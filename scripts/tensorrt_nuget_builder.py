"""Build ONNX through the actual TensorRT10 NuGet in an isolated .NET consumer."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def builder_version():
    path = Path(__file__).resolve().parents[1] / "csharp/alanxinn.TensorRtOnnxBuilder/alanxinn.TensorRtOnnxBuilder.csproj"
    return ET.parse(path).findtext("./PropertyGroup/Version")


def build_engine(onnx, engine, precision="fp32", workspace_mib=2048,
                 packages="package/TensorRT10", version=None, cache="cache/trt10-nuget-builder"):
    repo = Path(__file__).resolve().parents[1]
    feed, cache = Path(packages).resolve(), Path(cache).resolve()
    version = version or builder_version()
    ids = ["alanxinn.TensorRtOnnxBuilder.TensorRT10"] + [
        "alanxinn.TensorRtRuntime." + name + ".TensorRT10" for name in
        ["Core", "SM86", "SM89", "PTX.Part1", "PTX.Part2", "PTX.Part3", "SM120.Part1", "SM120.Part2"]]
    inventory = []
    for package_id in ids:
        candidates = [feed / f"{package_id}.{version}.nupkg"] if package_id == ids[0] else sorted(feed.glob(package_id + ".*.nupkg"))
        if not candidates:
            raise FileNotFoundError("No local runtime release found for " + package_id)
        for path in candidates:
            with zipfile.ZipFile(path) as archive:
                metadata = ET.fromstring(archive.read(next(name for name in archive.namelist() if name.endswith(".nuspec"))))
                actual_version = metadata.find(".//{*}metadata/{*}version").text
            inventory.append({"id": package_id, "version": actual_version, "file": str(path), "sha256": sha256(path)})
    # Rebuilt packages with an unchanged version must never reuse stale extracted DLLs.
    fingerprint = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()[:16]
    consumer = cache / fingerprint
    consumer.mkdir(parents=True, exist_ok=True)
    project = ET.Element("Project", Sdk="Microsoft.NET.Sdk")
    properties = ET.SubElement(project, "PropertyGroup")
    for key, value in {"OutputType": "Exe", "TargetFramework": "net8.0", "PlatformTarget": "x64",
                       "ImplicitUsings": "enable", "Nullable": "enable", "EnableDefaultCompileItems": "false",
                       "AssemblyName": "Trt10NugetBuilder", "RestorePackagesPath": str(consumer / "packages")}.items():
        ET.SubElement(properties, key).text = value
    items = ET.SubElement(project, "ItemGroup")
    ET.SubElement(items, "Compile", Include=str(repo / "csharp/alanxinn.TensorRtOnnxBuilderDemo/Program.cs"))
    ET.SubElement(items, "PackageReference", Include=ids[0], Version=f"[{version}]")
    for package_id in ids[1:]:
        ET.SubElement(items, "PackageReference", Include=package_id, Version="*")
    project_path = consumer / "Builder.csproj"
    ET.ElementTree(project).write(project_path, encoding="utf-8", xml_declaration=True)
    config = ET.Element("configuration")
    sources = ET.SubElement(config, "packageSources")
    ET.SubElement(sources, "clear")
    ET.SubElement(sources, "add", key="local-trt10", value=str(feed))
    config_path = consumer / "NuGet.Config"
    ET.ElementTree(config).write(config_path, encoding="utf-8", xml_declaration=True)
    env = {key: value for key, value in os.environ.items() if not key.startswith("ALANXINN_TENSORRT_")}
    runtime = consumer / "runtime"
    env.update(ALANXINN_TENSORRT_REQUIRE_BUNDLED="1", ALANXINN_TENSORRT_NO_DOWNLOAD="1",
               ALANXINN_TENSORRT_RUNTIME_CACHE=str(runtime))
    subprocess.run(["dotnet", "build", str(project_path), "-c", "Release",
                    "--configfile", str(config_path)], env=env, check=True)
    engine = Path(engine).resolve()
    engine.parent.mkdir(parents=True, exist_ok=True)
    report_path = engine.with_suffix(".build.json")
    report_path.unlink(missing_ok=True)
    subprocess.run(["dotnet", str(consumer / "bin/Release/net8.0/Trt10NugetBuilder.dll"),
                    "--onnx", str(Path(onnx).resolve()), "--engine", str(engine),
                    "--precision", precision, "--no-tf32", "--workspace-mib", str(workspace_mib),
                    "--report", str(report_path)], env=env, check=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    def within(path, root):
        return Path(path).resolve().is_relative_to(root.resolve())
    binary = consumer / "bin/Release/net8.0"
    if not report["runtime_directory"] or not within(report["runtime_directory"], runtime):
        raise RuntimeError("Builder did not use NuGet bundled runtime")
    if not report["modules"] or any(not within(path, binary) and not within(path, runtime) for path in report["modules"]):
        raise RuntimeError("Builder loaded native DLLs outside the NuGet output/runtime cache")
    if sha256(engine) != report["engine_sha256"]:
        raise RuntimeError("Built engine SHA256 mismatch")
    assets = json.loads((consumer / "obj/project.assets.json").read_text(encoding="utf-8"))
    resolved = {name.rsplit("/", 1)[0].lower(): name.rsplit("/", 1)[1] for name in assets["libraries"]}
    inventory = [item for item in inventory if resolved.get(item["id"].lower()) == item["version"]]
    if {item["id"] for item in inventory} != set(ids):
        raise RuntimeError("Resolved runtime packages differ from the verified local inventory")
    report.update(converter=ids[0] + "/" + version, packages=inventory, package_fingerprint=fingerprint)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
