"""Pack RTX30/40/50 runtime dependencies, each below NuGet.org's size limit.

Large PTX/SM120 DLLs are split into native assets and reassembled/verified by
TensorRtRuntime on first use. Consumers install only the public top-level IDs.
"""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

CHUNK_BYTES = 240_000_000
MAX_PACKAGE_BYTES = 250_000_000


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def vc_runtime_files():
    # App-local MSVC runtime avoids a separate VC++ Redistributable installer.
    vswhere = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
    installation = subprocess.check_output([str(vswhere), "-latest", "-products", "*", "-requires",
                                           "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], text=True).strip()
    candidates = list((Path(installation) / "VC/Redist/MSVC").glob("*/x64/Microsoft.VC*.CRT"))
    if not candidates: raise FileNotFoundError("No redistributable x64 MSVC runtime found")
    selected = max(candidates, key=lambda x: tuple(int(p) for p in x.parents[1].name.split(".")))
    files = list(selected.glob("*.dll"))
    for required in ["msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"]:
        if not any(x.name.lower() == required for x in files): raise FileNotFoundError(required)
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("build/trt10-release/builder_runtime"))
    p.add_argument("--output", type=Path, default=Path("package/TensorRT10"))
    p.add_argument("--version", default="0.2.1-trt10")
    a = p.parse_args()
    source, output = a.source.resolve(), a.output.resolve()
    repo = Path(__file__).resolve().parents[1]
    staging = repo / "cache/trt10-runtime-packages" / a.version
    staging.mkdir(parents=True, exist_ok=True); output.mkdir(parents=True, exist_ok=True)
    groups = {"Core": [source / x for x in ["nvinfer_10.dll", "nvinfer_plugin_10.dll", "cudart64_12.dll"]],
              "SM86": [source / "nvinfer_builder_resource_sm86_10.dll"],
              "SM89": [source / "nvinfer_builder_resource_sm89_10.dll"]}
    groups["Core"].extend(vc_runtime_files())
    manifest = ["cvinfer-trt10-parts-v1"]
    for component, filename, count in [("PTX", "nvinfer_builder_resource_ptx_10.dll", 3),
                                       ("SM120", "nvinfer_builder_resource_sm120_10.dll", 2)]:
        original = source / filename
        if (original.stat().st_size + CHUNK_BYTES - 1) // CHUNK_BYTES != count:
            raise ValueError("SDK file size changed; update runtime dependency part counts: " + filename)
        parts = []
        with original.open("rb") as reader:
            for index in range(1, count + 1):
                target = staging / (filename + ".part" + str(index))
                remaining = CHUNK_BYTES
                with target.open("wb") as writer:
                    while remaining:
                        block = reader.read(min(1024 * 1024, remaining))
                        if not block: break
                        writer.write(block); remaining -= len(block)
                groups[component + ".Part" + str(index)] = [target]
                parts.append(target.name)
        manifest.append("\t".join([filename, digest(original), *parts]))
    index_file = staging / "tensorrt10-runtime.tsv"
    index_file.write_text("\n".join(manifest) + "\n", encoding="utf-8")
    groups["Core"].append(index_file)
    for component, files in groups.items():
        package_id = "alanxinn.TensorRtRuntime." + component + ".TensorRT10"
        project = ET.Element("Project", Sdk="Microsoft.NET.Sdk")
        properties = ET.SubElement(project, "PropertyGroup")
        for key, value in {"TargetFramework": "net8.0", "PackageId": package_id, "Version": a.version,
                           "Authors": "alanxinn", "Description": "TensorRT10 Windows x64 runtime assets for RTX30/40/50: " + component,
                           "IncludeBuildOutput": "false", "SuppressDependenciesWhenPacking": "true", "EnableDefaultItems": "false",
                           "PackageLicenseFile": "LICENSE.txt", "NoWarn": "NU5128"}.items():
            ET.SubElement(properties, key).text = value
        items = ET.SubElement(project, "ItemGroup")
        for file in files:
            if not file.is_file(): raise FileNotFoundError(file)
            ET.SubElement(items, "None", Include=str(file), Pack="true", PackagePath="runtimes/win-x64/native/" + file.name)
        ET.SubElement(items, "None", Include=str(repo / "LICENSE.txt"), Pack="true", PackagePath="LICENSE.txt")
        folder = staging / component; folder.mkdir(exist_ok=True)
        csproj = folder / "Runtime.csproj"
        ET.ElementTree(project).write(csproj, encoding="utf-8", xml_declaration=True)
        print("Packing", package_id, flush=True)
        with (folder / "pack.log").open("w", encoding="utf-8") as log:
            subprocess.run(["dotnet", "restore", str(csproj)], stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run(["dotnet", "pack", str(csproj), "--no-build", "-c", "Release", "-o", str(output)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        package = output / (package_id + "." + a.version + ".nupkg")
        if package.stat().st_size > MAX_PACKAGE_BYTES:
            raise ValueError("Package exceeds NuGet.org's 250MB limit: " + str(package))


if __name__ == "__main__":
    main()
