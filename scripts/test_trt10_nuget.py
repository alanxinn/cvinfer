"""Restore the TRT10 NuGets into an isolated consumer and discover local model tests.

No ProjectReference or manual native-DLL copy is allowed. Runtime comes from
the deployment ZIP, while native inference/builder APIs come from the NuGets.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET


def run(command, log, **kwargs):
    with log.open("w", encoding="utf-8") as out:
        subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, check=True, **kwargs)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def runtime_versions(runtime):
    import ctypes as c
    # cuDNN's forwarding DLL uses its own LoadLibrary calls for components,
    # just as the C# runtime initializer provisions PATH/SetDllDirectory.
    directory = os.add_dll_directory(str(runtime))
    os.environ["PATH"] = str(runtime) + ";" + os.environ.get("PATH", "")
    kernel = c.WinDLL("kernel32", use_last_error=True)
    kernel.SetDllDirectoryW.argtypes = [c.c_wchar_p]
    kernel.SetDllDirectoryW.restype = c.c_int
    if not kernel.SetDllDirectoryW(str(runtime)): raise c.WinError(c.get_last_error())
    cuda = c.WinDLL(str(runtime / "cudart64_12.dll"))
    version = c.c_int()
    cuda.cudaRuntimeGetVersion.argtypes = [c.POINTER(c.c_int)]
    cuda.cudaRuntimeGetVersion.restype = c.c_int
    assert cuda.cudaRuntimeGetVersion(c.byref(version)) == 0
    trt = c.WinDLL(str(runtime / "nvinfer_10.dll")); trt.getInferLibVersion.restype = c.c_int
    cudnn_version = None
    if (runtime / "cudnn64_9.dll").is_file():
        cudnn = c.WinDLL(str(runtime / "cudnn64_9.dll")); cudnn.cudnnGetVersion.restype = c.c_size_t
        cudnn_version = cudnn.cudnnGetVersion()
        assert cudnn_version == 92700
    result = {"cudaRuntimeGetVersion": version.value, "cudnnGetVersion": cudnn_version, "getInferLibVersion": trt.getInferLibVersion()}
    assert result["cudaRuntimeGetVersion"] == 12090 and result["getInferLibVersion"] == 101601, result
    directory.close()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packages", type=Path, default=Path("package/TensorRT10"))
    p.add_argument("--runtime-zip", type=Path, help="Deployment ZIP selected by Pack-Trt10.ps1")
    p.add_argument("--output", type=Path, default=Path("cache/nuget-trt10-validation"))
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    feed, output = a.packages.resolve(), a.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation.json").write_text(json.dumps({"status": "RUNNING"}))
    packages = {}
    inventory = []
    for name in ["alanxinn.CvinferSharp.TensorRT10", "alanxinn.TensorRtOnnxBuilder.TensorRT10"]:
        candidates = list(feed.glob(name + ".*.nupkg"))
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one {name} nupkg in {feed}")
        path = candidates[0]
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            metadata = ET.fromstring(archive.read(next(x for x in names if x.endswith(".nuspec"))))
            fields = {x.tag.split("}")[-1]: x.text for x in metadata.find("{*}metadata")}
            assert fields["id"] == name
            for tfm in ["net472", "net6.0", "net8.0"]:
                assert any(x.startswith("lib/" + tfm + "/") and x.endswith(".dll") for x in names), tfm
            assert not any("cvinfer_cuda" in x or "denglin" in x.lower() or "nvinfer_8.dll" in x for x in names)
            if name.startswith("alanxinn.CvinferSharp"):
                assert f"buildTransitive/{name}.targets" in names
                for dll in ["alanxinn_cvinfer_trt.dll", "alanxinn_cvinfer_sam3.dll", "alanxinn_cvinfer_dinov3.dll", "nvinfer_10.dll", "cudart64_12.dll"]:
                    assert "runtimes/win-x64/native/" + dll in names, dll
            else:
                assert "runtimes/win-x64/native/alanxinn_tensorrt_builder.dll" in names
        packages[name] = fields["version"]
        inventory.append({"package_id": name, "version": fields["version"], "file": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    runtime_zip = a.runtime_zip.resolve() if a.runtime_zip else feed / "tensorrt10.16.1.11-cuda12.9-sm89-runtime-win-x64.zip"
    runtime_hash = sha256(runtime_zip)
    # Never let DLLs extracted from an earlier full SDK mask missing minimal dependencies.
    runtime = output / "runtime" / runtime_hash[:16]
    runtime.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(runtime_zip) as archive:
        runtime_manifest = json.loads(archive.read("runtime-manifest.json").decode("utf-8-sig"))
        for item in archive.infolist():
            target = (runtime / item.filename).resolve()
            if not target.is_relative_to(runtime.resolve()): raise ValueError("Unsafe runtime ZIP path")
            archive.extract(item, runtime)
        names = {Path(x).name for x in archive.namelist() if x.endswith(".dll")}
        assert names == set(runtime_manifest["dlls"])
        assert any(x.startswith("nvinfer_builder_resource_sm") for x in names)
        for dll in ["nvinfer_10.dll", "nvinfer_plugin_10.dll", "nvonnxparser_10.dll", "nvinfer_builder_resource_ptx_10.dll", "cudart64_12.dll"]:
            assert dll in names, dll
        if not runtime_manifest["optionalLibraries"]:
            assert not any(x.startswith(("cudnn", "cublas", "nvrtc")) for x in names)
    inventory.append({"file": str(runtime_zip), "bytes": runtime_zip.stat().st_size, "sha256": runtime_hash, "dlls": sorted(names)})
    # A package rebuilt with the same NuGet version must not silently reuse
    # an older extracted package or old native files from a previous run.
    fingerprint = hashlib.sha256("".join(x["sha256"] for x in inventory[:2]).encode()).hexdigest()[:16]
    consumer = output / "consumer" / fingerprint
    consumer.mkdir(parents=True, exist_ok=True)
    project = ET.Element("Project", Sdk="Microsoft.NET.Sdk")
    props = ET.SubElement(project, "PropertyGroup")
    for key, value in {"OutputType": "Exe", "TargetFramework": "net8.0-windows", "PlatformTarget": "x64", "ImplicitUsings": "enable", "Nullable": "enable", "RestorePackagesPath": str(output / "packages" / fingerprint)}.items():
        ET.SubElement(props, key).text = value
    items = ET.SubElement(project, "ItemGroup")
    for name, version in packages.items(): ET.SubElement(items, "PackageReference", Include=name, Version=version)
    ET.ElementTree(project).write(consumer / "Consumer.csproj", encoding="utf-8", xml_declaration=True)
    config = ET.Element("configuration")
    sources = ET.SubElement(config, "packageSources"); ET.SubElement(sources, "clear")
    ET.SubElement(sources, "add", key="TRT10-local", value=str(feed))
    ET.ElementTree(config).write(consumer / "NuGet.Config", encoding="utf-8", xml_declaration=True)
    (consumer / "Program.cs").write_bytes((repo / "scripts/nuget_trt10_smoke/Program.cs").read_bytes())
    run(["dotnet", "build", str(consumer / "Consumer.csproj"), "-c", "Release"], output / "consumer-build.log")
    binary = consumer / "bin/Release/net8.0-windows"
    # Matching target name must cause native DLLs to land in the isolated path.
    for dll in ["alanxinn_cvinfer_trt.dll", "alanxinn_cvinfer_sam3.dll", "alanxinn_cvinfer_dinov3.dll", "nvinfer_10.dll"]:
        assert (binary / "runtimes/win-x64/native" / dll).is_file(), dll
        assert not (binary / dll).is_file(), "Native assets were incorrectly flattened: " + dll
    image = repo / "cache/trt10-contract/bus.bgr"
    if not image.is_file():
        import cv2
        pixels = cv2.imread(str(repo / "cache/trt10-models/bus.jpg"))
        if pixels is None: raise ValueError("No local bus test image found")
        image = output / "bus.bgr"; pixels.tofile(image)
    assert image.stat().st_size == 810 * 1080 * 3
    models_path = repo / "cache/trt10-models/models.json"
    manifest = json.loads(models_path.read_text())
    tests = []
    for record in manifest["models"]:
        name = record["name"]
        candidates = [repo / "cache/trt10-validation" / (name + ".engine")]
        candidates += sorted((repo / "models").rglob(name + ".engine"))
        engine = next((x for x in candidates if x.is_file()), output / (name + ".engine"))
        rebuild = name == "yolo11n" and Path(record["onnx"]).is_file()
        if rebuild: engine = output / (name + ".engine")
        if not engine.is_file() and not Path(record["onnx"]).is_file():
            raise FileNotFoundError(f"No local engine or ONNX for {name}")
        tests.append({**record, "mode": "model", "engine": str(engine), "rebuild": rebuild,
                      "contract_onnx": str(repo / "cache/trt10-contract/int64.onnx"), "contract_engine": str(output / "int64.engine")})
    sam = next((x.parent for x in [repo / "cache/sam31/engines/bundle.json", repo / "cache/sam3/engines/bundle.json"]
                if x.is_file() and not json.loads(x.read_text()).get("synthetic", False)), None)
    if sam: tests.append({"mode": "sam3", "name": "sam3.1", "engine": str(sam)})
    # Authorized pretrained DINOv3 is preferred; otherwise clearly identify
    # the existing real-operator/random-weight bundles as synthetic checks.
    dino = list((repo / "cache/dinov3/engine").glob("bundle.json"))
    if not dino: dino = sorted((repo / "cache/dinov3-tiny").glob("*/engine/bundle.json"))
    for path in dino:
        bundle = json.loads(path.read_text())
        tests.append({"mode": "dinov3", "name": "dinov3-" + bundle["model_type"], "engine": str(path.parent / "model.engine"), "synthetic": bundle.get("synthetic", True)})
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("ALANXINN_"): del env[key]
    env["ALANXINN_TENSORRT_RUNTIME_DIR"] = str(runtime)
    env["ALANXINN_TENSORRT_NO_DOWNLOAD"] = "1"
    results = []
    for case in tests:
        name = case["name"]
        request = {**case, "image": str(image), "width": 810, "height": 1080, "runtime": str(runtime),
                   "optional_libraries": runtime_manifest["optionalLibraries"], "report": str(output / (name + ".json"))}
        spec = output / (name + "-input.json"); spec.write_text(json.dumps(request))
        print("Testing NuGet", name, flush=True)
        try:
            run(["dotnet", str(binary / "Consumer.dll"), str(spec)], output / (name + ".log"), env=env)
            result = json.loads(Path(request["report"]).read_text())
        except subprocess.CalledProcessError as error:
            result = {"name": name, "status": "FAIL", "error": str(error), "log": str(output / (name + ".log"))}
        results.append(result)
    report = {"status": "PASS" if all(x["status"] == "PASS" for x in results) else "FAIL", "packages": inventory, "tests": results,
              "runtime_versions": runtime_versions(runtime),
              "consumer": str(consumer), "source": "PackageReference from local feed; no ProjectReference or native DLL copying"}
    (output / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": report["status"], "tests": len(results), "report": str(output / "validation.json")}), flush=True)
    if report["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__":
    main()
