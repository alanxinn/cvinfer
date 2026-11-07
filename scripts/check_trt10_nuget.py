"""Restore the TRT10 NuGets into an isolated CvinferDemo and discover local model tests.

No ProjectReference, manual native-DLL copy, SDK or runtime ZIP is needed.
Runtime resources are restored as transitive dependencies and assembled automatically.
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
    p.add_argument("--version", default="0.2.0")
    p.add_argument("--output", type=Path, default=Path("cache/nuget-trt10-validation"))
    p.add_argument("--reuse-engines", action="store_true", help="Reuse existing engines instead of forcing a YOLO rebuild; validate builder separately")
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    feed, output = a.packages.resolve(), a.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation.json").write_text(json.dumps({"status": "RUNNING"}))
    packages = {}
    inventory = []
    for name in ["alanxinn.CvinferSharp.TensorRT10", "alanxinn.TensorRtOnnxBuilder.TensorRT10"]:
        path = feed / (name + "." + a.version + ".nupkg")
        if not path.is_file(): raise FileNotFoundError(path)
        assert path.stat().st_size <= 250_000_000, "Package exceeds NuGet.org limit"
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            metadata = ET.fromstring(archive.read(next(x for x in names if x.endswith(".nuspec"))))
            fields = {x.tag.split("}")[-1]: x.text for x in metadata.find("{*}metadata")}
            assert fields["id"] == name
            dependencies = {x.attrib["id"] for x in metadata.findall(".//{*}dependency")}
            assert "alanxinn.TensorRtRuntime.Core.TensorRT10" in dependencies
            for tfm in ["net472", "net6.0", "net8.0"]:
                assert any(x.startswith("lib/" + tfm + "/") and x.endswith(".dll") for x in names), tfm
            assert not any("cvinfer_cuda" in x or "denglin" in x.lower() or "nvinfer_8.dll" in x for x in names)
            if name.startswith("alanxinn.CvinferSharp"):
                assert f"buildTransitive/{name}.targets" in names
                for dll in ["alanxinn_cvinfer_trt.dll", "alanxinn_cvinfer_sam3.dll", "alanxinn_cvinfer_sam2.dll", "alanxinn_cvinfer_dinov3.dll", "alanxinn_cvinfer_mediapipe.dll"]:
                    assert "runtimes/win-x64/native/" + dll in names, dll
            else:
                assert "runtimes/win-x64/native/alanxinn_tensorrt_builder.dll" in names
                for component in ["SM86", "SM89", "PTX.Part1", "PTX.Part2", "PTX.Part3", "SM120.Part1", "SM120.Part2"]:
                    assert "alanxinn.TensorRtRuntime." + component + ".TensorRT10" in dependencies
        packages[name] = fields["version"]
        inventory.append({"package_id": name, "version": fields["version"], "file": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    for component in ["Core", "SM86", "SM89", "PTX.Part1", "PTX.Part2", "PTX.Part3", "SM120.Part1", "SM120.Part2"]:
        package_id = "alanxinn.TensorRtRuntime." + component + ".TensorRT10"
        path = feed / (package_id + "." + next(iter(packages.values())) + ".nupkg")
        assert path.is_file() and path.stat().st_size <= 250_000_000, package_id
        inventory.append({"package_id": package_id, "file": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    # A package rebuilt with the same NuGet version must not silently reuse
    # an older extracted package or old native files from a previous run.
    fingerprint = hashlib.sha256("".join(x["sha256"] for x in inventory).encode()).hexdigest()[:16]
    runtime = output / "runtime" / fingerprint  # Empty cache; no DLLs are supplied by this script.
    consumer = output / "consumer" / fingerprint
    consumer.mkdir(parents=True, exist_ok=True)
    config = ET.Element("configuration")
    sources = ET.SubElement(config, "packageSources"); ET.SubElement(sources, "clear")
    ET.SubElement(sources, "add", key="TRT10-local", value=str(feed))
    ET.SubElement(sources, "add", key="nuget.org", value="https://api.nuget.org/v3/index.json")
    config_path = consumer / "NuGet.Config"
    ET.ElementTree(config).write(config_path, encoding="utf-8", xml_declaration=True)
    # Build the real Demo, with only PackageReference and fingerprint-isolated
    # intermediates/output. CopyLocalTensorRtRuntime is disabled by this mode.
    project = repo / "csharp/alanxinn.CvinferDemo/alanxinn.CvinferDemo.csproj"
    run(["dotnet", "build", str(project), "-c", "Release",
         "-p:UseLocalTensorRtProject=false", "-p:UseLocalDenglinProject=false",
         "-p:CvinferSharpPackageVersion=" + a.version,
         "-p:RestoreConfigFile=" + str(config_path),
         "-p:RestorePackagesPath=" + str(output / "packages" / fingerprint),
         "-p:BaseOutputPath=" + (consumer / "bin").as_posix() + "/",
         "-p:BaseIntermediateOutputPath=" + (consumer / "obj").as_posix() + "/",
         "-p:MSBuildProjectExtensionsPath=" + (consumer / "obj").as_posix() + "/"],
        output / "consumer-build.log")
    assets = json.loads((consumer / "obj/project.assets.json").read_text(encoding="utf-8"))
    assert not any(x.get("type") == "project" for x in assets["libraries"].values()), "ProjectReference leaked into package test"
    for name in packages:
        assert name + "/" + a.version in assets["libraries"], name
    binary = consumer / "bin/Release/net8.0-windows"
    # Matching target name must cause native DLLs to land in the isolated path.
    for dll in ["alanxinn_cvinfer_trt.dll", "alanxinn_cvinfer_sam3.dll", "alanxinn_cvinfer_sam2.dll", "alanxinn_cvinfer_dinov3.dll", "alanxinn_cvinfer_mediapipe.dll", "nvinfer_10.dll"]:
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
        rebuild = not a.reuse_engines and name == "yolo11n" and Path(record["onnx"]).is_file()
        if rebuild: engine = output / (name + ".engine")
        if not engine.is_file() and not Path(record["onnx"]).is_file():
            raise FileNotFoundError(f"No local engine or ONNX for {name}")
        tests.append({**record, "mode": "model", "engine": str(engine), "rebuild": rebuild,
                      "contract_onnx": str(repo / "cache/trt10-contract/int64.onnx"), "contract_engine": str(output / "int64.engine")})
    sam = next((x.parent for x in [repo / "cache/sam31/engines/bundle.json", repo / "cache/sam3/engines/bundle.json"]
                if x.is_file() and not json.loads(x.read_text()).get("synthetic", False)), None)
    if sam: tests.append({"mode": "sam3", "name": "sam3.1", "engine": str(sam)})
    # Only real, complete SAM2.1 bundles; absence is recorded, never passed off
    # as a successful test. Also discover variant-specific cache/model folders.
    sam2_bundles = sorted(set((repo / "cache").glob("sam2*/**/bundle.json")) |
                          set((repo / "models").glob("sam2*/**/bundle.json")))
    sam2_count = 0
    for path in sam2_bundles:
        bundle = json.loads(path.read_text(encoding="utf-8"))
        if (bundle.get("format") == "cvinfer-sam2.1-v1" and bundle.get("source", {}).get("sha256") and
            all((path.parent / (name + ".engine")).is_file() for name in ["image_encoder", "mask_decoder"])):
            sam2_count += 1
            variant = bundle["source"]["variant"]
            tests.append({"mode": "sam2", "name": f"sam2.1-{variant}-{sam2_count}",
                          "variant": variant, "engine": str(path.parent)})
    # Authorized pretrained DINOv3 is preferred; otherwise clearly identify
    # the existing real-operator/random-weight bundles as synthetic checks.
    dino = list((repo / "cache/dinov3/engine").glob("bundle.json"))
    if not dino: dino = sorted((repo / "cache/dinov3-tiny").glob("*/engine/bundle.json"))
    for path in dino:
        bundle = json.loads(path.read_text())
        tests.append({"mode": "dinov3", "name": "dinov3-" + bundle["model_type"], "engine": str(path.parent / "model.engine"), "synthetic": bundle.get("synthetic", True)})
    import cv2
    hands_image = cv2.imread(str(repo / "cache/mediapipe-hands/woman_hands.jpg"))
    if hands_image is None:
        raise FileNotFoundError("Prepare MediaPipe models and woman_hands.jpg before testing the NuGet")
    hand_pixels = output / "hands.bgr"
    hands_image.tofile(hand_pixels)
    for precision in ("fp32", "fp16"):
        palm_engine = repo / f"cache/mediapipe-hands/palm-{precision}.engine"
        hand_engine = repo / f"cache/mediapipe-hands/hand-{precision}.engine"
        if not palm_engine.is_file() or not hand_engine.is_file():
            raise FileNotFoundError(f"Missing MediaPipe {precision} engines")
        tests.append({"mode": "mediapipe", "name": "mediapipe-hands-" + precision,
                      "engine": str(hand_engine), "palm_engine": str(palm_engine), "precision": precision,
                      "image": str(hand_pixels), "width": hands_image.shape[1], "height": hands_image.shape[0]})
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("ALANXINN_"): del env[key]
    env["ALANXINN_TENSORRT_RUNTIME_CACHE"] = str(runtime)
    env["ALANXINN_TENSORRT_REQUIRE_BUNDLED"] = "1"
    env["ALANXINN_TENSORRT_NO_DOWNLOAD"] = "1"
    results = []
    for case in tests:
        name = case["name"]
        request = {**case, "image": case.get("image", str(image)), "width": case.get("width", 810), "height": case.get("height", 1080), "runtime": str(runtime),
                   "optional_libraries": False, "report": str(output / (name + ".json"))}
        spec = output / (name + "-input.json"); spec.write_text(json.dumps(request))
        print("Testing NuGet", name, flush=True)
        try:
            run(["dotnet", str(binary / "alanxinn.CvinferDemo.dll"), "--test", "nuget", "--reference", str(spec)], output / (name + ".log"), env=env)
            result = json.loads(Path(request["report"]).read_text())
        except subprocess.CalledProcessError as error:
            result = {"name": name, "status": "FAIL", "error": str(error), "log": str(output / (name + ".log"))}
        results.append(result)
    report = {"status": "PASS" if all(x["status"] == "PASS" for x in results) else "FAIL", "packages": inventory, "tests": results,
              "runtime_versions": runtime_versions(Path(results[0]["runtime_directory"])) if results[0].get("runtime_directory") else None,
              "consumer": str(consumer), "source": "CvinferDemo with PackageReference from local feed; no ProjectReference or native DLL copying",
              "sam2": {"real_bundles": sam2_count, "status": "TESTED" if sam2_count else "SKIPPED_NO_ENGINE_BUNDLE"}}
    (output / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": report["status"], "tests": len(results), "report": str(output / "validation.json")}), flush=True)
    if report["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__":
    main()
