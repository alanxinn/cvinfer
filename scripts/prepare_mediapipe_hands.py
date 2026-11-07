"""Download pinned OpenCV MediaPipe ONNX models and build TensorRT 10 engines."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import urllib.request

from trt10_runtime import BuilderConfig, bind, load_runtime

MODELS = {
    "palm": ("palm_detection_mediapipe", "palm_detection_mediapipe_2023feb.onnx",
             "78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c"),
    "hand": ("handpose_estimation_mediapipe", "handpose_estimation_mediapipe_2023feb.onnx",
             "db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c"),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url, path, expected=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (expected is None or sha256(path) == expected):
        return
    partial = path.with_suffix(path.suffix + ".download")
    try:
        print(f"Downloading {url}", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "cvinfer-mediapipe"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if expected is not None and sha256(partial) != expected:
            raise ValueError(f"SHA256 mismatch: {path.name}")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def prepare(root=Path("cache/mediapipe-hands"), build=Path("build/trt10-release"),
            precision="fp32", download_only=False):
    import onnx
    if precision not in ("fp32", "fp16"):
        raise ValueError("MediaPipe precision must be fp32 or fp16")
    root, build = Path(root).resolve(), Path(build).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "bundle.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {"format": "cvinfer-mediapipe-hands-v1", "precision": precision, "models": {}}
    dll = None
    for key, (directory, filename, checksum) in MODELS.items():
        source = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/{directory}/{filename}"
        original = root / "weights" / filename
        download(source, original, checksum)
        target, engine = root / f"{key}.onnx", root / f"{key}-{precision}.engine"
        # Normalize names once; both models retain their original NHWC FP32 IO.
        graph = onnx.load(original)
        if key == "palm":
            outputs = {}
            for output in graph.graph.output:
                last = output.type.tensor_type.shape.dim[-1].dim_value
                outputs[output.name] = "palm_boxes" if last == 18 else "palm_scores"
        else:
            outputs = dict(zip(["Identity", "Identity_1", "Identity_2", "Identity_3"],
                               ["landmarks", "confidence", "handedness", "world_landmarks"]))
            if set(outputs) != {x.name for x in graph.graph.output}:
                raise ValueError("Unexpected MediaPipe hand model output contract")
        rename = {graph.graph.input[0].name: key + "_input", **outputs}
        for node in graph.graph.node:
            for names in (node.input, node.output):
                for i, name in enumerate(names):
                    names[i] = rename.get(name, name)
        for value in list(graph.graph.input) + list(graph.graph.output) + list(graph.graph.value_info):
            value.name = rename.get(value.name, value.name)
        onnx.checker.check_model(graph)
        serialized = graph.SerializeToString()
        if not target.exists() or target.read_bytes() != serialized:
            target.write_bytes(serialized)
        record = {"source": source, "source_sha256": checksum, "onnx": target.name,
                  "onnx_sha256": sha256(target), "engine": engine.name}
        old = previous.get("models", {}).get(key, {})
        valid_cache = (previous.get("precision") == precision and old.get("onnx_sha256") == record["onnx_sha256"]
                       and engine.exists() and old.get("engine_sha256") == sha256(engine))
        if not download_only:
            if not valid_cache:
                if dll is None:
                    handles = load_runtime(build)  # Retain DLL search handles through all builds.
                    dll = C.CDLL(str(build / "tensorrt_builder/alanxinn_tensorrt_builder.dll"))
                config = BuilderConfig()
                bind(dll, "trt_builder_config_init", None, C.POINTER(BuilderConfig))(C.byref(config))
                config.precision, config.enable_tf32, config.workspace_mib = int(precision == "fp16"), 0, 1024
                print(f"Building {key} {precision} engine on device 0...", flush=True)
                run = bind(dll, "trt_builder_build_onnx", C.c_int, C.c_char_p, C.c_char_p,
                           C.POINTER(BuilderConfig), C.c_void_p, C.c_size_t)
                partial = engine.with_suffix(".building.engine")
                try:
                    status = run(str(target).encode(), str(partial).encode(), C.byref(config), None, 0)
                    if status:
                        raise RuntimeError(bind(dll, "trt_builder_last_error", C.c_char_p)().decode())
                    partial.replace(engine)
                finally:
                    partial.unlink(missing_ok=True)
            record["engine_sha256"] = sha256(engine)
        manifest["models"][key] = record
        print(f"Ready: {key} ({'ONNX' if download_only else precision + ' engine'})", flush=True)
    # Download-only must not invalidate a previously built engine manifest.
    if not download_only:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("cache/mediapipe-hands"))
    parser.add_argument("--build", type=Path, default=Path("build/trt10-release"))
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    prepare(args.output, args.build, args.precision, args.download_only)


if __name__ == "__main__":
    main()
