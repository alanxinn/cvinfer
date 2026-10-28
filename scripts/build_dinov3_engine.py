"""Build a DINOv3 engine using this project's native TensorRT 10 ONNX builder."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import shutil
from validate_trt10 import BuilderConfig, bind, load_runtime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", type=Path, default=Path("cache/dinov3/onnx"))
    p.add_argument("--output", type=Path, default=Path("cache/dinov3/engine"))
    p.add_argument("--build", type=Path, default=Path("build/trt10-release"))
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    a = p.parse_args()
    manifest = json.loads((a.onnx / "bundle.json").read_text())
    if manifest["format"] != "cvinfer-dinov3-v1":
        p.error("Unsupported DINOv3 bundle format")
    source = a.onnx / manifest["onnx"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["onnx_sha256"]:
        raise ValueError("DINOv3 ONNX SHA256 does not match the export manifest")
    output, build = a.output.resolve(), a.build.resolve()
    output.mkdir(parents=True, exist_ok=True)
    handles = load_runtime(build)
    dll = C.CDLL(str(build / "tensorrt_builder/alanxinn_tensorrt_builder.dll"))
    config = BuilderConfig()
    bind(dll, "trt_builder_config_init", None, C.POINTER(BuilderConfig))(C.byref(config))
    config.precision, config.enable_tf32, config.workspace_mib = int(a.precision == "fp16"), 0, 2048
    run = bind(dll, "trt_builder_build_onnx", C.c_int, C.c_char_p, C.c_char_p, C.POINTER(BuilderConfig), C.c_void_p, C.c_size_t)
    status = run(str(source.resolve()).encode(), str(output / "model.engine").encode(), C.byref(config), None, 0)
    if status:
        raise RuntimeError(bind(dll, "trt_builder_last_error", C.c_char_p)().decode())
    if (a.onnx / "processor").exists() and (a.onnx / "processor").resolve() != (output / "processor").resolve():
        shutil.copytree(a.onnx / "processor", output / "processor", dirs_exist_ok=True)
    manifest.update({"engine": "model.engine", "precision": a.precision, "status": "ENGINE_BUILT_NATIVE_VALIDATION_REQUIRED"})
    (output / "bundle.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Built", output / "model.engine", flush=True)


if __name__ == "__main__":
    main()
