"""Build a DINOv3 engine through TensorRtOnnxBuilder.TensorRT10 (or explicit native mode)."""
import argparse
import ctypes as C
import json
from pathlib import Path
import shutil
from trt10_runtime import BuilderConfig, bind, load_runtime
from tensorrt_nuget_builder import build_engine, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", type=Path, default=Path("cache/dinov3/onnx"))
    p.add_argument("--output", type=Path, default=Path("cache/dinov3/engine"))
    p.add_argument("--build", type=Path, default=Path("build/trt10-release"))
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--builder", choices=["nuget", "native"], default="nuget")
    p.add_argument("--packages", type=Path, default=Path("package/TensorRT10"))
    p.add_argument("--package-version", default=None, help="Main builder package version; defaults to the current source version")
    p.add_argument("--nuget-cache", type=Path, default=Path("cache/trt10-nuget-builder"))
    p.add_argument("--workspace-mib", type=int, default=2048)
    a = p.parse_args()
    manifest = json.loads((a.onnx / "bundle.json").read_text())
    if manifest["format"] != "cvinfer-dinov3-v1":
        p.error("Unsupported DINOv3 bundle format")
    source = a.onnx / manifest["onnx"]
    if a.workspace_mib <= 0:
        p.error("--workspace-mib must be positive")
    if sha256(source) != manifest["onnx_sha256"]:
        raise ValueError("DINOv3 ONNX SHA256 does not match the export manifest")
    output, build = a.output.resolve(), a.build.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "bundle.json").unlink(missing_ok=True)
    if a.builder == "nuget":
        receipt = build_engine(source, output / "model.engine", a.precision, a.workspace_mib,
                               a.packages, a.package_version, a.nuget_cache)
    else:
        handles = load_runtime(build)
        dll = C.CDLL(str(build / "tensorrt_builder/alanxinn_tensorrt_builder.dll"))
        config = BuilderConfig()
        bind(dll, "trt_builder_config_init", None, C.POINTER(BuilderConfig))(C.byref(config))
        config.precision, config.enable_tf32, config.workspace_mib = int(a.precision == "fp16"), 0, a.workspace_mib
        run = bind(dll, "trt_builder_build_onnx", C.c_int, C.c_char_p, C.c_char_p, C.POINTER(BuilderConfig), C.c_void_p, C.c_size_t)
        status = run(str(source.resolve()).encode(), str(output / "model.engine").encode(), C.byref(config), None, 0)
        if status:
            raise RuntimeError(bind(dll, "trt_builder_last_error", C.c_char_p)().decode())
        receipt = {"converter": "native C ABI", "engine_sha256": sha256(output / "model.engine")}
    if (a.onnx / "processor").exists() and (a.onnx / "processor").resolve() != (output / "processor").resolve():
        shutil.copytree(a.onnx / "processor", output / "processor", dirs_exist_ok=True)
    manifest.update({"engine": "model.engine", "precision": a.precision, "engine_sha256": receipt["engine_sha256"],
                     "converter": receipt["converter"], "workspace_mib": a.workspace_mib,
                     "status": "ENGINE_BUILT_REAL_VALIDATION_REQUIRED"})
    (output / "bundle.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Built", output / "model.engine", flush=True)


if __name__ == "__main__":
    main()
