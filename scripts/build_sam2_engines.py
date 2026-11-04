"""Build a SAM2.1 ONNX bundle with the project's TensorRT10 builder."""
import argparse
import ctypes as C
import json
from pathlib import Path
from trt10_runtime import BuilderConfig, Profile, bind, load_runtime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", type=Path, default=Path("cache/sam2.1/onnx"))
    p.add_argument("--output", type=Path, default=Path("cache/sam2.1/engines"))
    p.add_argument("--build", type=Path, default=Path("build/trt10-release"))
    p.add_argument("--workspace-mib", type=int, default=2048)
    p.add_argument("--encoder-fp16", action="store_true", help="Opt-in; verify accuracy before deployment")
    a = p.parse_args()
    if a.workspace_mib < 1:
        p.error("--workspace-mib must be positive")
    manifest = json.loads((a.onnx / "bundle.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "cvinfer-sam2.1-v1" or {r["name"] for r in manifest["graphs"]} != {"image_encoder", "mask_decoder"}:
        p.error("Expected a complete export_sam2.py bundle")
    if not isinstance(manifest.get("max_points"), int) or not 1 <= manifest["max_points"] <= 4096:
        p.error("Invalid bundle max_points")
    output, build = a.output.resolve(), a.build.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "bundle.json").unlink(missing_ok=True)
    handles = load_runtime(build)  # Keep DLL search-directory handles alive.
    dll = C.CDLL(str(build / "tensorrt_builder/alanxinn_tensorrt_builder.dll"))
    config = BuilderConfig()
    bind(dll, "trt_builder_config_init", None, C.POINTER(BuilderConfig))(C.byref(config))
    config.workspace_mib, config.enable_tf32 = a.workspace_mib, 0
    run = bind(dll, "trt_builder_build_onnx", C.c_int, C.c_char_p, C.c_char_p,
               C.POINTER(BuilderConfig), C.POINTER(Profile), C.c_size_t)
    error = bind(dll, "trt_builder_last_error", C.c_char_p)
    engines = []
    for graph in manifest["graphs"]:
        name = graph["name"]
        config.precision = int(name == "image_encoder" and a.encoder_fp16)
        profile = None
        if name == "mask_decoder":
            def shapes(n):
                return ("f0:1x32x256x256,f1:1x64x128x128,f2:1x256x64x64,"
                        f"point_coords:1x{n}x2,point_labels:1x{n},"
                        "mask_input:1x1x256x256,has_mask_input:1").encode()
            # Box corners are additional to the user-configured point limit.
            profile = Profile(shapes(1), shapes(min(4, manifest["max_points"] + 2)), shapes(manifest["max_points"] + 2))
        # Resolve by bundle-relative filename so moving the ONNX folder works.
        source = a.onnx / (name + ".onnx")
        print("Building", name, flush=True)
        status = run(str(source.resolve()).encode(), str(output / (name + ".engine")).encode(), C.byref(config),
                     C.byref(profile) if profile else None, int(profile is not None))
        if status:
            raise RuntimeError(error().decode("utf-8", errors="replace"))
        engines.append({"name": name, "precision": "fp16" if config.precision else "fp32"})
    manifest.update(engines=engines, workspace_mib=a.workspace_mib, status="ENGINES_BUILT_REAL_IMAGE_VALIDATION_REQUIRED")
    (output / "bundle.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
