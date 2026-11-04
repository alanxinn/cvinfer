"""Compile an export_sam3.py bundle with the project's TensorRT 10 builder."""
import argparse
import ctypes as C
import json
from pathlib import Path
import shutil
from trt10_runtime import BuilderConfig, Profile, bind, load_runtime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", type=Path, default=Path("cache/sam3/onnx"))
    p.add_argument("--output", type=Path, default=Path("cache/sam3/engines"))
    p.add_argument("--build", type=Path, default=Path("build/trt10-release"))
    p.add_argument("--vision-fp16", action="store_true", help="Opt-in; verify accuracy against FP32 before deployment")
    p.add_argument("--graphs", nargs="+", choices=["detector_vision", "tracker_vision", "text_encoder", "text_decoder", "box_decoder", "interactive_decoder"],
                   help="Build selected graphs (retain other existing engine records)")
    p.add_argument("--workspace-mib", type=int, default=2048)
    a = p.parse_args()
    if a.workspace_mib < 1:
        p.error("--workspace-mib must be positive")
    manifest = json.loads((a.onnx / "bundle.json").read_text())
    output, build = a.output.resolve(), a.build.resolve()
    output.mkdir(parents=True, exist_ok=True)
    handles = load_runtime(build)
    dll = C.CDLL(str(build / "tensorrt_builder/alanxinn_tensorrt_builder.dll"))
    config = BuilderConfig()
    bind(dll, "trt_builder_config_init", None, C.POINTER(BuilderConfig))(C.byref(config))
    config.workspace_mib, config.enable_tf32 = a.workspace_mib, 0
    run = bind(dll, "trt_builder_build_onnx", C.c_int, C.c_char_p, C.c_char_p,
               C.POINTER(BuilderConfig), C.POINTER(Profile), C.c_size_t)
    error = bind(dll, "trt_builder_last_error", C.c_char_p)
    previous = output / "bundle.json"
    profiles = json.loads(previous.read_text()).get("engines", []) if a.graphs and previous.exists() else []
    for record in manifest["graphs"]:
        name = record["name"]
        if a.graphs and name not in a.graphs:
            continue
        config.precision = int(a.vision_fp16 and name.endswith("vision"))
        profile = None
        if name == "interactive_decoder":
            import onnx
            graph = onnx.load(record["onnx"], load_external_data=False)
            fixed = []
            for value in graph.graph.input:
                if value.name not in ("input_points", "input_labels"):
                    dims = [d.dim_value for d in value.type.tensor_type.shape.dim]
                    if not all(d > 0 for d in dims):
                        raise ValueError("Unexpected dynamic SAM3 input: " + value.name)
                    fixed.append(value.name + ":" + "x".join(map(str, dims)))
            def shapes(n):
                return ",".join(fixed + [f"input_points:1x1x{n}x2", f"input_labels:1x1x{n}"]).encode()
            profile = Profile(shapes(1), shapes(min(4, manifest["max_points"])), shapes(manifest["max_points"]))
        print("Building", name, flush=True)
        status = run(str(Path(record["onnx"]).resolve()).encode(), str(output / (name + ".engine")).encode(),
                     C.byref(config), C.byref(profile) if profile else None, int(profile is not None))
        if status:
            raise RuntimeError(error().decode())
        profiles[:] = [r for r in profiles if r["name"] != name]
        profiles.append({"name": name, "precision": "fp16" if config.precision else "fp32"})
    for name in ["vocab.tsv", "merges.tsv"]:
        shutil.copyfile(a.onnx / name, output / name)
    required = {"detector_vision", "tracker_vision", "text_encoder", "text_decoder", "box_decoder", "interactive_decoder"}
    complete = required <= {record["name"] for record in profiles} and all((output / (name + ".engine")).is_file() for name in required)
    manifest.update({"engines": profiles, "workspace_mib": a.workspace_mib,
                     "status": "ENGINES_BUILT_REAL_IMAGE_VALIDATION_REQUIRED" if complete else "PARTIAL_ENGINES_BUILT"})
    (output / "bundle.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
