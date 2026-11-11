"""Build all WeVisOne ONNX graphs with the existing TensorRT10 native builder.

Language has one shared prefill/decode graph, avoiding a second copy of LLM
weights in GPU memory. Pixel decoding always uses FP32, as required upstream.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import shutil
from trt10_runtime import BuilderConfig, Profile, bind, load_runtime
from wevisone_checkpoint import sha256


def asset(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Bundle asset escapes source directory")
    return path


def shapes(manifest, sequence, previous, objects):
    l, v, p = (manifest[key] for key in ("language", "vision", "pixel"))
    h, layers, kv, d = (l[key] for key in ("hidden_size", "layers", "kv_heads", "head_dim"))
    gh, gw = v["height"] // v["patch_size"], v["width"] // v["patch_size"]
    return {
        "language": {"input_ids": [1, sequence], "position_ids": [3, 1, sequence],
                     "image_features": [1, sequence, h], "deepstack": [3, 1, sequence, h],
                     "visual_mask": [1, sequence, 1], "past": [layers, 2, kv, previous, d]},
        "pixel": {"image": [1, 3, p["image_size"], p["image_size"]],
                  **{f"visual_{i}": [1, v["hidden_size"], gh, gw] for i in range(4)},
                  "t_seg": [objects, h], "boxes": [objects, 4]},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", type=Path, default=Path("cache/wevisone/2B/onnx"))
    p.add_argument("--output", type=Path, default=Path("cache/wevisone/2B/engines"))
    p.add_argument("--build", type=Path, default=Path("build/trt10-release"))
    p.add_argument("--workspace-mib", type=int, default=4096)
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--max-objects", type=int, help="Pixel profile object limit (default: min(4, export limit)); no weight re-export needed")
    p.add_argument("--graphs", nargs="+", choices=["vision", "language", "pixel"],
                   help="Rebuild selected graphs; unselected engines require matching hashes/profiles in an existing bundle")
    a = p.parse_args()
    if a.workspace_mib <= 0 or a.device_id < 0:
        p.error("Workspace must be positive and device ID nonnegative")
    if a.onnx.resolve() == a.output.resolve():
        p.error("ONNX and engine directories must differ")
    manifest = json.loads((a.onnx / "bundle.json").read_text(encoding="utf-8"))
    if manifest["format"] != "cvinfer-wevisone-v1" or set(manifest["graphs"]) != {"vision", "language", "pixel"}:
        raise ValueError("Unsupported WeVisOne export bundle")
    export_limit = manifest["pixel"]["max_objects"]
    object_limit = a.max_objects if a.max_objects is not None else min(4, export_limit)
    if not 1 <= object_limit <= export_limit:
        p.error("--max-objects must be within the exported object limit")
    manifest["pixel"]["export_max_objects"] = export_limit
    manifest["pixel"]["max_objects"] = object_limit
    if sha256(a.onnx / "tokenizer_native.json") != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer SHA256 differs from export manifest")
    for record in manifest["graphs"].values():
        for relative, expected in record["files"].items():
            if sha256(asset(a.onnx, relative)) != expected:
                raise ValueError("ONNX/external weights SHA256 mismatch: " + relative)
    a.output.mkdir(parents=True, exist_ok=True)
    previous_path = a.output / "bundle.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else {}
    previous_path.unlink(missing_ok=True)
    runtime_handles = load_runtime(a.build.resolve())
    dll = C.CDLL(str(a.build.resolve() / "tensorrt_builder/alanxinn_tensorrt_builder.dll"))
    config = BuilderConfig()
    bind(dll, "trt_builder_config_init", None, C.POINTER(BuilderConfig))(C.byref(config))
    config.workspace_mib, config.device_id, config.enable_tf32 = a.workspace_mib, a.device_id, 0
    run = bind(dll, "trt_builder_build_onnx", C.c_int, C.c_char_p, C.c_char_p,
               C.POINTER(BuilderConfig), C.POINTER(Profile), C.c_size_t)
    error = bind(dll, "trt_builder_last_error", C.c_char_p)
    l, pixel = manifest["language"], manifest["pixel"]
    limits = [shapes(manifest, 1, 1, 1),
              shapes(manifest, min(256, l["max_prefill"]), 1, min(4, pixel["max_objects"])),
              shapes(manifest, l["max_prefill"], l["max_context"], pixel["max_objects"])]
    for name, record in manifest["graphs"].items():
        precision = "fp16" if name != "pixel" and l["cache_dtype"] == "fp16" else "fp32"
        strongly_typed = name != "pixel"
        config.precision = 0
        config.strict_types = 2 if strongly_typed else 0
        profile = None
        if name in ("language", "pixel"):
            def encode(values):
                return ",".join(key + ":" + "x".join(map(str, shape)) for key, shape in values[name].items()).encode()
            profile = Profile(*(encode(limit) for limit in limits))
        engine = a.output.resolve() / (name + ".engine")
        profile_record = [limit[name] for limit in limits] if profile else []
        if a.graphs and name not in a.graphs:
            old = previous.get("graphs", {}).get(name, {})
            if (old.get("files") != record["files"] or old.get("profiles") != profile_record
                    or old.get("precision") != precision or old.get("strongly_typed", False) != strongly_typed
                    or not engine.is_file() or old.get("engine_sha256") != sha256(engine)):
                raise ValueError("Unselected engine lacks matching SHA256/profile receipt: " + name)
            record.update({key: old[key] for key in ("engine", "engine_sha256", "precision", "profiles", "workspace_mib")})
            record["strongly_typed"] = strongly_typed
            manifest.update(status="PARTIAL_ENGINES_BUILT")
            previous_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print("Reusing verified", name, flush=True)
            continue
        print("Building", name, flush=True)
        status = run(str(asset(a.onnx, record["onnx"])).encode("utf-8"), str(engine).encode("utf-8"),
                     C.byref(config), C.byref(profile) if profile else None, int(profile is not None))
        if status:
            raise RuntimeError(error().decode("utf-8"))
        record.update(engine=engine.name, engine_sha256=sha256(engine),
                      precision=precision, strongly_typed=strongly_typed, profiles=profile_record,
                      workspace_mib=a.workspace_mib)
        # Preserve verified progress after each graph. An interrupted/failed
        # build remains explicitly non-loadable until all three graphs succeed.
        manifest.update(status="PARTIAL_ENGINES_BUILT")
        previous_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.copy2(a.onnx / "tokenizer_native.json", a.output / "tokenizer_native.json")
    if (a.onnx / "processor").is_dir():
        shutil.copytree(a.onnx / "processor", a.output / "processor", dirs_exist_ok=True)
    manifest.update(status="ENGINE_BUILT_REAL_VALIDATION_REQUIRED", workspace_mib=a.workspace_mib,
                    device_id=a.device_id, converter="cvinfer TensorRT10 native builder")
    (a.output / "bundle.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Built bundle", a.output)


if __name__ == "__main__":
    main()
