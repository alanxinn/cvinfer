"""Build engines with the project's C ABI and verify real image inference.

Each engine build/inference runs in a separate process so CUDA/torch versions
cannot contaminate one another and failures remain visible in validation.json.
"""
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


class BuilderConfig(C.Structure):
    _fields_ = [("precision", C.c_int), ("workspace_mib", C.c_int64),
                ("device_id", C.c_int), ("enable_tf32", C.c_int),
                ("enable_sparse_weights", C.c_int), ("strict_types", C.c_int)]


class ModelConfig(C.Structure):
    _fields_ = [("input_width", C.c_int), ("input_height", C.c_int), ("class_count", C.c_int),
                ("confidence", C.c_float), ("nms", C.c_float), ("mask", C.c_float),
                ("agnostic", C.c_int), ("input_name", C.c_char_p), ("output_name", C.c_char_p),
                ("prototype_name", C.c_char_p), ("preprocess", C.c_int), ("postprocess", C.c_int),
                ("keypoint_count", C.c_int), ("keypoint_dims", C.c_int), ("backend", C.c_int)]


class Box(C.Structure):
    _fields_ = [(f, C.c_float) for f in ["x1", "y1", "x2", "y2", "score"]] + [("class_id", C.c_int)]


class Segmentation(C.Structure):
    _fields_ = [("box", Box), ("width", C.c_int), ("height", C.c_int), ("bytes", C.c_size_t)]


class Pose(C.Structure):
    _fields_ = [("box", Box), ("count", C.c_int), ("dims", C.c_int), ("bytes", C.c_size_t)]


class Keypoint(C.Structure):
    _fields_ = [("x", C.c_float), ("y", C.c_float), ("score", C.c_float)]


class Timing(C.Structure):
    _fields_ = [(f, C.c_double) for f in ["preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"]]


def bind(dll, name, restype, *args):
    f = getattr(dll, name)
    f.restype, f.argtypes = restype, list(args)
    return f


def load_runtime(build):
    # Keep AddDllDirectory handles alive for the entire worker lifetime.
    paths = [build / "bin", build / "bintrt", build / "builder_runtime", build / "tensorrt_builder"]
    handles = [os.add_dll_directory(str(p)) for p in paths if p.exists()]
    os.environ["PATH"] = ";".join(str(p) for p in paths) + ";" + os.environ.get("PATH", "")
    return handles


def build_engine(build, record, engine):
    dll = C.CDLL(str(build / "tensorrt_builder/alanxinn_tensorrt_builder.dll"))
    config = BuilderConfig()
    bind(dll, "trt_builder_config_init", None, C.POINTER(BuilderConfig))(C.byref(config))
    config.precision = 1 if record["precision"] == "fp16" else 0
    config.workspace_mib = 2048
    config.enable_tf32 = 0 if record["type"] in (3, 4) else 1
    start = time.monotonic()
    status = bind(dll, "trt_builder_build_onnx", C.c_int, C.c_char_p, C.c_char_p,
                  C.POINTER(BuilderConfig), C.c_void_p, C.c_size_t)(
                      record["onnx"].encode(), str(engine).encode(), C.byref(config), None, 0)
    if status:
        raise RuntimeError(bind(dll, "trt_builder_last_error", C.c_char_p)().decode())
    return {"engine": str(engine), "engine_bytes": engine.stat().st_size,
            "onnx_sha256": hashlib.sha256(Path(record["onnx"]).read_bytes()).hexdigest(),
            "build_seconds": time.monotonic() - start, "precision": record["precision"]}


def infer(build, record, engine, image_path, output, compute):
    import cv2
    import numpy as np
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Image cannot be read: {image_path}")
    dll = C.CDLL(str(build / "bin/alanxinn_cvinfer_trt.dll"))
    ptr, size = C.c_void_p, C.c_size_t
    error = bind(dll, "cvinfer_last_error", C.c_char_p)

    def check(status):
        if status:
            raise RuntimeError(error().decode())

    config = ModelConfig()
    bind(dll, "cvinfer_model_config_v2_init", None, C.POINTER(ModelConfig))(C.byref(config))
    config.preprocess = compute
    # The existing NVIDIA backend has CUDA decoders for YOLO11 det/seg and
    # RF-DETR. End-to-end YOLO26 and pose use the public CPU decoder while
    # retaining CUDA preprocessing and TensorRT GPU inference.
    config.postprocess = 0 if compute == 2 and record["type"] in (5, 6, 7, 8) else compute
    model, result = ptr(), ptr()
    check(bind(dll, "cvinfer_model_create_v2", C.c_int, C.c_char_p, C.c_int, C.POINTER(ModelConfig),
               C.POINTER(ptr))(str(engine).encode(), record["type"], C.byref(config), C.byref(model)))
    run = bind(dll, "cvinfer_model_infer_bgr", C.c_int, ptr, ptr, C.c_int, C.c_int, C.c_int, C.POINTER(ptr))
    destroy_result = bind(dll, "cvinfer_result_destroy", None, ptr)
    destroy_model = bind(dll, "cvinfer_model_destroy", None, ptr)
    count_fn = bind(dll, "cvinfer_result_count", size, ptr)
    get_det = bind(dll, "cvinfer_result_get_detection", C.c_int, ptr, size, C.POINTER(Box))
    get_seg = bind(dll, "cvinfer_result_get_segmentation", C.c_int, ptr, size, C.POINTER(Segmentation))
    get_pose = bind(dll, "cvinfer_result_get_pose", C.c_int, ptr, size, C.POINTER(Pose))
    get_mask = bind(dll, "cvinfer_result_copy_mask", C.c_int, ptr, size, ptr, size, C.POINTER(size))
    get_points = bind(dll, "cvinfer_result_copy_keypoints", C.c_int, ptr, size, ptr, size, C.POINTER(size))
    get_timing = bind(dll, "cvinfer_result_get_timing", C.c_int, ptr, C.POINTER(Timing))
    check(bind(dll, "cvinfer_model_set_timing_enabled", C.c_int, ptr, C.c_int)(model, 1))
    try:
        # Exercise model reuse and buffer lifetime, including warmup.
        timings, counts = [], []
        for iteration in range(4):
            check(run(model, image.ctypes.data, image.shape[1], image.shape[0], 3, C.byref(result)))
            count = count_fn(result)
            if count == 0:
                raise AssertionError("No objects detected in the standard bus image")
            counts.append(count)
            t = Timing()
            check(get_timing(result, C.byref(t)))
            if iteration:
                timings.append({field: getattr(t, field) for field, _ in t._fields_})
            if iteration < 3:
                destroy_result(result)
                result = ptr()
        items, nonempty_masks, valid_points = [], 0, 0
        canvas = image.copy()
        for i in range(counts[-1]):
            model_type = record["type"]
            if model_type in (2, 4, 7):
                info = Segmentation()
                check(get_seg(result, i, C.byref(info)))
                box = info.box
                mask = np.zeros(info.bytes, dtype=np.uint8)
                copied = size()
                check(get_mask(result, i, mask.ctypes.data, mask.nbytes, C.byref(copied)))
                assert copied.value == mask.nbytes == info.width * info.height
                assert np.isin(mask, [0, 255]).all(), "Mask contains unexpected values"
                pixels = int(np.count_nonzero(mask))
                nonempty_masks += pixels > 0
                # Native masks may cover the full image or only a box ROI.
                mask = mask.reshape(info.height, info.width)
                cv2.imwrite(str(output / f"{record['name']}-{compute}-mask-{i}.png"), mask)
                left, top = math.floor(box.x1), math.floor(box.y1)
                right, bottom = min(image.shape[1], left + info.width), min(image.shape[0], top + info.height)
                x, y = max(0, left), max(0, top)
                if right > x and bottom > y:
                    selected = mask[y - top:bottom - top, x - left:right - left] != 0
                    roi = canvas[y:bottom, x:right]
                    tint = np.array([(37 * i + 50) % 255, (83 * i + 100) % 255, (151 * i + 80) % 255])
                    roi[selected] = (.6 * roi[selected] + .4 * tint).astype(np.uint8)
            elif model_type in (5, 8):
                info = Pose()
                check(get_pose(result, i, C.byref(info)))
                box = info.box
                assert info.count == 17 and info.dims == 3
                points = (Keypoint * info.count)()
                copied = size()
                check(get_points(result, i, C.cast(points, ptr), C.sizeof(points), C.byref(copied)))
                assert copied.value == C.sizeof(points)
                for point in points:
                    assert all(math.isfinite(v) for v in [point.x, point.y, point.score])
                    if point.score >= .25:
                        valid_points += 1
                        cv2.circle(canvas, (int(point.x), int(point.y)), 3, (0, 255, 255), -1)
            else:
                box = Box()
                check(get_det(result, i, C.byref(box)))
            values = {field: getattr(box, field) for field, _ in box._fields_}
            assert all(math.isfinite(v) for v in values.values())
            assert 0 <= box.score <= 1 and box.x2 > box.x1 and box.y2 > box.y1
            assert box.class_id >= 0
            cv2.rectangle(canvas, (round(box.x1), round(box.y1)), (round(box.x2), round(box.y2)), (0, 255, 0), 2)
            cv2.putText(canvas, f"{box.class_id} {box.score:.3f}", (round(box.x1), max(15, round(box.y1))),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
            items.append(values)
        if record["type"] in (2, 4, 7):
            assert nonempty_masks > 0, "All segmentation masks are empty"
        if record["type"] in (5, 8):
            assert valid_points > 0, "All pose keypoints are below threshold"
        assert len(set(counts)) == 1, "Repeated inference returned inconsistent object counts"
        output.mkdir(exist_ok=True)
        rendered = output / f"{record['name']}-{compute}.jpg"
        assert cv2.imwrite(str(rendered), canvas)
        return {"count": counts[-1], "preprocess_backend": config.preprocess,
                "postprocess_backend": config.postprocess,
                "nonempty_masks": nonempty_masks, "valid_keypoints": valid_points,
                "objects": items, "mean_timing": {f: sum(t[f] for t in timings) / len(timings) for f in timings[0]},
                "image": str(rendered)}
    finally:
        if result:
            destroy_result(result)
        destroy_model(model)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, default=Path("cache/trt10-models/models.json"))
    p.add_argument("--build", type=Path, default=Path("build/trt10-release"))
    p.add_argument("--output", type=Path, default=Path("cache/trt10-validation"))
    p.add_argument("--compute", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--worker", choices=["build", "infer"])
    p.add_argument("--name")
    p.add_argument("--reuse-engines", action="store_true")
    a = p.parse_args()
    a.models, a.build, a.output = a.models.resolve(), a.build.resolve(), a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    records = json.loads(a.models.read_text())["models"]
    if a.worker:
        handles = load_runtime(a.build)
        record = next(m for m in records if m["name"] == a.name)
        engine = a.output / (a.name + ".engine")
        data = build_engine(a.build, record, engine) if a.worker == "build" else infer(
            a.build, record, engine, a.models.parent / "bus.jpg", a.output, 2 if a.compute == "cuda" else 1)
        (a.output / f"{a.name}-{a.worker}-{a.compute}.json").write_text(json.dumps(data, indent=2))
        print(json.dumps(data), flush=True)
        return
    results = []
    for record in records:
        if a.name and record["name"] != a.name:
            continue
        row = {"model": record["name"], "status": "FAIL", "compute": a.compute}
        for stage in ["build", "infer"]:
            if stage == "build" and a.reuse_engines and (a.output / (record["name"] + ".engine")).exists():
                # Build metadata belongs to the original CPU run even when
                # the second pass exercises CUDA preprocessing/postprocessing.
                metadata = a.output / f"{record['name']}-build-cpu.json"
                if metadata.exists():
                    row["build"] = json.loads(metadata.read_text())
                    if row["build"].get("onnx_sha256", record["onnx_sha256"]) != record["onnx_sha256"]:
                        raise ValueError("Cached engine belongs to a different ONNX; remove --reuse-engines")
                continue
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", stage, "--name", record["name"],
                       "--models", str(a.models), "--build", str(a.build), "--output", str(a.output), "--compute", a.compute]
            print(stage.upper(), record["name"], flush=True)
            log = a.output / f"{record['name']}-{stage}-{a.compute}.log"
            with open(log, "w", encoding="utf-8") as stream:
                try:
                    completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
                    row[stage + "_exit"] = completed.returncode
                except subprocess.TimeoutExpired:
                    row[stage + "_exit"] = "timeout"
            if row[stage + "_exit"] != 0:
                row["error_log"] = str(log)
                break
            row[stage] = json.loads((a.output / f"{record['name']}-{stage}-{a.compute}.json").read_text())
        else:
            row["status"] = "PASS"
        results.append(row)
        (a.output / f"validation-{a.compute}.json").write_text(json.dumps(results, indent=2))
        print(row["status"], record["name"], flush=True)
    if not results or any(r["status"] != "PASS" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
