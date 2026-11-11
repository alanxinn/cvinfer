"""Run the native WeVisOne C ABI for all eleven tasks and save JSON/PNG/FP32 alpha."""
import argparse
import ctypes as C
import json
from pathlib import Path
from trt10_runtime import bind, load_runtime
from wevisone_checkpoint import TASKS


class Box(C.Structure):
    _fields_ = [(key, C.c_float) for key in ("x1", "y1", "x2", "y2")]


class Options(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("max_new_tokens", C.c_int32),
                ("temperature", C.c_float), ("top_p", C.c_float), ("top_k", C.c_int32),
                ("repetition_penalty", C.c_float), ("mask_threshold", C.c_float),
                ("seed", C.c_uint64), ("decode_pixels", C.c_int32)]


class Summary(C.Structure):
    _fields_ = [(key, C.c_int32) for key in ("count", "rejected", "truncated")] + [
        ("latency_ms", C.c_double)] + [(key, C.c_size_t) for key in ("annotations", "masks", "absent_labels")]


class Annotation(C.Structure):
    _fields_ = [("box", Box), ("x", C.c_float), ("y", C.c_float),
                ("is_point", C.c_int32), ("mask_index", C.c_int32)]


class MaskInfo(C.Structure):
    _fields_ = [("box", Box), ("width", C.c_int32), ("height", C.c_int32),
                ("alpha_elements", C.c_size_t), ("binary_bytes", C.c_size_t)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("cache/wevisone/2B/engines"))
    p.add_argument("--build", type=Path, default=Path("build/trt10-release"))
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--task", choices=TASKS, default="detection")
    p.add_argument("--query", action="append", default=[], help="Repeat for multiple detection categories")
    p.add_argument("--prompt")
    p.add_argument("--box", type=float, nargs=4, action="append", default=[], metavar=("X1", "Y1", "X2", "Y2"))
    p.add_argument("--box-matting", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--top-p", type=float, default=.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--repetition-penalty", type=float, default=1)
    p.add_argument("--mask-threshold", type=float, default=.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-pixels", action="store_true")
    p.add_argument("--output", type=Path, default=Path("wevisone_result"))
    a = p.parse_args()
    if a.box_matting and not a.box:
        p.error("--box-matting requires --box")
    import cv2
    import numpy as np
    image = cv2.imdecode(np.fromfile(a.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot read image")
    image = np.ascontiguousarray(image)
    runtime_handles = load_runtime(a.build.resolve())
    dll = C.CDLL(str(a.build.resolve() / "bin/alanxinn_cvinfer_wevisone.dll"))
    error = bind(dll, "wevisone_last_error", C.c_char_p)

    def check(status):
        if status:
            raise RuntimeError(error().decode("utf-8"))

    create = bind(dll, "wevisone_model_create", C.c_int, C.c_char_p, C.POINTER(C.c_void_p))
    destroy_model = bind(dll, "wevisone_model_destroy", None, C.c_void_p)
    destroy_result = bind(dll, "wevisone_result_destroy", None, C.c_void_p)
    set_image = bind(dll, "wevisone_model_set_image", C.c_int, C.c_void_p, C.c_void_p, C.c_int32, C.c_int32, C.c_int32)
    predict = bind(dll, "wevisone_model_predict", C.c_int, C.c_void_p, C.c_char_p, C.c_char_p, C.c_char_p,
                   C.POINTER(Box), C.c_size_t, C.POINTER(Options), C.POINTER(C.c_void_p))
    matte = bind(dll, "wevisone_model_matte_boxes", C.c_int, C.c_void_p, C.POINTER(Box), C.c_size_t,
                 C.POINTER(Options), C.POINTER(C.c_void_p))
    summary_fn = bind(dll, "wevisone_result_summary", C.c_int, C.c_void_p, C.POINTER(Summary))
    text_fn = bind(dll, "wevisone_result_copy_text", C.c_int, C.c_void_p, C.c_int32, C.c_size_t,
                   C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t))
    annotation_fn = bind(dll, "wevisone_result_annotation", C.c_int, C.c_void_p, C.c_size_t, C.POINTER(Annotation))
    mask_fn = bind(dll, "wevisone_result_mask_info", C.c_int, C.c_void_p, C.c_size_t, C.POINTER(MaskInfo))
    alpha_fn = bind(dll, "wevisone_result_copy_alpha", C.c_int, C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t))
    binary_fn = bind(dll, "wevisone_result_copy_binary", C.c_int, C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t))
    model, result = C.c_void_p(), C.c_void_p()
    options = Options()
    bind(dll, "wevisone_options_init", None, C.POINTER(Options))(C.byref(options))
    for key in ("max_new_tokens", "temperature", "top_p", "top_k", "repetition_penalty", "mask_threshold", "seed"):
        setattr(options, key, getattr(a, key))
    options.decode_pixels = int(not a.no_pixels)
    boxes = (Box * len(a.box))(*(Box(*box) for box in a.box))

    def save_png(name, pixels):
        ok, encoded = cv2.imencode(".png", pixels)
        if not ok:
            raise OSError("Cannot encode " + name)
        encoded.tofile(a.output / name)

    try:
        check(create(str(a.model.resolve()).encode("utf-8"), C.byref(model)))
        check(set_image(model, image.ctypes.data, image.shape[1], image.shape[0], 3))
        if a.box_matting:
            check(matte(model, boxes, len(boxes), C.byref(options), C.byref(result)))
        else:
            query = "\n".join(a.query) if a.task in ("detection", "point") else " ".join(a.query)
            check(predict(model, a.task.encode(), query.encode("utf-8"), a.prompt.encode("utf-8") if a.prompt else None,
                          boxes, len(boxes), C.byref(options), C.byref(result)))
        summary = Summary()
        check(summary_fn(result, C.byref(summary)))

        def text(field, index=0):
            size = C.c_size_t()
            check(text_fn(result, field, index, None, 0, C.byref(size)))
            buffer = C.create_string_buffer(size.value)
            check(text_fn(result, field, index, buffer, len(buffer), C.byref(size)))
            return buffer.value.decode("utf-8", errors="replace")

        metadata = {"task": a.task, "prompt": text(1), "response": text(0), "count": summary.count,
                    "rejected": bool(summary.rejected), "truncated": bool(summary.truncated),
                    "latency_ms": summary.latency_ms, "annotations": [], "masks": [],
                    "absent_labels": [text(3, i) for i in range(summary.absent_labels)]}
        a.output.mkdir(parents=True, exist_ok=True)
        overlay = image.copy()
        for i in range(summary.annotations):
            value = Annotation()
            check(annotation_fn(result, i, C.byref(value)))
            box = [value.box.x1, value.box.y1, value.box.x2, value.box.y2]
            metadata["annotations"].append({"label": text(2, i), "box": box, "point": [value.x, value.y] if value.is_point else None,
                                            "mask_index": value.mask_index})
            if value.is_point:
                cv2.circle(overlay, (int(value.x), int(value.y)), 4, (0, 255, 0), -1)
            else:
                cv2.rectangle(overlay, tuple(map(int, box[:2])), tuple(map(int, box[2:])), (0, 255, 0), 2)
        for i in range(summary.masks):
            info, copied = MaskInfo(), C.c_size_t()
            check(mask_fn(result, i, C.byref(info)))
            if info.alpha_elements != info.width * info.height:
                raise ValueError("Native alpha shape mismatch")
            alpha = np.empty((info.height, info.width), dtype=np.float32)
            check(alpha_fn(result, i, alpha.ctypes.data, alpha.size, C.byref(copied)))
            if copied.value != alpha.size:
                raise ValueError("Native alpha copy size mismatch")
            alpha.tofile(a.output / f"alpha_{i:03d}.f32")
            alpha_png = np.clip(np.rint(alpha * 255), 0, 255).astype(np.uint8)
            save_png(f"alpha_{i:03d}.png", alpha_png)
            rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = alpha_png
            save_png(f"cutout_{i:03d}.png", rgba)
            if info.binary_bytes:
                if info.binary_bytes != alpha.size:
                    raise ValueError("Native binary shape mismatch")
                binary = np.empty(alpha.shape, dtype=np.uint8)
                check(binary_fn(result, i, binary.ctypes.data, binary.size, C.byref(copied)))
                if copied.value != binary.size:
                    raise ValueError("Native binary copy size mismatch")
                save_png(f"mask_{i:03d}.png", binary)
            metadata["masks"].append({"width": info.width, "height": info.height,
                                      "box": [info.box.x1, info.box.y1, info.box.x2, info.box.y2],
                                      "alpha": f"alpha_{i:03d}.f32", "binary": f"mask_{i:03d}.png" if info.binary_bytes else None})
        save_png("overlay.png", overlay)
        (a.output / "result.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(metadata["response"])
    finally:
        if result:
            destroy_result(result)
        if model:
            destroy_model(model)


if __name__ == "__main__":
    main()
