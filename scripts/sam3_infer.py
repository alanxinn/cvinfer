"""Run the native SAM3 TensorRT adapter with a box, positive/negative clicks, or text."""
import argparse
import ctypes as C
import json
from pathlib import Path
import numpy as np
from trt10_runtime import bind, load_runtime


class MaskInfo(C.Structure):
    _fields_ = [(k, C.c_float) for k in ["x1", "y1", "x2", "y2", "score"]] + [
        ("width", C.c_int), ("height", C.c_int), ("bytes", C.c_size_t)]


class Sam3Session:
    """Native session; images and points use original-image pixel coordinates."""
    def __init__(self, engines, build="build/trt10-release", confidence=.3, mask_threshold=.5):
        self.handles = load_runtime(Path(build).resolve())
        self.dll = C.CDLL(str(Path(build).resolve() / "bin/alanxinn_cvinfer_sam3.dll"))
        ptr = C.c_void_p
        self.error = bind(self.dll, "sam3_last_error", C.c_char_p)
        self.create = bind(self.dll, "sam3_model_create", C.c_int, C.c_char_p, C.c_float, C.c_float, C.c_int, C.POINTER(ptr))
        self.destroy = bind(self.dll, "sam3_model_destroy", None, ptr)
        self.set_image_fn = bind(self.dll, "sam3_model_set_image", C.c_int, ptr, ptr, C.c_int, C.c_int, C.c_int)
        self.box_fn = bind(self.dll, "sam3_model_set_box", C.c_int, ptr, C.c_float, C.c_float, C.c_float, C.c_float)
        self.point_fn = bind(self.dll, "sam3_model_add_point", C.c_int, ptr, C.c_float, C.c_float, C.c_int)
        self.clear_fn = bind(self.dll, "sam3_model_clear_points", C.c_int, ptr)
        self.predict_fn = bind(self.dll, "sam3_model_predict", C.c_int, ptr, C.POINTER(ptr))
        self.text_fn = bind(self.dll, "sam3_model_predict_text", C.c_int, ptr, C.c_char_p, C.POINTER(ptr))
        self.tokens_fn = bind(self.dll, "sam3_model_predict_tokens", C.c_int, ptr, ptr, ptr, C.c_size_t, C.POINTER(ptr))
        self.count_fn = bind(self.dll, "sam3_result_count", C.c_size_t, ptr)
        self.info_fn = bind(self.dll, "sam3_result_get_mask", C.c_int, ptr, C.c_size_t, C.POINTER(MaskInfo))
        self.copy_fn = bind(self.dll, "sam3_result_copy_mask", C.c_int, ptr, C.c_size_t, ptr, C.c_size_t, C.POINTER(C.c_size_t))
        self.result_destroy = bind(self.dll, "sam3_result_destroy", None, ptr)
        manifest = Path(engines) / "bundle.json"
        self.bundle = json.loads(manifest.read_text()) if manifest.exists() else {"max_points": 32}
        self.model = ptr()
        self.check(self.create(str(Path(engines).resolve()).encode(), confidence, mask_threshold,
                               self.bundle["max_points"], C.byref(self.model)))
    def check(self, status):
        if status:
            raise RuntimeError(self.error().decode("utf-8", errors="replace"))
    def close(self):
        if self.model:
            self.destroy(self.model)
            self.model = C.c_void_p()
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()
    def set_image(self, image):
        image = np.ascontiguousarray(image, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a 3-channel BGR image")
        self.check(self.set_image_fn(self.model, image.ctypes.data, image.shape[1], image.shape[0], 3))
    def set_box(self, x1, y1, x2, y2):
        self.check(self.box_fn(self.model, x1, y1, x2, y2))
    def add_point(self, x, y, positive=True):
        self.check(self.point_fn(self.model, x, y, int(positive)))
    def clear_points(self):
        self.check(self.clear_fn(self.model))
    def _read(self, result):
        try:
            values = []
            for i in range(self.count_fn(result)):
                info = MaskInfo()
                self.check(self.info_fn(result, i, C.byref(info)))
                if info.bytes != info.width * info.height:
                    raise ValueError("Native SAM3 mask size mismatch")
                mask = np.zeros((info.height, info.width), dtype=np.uint8)
                copied = C.c_size_t()
                self.check(self.copy_fn(result, i, mask.ctypes.data, mask.nbytes, C.byref(copied)))
                assert copied.value == mask.nbytes
                values.append({"box": [info.x1, info.y1, info.x2, info.y2], "score": info.score, "mask": mask})
            return values
        finally:
            self.result_destroy(result)
    def predict(self):
        result = C.c_void_p()
        self.check(self.predict_fn(self.model, C.byref(result)))
        return self._read(result)
    def predict_text(self, text):
        result = C.c_void_p()
        self.check(self.text_fn(self.model, text.encode(), C.byref(result)))
        return self._read(result)
    def predict_tokens(self, ids, attention):
        ids, attention = np.ascontiguousarray(ids, dtype=np.int64), np.ascontiguousarray(attention, dtype=np.int64)
        if ids.size != attention.size:
            raise ValueError("Token and attention lengths differ")
        result = C.c_void_p()
        self.check(self.tokens_fn(self.model, ids.ctypes.data, attention.ctypes.data, ids.size, C.byref(result)))
        return self._read(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engines", type=Path, default=Path("cache/sam3/engines"))
    p.add_argument("--build", type=Path, default=Path("build/trt10-release"))
    p.add_argument("--image", type=Path, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--box", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    mode.add_argument("--text")
    p.add_argument("--positive", type=float, nargs=2, action="append", default=[])
    p.add_argument("--negative", type=float, nargs=2, action="append", default=[])
    p.add_argument("--tokenizer", help="Official HF tokenizer repository/local path; enables Unicode through encoded-token API")
    p.add_argument("--output", type=Path, default=Path("cache/sam3/result"))
    p.add_argument("--confidence", type=float, default=.3)
    a = p.parse_args()
    if a.text and (a.positive or a.negative):
        p.error("Points refine --box mode; text concept masks are a separate prediction mode")
    import cv2
    image = cv2.imread(str(a.image))
    if image is None:
        p.error("Image could not be read")
    a.output.mkdir(parents=True, exist_ok=True)
    with Sam3Session(a.engines, a.build, a.confidence) as session:
        session.set_image(image)
        if a.box:
            session.set_box(*a.box)
            stages = [("box", session.predict())]
            for label, points in [("positive", a.positive), ("negative", a.negative)]:
                for i, point in enumerate(points):
                    session.add_point(*point, positive=label == "positive")
                    stages.append((f"{label}-{i + 1}", session.predict()))
        elif a.tokenizer:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(a.tokenizer)
            encoded = tokenizer(a.text, padding="max_length", truncation=True,
                                max_length=session.bundle["text_length"], return_tensors="np")
            stages = [("text", session.predict_tokens(encoded.input_ids, encoded.attention_mask))]
        else:
            stages = [("text", session.predict_text(a.text))]
        report = []
        for stage, masks in stages:
            canvas = image.copy()
            items = []
            for i, item in enumerate(masks):
                path = a.output / f"{stage}-{i}.png"
                assert cv2.imwrite(str(path), item["mask"])
                selected = item["mask"] != 0
                color = np.array([(50 + 83 * i) % 255, (130 + 37 * i) % 255, (90 + 151 * i) % 255])
                canvas[selected] = (.6 * canvas[selected] + .4 * color).astype(np.uint8)
                items.append({"box": item["box"], "score": item["score"], "mask": str(path.resolve())})
            assert cv2.imwrite(str(a.output / (stage + ".jpg")), canvas)
            report.append({"stage": stage, "masks": items})
        (a.output / "result.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
