"""Run native SAM2.1 image segmentation with box/points and iterative refinement."""
import argparse
import ctypes as C
import json
from pathlib import Path
import numpy as np
from trt10_runtime import bind, load_runtime


class MaskInfo(C.Structure):
    _fields_ = [(k, C.c_float) for k in ["x1", "y1", "x2", "y2", "score"]] + [
        ("width", C.c_int), ("height", C.c_int), ("bytes", C.c_size_t)]


class Sam2Session:
    """Native session; images and points use original-image pixel coordinates."""
    def __init__(self, engines, build="build/trt10-release", mask_threshold=.5):
        self.handles = load_runtime(Path(build).resolve())
        self.dll = C.CDLL(str(Path(build).resolve() / "bin/alanxinn_cvinfer_sam2.dll"))
        ptr = C.c_void_p
        self.error = bind(self.dll, "sam2_last_error", C.c_char_p)
        self.create = bind(self.dll, "sam2_model_create", C.c_int, C.c_char_p, C.c_float, C.c_int, C.POINTER(ptr))
        self.destroy = bind(self.dll, "sam2_model_destroy", None, ptr)
        self.set_image_fn = bind(self.dll, "sam2_model_set_image", C.c_int, ptr, ptr, C.c_int, C.c_int, C.c_int)
        self.box_fn = bind(self.dll, "sam2_model_set_box", C.c_int, ptr, C.c_float, C.c_float, C.c_float, C.c_float)
        self.point_fn = bind(self.dll, "sam2_model_add_point", C.c_int, ptr, C.c_float, C.c_float, C.c_int)
        self.clear_fn = bind(self.dll, "sam2_model_clear_points", C.c_int, ptr)
        self.clear_box_fn = bind(self.dll, "sam2_model_clear_box", C.c_int, ptr)
        self.clear_prompts_fn = bind(self.dll, "sam2_model_clear_prompts", C.c_int, ptr)
        self.remove_fn = bind(self.dll, "sam2_model_remove_point", C.c_int, ptr, C.c_size_t)
        self.predict_fn = bind(self.dll, "sam2_model_predict", C.c_int, ptr, C.c_int, C.POINTER(ptr))
        self.count_fn = bind(self.dll, "sam2_result_count", C.c_size_t, ptr)
        self.info_fn = bind(self.dll, "sam2_result_get_mask", C.c_int, ptr, C.c_size_t, C.POINTER(MaskInfo))
        self.copy_fn = bind(self.dll, "sam2_result_copy_mask", C.c_int, ptr, C.c_size_t, ptr, C.c_size_t, C.POINTER(C.c_size_t))
        self.result_destroy = bind(self.dll, "sam2_result_destroy", None, ptr)
        manifest = Path(engines) / "bundle.json"
        self.bundle = json.loads(manifest.read_text(encoding="utf-8"))
        if self.bundle.get("format") != "cvinfer-sam2.1-v1":
            raise ValueError("Expected a build_sam2_engines.py bundle")
        self.model = ptr()
        self.check(self.create(str(Path(engines).resolve()).encode(), mask_threshold,
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
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise ValueError("Expected a uint8 BGR image")
        image = np.ascontiguousarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a 3-channel BGR image")
        self.check(self.set_image_fn(self.model, image.ctypes.data, image.shape[1], image.shape[0], 3))
    def set_box(self, x1, y1, x2, y2):
        self.check(self.box_fn(self.model, x1, y1, x2, y2))
    def add_point(self, x, y, positive=True):
        self.check(self.point_fn(self.model, x, y, int(positive)))
    def clear_points(self):
        self.check(self.clear_fn(self.model))
    def clear_box(self):
        self.check(self.clear_box_fn(self.model))
    def clear_prompts(self):
        self.check(self.clear_prompts_fn(self.model))
    def remove_point(self, index):
        if index < 0:
            raise ValueError("Point index must be nonnegative")
        self.check(self.remove_fn(self.model, index))
    def _read(self, result):
        try:
            values = []
            for i in range(self.count_fn(result)):
                info = MaskInfo()
                self.check(self.info_fn(result, i, C.byref(info)))
                if info.bytes != info.width * info.height:
                    raise ValueError("Native SAM2.1 mask size mismatch")
                mask = np.zeros((info.height, info.width), dtype=np.uint8)
                copied = C.c_size_t()
                self.check(self.copy_fn(result, i, mask.ctypes.data, mask.nbytes, C.byref(copied)))
                assert copied.value == mask.nbytes
                values.append({"box": [info.x1, info.y1, info.x2, info.y2], "score": info.score, "mask": mask})
            return values
        finally:
            self.result_destroy(result)
    def predict(self, use_previous_mask=False):
        result = C.c_void_p()
        self.check(self.predict_fn(self.model, int(use_previous_mask), C.byref(result)))
        return self._read(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engines", type=Path, default=Path("cache/sam2.1/engines"))
    p.add_argument("--build", type=Path, default=Path("build/trt10-release"))
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--box", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    p.add_argument("--positive", type=float, nargs=2, action="append", default=[])
    p.add_argument("--negative", type=float, nargs=2, action="append", default=[])
    p.add_argument("--refine", action="store_true", help="Feed previous mask logits after each added click")
    p.add_argument("--output", type=Path, default=Path("cache/sam2.1/result"))
    a = p.parse_args()
    if not a.box and not (a.positive or a.negative):
        p.error("Provide a box or at least one point")
    import cv2
    image = cv2.imread(str(a.image))
    if image is None:
        p.error("Image could not be read")
    a.output.mkdir(parents=True, exist_ok=True)
    report = []
    with Sam2Session(a.engines, a.build) as session:
        session.set_image(image)

        def save(stage, refine=False):
            item = session.predict(refine)[0]
            mask_path = a.output / (stage + ".png")
            canvas = image.copy(); chosen = item["mask"] != 0
            canvas[chosen] = (.6 * canvas[chosen] + .4 * np.array([30, 210, 90])).astype(np.uint8)
            if not cv2.imwrite(str(mask_path), item["mask"]) or not cv2.imwrite(str(a.output / (stage + ".jpg")), canvas):
                raise IOError("Could not write SAM2 result images")
            report.append({"stage": stage, "score": item["score"], "box": item["box"], "mask": str(mask_path.resolve())})

        if a.box:
            session.set_box(*a.box); save("box")
        for label, points in [("positive", a.positive), ("negative", a.negative)]:
            for i, point in enumerate(points):
                session.add_point(*point, positive=label == "positive")
                save(f"{label}-{i + 1}", a.refine and bool(report))
    (a.output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
