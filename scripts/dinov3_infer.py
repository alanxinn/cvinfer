"""Native DINOv3 image feature inference; save embeddings/tokens/maps as NumPy arrays."""
import argparse
import ctypes as C
import json
from pathlib import Path
import numpy as np
from trt10_runtime import bind, load_runtime

TENSOR_NAMES = ["global_embedding", "normalized_embedding", "patch_tokens", "feature_map"]


class TensorInfo(C.Structure):
    _fields_ = [("rank", C.c_int32), ("shape", C.c_int64 * 8), ("elements", C.c_size_t)]


class DinoV3Session:
    def __init__(self, engine, build="build/trt10-release"):
        self.handles = load_runtime(Path(build).resolve())
        self.dll = C.CDLL(str(Path(build).resolve() / "bin/alanxinn_cvinfer_dinov3.dll"))
        ptr = C.c_void_p
        self.error = bind(self.dll, "dinov3_last_error", C.c_char_p)
        create = bind(self.dll, "dinov3_model_create", C.c_int, C.c_char_p, C.POINTER(ptr))
        self.destroy = bind(self.dll, "dinov3_model_destroy", None, ptr)
        input_size = bind(self.dll, "dinov3_model_input_size", C.c_int, ptr, C.POINTER(C.c_int32), C.POINTER(C.c_int32))
        self.bgr = bind(self.dll, "dinov3_model_infer_bgr", C.c_int, ptr, ptr, C.c_int32, C.c_int32, C.c_int32, C.POINTER(ptr))
        self.chw = bind(self.dll, "dinov3_model_infer_chw", C.c_int, ptr, ptr, C.c_size_t, C.POINTER(ptr))
        self.destroy_result = bind(self.dll, "dinov3_result_destroy", None, ptr)
        self.info = bind(self.dll, "dinov3_result_tensor_info", C.c_int, ptr, C.c_char_p, C.POINTER(TensorInfo))
        self.copy = bind(self.dll, "dinov3_result_copy_tensor", C.c_int, ptr, C.c_char_p, ptr, C.c_size_t, C.POINTER(C.c_size_t))
        self.model = ptr()
        self.check(create(str(Path(engine).resolve()).encode(), C.byref(self.model)))
        width, height = C.c_int32(), C.c_int32()
        self.check(input_size(self.model, C.byref(width), C.byref(height)))
        self.input_width, self.input_height = width.value, height.value
    def check(self, status):
        if status:
            raise RuntimeError(self.error().decode("utf-8", errors="replace"))
    def close(self):
        if self.model:
            self.destroy(self.model); self.model = C.c_void_p()
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()
    def _read(self, result):
        try:
            features = {}
            for name in TENSOR_NAMES:
                info = TensorInfo()
                self.check(self.info(result, name.encode(), C.byref(info)))
                if not 1 <= info.rank <= 8:
                    raise ValueError("Invalid DINOv3 tensor rank")
                shape = list(info.shape[:info.rank])
                if any(d <= 0 for d in shape) or int(np.prod(shape)) != info.elements:
                    raise ValueError("DINOv3 tensor shape/element count mismatch")
                data = np.empty(shape, dtype=np.float32)
                copied = C.c_size_t()
                self.check(self.copy(result, name.encode(), data.ctypes.data, data.size, C.byref(copied)))
                assert copied.value == data.size
                features[name] = data
            return features
        finally:
            self.destroy_result(result)
    def infer_bgr(self, image):
        image = np.asarray(image)
        if image.dtype != np.uint8:
            raise ValueError("DINOv3 BGR pixels must be uint8; use infer_chw for normalized float input")
        image = np.ascontiguousarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or not image.size:
            raise ValueError("DINOv3 requires a non-empty 3-channel BGR image")
        result = C.c_void_p()
        self.check(self.bgr(self.model, image.ctypes.data, image.shape[1], image.shape[0], 3, C.byref(result)))
        return self._read(result)
    def infer_chw(self, values):
        values = np.ascontiguousarray(values, dtype=np.float32)
        if values.size != 3 * self.input_width * self.input_height:
            raise ValueError("DINOv3 CHW input element count does not match the engine")
        result = C.c_void_p()
        self.check(self.chw(self.model, values.ctypes.data, values.size, C.byref(result)))
        return self._read(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", type=Path, default=Path("cache/dinov3/engine/model.engine"))
    p.add_argument("--build", type=Path, default=Path("build/trt10-release"))
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--compare", type=Path, help="Optional second image for cosine similarity")
    p.add_argument("--output", type=Path, default=Path("cache/dinov3/result"))
    p.add_argument("--visualize", action="store_true", help="Save a PCA feature-map preview, not a semantic mask")
    a = p.parse_args()
    import cv2
    image = cv2.imread(str(a.image))
    if image is None:
        p.error("Image cannot be read")
    a.output.mkdir(parents=True, exist_ok=True)
    with DinoV3Session(a.engine, a.build) as session:
        features = session.infer_bgr(image)
        np.savez_compressed(a.output / "features.npz", **features)
        report = {"image": str(a.image.resolve()), "input_size": [session.input_height, session.input_width],
                  "features": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in features.items()},
                  "output": str((a.output / "features.npz").resolve())}
        bundle = a.engine.parent / "bundle.json"
        if bundle.exists():
            metadata = json.loads(bundle.read_text())
            report["model"] = {key: metadata.get(key) for key in ["model_type", "checkpoint", "synthetic", "precision"]}
        if a.compare:
            other = cv2.imread(str(a.compare))
            if other is None:
                p.error("Comparison image cannot be read")
            vector = session.infer_bgr(other)["global_embedding"].astype(np.float64).ravel()
            first = features["global_embedding"].astype(np.float64).ravel()
            denominator = np.linalg.norm(first) * np.linalg.norm(vector)
            if denominator == 0:
                raise ValueError("Cosine similarity is undefined for zero feature vectors")
            report["cosine_similarity"] = float(np.clip(np.dot(first, vector) / denominator, -1, 1))
        if a.visualize:
            fmap = features["feature_map"][0].transpose(1, 2, 0)
            flat = fmap.reshape(-1, fmap.shape[-1]); flat = flat - flat.mean(0)
            _, _, projection = np.linalg.svd(flat, full_matrices=False)
            rgb = flat @ projection[:3].T
            rgb = (rgb - rgb.min(0)) / np.maximum(np.ptp(rgb, axis=0), 1e-8)
            if rgb.shape[1] < 3:
                rgb = np.pad(rgb, ((0, 0), (0, 3 - rgb.shape[1])))
            rgb = (rgb.reshape(fmap.shape[0], fmap.shape[1], 3) * 255).astype(np.uint8)
            preview = cv2.resize(rgb[:, :, ::-1], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            path = a.output / "feature_map_pca.png"
            if not cv2.imwrite(str(path), preview):
                raise RuntimeError("Could not save DINOv3 feature preview")
            report["pca_preview"] = str(path.resolve())
        (a.output / "result.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
