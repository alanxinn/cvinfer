"""Validate native DINOv3 against official preprocessing + ONNX Runtime on RANDOM tiny models."""
import argparse
import ctypes as C
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from dinov3_infer import DinoV3Session, TensorInfo


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("cache/dinov3-tiny"))
    p.add_argument("--build", type=Path, default=Path("build/trt10-release"))
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    a = p.parse_args()
    from transformers import DINOv3ViTImageProcessor
    assert C.sizeof(TensorInfo) == 80
    processor = DINOv3ViTImageProcessor(size={"height": 64, "width": 64})
    rng = np.random.default_rng(123)
    images = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for h, w in [(43, 97), (113, 191), (17, 21), (64, 64), (1, 1)]]
    options = ort.SessionOptions(); options.intra_op_num_threads = 2
    reports = []
    for kind in ["vit", "convnext"]:
        root = a.root / kind
        engine_dir = root / ("engine" if a.precision == "fp32" else "engine-fp16")
        session = ort.InferenceSession(str(root / "onnx/model.onnx"), options, providers=["CPUExecutionProvider"])
        comparisons = []
        with DinoV3Session(engine_dir / "model.engine", a.build) as native:
            assert (native.input_width, native.input_height) == (64, 64)
            for index, image in enumerate(images):
                values = processor(images=image[:, :, ::-1].copy(), input_data_format="channels_last", return_tensors="np").pixel_values
                expected = dict(zip([v.name for v in session.get_outputs()], session.run(None, {"pixel_values": values})))
                expected["normalized_embedding"] = expected["global_embedding"] / np.linalg.norm(expected["global_embedding"], axis=-1, keepdims=True)
                raw = native.infer_chw(values)
                result = native.infer_bgr(image)
                errors = {}
                for name in expected:
                    # FP16 is opt-in and uses a separate, declared tolerance.
                    atol, rtol = (3e-4, 3e-4) if a.precision == "fp32" else (.03, .03)
                    np.testing.assert_allclose(raw[name], expected[name], atol=atol, rtol=rtol)
                    np.testing.assert_allclose(result[name], expected[name], atol=atol, rtol=rtol)
                    errors[name] = float(np.max(np.abs(result[name] - expected[name])))
                patches, fmap = result["patch_tokens"], result["feature_map"]
                np.testing.assert_array_equal(patches, fmap.transpose(0, 2, 3, 1).reshape(patches.shape))
                assert patches.shape == (1, 16 if kind == "vit" else 4, 32)
                np.testing.assert_allclose(np.linalg.norm(result["normalized_embedding"]), 1, atol=1e-6)
                saved = result["global_embedding"].copy()
                native.infer_bgr(image[:, ::-1])  # non-contiguous view is copied by the Python API
                np.testing.assert_array_equal(saved, result["global_embedding"])
                comparisons.append({"input_shape": list(image.shape), "max_abs_errors": errors})
            def rejected(action):
                try:
                    action()
                except (RuntimeError, ValueError):
                    return
                raise AssertionError("Invalid DINOv3 input was accepted")
            rejected(lambda: native.infer_bgr(np.zeros((3, 3), np.uint8)))
            rejected(lambda: native.infer_bgr(np.zeros((3, 3, 3), np.float32)))
            rejected(lambda: native.infer_chw(np.zeros(3, np.float32)))
            invalid = values.copy(); invalid.ravel()[0] = np.nan
            rejected(lambda: native.infer_chw(invalid))
            # Verify native size-query, error status and no partial result leaks.
            result_ptr = C.c_void_p()
            assert native.bgr(native.model, None, 1, 1, 3, C.byref(result_ptr)) == 1
            assert not result_ptr.value
            valid = np.ascontiguousarray(values)
            native.check(native.chw(native.model, valid.ctypes.data, valid.size, C.byref(result_ptr)))
            try:
                copied = C.c_size_t()
                assert native.copy(result_ptr, b"global_embedding", None, 0, C.byref(copied)) == 4
                assert copied.value == 32
                info = TensorInfo()
                assert native.info(result_ptr, b"not-a-tensor", C.byref(info)) == 1
                native.check(native.info(result_ptr, b"patch_tokens", C.byref(info)))
                assert info.rank == 3
            finally:
                native.destroy_result(result_ptr)
            # Recovery after invalid requests, and independent model instances.
            with DinoV3Session(engine_dir / "model.engine", a.build) as other:
                recovered = native.infer_chw(valid)
                np.testing.assert_array_equal(recovered["global_embedding"], other.infer_chw(valid)["global_embedding"])
            zero = native.infer_chw(np.zeros_like(valid))
            normalized = zero["global_embedding"] / np.maximum(np.linalg.norm(zero["global_embedding"], axis=-1, keepdims=True), 1e-12)
            np.testing.assert_allclose(zero["normalized_embedding"], normalized, atol=1e-6)
        row = {"model": kind, "precision": a.precision, "status": "PASS_SYNTHETIC_ONLY", "comparisons": comparisons}
        reports.append(row)
        print(json.dumps(row), flush=True)
    (a.root / f"validation-{a.precision}.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
