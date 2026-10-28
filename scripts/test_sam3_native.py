"""Compare native SAM3 state/coordinates/masks with ONNX Runtime using RANDOM tiny SAM3.

This validates the implementation contract, not pretrained segmentation quality.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort
from sam3_infer import Sam3Session


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", type=Path, default=Path("cache/sam3-tiny/onnx"))
    p.add_argument("--engines", type=Path, default=Path("cache/sam3-tiny/engines"))
    p.add_argument("--build", type=Path, default=Path("build/trt10-release"))
    a = p.parse_args()
    options = ort.SessionOptions(); options.intra_op_num_threads = 2
    sessions = {name: ort.InferenceSession(str(a.onnx / (name + ".onnx")), options, providers=["CPUExecutionProvider"])
                for name in ["detector_vision", "tracker_vision", "text_encoder", "text_decoder", "box_decoder", "interactive_decoder"]}
    image = np.random.default_rng(123).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    # Use the official processor implementation as the preprocessing reference.
    from transformers import Sam3ImageProcessor
    processor = Sam3ImageProcessor(size={"height": 28, "width": 28})
    pixel_values = processor(images=image[:, :, ::-1].copy(), return_tensors="np")["pixel_values"]

    def run(name, inputs):
        s = sessions[name]
        return dict(zip([o.name for o in s.get_outputs()], s.run(None, {i.name: inputs[i.name] for i in s.get_inputs()})))

    features = run("tracker_vision", {"pixel_values": pixel_values})
    box = [4., 3., 42., 29.]
    encoded_box = np.array(box, np.float32).reshape(1, 1, 4) * np.array([28 / 48, 28 / 32, 28 / 48, 28 / 32], np.float32)
    all_points, all_labels = [], []

    def reference():
        inputs = {**features, "input_boxes": encoded_box}
        if all_points:
            inputs["input_points"] = np.array(all_points, np.float32).reshape(1, 1, -1, 2) * np.array([28 / 48, 28 / 32], np.float32)
            inputs["input_labels"] = np.array(all_labels, np.int64).reshape(1, 1, -1)
        values = run("interactive_decoder" if all_points else "box_decoder", inputs)
        best = int(values["iou_scores"].reshape(-1).argmax())
        low = values["masks"][0, 0, best]
        return (cv2.resize(low, (48, 32)) > 0).astype(np.uint8) * 255

    checks = []
    with Sam3Session(a.engines, a.build, confidence=0) as native:
        def rejected(action):
            try:
                action()
            except RuntimeError:
                return
            raise AssertionError("Invalid SAM3 call was accepted")
        rejected(native.predict)
        native.set_image(image)
        rejected(lambda: native.add_point(1, 1))
        rejected(lambda: native.set_box(40, 0, 1, 20))
        native.set_box(*box)
        def compare(stage):
            actual = native.predict()[0]["mask"]
            expected = reference()
            agreement = float(np.mean(actual == expected))
            assert actual.shape == (32, 48) and np.isin(actual, [0, 255]).all()
            assert agreement > .98, (stage, agreement)
            checks.append({"stage": stage, "native_onnx_mask_pixel_agreement": agreement})
        compare("box")
        native.add_point(20, 18, True); all_points.append([20, 18]); all_labels.append(1)
        compare("box+positive")
        native.add_point(6, 5, False); all_points.append([6, 5]); all_labels.append(0)
        compare("box+positive+negative")
        rejected(lambda: native.add_point(float("nan"), 1))
        rejected(lambda: native.add_point(48, 32))
        native.clear_points(); all_points.clear(); all_labels.clear()
        compare("clear-points-preserves-box")
        ids = np.array([[1, 7, 2, 0, 0, 0, 0, 0]], np.int64)
        attention = (ids != 0).astype(np.int64)
        text_result = native.predict_text("BUS")
        token_result = native.predict_tokens(ids, attention)
        assert len(text_result) == len(token_result) == 2
        df = run("detector_vision", {"pixel_values": pixel_values})
        tf = run("text_encoder", {"input_ids": ids, "attention_mask": attention})
        expected = run("text_decoder", {**df, **tf, "attention_mask": attention})
        for i, (aa, bb) in enumerate(zip(text_result, token_result)):
            np.testing.assert_array_equal(aa["mask"], bb["mask"])
            assert abs(aa["score"] - float(expected["scores"][0, i])) < 1e-4
            np.testing.assert_allclose(aa["box"], expected["boxes"][0, i] * [48, 32, 48, 32], atol=1e-3)
            low = 1 / (1 + np.exp(-expected["masks"][0, i]))
            ref = (cv2.resize(low, (48, 32)) > .5).astype(np.uint8) * 255
            assert np.mean(aa["mask"] == ref) > .98
        checks.append({"stage": "text/BPE/encoded-tokens", "status": "PASS"})
        rejected(lambda: native.predict_text("中文"))
        rejected(lambda: native.predict_text("   "))
        rejected(lambda: native.predict_tokens(ids.ravel()[:-1], attention.ravel()[:-1]))
        rejected(lambda: native.predict_tokens(ids + 1000000, attention))
        native.set_image(image)
        rejected(native.predict)
        rejected(lambda: native.add_point(1, 1))
        native.set_box(*box)
        for i in range(native.bundle["max_points"]):
            native.add_point(20, 18, i % 2 == 0)
        native.predict()
        rejected(lambda: native.add_point(20, 18))
        checks.append({"stage": "new-image-reset/point-limit/invalid-arguments", "status": "PASS"})
    report = {"status": "PASS_SYNTHETIC_CONTRACT_ONLY", "checkpoint": "RANDOM_TINY_SAM3_NOT_PRETRAINED", "checks": checks}
    (a.engines.parent / "native-validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
