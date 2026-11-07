"""Validate real MediaPipe TensorRT engines against ONNX Runtime, without compiling.

Use --native after building the new adapter to include C ABI/geometry checks.
"""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np
import onnxruntime as ort

from mediapipe_hands_infer import HandsSession, RawEngineRunner
from mediapipe_hands_pipeline import anchors, draw, hand_input, infer
from prepare_mediapipe_hands import download, prepare


def check_demo(root, precision, demo):
    """Exercise the public image CLI and compare every field with native fixtures."""
    root, demo = root.resolve(), demo.resolve()
    command = ["dotnet", str(demo)] if demo.suffix.lower() == ".dll" else [str(demo)]
    fixtures, output = root / "interface-fixtures", root / f"demo-{precision}"
    output.mkdir(parents=True, exist_ok=True)
    base = ["--type", "mediapipe-hands", "--palm-model", str(root / f"palm-{precision}.engine"),
            "--model", str(root / f"hand-{precision}.engine")]
    env = os.environ.copy()
    env.pop("ALANXINN_MEDIAPIPE_NATIVE_DIR", None)  # Verify the Demo's deployed DLL search path.
    cases = []
    for line in (fixtures / f"manifest-{precision}.tsv").read_text().splitlines():
        name, width, height, count = line.split("\t")
        image = np.fromfile(fixtures / f"{name}.bgr", np.uint8).reshape(int(height), int(width), 3)
        source, target = output / f"{name}.png", output / f"{name}-result.png"
        assert cv2.imwrite(str(source), image)
        args = base + ["--img", str(source), "--output", str(target)]
        if not cases:
            args += ["--warmup", "1", "--iterations", "2", "--timing"]
        completed = subprocess.run(command + args, env=env, capture_output=True, text=True, timeout=180)
        assert completed.returncode == 0, completed.stderr or completed.stdout
        if not cases:
            assert "benchmark warmup=1 iterations=2" in completed.stdout and "total avg=" in completed.stdout
        actual = json.loads(target.with_suffix(".json").read_text())
        assert actual["width"] == int(width) and actual["height"] == int(height)
        assert len(actual["hands"]) == int(count)
        values = []
        for hand in actual["hands"]:
            assert len(hand["box"]) == 4 and len(hand["landmarks"]) == len(hand["worldLandmarks"]) == 63
            assert hand["label"] == ("Right" if hand["handedness"] >= .5 else "Left")
            values.extend(hand["box"] + [hand["palmScore"], hand["confidence"], hand["handedness"]] +
                          hand["landmarks"] + hand["worldLandmarks"])
        actual_values = np.asarray(values, dtype=np.float32)
        expected = np.fromfile(fixtures / f"{name}.hands.bin", dtype="<f4")
        np.testing.assert_array_equal(actual_values, expected)
        assert target.exists() and cv2.imread(str(target)).shape == image.shape
        cases.append({"name": name, "hands": int(count), "max_abs_error": 0.})
    first = output / f"{precision}-original.png"
    one = output / "one-hand.png"
    completed = subprocess.run(command + base + ["--img", str(first), "--output", str(one), "--max-hands", "1"],
                               env=env, capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stderr
    assert len(json.loads(one.with_suffix(".json").read_text())["hands"]) == 1
    for options in (["--max-hands", "0"], ["--palm-threshold", "NaN"], ["--hand-threshold", "2"],
                    ["--nms", "-1"], ["--iterations", "0"], ["--backend", "denglin"]):
        completed = subprocess.run(command + base + ["--img", str(first)] + options,
                                   env=env, capture_output=True, text=True, timeout=180)
        assert completed.returncode == 2, (options, completed.returncode, completed.stderr)
    swapped = base.copy()
    swapped[swapped.index("--palm-model") + 1], swapped[swapped.index("--model") + 1] = (
        str(root / f"hand-{precision}.engine"), str(root / f"palm-{precision}.engine"))
    completed = subprocess.run(command + swapped + ["--img", str(first)],
                               env=env, capture_output=True, text=True, timeout=180)
    assert completed.returncode == 1 and completed.stderr.strip(), "Native model error was not propagated"
    report = {"status": "PASS", "precision": precision, "demo": str(demo), "cases": cases,
              "max_hands": "PASS", "invalid_inputs": "PASS", "native_error_propagation": "PASS",
              "warmup_iterations_timing": "PASS", "annotated_images_and_json": "PASS"}
    (root / f"validation-csharp-demo-{precision}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("cache/mediapipe-hands"))
    parser.add_argument("--build", type=Path, default=Path("build/trt10-release"))
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true", help="Use existing models/engines without downloading or rebuilding")
    parser.add_argument("--demo", type=Path, help="Also validate CvinferDemo executable/DLL using the native fixtures")
    args = parser.parse_args()
    args.native = args.native or args.demo is not None
    if not args.skip_prepare:
        prepare(args.root, args.build, args.precision)
    source = "https://storage.googleapis.com/mediapipe-assets/woman_hands.jpg"
    sample = args.root / "woman_hands.jpg"
    download(source, sample)
    image = cv2.imread(str(sample))
    assert image is not None
    raw = RawEngineRunner(args.root, args.build, args.precision)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    sessions = {kind: ort.InferenceSession(str(args.root / f"{kind}.onnx"), options,
                providers=["CPUExecutionProvider"]) for kind in ("palm", "hand")}

    def reference(kind, values):
        session = sessions[kind]
        return dict(zip([o.name for o in session.get_outputs()], session.run(None, {kind + "_input": values})))

    parity = []
    def checked(kind, values):
        actual, expected = raw(kind, values), reference(kind, values)
        errors = {}
        for name, value in expected.items():
            assert actual[name].shape == value.shape
            assert np.isfinite(actual[name]).all()
            delta = actual[name].astype(np.float64) - value
            relative = float(np.sqrt(np.mean(delta ** 2)) / max(np.sqrt(np.mean(value.astype(np.float64) ** 2)), 1e-6))
            limit = .003 if args.precision == "fp32" else .03
            assert relative < limit, (kind, name, relative, limit)
            errors[name] = {"relative_rmse": relative, "max_abs": float(np.max(np.abs(delta)))}
        parity.append({"model": kind, "outputs": errors})
        return actual

    assert anchors().shape == (2016, 2)
    # Independent affine round-trip, including points outside image boundaries.
    palm = {"points": np.array([[30, 60], [20, 40], [40, 30], [50, 40], [60, 50], [10, 60], [20, 70]]),
            "box": np.array([10, 20, 60, 70]), "score": 1.}
    _, inverse, _, _ = hand_input(image, palm)
    affine = cv2.invertAffineTransform(inverse)
    points = np.array([[0., 0.], [image.shape[1], image.shape[0]], [-20., -30.]])
    restored = (points @ affine[:, :2].T + affine[:, 2]) @ inverse[:, :2].T + inverse[:, 2]
    np.testing.assert_allclose(restored, points, atol=1e-9)
    variants = {"original": image, "rotated90": cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
                "mirrored": image[:, ::-1], "wide": cv2.copyMakeBorder(image, 0, 0, 100, 100, cv2.BORDER_CONSTANT)}
    results = {}
    fixture_rows = []
    fixtures = args.root / "interface-fixtures"
    def save_fixture(name, frame, hands):
        fixtures.mkdir(parents=True, exist_ok=True)
        prefix = f"{args.precision}-{name}"
        np.ascontiguousarray(frame).tofile(fixtures / f"{prefix}.bgr")
        floats = [value for hand in hands for value in
                  (hand["box"] + [hand["palm_score"], hand["confidence"], hand["handedness"]] +
                   np.asarray(hand["landmarks"]).ravel().tolist() +
                   np.asarray(hand["world_landmarks"]).ravel().tolist())]
        np.asarray(floats, dtype="<f4").tofile(fixtures / f"{prefix}.hands.bin")
        fixture_rows.append(f"{prefix}\t{frame.shape[1]}\t{frame.shape[0]}\t{len(hands)}")
    native = HandsSession(args.root, args.build, args.precision) if args.native else None
    try:
        for name, frame in variants.items():
            hands = infer(frame, checked)
            expected = infer(frame, reference)
            assert len(hands) == len(expected) == 2, (name, len(hands), len(expected))
            # Match by wrist distance to allow confidence-based NMS ordering changes.
            remaining = list(expected)
            for hand in hands:
                wrist = np.array(hand["landmarks"])[0, :2]
                match = min(remaining, key=lambda h: np.linalg.norm(np.array(h["landmarks"])[0, :2] - wrist))
                remaining.remove(match)
                assert hand["label"] == match["label"]
                np.testing.assert_allclose(hand["landmarks"], match["landmarks"], atol=1.5 if args.precision == "fp32" else 5, rtol=.01)
                np.testing.assert_allclose(hand["world_landmarks"], match["world_landmarks"], atol=.003, rtol=.05)
            if native:
                native_hands = native.infer_bgr(frame)
                assert len(native_hands) == len(hands)
                for hand, other in zip(hands, native_hands):
                    np.testing.assert_allclose(hand["landmarks"], other["landmarks"], atol=2., rtol=.01)
                    np.testing.assert_allclose(hand["world_landmarks"], other["world_landmarks"], atol=.003, rtol=.05)
                    assert hand["label"] == other["label"]
                save_fixture(name, frame, native_hands)
            results[name] = {"hands": len(hands), "labels": [h["label"] for h in hands]}
            assert cv2.imwrite(str(args.root / f"checked-{name}.jpg"), draw(frame, hands))
            (args.root / f"checked-{name}.json").write_text(json.dumps(hands, indent=2))
        assert infer(np.zeros((256, 256, 3), np.uint8), checked) == []
        assert len(infer(image, raw, max_hands=1)) == 1
        for action in [lambda: infer(np.zeros((2, 2), np.uint8), raw),
                       lambda: infer(image.astype(np.float32), raw),
                       lambda: infer(image, raw, palm_threshold=float("nan")),
                       lambda: infer(image, raw, max_hands=0)]:
            try:
                action()
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid request was accepted")
        if native:
            blank = np.zeros((256, 256, 3), np.uint8)
            assert native.infer_bgr(blank) == []
            save_fixture("blank", blank, [])
            count = C.c_size_t(999)
            assert native.run(native.model, None, 1, 1, 3, None, 0, C.byref(count)) == 1 and count.value == 0
            contiguous = np.ascontiguousarray(image)
            assert native.run(native.model, contiguous.ctypes.data, image.shape[1], image.shape[0], 3,
                              None, 0, C.byref(count)) == 4 and count.value == 2
            assert len(native.infer_bgr(image)) == 2
            native.close()
            try:
                native.infer_bgr(image)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Disposed session accepted inference")
    finally:
        if native:
            native.close()
    report = {"status": "PASS", "precision": args.precision, "source": source,
              "backend": "native-adapter-and-core" if args.native else "existing-native-core-with-python-geometry",
              "native_adapter_tested": args.native, "cases": results, "blank_image": "PASS",
              "invalid_inputs": "PASS", "max_hands": "PASS", "affine_round_trip": "PASS", "parity": parity}
    (args.root / f"validation-{args.precision}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.native:
        (fixtures / f"manifest-{args.precision}.tsv").write_text("\n".join(fixture_rows) + "\n", encoding="utf-8")
    if args.demo:
        check_demo(args.root, args.precision, args.demo)
    print(json.dumps({key: value for key, value in report.items() if key != "parity"}), flush=True)


if __name__ == "__main__":
    main()
