"""Compare the migrated C++ tensor API outputs against ONNX Runtime CPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, default=Path("cache/trt10-models/models.json"))
    p.add_argument("--engines", type=Path, default=Path("cache/trt10-validation"))
    p.add_argument("--build", type=Path, default=Path("build/trt10-release"))
    p.add_argument("--name")
    a = p.parse_args()
    import numpy as np
    import cv2
    import onnxruntime as ort
    build = a.build.resolve()
    env = os.environ.copy()
    env["PATH"] = str(build / "bintrt") + ";" + env.get("PATH", "")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    report = []
    for record in json.loads(a.models.read_text())["models"]:
        if a.name and record["name"] != a.name:
            continue
        engine = a.engines.resolve() / (record["name"] + ".engine")
        if not engine.exists():
            raise FileNotFoundError(engine)
        folder = a.engines.resolve() / "raw" / record["name"]
        folder.mkdir(parents=True, exist_ok=True)
        session = ort.InferenceSession(record["onnx"], options, providers=["CPUExecutionProvider"])
        reference_input = Path(record["onnx"]).parent / "input.npy"
        if reference_input.exists():
            x = np.load(reference_input)
        else:
            image = cv2.imread(str(a.models.parent / "bus.jpg"))
            _, _, height, width = session.get_inputs()[0].shape
            scale = min(width / image.shape[1], height / image.shape[0])
            resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
            canvas = np.full((height, width, 3), 114, np.uint8)
            left, top = (width - resized.shape[1]) // 2, (height - resized.shape[0]) // 2
            canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
            x = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
        x.astype(np.float32).tofile(folder / "input.bin")
        expected = dict(zip([o.name for o in session.get_outputs()], session.run(None, {session.get_inputs()[0].name: x})))
        completed = subprocess.run([str(build / "bin/trt_raw_outputs.exe"), str(engine),
                                    str(folder / "input.bin"), str(folder)], env=env,
                                   capture_output=True, text=True, timeout=300)
        if completed.returncode:
            raise RuntimeError(completed.stderr or completed.stdout)
        comparisons = {}
        for line in (folder / "outputs.tsv").read_text().splitlines():
            name, filename, dtype, *shape = line.split("\t")
            actual = np.fromfile(folder / filename, dtype={"FP32": np.float32, "FP16": np.float16,
                                                         "INT64": np.int64, "INT32": np.int32}[dtype]).reshape([int(d) for d in shape])
            ref = expected[name]
            assert actual.shape == ref.shape
            assert np.isfinite(actual).all(), f"{name}: non-finite values"
            diff = actual.astype(np.float64) - ref.astype(np.float64)
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            rel = rmse / max(float(np.sqrt(np.mean(ref.astype(np.float64) ** 2))), 1e-6)
            max_abs = float(np.max(np.abs(diff)))
            # Low-confidence FP16 box coordinates can have outliers. Preserve
            # the maximum error in the report and compare complete tensors by
            # relative RMS, with tighter bounds for FP32 transformer logits.
            limit = .01 if record["precision"] == "fp16" else .002
            passed = rel < limit
            comparisons[name] = {"shape": list(actual.shape), "max_abs_error": max_abs,
                                 "rmse": rmse, "relative_rmse": rel, "limit": limit, "pass": passed}
            if record["type"] in (6, 7, 8) and actual.ndim == 3:
                # YOLO26 TopK returns a set of detections. FP16 changes the
                # ranking of near-equal scores, so comparing row i with row i
                # is invalid. Match confident detections by class and box IoU.
                from scipy.optimize import linear_sum_assignment
                aa, rr = actual[0], ref[0]
                aa, rr = aa[aa[:, 4] > .25], rr[rr[:, 4] > .25]
                if len(aa) and len(aa) == len(rr):
                    inter_min = np.maximum(aa[:, None, :2], rr[None, :, :2])
                    inter_max = np.minimum(aa[:, None, 2:4], rr[None, :, 2:4])
                    inter = np.maximum(0, inter_max - inter_min).prod(axis=-1)
                    areas_a = np.maximum(0, aa[:, 2:4] - aa[:, :2]).prod(axis=-1)
                    areas_r = np.maximum(0, rr[:, 2:4] - rr[:, :2]).prod(axis=-1)
                    ious = inter / np.maximum(areas_a[:, None] + areas_r[None, :] - inter, 1e-9)
                    same_class = aa[:, None, 5].round() == rr[None, :, 5].round()
                    ii, jj = linear_sum_assignment(1 - ious + (~same_class) * 100)
                    minimum_iou = float(ious[ii, jj].min())
                    score_error = float(np.abs(aa[ii, 4] - rr[jj, 4]).max())
                    classes_match = bool(same_class[ii, jj].all())
                    set_pass = minimum_iou > .95 and score_error < .05 and classes_match
                else:
                    minimum_iou, score_error, classes_match, set_pass = 0., None, False, False
                comparisons[name].update({"comparison": "class-and-IoU matching after TopK",
                    "reference_count": len(rr), "actual_count": len(aa), "minimum_matched_iou": minimum_iou,
                    "confidence_max_abs_error": score_error, "classes_match": classes_match, "pass": set_pass})
            if record["type"] in (1, 2, 5) and actual.ndim == 3:
                class_count = 1 if record["type"] == 5 else 80
                conf_error = float(np.max(np.abs(actual[:, 4:4 + class_count] - ref[:, 4:4 + class_count])))
                comparisons[name]["confidence_max_abs_error"] = conf_error
                confidence_limit = .05
                comparisons[name]["confidence_error_limit"] = confidence_limit
                reference_scores = ref[:, 4:4 + class_count]
                actual_scores = actual[:, 4:4 + class_count]
                selected = reference_scores.max(axis=1) > .5
                agreement = float(np.mean(reference_scores.argmax(axis=1)[selected] == actual_scores.argmax(axis=1)[selected])) if selected.any() else 1.
                comparisons[name]["high_confidence_class_agreement"] = agreement
                comparisons[name]["pass"] &= conf_error < confidence_limit and agreement >= .99
            if reference_input.exists():
                pytorch_file = reference_input.parent / "pytorch.npz"
                if pytorch_file.exists():
                    pytorch = np.load(pytorch_file)[name]
                    comparisons[name]["onnx_pytorch_max_abs_error"] = float(np.max(np.abs(ref - pytorch)))
        row = {"model": record["name"], "outputs": comparisons,
               "status": "PASS" if all(v["pass"] for v in comparisons.values()) else "FAIL"}
        report.append(row)
        print(json.dumps(row), flush=True)
        (a.engines / "numerical-parity.json").write_text(json.dumps(report, indent=2))
    if not report or any(r["status"] != "PASS" for r in report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
