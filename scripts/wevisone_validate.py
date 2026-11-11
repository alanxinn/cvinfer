"""Exercise all eleven tasks on real images through C ABI, C# and C++.

This checks observed behavior and binding parity; it is not a benchmark accuracy
claim. Requires a genuinely built WeVisOne bundle (never substitutes mock engines).
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request

from wevisone_checkpoint import TASKS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("cache/wevisone/2B/engines"))
    p.add_argument("--build", type=Path, default=Path("build/trt10-release"))
    p.add_argument("--output", type=Path, default=Path("cache/wevisone/validation/real"))
    p.add_argument("--bindings", nargs="+", choices=["native", "csharp", "cpp"], default=["native", "csharp", "cpp"])
    p.add_argument("--timeout", type=int, default=240)
    a = p.parse_args()
    a.bindings = [name for name in ("native", "csharp", "cpp") if name in a.bindings]
    import cv2
    import numpy as np
    repo = Path(__file__).resolve().parents[1]
    a.model, a.build, a.output = (path.resolve() for path in (a.model, a.build, a.output))
    a.output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join(str(a.build / path) for path in ("bin", "bintrt")) + os.pathsep + environment.get("PATH", "")
    report = {"model": str(a.model), "scope": "real-image behavior and binding parity, not benchmark accuracy", "cases": []}

    def save():
        (a.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if not (a.model / "bundle.json").is_file():
        report.update(status="BLOCKED", reason="Real TensorRT engine bundle is missing")
        report["cases"] = [{"task": task, "status": "NOT_RUN"} for task in TASKS]
        save()
        raise SystemExit("Real TensorRT engine bundle is missing; no task has been declared passed")
    manifest = json.loads((a.model / "bundle.json").read_text(encoding="utf-8"))
    if manifest.get("synthetic") or manifest.get("format") != "cvinfer-wevisone-v1":
        raise ValueError("Real official-checkpoint bundle is required")
    inputs = a.output / "inputs"
    inputs.mkdir(exist_ok=True)
    images = {}
    for name in ("livingroom", "coffee_sign"):
        source = repo / "cache/wevisone/validation/inputs" / (name + ".jpg")
        if not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve("https://raw.githubusercontent.com/Tencent/WeVisOne-VL/9a6bcd85404505df0d05cfb5d97512ccacf35dee/demo/examples/images/" + name + ".jpg", source)
        image = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Cannot read official image " + name)
        path = inputs / (name + ".png")
        cv2.imencode(".png", image)[1].tofile(path)  # One lossless fixture for every binding.
        images[name] = (path, image)
    cases = [(task, "coffee_sign" if task.startswith("ocr") else "livingroom",
              "METERS" if task == "ocr_grounding" else "" if task in ("ocr", "visual_prompt") else
              "Describe the image in one short sentence." if task == "freeform" else "sofa") for task in TASKS]
    cases += [("absent_target", "livingroom", "a purple elephant"), ("box_matting", "livingroom", "")]
    for task, image_name, query in cases:
        record = {"task": task, "bindings": []}
        baseline = None
        image_path, image = images[image_name]
        height, width = image.shape[:2]
        box = [20, 280, 620, 530] if task in ("visual_prompt", "box_matting") else []
        actual_task = "referring" if task == "absent_target" else "matting" if task == "box_matting" else task
        for binding in a.bindings:
            if binding == "cpp" and task == "box_matting":
                continue  # C++ demo has no box-matting flag; native and C# exercise it.
            destination = a.output / task / binding
            destination.mkdir(parents=True, exist_ok=True)
            if binding == "native":
                cmd = [sys.executable, str(repo / "scripts/wevisone_infer.py"), "--model", str(a.model), "--build", str(a.build),
                       "--image", str(image_path), "--task", actual_task, "--output", str(destination)]
                if query:
                    cmd += ["--query", query]
                if box:
                    cmd += ["--box", *map(str, box)]
                if task == "box_matting":
                    cmd += ["--box-matting"]
            elif binding == "csharp":
                cmd = ["dotnet", str(repo / "csharp/alanxinn.CvinferDemo/bin/Release/net8.0-windows/alanxinn.CvinferDemo.dll"),
                       "--type", "wevisone", "--model", str(a.model), "--img", str(image_path), "--vl-task", actual_task,
                       "--output", str(destination)]
                if query:
                    cmd += ["--query", query]
                if box:
                    cmd += ["--boxes", ",".join(map(str, box))]
                if task == "box_matting":
                    cmd += ["--box-matting"]
            else:
                cmd = [str(a.build / "bin/alanxinn_wevisone_demo.exe"), str(a.model), str(image_path), actual_task, query, *map(str, box)]
            started = time.monotonic()
            value = {"binding": binding, "command": cmd}
            try:
                run = subprocess.run(cmd, cwd=destination, env=environment, capture_output=True, timeout=a.timeout)
                stdout, stderr = run.stdout.decode("utf-8", errors="replace"), run.stderr.decode("utf-8", errors="replace")
                (destination / "stdout.log").write_text(stdout, encoding="utf-8")
                (destination / "stderr.log").write_text(stderr, encoding="utf-8")
                if run.returncode:
                    raise RuntimeError(f"exit={run.returncode}: {stderr[-1800:]}")
                if binding == "native":
                    result = json.loads((destination / "result.json").read_text(encoding="utf-8"))
                elif binding == "csharp":
                    data = json.loads((destination / "result.json").read_text(encoding="utf-8-sig"))
                    result = {"response": data["Response"], "count": data["Count"], "rejected": data["Rejected"],
                              "truncated": data["Truncated"], "annotations": data["Annotations"], "masks": data["Masks"]}
                else:
                    match = re.search(r"\ncount=(-?\d+) rejected=(\d+) truncated=(\d+) latency_ms=", stdout)
                    if not match:
                        raise ValueError("C++ result summary is missing")
                    result = {"response": stdout[:match.start()].strip(), "count": int(match[1]),
                              "rejected": bool(int(match[2])), "truncated": bool(int(match[3]))}
                if result["truncated"] or not result["response"].strip():
                    raise ValueError("Response is empty or truncated")
                if task == "absent_target":
                    if not result["rejected"]:
                        raise ValueError("Expected absent-target rejection")
                elif result["rejected"]:
                    raise ValueError("Present target was rejected")
                if task in ("count_box", "count_point") and result["count"] != 1:
                    raise ValueError(f"Expected one sofa, observed {result['count']}")
                if binding != "cpp":
                    annotations, masks = result["annotations"], result["masks"]
                    if task not in ("freeform", "absent_target") and not annotations:
                        raise ValueError("Present-target annotations are empty")
                    for annotation in annotations:
                        is_point = annotation.get("IsPoint", annotation.get("point") is not None)
                        if task in ("point", "count_point") and not is_point:
                            raise ValueError("Point task returned boxes instead of points")
                        coordinates = [annotation["X"], annotation["Y"]] if is_point and binding == "csharp" else annotation.get("point") if is_point else annotation.get("Box", annotation.get("box"))
                        if not np.isfinite(coordinates).all() or any(x < 0 or x > (width if i % 2 == 0 else height) for i, x in enumerate(coordinates)):
                            raise ValueError("Annotation is outside the original image")
                    labels = " ".join(item.get("Label", item.get("label", "")) for item in annotations).upper()
                    if task == "ocr_grounding" and "METERS" not in labels:
                        raise ValueError("Grounded OCR text does not match the requested word")
                    if task == "ocr" and not any(word in labels for word in ("METERS", "COFFEE", "AHEAD", "RIGHT", "ALLEY")):
                        raise ValueError("OCR did not recognize any expected sign words")
                    if task in ("segmentation", "matting", "box_matting"):
                        if not masks:
                            raise ValueError("Pixel task produced no masks")
                        for i in range(len(masks)):
                            alpha = np.fromfile(destination / f"alpha_{i:03d}.f32", dtype="<f4")
                            if alpha.size != width * height or not np.isfinite(alpha).all() or alpha.min() < 0 or alpha.max() > 1:
                                raise ValueError("Invalid alpha data")
                            if alpha.max() < .5 or alpha.min() > .5:
                                raise ValueError("Mask is empty or full")
                            if i == 0 and (alpha.reshape(height, width)[400, 300] < .5 or alpha[0] > .5):
                                raise ValueError("Sofa mask does not cover the known foreground/background sample pixels")
                            if baseline is not None:
                                expected = np.fromfile(baseline / f"alpha_{i:03d}.f32", dtype="<f4")
                                if alpha.shape != expected.shape or not np.allclose(alpha, expected, rtol=0, atol=1e-6):
                                    raise ValueError("Alpha differs across bindings")
                            if task == "segmentation":
                                binary = cv2.imread(str(destination / f"mask_{i:03d}.png"), cv2.IMREAD_GRAYSCALE)
                                if binary is None or set(np.unique(binary)) != {0, 255}:
                                    raise ValueError("Invalid binary segmentation mask")
                if baseline is not None:
                    reference = json.loads((baseline / "normalized.json").read_text(encoding="utf-8"))
                    for key in ("response", "count", "rejected", "truncated"):
                        if result[key] != reference[key]:
                            raise ValueError("Binding output differs: " + key)
                    if binding == "cpp" and task in ("segmentation", "matting"):
                        for i in range(len(reference.get("masks", []))):
                            actual = cv2.imread(str(destination / f"wevisone_alpha_{i}.png"), cv2.IMREAD_GRAYSCALE)
                            expected = cv2.imread(str(baseline / f"alpha_{i:03d}.png"), cv2.IMREAD_GRAYSCALE)
                            if actual is None or expected is None or actual.shape != expected.shape or np.max(np.abs(actual.astype(np.int16) - expected)) > 1:
                                raise ValueError("C++ alpha PNG differs across bindings")
                else:
                    baseline = destination
                (destination / "normalized.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                value.update(status="PASS", response=result["response"], count=result["count"])
            except Exception as error:
                value.update(status="FAIL", error=str(error))
            value["elapsed_s"] = round(time.monotonic() - started, 3)
            record["bindings"].append(value)
            print(task, binding, value["status"], value.get("error", ""), flush=True)
        record["status"] = "SKIP" if not record["bindings"] else "PASS" if all(x["status"] == "PASS" for x in record["bindings"]) else "FAIL"
        report["cases"].append(record)
        save()
    report["status"] = "PASS" if all(case["status"] in ("PASS", "SKIP") for case in report["cases"]) else "FAIL"
    report["task_passed"] = sum(case["status"] == "PASS" for case in report["cases"] if case["task"] in TASKS)
    save()
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
