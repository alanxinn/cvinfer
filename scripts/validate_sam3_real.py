"""Validate an AUTHORIZED pretrained SAM3 bundle against official PyTorch CPU inference."""
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
from sam3_infer import Sam3Session
from sam3_checkpoint import component_path, load_sam3_component, source_metadata


def iou(a, b):
    a, b = a != 0, b != 0
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 1.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="facebook/sam3")
    p.add_argument("--engines", type=Path, default=Path("cache/sam3/engines"))
    p.add_argument("--build", type=Path, default=Path("build/trt10-release"))
    p.add_argument("--image", type=Path, default=Path("cache/trt10-models/bus.jpg"))
    p.add_argument("--box", type=float, nargs=4, default=[50, 390, 245, 910])
    p.add_argument("--positive", type=float, nargs=2, default=[150, 600])
    p.add_argument("--negative", type=float, nargs=2, default=[240, 850])
    p.add_argument("--text", default="person")
    p.add_argument("--minimum-iou", type=float, default=.95)
    p.add_argument("--output", type=Path, default=Path("cache/sam3/real-validation.json"))
    a = p.parse_args()
    from huggingface_hub import get_token
    if not Path(a.checkpoint).exists() and not get_token():
        p.error("Official SAM3 checkpoint access is required. Log in locally or pass a local HF snapshot directory.")
    import torch
    import torch.nn.functional as F
    import cv2
    from PIL import Image
    from transformers import Sam3Model, Sam3TrackerModel, Sam3Processor, Sam3TrackerProcessor
    from scipy.optimize import linear_sum_assignment
    torch.set_num_threads(4)
    image = cv2.imread(str(a.image))
    if image is None:
        p.error("Image cannot be read")
    h, w = image.shape[:2]
    pil = Image.fromarray(image[:, :, ::-1])
    checks = []
    artifacts = a.output.parent / (a.output.stem + "-images")
    artifacts.mkdir(parents=True, exist_ok=True)
    def save_masks(stage, results):
        canvas = image.copy()
        for index, item in enumerate(results):
            path = artifacts / f"{stage}-{index}.png"
            if not cv2.imwrite(str(path), item["mask"]):
                raise RuntimeError("Could not write validation mask")
            item["mask"].tofile(artifacts / f"{stage}-{index}.mask")
            selected = item["mask"] != 0
            color = np.array([(50 + 83 * index) % 255, (130 + 37 * index) % 255, (90 + 151 * index) % 255])
            canvas[selected] = (.6 * canvas[selected] + .4 * color).astype(np.uint8)
        cv2.imwrite(str(artifacts / (stage + ".jpg")), canvas)
    manifest = json.loads((a.engines / "bundle.json").read_text())
    if manifest.get("synthetic") or manifest.get("source", {}).get("synthetic"):
        p.error("Real-model validation cannot use synthetic engines")
    started = time.perf_counter()
    with Sam3Session(a.engines, a.build) as native:
        native.set_image(image)
        native.set_box(*a.box)
        processor = Sam3TrackerProcessor.from_pretrained(component_path(a.checkpoint, "tracker"))
        tracker = load_sam3_component(a.checkpoint, "tracker")
        pixels = processor.image_processor(images=pil, return_tensors="pt").pixel_values
        scale = torch.tensor([pixels.shape[-1] / w, pixels.shape[-2] / h])
        box = torch.tensor(a.box).reshape(1, 1, 4) * scale.repeat(2)
        with torch.inference_mode():
            embeddings = tracker.get_image_embeddings(pixels)
        xy, labels = [], []
        for stage, point, positive in [("box", None, True), ("positive", a.positive, True), ("negative", a.negative, False)]:
            if point is not None:
                native.add_point(*point, positive=positive)
                xy.append(point); labels.append(int(positive))
            kwargs = {"image_embeddings": embeddings, "input_boxes": box, "multimask_output": True}
            if xy:
                kwargs.update(input_points=torch.tensor(xy).reshape(1, 1, -1, 2) * scale,
                              input_labels=torch.tensor(labels, dtype=torch.int64).reshape(1, 1, -1))
            with torch.inference_mode():
                ref = tracker(**kwargs)
            best = int(ref.iou_scores.reshape(-1).argmax())
            expected = F.interpolate(ref.pred_masks[:, :, best], size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy() > 0
            before = time.perf_counter()
            actual = native.predict()[0]
            elapsed = time.perf_counter() - before
            save_masks(stage, [actual])
            overlap = iou(expected, actual["mask"])
            score_error = abs(actual["score"] - float(ref.iou_scores.reshape(-1)[best]))
            passed = overlap >= a.minimum_iou and score_error < .05 and bool(np.count_nonzero(actual["mask"]))
            checks.append({"stage": stage, "mask_iou": overlap, "score_error": score_error,
                           "native_seconds": elapsed, "foreground_pixels": int(np.count_nonzero(actual["mask"])), "pass": passed})
            print(checks[-1], flush=True)
        del embeddings, tracker, ref, processor
        gc.collect()
        concept_processor = Sam3Processor.from_pretrained(component_path(a.checkpoint, "detector"))
        detector = load_sam3_component(a.checkpoint, "detector")
        encoded = concept_processor.tokenizer(a.text, padding="max_length", max_length=native.bundle["text_length"], truncation=True, return_tensors="pt")
        (artifacts / "text-tokens.json").write_text(json.dumps({"text": a.text, "ids": encoded.input_ids[0].tolist(),
                                                              "attention": encoded.attention_mask[0].tolist()}))
        pixels = concept_processor.image_processor(images=pil, return_tensors="pt").pixel_values
        with torch.inference_mode():
            output = detector(pixel_values=pixels, input_ids=encoded.input_ids, attention_mask=encoded.attention_mask)
        expected = concept_processor.post_process_instance_segmentation(output, target_sizes=[(h, w)], threshold=.3)[0]
        before = time.perf_counter()
        actual = native.predict_tokens(encoded.input_ids.numpy(), encoded.attention_mask.numpy())
        elapsed = time.perf_counter() - before
        save_masks("text", actual)
        # Confirm that the native BPE path produces the same model inputs/results
        # for the documented English prompt path.
        if a.text.isascii():
            direct = native.predict_text(a.text)
            assert len(actual) == len(direct)
            for x, y in zip(actual, direct):
                np.testing.assert_array_equal(x["mask"], y["mask"])
        boxes = expected["boxes"].numpy()
        overlaps, score_errors = [], []
        if len(actual) == len(boxes) and actual:
            aa = np.array([m["box"] for m in actual])
            left = np.maximum(aa[:, None, :2], boxes[None, :, :2])
            right = np.minimum(aa[:, None, 2:], boxes[None, :, 2:])
            inter = np.maximum(0, right - left).prod(-1)
            union = np.maximum(0, aa[:, 2:] - aa[:, :2]).prod(-1)[:, None] + np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(-1)[None] - inter
            ii, jj = linear_sum_assignment(1 - inter / np.maximum(union, 1e-9))
            for i, j in zip(ii, jj):
                overlaps.append(iou(actual[i]["mask"], expected["masks"][j].numpy()))
                score_errors.append(abs(actual[i]["score"] - float(expected["scores"][j])))
        passed = bool(overlaps) and min(overlaps) >= a.minimum_iou and max(score_errors) < .05
        checks.append({"stage": "text", "native_count": len(actual), "pytorch_count": len(boxes),
                       "mask_ious": overlaps, "score_errors": score_errors, "native_seconds": elapsed, "pass": passed})
    report = {"status": "PASS_REAL_PRETRAINED" if all(v["pass"] for v in checks) else "FAIL", "checkpoint": a.checkpoint,
              "image": str(a.image.resolve()), "checks": checks, "source": source_metadata(a.checkpoint),
              "total_seconds": time.perf_counter() - started, "artifacts": str(artifacts.resolve())}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if report["status"] != "PASS_REAL_PRETRAINED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
