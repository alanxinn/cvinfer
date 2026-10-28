"""Compare native DINOv3 image inference with authorized pretrained PyTorch weights."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from dinov3_infer import DinoV3Session


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=Path("cache/dinov3/engine"))
    p.add_argument("--checkpoint", help="Optional local HF snapshot or authorized repository; defaults to bundle source")
    p.add_argument("--image", type=Path, default=Path("cache/trt10-models/bus.jpg"))
    p.add_argument("--build", type=Path, default=Path("build/trt10-release"))
    p.add_argument("--output", type=Path, default=Path("cache/dinov3/real-validation.json"))
    a = p.parse_args()
    manifest = json.loads((a.bundle / "bundle.json").read_text())
    if manifest.get("synthetic") is not False:
        p.error("This command requires a pretrained bundle; use test_dinov3_native.py for synthetic models")
    checkpoint = a.checkpoint or manifest["checkpoint"]
    from huggingface_hub import get_token
    if not Path(checkpoint).exists() and not get_token():
        p.error("Authorized DINOv3 weights are required. Log in locally or provide --checkpoint LOCAL_HF_SNAPSHOT")
    import torch
    import cv2
    from transformers import AutoConfig, AutoModel, DINOv3ViTImageProcessor
    from export_dinov3 import feature_wrapper, OUTPUTS
    torch.set_num_threads(4)
    image = cv2.imread(str(a.image))
    if image is None:
        p.error("Image cannot be read")
    revision = {"revision": manifest["revision"]} if manifest.get("revision") and not a.checkpoint and not Path(checkpoint).exists() else {}
    config = AutoConfig.from_pretrained(checkpoint, **revision)
    if config.model_type != manifest["model_type"]:
        p.error("Checkpoint architecture differs from the exported engine")
    attention = {"attn_implementation": "eager"} if config.model_type == "dinov3_vit" else {}
    model = AutoModel.from_pretrained(checkpoint, config=config, dtype=torch.float32, **revision, **attention).eval()
    wrapper, _ = feature_wrapper(torch, model)
    processor = DINOv3ViTImageProcessor.from_pretrained(a.bundle / "processor")
    precision = manifest["precision"]
    reports = []
    with DinoV3Session(a.bundle / manifest["engine"], a.build) as native:
        for name, pixels in [("original", image), ("horizontal-flip", image[:, ::-1].copy())]:
            values = processor(images=pixels[:, :, ::-1].copy(), input_data_format="channels_last",
                               size={"height": native.input_height, "width": native.input_width}, return_tensors="pt").pixel_values
            with torch.inference_mode():
                outputs = wrapper(values)
            expected = dict(zip(OUTPUTS, [v.numpy() for v in outputs]))
            expected["normalized_embedding"] = expected["global_embedding"] / np.maximum(np.linalg.norm(expected["global_embedding"], axis=-1, keepdims=True), 1e-12)
            actual = native.infer_bgr(pixels)
            tensors = {}
            passed = True
            for key, ref in expected.items():
                diff = actual[key].astype(np.float64) - ref.astype(np.float64)
                rel = float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1e-12))
                limit = .003 if precision == "fp32" else .03
                tensors[key] = {"shape": list(ref.shape), "max_abs_error": float(np.max(np.abs(diff))),
                                "relative_l2_error": rel, "limit": limit, "pass": rel < limit}
                passed &= tensors[key]["pass"]
            cosine = float(np.sum(actual["normalized_embedding"] * expected["normalized_embedding"]))
            passed &= cosine > (.999 if precision == "fp32" else .995)
            reports.append({"image": name, "tensors": tensors, "global_cosine": cosine, "pass": bool(passed)})
    result = {"status": "PASS_REAL_PRETRAINED" if all(v["pass"] for v in reports) else "FAIL", "checkpoint": checkpoint,
              "image": str(a.image.resolve()), "image_sha256": hashlib.sha256(a.image.read_bytes()).hexdigest(),
              "precision": precision, "checks": reports}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if result["status"] != "PASS_REAL_PRETRAINED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
