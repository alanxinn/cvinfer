"""Export authorized DINOv3 ViT/ConvNeXt image-feature models to a fixed-size ONNX bundle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

OUTPUTS = ["global_embedding", "patch_tokens", "feature_map"]


def feature_wrapper(torch, model):
    kind = model.config.model_type
    if kind == "dinov3_vit":
        stride = model.config.patch_size
        if not isinstance(stride, int):
            raise ValueError("Only square DINOv3 ViT patches are supported")
        prefix = 1 + model.config.num_register_tokens
    elif kind == "dinov3_convnext":
        stride, prefix = 4 * 2 ** (len(model.config.hidden_sizes) - 1), 1
    else:
        raise ValueError("Expected DINOv3 ViT or ConvNeXt, got " + kind)

    class Features(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.model = model
        def forward(self, pixel_values):
            output = self.model(pixel_values=pixel_values)
            patches = output.last_hidden_state[:, prefix:]
            fmap = patches.transpose(1, 2).reshape(pixel_values.shape[0], patches.shape[-1],
                                                  pixel_values.shape[-2] // stride, pixel_values.shape[-1] // stride)
            return output.pooler_output, patches, fmap
    return Features().eval(), stride


def export_bundle(model, processor, output, image_size, checkpoint, synthetic=False):
    import torch
    import onnx
    import numpy as np
    import onnxruntime as ort
    import transformers
    module, stride = feature_wrapper(torch, model.eval())
    if image_size <= 0 or image_size % stride:
        raise ValueError(f"Image size must be a positive multiple of {stride}")
    # The native adapter uses this exact preprocessing contract. Reject
    # checkpoints with incompatible transforms instead of silently misusing them.
    if processor.__class__.__name__ != "DINOv3ViTImageProcessor" or not all(
            getattr(processor, flag, False) for flag in ["do_resize", "do_rescale", "do_normalize"]):
        raise ValueError("Use the canonical DINOv3 image processor with resize/rescale/normalize enabled")
    if not np.allclose(processor.image_mean, [.485, .456, .406]) or not np.allclose(processor.image_std, [.229, .224, .225]):
        raise ValueError("Unsupported DINOv3 processor normalization")
    if not np.isclose(processor.rescale_factor, 1 / 255) or int(processor.resample) != 2 or getattr(processor, "do_center_crop", False):
        raise ValueError("DINOv3 native preprocessing requires rescale + bilinear resize without center crop")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sample = torch.randn(1, 3, image_size, image_size)
    target = output / "model.onnx"
    with torch.inference_mode():
        reference = module(sample)
        torch.onnx.export(module, (sample,), str(target), input_names=["pixel_values"], output_names=OUTPUTS,
                          opset_version=17, dynamo=False, do_constant_folding=True)
    onnx.checker.check_model(str(target))
    options = ort.SessionOptions(); options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(target), options, providers=["CPUExecutionProvider"])
    actual = session.run(None, {"pixel_values": sample.numpy()})
    parity = {}
    for name, expected, values in zip(OUTPUTS, reference, actual):
        ref = expected.numpy()
        np.testing.assert_allclose(values, ref, atol=2e-4, rtol=2e-3)
        parity[name] = {"shape": list(values.shape), "max_abs_error": float(np.max(np.abs(values - ref)))}
    processor.save_pretrained(output / "processor")
    model.config.save_pretrained(output / "config")
    manifest = {"format": "cvinfer-dinov3-v1", "checkpoint": checkpoint, "model_type": model.config.model_type,
                "revision": getattr(model.config, "_commit_hash", None),
                "synthetic": synthetic, "image_size": image_size, "stride": stride,
                "register_tokens_excluded": getattr(model.config, "num_register_tokens", 0),
                "onnx": "model.onnx", "onnx_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "preprocess": "RGB rescale -> float antialiased bilinear resize -> ImageNet normalization",
                "torch": torch.__version__, "transformers": transformers.__version__, "outputs": parity,
                "status": "SYNTHETIC_ONNX_VERIFIED" if synthetic else "ONNX_VERIFIED_TENSORRT_VALIDATION_REQUIRED"}
    (output / "bundle.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="facebook/dinov3-vits16-pretrain-lvd1689m", help="Authorized HF repository or local HF snapshot directory")
    p.add_argument("--output", type=Path, default=Path("cache/dinov3/onnx"))
    p.add_argument("--image-size", type=int, default=224)
    a = p.parse_args()
    from huggingface_hub import get_token
    if not Path(a.checkpoint).exists() and not get_token():
        p.error("DINOv3 official weights require authorized access. Log in locally with hf auth login or pass --checkpoint LOCAL_HF_SNAPSHOT.")
    import torch
    import transformers
    from transformers import AutoConfig, AutoModel, AutoImageProcessor
    if transformers.__version__ != "5.8.0":
        p.error("Use transformers==5.8.0 for the validated export API")
    torch.set_num_threads(4)
    config = AutoConfig.from_pretrained(a.checkpoint)
    if config.model_type not in ("dinov3_vit", "dinov3_convnext"):
        p.error("Checkpoint must be a DINOv3 ViT or ConvNeXt backbone")
    processor = AutoImageProcessor.from_pretrained(a.checkpoint)
    attention = {"attn_implementation": "eager"} if config.model_type == "dinov3_vit" else {}
    model = AutoModel.from_pretrained(a.checkpoint, config=config, dtype=torch.float32, **attention).eval()
    export_bundle(model, processor, a.output, a.image_size, a.checkpoint)


if __name__ == "__main__":
    main()
