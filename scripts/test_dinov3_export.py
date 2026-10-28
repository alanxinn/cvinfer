"""Export REAL DINOv3 operators with RANDOM tiny weights; not pretrained quality validation."""
import argparse
from pathlib import Path
import torch
from export_dinov3 import export_bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("cache/dinov3-tiny"))
    a = p.parse_args()
    from transformers import (DINOv3ViTConfig, DINOv3ViTModel, DINOv3ConvNextConfig, DINOv3ConvNextModel, DINOv3ViTImageProcessor)
    torch.manual_seed(123)
    torch.set_num_threads(2)
    processor = DINOv3ViTImageProcessor(size={"height": 64, "width": 64})
    vit = DINOv3ViTConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                          image_size=64, patch_size=16, num_register_tokens=4)
    vit._attn_implementation = "eager"
    conv = DINOv3ConvNextConfig(hidden_sizes=[8, 16, 24, 32], depths=[1, 1, 1, 1], image_size=64)
    for name, model in [("vit", DINOv3ViTModel(vit)), ("convnext", DINOv3ConvNextModel(conv))]:
        export_bundle(model, processor, a.output / name / "onnx", 64, "RANDOM_TINY_DINOV3_NOT_PRETRAINED", synthetic=True)


if __name__ == "__main__":
    main()
