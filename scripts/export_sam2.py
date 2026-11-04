"""Export official SAM2.1 image encoder and prompt/mask decoder to ONNX."""
import argparse
import json
from pathlib import Path
from sam2_checkpoint import VARIANTS, load_model


def wrappers(torch):
    class ImageEncoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__(); self.model = model

        def forward(self, pixel_values):
            features = self.model.forward_image(pixel_values)["backbone_fpn"]
            f0, f1, f2 = features[-3:]
            if self.model.directly_add_no_mem_embed:
                f2 = f2 + self.model.no_mem_embed.reshape(1, -1, 1, 1)
            return f0, f1, f2

    class MaskDecoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.prompt = model.sam_prompt_encoder
            self.decoder = model.sam_mask_decoder

        def forward(self, f0, f1, f2, point_coords, point_labels, mask_input, has_mask_input):
            # Labels 2/3 encode the box corners before ordinary points. The
            # official image predictor also appends a not-a-point token here.
            sparse, no_mask = self.prompt(points=(point_coords, point_labels), boxes=None, masks=None)
            mask = self.prompt._embed_masks(mask_input)
            present = has_mask_input.reshape(1, 1, 1, 1)
            dense = present * mask + (1 - present) * no_mask
            masks, scores, _, _ = self.decoder(
                image_embeddings=f2, image_pe=self.prompt.get_dense_pe(),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                multimask_output=True, repeat_image=False, high_res_features=[f0, f1])
            return masks, scores

    return ImageEncoder, MaskDecoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=VARIANTS, default="tiny")
    p.add_argument("--checkpoint", type=Path, help="Local official .pt; must match --variant")
    p.add_argument("--download", action=argparse.BooleanOptionalAction, default=True, help="Download missing weights from Meta (default: enabled)")
    p.add_argument("--output", type=Path, default=Path("cache/sam2.1/onnx"))
    p.add_argument("--max-points", type=int, default=32)
    a = p.parse_args()
    if not 1 <= a.max_points <= 4096:
        p.error("--max-points must be 1..4096")
    import torch
    import onnx
    torch.set_num_threads(4)
    model, source = load_model(a.variant, a.checkpoint, a.download)
    output = a.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    # Invalidate any previous manifest before writing graphs from a new source.
    (output / "bundle.json").unlink(missing_ok=True)
    ImageEncoder, MaskDecoder = wrappers(torch)
    pixels = torch.zeros(1, 3, 1024, 1024)
    with torch.inference_mode():
        features = ImageEncoder(model)(pixels)
    coords = torch.tensor([[[100., 100.], [900., 900.]]])
    labels = torch.tensor([[2, 3]], dtype=torch.int32)
    graphs = [
        ("image_encoder", ImageEncoder(model), (pixels,), ["pixel_values"], ["f0", "f1", "f2"], None),
        ("mask_decoder", MaskDecoder(model), (*features, coords, labels, torch.zeros(1, 1, 256, 256), torch.zeros(1)),
         ["f0", "f1", "f2", "point_coords", "point_labels", "mask_input", "has_mask_input"],
         ["masks", "iou_scores"], {"point_coords": {1: "prompts"}, "point_labels": {1: "prompts"}}),
    ]
    records = []
    for name, module, inputs, names, outputs, dynamic in graphs:
        path = output / (name + ".onnx")
        with torch.inference_mode():
            torch.onnx.export(module.eval(), inputs, str(path), input_names=names, output_names=outputs,
                              opset_version=17, dynamic_axes=dynamic, dynamo=False, do_constant_folding=True)
        onnx.checker.check_model(str(path))
        records.append({"name": name, "onnx": str(path), "inputs": names, "outputs": outputs})
        print("Exported", name, flush=True)
    manifest = {"format": "cvinfer-sam2.1-v1", "source": source, "image_size": 1024,
                "max_points": a.max_points, "graphs": records, "torch": torch.__version__,
                "preprocess": "RGB float /255; antialiased bilinear 1024 square; ImageNet mean/std",
                "mask_mode": "three candidates; highest IoU; optional previous low-resolution logits",
                "status": "ONNX_EXPORTED_ENGINE_VALIDATION_REQUIRED"}
    (output / "bundle.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
