"""Convert local Meta SAM 3.1 multiplex weights to strictly loaded IMAGE components.

Video propagation/multiplex tokens are deliberately excluded. The separate
interactive_sam_* head is the image prompt head; sam_mask_decoder is NOT it.
"""
import argparse
import gc
import hashlib
import json
import os
import re
from pathlib import Path
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
import torch
from transformers import (Sam3Config, Sam3Model, Sam3TrackerConfig, Sam3TrackerModel,
                          Sam3ImageProcessor, Sam3Processor, Sam3TrackerProcessor, CLIPTokenizer)
from sam3_checkpoint_mapping import convert_old_keys_to_new_keys, split_qkv


def vision_weights(raw):
    mapping = convert_old_keys_to_new_keys(list(raw))
    converted = {}
    for old, value in raw.items():
        key = mapping[old]
        if key.endswith(".rotary_emb.rope_embeddings"):
            continue  # HF computes the same non-persistent RoPE buffer.
        if key == "vision_encoder.backbone.embeddings.position_embeddings":
            value = value[:, 1:, :]  # Meta retains the unused pretrained CLS position.
        converted[key] = value
    return split_qkv(converted)


def interactive_weights(source):
    values = {}
    replacements = {
        "point_embeddings.": "point_embed.",
        "pe_layer.positional_encoding_gaussian_matrix": "shared_embedding.positional_embedding",
        "mask_downscaling.0.": "mask_embed.conv1.", "mask_downscaling.1.": "mask_embed.layer_norm1.",
        "mask_downscaling.3.": "mask_embed.conv2.", "mask_downscaling.4.": "mask_embed.layer_norm2.",
        "mask_downscaling.6.": "mask_embed.conv3.", "output_upscaling.0.": "upscale_conv1.",
        "output_upscaling.1.": "upscale_layer_norm.", "output_upscaling.3.": "upscale_conv2.",
        ".out_proj.": ".o_proj.", ".mlp.lin1.": ".mlp.proj_in.", ".mlp.lin2.": ".mlp.proj_out.",
        ".norm_final_attn.": ".layer_norm_final_attn.",
    }
    for key, value in source.items():
        if not key.startswith("tracker.model.interactive_sam_"):
            continue
        key = key.removeprefix("tracker.model.interactive_sam_")
        for old, new in replacements.items():
            key = key.replace(old, new)
        key = re.sub(r"\.norm([1-4])\.", r".layer_norm\1.", key)
        if any(part in key for part in ["iou_prediction_head.layers.", "pred_obj_score_head.layers.", "output_hypernetworks_mlps."]):
            key = re.sub(r"\.layers\.(\d+)\.", lambda m: [".proj_in.", ".layers.0.", ".proj_out."][int(m[1])], key)
        if key in values:
            raise ValueError("Duplicate converted tensor: " + key)
        values[key] = value
    values["no_memory_embedding"] = source["tracker.model.interactivity_no_mem_embed"]
    values["shared_image_embedding.positional_embedding"] = values["prompt_encoder.shared_embedding.positional_embedding"]
    values["prompt_encoder.point_embed.weight"] = torch.cat([values.pop(f"prompt_encoder.point_embed.{i}.weight") for i in range(4)])
    return values


def save_strict(model_type, config, values, target, processor):
    with torch.device("meta"):
        model = model_type(config)
    expected = model.state_dict()
    missing, extra = sorted(expected.keys() - values.keys()), sorted(values.keys() - expected.keys())
    shapes = [k for k in expected.keys() & values.keys() if expected[k].shape != values[k].shape]
    if missing or extra or shapes:
        raise ValueError(f"{model_type.__name__}: missing={missing}, extra={extra}, shape mismatch={shapes}")
    # Convert every loaded tensor to FP32 and own storage (the original mmap may
    # share point embeddings; safetensors must not silently omit aliases).
    values = {k: v.float().contiguous().clone() for k, v in values.items()}
    model.load_state_dict(values, strict=True, assign=True)
    model.save_pretrained(target, max_shard_size="2GB")
    processor.save_pretrained(target)
    record = {"directory": str(target.resolve()), "tensors": len(expected),
              "parameters": sum(p.numel() for p in model.parameters()), "missing": [], "unexpected": []}
    print(record, flush=True)
    del model, values
    gc.collect()
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("models/sam3.1/sam3.1_multiplex.pt"))
    p.add_argument("--assets", type=Path, default=Path("models/sam3.1"))
    p.add_argument("--output", type=Path, default=Path("cache/sam31/hf"))
    a = p.parse_args()
    torch.set_num_threads(4)
    import transformers
    if transformers.__version__ != "5.8.0":
        p.error("Use transformers==5.8.0")
    source = torch.load(a.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if "model" in source:
        source = source["model"]
    if "tracker.model.interactivity_no_mem_embed" not in source:
        p.error("Expected a SAM 3.1 multiplex checkpoint with the independent interactive image head")
    root_config = json.loads((a.assets / "config.json").read_text())
    config = Sam3Config(**root_config["detector_config"])
    # SAM 3.1 has three neck levels. The HF SAM3 detector normally drops a
    # fourth level; load_sam3_component appends an unused view, with no weights.
    config.vision_config.scale_factors = [4., 2., 1.]
    tokenizer = CLIPTokenizer.from_pretrained(a.assets, local_files_only=True)
    processor = Sam3Processor(image_processor=Sam3ImageProcessor(), tokenizer=tokenizer)
    raw = {k.removeprefix("detector."): v for k, v in source.items() if k.startswith("detector.")
           and not any(part in k for part in [".interactive_convs.", ".propagation_convs."])
           and not k.startswith("detector.geometry_encoder.points_")}
    values = vision_weights(raw)
    values["text_encoder.text_projection.weight"] = values["text_encoder.text_projection.weight"].T
    a.output.mkdir(parents=True, exist_ok=True)
    detector = save_strict(Sam3Model, config, values, a.output / "detector", processor)
    del raw, values
    gc.collect()
    raw = {k.removeprefix("detector.").replace(".interactive_convs.", ".convs."): v
           for k, v in source.items() if k.startswith("detector.backbone.vision_backbone.")
           and (".trunk." in k or ".interactive_convs." in k)}
    values = vision_weights(raw)
    values.update(interactive_weights(source))
    tracker_config = Sam3TrackerConfig()
    tracker_config.vision_config.scale_factors = [4., 2., 1.]
    tracker = save_strict(Sam3TrackerModel, tracker_config, values, a.output / "tracker",
                         Sam3TrackerProcessor(image_processor=Sam3ImageProcessor()))
    h = hashlib.sha256()
    with a.checkpoint.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(block)
    sha = h.hexdigest()
    manifest = {"format": "cvinfer-sam31-images-v1", "checkpoint": str(a.checkpoint.resolve()),
                "checkpoint_sha256": sha, "synthetic": False, "transformers": transformers.__version__,
                "components": {"detector": detector, "tracker": tracker},
                "scope": "Image PCS and independent interactive PVS; video multiplex propagation excluded",
                "status": "STRICT_WEIGHT_MAPPING_PASSED_INFERENCE_VALIDATION_REQUIRED"}
    (a.output / "sam31.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
