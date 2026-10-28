"""Export an authorized SAM3 Hugging Face checkpoint into six TensorRT subgraphs.

Requires transformers==5.8.0 and torch>=2.7 in a separate environment from RF-DETR.
No checkpoint or authentication token is bundled in the repository.
"""
import argparse
import gc
import json
import os
from pathlib import Path
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")


def export_tokenizer(tokenizer, output):
    # CLIPTokenizer's slow and fast implementations expose the same BPE data.
    if hasattr(tokenizer, "backend_tokenizer"):
        model = json.loads(tokenizer.backend_tokenizer.to_str())["model"]
        vocab, merges = model["vocab"], model["merges"]
        merges = [tuple(pair.split(" ")) if isinstance(pair, str) else tuple(pair) for pair in merges]
    else:
        vocab = tokenizer.get_vocab()
        merges = [pair for pair, rank in sorted(tokenizer.bpe_ranks.items(), key=lambda item: item[1])]
    header = f"{tokenizer.bos_token_id}\t{tokenizer.eos_token_id}\t{tokenizer.pad_token_id}\n"
    (output / "vocab.tsv").write_text(header + "".join(f"{i}\t{word.encode().hex()}\n" for word, i in vocab.items()), encoding="utf-8")
    (output / "merges.tsv").write_text("".join(f"{i}\t{a.encode().hex()}\t{b.encode().hex()}\n" for i, (a, b) in enumerate(merges)), encoding="utf-8")


def wrappers(torch):
    from transformers.models.sam3 import modeling_sam3
    from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput
    # Replace only the tracing path of the library's padding-mask factory.
    # Non-traced PyTorch reference inference still uses the original factory.
    if not getattr(modeling_sam3.create_bidirectional_mask, "_cvinfer_export", False):
        original_mask = modeling_sam3.create_bidirectional_mask
        def export_mask(config, inputs_embeds, attention_mask, encoder_hidden_states=None, **kwargs):
            if not torch.jit.is_tracing():
                return original_mask(config=config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                     encoder_hidden_states=encoder_hidden_states, **kwargs)
            if attention_mask is None or attention_mask.dim() == 4:
                return attention_mask
            blocked = ~attention_mask.bool()[:, None, None, :]
            bias = torch.where(blocked, torch.full((), torch.finfo(inputs_embeds.dtype).min,
                               dtype=inputs_embeds.dtype, device=inputs_embeds.device),
                               torch.zeros((), dtype=inputs_embeds.dtype, device=inputs_embeds.device))
            return bias.expand(inputs_embeds.shape[0], 1, inputs_embeds.shape[1], attention_mask.shape[-1])
        export_mask._cvinfer_export = True
        modeling_sam3.create_bidirectional_mask = export_mask

    class DetectorVision(torch.nn.Module):
        def __init__(self, model):
            super().__init__(); self.model = model
        def forward(self, pixel_values):
            v = self.model.get_vision_features(pixel_values)
            return tuple(v.fpn_hidden_states[:-1]) + tuple(v.fpn_position_encoding[:-1])

    class TextEncoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__(); self.model = model
        def forward(self, input_ids, attention_mask):
            # Build CLIP's additive causal+padding mask explicitly. The new
            # Transformers mask factory uses tracing-incompatible vmap helpers
            # with torch 2.7; the layers/weights and attention math stay identical.
            clip = self.model.text_encoder.text_model
            hidden = clip.embeddings(input_ids=input_ids)
            length = input_ids.shape[1]
            causal = torch.triu(torch.ones(length, length, device=input_ids.device, dtype=torch.bool), diagonal=1)
            blocked = causal[None, None] | (~attention_mask.bool())[:, None, None, :]
            bias = torch.where(blocked, torch.full((), torch.finfo(hidden.dtype).min, dtype=hidden.dtype, device=hidden.device),
                               torch.zeros((), dtype=hidden.dtype, device=hidden.device))
            encoded = clip.encoder(inputs_embeds=hidden, attention_mask=bias).last_hidden_state
            return self.model.text_projection(clip.final_layer_norm(encoded))

    class TextDecoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__(); self.model = model
        def forward(self, f0, f1, f2, p0, p1, p2, text_features, attention_mask):
            # Sam3Model drops the final FPN level before decoding. Its value is
            # unused, so duplicate the last input to preserve the public contract.
            vision = Sam3VisionEncoderOutput(fpn_hidden_states=(f0, f1, f2, f2),
                                             fpn_position_encoding=(p0, p1, p2, p2))
            v = self.model(vision_embeds=vision, text_embeds=text_features, attention_mask=attention_mask)
            scores = v.pred_logits.sigmoid() * v.presence_logits.sigmoid()
            return v.pred_boxes, scores, v.pred_masks

    class TrackerVision(torch.nn.Module):
        def __init__(self, model):
            super().__init__(); self.model = model
        def forward(self, pixel_values):
            return tuple(self.model.get_image_embeddings(pixel_values))

    class BoxDecoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__(); self.model = model
        def forward(self, f0, f1, f2, input_boxes):
            v = self.model(image_embeddings=[f0, f1, f2], input_boxes=input_boxes, multimask_output=True)
            return v.pred_masks, v.iou_scores

    class InteractiveDecoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__(); self.model = model
        def forward(self, f0, f1, f2, input_boxes, input_points, input_labels):
            v = self.model(image_embeddings=[f0, f1, f2], input_boxes=input_boxes,
                           input_points=input_points, input_labels=input_labels, multimask_output=True)
            return v.pred_masks, v.iou_scores
    return DetectorVision, TextEncoder, TextDecoder, TrackerVision, BoxDecoder, InteractiveDecoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="facebook/sam3", help="Authorized HF repository or local snapshot directory")
    p.add_argument("--output", type=Path, default=Path("cache/sam3/onnx"))
    p.add_argument("--max-points", type=int, default=32)
    a = p.parse_args()
    if a.max_points < 1:
        p.error("--max-points must be positive")
    from huggingface_hub import get_token
    if not Path(a.checkpoint).exists() and not get_token():
        p.error("SAM3 official weights require approved Hugging Face access. Log in locally with hf auth login or pass --checkpoint LOCAL_SNAPSHOT; do not put a token in this command.")
    import torch
    import onnx
    import transformers
    from transformers import Sam3Model, Sam3Processor, Sam3TrackerModel
    if transformers.__version__ != "5.8.0":
        p.error("Use the validated exporter API version transformers==5.8.0 in a separate environment")
    torch.set_num_threads(4)
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    DVision, Text, DHead, TVision, Box, Click = wrappers(torch)
    records = []

    def save(name, module, inputs, input_names, output_names, dynamic=None):
        path = output / (name + ".onnx")
        module.eval()
        with torch.inference_mode():
            torch.onnx.export(module, inputs, str(path), input_names=input_names, output_names=output_names,
                              opset_version=17, dynamic_axes=dynamic, dynamo=False, do_constant_folding=True)
        # Large encoders can use ONNX external data. Preserve the entire folder.
        onnx.checker.check_model(str(path))
        records.append({"name": name, "onnx": str(path), "inputs": input_names, "outputs": output_names})
        print("Exported", name, flush=True)

    processor = Sam3Processor.from_pretrained(a.checkpoint)
    export_tokenizer(processor.tokenizer, output)
    detector = Sam3Model.from_pretrained(a.checkpoint, attn_implementation="eager", dtype=torch.float32).eval()
    size = detector.config.vision_config.image_size
    if not isinstance(size, int):
        raise ValueError("This exporter requires square SAM3 image inputs")
    length = detector.config.text_config.max_position_embeddings
    pixels = torch.zeros(1, 3, size, size)
    encoded = processor.tokenizer("person", padding="max_length", max_length=length, truncation=True, return_tensors="pt")
    ids, attention = encoded.input_ids, encoded.attention_mask
    with torch.inference_mode():
        features = DVision(detector)(pixels)
        text = Text(detector)(ids, attention)
    save("detector_vision", DVision(detector), (pixels,), ["pixel_values"], ["f0", "f1", "f2", "p0", "p1", "p2"])
    save("text_encoder", Text(detector), (ids, attention), ["input_ids", "attention_mask"], ["text_features"])
    save("text_decoder", DHead(detector), (*features, text, attention),
         ["f0", "f1", "f2", "p0", "p1", "p2", "text_features", "attention_mask"], ["boxes", "scores", "masks"])
    del detector, features, text
    gc.collect()
    tracker = Sam3TrackerModel.from_pretrained(a.checkpoint, attn_implementation="eager", dtype=torch.float32).eval()
    tracker_size = tracker.config.vision_config.image_size
    tracker_pixels = torch.zeros(1, 3, tracker_size, tracker_size)
    with torch.inference_mode():
        tf = TVision(tracker)(tracker_pixels)
    box = torch.tensor([[[.1 * tracker_size, .1 * tracker_size, .8 * tracker_size, .8 * tracker_size]]])
    points = torch.tensor([[[[.4 * tracker_size, .4 * tracker_size], [.15 * tracker_size, .15 * tracker_size]]]])
    labels = torch.tensor([[[1, 0]]], dtype=torch.int64)
    save("tracker_vision", TVision(tracker), (tracker_pixels,), ["pixel_values"], ["f0", "f1", "f2"])
    save("box_decoder", Box(tracker), (*tf, box), ["f0", "f1", "f2", "input_boxes"], ["masks", "iou_scores"])
    save("interactive_decoder", Click(tracker), (*tf, box, points, labels),
         ["f0", "f1", "f2", "input_boxes", "input_points", "input_labels"], ["masks", "iou_scores"],
         {"input_points": {2: "points"}, "input_labels": {2: "points"}})
    manifest = {"checkpoint": a.checkpoint, "transformers": transformers.__version__, "torch": torch.__version__,
                "detector_size": size, "tracker_size": tracker_size, "text_length": length,
                "max_points": a.max_points, "graphs": records, "status": "ONNX_EXPORTED_ENGINE_VALIDATION_REQUIRED"}
    (output / "bundle.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
