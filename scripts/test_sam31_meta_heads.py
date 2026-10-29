"""Check converted image prompt/mask heads against unmodified Meta SAM3 source.

Pass a local clone of https://github.com/facebookresearch/sam3. Only independent
image-head modules are imported, without the optional video/training stack.
"""
import argparse
import json
import os
import subprocess
import sys
import types
from pathlib import Path
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
import torch
from convert_sam31_checkpoint import interactive_weights


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta-source", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=Path("models/sam3.1/sam3.1_multiplex.pt"))
    p.add_argument("--output", type=Path, default=Path("cache/sam31/meta-head-validation.json"))
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(7)
    package = types.ModuleType("sam3")
    package.__path__ = [str((a.meta_source / "sam3").resolve())]
    sys.modules["sam3"] = package
    from sam3.sam.prompt_encoder import PromptEncoder
    from sam3.sam.mask_decoder import MaskDecoder
    from sam3.sam.transformer import TwoWayTransformer
    from transformers import Sam3TrackerConfig
    from transformers.models.sam3_tracker.modeling_sam3_tracker import Sam3TrackerPromptEncoder, Sam3TrackerMaskDecoder
    source = torch.load(a.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    prompt = PromptEncoder(256, (72, 72), (1008, 1008), 16).eval()
    decoder = MaskDecoder(transformer_dim=256, transformer=TwoWayTransformer(2, 256, 8, 2048),
                          use_high_res_features=True, pred_obj_scores=True, pred_obj_scores_mlp=True,
                          use_multimask_token_for_obj_ptr=True).eval()
    for module, prefix in [(prompt, "prompt_encoder"), (decoder, "mask_decoder")]:
        sd = {k.removeprefix(f"tracker.model.interactive_sam_{prefix}."): v.float()
              for k, v in source.items() if k.startswith(f"tracker.model.interactive_sam_{prefix}.")}
        module.load_state_dict(sd, strict=True)
    config = Sam3TrackerConfig()
    config.mask_decoder_config._attn_implementation = "eager"
    hp = Sam3TrackerPromptEncoder(config.prompt_encoder_config).eval()
    hd = Sam3TrackerMaskDecoder(config.mask_decoder_config).eval()
    hd.iou_prediction_head.sigmoid_output = False
    converted = interactive_weights(source)
    for module, prefix in [(hp, "prompt_encoder"), (hd, "mask_decoder")]:
        module.load_state_dict({k.removeprefix(prefix + "."): v.float() for k, v in converted.items()
                                if k.startswith(prefix + ".")}, strict=True)
    features = torch.randn(1, 256, 72, 72)
    hi = [torch.randn(1, 32, 288, 288), torch.randn(1, 64, 144, 144)]
    box = torch.tensor([[100., 150., 700., 950.]])
    points = torch.tensor([[[350., 500.], [650., 900.]]])
    labels = torch.tensor([[1, 0]])
    checks = []
    with torch.inference_mode():
        for n in [0, 1, 2]:
            # Meta's public image predictor passes box corners as points with
            # labels 2/3, hence PromptEncoder adds its not-a-point padding token.
            # Decoder attention has no sequence-position encoding, so ordering
            # the foreground/background points before corners is equivalent.
            coords = torch.cat([points[:, :n], box.reshape(1, 2, 2)], dim=1)
            kinds = torch.cat([labels[:, :n], torch.tensor([[2, 3]])], dim=1)
            sparse, dense = prompt((coords, kinds), None, None)
            hs, he = hp(input_points=points[:, None, :n] if n else None,
                        input_labels=labels[:, None, :n] if n else None, input_boxes=box[:, None], input_masks=None)
            torch.testing.assert_close(sparse, hs[:, 0], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(dense, he, atol=1e-6, rtol=1e-6)
            meta = decoder(image_embeddings=features, image_pe=prompt.get_dense_pe(),
                           sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                           multimask_output=True, repeat_image=False, high_res_features=hi)
            hf = hd(image_embeddings=features, image_positional_embeddings=prompt.get_dense_pe(),
                    sparse_prompt_embeddings=hs, dense_prompt_embeddings=he,
                    multimask_output=True, high_resolution_features=hi)
            torch.testing.assert_close(meta[0], hf[0][:, 0], atol=3e-4, rtol=3e-4)
            torch.testing.assert_close(meta[1], hf[1][:, 0], atol=1e-5, rtol=1e-5)
            checks.append({"points": n, "mask_max_abs_error": float((meta[0] - hf[0][:, 0]).abs().max()),
                           "iou_max_abs_error": float((meta[1] - hf[1][:, 0]).abs().max()), "pass": True})
    revision = subprocess.check_output(["git", "-C", str(a.meta_source), "rev-parse", "HEAD"], text=True).strip()
    report = {"status": "PASS_META_IMAGE_HEADS", "meta_revision": revision, "checks": checks,
              "scope": "Real pretrained independent interactive prompt and mask heads; random input features"}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
