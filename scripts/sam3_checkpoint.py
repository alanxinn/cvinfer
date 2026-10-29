"""Load a HF SAM3 snapshot or the strictly converted SAM 3.1 IMAGE bundle."""
import json
from pathlib import Path
from types import MethodType


def component_path(checkpoint, component):
    root = Path(checkpoint)
    return str(root / component) if (root / "sam31.json").is_file() else str(checkpoint)


def load_sam3_component(checkpoint, component):
    import torch
    from transformers import Sam3Model, Sam3TrackerModel
    path = component_path(checkpoint, component)
    kind = Sam3Model if component == "detector" else Sam3TrackerModel
    model, info = kind.from_pretrained(path, attn_implementation="eager", dtype=torch.float32, output_loading_info=True)
    if any(info.get(key) for key in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]):
        raise ValueError("SAM3 component did not load strictly: " + str(info))
    if component == "detector" and (Path(checkpoint) / "sam31.json").is_file():
        if len(model.vision_encoder.neck.fpn_layers) != 3:
            raise ValueError("SAM 3.1 image bundle requires three FPN levels")
        original = model.vision_encoder.neck.forward
        def forward(neck, hidden_states):
            features, positions = original(hidden_states)
            # HF drops its fourth level. Meta SAM3.1 already contains only the
            # three consumed levels, so append an unused alias, not new weights.
            return features + (features[-1],), positions + (positions[-1],)
        model.vision_encoder.neck.forward = MethodType(forward, model.vision_encoder.neck)
    if component == "tracker" and (Path(checkpoint) / "sam31.json").is_file():
        # Meta SAM3.1 sets iou_prediction_use_sigmoid=False. Transformers 5.8
        # hardcodes True; preserve the checkpoint's original score contract.
        model.mask_decoder.iou_prediction_head.sigmoid_output = False
    return model.eval()


def source_metadata(checkpoint):
    path = Path(checkpoint) / "sam31.json"
    return json.loads(path.read_text()) if path.is_file() else {"checkpoint": str(checkpoint), "synthetic": False}
