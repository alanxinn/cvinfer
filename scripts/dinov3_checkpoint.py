"""Strict loading and local source fingerprints for pretrained DINOv3 backbones."""
import json
from pathlib import Path
from tensorrt_nuget_builder import sha256


def checkpoint_metadata(checkpoint):
    root = Path(checkpoint)
    if not root.exists():
        return None  # A remote repository is identified by its resolved revision.
    if not root.is_dir():
        raise ValueError("Use a local Hugging Face snapshot directory, not a raw .pth file")
    weights = sorted(root.glob("*.safetensors"))
    if not weights:
        raise ValueError("Local DINOv3 snapshot has no safetensors weights")
    files = [root / "config.json", root / "preprocessor_config.json"] + weights
    index = root / "model.safetensors.index.json"
    if index.exists():
        files.append(index)
        expected = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        if expected != {path.name for path in weights}:
            raise ValueError("DINOv3 weight shards differ from the safetensors index")
    return [{"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files]


def load_pretrained(checkpoint, config, **kwargs):
    import torch
    from transformers import AutoModel
    attention = {"attn_implementation": "eager"} if config.model_type == "dinov3_vit" else {}
    model, info = AutoModel.from_pretrained(checkpoint, config=config, dtype=torch.float32,
                                            output_loading_info=True, **attention, **kwargs)
    # from_pretrained can otherwise initialize missing parameters randomly and
    # still return a runnable model. Such a model is not a validated checkpoint.
    problems = {key: info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(key)}
    if problems:
        raise ValueError("DINOv3 checkpoint did not load strictly: " + str(problems))
    return model.eval()
