"""Official SAM2.1 configurations and checkpoint acquisition (no random fallback)."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import uuid

VARIANTS = {
    "tiny": ("sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": ("sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "base-plus": ("sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large": ("sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}
OFFICIAL_ROOT = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_path(variant, checkpoint=None, download=False):
    filename = VARIANTS[variant][1]
    path = Path(checkpoint) if checkpoint else Path("models/sam2.1") / filename
    if not path.is_file():
        if checkpoint or not download:
            raise FileNotFoundError(f"SAM2.1 checkpoint missing: {path}; provide --checkpoint or --download")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
        try:
            print("Downloading official SAM2.1 checkpoint:", filename, flush=True)
            with urllib.request.urlopen(OFFICIAL_ROOT + filename, timeout=120) as source, partial.open("wb") as target:
                size = 0
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    target.write(block); size += len(block)
                expected = source.headers.get("Content-Length")
                if not size or (expected is not None and size != int(expected)):
                    raise IOError("Incomplete SAM2.1 checkpoint download")
            os.replace(partial, path)
        finally:
            partial.unlink(missing_ok=True)
    return path.resolve()


def load_model(variant, checkpoint=None, download=False):
    from sam2.build_sam import build_sam2
    path = checkpoint_path(variant, checkpoint, download)
    config = "configs/sam2.1/" + VARIANTS[variant][0]
    # Image masks use the three-candidate mode, no video postprocessing kernels.
    model = build_sam2(config, str(path), device="cpu", apply_postprocessing=False).eval()
    metadata = {"variant": variant, "checkpoint": str(path), "sha256": sha256(path),
                "config": config, "official_url": OFFICIAL_ROOT + VARIANTS[variant][1]}
    return model, metadata


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=VARIANTS, default="tiny")
    a = p.parse_args()
    path = checkpoint_path(a.variant, download=True)
    print(json.dumps({"checkpoint": str(path), "sha256": sha256(path)}, indent=2))
