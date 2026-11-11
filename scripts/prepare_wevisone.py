"""Download the official 2B/4B checkpoint and pinned inference sources; never runs inference."""
import argparse
import io
import json
from pathlib import Path
import urllib.request
import zipfile

SOURCE_REVISION = "9a6bcd85404505df0d05cfb5d97512ccacf35dee"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", choices=["2B", "4B"], default="2B")
    p.add_argument("--revision", default="main", help="HF checkpoint revision (resolved commit is recorded)")
    p.add_argument("--output", type=Path, default=Path("cache/wevisone"))
    p.add_argument("--source-revision", default=SOURCE_REVISION)
    a = p.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    model_id = f"tencent/WeVisOne-VL-{a.size}"
    revision = HfApi().model_info(model_id, revision=a.revision).sha
    checkpoint = a.output / a.size / "checkpoint"
    snapshot_download(model_id, revision=revision, local_dir=checkpoint,
                      allow_patterns=["*.json", "*.jinja", "*.safetensors", "mask_decoder.pt", "LICENSE", "README.md"])
    source = a.output / "source" / a.source_revision
    if not (source / "wevisone_vl/engine.py").is_file():
        url = f"https://github.com/Tencent/WeVisOne-VL/archive/{a.source_revision}.zip"
        request = urllib.request.Request(url, headers={"User-Agent": "cvinfer"})
        with urllib.request.urlopen(request) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        source.mkdir(parents=True, exist_ok=True)
        for item in archive.infolist():
            parts = Path(item.filename).parts[1:]
            if not parts or item.is_dir():
                continue
            destination = source.joinpath(*parts).resolve()
            if not destination.is_relative_to(source.resolve()):
                raise ValueError("Source archive path escapes output directory")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(item))
    receipt = {"model_id": model_id, "revision": revision, "checkpoint": str(checkpoint.resolve()),
               "source_revision": a.source_revision, "source": str(source.resolve())}
    (checkpoint.parent / "source.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
