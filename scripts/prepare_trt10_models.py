"""Download official weights and export reproducible, raw-output ONNX models.

Run in a dedicated CPython environment; this script never installs packages.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import urllib.request

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

YOLO_MODELS = [
    ("yolo11n", 1), ("yolo11n-seg", 2), ("yolo11n-pose", 5),
    ("yolo26n", 6), ("yolo26n-seg", 7), ("yolo26n-pose", 8),
]
RF_MODELS = [
    ("rfdetr-nano", "rf-detr-nano.pth", "RFDETRNano", 3,
     "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
     "fb6504cce7fbdc783f7a46991f07639f"),
    ("rfdetr-seg-nano", "rf-detr-seg-nano.pt", "RFDETRSegNano", 4,
     "https://storage.googleapis.com/rfdetr/rf-detr-seg-n-ft.pth",
     "9995497791d0ff1664a1d9ddee9cfd20"),
]


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url, path, md5=None):
    if path.exists() and (not md5 or digest(path, "md5") == md5):
        return
    partial = path.with_suffix(path.suffix + ".download")
    for attempt in range(3):
        try:
            print(f"Downloading {url}", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "cvinfer-trt10"})
            with urllib.request.urlopen(request, timeout=120) as response, open(partial, "wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            if md5 and digest(partial, "md5") != md5:
                raise ValueError(f"Official weight MD5 mismatch: {path.name}")
            partial.replace(path)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("cache/trt10-models"))
    parser.add_argument("--models", nargs="*", help="Optional model names; default: all eight models")
    args = parser.parse_args()
    root = args.output.resolve()
    weights = root / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import onnx
    import torch
    import onnxslim
    from ultralytics import YOLO
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    download("https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg", root / "bus.jpg")
    manifest_path = root / "models.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"models": []}
    records = {m["name"]: m for m in manifest["models"]}

    def save(record):
        records[record["name"]] = record
        manifest.update({"packages": {p: importlib.metadata.version(p) for p in
                         ["torch", "ultralytics", "rfdetr", "onnx", "onnxruntime", "onnxslim"]},
                         "models": list(records.values())})
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for name, model_type in YOLO_MODELS:
        if args.models and name not in args.models:
            continue
        folder = root / name
        folder.mkdir(exist_ok=True)
        weight = weights / (name + ".pt")
        # Pin the official asset release, independent of future package defaults.
        source = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{weight.name}"
        download(source, weight)
        target = folder / "model.onnx"
        if not target.exists():
            model = YOLO(str(weight))
            exported = Path(model.export(format="onnx", imgsz=640, batch=1, dynamic=False,
                                        opset=17, simplify=True, device="cpu", half=False, nms=False))
            exported.replace(target)
        onnx.checker.check_model(str(target))
        save({"name": name, "type": model_type, "onnx": str(target), "weight": str(weight),
              "source": source, "weight_sha256": digest(weight), "onnx_sha256": digest(target),
              "precision": "fp16", "resolution": 640})
        print(f"Ready: {name}", flush=True)

    import rfdetr
    from PIL import Image
    import torchvision.transforms.functional as F
    for name, filename, class_name, model_type, source, md5 in RF_MODELS:
        if args.models and name not in args.models:
            continue
        folder = root / name
        folder.mkdir(exist_ok=True)
        weight = weights / filename
        download(source, weight, md5)
        target = folder / "model.onnx"
        if not target.exists():
            model = getattr(rfdetr, class_name)(pretrain_weights=str(weight), device="cpu")
            core = model.model.model.eval()
            size = model.model.resolution
            image = Image.open(root / "bus.jpg").convert("RGB")
            x = F.resize(F.normalize(F.to_tensor(image), [.485, .456, .406], [.229, .224, .225]), [size, size]).unsqueeze(0)
            with torch.inference_mode():
                raw = core(x)
            values = {"dets": raw["pred_boxes"].numpy(), "labels": raw["pred_logits"].numpy()}
            if "pred_masks" in raw:
                values["masks"] = raw["pred_masks"].numpy()
            np.save(folder / "input.npy", x.numpy())
            np.savez_compressed(folder / "pytorch.npz", **values)
            core.export()
            with torch.inference_mode():
                torch.onnx.export(core, x, str(target), input_names=["input"], output_names=list(values),
                                  opset_version=17, dynamo=False, do_constant_folding=True)
            graph = onnxslim.slim(onnx.load(target))
            onnx.checker.check_model(graph)
            onnx.save(graph, target)
            del model, core
        graph = onnx.load(target)
        size = graph.graph.input[0].type.tensor_type.shape.dim[2].dim_value
        save({"name": name, "type": model_type, "onnx": str(target), "weight": str(weight),
              "source": source, "weight_md5": md5, "weight_sha256": digest(weight),
              "onnx_sha256": digest(target), "precision": "fp32", "resolution": size})
        print(f"Ready: {name}", flush=True)


if __name__ == "__main__":
    main()
