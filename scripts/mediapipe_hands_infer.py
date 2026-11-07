"""Auto-prepare MediaPipe engines, infer hands, and write JSON/annotated image."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np
from trt10_runtime import bind, load_runtime
from mediapipe_hands_pipeline import draw, image_bgr, infer


class Config(C.Structure):
    _fields_ = [("palm_threshold", C.c_float), ("hand_threshold", C.c_float),
                ("nms_threshold", C.c_float), ("max_hands", C.c_int32)]


class Hand(C.Structure):
    _fields_ = [("box", C.c_float * 4), ("palm_score", C.c_float),
                ("confidence", C.c_float), ("handedness", C.c_float),
                ("landmarks", C.c_float * 63), ("world_landmarks", C.c_float * 63)]


class HandsSession:
    def __init__(self, root="cache/mediapipe-hands", build="build/trt10-release",
                 precision="fp32", palm_threshold=.5, hand_threshold=.5, nms=.3, max_hands=2):
        root, build = Path(root).resolve(), Path(build).resolve()
        self.handles = load_runtime(build)
        self.dll = C.CDLL(str(build / "bin/alanxinn_cvinfer_mediapipe.dll"))
        self.error = bind(self.dll, "hands_last_error", C.c_char_p)
        self.destroy = bind(self.dll, "hands_model_destroy", None, C.c_void_p)
        create = bind(self.dll, "hands_model_create", C.c_int, C.c_char_p, C.c_char_p,
                      C.POINTER(Config), C.POINTER(C.c_void_p))
        self.run = bind(self.dll, "hands_model_infer_bgr", C.c_int, C.c_void_p, C.c_void_p,
                        C.c_int32, C.c_int32, C.c_int32, C.POINTER(Hand), C.c_size_t, C.POINTER(C.c_size_t))
        config = Config(palm_threshold, hand_threshold, nms, max_hands)
        self.model = C.c_void_p()
        self.check(create(str(root / f"palm-{precision}.engine").encode(),
                          str(root / f"hand-{precision}.engine").encode(), C.byref(config), C.byref(self.model)))
        self.capacity = max_hands

    def check(self, status):
        if status:
            raise RuntimeError(self.error().decode("utf-8", errors="replace"))

    def infer_bgr(self, image):
        if not self.model:
            raise RuntimeError("MediaPipe session is closed")
        image = image_bgr(image)
        output, count = (Hand * self.capacity)(), C.c_size_t()
        self.check(self.run(self.model, image.ctypes.data, image.shape[1], image.shape[0], 3,
                            output, self.capacity, C.byref(count)))
        results = []
        for hand in output[:count.value]:
            result = {name: float(getattr(hand, name)) for name in ("palm_score", "confidence", "handedness")}
            result.update({"box": list(hand.box), "label": "Right" if hand.handedness >= .5 else "Left",
                           "landmarks": np.array(hand.landmarks).reshape(21, 3).tolist(),
                           "world_landmarks": np.array(hand.world_landmarks).reshape(21, 3).tolist()})
            results.append(result)
        return results

    def close(self):
        if self.model:
            self.destroy(self.model)
            self.model = C.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class RawEngineRunner:
    """Run existing native core executable without compiling the new adapter.

    This is a validation/debug backend; each inference starts a process.
    Production uses HandsSession and resident engines.
    """
    def __init__(self, root, build, precision="fp32"):
        self.root, self.build, self.precision = Path(root).resolve(), Path(build).resolve(), precision
        self.env = os.environ.copy()
        self.env["PATH"] = str(self.build / "bintrt") + os.pathsep + self.env.get("PATH", "")

    def __call__(self, kind, values):
        with tempfile.TemporaryDirectory(prefix="hands-", dir=self.root) as folder:
            folder = Path(folder)
            np.ascontiguousarray(values, dtype=np.float32).tofile(folder / "input.bin")
            completed = subprocess.run([str(self.build / "bin/trt_raw_outputs.exe"),
                str(self.root / f"{kind}-{self.precision}.engine"), str(folder / "input.bin"), str(folder)],
                env=self.env, capture_output=True, text=True, timeout=180)
            if completed.returncode:
                raise RuntimeError(completed.stderr or completed.stdout)
            outputs = {}
            for line in (folder / "outputs.tsv").read_text().splitlines():
                name, filename, dtype, *shape = line.split("\t")
                if dtype != "FP32":
                    raise ValueError("MediaPipe requires FP32 engine IO")
                outputs[name] = np.fromfile(folder / filename, dtype=np.float32).reshape(tuple(map(int, shape)))
            return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("cache/mediapipe-hands"))
    parser.add_argument("--build", type=Path, default=Path("build/trt10-release"))
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("cache/mediapipe-hands/result"))
    parser.add_argument("--backend", choices=["native", "raw"], default="native")
    parser.add_argument("--palm-threshold", type=float, default=.5)
    parser.add_argument("--hand-threshold", type=float, default=.5)
    parser.add_argument("--nms", type=float, default=.3)
    parser.add_argument("--max-hands", type=int, default=2)
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error("Cannot read input image")
    from prepare_mediapipe_hands import prepare
    prepare(args.root, args.build, args.precision)
    options = dict(palm_threshold=args.palm_threshold, hand_threshold=args.hand_threshold,
                   nms=args.nms, max_hands=args.max_hands)
    if args.backend == "raw":
        hands = infer(image, RawEngineRunner(args.root, args.build, args.precision), **options)
    else:
        with HandsSession(args.root, args.build, args.precision, **options) as session:
            hands = session.infer_bgr(image)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"image": str(args.image.resolve()), "backend": args.backend, "precision": args.precision, "hands": hands}
    (args.output / "hands.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not cv2.imwrite(str(args.output / "hands.jpg"), draw(image, hands)):
        raise OSError("Cannot write annotated image")
    print(f"Detected {len(hands)} hands; results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
