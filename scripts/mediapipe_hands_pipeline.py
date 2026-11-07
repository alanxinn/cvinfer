"""MediaPipe palm/landmark geometry shared by CLI and independent engine tests.

The pinned OpenCV Zoo models use RGB NHWC [0,1] input, 2016 palm anchors,
and 224-pixel landmark coordinates. Inference is supplied by a callable.
"""
import math
import cv2
import numpy as np

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
               (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def image_bgr(image):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("Expected nonempty uint8 HWC BGR image")
    return np.ascontiguousarray(image)


def anchors():
    return np.array([((x + .5) / side, (y + .5) / side)
                     for side, repeats in [(24, 2), (12, 6)]
                     for y in range(side) for x in range(side) for _ in range(repeats)], np.float32)


def palm_input(image):
    height, width = image.shape[:2]
    scale = 192 / max(width, height)
    w, h = int(width * scale), int(height * scale)
    w, h = max(1, w), max(1, h)
    left, top = (192 - w) // 2, (192 - h) // 2
    canvas = np.zeros((192, 192, 3), np.uint8)
    canvas[top:top+h, left:left+w] = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(canvas[None, :, :, ::-1], dtype=np.float32) / 255, (scale, left, top)


def decode_palms(outputs, transform, threshold=.5, nms=.3, max_hands=2):
    raw, logits = outputs["palm_boxes"].reshape(2016, 18), outputs["palm_scores"].reshape(2016)
    if not np.isfinite(raw).all() or not np.isfinite(logits).all():
        raise ValueError("Non-finite palm model outputs")
    scores = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
    selected = np.flatnonzero(scores >= threshold)
    scale, left, top = transform
    candidates = []
    centers = anchors() * 192
    for index in selected[np.argsort(-scores[selected], kind="stable")]:
        values = raw[index]
        center = (values[:2] + centers[index] - [left, top]) / scale
        size = values[2:4] / scale
        if np.any(size <= 0):
            continue
        box = np.r_[center - size / 2, center + size / 2]
        keep = True
        for other in candidates:
            intersection = np.maximum(0, np.minimum(box[2:], other["box"][2:]) -
                                      np.maximum(box[:2], other["box"][:2])).prod()
            union = size.prod() + np.prod(other["box"][2:] - other["box"][:2]) - intersection
            if intersection / max(union, 1e-12) > nms:
                keep = False
                break
        if keep:
            points = (values[4:].reshape(7, 2) + centers[index] - [left, top]) / scale
            candidates.append({"box": box, "points": points, "score": float(scores[index])})
            if len(candidates) == max_hands:
                break
    return candidates


def hand_input(image, palm):
    # Align wrist -> middle-finger MCP with the crop's upward axis. Construct
    # the affine transform directly, including border padding outside the image.
    direction = palm["points"][2] - palm["points"][0]
    angle = math.pi / 2 + math.atan2(float(direction[1]), float(direction[0]))
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.array([[cosine, sine], [-sine, cosine]])
    points = palm["points"] @ rotation.T
    lo, hi = points.min(0), points.max(0)
    center = (lo + hi) / 2 + [0, -.4 * (hi[1] - lo[1])]
    side = max(hi - lo) * 3
    if not np.isfinite(side) or side < 1e-3:
        return None
    factor = 224 / side
    affine = np.c_[rotation * factor, 112 - center * factor]
    crop = cv2.warpAffine(image, affine, (224, 224), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    values = np.ascontiguousarray(crop[None, :, :, ::-1], dtype=np.float32) / 255
    return values, cv2.invertAffineTransform(affine), rotation, side


def decode_hand(outputs, palm, geometry, threshold=.5):
    if any(not np.isfinite(value).all() for value in outputs.values()):
        raise ValueError("Non-finite hand model outputs")
    confidence = float(outputs["confidence"].ravel()[0])
    if confidence < threshold:
        return None
    _, inverse, rotation, side = geometry
    landmarks = outputs["landmarks"].reshape(21, 3).astype(np.float64).copy()
    landmarks[:, :2] = landmarks[:, :2] @ inverse[:, :2].T + inverse[:, 2]
    landmarks[:, 2] *= side / 224
    world = outputs["world_landmarks"].reshape(21, 3).astype(np.float64).copy()
    world[:, :2] = world[:, :2] @ rotation
    box = np.r_[landmarks[:, :2].min(0), landmarks[:, :2].max(0)]
    handedness = float(outputs["handedness"].ravel()[0])
    return {"box": box.tolist(), "palm_score": palm["score"], "confidence": confidence,
            "handedness": handedness, "label": "Right" if handedness >= .5 else "Left",
            "landmarks": landmarks.tolist(), "world_landmarks": world.tolist()}


def infer(image, run, palm_threshold=.5, hand_threshold=.5, nms=.3, max_hands=2):
    image = image_bgr(image)
    if any(not math.isfinite(t) or not 0 <= t <= 1 for t in (palm_threshold, hand_threshold, nms)):
        raise ValueError("Thresholds must be finite and in [0,1]")
    if not isinstance(max_hands, int) or not 1 <= max_hands <= 64:
        raise ValueError("max_hands must be in [1,64]")
    values, transform = palm_input(image)
    palms = decode_palms(run("palm", values), transform, palm_threshold, nms, max_hands)
    results = []
    for palm in palms:
        geometry = hand_input(image, palm)
        if geometry is not None:
            result = decode_hand(run("hand", geometry[0]), palm, geometry, hand_threshold)
            if result is not None:
                results.append(result)
    return results


def draw(image, hands):
    image = image.copy()
    for hand in hands:
        points = np.rint(np.array(hand["landmarks"])[:, :2]).astype(int)
        for a, b in CONNECTIONS:
            cv2.line(image, tuple(points[a]), tuple(points[b]), (0, 220, 0), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(image, tuple(point), 3, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(image, f'{hand["label"]} {hand["confidence"]:.3f}', tuple(points[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 0, 0), 2)
    return image
