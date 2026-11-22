"""Frame validation and detection filtering. Pure NumPy/OpenCV: no model or torch needed.

Nothing in here assigns or changes a class. Filters can only DROP detections (invalid
geometry, below threshold, or a near-identical box where the model itself was more
confident in a different class). Classes always come from the model output."""
import math
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Det:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int

    @property
    def w(self): return self.x2 - self.x1
    @property
    def h(self): return self.y2 - self.y1


def prepare_frame(frame, max_width=1280):
    """Validate a decoded frame and cap its width.

    Frames are BGR uint8 (what cv2.imdecode / VideoCapture return and what Ultralytics
    expects for numpy input; it converts to RGB internally, so do NOT convert here).
    Downscaling uses INTER_AREA, which keeps detail better than the default bilinear.
    """
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError("frame must be a uint8 numpy array")
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"unsupported frame shape {frame.shape}")
    h, w = frame.shape[:2]
    if h < 32 or w < 32:
        raise ValueError(f"frame too small ({w}x{h})")
    if w > max_width:
        s = max_width / w
        frame = cv2.resize(frame, (max_width, max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(frame)


def iou(a: Det, b: Det) -> float:
    iw = min(a.x2, b.x2) - max(a.x1, b.x1)
    ih = min(a.y2, b.y2) - max(a.y1, b.y1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def clean_detections(dets, frame_w, frame_h, valid_class_ids, conf_min, min_side=12, dup_iou=0.85):
    """Return (kept, Counter of drop reasons).

    Drop reasons: invalid (NaN/inf, unknown class id, empty box), too_small, low_conf,
    duplicate_class. Boxes are clipped to the frame so downstream crops never go out of range.
    """
    dropped = Counter()
    candidates = []
    for d in dets:
        vals = (d.x1, d.y1, d.x2, d.y2, d.conf)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals) or d.cls not in valid_class_ids:
            dropped["invalid"] += 1
            continue
        if d.conf < conf_min:
            dropped["low_conf"] += 1
            continue
        c = Det(max(0.0, min(d.x1, d.x2)), max(0.0, min(d.y1, d.y2)),
                min(float(frame_w), max(d.x1, d.x2)), min(float(frame_h), max(d.y1, d.y2)), d.conf, d.cls)
        if c.w <= 0 or c.h <= 0:
            dropped["invalid"] += 1
        elif min(c.w, c.h) < min_side:
            dropped["too_small"] += 1
        else:
            candidates.append(c)

    # Cross-class duplicates: when two boxes cover (almost) exactly the same region but name
    # different classes, they describe one physical object. Keep the class the model is more
    # confident in. Requires very high overlap so e.g. a person sitting on a chair is untouched.
    kept = []
    for d in sorted(candidates, key=lambda x: x.conf, reverse=True):
        if any(k.cls != d.cls and iou(k, d) >= dup_iou for k in kept):
            dropped["duplicate_class"] += 1
        else:
            kept.append(d)
    return kept, dropped
