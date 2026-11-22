"""Live detection settings: validation first, then apply, so a bad value never leaves settings half-changed.

Kept separate from the web layer so the same rules apply to the REST endpoint, the WebSocket and the tests.
"""
from .config import CONF_MAX, CONF_MIN, EMBEDDER_CHOICES, IMGSZ_CHOICES, IOU_MAX, IOU_MIN, STRIDE_MAX, STRIDE_MIN


def current_settings(eng) -> dict:
    return {"conf": eng.conf, "iou": eng.iou, "imgsz": eng.imgsz or "auto", "imgsz_effective": eng.effective_imgsz(),
            "class_names": eng.class_names, "frame_stride": eng.frame_stride, "tracker_embedder": eng.embedder,
            "imgsz_choices": list(IMGSZ_CHOICES)}


def apply_settings(eng, det, d: dict) -> dict:
    new = {}
    if d.get("conf") is not None:
        c = float(d["conf"])
        if not CONF_MIN <= c <= CONF_MAX:
            raise ValueError(f"conf must be between {CONF_MIN} and {CONF_MAX}")
        new["conf"] = c
    if d.get("iou") is not None:
        i = float(d["iou"])
        if not IOU_MIN <= i <= IOU_MAX:
            raise ValueError(f"iou must be between {IOU_MIN} and {IOU_MAX}")
        new["iou"] = i
    if d.get("imgsz") is not None:
        s = 0 if str(d["imgsz"]).lower() == "auto" else int(d["imgsz"])
        if s != 0 and s not in IMGSZ_CHOICES:
            raise ValueError(f"imgsz must be 'auto' or one of {list(IMGSZ_CHOICES)}")
        new["imgsz"] = s
    if d.get("frame_stride") is not None:
        n = int(d["frame_stride"])
        if not STRIDE_MIN <= n <= STRIDE_MAX:
            raise ValueError(f"frame_stride must be between {STRIDE_MIN} and {STRIDE_MAX}")
        new["frame_stride"] = n
    if d.get("tracker_embedder") is not None:
        e = str(d["tracker_embedder"]).lower()
        if e not in EMBEDDER_CHOICES:
            raise ValueError(f"tracker_embedder must be one of {list(EMBEDDER_CHOICES)}")
        new["embedder"] = e                                   # takes effect at the next start/reset
    names = d.get("class_names")
    if names is None and d.get("classes") is not None:        # legacy clients sent class indices
        if not det.names:
            raise ValueError("Model not loaded yet; cannot map class indices.")
        names = [det.names[int(c)] for c in d["classes"] if int(c) in det.names]
    if names is not None:
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ValueError("class_names must be a list of strings")
        unknown = [n for n in names if det.names and n not in det.names.values()]
        if unknown:
            raise ValueError(f"Unknown class name(s): {', '.join(unknown)}")
        new["class_names"] = names
    for k, v in new.items():
        setattr(eng, k, v)
    return current_settings(eng)
