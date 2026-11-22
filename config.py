"""Central configuration. Every value can be overridden with an environment variable
(see .env.example). Values are validated and clamped so a typo cannot silently
disable filtering."""
import os
from dataclasses import dataclass
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent

# Models selectable from the UI / API. Weights are downloaded by Ultralytics on first use.
MODEL_CHOICES = {
    "yolo11n.pt": "YOLO11n - fastest, least accurate",
    "yolo11s.pt": "YOLO11s - balanced (default)",
    "yolo11m.pt": "YOLO11m - most accurate, slowest",
}
IMGSZ_CHOICES = (320, 416, 640, 960)   # selectable sizes (multiples of 32); 0 means "auto" (960 if hardware is fast enough, else 640)
EMBEDDER_CHOICES = ("mobilenet", "none")   # tracker appearance model: "none" = motion/IoU only (faster on CPU, less stable IDs)
STRIDE_MIN, STRIDE_MAX = 1, 10             # uploaded video: process every Nth frame

CONF_MIN, CONF_MAX = 0.05, 0.95
IOU_MIN, IOU_MAX = 0.30, 0.90


def _num(name, default, lo, hi, cast=float):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return min(max(cast(raw), lo), hi)
    except ValueError:
        raise SystemExit(f"Invalid value for {name}: {raw!r} (expected a number between {lo} and {hi})")


def _flag(name, default):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    v = raw.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise SystemExit(f"Invalid value for {name}: {raw!r} (use 1/0, true/false)")


def _choice(name, default, choices):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    v = raw.strip().lower()
    if v not in choices:
        raise SystemExit(f"Invalid value for {name}: {raw!r} (expected one of {', '.join(choices)})")
    return v


def _device(raw):
    """auto | cpu | cuda | cuda:N  (a bare number means that CUDA device)."""
    v = (raw or "auto").strip().lower()
    if v.isdigit():
        v = f"cuda:{v}"
    if v in ("auto", "cpu", "cuda") or (v.startswith("cuda:") and v[5:].isdigit()):
        return v
    raise SystemExit(f"Invalid DEVICE: {raw!r} (use auto, cpu, cuda or cuda:N)")


def _imgsz(raw):
    if raw is None or raw.strip().lower() in ("", "auto", "0"):
        return 0
    try:
        v = int(raw)
    except ValueError:
        raise SystemExit(f"Invalid IMGSZ: {raw!r} (use 'auto' or a multiple of 32 such as 640 / 960)")
    if v % 32 or not 320 <= v <= 1280:
        raise SystemExit(f"Invalid IMGSZ: {v} (must be a multiple of 32 between 320 and 1280)")
    return v


@dataclass(frozen=True)
class Config:
    default_model: str = "yolo11s.pt"
    conf: float = 0.40               # default confidence threshold
    iou: float = 0.50                # NMS IoU: lower = more aggressive duplicate removal
    imgsz: int = 0                   # 0 = auto
    max_det: int = 100
    latency_budget_ms: float = 150.0  # auto mode keeps 960 only if one inference is at most this slow
    min_box_px: int = 12             # boxes thinner than this (after clipping) are discarded
    dup_iou: float = 0.85            # near-identical boxes with different classes -> keep the more confident one
    max_frame_width: int = 1280      # larger frames are downscaled (INTER_AREA) before inference
    weights_dir: Path = BACKEND_DIR
    device: str = "auto"             # auto = CUDA GPU if PyTorch can really use one, else CPU
    half: bool = True                # FP16 inference (only ever used on CUDA)
    jpeg_quality: int = 80           # quality of the annotated frames streamed to the browser
    frame_stride: int = 1            # uploaded video: process every Nth frame (1 = every frame)
    tracker_embedder: str = "mobilenet"   # mobilenet (appearance + motion) | none (motion only)
    track_max_age: int = 30          # frames a lost track is remembered (lets people re-appear with the same ID)
    track_n_init: int = 3            # consecutive matches before a track is confirmed and drawn
    track_max_cosine: float = 0.30   # appearance distance gate (DeepSORT default 0.2 is strict -> ID switches)
    track_max_iou: float = 0.70      # IoU distance gate
    upload_ttl_hours: float = 24.0   # uploaded videos older than this are deleted


def load_config() -> Config:
    return Config(
        default_model=os.getenv("MODEL", "yolo11s.pt").strip() or "yolo11s.pt",
        conf=_num("CONF", 0.40, CONF_MIN, CONF_MAX),
        iou=_num("IOU", 0.50, IOU_MIN, IOU_MAX),
        imgsz=_imgsz(os.getenv("IMGSZ")),
        max_det=_num("MAX_DET", 100, 1, 300, int),
        latency_budget_ms=_num("LATENCY_BUDGET_MS", 150.0, 20.0, 5000.0),
        min_box_px=_num("MIN_BOX_PX", 12, 1, 200, int),
        dup_iou=_num("DUP_IOU", 0.85, 0.5, 1.0),
        max_frame_width=_num("MAX_FRAME_WIDTH", 1280, 320, 3840, int),
        device=_device(os.getenv("DEVICE")),
        half=_flag("HALF", True),
        jpeg_quality=_num("JPEG_QUALITY", 80, 40, 95, int),
        frame_stride=_num("FRAME_STRIDE", 1, STRIDE_MIN, STRIDE_MAX, int),
        tracker_embedder=_choice("TRACKER_EMBEDDER", "mobilenet", EMBEDDER_CHOICES),
        track_max_age=_num("TRACK_MAX_AGE", 30, 1, 300, int),
        track_n_init=_num("TRACK_N_INIT", 3, 1, 10, int),
        track_max_cosine=_num("TRACK_MAX_COSINE", 0.30, 0.05, 1.0),
        track_max_iou=_num("TRACK_MAX_IOU", 0.70, 0.1, 1.0),
        upload_ttl_hours=_num("UPLOAD_TTL_HOURS", 24.0, 0.0, 24.0 * 365),
    )
