"""YOLO model management: loading, runtime switching, class-name validation, inference.

torch / ultralytics are imported lazily inside _load_model so the rest of the package
(filters, consistency, drawing, engine) can be imported and tested without them.
"""
import logging
import os
import platform
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..config import MODEL_CHOICES, Config
from .filters import Det

log = logging.getLogger("detector")
FALLBACK_ORDER = ("yolo11s.pt", "yolo11m.pt", "yolo11n.pt")


def validate_names(names, nc=None) -> dict:
    """Check the class-name table that ships INSIDE the weights file and return {int: str}.

    This is the single source of truth for id -> label: nothing in the app keeps its own
    class list. Fails loudly if ids are not contiguous 0..N-1, any name is blank, or the
    detection head's class count (nc) disagrees with the number of names.
    """
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    if not hasattr(names, "items") or not names:
        raise ValueError("model has no class names")
    out = {int(k): str(v).strip() for k, v in names.items()}
    if sorted(out) != list(range(len(out))):
        raise ValueError("class ids are not contiguous 0..N-1")
    if any(not v for v in out.values()):
        raise ValueError("model has a blank class name")
    if nc is not None and int(nc) != len(out):
        raise ValueError(f"detection head outputs {nc} classes but the model lists {len(out)} names")
    return out


CUDA_INSTALL_HINT = ("pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu126  "
                     "(use cu128 or cu130 for newer GPUs/drivers; run `nvidia-smi` to see which CUDA version your driver supports)")


@dataclass(frozen=True)
class DeviceChoice:
    """What the app WILL try to use and why. The device that is really used is read back from the
    loaded model's weights afterwards (see Detector._load_model), so the UI never reports a guess."""
    device: str                    # "cpu" | "cuda:N"
    reason: str                    # human-readable explanation shown in the UI
    gpu_name: str | None
    cuda_available: bool
    torch_version: str
    torch_cuda: str | None         # CUDA version this PyTorch wheel was built for (None = CPU-only wheel)


def resolve_device(requested: str, torch) -> DeviceChoice:
    """Decide between CPU and CUDA. `torch` is passed in so this is testable without PyTorch.

    requested: auto | cpu | cuda | cuda:N. A CUDA request that cannot be honoured falls back to CPU
    (with the reason) instead of crashing, so CPU fallback always works."""
    tv = str(getattr(torch, "__version__", "?"))
    built = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        avail = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - broken driver etc.
        avail = False
    req = (requested or "auto").lower()

    def cpu(reason):
        return DeviceChoice("cpu", reason, None, avail, tv, built)

    if req == "cpu":
        return cpu("CPU selected with DEVICE=cpu.")
    if not avail:
        if built is None:
            why = (f"This PyTorch ({tv}) is a CPU-only build, so the GPU cannot be used even if you have one. "
                   f"On Windows pip installs the CPU build by default. To enable NVIDIA GPU acceleration run: {CUDA_INSTALL_HINT}")
        else:
            why = (f"PyTorch {tv} has CUDA {built} support but no usable NVIDIA GPU/driver was found "
                   f"(check `nvidia-smi`; update the NVIDIA driver if the card is present).")
        return cpu("Running on CPU. " + why)
    idx = 0
    if req.startswith("cuda:"):
        idx = int(req[5:])
    try:
        count = int(torch.cuda.device_count())
        if idx >= count:
            return cpu(f"Running on CPU: {requested} was requested but only {count} CUDA device(s) exist.")
        name = str(torch.cuda.get_device_name(idx))
    except Exception as e:  # noqa: BLE001
        return cpu(f"Running on CPU: could not query CUDA device {idx} ({type(e).__name__}: {e}).")
    return DeviceChoice(f"cuda:{idx}", f"NVIDIA GPU detected: {name} (CUDA {built}).", name, True, tv, built)


@dataclass(frozen=True)
class Loaded:
    name: str
    model: object
    names: dict
    device: str
    auto_imgsz: int
    latency_ms: float | None
    half: bool = False             # FP16 inference in use (CUDA only)
    device_name: str = "CPU"


class Detector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.status = "loading"          # loading | loaded | error  (error = nothing could be loaded)
        self.error = None                # fatal error, only while status == "error"
        self.last_error = None           # last problem (failed switch / fallback notice); UI shows it
        self.loading = None              # name of the model currently being loaded
        self.generation = 0              # increments whenever the active model changes
        self.choice: DeviceChoice | None = None   # set at the start of load(); None until PyTorch has been inspected
        self.device_note = None          # why this device was chosen / why the GPU is not in use
        self._active: Loaded | None = None
        self._state = threading.Lock()

    # ---- convenience accessors (kept for API compatibility) ----
    @property
    def active(self): return self._active
    @property
    def name(self): return self._active.name if self._active else self.cfg.default_model
    @property
    def device(self):
        """The device the ACTIVE model really runs on; None until a model is loaded (never a guess)."""
        return self._active.device if self._active else None
    @property
    def names(self): return self._active.names if self._active else {}

    def weights_path(self, name) -> Path:
        p = Path(name)
        return p if p.is_absolute() else self.cfg.weights_dir / p

    # ---- loading ----
    def load(self):
        """Initial load; run it in a background thread. Falls back to any locally available
        weights if the default cannot be loaded (e.g. first run without internet), and says so."""
        default = self.cfg.default_model
        order = [default] + [m for m in FALLBACK_ORDER if m != default and self.weights_path(m).exists()]
        errors = []
        try:
            for name in order:
                self.loading = name
                try:
                    self._activate(self._load_model(name))
                    if errors:
                        self.last_error = (f"{default} could not be loaded ({errors[0]}); "
                                           f"running on {name} instead.")
                    return
                except Exception as e:  # noqa: BLE001 - report any load failure to the UI
                    errors.append(f"{name}: {type(e).__name__}: {e}")
                    log.error("failed to load %s: %s", name, e)
            self.status, self.error = "error", "; ".join(errors)
        finally:
            self.loading = None

    def select(self, name: str) -> bool:
        """Start loading `name` in the background; the current model keeps serving meanwhile.
        Returns False if it is already active. Raises ValueError (unknown) / RuntimeError (busy)."""
        if name not in MODEL_CHOICES and name != self.cfg.default_model:
            raise ValueError(f"Unknown model {name!r}. Choose one of: {', '.join(MODEL_CHOICES)}")
        with self._state:
            if self.loading:
                raise RuntimeError(f"Already loading {self.loading}; wait for it to finish.")
            if self._active and self._active.name == name:
                return False
            self.loading, self.last_error = name, None
        threading.Thread(target=self._switch, args=(name,), daemon=True).start()
        return True

    def _switch(self, name):
        try:
            self._activate(self._load_model(name))
        except Exception as e:  # noqa: BLE001
            self.last_error = (f"Could not load {name}: {type(e).__name__}: {e}. "
                               f"Still using {self.name}. (First use needs internet to download the weights.)")
            log.error(self.last_error)
        finally:
            self.loading = None

    def _activate(self, loaded: Loaded):
        self._active = loaded
        self.generation += 1
        self.status, self.error = "loaded", None
        log.info("active model %s on %s, auto imgsz %s (%.0f ms)", loaded.name, loaded.device,
                 loaded.auto_imgsz, loaded.latency_ms or -1)

    def _load_model(self, name) -> Loaded:
        import torch
        from ultralytics import YOLO
        choice = resolve_device(self.cfg.device, torch)
        self.choice, self.device_note = choice, choice.reason
        model = YOLO(str(self.weights_path(name)))   # downloads official weights on first use
        try:
            model.to(choice.device)
        except Exception as e:  # noqa: BLE001 - e.g. CUDA out of memory / driver problem: fall back, say so
            if choice.device == "cpu":
                raise
            log.warning("could not move the model to %s (%s); falling back to CPU", choice.device, e)
            choice = replace(choice, device="cpu", gpu_name=None,
                             reason=f"Running on CPU: moving the model to {choice.device} failed ({type(e).__name__}: {e}).")
            self.choice, self.device_note = choice, choice.reason
            model.to("cpu")
        device = self._actual_device(model, choice.device)      # trust the weights, not the request
        if device != choice.device:
            log.warning("requested %s but the weights are on %s", choice.device, device)
            self.device_note = f"Requested {choice.device} but the model is on {device}."
        names = validate_names(model.names, getattr(getattr(model, "model", None), "nc", None))
        half = bool(self.cfg.half and device.startswith("cuda"))
        imgsz, ms = self._probe(model, device, half)
        dev_name = (choice.gpu_name or "GPU") if device.startswith("cuda") else (platform.processor() or "CPU")
        return Loaded(name, model, names, device, imgsz, ms, half, dev_name)

    @staticmethod
    def _actual_device(model, fallback: str) -> str:
        """Read the device from the first weight tensor (the only trustworthy source)."""
        try:
            d = next(model.model.parameters()).device
            return d.type if d.type == "cpu" else f"{d.type}:{d.index or 0}"
        except Exception:  # noqa: BLE001
            return fallback

    def _probe(self, model, device, half=False):
        """Time one 1280x720 inference at 960. Auto mode keeps 960 only if it fits the latency
        budget, otherwise 640. The blank image is used for TIMING ONLY; its output is ignored.
        The first calls double as the warm-up, so the first real frame is not slow."""
        blank = np.full((720, 1280, 3), 114, np.uint8)
        try:
            times = []
            for i in range(4):                      # first calls include one-time warm-up cost
                t = time.perf_counter()
                model.predict(blank, imgsz=960, conf=0.99, verbose=False, device=device, half=half)
                if i >= 2:
                    times.append((time.perf_counter() - t) * 1000)
            ms = sum(times) / len(times)
            return (960 if ms <= self.cfg.latency_budget_ms else 640), ms
        except Exception as e:  # noqa: BLE001 - e.g. GPU out of memory at 960
            log.warning("latency probe failed (%s); defaulting to 640", e)
            return 640, None

    # ---- inference ----
    def detect(self, frame, conf, iou, imgsz, class_ids, active: Loaded | None = None):
        """Run the model on a BGR frame and return raw detections (class ids exactly as the
        model predicted them). `active` pins the model snapshot so ids and names always match."""
        a = active or self._active
        if a is None:
            raise RuntimeError("model not loaded")
        r = a.model.predict(frame, conf=conf, iou=iou, imgsz=imgsz or a.auto_imgsz,
                            classes=class_ids or None, max_det=self.cfg.max_det,
                            agnostic_nms=False, verbose=False, device=a.device,
                            half=bool(getattr(a, "half", False)))[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return []
        xyxy, cf, cl = b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)
        return [Det(float(xyxy[i, 0]), float(xyxy[i, 1]), float(xyxy[i, 2]), float(xyxy[i, 3]),
                    float(cf[i]), int(cl[i])) for i in range(len(cf))]

    def info(self) -> dict:
        a = self._active
        c = self.choice
        return dict(
            status=self.status, error=self.error, last_error=self.last_error, loading=self.loading,
            active=a.name if a else None, model=self.name, device=self.device,
            device_name=a.device_name if a else None, fp16=bool(a.half) if a else False,
            gpu_active=bool(a and a.device.startswith("cuda")), device_note=self.device_note,
            cuda_available=c.cuda_available if c else None, torch_version=c.torch_version if c else None,
            torch_cuda=c.torch_cuda if c else None,
            auto_imgsz=a.auto_imgsz if a else None,
            latency_ms=round(a.latency_ms, 1) if a and a.latency_ms else None,
            hint=self.performance_hint(),
            classes=list(a.names.values()) if a else [],
            available=[dict(name=n, label=l, downloaded=self.weights_path(n).exists())
                       for n, l in MODEL_CHOICES.items()],
        )

    def performance_hint(self):
        """Advice based only on what was measured at load time; None when there is nothing to say."""
        a = self._active
        if not a or a.device.startswith("cuda"):
            return None
        if a.latency_ms and a.latency_ms > self.cfg.latency_budget_ms:
            return ("Running on CPU, which is slow for real-time video. For smoother FPS pick YOLO11n and an "
                    "inference size of 416 or 320 px in Settings, or enable the NVIDIA GPU (see device note).")
        return None

    def system_info(self) -> dict:
        """Hardware/software facts for the Settings page. Everything here is queried, nothing assumed."""
        out = dict(python=platform.python_version(), os=f"{platform.system()} {platform.release()}",
                   cpu=platform.processor() or platform.machine(), cpu_threads=os.cpu_count(),
                   requested_device=self.cfg.device, fp16_enabled_in_config=self.cfg.half, cuda_install_hint=CUDA_INSTALL_HINT)
        try:
            import torch
            out.update(torch_version=str(torch.__version__), torch_cuda=torch.version.cuda,
                       cuda_available=bool(torch.cuda.is_available()), gpus=[])
            if out["cuda_available"]:
                for i in range(torch.cuda.device_count()):
                    p = torch.cuda.get_device_properties(i)
                    out["gpus"].append(dict(index=i, name=p.name, memory_gb=round(p.total_memory / 2**30, 1)))
        except Exception as e:  # noqa: BLE001 - torch missing or broken
            out.update(torch_version=None, torch_cuda=None, cuda_available=False, gpus=[],
                       torch_error=f"{type(e).__name__}: {e}")
        out.update({k: v for k, v in self.info().items() if k in ("device", "device_name", "fp16", "gpu_active", "device_note", "hint")})
        return out
