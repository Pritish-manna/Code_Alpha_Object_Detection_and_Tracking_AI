import threading
import time
from collections import Counter, deque

from ..config import Config
from .consistency import ClassConsistency
from .draw import class_color, draw_tracks
from .filters import clean_detections, prepare_frame
from .tracker import Tracker

FPS_WINDOW = 30          # frames used for the "current FPS" figure


class Engine:
    """detect -> validate/filter -> track -> temporal class vote -> draw.

    Every number and label that reaches the screen comes from the model's own output for
    that frame (or a short smoothed history of it). There are no class lists, overrides or
    default detections anywhere in this pipeline.

    Threading: `process()` / `reset()` are serialised by `_proc` (they may take a few hundred ms).
    All shared counters are mutated and read under `_data`, which is only ever held briefly, so
    `stats()` / `analytics()` can be called from the event loop or API threads at any moment without
    blocking on inference and without seeing half-updated or mutating dictionaries.
    """

    def __init__(self, detector, cfg: Config, tracker_factory=None):
        self.d, self.cfg = detector, cfg
        self._custom_tracker_factory = tracker_factory      # tests inject their own tracker here
        self._proc = threading.RLock()
        self._data = threading.RLock()
        self.conf, self.iou, self.imgsz = cfg.conf, cfg.iou, cfg.imgsz   # imgsz 0 = auto
        self.frame_stride = cfg.frame_stride                 # uploaded video: process every Nth frame
        self.embedder = cfg.tracker_embedder                 # mobilenet | none (applies from the next start)
        self.class_names: list[str] = []                     # empty = all classes
        self.running = False
        self._paused = False
        self.mode = None
        self.history, self.run_no = [], 0
        self.tracker = None
        self._tracker_embedder = None
        self._active_seen = None
        self._reset_state()

    # ---- pause bookkeeping (kept off the FPS figures) ----
    @property
    def paused(self):
        return self._paused

    @paused.setter
    def paused(self, value):
        self.pause() if value else self.resume()

    def pause(self):
        with self._data:
            if not self._paused:
                self._paused, self._pause_started = True, time.perf_counter()

    def resume(self):
        with self._data:
            if self._paused:
                self._paused = False
                if self._first_t is not None:
                    self._paused_total += time.perf_counter() - self._pause_started
                self._window.clear()                          # the gap must not drag the current FPS down

    # ---- state ----
    def _reset_state(self):
        with self._data:
            self.votes = ClassConsistency()
            self._smooth, self._hist, self.last_tracks = {}, {}, []
            self.run_no += 1
            self.ids, self.frames, self.total, self.conf_sum, self.tracked = set(), 0, 0, 0.0, 0
            self.skipped = 0                                  # frames deliberately not processed (frame stride)
            self.latency = 0.0
            self.timing = {}                                  # EMA of ms per pipeline stage
            self._window = deque(maxlen=FPS_WINDOW)
            self._first_t = self._last_t = None
            self._paused, self._pause_started, self._paused_total = False, 0.0, 0.0
            self.filtered = Counter()
            # per-class analytics, keyed by the displayed (voted) class name
            self.cls_total, self.cls_conf, self.cls_ids = Counter(), Counter(), {}
            self.started_at = time.time()

    def _build_tracker(self):
        if self._custom_tracker_factory is not None:
            return self._custom_tracker_factory()
        c = self.cfg
        self._tracker_embedder = self.embedder
        return Tracker(embedder=None if self.embedder == "none" else "mobilenet", max_age=c.track_max_age,
                       n_init=c.track_n_init, max_iou_distance=c.track_max_iou, max_cosine_distance=c.track_max_cosine)

    def reset(self):
        """Clear all session data and tracks. Waits for a frame that is currently being processed."""
        with self._proc:
            self._reset_state()
            stale = self._custom_tracker_factory is None and self._tracker_embedder != self.embedder
            if self.tracker is None or stale:
                self.tracker = self._build_tracker()          # (re)built only if missing or appearance mode changed
            else:
                self.tracker.reset()

    def effective_imgsz(self):
        a = self.d.active
        return self.imgsz or (a.auto_imgsz if a else None)

    def note_timing(self, name, ms):
        """Record a stage measured outside the engine (decode / encode) in the same EMA table."""
        with self._data:
            self._ema(name, ms)

    def note_skipped(self, n=1):
        with self._data:
            self.skipped += n

    def _ema(self, name, ms):
        prev = self.timing.get(name)
        self.timing[name] = ms if prev is None else 0.8 * prev + 0.2 * ms

    # ---- statistics ----
    def _fps_now(self):
        w = self._window
        return (len(w) - 1) / (w[-1] - w[0]) if len(w) >= 2 and w[-1] > w[0] else 0.0

    def _fps_avg(self):
        """Frames per second of ACTIVE processing time over the whole session (pauses excluded)."""
        if self.frames < 2 or self._first_t is None:
            return 0.0
        active = (self._last_t - self._first_t) - self._paused_total
        return (self.frames - 1) / active if active > 0 else 0.0

    def stats(self):
        with self._data:
            f = self.filtered
            return dict(
                total_detections=self.total, tracked_now=self.tracked, unique_ids=len(self.ids),
                fps=round(self._fps_now(), 1), avg_fps=round(self._fps_avg(), 1),
                frames=self.frames, frames_skipped=self.skipped,
                avg_confidence=round(100 * self.conf_sum / self.total, 1) if self.total else 0,
                latency_ms=round(self.latency, 1), timing_ms={k: round(v, 1) for k, v in self.timing.items()},
                model=self.d.name, imgsz=self.effective_imgsz(),
                device=getattr(self.d, "device", None), device_name=getattr(getattr(self.d, "active", None), "device_name", None),
                fp16=bool(getattr(getattr(self.d, "active", None), "half", False)),
                filtered_low_conf=f["low_conf"] + f["hidden_low_conf"], filtered_invalid=f["invalid"] + f["too_small"],
                filtered_duplicates=f["duplicate_class"],
                class_switches=self.votes.switches, class_flips_suppressed=self.votes.suppressed)

    def class_summary(self):
        """Per-class totals built only from detections that were actually drawn."""
        with self._data:
            total = self.total
            rows = [dict(cls=n, detections=c, unique_ids=len(self.cls_ids.get(n, ())),
                         avg_confidence=round(100 * self.cls_conf[n] / c, 1),
                         share_pct=round(100 * c / total, 1) if total else 0.0)
                    for n, c in self.cls_total.items() if c]
        return sorted(rows, key=lambda r: (-r["detections"], r["cls"]))

    def analytics(self):
        with self._data:
            elapsed = ((self._last_t - self._first_t) - self._paused_total) if self._first_t is not None else 0.0
            return dict(stats=self.stats(), classes=self.class_summary(), mode=self.mode,
                        running=self.running, paused=self._paused, run_no=self.run_no,
                        has_data=self.frames > 0, started_at=self.started_at,
                        active_seconds=round(max(elapsed, 0.0), 1), history_entries=len(self.history))

    def history_snapshot(self, n=100):
        with self._data:
            return [dict(h) for h in self.history[-n:][::-1]]

    # ---- helpers ----
    def _class_ids(self, active):
        if not self.class_names:
            return None
        wanted = set(self.class_names)
        return [i for i, n in active.names.items() if n in wanted]

    def _smooth_box(self, tid, box, w, h, alpha=0.7):
        prev = self._smooth.get(tid)
        cur = tuple(box) if prev is None else tuple(alpha * b + (1 - alpha) * p for b, p in zip(box, prev))
        self._smooth[tid] = cur
        return (max(0.0, cur[0]), max(0.0, cur[1]), min(float(w), cur[2]), min(float(h), cur[3]))

    def _work_width(self, active):
        """Frames wider than the inference size carry no extra information for the model (it letterboxes
        to imgsz), so they are shrunk first: cheaper tracking, drawing, JPEG encoding and network."""
        size = self.imgsz or getattr(active, "auto_imgsz", None)
        cap = self.cfg.max_frame_width
        return min(cap, max(int(size), 640)) if size else cap

    # ---- main entry ----
    def process(self, frame):
        with self._proc:
            t0 = time.perf_counter()
            active = self.d.active                      # one snapshot per frame: ids and names always match
            if active is None:
                raise RuntimeError("model not loaded")
            frame = prepare_frame(frame, self._work_width(active))
            if self.tracker is None:
                self.tracker = self._build_tracker()
            if active is not self._active_seen:         # model switched: old tracks/votes are meaningless
                if self._active_seen is not None:
                    self.tracker.reset()
                    with self._data:
                        self.votes, self._smooth = ClassConsistency(), {}
                self._active_seen = active

            ids = self._class_ids(active)
            t1 = time.perf_counter()
            raw = [] if ids == [] else self.d.detect(frame, self.conf, self.iou, self.imgsz, ids, active=active)
            t2 = time.perf_counter()

            h, w = frame.shape[:2]
            dets, dropped = clean_detections(raw, w, h, active.names, self.conf,
                                             self.cfg.min_box_px, self.cfg.dup_iou)
            obs, alive = self.tracker.update(dets, frame)
            t3 = time.perf_counter()

            items = []
            with self._data:
                ms = (t2 - t1) * 1000
                self.latency = ms if not self.latency else 0.8 * self.latency + 0.2 * ms
                self.filtered.update(dropped)
                self.votes.prune(alive)
                self._smooth = {k: v for k, v in self._smooth.items() if k in alive}
                self.last_tracks = []
                for o in obs:
                    v = self.votes.observe(o.track_id, o.cls, o.conf)
                    name = active.names.get(v.label)
                    conf = v.confidence                      # confidence of the DISPLAYED class
                    if name is None or conf + 1e-6 < self.conf:
                        self.filtered["hidden_low_conf"] += 1
                        continue
                    box = self._smooth_box(o.track_id, o.box, w, h)
                    self.ids.add(o.track_id)
                    self.total += 1
                    self.conf_sum += conf
                    self.cls_total[name] += 1
                    self.cls_conf[name] += conf
                    self.cls_ids.setdefault(name, set()).add(o.track_id)
                    items.append(dict(box=box, conf=conf, color=class_color(v.label),
                                      label=f"{name.upper()} | ID: {o.track_id:02d} | {int(round(conf * 100))}%"))
                    self.last_tracks.append(dict(id=o.track_id, cls=name, conf=conf, box=box,
                                                 raw_cls=active.names.get(o.cls), raw_conf=o.conf))
                    entry = self._hist.get(o.track_id)
                    if entry is None:
                        entry = self._hist[o.track_id] = dict(t=time.strftime("%H:%M:%S"), id=o.track_id,
                                                              cls=name, conf=round(conf, 2), run=self.run_no)
                        self.history.append(entry)
                    else:
                        entry["cls"] = name                   # keep history in step with genuine class changes
                self.history = self.history[-500:]
                self.tracked = len(items)

            draw_tracks(frame, items)
            t4 = time.perf_counter()
            with self._data:
                self.frames += 1
                self._window.append(t4)
                if self._first_t is None:
                    self._first_t = t4
                self._last_t = t4
                self._ema("prep", (t1 - t0) * 1000)
                self._ema("detect", (t2 - t1) * 1000)
                self._ema("track", (t3 - t2) * 1000)
                self._ema("draw", (t4 - t3) * 1000)
                self._ema("total", (t4 - t0) * 1000)
            return frame
