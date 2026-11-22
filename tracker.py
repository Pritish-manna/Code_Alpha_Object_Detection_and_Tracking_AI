import inspect
import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("tracker")


@dataclass(frozen=True)
class TrackObs:
    track_id: int
    box: tuple          # x1, y1, x2, y2 (float): the matched detection's own box this frame
    cls: int            # class of the detection matched THIS frame (raw, before temporal voting)
    conf: float         # confidence of that detection


class Tracker:
    """Thin wrapper around DeepSORT.

    embedder="mobilenet" (default) uses appearance + motion. embedder=None tracks by motion/IoU
    only, which needs no PyTorch embedder and is handy for tests or very weak machines.

    Only confirmed tracks that were matched to a real detection in the current frame are
    returned: a track that is merely being coasted by the Kalman filter is never drawn,
    so nothing is displayed that the detector did not actually see in this frame.
    """

    def __init__(self, embedder="mobilenet", max_age=30, n_init=3, max_iou_distance=0.7,
                 max_cosine_distance=0.3, nn_budget=100):
        from deep_sort_realtime.deepsort_tracker import DeepSort
        self._appearance = embedder is not None
        wanted = dict(max_age=max_age, n_init=n_init, max_iou_distance=max_iou_distance,
                      max_cosine_distance=max_cosine_distance, nn_budget=nn_budget, embedder=embedder)
        # Pass only what this installed deep-sort-realtime version accepts, so a version difference
        # can never stop tracking from starting.
        accepted = set(inspect.signature(DeepSort.__init__).parameters)
        kwargs = {k: v for k, v in wanted.items() if k in accepted}
        for k in set(wanted) - set(kwargs):
            log.warning("deep-sort-realtime does not support %r; using its default", k)
        self.t = DeepSort(**kwargs)

    def reset(self):
        self.t.delete_all_tracks()   # also restarts IDs at 1, without reloading the embedder

    def update(self, dets, frame):
        raw = [([d.x1, d.y1, d.w, d.h], d.conf, d.cls) for d in dets]
        embeds = None
        if not self._appearance:
            embeds = [np.ones(8, dtype=np.float32) / np.sqrt(8)] * len(raw)
        tracks = self.t.update_tracks(raw, frame=frame, embeds=embeds)
        alive = {int(t.track_id) for t in tracks}
        out = []
        for t in tracks:
            if not t.is_confirmed() or t.time_since_update != 0:
                continue
            box = t.to_ltrb(orig=True, orig_strict=True)
            cls, conf = t.get_det_class(), t.get_det_conf()
            if box is None or cls is None or conf is None:
                continue
            out.append(TrackObs(int(t.track_id), tuple(float(v) for v in box), int(cls), float(conf)))
        return out, alive
