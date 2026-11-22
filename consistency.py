"""Temporal class consistency.

DeepSORT associates detections to tracks by position/appearance only and overwrites a
track's class with whatever the latest matched detection said. A single bad frame
("motorcycle" for one frame on a stack of books) therefore relabels the whole track.

ClassVoter keeps an exponentially-decayed, confidence-weighted vote per track:

  * every observation adds its confidence to its class and decays all other votes;
  * the displayed class only switches when a challenger's score exceeds the current
    class's score by `margin` (hysteresis).

With the defaults (decay 0.8, margin 1.25) a 1-3 frame flip can never change the label,
while a genuine change (a different object now stays in that place) takes over after
~4 consecutive frames. Classes are never invented: the label is always a class the model
actually predicted for this track.
"""


class ClassVoter:
    __slots__ = ("decay", "margin", "scores", "conf", "label", "switches", "suppressed")

    def __init__(self, decay=0.8, margin=1.25):
        self.decay, self.margin = decay, margin
        self.scores: dict[int, float] = {}
        self.conf: dict[int, float] = {}      # smoothed model confidence per class
        self.label = None
        self.switches = 0                      # genuine label changes
        self.suppressed = 0                    # frames where the raw class disagreed but the label held

    def observe(self, cls: int, conf: float) -> int:
        for k in list(self.scores):
            self.scores[k] *= self.decay
            if self.scores[k] < 1e-3:
                del self.scores[k]
        self.scores[cls] = self.scores.get(cls, 0.0) + conf
        self.conf[cls] = conf if cls not in self.conf else 0.6 * self.conf[cls] + 0.4 * conf

        if self.label is None:
            self.label = cls
        elif cls != self.label:
            best = max(self.scores, key=self.scores.get)
            if best != self.label and self.scores[best] > self.margin * self.scores.get(self.label, 0.0):
                self.label = best
                self.switches += 1
            else:
                self.suppressed += 1
        return self.label

    @property
    def confidence(self) -> float:
        """Smoothed confidence of the DISPLAYED class (never the confidence of a different class)."""
        return self.conf.get(self.label, 0.0)


class ClassConsistency:
    def __init__(self, decay=0.8, margin=1.25):
        self.decay, self.margin = decay, margin
        self.voters: dict[int, ClassVoter] = {}
        self._retired_switches = self._retired_suppressed = 0

    def observe(self, track_id: int, cls: int, conf: float) -> ClassVoter:
        v = self.voters.get(track_id)
        if v is None:
            v = self.voters[track_id] = ClassVoter(self.decay, self.margin)
        v.observe(cls, conf)
        return v

    def prune(self, alive_ids):
        for tid in [t for t in self.voters if t not in alive_ids]:
            v = self.voters.pop(tid)
            self._retired_switches += v.switches
            self._retired_suppressed += v.suppressed

    @property
    def switches(self): return self._retired_switches + sum(v.switches for v in self.voters.values())
    @property
    def suppressed(self): return self._retired_suppressed + sum(v.suppressed for v in self.voters.values())
