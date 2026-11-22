"""End-to-end tests of the Engine (validate -> real DeepSORT -> class vote -> draw).

The YOLO model is replaced by ScriptedDetector, a TEST DOUBLE that replays a list of raw
detections per frame. These tests therefore verify the filtering / tracking / voting / drawing
logic only; they say nothing about YOLO's accuracy (use evaluation/benchmark.py for that).
Requires deep-sort-realtime (embedder disabled, so no PyTorch needed).
"""
import math
import unittest
from types import SimpleNamespace

import numpy as np

from app.config import Config
from app.detection.filters import Det
from app.detection.processor import Engine
from app.detection.tracker import Tracker

NAMES = {0: "person", 1: "book", 2: "motorcycle", 3: "chair"}
BOX = (200, 150, 420, 330)


def make_active(name="yolo11s.pt", names=NAMES):
    return SimpleNamespace(name=name, names=names, auto_imgsz=960, model=None, device="cpu", latency_ms=None)


class ScriptedDetector:
    def __init__(self, script):
        self.script, self.i, self.calls, self.last_class_ids = script, 0, 0, "unset"
        self._active, self.generation = make_active(), 1
        self.name = self._active.name

    @property
    def active(self): return self._active

    def detect(self, frame, conf, iou, imgsz, class_ids, active=None):
        self.calls += 1
        self.last_class_ids = class_ids
        dets = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return list(dets)


def run(script, **engine_kw):
    det = ScriptedDetector(script)
    eng = Engine(det, Config(), tracker_factory=lambda: Tracker(embedder=None))
    for k, v in engine_kw.items():
        setattr(eng, k, v)
    eng.reset()
    return det, eng


def frame():
    return np.full((480, 640, 3), 60, np.uint8)


def step(eng):
    out = eng.process(frame())
    return out, list(eng.last_tracks)


class EngineTests(unittest.TestCase):
    def test_transient_motorcycle_flip_on_a_book_is_not_displayed(self):
        script = [[Det(*BOX, 0.80, 1)]] * 8 + [[Det(*BOX, 0.92, 2)]] + [[Det(*BOX, 0.80, 1)]] * 6
        det, eng = run(script)
        shown, raw = [], []
        for _ in script:
            _, tr = step(eng)
            shown += [t["cls"] for t in tr]; raw += [t["raw_cls"] for t in tr]
        self.assertIn("motorcycle", raw, "scenario must actually contain the raw flip")
        self.assertEqual(set(shown), {"book"})
        self.assertGreaterEqual(eng.stats()["class_flips_suppressed"], 1)
        self.assertEqual(eng.stats()["class_switches"], 0)

    def test_genuine_class_change_is_followed(self):
        script = [[Det(*BOX, 0.80, 1)]] * 10 + [[Det(*BOX, 0.80, 2)]] * 12
        det, eng = run(script)
        shown = []
        for _ in script:
            _, tr = step(eng)
            shown.append(tr[0]["cls"] if tr else None)
        self.assertEqual(shown[9], "book")
        self.assertEqual(shown[-1], "motorcycle")
        self.assertLessEqual(shown.index("motorcycle") - 10, 5)
        self.assertEqual(eng.stats()["class_switches"], 1)

    def test_low_confidence_and_invalid_detections_never_displayed(self):
        bad = [Det(*BOX, 0.30, 1), Det(math.nan, 0, 50, 50, 0.9, 1), Det(10, 10, 80, 80, 0.9, 77)]
        det, eng = run([bad] * 10)
        for _ in range(10):
            out, tr = step(eng)
            self.assertEqual(tr, [])
        s = eng.stats()
        self.assertEqual((s["total_detections"], s["tracked_now"]), (0, 0))
        self.assertEqual(s["filtered_low_conf"], 10)
        self.assertEqual(s["filtered_invalid"], 20)

    def test_frame_is_untouched_when_nothing_is_detected(self):
        det, eng = run([[]] * 5)
        for _ in range(5):
            out, tr = step(eng)
            self.assertTrue((out == 60).all(), "pixels were drawn although nothing was detected")

    def test_detection_is_drawn_once_confirmed(self):
        det, eng = run([[Det(*BOX, 0.85, 1)]] * 6)
        outs = [step(eng)[0] for _ in range(6)]
        self.assertTrue((outs[0] == 60).all())                    # tentative track: not shown yet
        self.assertFalse((outs[-1] == 60).all())

    def test_cross_class_duplicate_shows_only_the_more_confident_class(self):
        det, eng = run([[Det(*BOX, 0.62, 1), Det(*BOX, 0.47, 2)]] * 6)
        for _ in range(6):
            _, tr = step(eng)
        self.assertEqual([t["cls"] for t in tr], ["book"])
        self.assertEqual(eng.stats()["filtered_duplicates"], 6)

    def test_displayed_confidence_never_exceeds_what_the_model_reported(self):
        det, eng = run([[Det(*BOX, 0.55, 1)]] * 8 + [[Det(*BOX, 0.95, 2)]])
        for _ in range(9):
            _, tr = step(eng)
        self.assertAlmostEqual(tr[0]["conf"], 0.55, places=6)    # not the 0.95 of the suppressed frame

    def test_raising_threshold_hides_existing_tracks(self):
        det, eng = run([[Det(*BOX, 0.50, 1)]] * 12)
        for _ in range(6):
            _, tr = step(eng)
        self.assertEqual(len(tr), 1)
        eng.conf = 0.60
        _, tr = step(eng)
        self.assertEqual(tr, [])

    def test_class_filter_is_mapped_by_name_and_unknown_name_yields_nothing(self):
        det, eng = run([[Det(*BOX, 0.85, 1)]] * 4)
        eng.class_names = ["book", "person"]
        step(eng)
        self.assertEqual(sorted(det.last_class_ids), [0, 1])
        calls = det.calls
        eng.class_names = ["unicorn"]
        _, tr = step(eng)
        self.assertEqual((det.calls, tr), (calls, []))            # detector not even called; nothing shown

    def test_model_switch_resets_tracks(self):
        det, eng = run([[Det(*BOX, 0.85, 1)]] * 12)
        for _ in range(5):
            step(eng)
        first_ids = {t["id"] for t in eng.last_tracks}
        det._active = make_active("yolo11m.pt"); det.generation += 1
        for _ in range(5):
            step(eng)
        self.assertTrue(first_ids)
        self.assertEqual({t["id"] for t in eng.last_tracks}, {1})  # IDs restarted from a clean tracker

    def test_history_follows_genuine_changes(self):
        script = [[Det(*BOX, 0.8, 1)]] * 10 + [[Det(*BOX, 0.8, 2)]] * 12
        det, eng = run(script)
        for _ in script:
            step(eng)
        self.assertEqual([h["cls"] for h in eng.history], ["motorcycle"])

    def test_two_separate_objects_get_two_ids(self):
        a, b = (50, 60, 200, 220), (380, 240, 600, 450)
        det, eng = run([[Det(*a, 0.9, 1), Det(*b, 0.8, 3)]] * 6)
        for _ in range(6):
            _, tr = step(eng)
        self.assertEqual(sorted(t["cls"] for t in tr), ["book", "chair"])
        self.assertEqual(len({t["id"] for t in tr}), 2)


if __name__ == "__main__":
    unittest.main()
