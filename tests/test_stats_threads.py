"""Statistics correctness (FPS, pause handling, class distribution, empty sessions) and thread safety."""
import threading
import time
import unittest

import numpy as np

from app.config import Config
from app.detection.filters import Det
from app.detection.processor import Engine
from app.detection.tracker import Tracker
from tests.helpers import BOX, NAMES, ScriptedDetector


def engine(dets=None, delay=0.0, **cfg):
    det = ScriptedDetector(dets, delay)
    return det, Engine(det, Config(**cfg), tracker_factory=lambda: Tracker(embedder=None))


def frame(): return np.full((240, 320, 3), 60, np.uint8)


class StatsTests(unittest.TestCase):
    def test_empty_session_is_all_zeros_and_reports_no_data(self):
        _, eng = engine(); eng.reset()
        a = eng.analytics()
        self.assertFalse(a["has_data"])
        s = a["stats"]
        for k in ("total_detections", "tracked_now", "unique_ids", "fps", "avg_fps", "frames", "avg_confidence"):
            self.assertEqual(s[k], 0, k)
        self.assertEqual(a["classes"], [])

    def test_class_distribution_matches_totals(self):
        a, b = (10, 10, 100, 100), (150, 120, 300, 230)
        _, eng = engine([Det(*a, 0.9, 1), Det(*b, 0.8, 3)]); eng.reset()
        for _ in range(8):
            eng.process(frame())
        an = eng.analytics()
        rows = {r["cls"]: r for r in an["classes"]}
        self.assertEqual(set(rows), {"book", "chair"})
        self.assertEqual(sum(r["detections"] for r in an["classes"]), an["stats"]["total_detections"])
        self.assertAlmostEqual(sum(r["share_pct"] for r in an["classes"]), 100.0, delta=0.2)
        self.assertEqual(an["stats"]["unique_ids"], 2)
        self.assertEqual(an["stats"]["tracked_now"], 2)
        self.assertTrue(an["has_data"])
        self.assertAlmostEqual(rows["book"]["avg_confidence"], 90.0, delta=0.1)

    def test_average_fps_counts_active_time_only(self):
        _, eng = engine(delay=0.02); eng.reset()
        for _ in range(10):
            eng.process(frame())
        before = eng.stats()["avg_fps"]
        self.assertTrue(20 < before < 60, before)               # ~50 fps worth of 20 ms work + overhead
        eng.pause(); time.sleep(0.6); eng.resume()
        for _ in range(10):
            eng.process(frame())
        after = eng.stats()["avg_fps"]
        self.assertAlmostEqual(after, before, delta=before * 0.35, msg="a pause must not drag the average FPS down")
        self.assertGreater(eng.stats()["fps"], 20)               # current FPS recovers immediately after resume

    def test_stats_timing_breakdown_present(self):
        _, eng = engine(); eng.reset()
        for _ in range(3):
            eng.process(frame())
        t = eng.stats()["timing_ms"]
        for k in ("prep", "detect", "track", "draw", "total"):
            self.assertIn(k, t)

    def test_reset_clears_everything_but_keeps_history_log(self):
        _, eng = engine(); eng.reset()
        for _ in range(6):
            eng.process(frame())
        self.assertGreater(eng.stats()["total_detections"], 0)
        eng.reset()
        s = eng.stats()
        self.assertEqual((s["total_detections"], s["unique_ids"], s["frames"], s["fps"]), (0, 0, 0, 0))
        self.assertEqual(eng.class_summary(), [])

    def test_frames_are_downscaled_to_the_inference_size_not_beyond(self):
        _, eng = engine(); eng.reset()
        big = np.full((720, 1280, 3), 60, np.uint8)
        eng.imgsz = 416
        self.assertEqual(eng.process(big).shape[1], 640)         # floor of 640 keeps the preview sharp
        eng.imgsz = 960
        self.assertEqual(eng.process(big).shape[1], 960)
        eng.imgsz = 0                                            # auto -> 960 for the scripted model
        self.assertEqual(eng.process(big).shape[1], 960)


class ThreadSafetyTests(unittest.TestCase):
    def test_polling_stats_and_analytics_while_frames_are_processed(self):
        """Regression: analytics iterated dicts that the worker thread was mutating ('dictionary changed size')."""
        _, eng = engine([Det(10, 10, 100, 100, 0.9, 1), Det(150, 120, 300, 230, 0.8, 3)]); eng.reset()
        errors, stop = [], threading.Event()

        def poll():
            while not stop.is_set():
                try:
                    eng.analytics(); eng.stats(); eng.history_snapshot(); eng.class_summary()
                except Exception as e:  # noqa: BLE001
                    errors.append(repr(e)); return

        threads = [threading.Thread(target=poll) for _ in range(4)]
        [t.start() for t in threads]
        for i in range(300):
            eng.process(frame())
            if i % 50 == 49:
                eng.reset()
        stop.set(); [t.join() for t in threads]
        self.assertEqual(errors, [])

    def test_reset_waits_for_a_frame_in_progress(self):
        _, eng = engine(delay=0.3); eng.reset()
        t = threading.Thread(target=lambda: eng.process(frame())); t.start()
        time.sleep(0.05)
        t0 = time.perf_counter(); eng.reset(); waited = time.perf_counter() - t0
        t.join()
        self.assertGreater(waited, 0.15)
        self.assertEqual(eng.stats()["frames"], 0, "the frame finishing after reset must not leak into the new session")


if __name__ == "__main__":
    unittest.main()
