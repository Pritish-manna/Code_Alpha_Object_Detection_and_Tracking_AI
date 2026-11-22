import math
import unittest

import numpy as np

from app.detection.filters import Det, clean_detections, prepare_frame

NAMES = {0: "person", 1: "book", 2: "motorcycle"}
W, H = 640, 480


def clean(dets, conf=0.40, **kw):
    return clean_detections(dets, W, H, NAMES, conf, **kw)


class PrepareFrameTests(unittest.TestCase):
    def test_gray_and_bgra_become_bgr(self):
        self.assertEqual(prepare_frame(np.zeros((100, 100), np.uint8)).shape, (100, 100, 3))
        self.assertEqual(prepare_frame(np.zeros((100, 100, 4), np.uint8)).shape, (100, 100, 3))

    def test_large_frame_downscaled_keeping_aspect(self):
        out = prepare_frame(np.zeros((2160, 3840, 3), np.uint8), 1280)
        self.assertEqual(out.shape, (720, 1280, 3))

    def test_small_frame_not_resized_and_channels_untouched(self):
        f = np.zeros((480, 640, 3), np.uint8); f[..., 0] = 200   # blue channel only
        out = prepare_frame(f)
        self.assertEqual(out.shape, f.shape)
        self.assertTrue((out[..., 0] == 200).all() and (out[..., 1:] == 0).all())   # no BGR/RGB swap

    def test_rejects_bad_input(self):
        for bad in (None, "x", np.zeros((10, 10, 3), np.uint8), np.zeros((100, 100, 3), np.float32), np.zeros((100, 100, 2), np.uint8)):
            with self.assertRaises(ValueError):
                prepare_frame(bad)


class CleanDetectionTests(unittest.TestCase):
    def test_valid_detection_kept_unchanged(self):
        kept, dropped = clean([Det(100, 100, 300, 300, 0.9, 1)])
        self.assertEqual(kept, [Det(100, 100, 300, 300, 0.9, 1)])
        self.assertFalse(dropped)

    def test_non_finite_and_unknown_class_dropped(self):
        dets = [Det(math.nan, 0, 100, 100, 0.9, 1), Det(0, 0, math.inf, 100, 0.9, 1), Det(0, 0, 100, 100, math.nan, 1),
                Det(0, 0, 100, 100, 0.9, 99), Det(0, 0, 100, 100, 0.9, -1)]
        kept, dropped = clean(dets)
        self.assertEqual(kept, [])
        self.assertEqual(dropped["invalid"], 5)

    def test_confidence_threshold_is_inclusive_and_enforced(self):
        kept, dropped = clean([Det(0, 0, 100, 100, 0.39, 1), Det(200, 0, 300, 100, 0.40, 1)])
        self.assertEqual([d.conf for d in kept], [0.40])
        self.assertEqual(dropped["low_conf"], 1)

    def test_boxes_are_clipped_to_frame_and_corners_normalised(self):
        kept, _ = clean([Det(-50, -20, 700, 500, 0.9, 1), Det(300, 300, 200, 200, 0.9, 2)])
        a, b = sorted(kept, key=lambda d: d.cls)
        self.assertEqual((a.x1, a.y1, a.x2, a.y2), (0, 0, W, H))
        self.assertEqual((b.x1, b.y1, b.x2, b.y2), (200, 200, 300, 300))

    def test_empty_and_tiny_boxes_dropped(self):
        kept, dropped = clean([Det(100, 100, 100, 200, 0.9, 1), Det(700, 10, 800, 90, 0.9, 1), Det(10, 10, 15, 200, 0.9, 1)])
        self.assertEqual(kept, [])
        self.assertEqual(dropped["invalid"] + dropped["too_small"], 3)

    def test_cross_class_duplicate_keeps_more_confident_class(self):
        box = (100, 100, 400, 300)
        kept, dropped = clean([Det(*box, 0.45, 2), Det(*box, 0.62, 1)])
        self.assertEqual([(d.cls, d.conf) for d in kept], [(1, 0.62)])
        self.assertEqual(dropped["duplicate_class"], 1)

    def test_partially_overlapping_different_classes_both_kept(self):
        # e.g. a person sitting on a chair: overlap is high but far from identical
        kept, _ = clean([Det(100, 100, 300, 400, 0.9, 0), Det(150, 200, 330, 420, 0.8, 2)])
        self.assertEqual(len(kept), 2)

    def test_same_class_boxes_are_left_to_nms(self):
        box = (100, 100, 400, 300)
        kept, _ = clean([Det(*box, 0.9, 1), Det(*box, 0.8, 1)])
        self.assertEqual(len(kept), 2)


if __name__ == "__main__":
    unittest.main()
