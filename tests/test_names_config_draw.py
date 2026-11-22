import os
import unittest
from unittest import mock

import numpy as np

from app.config import load_config
from app.detection.detector import validate_names
from app.detection.draw import class_color, draw_tracks


class ValidateNamesTests(unittest.TestCase):
    def test_accepts_dict_and_list(self):
        self.assertEqual(validate_names({0: "person", 1: "book"}, nc=2), {0: "person", 1: "book"})
        self.assertEqual(validate_names(["person", "book"]), {0: "person", 1: "book"})

    def test_rejects_bad_tables(self):
        for bad, nc in (({}, None), ({0: "a", 2: "b"}, None), ({0: "a", 1: " "}, None), ({0: "a", 1: "b"}, 3)):
            with self.assertRaises(ValueError):
                validate_names(bad, nc)


class ConfigTests(unittest.TestCase):
    def test_defaults_match_requirements(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            c = load_config()
        self.assertEqual((c.default_model, c.conf, c.iou, c.imgsz), ("yolo11s.pt", 0.40, 0.50, 0))

    def test_values_clamped_and_invalid_rejected(self):
        with mock.patch.dict(os.environ, {"CONF": "5", "IOU": "0.01"}, clear=True):
            c = load_config()
        self.assertEqual((c.conf, c.iou), (0.95, 0.30))
        for bad in ("abc", "1000", "950"):
            with mock.patch.dict(os.environ, {"IMGSZ": bad}, clear=True), self.assertRaises(SystemExit):
                load_config()
        with mock.patch.dict(os.environ, {"IMGSZ": "960"}, clear=True):
            self.assertEqual(load_config().imgsz, 960)


class DrawTests(unittest.TestCase):
    def item(self, box, conf=0.8):
        return dict(box=box, conf=conf, color=class_color(1), label="BOOK | ID: 01 | 80%")

    def test_no_items_leaves_frame_untouched(self):
        f = np.full((480, 640, 3), 90, np.uint8); before = f.copy()
        draw_tracks(f, [])
        self.assertTrue((f == before).all())

    def test_draws_pixels_and_survives_edge_boxes(self):
        for box in ((100, 100, 300, 300), (0, 0, 80, 60), (560, 420, 700, 600), (-30, -30, 50, 50), (300, 200, 301, 201)):
            f = np.full((480, 640, 3), 90, np.uint8)
            draw_tracks(f, [self.item(box)])
            self.assertFalse((f == 90).all(), f"nothing drawn for {box}")

    def test_label_chip_stays_inside_frame_at_top_edge(self):
        f = np.zeros((480, 640, 3), np.uint8)
        draw_tracks(f, [dict(box=(10, 0, 200, 200), conf=1.0, color=(0, 255, 0), label="BOOK | ID: 01 | 100%")])
        self.assertGreater(f[:30, 10:200].sum(), 0)               # chip rendered inside, not clipped away

    def test_class_colors_stable_and_distinct(self):
        self.assertEqual(class_color(3), class_color(3))
        self.assertEqual(len({class_color(i) for i in range(80)}), 80)


if __name__ == "__main__":
    unittest.main()
