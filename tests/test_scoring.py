import unittest
from app.detection.filters import Det
from evaluation.scoring import aggregate, score_image

N = {0: "person", 1: "book", 2: "motorcycle"}
d = lambda c: Det(0, 0, 50, 50, .9, c)


class ScoringTests(unittest.TestCase):
    def test_hit_miss_false_and_confusion(self):
        self.assertEqual(score_image({"book"}, set(), [d(1)], N)["hit"], True)
        r = score_image({"book"}, set(), [d(2)], N)
        self.assertTrue(r["missed"]); self.assertEqual((r["false"], r["confused_as"]), (["motorcycle"], ["motorcycle"]))
        self.assertEqual(score_image({"book"}, {"person"}, [d(1), d(0)], N)["false"], [])      # tolerated class
        self.assertEqual(score_image({"book"}, set(), [d(1), d(0)], N)["false"], ["person"])
        n = score_image(set(), set(), [d(2)], N)                                              # negative image
        self.assertEqual((n["hit"], n["missed"], n["false"]), (None, False, ["motorcycle"]))

    def test_aggregate(self):
        rows = [dict(score_image({"book"}, set(), k, N), object="books") for k in ([d(1)], [d(2)], [], [d(1), d(2)])]
        a = aggregate(rows, "object")["books"]
        self.assertEqual((a["images"], a["recall"], a["missed"], a["false_dets"]), (4, 0.5, 2, 2))
        self.assertEqual(a["confused_as"], {"motorcycle": 1})


if __name__ == "__main__":
    unittest.main()
