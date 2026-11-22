import random
import unittest

from app.detection.consistency import ClassConsistency, ClassVoter

BOOK, MOTO = 1, 2


def feed(v, seq):
    return [v.observe(c, conf) for c, conf in seq]


class ClassVoterTests(unittest.TestCase):
    def test_first_observation_sets_label(self):
        v = ClassVoter(); v.observe(BOOK, 0.7)
        self.assertEqual(v.label, BOOK)

    def test_short_flips_never_change_label(self):
        for flip_len in (1, 2, 3):
            v = ClassVoter()
            feed(v, [(BOOK, 0.8)] * 10)
            labels = feed(v, [(MOTO, 0.9)] * flip_len)           # even MORE confident than the book frames
            self.assertEqual(set(labels), {BOOK}, f"flip of {flip_len} frames changed the label")
            self.assertEqual(v.suppressed, flip_len)

    def test_sustained_change_is_followed_quickly(self):
        v = ClassVoter()
        feed(v, [(BOOK, 0.8)] * 10)
        labels = feed(v, [(MOTO, 0.8)] * 8)
        first = labels.index(MOTO) + 1
        self.assertLessEqual(first, 5, "genuine change should take over within ~5 frames")
        self.assertEqual(labels[-1], MOTO)
        self.assertEqual(v.switches, 1)

    def test_alternating_equal_confidence_does_not_flicker(self):
        v = ClassVoter()
        labels = feed(v, [(BOOK if i % 2 == 0 else MOTO, 0.6) for i in range(40)])
        self.assertEqual(set(labels), {BOOK})
        self.assertEqual(v.switches, 0)

    def test_confidence_belongs_to_displayed_class(self):
        v = ClassVoter()
        feed(v, [(BOOK, 0.55)] * 6)
        v.observe(MOTO, 0.95)
        self.assertEqual(v.label, BOOK)
        self.assertAlmostEqual(v.confidence, 0.55, places=6)      # NOT 0.95 from the suppressed class

    def test_label_is_always_a_class_that_was_observed(self):
        rng = random.Random(7)
        for _ in range(200):
            v, seen = ClassVoter(), set()
            for _ in range(60):
                c = rng.choice([0, 1, 2, 5, 9]); seen.add(c)
                self.assertIn(v.observe(c, rng.uniform(0.4, 1.0)), seen)


class ClassConsistencyTests(unittest.TestCase):
    def test_tracks_are_independent_and_pruned_counts_survive(self):
        cc = ClassConsistency()
        for _ in range(10):
            cc.observe(1, BOOK, 0.8); cc.observe(2, MOTO, 0.8)
        cc.observe(1, MOTO, 0.9)                                   # one suppressed flip on track 1
        self.assertEqual((cc.voters[1].label, cc.voters[2].label), (BOOK, MOTO))
        cc.prune({2})
        self.assertNotIn(1, cc.voters)
        self.assertEqual(cc.suppressed, 1)


if __name__ == "__main__":
    unittest.main()
