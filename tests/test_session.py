"""Session/WebSocket behaviour: start, stop, pause, resume, reset, races, cleanup, webcam frame handling.
Uses real video files, real OpenCV decode/encode, the real Engine and the real DeepSORT wrapper; only YOLO is scripted."""
import asyncio
import os
import unittest

import cv2
import numpy as np

from tests.helpers import (FakeTransport, ScriptedDetector, build, jpeg, make_video, tmpdir, wait_for)
from app.session import Connection


class Harness:
    def __init__(self, tc, frames=30, delay=0.0, **cfg_kw):
        self.tmp = tmpdir(); tc.addCleanup(self.tmp.cleanup)
        self.det, self.eng, self.ctl, self.cfg = build(self.tmp.name, ScriptedDetector(delay=delay), **cfg_kw)
        self.video = "clip.avi"
        make_video(os.path.join(self.tmp.name, self.video), frames)
        self.t = FakeTransport()
        self.conn = Connection(self.ctl, self.t)
        self.serving = None

    async def __aenter__(self):
        self.serving = asyncio.create_task(self.conn.serve())
        await wait_for(lambda: self.t.msgs("state"))
        return self

    async def __aexit__(self, *exc):
        self.t.client_disconnect()
        await asyncio.wait_for(self.serving, 10)

    def states(self): return [m["state"] for m in self.t.msgs("state")]


class VideoSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_video_is_processed_and_reports_done(self):
        async with Harness(self, frames=30) as h:
            h.t.client_text(action="start", mode="video", file_id=h.video)
            self.assertTrue(await wait_for(lambda: h.t.msgs("done")))
            done = h.t.msgs("done")[0]
            self.assertEqual(done["stats"]["frames"], 30)
            self.assertEqual(len(h.t.msgs("frame")), 30)
            self.assertEqual(sum(1 for k, _ in h.t.sent if k == "bytes"), 30)
            self.assertEqual(h.t.msgs("frame")[-1]["progress"], 100.0)
            # every binary message is a decodable JPEG following its frame message
            for (k1, a), (k2, b) in zip(h.t.sent, h.t.sent[1:]):
                if k1 == "json" and a.get("type") == "frame":
                    self.assertEqual(k2, "bytes")
                    self.assertIsNotNone(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR))
            self.assertEqual(h.ctl.state, "idle")
            self.assertFalse(h.eng.running)
            self.assertIsNone(h.ctl.task)
            self.assertEqual(h.states()[-1], "idle")

    async def test_second_start_is_rejected_and_first_keeps_running(self):
        async with Harness(self, frames=60, delay=0.02) as h:
            h.t.client_text(action="start", mode="video", file_id=h.video)
            await wait_for(lambda: h.t.msgs("frame"))
            h.t.client_text(action="start", mode="video", file_id=h.video)
            self.assertTrue(await wait_for(lambda: h.t.msgs("error")))
            self.assertEqual(h.t.msgs("error")[0]["code"], 409)
            self.assertEqual(h.ctl.state, "running")
            self.assertTrue(await wait_for(lambda: h.t.msgs("done")))
            self.assertEqual(h.t.msgs("done")[0]["stats"]["frames"], 60)

    async def test_stop_then_immediate_start_is_not_killed_by_the_old_task(self):
        """Regression: the old task's cleanup used to switch off a session started right after Stop."""
        async with Harness(self, frames=80, delay=0.02) as h:
            h.t.client_text(action="start", mode="video", file_id=h.video)
            await wait_for(lambda: len(h.t.msgs("frame")) >= 3)
            h.t.client_text(action="stop")
            h.t.client_text(action="start", mode="video", file_id=h.video)
            self.assertTrue(await wait_for(lambda: h.t.msgs("done"), timeout=30))
            self.assertEqual([m["code"] for m in h.t.msgs("error")], [])
            self.assertEqual(h.t.msgs("done")[0]["stats"]["frames"], 80, "second run must process the whole video")

    async def test_pause_stops_frames_and_resume_continues(self):
        async with Harness(self, frames=200, delay=0.01) as h:
            h.t.client_text(action="start", mode="video", file_id=h.video)
            await wait_for(lambda: len(h.t.msgs("frame")) >= 5)
            h.t.client_text(action="pause")
            await wait_for(lambda: h.ctl.state == "paused")
            await asyncio.sleep(0.3)                       # let the frame in flight drain
            n = len(h.t.msgs("frame")); await asyncio.sleep(0.4)
            self.assertEqual(len(h.t.msgs("frame")), n, "frames kept flowing while paused")
            self.assertTrue(h.eng.paused)
            h.t.client_text(action="resume")
            self.assertTrue(await wait_for(lambda: len(h.t.msgs("frame")) > n + 3))
            self.assertFalse(h.eng.paused)
            h.t.client_text(action="stop")
            await wait_for(lambda: h.ctl.state == "idle")

    async def test_stop_releases_everything_and_keeps_stats(self):
        async with Harness(self, frames=200, delay=0.01) as h:
            h.t.client_text(action="start", mode="video", file_id=h.video)
            await wait_for(lambda: len(h.t.msgs("frame")) >= 5)
            h.t.client_text(action="stop")
            self.assertTrue(await wait_for(lambda: [a for a in h.t.msgs("ack") if a["action"] == "stop"]))
            ack = [a for a in h.t.msgs("ack") if a["action"] == "stop"][0]
            self.assertEqual(ack["state"], "idle")
            self.assertGreaterEqual(ack["stats"]["frames"], 5)
            self.assertIsNone(h.ctl.task)
            self.assertEqual(h.t.msgs("done"), [], "a stopped video must not be reported as finished")
            n = len(h.t.msgs("frame")); await asyncio.sleep(0.3)
            self.assertEqual(len(h.t.msgs("frame")), n)

    async def test_reset_while_running_stops_and_clears(self):
        async with Harness(self, frames=200, delay=0.01) as h:
            h.t.client_text(action="start", mode="video", file_id=h.video)
            await wait_for(lambda: len(h.t.msgs("frame")) >= 5)
            h.t.client_text(action="reset")
            self.assertTrue(await wait_for(lambda: [a for a in h.t.msgs("ack") if a["action"] == "reset"]))
            ack = [a for a in h.t.msgs("ack") if a["action"] == "reset"][0]
            self.assertEqual((ack["state"], ack["stats"]["frames"], ack["stats"]["unique_ids"]), ("idle", 0, 0))
            self.assertIsNone(h.ctl.task)
            self.assertFalse(h.eng.analytics()["has_data"])

    async def test_frame_stride_skips_and_counts(self):
        async with Harness(self, frames=30, frame_stride=3) as h:
            h.t.client_text(action="start", mode="video", file_id=h.video)
            self.assertTrue(await wait_for(lambda: h.t.msgs("done")))
            s = h.t.msgs("done")[0]["stats"]
            self.assertEqual(s["frames"], 10)
            self.assertEqual(s["frames_skipped"], 20)
            self.assertEqual(h.t.msgs("frame")[-1]["progress"], 100.0)

    async def test_bad_requests_get_clear_errors(self):
        async with Harness(self) as h:
            h.t.client_text(action="start", mode="video", file_id="nope.avi")
            h.t.client_text(action="start", mode="video", file_id="../../etc/passwd")
            h.t.client_text(action="start", mode="banana")
            h.t.client_text(action="pause")
            h.t.client_text(action="explode")
            await wait_for(lambda: len(h.t.msgs("error")) >= 5)
            self.assertEqual([m["code"] for m in h.t.msgs("error")], [404, 404, 400, 409, 400])
            self.assertEqual(h.ctl.state, "idle")
            self.assertFalse(h.eng.running)

    async def test_model_not_loaded_blocks_start(self):
        async with Harness(self) as h:
            h.det.status = "loading"
            h.t.client_text(action="start", mode="webcam")
            await wait_for(lambda: h.t.msgs("error"))
            self.assertEqual(h.t.msgs("error")[0]["code"], 503)

    async def test_disconnect_stops_the_session_it_owned(self):
        h = Harness(self, frames=300, delay=0.01)
        await h.__aenter__()
        h.t.client_text(action="start", mode="video", file_id=h.video)
        await wait_for(lambda: len(h.t.msgs("frame")) >= 3)
        h.t.client_disconnect()
        await asyncio.wait_for(h.serving, 10)
        self.assertEqual(h.ctl.state, "idle")
        self.assertFalse(h.eng.running)
        self.assertIsNone(h.ctl.task)


class WebcamTests(unittest.IsolatedAsyncioTestCase):
    async def test_webcam_frames_are_processed_and_answered(self):
        async with Harness(self) as h:
            h.t.client_text(action="start", mode="webcam")
            await wait_for(lambda: [a for a in h.t.msgs("ack") if a["action"] == "start"])
            for _ in range(8):
                before = len(h.t.msgs("frame"))
                h.t.client_bytes(jpeg())
                self.assertTrue(await wait_for(lambda: len(h.t.msgs("frame")) > before))
            self.assertEqual(h.t.msgs("frame")[-1]["stats"]["frames"], 8)
            self.assertGreaterEqual(h.t.msgs("frame")[-1]["stats"]["tracked_now"], 1)

    async def test_frames_outside_a_session_or_while_paused_get_skip(self):
        async with Harness(self) as h:
            h.t.client_bytes(jpeg())
            await wait_for(lambda: h.t.msgs("skip"))
            self.assertEqual((len(h.t.msgs("skip")), len(h.t.msgs("frame"))), (1, 0))
            h.t.client_text(action="start", mode="webcam"); h.t.client_text(action="pause")
            await wait_for(lambda: h.ctl.state == "paused")
            h.t.client_bytes(jpeg())
            await wait_for(lambda: len(h.t.msgs("skip")) == 2)
            self.assertEqual(h.eng.frames, 0)

    async def test_bad_frame_reports_frame_error_and_session_survives(self):
        async with Harness(self) as h:
            h.t.client_text(action="start", mode="webcam")
            await wait_for(lambda: h.ctl.state == "running")
            h.t.client_bytes(b"not a jpeg")
            self.assertTrue(await wait_for(lambda: h.t.msgs("frame_error")))
            h.t.client_bytes(jpeg())
            self.assertTrue(await wait_for(lambda: h.t.msgs("frame")))
            self.assertEqual(h.ctl.state, "running")

    async def test_every_frame_gets_exactly_one_reply_and_newest_wins(self):
        async with Harness(self, delay=0.15) as h:
            h.t.client_text(action="start", mode="webcam")
            await wait_for(lambda: h.ctl.state == "running")
            for _ in range(6):
                h.t.client_bytes(jpeg())
            self.assertTrue(await wait_for(lambda: len(h.t.msgs("frame")) + len(h.t.msgs("skip")) == 6, timeout=20))
            await asyncio.sleep(0.3)
            self.assertEqual(len(h.t.msgs("frame")) + len(h.t.msgs("skip")), 6)
            self.assertLessEqual(len(h.t.msgs("frame")), 3, "stale frames should have been dropped")

    async def test_stop_and_reset_end_a_webcam_session(self):
        async with Harness(self) as h:
            h.t.client_text(action="start", mode="webcam")
            await wait_for(lambda: h.ctl.state == "running")
            h.t.client_bytes(jpeg()); await wait_for(lambda: h.t.msgs("frame"))
            h.t.client_text(action="stop"); await wait_for(lambda: h.ctl.state == "idle")
            h.t.client_bytes(jpeg()); await wait_for(lambda: h.t.msgs("skip"))
            h.t.client_text(action="start", mode="webcam"); await wait_for(lambda: h.ctl.state == "running")
            self.assertEqual(h.eng.frames, 0, "a new session must start from zero")
            h.t.client_text(action="reset"); await wait_for(lambda: h.ctl.state == "idle")


class SettingsOverSocketTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_and_invalid_settings(self):
        async with Harness(self) as h:
            h.t.client_text(action="start", mode="webcam"); await wait_for(lambda: h.ctl.state == "running")
            h.t.client_text(action="settings", conf=0.55, imgsz=416, frame_stride=2, tracker_embedder="none", class_names=["person"])
            await wait_for(lambda: [a for a in h.t.msgs("ack") if a["action"] == "settings"])
            self.assertEqual((h.eng.conf, h.eng.imgsz, h.eng.frame_stride, h.eng.embedder, h.eng.class_names), (0.55, 416, 2, "none", ["person"]))
            h.t.client_text(action="settings", conf=7)
            h.t.client_text(action="settings", imgsz=123)
            h.t.client_text(action="settings", class_names=["unicorn"])
            await wait_for(lambda: len(h.t.msgs("settings_error")) == 3)
            self.assertEqual(h.eng.conf, 0.55)
            self.assertEqual(h.ctl.state, "running", "a rejected setting must not stop the session")


if __name__ == "__main__":
    unittest.main()
