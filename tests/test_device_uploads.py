"""Device selection honesty, upload handling and config validation."""
import asyncio
import os
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from app.config import load_config
from app.detection.detector import CUDA_INSTALL_HINT, Detector, resolve_device
from app.settings import apply_settings, current_settings
from app.uploads import UploadError, cleanup_uploads, probe_video, safe_upload_path, save_upload
from tests.helpers import ScriptedDetector, build, make_video, tmpdir


def fake_torch(version="2.5.0+cpu", built=None, avail=False, devices=("NVIDIA GeForce RTX 3060",), boom=False):
    def is_available():
        if boom: raise RuntimeError("driver exploded")
        return avail
    cuda = SimpleNamespace(is_available=is_available, device_count=lambda: len(devices) if avail else 0,
                           get_device_name=lambda i: devices[i])
    return SimpleNamespace(__version__=version, version=SimpleNamespace(cuda=built), cuda=cuda)


class DeviceTests(unittest.TestCase):
    def test_cpu_only_wheel_is_explained_with_the_fix(self):
        c = resolve_device("auto", fake_torch())
        self.assertEqual(c.device, "cpu")
        self.assertIn("CPU-only", c.reason)
        self.assertIn("download.pytorch.org/whl/cu", c.reason)
        self.assertIsNone(c.torch_cuda)

    def test_cuda_build_but_no_gpu_points_at_the_driver(self):
        c = resolve_device("auto", fake_torch("2.5.0+cu124", "12.4", avail=False))
        self.assertEqual(c.device, "cpu")
        self.assertIn("nvidia-smi", c.reason)

    def test_gpu_is_used_when_really_available(self):
        c = resolve_device("auto", fake_torch("2.5.0+cu124", "12.4", avail=True))
        self.assertEqual((c.device, c.gpu_name), ("cuda:0", "NVIDIA GeForce RTX 3060"))
        self.assertIn("RTX 3060", c.reason)

    def test_forced_cpu_wins_even_with_a_gpu(self):
        self.assertEqual(resolve_device("cpu", fake_torch("2.5.0+cu124", "12.4", avail=True)).device, "cpu")

    def test_unavailable_cuda_index_falls_back_to_cpu(self):
        c = resolve_device("cuda:3", fake_torch("2.5.0+cu124", "12.4", avail=True))
        self.assertEqual(c.device, "cpu"); self.assertIn("only 1", c.reason)

    def test_broken_driver_falls_back_instead_of_crashing(self):
        self.assertEqual(resolve_device("auto", fake_torch("2.5.0+cu124", "12.4", boom=True)).device, "cpu")

    def test_device_is_unknown_not_cpu_until_a_model_is_loaded(self):
        d = Detector(load_config())
        self.assertIsNone(d.device)
        info = d.info()
        self.assertIsNone(info["device"]); self.assertFalse(info["gpu_active"])

    def test_hint_command_targets_the_pytorch_index(self):
        self.assertIn("--index-url https://download.pytorch.org/whl/cu", CUDA_INSTALL_HINT)

    def test_actual_device_is_read_from_the_weights(self):
        class P:  # a parameter living on cuda:0
            device = SimpleNamespace(type="cuda", index=0)
        model = SimpleNamespace(model=SimpleNamespace(parameters=lambda: iter([P()])))
        self.assertEqual(Detector._actual_device(model, "cpu"), "cuda:0")
        self.assertEqual(Detector._actual_device(SimpleNamespace(), "cpu"), "cpu")


class UploadTests(unittest.IsolatedAsyncioTestCase):
    def chunks(self, data, n=1 << 16):
        it = iter([data[i:i + n] for i in range(0, len(data), n)] + [b""])
        async def read(_): return next(it)
        return read

    async def test_save_validate_and_lookup(self):
        with tmpdir() as d:
            src = make_video(os.path.join(d, "src.avi"), 12)
            with open(src, "rb") as f: data = f.read()
            fid, path = await save_upload(self.chunks(data), "My Clip.AVI", d, 10 << 20)
            self.assertTrue(fid.endswith(".avi") and os.path.isfile(path))
            info = probe_video(path)
            self.assertEqual((info["frames"], info["width"], info["height"]), (12, 320, 240))
            self.assertEqual(safe_upload_path(d, fid), path)

    async def test_rejects_wrong_extension_oversize_empty_and_corrupt(self):
        with tmpdir() as d:
            with self.assertRaises(UploadError) as e: await save_upload(self.chunks(b"x"), "a.txt", d, 1 << 20)
            self.assertEqual(e.exception.status, 400)
            with self.assertRaises(UploadError) as e: await save_upload(self.chunks(b"x" * 5000), "a.mp4", d, 1000)
            self.assertEqual(e.exception.status, 413)
            with self.assertRaises(UploadError) as e: await save_upload(self.chunks(b""), "a.mp4", d, 1000)
            self.assertEqual(e.exception.status, 400)
            self.assertEqual(os.listdir(d), [], "partial files must be removed")
            fid, path = await save_upload(self.chunks(b"this is not a video" * 50), "bad.mp4", d, 1 << 20)
            with self.assertRaises(UploadError) as e: probe_video(path)
            self.assertEqual(e.exception.status, 422)

    def test_path_traversal_is_neutralised(self):
        with tmpdir() as d:
            for bad in ("../../etc/passwd", "..\\..\\x.avi", "", None, "missing.avi"):
                with self.assertRaises(UploadError): safe_upload_path(d, bad)

    def test_cleanup_only_removes_old_videos_and_keeps_requested(self):
        with tmpdir() as d:
            old, new, keep, other = (os.path.join(d, n) for n in ("old.mp4", "new.mp4", "keep.mp4", "notes.txt"))
            for p in (old, new, keep, other):
                with open(p, "w") as f: f.write("x")
            past = time.time() - 48 * 3600
            for p in (old, keep, other): os.utime(p, (past, past))
            self.assertEqual(cleanup_uploads(d, 24, keep={"keep.mp4"}), ["old.mp4"])
            self.assertEqual(sorted(os.listdir(d)), ["keep.mp4", "new.mp4", "notes.txt"])
            self.assertEqual(cleanup_uploads(d, 0), [])


class SettingsConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tmpdir(); self.addCleanup(self.tmp.cleanup)
        self.det, self.eng, _, _ = build(self.tmp.name)

    def test_new_inference_sizes_are_accepted(self):
        for s in (320, 416, 640, 960, "auto"):
            self.assertEqual(apply_settings(self.eng, self.det, {"imgsz": s})["imgsz"], s if s != "auto" else "auto")
        with self.assertRaises(ValueError): apply_settings(self.eng, self.det, {"imgsz": 480})

    def test_invalid_values_change_nothing(self):
        before = current_settings(self.eng)
        for bad in ({"conf": 0.5, "frame_stride": 99}, {"conf": 0.5, "tracker_embedder": "x"}, {"iou": 2}):
            with self.assertRaises(ValueError): apply_settings(self.eng, self.det, bad)
        self.assertEqual(current_settings(self.eng), before)

    def test_config_env_validation(self):
        with mock.patch.dict(os.environ, {"DEVICE": "cuda:1", "HALF": "0", "FRAME_STRIDE": "4", "TRACKER_EMBEDDER": "none", "IMGSZ": "416"}, clear=True):
            c = load_config()
        self.assertEqual((c.device, c.half, c.frame_stride, c.tracker_embedder, c.imgsz), ("cuda:1", False, 4, "none", 416))
        for k, v in (("DEVICE", "tpu"), ("HALF", "maybe"), ("TRACKER_EMBEDDER", "resnet"), ("IMGSZ", "300")):
            with mock.patch.dict(os.environ, {k: v}, clear=True), self.assertRaises(SystemExit):
                load_config()


if __name__ == "__main__":
    unittest.main()
