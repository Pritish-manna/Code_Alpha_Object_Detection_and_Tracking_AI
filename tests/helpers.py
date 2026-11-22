"""Shared test doubles. ScriptedDetector replaces YOLO (it replays fixed detections); it exists so the
pipeline, session and WebSocket logic can be tested without model weights. It says nothing about accuracy."""
import asyncio
import json
import os
import tempfile
from types import SimpleNamespace

import cv2
import numpy as np

from app.config import Config
from app.detection.filters import Det
from app.detection.processor import Engine
from app.detection.tracker import Tracker
from app.session import Connection, SessionController

NAMES = {0: "person", 1: "book", 2: "motorcycle", 3: "chair"}
BOX = (80, 60, 220, 200)


def make_active(name="yolo11s.pt", device="cpu", half=False):
    return SimpleNamespace(name=name, names=NAMES, auto_imgsz=960, model=None, device=device, latency_ms=None,
                           half=half, device_name="test device")


class ScriptedDetector:
    status, error = "loaded", None

    def __init__(self, dets=None, delay=0.0):
        self.dets = [Det(*BOX, 0.9, 1)] if dets is None else dets
        self.delay, self.calls = delay, 0
        self._active, self.generation = make_active(), 1
        self.name = self._active.name

    @property
    def active(self): return self._active
    @property
    def device(self): return self._active.device
    @property
    def names(self): return NAMES

    def detect(self, frame, conf, iou, imgsz, class_ids, active=None):
        import time
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return list(self.dets)


def make_video(path, frames=30, size=(320, 240)):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25.0, size)
    assert w.isOpened(), "OpenCV cannot write MJPG video in this environment"
    for i in range(frames):
        f = np.full((size[1], size[0], 3), 40 + i % 50, np.uint8)
        cv2.rectangle(f, (20 + i, 40), (120 + i, 140), (200, 200, 200), -1)
        w.write(f)
    w.release()
    return path


def jpeg(color=90, size=(320, 240)):
    ok, b = cv2.imencode(".jpg", np.full((size[1], size[0], 3), color, np.uint8))
    return b.tobytes()


class FakeTransport:
    """Starlette-like WebSocket stand-in: tests push client messages, and read what the server sent."""

    def __init__(self):
        self.inbox = asyncio.Queue()
        self.sent = []                     # ("json", dict) | ("bytes", bytes)

    async def receive(self): return await self.inbox.get()
    async def send_text(self, t): self.sent.append(("json", json.loads(t)))
    async def send_bytes(self, b): self.sent.append(("bytes", b))

    def client_text(self, **msg): self.inbox.put_nowait({"type": "websocket.receive", "text": json.dumps(msg)})
    def client_bytes(self, b): self.inbox.put_nowait({"type": "websocket.receive", "bytes": b})
    def client_disconnect(self): self.inbox.put_nowait({"type": "websocket.disconnect"})

    def msgs(self, type_=None):
        out = [m for k, m in self.sent if k == "json"]
        return [m for m in out if m.get("type") == type_] if type_ else out


async def wait_for(pred, timeout=15.0, step=0.01):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        if pred():
            return True
        await asyncio.sleep(step)
    return False


def build(upload_dir, detector=None, **cfg_kw):
    cfg = Config(**cfg_kw)
    det = detector or ScriptedDetector()
    eng = Engine(det, cfg, tracker_factory=lambda: Tracker(embedder=None))
    return det, eng, SessionController(eng, det, upload_dir, cfg), cfg


def tmpdir():
    return tempfile.TemporaryDirectory()
