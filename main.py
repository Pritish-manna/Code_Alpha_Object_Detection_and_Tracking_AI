"""FastAPI wiring only. All behaviour lives in app.session / app.settings / app.uploads / app.detection."""
import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import load_config
from .detection.detector import Detector
from .detection.processor import Engine
from .session import Connection, SessionController, SessionError
from .settings import apply_settings, current_settings
from .uploads import UploadError, cleanup_uploads, probe_video, save_upload

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

UP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploads"))
os.makedirs(UP, exist_ok=True)
MAX = int(os.getenv("MAX_UPLOAD_MB", 200)) * 1024 * 1024
cfg = load_config()
det = Detector(cfg)
eng = Engine(det, cfg)
ctl = SessionController(eng, det, UP, cfg)


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=det.load, daemon=True, name="model-loader").start()
    removed = cleanup_uploads(UP, cfg.upload_ttl_hours)
    if removed:
        log.info("removed %d old uploaded video(s)", len(removed))
    yield
    await ctl.stop()


app = FastAPI(title="Object Detection & Tracking AI", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class SettingsBody(BaseModel):
    conf: float | None = None
    iou: float | None = None
    imgsz: int | str | None = None
    class_names: list[str] | None = None
    frame_stride: int | None = None
    tracker_embedder: str | None = None


class ModelSelect(BaseModel):
    model: str


# ---------- status / settings ----------
@app.get("/api/health")
def health():
    return {"status": "online", "running": eng.running, "state": ctl.state, "model_status": det.status}


@app.get("/api/model/status")
def model_status():
    return {**det.info(), **current_settings(eng)}


@app.get("/api/system")
def system():
    return det.system_info()


@app.post("/api/model/select", status_code=202)
def model_select(body: ModelSelect):
    try:
        changed = det.select(body.model)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"changed": changed, "loading": det.loading}


@app.get("/api/detection/settings")
def get_settings():
    return current_settings(eng)


@app.post("/api/detection/settings")
def set_settings(body: SettingsBody):
    try:
        return apply_settings(eng, det, body.model_dump(exclude_none=True))
    except (ValueError, TypeError) as e:
        raise HTTPException(422, str(e))


# ---------- statistics (real engine data only) ----------
@app.get("/api/detection/statistics")
def statistics():
    return eng.stats()


@app.get("/api/detection/analytics")
def analytics():
    return eng.analytics()


@app.get("/api/detection/history")
def history():
    return eng.history_snapshot(100)


# ---------- video upload ----------
@app.post("/api/video/upload")
async def upload(file: UploadFile = File(...)):
    loop = asyncio.get_running_loop()
    try:
        fid, path = await save_upload(file.read, file.filename, UP, MAX)
        try:
            info = await loop.run_in_executor(None, probe_video, path)
        except UploadError:
            os.remove(path)
            raise
    except UploadError as e:
        raise HTTPException(e.status, str(e))
    await loop.run_in_executor(None, cleanup_uploads, UP, cfg.upload_ttl_hours, {fid})
    return {"file_id": fid, "name": file.filename, **info}


# ---------- run control (REST mirrors the WebSocket actions; video is started over the WebSocket) ----------
async def _run(coro):
    try:
        await coro
    except SessionError as e:
        raise HTTPException(e.status, str(e))
    return {"running": ctl.state != "idle", "state": ctl.state, "mode": ctl.mode}


@app.post("/api/detection/start")
async def api_start(mode: str = "webcam"):
    return await _run(ctl.start(mode, owner=None))


@app.post("/api/detection/stop")
async def api_stop():
    return await _run(ctl.stop())


@app.post("/api/detection/pause")
async def api_pause():
    return await _run(ctl.pause())


@app.post("/api/detection/resume")
async def api_resume():
    return await _run(ctl.resume())


@app.post("/api/detection/reset")
async def api_reset():
    return await _run(ctl.reset())


class _Transport:
    """Adapts a Starlette WebSocket to the tiny interface app.session.Connection needs."""

    def __init__(self, sock: WebSocket):
        self.sock = sock

    async def receive(self):
        return await self.sock.receive()

    async def send_text(self, text):
        await self.sock.send_text(text)

    async def send_bytes(self, data):
        await self.sock.send_bytes(data)


@app.websocket("/ws/detection")
async def ws(sock: WebSocket):
    await sock.accept()
    try:
        await Connection(ctl, _Transport(sock)).serve()
    except (WebSocketDisconnect, RuntimeError):      # client went away mid-send
        pass
