"""Detection session control for webcam and uploaded video. No web-framework imports.

One process-wide SessionController owns the single detection session, so two sessions can never run at
once. Every start/stop/reset is serialised by one lock and tagged with a run id; a task only touches shared
state while its run id is still current. That removes the old race where a task from a stopped session
could switch off a session started a moment later.

WebSocket protocol (server -> client JSON unless noted):
  {type:"state", state:"idle|running|paused", mode:"webcam|video|null", run_id}     on connect and on every change
  {type:"ack", action, ok:true, state, mode, stats?}                               reply to every client action
  {type:"error", action, message, code}                                            a client action was refused
  {type:"frame", stats, progress?}  followed by ONE binary message (annotated JPEG)
  {type:"skip"} / {type:"frame_error", message}                                    answer to a webcam frame that was not processed
  {type:"done", reason:"finished", stats}                                          uploaded video fully processed
  {type:"settings_error", message}                                                 a rejected setting (never stops a session)
Client -> server: binary = webcam JPEG frame; text = {action: start|stop|pause|resume|reset|settings, ...}.
Every webcam frame receives exactly one reply (frame, skip or frame_error).
"""
import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .settings import apply_settings
from .uploads import UploadError, safe_upload_path

log = logging.getLogger("session")


class SessionError(Exception):
    """A refused request. `status` doubles as the HTTP status for REST callers."""

    def __init__(self, message, status=409):
        super().__init__(message)
        self.status = status


class FrameError(Exception):
    pass


def encode_jpeg(frame, quality) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise FrameError("Could not encode the annotated frame.")
    return buf.tobytes()


class SessionController:
    def __init__(self, engine, detector, upload_dir, cfg):
        self.eng, self.det, self.upload_dir, self.cfg = engine, detector, upload_dir, cfg
        self.state = "idle"                       # idle | running | paused
        self.mode = None                          # webcam | video | None
        self.run_id = 0
        self.owner = None
        self.task = None
        self.subscribers = set()                  # async callables receiving {type:"state"...}
        self.infer_pool = ThreadPoolExecutor(1, thread_name_prefix="infer")   # one inference at a time, in order
        self.io_pool = ThreadPoolExecutor(2, thread_name_prefix="encode")
        self._lock = asyncio.Lock()

    # ---------- observers ----------
    def state_msg(self):
        return {"type": "state", "state": self.state, "mode": self.mode, "run_id": self.run_id}

    def subscribe(self, cb):
        self.subscribers.add(cb)

    def unsubscribe(self, cb):
        self.subscribers.discard(cb)

    async def _emit_state(self):
        msg = self.state_msg()
        for cb in list(self.subscribers):
            try:
                await cb(msg)
            except Exception:  # noqa: BLE001 - a dead socket must not break the others
                log.debug("state broadcast failed", exc_info=True)

    # ---------- lifecycle ----------
    async def start(self, mode, owner=None, file_id=None, send=None):
        async with self._lock:
            if mode not in ("webcam", "video"):
                raise SessionError("mode must be 'webcam' or 'video'.", 400)
            if self.det.status != "loaded":
                raise SessionError(f"Model {self.det.status}: {self.det.error or 'still loading'}", 503)
            if self.state != "idle":
                raise SessionError("Detection is already running. Stop it first.", 409)
            path = None
            if mode == "video":
                if send is None:
                    raise SessionError("Video detection is started over the WebSocket (action 'start' with file_id).", 400)
                try:
                    path = safe_upload_path(self.upload_dir, file_id)
                except UploadError as e:
                    raise SessionError(str(e), e.status)
            if self.task is not None:                          # a finished/finishing task from the last session
                await asyncio.gather(self.task, return_exceptions=True)
                self.task = None
            await asyncio.get_running_loop().run_in_executor(self.io_pool, self.eng.reset)
            self.run_id += 1
            rid = self.run_id
            self.mode, self.owner, self.state = mode, owner, "running"
            self.eng.mode, self.eng.running = mode, True
            if mode == "video":
                self.task = asyncio.create_task(self._play(rid, path, send))
        await self._emit_state()
        return rid

    async def stop(self):
        """Stop whatever is running and wait until its task and video file are released."""
        async with self._lock:
            changed = await self._stop_locked()
        if changed:
            await self._emit_state()
        return changed

    async def _stop_locked(self):
        task, self.task = self.task, None
        if self.state == "idle" and task is None:
            return False
        self.run_id += 1                                       # invalidates the old task and any frame still in flight
        self.state, self.mode, self.owner = "idle", None, None
        self.eng.running = False
        self.eng.paused = False
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        return True

    async def reset(self):
        """Stop, release the video, and clear tracks and statistics."""
        async with self._lock:
            await self._stop_locked()
            await asyncio.get_running_loop().run_in_executor(self.io_pool, self.eng.reset)
        await self._emit_state()

    async def pause(self):
        if self.state != "running":
            raise SessionError("Nothing is running to pause.", 409)
        self.state = "paused"
        self.eng.pause()
        await self._emit_state()

    async def resume(self):
        if self.state != "paused":
            raise SessionError("Detection is not paused.", 409)
        self.state = "running"
        self.eng.resume()
        await self._emit_state()

    async def connection_closed(self, owner):
        """Called when a WebSocket ends: stop the session it owned, or any session nobody can see any more."""
        if (owner is not None and self.owner is owner) or (not self.subscribers and self.state != "idle"):
            await self.stop()

    # ---------- webcam ----------
    def _webcam_job(self, data):
        t0 = time.perf_counter()
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise FrameError("Could not decode camera frame.")
        self.eng.note_timing("decode", (time.perf_counter() - t0) * 1000)
        out = self.eng.process(frame)
        t1 = time.perf_counter()
        jpg = encode_jpeg(out, self.cfg.jpeg_quality)
        self.eng.note_timing("encode", (time.perf_counter() - t1) * 1000)
        return jpg

    async def process_webcam(self, data):
        """Returns (meta, jpeg) or None when the frame should be answered with 'skip'."""
        if self.state != "running" or self.mode != "webcam":
            return None
        rid = self.run_id
        jpg = await asyncio.get_running_loop().run_in_executor(self.infer_pool, self._webcam_job, data)
        if self.run_id != rid:
            return None                                        # stopped/reset while this frame was processed
        return {"type": "frame", "stats": self.eng.stats()}, jpg

    # ---------- uploaded video ----------
    async def _emit_frame(self, send, meta, frame):
        jpg = await asyncio.get_running_loop().run_in_executor(self.io_pool, self._encode_timed, frame)
        await send(meta, jpg)

    def _encode_timed(self, frame):
        t = time.perf_counter()
        jpg = encode_jpeg(frame, self.cfg.jpeg_quality)
        self.eng.note_timing("encode", (time.perf_counter() - t) * 1000)
        return jpg

    async def _play(self, rid, path, send):
        loop = asyncio.get_running_loop()
        reader = ThreadPoolExecutor(1, thread_name_prefix="reader")   # every cv2 call on this file goes through it, in order
        cap = None
        error = None
        emit = None
        pending = None
        reason = "finished"
        try:
            cap = await asyncio.wrap_future(reader.submit(cv2.VideoCapture, path))
            if not cap.isOpened():
                raise RuntimeError("Could not open the uploaded video.")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            def read_one():
                for _ in range(max(1, int(self.eng.frame_stride)) - 1):   # frame stride: skip without decoding
                    if not cap.grab():
                        return None
                    self.eng.note_skipped(1)
                t = time.perf_counter()
                ok, fr = cap.read()
                if not ok:
                    return None
                self.eng.note_timing("decode", (time.perf_counter() - t) * 1000)
                return fr, int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            def read_next():
                return asyncio.wrap_future(reader.submit(read_one))

            pending = read_next()
            while self.run_id == rid:
                if self.state == "paused":
                    await asyncio.sleep(0.05)
                    continue
                item = await pending
                if item is None:
                    break
                frame, pos = item
                pending = read_next()                                          # decode the next frame while this one is inferred
                out = await loop.run_in_executor(self.infer_pool, self.eng.process, frame)
                if self.run_id != rid:
                    break
                if emit is not None:
                    await emit                                                 # at most one frame waiting to be sent
                progress = round(min(100.0, 100.0 * pos / total), 1) if total > 0 else None
                emit = asyncio.create_task(self._emit_frame(send, {"type": "frame", "stats": self.eng.stats(), "progress": progress}, out))
            if emit is not None:
                await emit
                emit = None
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - reported to the client, never swallowed silently
            log.exception("video processing failed")
            error = f"{type(e).__name__}: {e}" if str(e) == "" else str(e)
        finally:
            for fut in (emit, pending):
                if fut is not None and not fut.done():
                    fut.cancel()
            if cap is not None:
                reader.submit(cap.release)                                     # queued behind any read still running
            reader.shutdown(wait=False)
        if self.run_id == rid:                                                 # not stopped/reset in the meantime
            await self._finish(rid, send, error)

    async def _finish(self, rid, send, error):
        self.state, self.mode, self.owner, self.task = "idle", None, None, None
        self.eng.running = False
        self.eng.paused = False
        try:
            if error:
                await send({"type": "error", "action": "start", "message": error, "code": 500}, None)
            else:
                await send({"type": "done", "reason": "finished", "stats": self.eng.stats()}, None)
        except Exception:  # noqa: BLE001 - client already gone
            log.debug("could not deliver the final message", exc_info=True)
        await self._emit_state()

    # ---------- misc ----------
    def apply_settings(self, d):
        return apply_settings(self.eng, self.det, d)


class Connection:
    """One WebSocket client. `transport` provides: async receive() -> starlette-style message dict,
    async send_text(str), async send_bytes(bytes)."""

    def __init__(self, ctl: SessionController, transport):
        self.ctl, self.t = ctl, transport
        self._send_lock = asyncio.Lock()
        self._slot = None                         # newest webcam frame not yet processed
        self._slot_ev = asyncio.Event()

    # ---- sending (one message pair at a time) ----
    async def send_json(self, **msg):
        async with self._send_lock:
            await self.t.send_text(json.dumps(msg))

    async def _on_state(self, msg):
        await self.send_json(**msg)

    async def send_frame(self, meta, jpg=None):
        """Used for video output: frames are a JSON message plus one binary message; other messages are JSON only."""
        async with self._send_lock:
            await self.t.send_text(json.dumps(meta))
            if jpg is not None:
                await self.t.send_bytes(jpg)

    # ---- main loop ----
    async def serve(self):
        self.ctl.subscribe(self._on_state)
        worker = asyncio.create_task(self._webcam_worker())
        try:
            await self.send_json(**self.ctl.state_msg())
            while True:
                m = await self.t.receive()
                if m["type"] == "websocket.disconnect":
                    break
                if m.get("bytes") is not None:
                    await self._on_frame(m["bytes"])
                elif m.get("text") is not None:
                    await self._on_text(m["text"])
        finally:
            self.ctl.unsubscribe(self._on_state)
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await self.ctl.connection_closed(self)

    # ---- webcam frames: newest frame wins, every frame is answered ----
    async def _on_frame(self, data):
        if self._slot is not None:
            await self.send_json(type="skip")                  # the older, unprocessed frame is dropped
        self._slot = data
        self._slot_ev.set()

    async def _webcam_worker(self):
        while True:
            await self._slot_ev.wait()
            self._slot_ev.clear()
            data, self._slot = self._slot, None
            if data is None:
                continue
            try:
                res = await self.ctl.process_webcam(data)
                if res is None:
                    await self.send_json(type="skip")
                else:
                    await self.send_frame(*res)
            except asyncio.CancelledError:
                raise
            except FrameError as e:
                await self.send_json(type="frame_error", message=str(e))
            except Exception as e:  # noqa: BLE001
                log.exception("frame processing failed")
                await self.send_json(type="frame_error", message=f"{type(e).__name__}: {e}")

    # ---- control messages ----
    async def _on_text(self, text):
        action = None
        try:
            d = json.loads(text)
            if not isinstance(d, dict):
                raise ValueError("message must be a JSON object")
            action = d.get("action")
            ctl = self.ctl
            extra = {}
            if action == "start":
                await ctl.start(d.get("mode", "webcam"), owner=self, file_id=d.get("file_id"), send=self.send_frame)
            elif action == "stop":
                await ctl.stop()
                extra["stats"] = ctl.eng.stats()
            elif action == "pause":
                await ctl.pause()
            elif action == "resume":
                await ctl.resume()
            elif action == "reset":
                await ctl.reset()
                extra["stats"] = ctl.eng.stats()
            elif action == "settings":
                try:
                    ctl.apply_settings(d)
                except (ValueError, TypeError) as e:           # a rejected setting must not stop a running session
                    await self.send_json(type="settings_error", message=str(e))
                    return
            else:
                raise ValueError(f"unknown action {action!r}")
            await self.send_json(type="ack", action=action, ok=True, state=ctl.state, mode=ctl.mode, **extra)
        except SessionError as e:
            await self.send_json(type="error", action=action, message=str(e), code=e.status)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
            await self.send_json(type="error", action=action, message=f"Bad request: {e}", code=400)
