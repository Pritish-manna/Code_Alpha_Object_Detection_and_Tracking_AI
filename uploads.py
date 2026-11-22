"""Video upload helpers: streaming save with a size limit, validation, safe lookup and clean-up.
No web framework imports, so everything here is unit-testable."""
import os
import time
import uuid
from pathlib import Path

import cv2

ALLOWED_EXT = {".mp4", ".avi", ".mov"}


class UploadError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def safe_upload_path(upload_dir, file_id) -> str:
    """Resolve an uploaded file id to a path inside upload_dir (no path traversal)."""
    name = os.path.basename(str(file_id or ""))
    path = os.path.join(upload_dir, name)
    if not name or os.path.splitext(name)[1].lower() not in ALLOWED_EXT or not os.path.isfile(path):
        raise UploadError("Uploaded video not found. Upload it again.", 404)
    return path


def probe_video(path) -> dict:
    """Open the file and read its properties. Raises UploadError if it cannot be decoded."""
    cap = cv2.VideoCapture(path)
    try:
        ok, _ = cap.read()
        if not ok:
            raise UploadError("Video is corrupted or unreadable.", 422)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        return dict(frames=n if n > 0 else None, fps=round(fps, 2) if fps > 0 else None,
                    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    duration_s=round(n / fps, 1) if n > 0 and fps > 0 else None)
    finally:
        cap.release()


async def save_upload(read, filename, upload_dir, max_bytes) -> tuple[str, str]:
    """Stream `await read(n)` chunks to disk. Returns (file_id, path). Removes partial files on any error."""
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise UploadError("Unsupported format. Use MP4, AVI or MOV.", 400)
    os.makedirs(upload_dir, exist_ok=True)
    fid = uuid.uuid4().hex + ext
    path = os.path.join(upload_dir, fid)
    size = 0
    try:
        with open(path, "wb") as f:
            while chunk := await read(1 << 20):
                size += len(chunk)
                if size > max_bytes:
                    raise UploadError(f"File too large (limit {max_bytes // (1 << 20)} MB).", 413)
                f.write(chunk)
        if size == 0:
            raise UploadError("The uploaded file is empty.", 400)
    except BaseException:
        if os.path.exists(path):
            os.remove(path)
        raise
    return fid, path


def cleanup_uploads(upload_dir, ttl_hours, keep=()) -> list[str]:
    """Delete uploaded videos older than ttl_hours (never the ones in `keep`). Returns removed names."""
    removed = []
    if ttl_hours <= 0 or not os.path.isdir(upload_dir):
        return removed
    cutoff = time.time() - ttl_hours * 3600
    for p in Path(upload_dir).iterdir():
        if p.is_file() and p.suffix.lower() in ALLOWED_EXT and p.name not in keep:
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink()
                    removed.append(p.name)
            except OSError:
                pass
    return removed
