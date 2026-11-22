# Object Detection & Tracking AI

FastAPI + YOLO11 + DeepSORT backend, React + Vite "Liquid Glass" dashboard.

## Run (Windows PowerShell)
```
cd backend
python -m venv venv; .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
# NVIDIA GPU (optional but ~10x faster): PyPI's Windows PyTorch is CPU-only, so install the CUDA build.
nvidia-smi                                            # shows your driver's max CUDA version
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu126    # cu128 / cu130 for newer drivers
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"                 # must print True for GPU use
uvicorn app.main:app --port 8000
# new terminal
cd frontend; npm install; npm run dev                 # http://localhost:5173
```
The header and Settings > Hardware acceleration show the device the model REALLY runs on (read from the loaded weights).
If it says CPU, the page explains why. Without a GPU everything still works on CPU: choose YOLO11n and 416/320 px in Settings.
Pages are addressed by URL hash (`#/settings`, `#/analytics`, ...), so reload and the browser Back button keep your place.
First start downloads `yolo11s.pt` (needs internet). If it can't, the app falls back to any weights already in
`backend/` (e.g. `yolo11n.pt`) and tells you in the Detection settings panel. `yolo11m.pt` downloads when first selected.

## Configuration (`backend/.env.example`)
`MODEL` (yolo11s.pt), `CONF` (0.40), `IOU` (0.50), `IMGSZ` (auto = 960 if one inference <= `LATENCY_BUDGET_MS`, else 640),
`MIN_BOX_PX`, `MAX_FRAME_WIDTH`. Also `DEVICE` (auto|cpu|cuda|cuda:N), `HALF` (FP16 on CUDA), `JPEG_QUALITY`, `FRAME_STRIDE`,
`TRACKER_EMBEDDER` (mobilenet|none), `TRACK_MAX_AGE`, `TRACK_N_INIT`, `TRACK_MAX_COSINE`. Inference sizes: 320, 416, 640, 960.
Model, size, confidence, frame stride and tracker appearance matching can also be changed live in Settings.

## How a frame is processed
browser (<=1280 px JPEG, newest frame wins, 2 in flight) -> decode (BGR) -> `prepare_frame` (validate, shrink to the inference size) -> YOLO (`imgsz`, `conf`, `iou`, class-aware NMS)
-> `clean_detections` (drop NaN/unknown-class/tiny/below-threshold; clip to frame; near-identical boxes with different classes -> keep the
more confident) -> DeepSORT -> `ClassVoter` per track (decayed confidence-weighted vote with hysteresis) -> draw.
Class names are read from the loaded weights and validated (`validate_names`); the app has no class list of its own.

Annotated frames come back as binary WebSocket messages (no base64). Stage timings (prep/detect/track/draw/encode) are shown on the
Dashboard so you can see where time goes. Protocol: see the docstring at the top of `backend/app/session.py`.

## Tests
```
cd backend
python -m unittest discover -s tests -t .
```
Unit/integration tests use a scripted stand-in for YOLO to test filtering/tracking/voting. They do NOT measure model accuracy.

## Measuring accuracy on your own images
```
python -m evaluation.capture --object books --lighting dim --angle top        # SPACE saves, Q quits; repeat per object/lighting/angle
python -m evaluation.capture --object _negative                               # scenes with no COCO objects
python -m evaluation.benchmark --models yolo11n.pt yolo11s.pt yolo11m.pt --imgsz 640 960 --conf 0.25 0.40 0.55 \
        --tolerate person --save-annotated evaluation/annotated
```
Reports recall, missed images, false detections, "confused as" (e.g. books -> motorcycle) per object / lighting / angle,
ms per image, and detections on blank/noise images (must be 0 - proves outputs depend on pixels). Results: `evaluation/results/`.
Folder aliases: `books`->book, `mobile_phone`->cell phone, `water_bottle`->bottle (`evaluation/object_map.json`, ground truth only).
