"""Collect labelled test images from your webcam.
  python -m evaluation.capture --object books --lighting dim --angle top
SPACE = save frame, Q = quit. Files go to evaluation/dataset/<object>/<lighting>__<angle>__NN.jpg
Use --object _negative for scenes with no COCO objects. Frames are saved UNPROCESSED at full quality."""
import argparse
from pathlib import Path

import cv2

ap = argparse.ArgumentParser(); ap.add_argument("--object", required=True); ap.add_argument("--lighting", default="normal")
ap.add_argument("--angle", default="front"); ap.add_argument("--camera", type=int, default=0); ap.add_argument("--root", default="evaluation/dataset")
a = ap.parse_args()
out = Path(a.root) / a.object; out.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(a.camera); cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
n = len(list(out.glob(f"{a.lighting}__{a.angle}__*.jpg")))
while True:
    ok, f = cap.read()
    if not ok: break
    cv2.imshow(f"{a.object} | {a.lighting} | {a.angle} | saved {n} | SPACE=save Q=quit", f)
    k = cv2.waitKey(1) & 0xFF
    if k == 32: n += 1; cv2.imwrite(str(out / f"{a.lighting}__{a.angle}__{n:02d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 98])
    elif k in (ord("q"), 27): break
cap.release(); cv2.destroyAllWindows()
