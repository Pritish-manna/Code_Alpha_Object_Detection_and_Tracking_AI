"""Measure missed / false detections of the REAL model on YOUR images, using the app's own pipeline
(detector -> clean_detections). Run from the backend folder:

  python -m evaluation.benchmark --data evaluation/dataset --models yolo11n.pt yolo11s.pt yolo11m.pt --imgsz 640 960

Dataset layout (collect it with `python -m evaluation.capture`):
  dataset/<object>/<lighting>__<angle>__<anything>.jpg   e.g. books/dim__top__01.jpg
  dataset/_negative/*.jpg   scenes containing NO COCO object (any detection there is a false positive)
<object> is a COCO class name or an alias from object_map.json (books, mobile_phone, water_bottle, ...).
"""
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np

from app.config import Config
from app.detection.detector import Detector
from app.detection.filters import clean_detections, prepare_frame
from .scoring import aggregate, score_image

IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover(root, alias):
    items = []
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        neg = d.name == "_negative"
        coco = None if neg else alias.get(d.name, d.name.replace("_", " "))
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMG:
                parts = f.stem.split("__")
                items.append(dict(path=f, object=d.name, coco=coco, lighting=parts[0] if len(parts) > 1 else "untagged",
                                  angle=parts[1] if len(parts) > 2 else "untagged"))
    return items


def synthetic():
    rng = np.random.default_rng(0)
    g = np.tile(np.linspace(0, 255, 960, dtype=np.uint8), (540, 1))
    return {"black": np.zeros((540, 960, 3), np.uint8), "white": np.full((540, 960, 3), 255, np.uint8),
            "gray": np.full((540, 960, 3), 127, np.uint8), "noise": rng.integers(0, 256, (540, 960, 3), dtype=np.uint8),
            "gradient": cv2.merge([g, g, g]), "checker": (np.indices((540, 960)).sum(0) // 60 % 2 * 255).astype(np.uint8)[..., None].repeat(3, 2)}


def evaluate(model, imgsz, conf, iou, items, tolerate, save_dir):
    det = Detector(Config(default_model=model, imgsz=imgsz, conf=conf, iou=iou)); det.load()
    if det.status != "loaded": sys.exit(f"Could not load {model}: {det.error}")
    a, names, rows, times = det.active, det.active.names, [], []
    inv = set(names.values())
    for it in items:                                                # ground-truth names must exist in the model's own table
        if it["coco"] and it["coco"] not in inv: sys.exit(f"'{it['coco']}' (folder {it['object']}) is not a class of {model}")
    size = imgsz or a.auto_imgsz
    def run(frame):
        frame = prepare_frame(frame); t = time.perf_counter()
        raw = det.detect(frame, conf, iou, size, None, active=a); times.append((time.perf_counter() - t) * 1000)
        return frame, clean_detections(raw, frame.shape[1], frame.shape[0], names, conf)[0]
    for it in items:
        img = cv2.imread(str(it["path"]))
        if img is None: print("  unreadable:", it["path"]); continue
        frame, kept = run(img)
        r = score_image({it["coco"]} if it["coco"] else set(), set(tolerate), kept, names)
        r.update(object=it["object"], lighting=it["lighting"], angle=it["angle"], file=it["path"].name,
                 seen=[(names[d.cls], round(d.conf, 2)) for d in kept])
        rows.append(r)
        if save_dir:
            from app.detection.draw import class_color, draw_tracks
            draw_tracks(frame, [dict(box=(d.x1, d.y1, d.x2, d.y2), conf=d.conf, color=class_color(d.cls), label=f"{names[d.cls]} {d.conf:.0%}") for d in kept])
            out = Path(save_dir) / f"{model}_{size}_{conf}"; out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / f"{it['object']}__{it['path'].name}"), frame)
    synth = {k: len(run(v)[1]) for k, v in synthetic().items()}
    return dict(model=model, imgsz=size, conf=conf, iou=iou, device=a.device, ms_per_image=round(float(np.mean(times)), 1),
                by_object=aggregate(rows, "object"), by_lighting=aggregate(rows, "lighting"), by_angle=aggregate(rows, "angle"),
                overall=aggregate([dict(r, all="all") for r in rows], "all").get("all"), synthetic_detections=synth, rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="evaluation/dataset"); ap.add_argument("--models", nargs="+", default=["yolo11s.pt"])
    ap.add_argument("--imgsz", nargs="+", type=int, default=[0], help="0 = auto"); ap.add_argument("--conf", nargs="+", type=float, default=[0.40])
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--tolerate", nargs="*", default=[], help="classes allowed to appear without counting as false (e.g. person chair)")
    ap.add_argument("--save-annotated", default=None); ap.add_argument("--out", default="evaluation/results")
    a = ap.parse_args()
    alias = {k: v for k, v in json.loads((Path(__file__).parent / "object_map.json").read_text()).items() if not k.startswith("_")}
    items = discover(a.data, alias)
    if not items: sys.exit(f"No images found under {a.data}. Collect some with: python -m evaluation.capture --help")
    print(f"{len(items)} images, {len({i['object'] for i in items})} folders")
    results = [evaluate(m, s, c, a.iou, items, a.tolerate, a.save_annotated) for m in a.models for s in a.imgsz for c in a.conf]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "results.json").write_text(json.dumps(results, indent=1, default=str))
    lines = ["# Detection benchmark", "", "| model | imgsz | conf | ms/img | recall | missed | false dets | false/img | synthetic-image detections |", "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        o = r["overall"]; lines.append(f"| {r['model']} | {r['imgsz']} | {r['conf']} | {r['ms_per_image']} | {o['recall']} | {o['missed']} | {o['false_dets']} | {o['false_per_image']} | {sum(r['synthetic_detections'].values())} |")
    for r in results:
        lines += ["", f"## {r['model']} @ {r['imgsz']}px, conf {r['conf']}", "", "| object | images | recall | missed | false dets | confused as (when missed) |", "|---|---|---|---|---|---|"]
        lines += [f"| {k} | {v['images']} | {v['recall']} | {v['missed']} | {v['false_dets']} | {v['confused_as']} |" for k, v in r["by_object"].items()]
        for t in ("by_lighting", "by_angle"):
            lines += ["", f"By {t[3:]}: " + "; ".join(f"{k}: recall {v['recall']}, {v['false_dets']} false" for k, v in r[t].items())]
    (Path(a.out) / "report.md").write_text("\n".join(lines)); print("\n".join(lines)); print(f"\nSaved to {a.out}/")


if __name__ == "__main__":
    main()
