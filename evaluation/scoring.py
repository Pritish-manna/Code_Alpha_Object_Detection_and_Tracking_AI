"""Pure scoring helpers for the benchmark (no model needed, unit-tested)."""
from collections import Counter


def score_image(expected, tolerated, kept, names):
    """expected: set of COCO names that ARE in the image. tolerated: names that may legitimately appear
    (e.g. a hand holding the phone -> 'person'). kept: detections after the app's filters."""
    seen = [names[d.cls] for d in kept]
    ok = expected | tolerated
    hit = bool(expected & set(seen)) if expected else None          # None = negative image
    false = [n for n in seen if n not in ok]
    return dict(hit=hit, missed=(hit is False), false=false, confused_as=sorted(set(false)) if hit is False else [])


def aggregate(rows, key):
    """rows: dicts with the score fields plus metadata. Groups by `key` (e.g. 'object', 'lighting')."""
    groups = {}
    for r in rows:
        g = groups.setdefault(r[key], dict(images=0, hits=0, missed=0, false_dets=0, imgs_with_false=0, confusions=Counter()))
        g["images"] += 1
        g["false_dets"] += len(r["false"])
        g["imgs_with_false"] += bool(r["false"])
        if r["hit"] is True: g["hits"] += 1
        if r["missed"]:
            g["missed"] += 1
            g["confusions"].update(r["confused_as"])
    out = {}
    for k, g in groups.items():
        pos = g["hits"] + g["missed"]
        out[k] = dict(images=g["images"], recall=round(g["hits"] / pos, 3) if pos else None, missed=g["missed"],
                      false_dets=g["false_dets"], false_per_image=round(g["false_dets"] / g["images"], 3),
                      confused_as=dict(g["confusions"].most_common(3)))
    return out
