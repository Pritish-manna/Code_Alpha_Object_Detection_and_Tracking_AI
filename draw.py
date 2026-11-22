"""Bounding-box rendering. Scales with frame size, anti-aliased, keeps labels inside the frame."""
import colorsys

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX


def class_color(cls_id: int):
    """Stable BGR colour per class id (golden-ratio hue spacing). Colour carries no class meaning."""
    r, g, b = colorsys.hsv_to_rgb((cls_id * 0.618033988749895) % 1.0, 0.72, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _text_color(bgr):
    luma = 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]
    return (20, 20, 20) if luma > 140 else (255, 255, 255)


def draw_tracks(frame, items):
    """items: list of dicts {box:(x1,y1,x2,y2), label:str, color:(b,g,r), conf:float 0..1}. Draws in place."""
    if not items:
        return frame
    h, w = frame.shape[:2]
    scale = max(0.5, min(h, w) / 800)
    thick = max(2, round(min(h, w) / 360))
    tthick = max(1, round(scale * 1.3))
    items = sorted(items, key=lambda i: (i["box"][2] - i["box"][0]) * (i["box"][3] - i["box"][1]), reverse=True)

    boxes = []
    for it in items:                                    # clamp to the frame
        x1, y1, x2, y2 = (int(round(v)) for v in it["box"])
        boxes.append((max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)))

    overlay = frame.copy()                              # one translucent fill pass for all boxes
    for (x1, y1, x2, y2), it in zip(boxes, items):
        cv2.rectangle(overlay, (x1, y1), (x2, y2), it["color"], -1)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, dst=frame)

    for (x1, y1, x2, y2), it in zip(boxes, items):
        c = it["color"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, max(1, thick - 1), cv2.LINE_AA)
        ln = int(max(10, min(40, 0.22 * min(x2 - x1, y2 - y1))))   # heavier corner brackets
        for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(frame, (px, py), (px + dx * ln, py), c, thick + 1, cv2.LINE_AA)
            cv2.line(frame, (px, py), (px, py + dy * ln), c, thick + 1, cv2.LINE_AA)

        (tw, th), base = cv2.getTextSize(it["label"], FONT, 0.55 * scale, tthick)
        pad = max(3, int(4 * scale))
        cw, ch = tw + 2 * pad, th + base + 2 * pad
        cx = min(max(0, x1), max(0, w - cw))
        cy = y1 - ch if y1 - ch >= 0 else min(y1, max(0, h - ch))   # above the box, or inside if no room
        cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), c, -1)
        cv2.putText(frame, it["label"], (cx + pad, cy + pad + th), FONT, 0.55 * scale, _text_color(c), tthick, cv2.LINE_AA)
        bar = int(cw * max(0.0, min(1.0, it.get("conf", 0.0))))         # confidence bar along the chip's base
        cv2.rectangle(frame, (cx, cy + ch - 3), (cx + bar, cy + ch), _text_color(c), -1)
    return frame
