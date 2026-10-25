"""Convert binary defect masks into tight bounding boxes (one per connected component)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class Box:
    x1: int  # inclusive pixel coords
    y1: int
    x2: int  # exclusive (x1 + width)
    y2: int
    area_px: int  # foreground pixels in the component (not box area)

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    def yolo(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        return ((self.x1 + self.x2) / 2 / img_w, (self.y1 + self.y2) / 2 / img_h, self.w / img_w, self.h / img_h)

    def as_dict(self) -> dict:
        return {**asdict(self), "w": self.w, "h": self.h}


def binarise(mask: np.ndarray, thresh: int = 127) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    return (mask > thresh).astype(np.uint8)


def components_to_boxes(mask: np.ndarray, min_area: int = 0, connectivity: int = 8) -> tuple[list[Box], np.ndarray]:
    """Return one Box per connected component with area >= min_area, plus the label image."""
    fg = binarise(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=connectivity)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area >= min_area:
            boxes.append(Box(x, y, x + w, y + h, area))
    boxes.sort(key=lambda b: (b.y1, b.x1))
    return boxes, labels


def merge_boxes(boxes: list[Box], gap: int = 0) -> list[Box]:
    """Merge boxes whose extents overlap or lie within `gap` px (iterated to a fixed point)."""
    bs = list(boxes)
    changed = True
    while changed:
        changed = False
        out: list[Box] = []
        while bs:
            b = bs.pop()
            i = 0
            while i < len(bs):
                o = bs[i]
                if (b.x1 - gap < o.x2 and o.x1 - gap < b.x2 and b.y1 - gap < o.y2 and o.y1 - gap < b.y2):
                    b = Box(min(b.x1, o.x1), min(b.y1, o.y1), max(b.x2, o.x2), max(b.y2, o.y2), b.area_px + o.area_px)
                    bs.pop(i)
                    changed = True
                else:
                    i += 1
            out.append(b)
        bs = out
    return sorted(bs, key=lambda b: (b.y1, b.x1))


def mask_to_boxes(mask: np.ndarray, policy: str = "components", min_area: int = 0, merge_gap: int = 0) -> list[Box]:
    """policy: 'components' (one box per CC), 'merged' (CCs within merge_gap px merged),
    'union' (single box around all foreground)."""
    boxes, _ = components_to_boxes(mask, min_area=min_area)
    if policy == "components" or not boxes:
        return boxes
    if policy == "merged":
        return merge_boxes(boxes, gap=merge_gap)
    if policy == "union":
        return [Box(min(b.x1 for b in boxes), min(b.y1 for b in boxes),
                    max(b.x2 for b in boxes), max(b.y2 for b in boxes), sum(b.area_px for b in boxes))]
    raise ValueError(policy)


def validate_yolo_line(cls: int, cx: float, cy: float, w: float, h: float) -> None:
    if cls < 0:
        raise ValueError("negative class id")
    for v in (cx, cy, w, h):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"YOLO value out of range: {v}")
    if w <= 0 or h <= 0:
        raise ValueError("non-positive box size")
    if cx - w / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6 or cy - h / 2 < -1e-6 or cy + h / 2 > 1 + 1e-6:
        raise ValueError("box extends outside image")
