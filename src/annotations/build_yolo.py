"""Build a YOLO-format dataset directory from a frozen manifest and the raw masks."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from .mask_to_boxes import Box, components_to_boxes, mask_to_boxes, validate_yolo_line

CLASS_NAMES_MULTI = ["bent", "color", "flip", "scratch"]


def sample_stem(r: dict) -> str:
    return r["id"].replace("/", "_")  # e.g. test_bent_000 — unique across origin folders


def build(rows: list[dict], cat_dir: Path, out_dir: Path, policy: str = "components", merge_gap: int = 0,
          min_area: int = 0, multiclass: bool = False, splits=("train", "val", "test", "unseen")) -> dict:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    ann_rows, stats = [], {"images": 0, "boxes": 0, "empty_label_files": 0, "dropped_by_min_area": 0}
    for r in rows:
        if r["split"] not in splits:
            continue
        stem = sample_stem(r)
        img_dst = out_dir / "images" / r["split"] / f"{stem}.png"
        lbl_dst = out_dir / "labels" / r["split"] / f"{stem}.txt"
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        lbl_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cat_dir / r["image"], img_dst)
        stats["images"] += 1
        lines = []
        if r["mask"]:
            mask = cv2.imread(str(cat_dir / r["mask"]), cv2.IMREAD_GRAYSCALE)
            h, w = mask.shape
            all_cc, _ = components_to_boxes(mask, min_area=0)
            boxes = mask_to_boxes(mask, policy=policy, min_area=min_area, merge_gap=merge_gap)
            stats["dropped_by_min_area"] += len(all_cc) - len(components_to_boxes(mask, min_area=min_area)[0])
            cls = CLASS_NAMES_MULTI.index(r["category"]) if multiclass else 0
            for k, b in enumerate(boxes):
                cx, cy, bw, bh = b.yolo(w, h)
                validate_yolo_line(cls, cx, cy, bw, bh)
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                ann_rows.append({"stem": stem, "split": r["split"], "category": r["category"], "image": r["image"],
                                 "mask": r["mask"], "box_idx": k, "class_id": cls, **b.as_dict(),
                                 "cx": round(cx, 6), "cy": round(cy, 6), "nw": round(bw, 6), "nh": round(bh, 6)})
                stats["boxes"] += 1
        if not lines:
            stats["empty_label_files"] += 1
        lbl_dst.write_text("\n".join(lines) + ("\n" if lines else ""))

    names = CLASS_NAMES_MULTI if multiclass else ["defect"]
    present = sorted({r["split"] for r in rows if r["split"] in splits})
    cfg = {"path": str(out_dir.resolve()), "names": dict(enumerate(names))}
    for s in present:
        cfg[s] = f"images/{s}"
    (out_dir / "dataset.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    with open(out_dir / "annotations.csv", "w", newline="") as f:
        if ann_rows:
            wr = csv.DictWriter(f, fieldnames=list(ann_rows[0]), lineterminator="\n")
            wr.writeheader()
            wr.writerows(ann_rows)
    stats.update({"policy": policy, "merge_gap": merge_gap, "min_area": min_area, "multiclass": multiclass,
                  "splits_present": present})
    return stats


def qa_panel(img: np.ndarray, mask: np.ndarray, boxes: list[Box], title: str) -> np.ndarray:
    """Original | mask | coloured components | image with boxes — 4 tiles side by side."""
    _, labels = components_to_boxes(mask)
    rng = np.random.default_rng(0)
    colours = rng.integers(60, 255, size=(labels.max() + 1, 3), dtype=np.uint8)
    colours[0] = 0
    comp = colours[labels]
    over = img.copy()
    for b in boxes:
        cv2.rectangle(over, (b.x1, b.y1), (b.x2 - 1, b.y2 - 1), (0, 0, 255), 3)
    tiles = [img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), comp, over]
    tiles = [cv2.resize(t, (350, 350), interpolation=cv2.INTER_AREA) for t in tiles]
    panel = np.hstack(tiles)
    bar = np.full((30, panel.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, f"{title} | n_components={labels.max()} n_boxes={len(boxes)}", (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, panel])
