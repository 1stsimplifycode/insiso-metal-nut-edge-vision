"""Collect predictions + ground truth for a dataset split, from a .pt or an .onnx model."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def stem_category(stem: str) -> str:
    return stem.split("_")[1]  # "<origin>_<category>_<idx>"


def load_gt(label_path: Path, w: int, h: int) -> list[list[float]]:
    out = []
    for line in label_path.read_text().split("\n"):
        if not line.strip():
            continue
        _, cx, cy, bw, bh = map(float, line.split())
        out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return out


def collect(model_path: str, dataset_dir: Path, split: str, imgsz: int | None = None, conf: float = 0.001,
            iou: float = 0.7, gt_dataset_dir: Path | None = None, device: str = "cpu") -> list[dict]:
    """GT is read from `gt_dataset_dir` (default: dataset_dir) so that models trained on a different
    annotation policy are still scored against the same fixed reference labels."""
    gt_dir = gt_dataset_dir or dataset_dir
    img_paths = sorted((dataset_dir / "images" / split).glob("*.png"))
    if model_path.endswith(".onnx"):
        from src.inference.onnx_detector import OnnxDetector
        det = OnnxDetector(model_path)
        run = lambda img: det(img, conf=conf, iou=iou)[0]  # noqa: E731
    else:
        from ultralytics import YOLO
        model = YOLO(model_path)

        def run(img):
            r = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)[0]
            b = r.boxes
            return np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None],
                                   b.cls.cpu().numpy()[:, None]], 1) if len(b) else np.zeros((0, 6))
    out = []
    for p in img_paths:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        d = run(img)
        out.append({"id": p.stem, "category": stem_category(p.stem), "image": str(p),
                    "gt": load_gt(gt_dir / "labels" / split / f"{p.stem}.txt", w, h),
                    "pred": [[float(v) for v in row[:5]] for row in d]})
    return out
