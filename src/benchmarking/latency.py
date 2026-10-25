"""Stage-separated latency measurement shared by the benchmark scripts."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from src.inference.onnx_detector import OnnxDetector, postprocess, preprocess

STAGES = ("load", "preprocess", "inference", "postprocess", "end_to_end")


def stats(v: list[float]) -> dict:
    a = np.asarray(v)
    return {"mean_ms": round(float(a.mean()), 3), "median_ms": round(float(np.median(a)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3), "p99_ms": round(float(np.percentile(a, 99)), 3),
            "std_ms": round(float(a.std(ddof=1)), 3), "min_ms": round(float(a.min()), 3)}


def measure(det: OnnxDetector, images: list[Path], warmup: int, iters: int, conf: float = 0.25) -> dict[str, list[float]]:
    """Raw per-iteration timings (ms) for each stage, batch size 1, cycling over real images."""
    for i in range(warmup):
        det(cv2.imread(str(images[i % len(images)])), conf=conf)
    rec = {k: [] for k in STAGES}
    for i in range(iters):
        p = images[i % len(images)]
        t0 = time.perf_counter()
        img = cv2.imread(str(p))
        t1 = time.perf_counter()
        x, info = preprocess(img, det.size)
        t2 = time.perf_counter()
        raw = det.raw(x)
        t3 = time.perf_counter()
        postprocess(raw, info, conf, 0.7)
        t4 = time.perf_counter()
        for k, dt_s in zip(STAGES, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0)):
            rec[k].append(dt_s * 1e3)
    return rec
