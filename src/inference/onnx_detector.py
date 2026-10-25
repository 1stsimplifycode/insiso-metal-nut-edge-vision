"""Framework-free YOLO detector on ONNX Runtime: letterbox -> ORT -> decode -> NMS -> rescale.

Only numpy, OpenCV and onnxruntime are needed, so this is what runs on the edge device.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class LetterboxInfo:
    gain: float
    pad_x: float
    pad_y: float
    src_w: int
    src_h: int


def letterbox(img_bgr: np.ndarray, size: int, colour: int = 114) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize keeping aspect ratio, pad to size x size (centred), as Ultralytics' LetterBox does."""
    h, w = img_bgr.shape[:2]
    g = min(size / h, size / w)
    nw, nh = int(round(w * g)), int(round(h * g))
    if (nw, nh) != (w, h):
        img_bgr = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(img_bgr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(colour,) * 3)
    return out, LetterboxInfo(g, left, top, w, h)


def preprocess(img_bgr: np.ndarray, size: int) -> tuple[np.ndarray, LetterboxInfo]:
    lb, info = letterbox(img_bgr, size)
    x = lb[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    x = np.ascontiguousarray(x, dtype=np.float32) / 255.0
    return x[None], info


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    order = scores.argsort()[::-1]
    keep = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.asarray(keep, int)


def postprocess(raw: np.ndarray, info: LetterboxInfo, conf: float, iou: float, max_det: int = 300) -> np.ndarray:
    """raw: (1, 4+nc, N) YOLOv8/11 head output. Returns (K, 6) [x1,y1,x2,y2,score,cls] in source pixels."""
    p = raw[0].T  # (N, 4+nc)
    cls_scores = p[:, 4:]
    cls = cls_scores.argmax(1)
    score = cls_scores[np.arange(len(p)), cls]
    m = score >= conf
    p, score, cls = p[m], score[m], cls[m]
    if not len(p):
        return np.zeros((0, 6), np.float32)
    xy, wh = p[:, :2], p[:, 2:4]
    boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1)
    # class-aware NMS via coordinate offset (same trick as Ultralytics)
    keep = nms(boxes + cls[:, None] * 7680.0, score, iou)[:max_det]
    boxes, score, cls = boxes[keep], score[keep], cls[keep]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - info.pad_x) / info.gain
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - info.pad_y) / info.gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, info.src_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, info.src_h)
    return np.concatenate([boxes, score[:, None], cls[:, None]], 1).astype(np.float32)


@dataclass
class Timings:
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    extra: dict = field(default_factory=dict)


class OnnxDetector:
    def __init__(self, path: str, providers=("CPUExecutionProvider",), intra_threads: int | None = None,
                 graph_opt: str = "all"):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if intra_threads:
            so.intra_op_num_threads = intra_threads
        so.graph_optimization_level = {
            "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL}[graph_opt]
        self.session = ort.InferenceSession(path, so, providers=list(providers))
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape
        self.size = int(inp.shape[2])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names = meta.get("names", "")
        self.path = path

    def raw(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: x})[0]

    def __call__(self, img_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.7) -> tuple[np.ndarray, Timings]:
        t = Timings()
        t0 = time.perf_counter()
        x, info = preprocess(img_bgr, self.size)
        t1 = time.perf_counter()
        raw = self.raw(x)
        t2 = time.perf_counter()
        det = postprocess(raw, info, conf, iou)
        t3 = time.perf_counter()
        t.preprocess_ms, t.inference_ms, t.postprocess_ms = (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3
        return det, t
