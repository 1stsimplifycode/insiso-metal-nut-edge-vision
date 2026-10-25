"""Classify every error at the operating point and render evidence panels."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .metrics import iou_matrix, match_image


def _ioa(inner: np.ndarray, outer: np.ndarray) -> np.ndarray:
    """Fraction of each `inner` box's area covered by each `outer` box."""
    if len(inner) == 0 or len(outer) == 0:
        return np.zeros((len(inner), len(outer)))
    x1 = np.maximum(inner[:, None, 0], outer[None, :, 0])
    y1 = np.maximum(inner[:, None, 1], outer[None, :, 1])
    x2 = np.minimum(inner[:, None, 2], outer[None, :, 2])
    y2 = np.minimum(inner[:, None, 3], outer[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (inner[:, 2] - inner[:, 0]) * (inner[:, 3] - inner[:, 1])
    return inter / (area[:, None] + 1e-9)


def region_contrast(img: np.ndarray, box, ring: int = 15) -> float:
    """|mean gray inside box - mean gray in a surrounding ring|: a crude visibility proxy."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    h, w = g.shape
    X1, Y1, X2, Y2 = max(0, x1 - ring), max(0, y1 - ring), min(w, x2 + ring), min(h, y2 + ring)
    outer = g[Y1:Y2, X1:X2].sum() - g[y1:y2, x1:x2].sum()
    n_outer = (X2 - X1) * (Y2 - Y1) - (x2 - x1) * (y2 - y1)
    inside = g[y1:y2, x1:x2].mean() if (x2 > x1 and y2 > y1) else 0.0
    return float(abs(inside - outer / max(n_outer, 1)))


def classify_errors(images: list[dict], conf: float) -> list[dict]:
    cases = []
    for im in images:
        allp = np.asarray(im["pred"], float).reshape(-1, 5)
        p = allp[allp[:, 4] >= conf]
        g = np.asarray(im["gt"], float).reshape(-1, 4)
        tp, gidx, best = match_image(p[:, :4], p[:, 4], g, 0.5)
        matched = set(int(j) for j in gidx if j >= 0)
        iou_pg = iou_matrix(p[:, :4], g)
        gt_in_pred = _ioa(g, p[:, :4])   # (G, P) how much of GT lies in each pred
        pred_in_gt = _ioa(p[:, :4], g)   # (P, G)
        # missed GT boxes
        for j in range(len(g)):
            if j in matched:
                continue
            iou_all = iou_matrix(allp[:, :4], g[j:j + 1])[:, 0] if len(allp) else np.zeros(0)
            below = np.where((iou_all >= 0.5) & (allp[:, 4] < conf))[0] if len(allp) else []
            best_iou = float(iou_pg[:, j].max()) if len(p) else 0.0
            if len(below):
                etype = "below_threshold"
                ref_conf = float(allp[below, 4].max())
            elif len(p) and gt_in_pred[j].max() >= 0.8:
                etype = "fragmentation_or_merged"   # GT fully inside a larger prediction
                ref_conf = float(p[int(gt_in_pred[j].argmax()), 4])
            elif best_iou >= 0.1:
                etype = "localisation"
                ref_conf = float(p[int(iou_pg[:, j].argmax()), 4])
            else:
                etype = "missed"
                ref_conf = float(allp[:, 4].max()) if len(allp) else 0.0
            cases.append({"id": im["id"], "image": im["image"], "category": im["category"], "kind": "FN",
                          "error_type": etype, "gt_box": g[j].tolist(), "best_iou_at_op": best_iou,
                          "conf": ref_conf, "image_flagged": bool(len(p))})
        # false-positive predictions
        for i in np.where(~tp)[0]:
            if im["category"] == "good":
                etype = "false_alarm_normal"
            elif len(g) and iou_pg[i].max() >= 0.5:
                etype = "duplicate"                 # GT already claimed by a higher-scoring box (NMS let both survive)
            elif len(g) and pred_in_gt[i].max() >= 0.8:
                etype = "fragmentation_or_merged"   # pred inside a GT region but IoU < 0.5 / duplicate
            elif len(g) and iou_pg[i].max() >= 0.1:
                etype = "localisation"
            else:
                etype = "spurious_on_defective"
            cases.append({"id": im["id"], "image": im["image"], "category": im["category"], "kind": "FP",
                          "error_type": etype, "pred_box": p[i, :4].tolist(), "conf": float(p[i, 4]),
                          "best_iou_at_op": float(iou_pg[i].max()) if len(g) else 0.0,
                          "image_flagged": True})
    return cases


def render_case(im: dict, conf: float, mask_path: Path | None, focus_box=None, title: str = "") -> np.ndarray:
    img = cv2.imread(im["image"])
    vis = img.copy()
    if mask_path and mask_path.exists():
        m = cv2.imread(str(mask_path), 0) > 127
        vis[m] = (0.55 * vis[m] + 0.45 * np.array([0, 200, 0])).astype(np.uint8)
    for b in im["gt"]:
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
    for b in im["pred"]:
        if b[4] < 0.05:
            continue
        col = (0, 0, 255) if b[4] >= conf else (0, 165, 255)
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2 if b[4] >= conf else 1)
        cv2.putText(vis, f"{b[4]:.2f}", (int(b[0]), max(12, int(b[1]) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1,
                    cv2.LINE_AA)
    if focus_box is not None:
        x1, y1, x2, y2 = (int(v) for v in focus_box)
        cv2.rectangle(vis, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), (255, 0, 255), 2)
        pad = 40
        crop = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
        crop = cv2.resize(crop, (350, 350), interpolation=cv2.INTER_CUBIC) if crop.size else np.zeros((350, 350, 3), np.uint8)
    else:
        crop = np.zeros((350, 350, 3), np.uint8)
    left = cv2.resize(img, (350, 350))
    mid = cv2.resize(vis, (350, 350))
    panel = np.hstack([left, mid, crop])
    bar = np.full((44, panel.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title[:110], (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(bar, "original | GT green (mask tint), pred>=thr red, pred<thr orange | zoom (magenta)", (8, 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1, cv2.LINE_AA)
    return np.vstack([bar, panel])
