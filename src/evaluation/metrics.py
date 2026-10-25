"""Box matching and detection metrics (numpy only, framework-independent).

Conventions
- Boxes are [x1, y1, x2, y2] in original-image pixels.
- Matching: per image, predictions sorted by descending confidence; each is matched to the
  unmatched GT with highest IoU if IoU >= thr (greedy, one-to-one) — PASCAL/COCO style.
- AP: COCO 101-point interpolation (mean over r in {0, .01, ..., 1} of the max precision at
  recall >= r, 0 where recall r is never reached). Note Ultralytics' validator instead
  integrates a trapezoid toward a (recall=1, precision=0) sentinel, so its numbers differ
  slightly (e.g. it reports 0.995 for a perfect detector).
"""
from __future__ import annotations

import numpy as np

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float), np.asarray(b, float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_image(pred_boxes, pred_scores, gt_boxes, thr: float):
    """Return (tp flags per prediction in input order, matched gt index per prediction or -1, best IoU per pred)."""
    n = len(pred_boxes)
    tp = np.zeros(n, bool)
    gt_idx = -np.ones(n, int)
    ious = iou_matrix(pred_boxes, gt_boxes)
    best = ious.max(1) if ious.size else np.zeros(n)
    used = np.zeros(len(gt_boxes), bool)
    for i in np.argsort(-np.asarray(pred_scores), kind="stable"):
        if len(gt_boxes) == 0:
            break
        cand = np.where(~used, ious[i], -1)
        j = int(cand.argmax())
        if cand[j] >= thr:
            tp[i], gt_idx[i], used[j] = True, j, True
    return tp, gt_idx, best


def average_precision(tp: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = tp[order].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / (ctp + cfp)
    # precision envelope: max precision at any recall >= r
    env = np.flip(np.maximum.accumulate(np.flip(precision)))
    out = []
    for r in np.linspace(0, 1, 101):
        k = np.searchsorted(recall, r - 1e-12, side="left")
        out.append(env[k] if k < len(env) else 0.0)
    return float(np.mean(out))


def evaluate(images: list[dict], conf: float) -> dict:
    """images: [{'id','category','gt': [[x1,y1,x2,y2],...], 'pred': [[x1,y1,x2,y2,score],...]}]
    Ranking metrics (AP) use all predictions; operating-point metrics use score >= conf."""
    n_gt = sum(len(im["gt"]) for im in images)
    ap = {}
    for t in IOU_THRESHOLDS:
        all_tp, all_s = [], []
        for im in images:
            p = np.asarray(im["pred"], float).reshape(-1, 5)
            tp, _, _ = match_image(p[:, :4], p[:, 4], np.asarray(im["gt"], float).reshape(-1, 4), t)
            all_tp.append(tp)
            all_s.append(p[:, 4])
        ap[round(float(t), 2)] = average_precision(np.concatenate(all_tp), np.concatenate(all_s), n_gt)

    # operating point at IoU 0.5
    tp_n = fp_n = 0
    per_cat = {}
    img_rows = []
    fp_on_normal = 0
    n_normal = n_anom = flagged_normal = flagged_anom = 0
    for im in images:
        p = np.asarray(im["pred"], float).reshape(-1, 5)
        p = p[p[:, 4] >= conf]
        g = np.asarray(im["gt"], float).reshape(-1, 4)
        tp, gidx, best = match_image(p[:, :4], p[:, 4], g, 0.5)
        tp_n += int(tp.sum())
        fp_n += int((~tp).sum())
        matched = set(int(j) for j in gidx if j >= 0)
        c = per_cat.setdefault(im["category"], {"gt_boxes": 0, "tp_boxes": 0, "images": 0, "flagged": 0})
        c["gt_boxes"] += len(g)
        c["tp_boxes"] += len(matched)
        c["images"] += 1
        flagged = len(p) > 0
        c["flagged"] += int(flagged)
        if im["category"] == "good":
            n_normal += 1
            flagged_normal += int(flagged)
            fp_on_normal += len(p)
        else:
            n_anom += 1
            flagged_anom += int(flagged)
        img_rows.append({"id": im["id"], "category": im["category"], "n_gt": len(g), "n_pred": len(p),
                         "tp": int(tp.sum()), "fp": int((~tp).sum()), "fn": len(g) - len(matched),
                         "flagged": flagged, "max_score": float(p[:, 4].max()) if len(p) else 0.0})
    fn_n = n_gt - tp_n
    prec = tp_n / (tp_n + fp_n) if tp_n + fp_n else float("nan")
    rec = tp_n / n_gt if n_gt else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    for c in per_cat.values():
        c["box_recall"] = c["tp_boxes"] / c["gt_boxes"] if c["gt_boxes"] else None
        c["image_flag_rate"] = c["flagged"] / c["images"]
    return {
        "conf_threshold": conf, "iou_threshold": 0.5,
        "n_images": len(images), "n_anomalous_images": n_anom, "n_normal_images": n_normal, "n_gt_boxes": n_gt,
        "box": {"tp": tp_n, "fp": fp_n, "fn": fn_n, "precision": prec, "recall": rec, "f1": f1},
        "mAP50": ap[0.5], "mAP50_95": float(np.mean(list(ap.values()))), "ap_by_iou": ap,
        "image": {"defective_flagged": flagged_anom, "defective_total": n_anom,
                  "sensitivity": flagged_anom / n_anom if n_anom else None,
                  "normal_flagged": flagged_normal, "normal_total": n_normal,
                  "false_positive_rate": flagged_normal / n_normal if n_normal else None,
                  "missed_defective_images": n_anom - flagged_anom},
        "fp_boxes_per_normal_image": fp_on_normal / n_normal if n_normal else None,
        "per_category": per_cat, "per_image": img_rows,
    }


def size_bin(box) -> str:
    s = np.sqrt((box[2] - box[0]) * (box[3] - box[1]))
    return "small(<50px)" if s < 50 else "medium(50-150px)" if s < 150 else "large(>=150px)"


def recall_by_size(images: list[dict], conf: float) -> dict:
    out = {}
    for im in images:
        p = np.asarray(im["pred"], float).reshape(-1, 5)
        p = p[p[:, 4] >= conf]
        g = np.asarray(im["gt"], float).reshape(-1, 4)
        _, gidx, _ = match_image(p[:, :4], p[:, 4], g, 0.5)
        matched = set(int(j) for j in gidx if j >= 0)
        for j, b in enumerate(g):
            d = out.setdefault(size_bin(b), {"gt": 0, "tp": 0})
            d["gt"] += 1
            d["tp"] += int(j in matched)
    for d in out.values():
        d["recall"] = d["tp"] / d["gt"]
    return dict(sorted(out.items()))


def zero_false_reject_threshold(images: list[dict], grid=np.round(np.arange(0.01, 0.96, 0.01), 2)) -> tuple[float, float]:
    """Recall-first rule: the LOWEST confidence at which no normal image is flagged.
    Returns (threshold, image sensitivity at that threshold)."""
    for c in grid:
        r = evaluate(images, float(c))
        if r["image"]["normal_flagged"] == 0:
            return float(c), r["image"]["sensitivity"]
    return float(grid[-1]), evaluate(images, float(grid[-1]))["image"]["sensitivity"]


def best_f1_threshold(images: list[dict], grid=np.linspace(0.05, 0.95, 91)) -> tuple[float, float]:
    """Confidence that maximises box-level F1 (ties -> lower threshold, i.e. favour recall)."""
    best = (0.25, -1.0)
    for c in grid:
        f1 = evaluate(images, float(c))["box"]["f1"]
        if f1 > best[1] + 1e-12:
            best = (round(float(c), 3), f1)
    return best
