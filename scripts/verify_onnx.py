"""ONNX parity test: PyTorch checkpoint vs ONNX Runtime (CPU EP) on identical images.

Level 1 (raw tensors): same preprocessed input -> compare raw head outputs.
Level 2 (detections): Ultralytics .pt predict pipeline vs our numpy ONNX pipeline,
          matched by IoU, compare boxes and scores.

    python scripts/verify_onnx.py --weights models/baseline.pt --onnx models/baseline.onnx
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import iou_matrix  # noqa: E402
from src.inference.onnx_detector import OnnxDetector, preprocess  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--images", default=str(ROOT / "data" / "yolo" / "baseline" / "images" / "test"))
    ap.add_argument("--conf", type=float, default=0.25)
    # Output rows 0-3 are box coords in letterboxed pixels (0..640); rows 4.. are 0..1 scores.
    # A single absolute tolerance across both units is wrong (first run of this check did that).
    ap.add_argument("--atol-box-px", type=float, default=1e-2, help="raw box-row abs tolerance (pixels)")
    ap.add_argument("--atol-score", type=float, default=1e-4, help="raw score-row abs tolerance")
    a = ap.parse_args()

    import torch
    from ultralytics import YOLO

    yolo = YOLO(a.weights)
    net = yolo.model.float().eval().fuse()
    det = OnnxDetector(a.onnx)
    paths = sorted(Path(a.images).glob("*.png"))

    raw_max, raw_mean, score_diffs, box_diffs = [], [], [], []
    count_mismatch = []
    for p in paths:
        img = cv2.imread(str(p))
        x, _ = preprocess(img, det.size)
        with torch.no_grad():
            t = net(torch.from_numpy(x))
            t = (t[0] if isinstance(t, (list, tuple)) else t).numpy()
        o = det.raw(x)
        d = np.abs(t - o)
        raw_max.append((float(d[:, :4].max()), float(d[:, 4:].max())))  # (box px, score)
        raw_mean.append(float(d.mean()))

        r = yolo.predict(img, imgsz=det.size, conf=a.conf, iou=0.7, device="cpu", verbose=False)[0]
        pt = np.concatenate([r.boxes.xyxy.numpy(), r.boxes.conf.numpy()[:, None]], 1) if len(r.boxes) else np.zeros((0, 5))
        ox, _ = det(img, conf=a.conf, iou=0.7)
        if len(pt) != len(ox):
            count_mismatch.append({"image": p.name, "pt": len(pt), "onnx": len(ox),
                                   "pt_scores": pt[:, 4].round(4).tolist(), "onnx_scores": ox[:, 4].round(4).tolist()})
        if len(pt) and len(ox):
            ious = iou_matrix(pt[:, :4], ox[:, :4])
            for i in range(len(pt)):
                j = int(ious[i].argmax())
                if ious[i, j] > 0.5:
                    score_diffs.append(abs(pt[i, 4] - ox[j, 4]))
                    box_diffs.append(float(np.abs(pt[i, :4] - ox[j, :4]).max()))

    rep = {
        "images": len(paths), "conf": a.conf,
        "raw_box_rows_max_abs_diff_px": max(b for b, _ in raw_max),
        "raw_score_rows_max_abs_diff": max(s for _, s in raw_max),
        "raw_output_mean_abs_diff": float(np.mean(raw_mean)),
        "raw_atol_box_px": a.atol_box_px, "raw_atol_score": a.atol_score,
        "raw_pass": max(b for b, _ in raw_max) <= a.atol_box_px and max(s for _, s in raw_max) <= a.atol_score,
        "matched_detections": len(score_diffs),
        "detection_score_max_abs_diff": float(max(score_diffs)) if score_diffs else None,
        "detection_box_max_abs_diff_px": float(max(box_diffs)) if box_diffs else None,
        "images_with_detection_count_mismatch": count_mismatch,
        "note": "raw outputs are compared on identical preprocessed tensors; detection-level differences can "
                "additionally arise from threshold-boundary scores and NMS tie-breaking.",
    }
    rep["detection_pass"] = (not count_mismatch) and (rep["detection_box_max_abs_diff_px"] or 0) < 1.0
    out = ROOT / "artifacts" / "onnx_parity" / f"{Path(a.onnx).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    sys.exit(0 if rep["raw_pass"] and rep["detection_pass"] else 1)


if __name__ == "__main__":
    main()
