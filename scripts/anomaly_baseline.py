"""Normal-only PatchCore baseline on the SAME frozen split as the detector.

Memory bank: train-split normal images only (no defect images used for fitting).
Image threshold: chosen on the val split (max image-level F1), then applied unchanged to test.
Compared with the detector at IMAGE level (the only level where both produce the same kind of output).

    python scripts/anomaly_baseline.py
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.anomaly.patchcore import PatchCore  # noqa: E402
from src.dataset.split import read_manifest, verify_frozen  # noqa: E402

RAW = ROOT / "data" / "raw" / "metal_nut"


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def img_metrics(y, s, thr):
    pred = s >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    return {"threshold": float(thr), "defective_flagged": tp, "defective_total": int((y == 1).sum()),
            "normal_flagged": fp, "normal_total": int((y == 0).sum()),
            "sensitivity": tp / max(1, (y == 1).sum()), "false_positive_rate": fp / max(1, (y == 0).sum())}


def main() -> None:
    import torch
    man = ROOT / "manifests" / "split_seed42.csv"
    verify_frozen(man)
    rows = read_manifest(man)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pc = PatchCore(device=dev)
    train_norm = [r for r in rows if r["split"] == "train" and r["category"] == "good"]
    t0 = time.time()
    pc.fit([cv2.imread(str(RAW / r["image"])) for r in train_norm])
    fit_s = time.time() - t0
    out = {"method": "PatchCore (WRN50 layer2+3, 256px, 10% greedy coreset)", "device": dev,
           "fit_images": len(train_norm), "memory_bank": int(len(pc.bank)), "fit_seconds": round(fit_s, 1)}
    scores = {}
    maps_dir = ROOT / "reports" / "anomaly_heatmaps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        rs = [r for r in rows if r["split"] == split]
        y, s, times = [], [], []
        for r in rs:
            img = cv2.imread(str(RAW / r["image"]))
            t = time.perf_counter()
            sc, amap = pc.predict(img)
            times.append((time.perf_counter() - t) * 1e3)
            y.append(int(r["category"] != "good"))
            s.append(sc)
            scores[r["id"]] = {"split": split, "category": r["category"], "score": sc}
            if split == "test":
                hm = cv2.applyColorMap(np.clip(amap / 6.0 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite(str(maps_dir / f"{r['id'].replace('/', '_')}.jpg"),
                            np.hstack([img, cv2.addWeighted(img, 0.5, hm, 0.5, 0)]))
        y, s = np.array(y), np.array(s)
        out[split] = {"image_auroc": auroc(y, s), "mean_ms_per_image_" + dev: round(float(np.mean(times[1:])), 1)}
        if split == "val":
            cands = np.unique(s)
            f1s = []
            for c in cands:
                m = img_metrics(y, s, c)
                p = m["defective_flagged"] / max(1, m["defective_flagged"] + m["normal_flagged"])
                r_ = m["sensitivity"]
                f1s.append(2 * p * r_ / (p + r_) if p + r_ else 0)
            thr = float(cands[int(np.argmax(f1s))])
            out["threshold_rule"] = "max image-level F1 on val"
        out[split]["at_val_threshold"] = img_metrics(y, s, thr)
        per_cat = {}
        for (rid, v) in scores.items():
            if v["split"] != split:
                continue
            c = per_cat.setdefault(v["category"], [0, 0])
            c[0] += int(v["score"] >= thr)
            c[1] += 1
        out[split]["flagged_by_category"] = {k: f"{a}/{b}" for k, a_b in per_cat.items() for a, b in [a_b]}
    res = ROOT / "artifacts" / "anomaly"
    res.mkdir(parents=True, exist_ok=True)
    (res / "patchcore_metrics.json").write_text(json.dumps(out, indent=2))
    (res / "patchcore_scores.json").write_text(json.dumps(scores, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
