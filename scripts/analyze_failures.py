"""Build the failure gallery from saved predictions at the val-selected operating point.

    python scripts/analyze_failures.py --name baseline --split test
Outputs reports/failures/<name>_<split>/ (panels + failures.md + failures.json) and a
threshold-sensitivity table.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.failures import classify_errors, region_contrast, render_case  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402

RAW = ROOT / "data" / "raw" / "metal_nut"


def mask_for(img_id: str) -> Path | None:
    origin, cat, idx = img_id.split("_")
    return None if cat == "good" else RAW / "ground_truth" / cat / f"{idx}_mask.png"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    ev = ROOT / "artifacts" / "eval"
    ims = json.loads((ev / f"{a.name}_{a.split}_predictions.json").read_text())
    conf = json.loads((ev / f"{a.name}_threshold.json").read_text())["conf"]
    by_id = {im["id"]: im for im in ims}

    out = ROOT / "reports" / "failures" / f"{a.name}_{a.split}"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.jpg"):
        f.unlink()
    cases = classify_errors(ims, conf)
    for k, c in enumerate(cases):
        im = by_id[c["id"]]
        box = c.get("gt_box") or c.get("pred_box")
        img = cv2.imread(im["image"])
        c["box_wh_px"] = [round(box[2] - box[0]), round(box[3] - box[1])]
        c["region_contrast_gray"] = round(region_contrast(img, box), 1)
        c["n_gt_boxes_in_image"] = len(im["gt"])
        title = (f"#{k:02d} {c['id']} | {c['kind']} {c['error_type']} | conf {c['conf']:.2f} | "
                 f"IoU {c['best_iou_at_op']:.2f} | box {c['box_wh_px'][0]}x{c['box_wh_px'][1]}px | thr {conf}")
        c["panel"] = f"{k:02d}_{c['id']}_{c['kind']}.jpg"
        cv2.imwrite(str(out / c["panel"]), render_case(im, conf, mask_for(c["id"]), box, title),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])

    # image-level failures
    img_fail = []
    for im in ims:
        p = [b for b in im["pred"] if b[4] >= conf]
        if im["category"] != "good" and not p:
            img_fail.append({"id": im["id"], "type": "defective part PASSED (missed)",
                             "max_score": max([b[4] for b in im["pred"]], default=0)})
        if im["category"] == "good" and p:
            img_fail.append({"id": im["id"], "type": "good part REJECTED (false alarm)",
                             "max_score": max(b[4] for b in p)})

    sens = []
    for c in np.round(np.arange(0.05, 0.96, 0.05), 2):
        r = evaluate(ims, float(c))
        sens.append({"conf": float(c), "box_P": r["box"]["precision"], "box_R": r["box"]["recall"],
                     "img_sens": f"{r['image']['defective_flagged']}/{r['image']['defective_total']}",
                     "img_fpr": f"{r['image']['normal_flagged']}/{r['image']['normal_total']}"})

    (out / "failures.json").write_text(json.dumps({"conf": conf, "cases": cases, "image_level": img_fail,
                                                    "threshold_sensitivity": sens}, indent=2))
    cnt = Counter((c["kind"], c["error_type"]) for c in cases)
    md = [f"# Failure gallery — {a.name} on {a.split} (conf ≥ {conf}, IoU 0.5)", "",
          "Auto-generated from saved predictions. Colours: green = ground truth (mask tinted), red = prediction "
          "≥ threshold, orange = prediction below threshold (≥ 0.05), magenta = the box this case is about.", "",
          f"Total box-level errors: **{len(cases)}** — " + ", ".join(f"{k[0]}/{k[1]}: {v}" for k, v in sorted(cnt.items())),
          f"Image-level errors: **{len(img_fail)}**", ""]
    md += ["| # | image | kind | error type | category | conf | IoU | box (px) | contrast | panel |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for k, c in enumerate(cases):
        md.append(f"| {k} | {c['id']} | {c['kind']} | {c['error_type']} | {c['category']} | {c['conf']:.2f} | "
                  f"{c['best_iou_at_op']:.2f} | {c['box_wh_px'][0]}×{c['box_wh_px'][1]} | {c['region_contrast_gray']} | "
                  f"![]({c['panel']}) |")
    md += ["", "## Image-level errors", "", "| image | type | max score |", "|---|---|---|"]
    md += [f"| {f['id']} | {f['type']} | {f['max_score']:.3f} |" for f in img_fail]
    md += ["", "## Threshold sensitivity (reported only; threshold was fixed on val)", "",
           "| conf | box P | box R | defective flagged | normal flagged |", "|---|---|---|---|---|"]
    md += [f"| {s['conf']} | {s['box_P']:.3f} | {s['box_R']:.3f} | {s['img_sens']} | {s['img_fpr']} |" for s in sens]
    (out / "failures.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:6]))
    for f in img_fail:
        print(f)


if __name__ == "__main__":
    main()
