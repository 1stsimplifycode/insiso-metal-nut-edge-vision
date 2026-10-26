"""Evaluate a model on val (chooses + saves the operating threshold) or test (uses the saved one).

    python scripts/evaluate.py --model models/baseline.pt --split val
    python scripts/evaluate.py --model models/baseline.pt --split test
The test split refuses to run without a threshold previously selected on val.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import best_f1_threshold, evaluate, recall_by_size, zero_false_reject_threshold  # noqa: E402
from src.evaluation.predict import collect  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default=None, help="results name (default: model file stem)")
    ap.add_argument("--dataset", default="baseline", help="data/yolo/<dataset> to read images from")
    ap.add_argument("--gt-dataset", default="baseline", help="data/yolo/<gt-dataset> labels used as reference")
    ap.add_argument("--split", required=True, choices=["val", "test", "unseen"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=None, help="override operating threshold (reported as such)")
    ap.add_argument("--threshold-from", default=None, help="name whose val threshold to use (default: --name)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threshold-rule", default="max_f1", choices=["max_f1", "zero_fpr"],
                    help="val-only rule for the operating point")
    a = ap.parse_args()

    name = a.name or Path(a.model).stem
    res_dir = ROOT / "artifacts" / "eval"
    res_dir.mkdir(parents=True, exist_ok=True)
    thr_file = res_dir / f"{a.threshold_from or name}_threshold.json"

    ims = collect(a.model, ROOT / "data" / "yolo" / a.dataset, a.split, a.imgsz,
                  gt_dataset_dir=ROOT / "data" / "yolo" / a.gt_dataset, device=a.device)
    if a.conf is not None:
        conf, source = a.conf, "manual override"
    elif a.split == "val":
        if a.threshold_rule == "max_f1":
            conf, f1 = best_f1_threshold(ims)
            thr_file.write_text(json.dumps({"conf": conf, "val_box_f1": f1, "rule": "max box-F1 on val"}, indent=2))
            source = "selected on val (max box-F1)"
        else:
            conf, sens = zero_false_reject_threshold(ims)
            thr_file.write_text(json.dumps({"conf": conf, "val_image_sensitivity": sens,
                                            "rule": "lowest conf with zero val false rejects"}, indent=2))
            source = "selected on val (lowest conf with zero false rejects)"
    else:
        if not thr_file.exists():
            sys.exit(f"no val-selected threshold at {thr_file}; run --split val first")
        conf, source = json.loads(thr_file.read_text())["conf"], f"from {thr_file.name} (val)"

    r = evaluate(ims, conf)
    r["recall_by_size"] = recall_by_size(ims, conf)
    r.update({"model": a.model, "split": a.split, "threshold_source": source, "imgsz": a.imgsz,
              "gt_reference": a.gt_dataset})
    tag = f"{name}_{a.split}" + ("_manualconf" if a.conf is not None else "")
    (res_dir / f"{tag}_metrics.json").write_text(json.dumps(r, indent=2))
    (res_dir / f"{tag}_predictions.json").write_text(json.dumps(ims))
    b, i = r["box"], r["image"]
    print(f"[{tag}] conf={conf} ({source})  images={r['n_images']} (defective {r['n_anomalous_images']}, "
          f"normal {r['n_normal_images']}), GT boxes={r['n_gt_boxes']}")
    print(f"  box  P={b['precision']:.3f} R={b['recall']:.3f} F1={b['f1']:.3f} (TP {b['tp']} FP {b['fp']} FN {b['fn']})"
          f"  mAP50={r['mAP50']:.3f} mAP50-95={r['mAP50_95']:.3f}")
    fpn = r["fp_boxes_per_normal_image"]  # None when the split has no normal images (e.g. `unseen`)
    print(f"  image sensitivity {i['defective_flagged']}/{i['defective_total']}  "
          f"FPR {i['normal_flagged']}/{i['normal_total']}  FP boxes/normal img "
          + ("n/a" if fpn is None else f"{fpn:.3f}"))
    for c, v in sorted(r["per_category"].items()):
        print(f"  {c:8s} box recall {v['tp_boxes']}/{v['gt_boxes']}  images flagged {v['flagged']}/{v['images']}")
    print(f"  recall by size: { {k: f'{v['tp']}/{v['gt']}' for k, v in r['recall_by_size'].items()} }")


if __name__ == "__main__":
    main()
