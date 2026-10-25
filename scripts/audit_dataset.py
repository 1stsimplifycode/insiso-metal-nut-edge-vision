"""Audit the extracted dataset and write reports/dataset_audit.{json,md}."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.audit import audit  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat-dir", default=str(ROOT / "data" / "raw" / "metal_nut"))
    ap.add_argument("--out", default=str(ROOT / "reports"))
    a = ap.parse_args()
    res = audit(Path(a.cat_dir))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_audit.json").write_text(json.dumps(res["summary"], indent=2))
    s = res["summary"]
    lines = ["# Dataset audit — MVTec AD metal_nut", "",
             f"- Total images: **{s['total_images']}** (normal {s['normal_images']}, anomalous {s['anomalous_images']})",
             f"- Normal:anomalous ratio: {s['imbalance_normal_to_anomalous']}:1",
             f"- Defect categories: {', '.join(s['defect_categories'])}",
             f"- Image shapes: {s['image_shapes']}", f"- Mask shapes: {s['mask_shapes']}",
             f"- Mask pixel-value sets: {s['mask_value_sets']}",
             f"- Problems (missing/invalid/orphan/size mismatch): {len(s['problems'])} {s['problems'][:10]}",
             f"- Exact duplicate groups (SHA-256): {len(s['exact_duplicate_groups'])}",
             f"- 256-bit dHash pairwise Hamming distance: {s['dhash_pairwise_hamming']}",
             f"- Near-duplicate pairs (Hamming <= {s['near_duplicate_threshold']}): {len(s['near_duplicate_pairs'])}",
             "", "| Folder | Images |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in s["per_folder"].items()]
    lines += ["", "| Category | mask area min | median | max (fraction of image) |", "|---|---|---|---|"]
    lines += [f"| {c} | {v['min']:.4f} | {v['median']:.4f} | {v['max']:.4f} |" for c, v in s["mask_area_fraction"].items()]
    (out / "dataset_audit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
