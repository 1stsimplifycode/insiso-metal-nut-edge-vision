"""Create frozen split manifests, convert masks to boxes and build YOLO datasets.

    python scripts/prepare_data.py                       # main split + baseline labels + QA images
    python scripts/prepare_data.py --policy merged --merge-gap 20 --name merged20
    python scripts/prepare_data.py --exclude-category scratch --name loco_scratch
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.annotations.build_yolo import build, qa_panel  # noqa: E402
from src.annotations.mask_to_boxes import mask_to_boxes  # noqa: E402
from src.dataset.audit import list_samples  # noqa: E402
from src.dataset.split import make_split, read_manifest, verify_frozen, write_manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cat-dir", default=str(ROOT / "data" / "raw" / "metal_nut"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--policy", default="components", choices=["components", "merged", "union"])
    ap.add_argument("--merge-gap", type=int, default=0)
    ap.add_argument("--min-area", type=int, default=0)
    ap.add_argument("--multiclass", action="store_true")
    ap.add_argument("--exclude-category", default=None, help="leave-one-defect-category-out")
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--qa", action="store_true", help="write annotation QA panels to reports/annotation_qa")
    a = ap.parse_args()

    cat_dir = Path(a.cat_dir)
    mdir = ROOT / "manifests"
    mname = f"split_seed{a.seed}" + (f"_loco_{a.exclude_category}" if a.exclude_category else "") + ".csv"
    mpath = mdir / mname
    if mpath.exists():
        verify_frozen(mpath)  # never silently regenerate a frozen split
        rows = read_manifest(mpath)
        print(f"using frozen manifest {mpath.name}")
    else:
        rows = make_split(list_samples(cat_dir), seed=a.seed, exclude_category=a.exclude_category)
        digest = write_manifest(rows, mpath)
        print(f"froze new manifest {mpath.name} sha256={digest[:16]}")

    # leakage check: each source image appears in exactly one split
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate source ids in manifest"
    counts = {}
    for r in rows:
        counts.setdefault(r["split"], {}).setdefault(r["category"], 0)
        counts[r["split"]][r["category"]] += 1
    print(json.dumps(counts, indent=1))

    out = ROOT / "data" / "yolo" / a.name
    stats = build(rows, cat_dir, out, policy=a.policy, merge_gap=a.merge_gap, min_area=a.min_area,
                  multiclass=a.multiclass)
    stats["manifest"] = mpath.name
    stats["split_counts"] = counts
    (out / "build_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "split_counts"}, indent=1))

    if a.qa:
        qa = ROOT / "reports" / "annotation_qa"
        qa.mkdir(parents=True, exist_ok=True)
        # every multi-component mask + first two of each category
        seen = {}
        for r in rows:
            if not r["mask"]:
                continue
            mask = cv2.imread(str(cat_dir / r["mask"]), cv2.IMREAD_GRAYSCALE)
            boxes = mask_to_boxes(mask, policy=a.policy, merge_gap=a.merge_gap, min_area=a.min_area)
            n_cc = len(mask_to_boxes(mask, "components"))
            seen[r["category"]] = seen.get(r["category"], 0) + 1
            if n_cc > 1 or seen[r["category"]] <= 2:
                img = cv2.imread(str(cat_dir / r["image"]))
                cv2.imwrite(str(qa / f"{a.name}_{r['id'].replace('/', '_')}.jpg"),
                            qa_panel(img, mask, boxes, f"{r['id']} [{a.policy}]"), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"QA panels in {qa}")


if __name__ == "__main__":
    main()
