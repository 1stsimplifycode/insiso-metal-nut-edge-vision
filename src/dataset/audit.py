"""Dataset audit: counts, dimensions, mask validity, image/mask correspondence, duplicates."""
from __future__ import annotations

import hashlib
import itertools
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def dhash(img_gray: np.ndarray, size: int = 16) -> int:
    small = cv2.resize(img_gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def list_samples(cat_dir: Path) -> list[dict]:
    """One record per image: origin split, defect category, image path, mask path (or None)."""
    recs = []
    for split in ("train", "test"):
        for d in sorted((cat_dir / split).iterdir()):
            if not d.is_dir():
                continue
            for img in sorted(d.glob("*.png")):
                mask = None
                if d.name != "good":
                    mask = cat_dir / "ground_truth" / d.name / f"{img.stem}_mask.png"
                recs.append({
                    "id": f"{split}/{d.name}/{img.stem}",
                    "origin_split": split,
                    "category": d.name,
                    "image": img.relative_to(cat_dir).as_posix(),
                    "mask": mask.relative_to(cat_dir).as_posix() if mask else None,
                })
    return recs


def audit(cat_dir: Path, near_dup_threshold: int = 4) -> dict:
    recs = list_samples(cat_dir)
    problems: list[str] = []
    img_shapes, mask_shapes = Counter(), Counter()
    mask_values = Counter()
    sha = defaultdict(list)
    hashes = {}
    mask_area = defaultdict(list)

    for r in recs:
        p = cat_dir / r["image"]
        raw = p.read_bytes()
        sha[hashlib.sha256(raw).hexdigest()].append(r["id"])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            problems.append(f"unreadable image {r['image']}")
            continue
        img_shapes[f"{img.shape[1]}x{img.shape[0]}x{img.shape[2] if img.ndim == 3 else 1}"] += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        hashes[r["id"]] = dhash(gray)
        if r["mask"]:
            mp = cat_dir / r["mask"]
            if not mp.exists():
                problems.append(f"missing mask {r['mask']}")
                continue
            m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
            mask_shapes[f"{m.shape[1]}x{m.shape[0]}x{m.shape[2] if m.ndim == 3 else 1}"] += 1
            if m.shape[:2] != img.shape[:2]:
                problems.append(f"mask/image size mismatch {r['id']}")
            vals = np.unique(m)
            mask_values[tuple(int(v) for v in vals)] += 1
            fg = int((m > 127).sum())
            if fg == 0:
                problems.append(f"empty mask for anomalous image {r['id']}")
            mask_area[r["category"]].append(fg / (m.shape[0] * m.shape[1]))

    # orphan masks (mask without image)
    for mp in (cat_dir / "ground_truth").rglob("*_mask.png"):
        img = cat_dir / "test" / mp.parent.name / mp.name.replace("_mask", "")
        if not img.exists():
            problems.append(f"orphan mask {mp.relative_to(cat_dir).as_posix()}")

    exact_dups = [ids for ids in sha.values() if len(ids) > 1]
    ids = list(hashes)
    dists = []
    near = []
    for a, b in itertools.combinations(ids, 2):
        d = bin(hashes[a] ^ hashes[b]).count("1")
        dists.append(d)
        if d <= near_dup_threshold:
            near.append((a, b, d))

    per_cat = Counter(f"{r['origin_split']}/{r['category']}" for r in recs)
    n_anom = sum(1 for r in recs if r["category"] != "good")
    return {
        "records": recs,
        "summary": {
            "total_images": len(recs),
            "normal_images": len(recs) - n_anom,
            "anomalous_images": n_anom,
            "per_folder": dict(sorted(per_cat.items())),
            "defect_categories": sorted({r["category"] for r in recs} - {"good"}),
            "image_shapes": dict(img_shapes),
            "mask_shapes": dict(mask_shapes),
            "mask_value_sets": {str(k): v for k, v in mask_values.items()},
            "mask_area_fraction": {
                c: {"min": float(np.min(v)), "median": float(np.median(v)), "max": float(np.max(v))}
                for c, v in sorted(mask_area.items())
            },
            "problems": problems,
            "exact_duplicate_groups": exact_dups,
            "dhash_bits": 256,
            "dhash_pairwise_hamming": {
                "min": int(np.min(dists)), "p1": float(np.percentile(dists, 1)),
                "median": float(np.median(dists)),
            },
            "near_duplicate_threshold": near_dup_threshold,
            "near_duplicate_pairs": [list(x) for x in near],
            "imbalance_normal_to_anomalous": round((len(recs) - n_anom) / max(n_anom, 1), 2),
        },
    }
