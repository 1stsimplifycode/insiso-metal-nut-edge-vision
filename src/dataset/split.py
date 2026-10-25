"""Deterministic, stratified, source-image-level train/val/test split.

Adaptation of MVTec AD (an unsupervised benchmark) to supervised detection:
anomalous images only exist in the original `test/`, so they are re-partitioned
per defect category. Normal images from original `train/good` and `test/good`
are pooled and re-partitioned (stratified by origin folder).
"""
from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path

FRACTIONS = {"train": 0.5, "val": 0.2, "test": 0.3}  # anomalous images, per category
NORMAL_COUNTS = {"val": 30, "test": 40}                # remaining normals -> train


def _split_group(items: list, frac: dict, rng: random.Random) -> dict:
    items = sorted(items, key=lambda r: r["id"])
    rng.shuffle(items)
    n = len(items)
    n_test = round(n * frac["test"])
    n_val = round(n * frac["val"])
    return {"test": items[:n_test], "val": items[n_test:n_test + n_val], "train": items[n_test + n_val:]}


def make_split(records: list[dict], seed: int = 42, exclude_category: str | None = None) -> list[dict]:
    """Return records with a `split` key. With `exclude_category`, that category is moved
    entirely to a separate `unseen` split (leave-one-defect-category-out)."""
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(dict(r))
    out = []
    for cat in sorted(c for c in by_cat if c != "good"):
        groups = _split_group(by_cat[cat], FRACTIONS, rng)  # always consume RNG -> identical other splits
        if cat == exclude_category:
            for r in by_cat[cat]:
                r["split"] = "unseen"
                out.append(r)
            continue
        for split, rs in groups.items():
            for r in rs:
                r["split"] = split
                out.append(r)
    # normals: stratify by origin folder proportionally
    normals = by_cat["good"]
    by_origin = defaultdict(list)
    for r in normals:
        by_origin[r["origin_split"]].append(r)
    total = len(normals)
    for origin in sorted(by_origin):
        group = sorted(by_origin[origin], key=lambda r: r["id"])
        rng.shuffle(group)
        share = len(group) / total
        n_test = round(NORMAL_COUNTS["test"] * share)
        n_val = round(NORMAL_COUNTS["val"] * share)
        for i, r in enumerate(group):
            r["split"] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            out.append(r)
    return sorted(out, key=lambda r: r["id"])


def write_manifest(rows: list[dict], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["id", "split", "category", "origin_split", "image", "mask"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "mask": r["mask"] or ""})
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n")
    return digest


def read_manifest(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["mask"] = r["mask"] or None
    return rows


def verify_frozen(path: Path) -> None:
    expected = path.with_suffix(".sha256").read_text().split()[0]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected != actual:
        raise RuntimeError(f"manifest {path} changed since it was frozen ({actual[:12]} != {expected[:12]})")
