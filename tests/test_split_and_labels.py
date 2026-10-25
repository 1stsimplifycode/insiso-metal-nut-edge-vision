"""Checks on the real prepared data (skipped if the dataset has not been prepared)."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.split import read_manifest, verify_frozen  # noqa: E402

MANIFEST = ROOT / "manifests" / "split_seed42.csv"
YOLO = ROOT / "data" / "yolo" / "baseline"
RAW = ROOT / "data" / "raw" / "metal_nut"

need = pytest.mark.skipif(not (MANIFEST.exists() and YOLO.exists() and RAW.exists()), reason="data not prepared")


@need
def test_manifest_frozen_and_disjoint():
    verify_frozen(MANIFEST)
    rows = read_manifest(MANIFEST)
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)) == 335
    by_split = {}
    for r in rows:
        by_split.setdefault(r["split"], set()).add(r["image"])
    s = list(by_split.values())
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            assert not (s[i] & s[j])


@need
def test_every_category_in_every_split():
    rows = read_manifest(MANIFEST)
    for split in ("train", "val", "test"):
        cats = {r["category"] for r in rows if r["split"] == split}
        assert cats == {"good", "bent", "color", "flip", "scratch"}


@need
def test_labels_match_images_and_normals_empty():
    rows = {r["id"].replace("/", "_"): r for r in read_manifest(MANIFEST)}
    for split in ("train", "val", "test"):
        imgs = {p.stem for p in (YOLO / "images" / split).glob("*.png")}
        lbls = {p.stem for p in (YOLO / "labels" / split).glob("*.txt")}
        assert imgs == lbls
        for stem in lbls:
            txt = (YOLO / "labels" / split / f"{stem}.txt").read_text().strip()
            assert bool(txt) == (rows[stem]["category"] != "good")


@need
def test_boxes_cover_all_mask_pixels():
    """Every foreground mask pixel must lie inside at least one generated box."""
    for lbl in (YOLO / "labels").rglob("*.txt"):
        txt = lbl.read_text().split("\n")
        txt = [t for t in txt if t]
        if not txt:
            continue
        split, cat, idx = lbl.stem.split("_")
        mask = cv2.imread(str(RAW / "ground_truth" / cat / f"{idx}_mask.png"), 0) > 127
        cover = np.zeros_like(mask)
        h, w = mask.shape
        for line in txt:
            _, cx, cy, bw, bh = map(float, line.split())
            x1, x2 = round((cx - bw / 2) * w), round((cx + bw / 2) * w)
            y1, y2 = round((cy - bh / 2) * h), round((cy + bh / 2) * h)
            cover[y1:y2, x1:x2] = True
        assert not (mask & ~cover).any(), lbl.name
