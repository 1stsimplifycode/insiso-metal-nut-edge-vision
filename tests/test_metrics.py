import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import average_precision, evaluate, iou_matrix, match_image  # noqa: E402


def test_iou_basic():
    assert iou_matrix([[0, 0, 10, 10]], [[0, 0, 10, 10]])[0, 0] == pytest.approx(1)
    assert iou_matrix([[0, 0, 10, 10]], [[5, 0, 15, 10]])[0, 0] == pytest.approx(50 / 150)
    assert iou_matrix([[0, 0, 1, 1]], [[2, 2, 3, 3]])[0, 0] == 0


def test_greedy_one_to_one():
    gt = np.array([[0, 0, 10, 10]])
    p = np.array([[0, 0, 10, 10], [0, 0, 10, 11]])
    tp, gidx, _ = match_image(p, np.array([0.5, 0.9]), gt, 0.5)
    assert tp.tolist() == [False, True]  # higher score claims the GT


def test_perfect_detector():
    ims = [{"id": "a", "category": "x", "gt": [[0, 0, 10, 10]], "pred": [[0, 0, 10, 10, 0.9]]},
           {"id": "b", "category": "good", "gt": [], "pred": []}]
    r = evaluate(ims, 0.5)
    assert r["mAP50"] == pytest.approx(1.0) and r["box"]["precision"] == 1 and r["box"]["recall"] == 1
    assert r["image"]["false_positive_rate"] == 0 and r["image"]["sensitivity"] == 1


def test_ap_half_recall():
    # precision 1 up to recall 0.5, never beyond -> 51 of 101 recall points
    assert average_precision(np.array([True]), np.array([0.9]), 2) == pytest.approx(51 / 101)


def test_ap_ranking_matters():
    # FP ranked above TP: precision 0.5 at recall 1
    assert average_precision(np.array([False, True]), np.array([0.9, 0.8]), 1) == pytest.approx(0.5)


def test_false_positive_on_normal_counts():
    ims = [{"id": "b", "category": "good", "gt": [], "pred": [[0, 0, 5, 5, 0.6], [1, 1, 4, 4, 0.2]]}]
    r = evaluate(ims, 0.5)
    assert r["fp_boxes_per_normal_image"] == 1 and r["image"]["normal_flagged"] == 1
