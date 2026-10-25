import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.onnx_detector import letterbox, nms, postprocess, preprocess  # noqa: E402
from src.inspection.decision import DecisionPolicy  # noqa: E402


def test_letterbox_square_and_rectangular():
    out, info = letterbox(np.zeros((700, 700, 3), np.uint8), 640)
    assert out.shape == (640, 640, 3) and info.pad_x == info.pad_y == 0
    out, info = letterbox(np.zeros((480, 640, 3), np.uint8), 640)
    assert out.shape == (640, 640, 3) and info.pad_x == 0 and info.pad_y == 80


def test_preprocess_layout_and_colour_order():
    img = np.zeros((10, 10, 3), np.uint8)
    img[..., 0] = 255  # blue in BGR
    x, _ = preprocess(img, 32)
    assert x.shape == (1, 3, 32, 32) and x.dtype == np.float32
    assert x[0, 2].max() == pytest.approx(1.0) and x[0, 0].max() == 0  # becomes channel 2 (B) in RGB


def test_nms_suppresses_overlap_keeps_separate():
    b = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], float)
    keep = nms(b, np.array([0.9, 0.8, 0.7]), 0.5)
    assert sorted(keep.tolist()) == [0, 2]


def test_postprocess_round_trip_to_source_pixels():
    # one anchor predicting a 64x64 box centred at (320, 400) in a 640 letterbox of a 480x640 image
    raw = np.zeros((1, 5, 3), np.float32)
    raw[0, :, 0] = [320, 400, 64, 64, 0.9]
    _, info = letterbox(np.zeros((480, 640, 3), np.uint8), 640)
    det = postprocess(raw, info, conf=0.5, iou=0.7)
    assert det.shape == (1, 6)
    np.testing.assert_allclose(det[0, :4], [288, 288, 352, 352], atol=1e-4)  # y shifted by pad 80


def test_decision_policy():
    p = DecisionPolicy(reject_conf=0.11, review_conf=0.05)
    d = lambda s: np.array([[0, 0, 1, 1, s, 0]], np.float32)  # noqa: E731
    assert p.decide(np.zeros((0, 6))) == "PASS"
    assert p.decide(d(0.08)) == "REVIEW"
    # not 0.11 exactly: float32(0.11) == 0.10999999940 < 0.11, same comparison as in evaluation
    assert p.decide(d(0.12)) == "REJECT"
    assert DecisionPolicy(0.11, None).decide(d(0.08)) == "PASS"
