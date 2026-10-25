import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.annotations.mask_to_boxes import Box, mask_to_boxes, merge_boxes, validate_yolo_line  # noqa: E402


def blank(h=100, w=200):
    return np.zeros((h, w), np.uint8)


def test_empty_mask_gives_no_boxes():
    assert mask_to_boxes(blank()) == []


def test_single_rectangle_is_tight():
    m = blank()
    m[10:20, 30:45] = 255
    (b,) = mask_to_boxes(m)
    assert (b.x1, b.y1, b.x2, b.y2, b.area_px) == (30, 10, 45, 20, 150)
    cx, cy, w, h = b.yolo(200, 100)
    assert (cx, cy, w, h) == pytest.approx((37.5 / 200, 15 / 100, 15 / 200, 10 / 100))


def test_disconnected_regions_give_separate_boxes():
    m = blank()
    m[5:10, 5:10] = 255
    m[50:60, 100:150] = 255
    assert len(mask_to_boxes(m)) == 2


def test_diagonal_pixels_are_one_component_with_8_connectivity():
    m = blank()
    m[10, 10] = m[11, 11] = m[12, 12] = 255
    (b,) = mask_to_boxes(m)
    assert (b.x1, b.y1, b.x2, b.y2) == (10, 10, 13, 13)


def test_single_pixel_kept_without_min_area():
    m = blank()
    m[0, 0] = 255
    (b,) = mask_to_boxes(m)
    assert b.w == b.h == 1


def test_min_area_filters_and_is_explicit():
    m = blank()
    m[0, 0] = 255
    m[20:30, 20:30] = 255
    assert len(mask_to_boxes(m, min_area=2)) == 1


def test_non_binary_mask_threshold():
    m = blank()
    m[0:5, 0:5] = 100  # below 127 -> background
    m[50:55, 50:55] = 200
    (b,) = mask_to_boxes(m)
    assert b.x1 == 50


def test_box_touching_border_is_valid_yolo():
    m = blank()
    m[90:100, 190:200] = 255
    (b,) = mask_to_boxes(m)
    validate_yolo_line(0, *b.yolo(200, 100))


def test_merge_policy_and_union():
    m = blank()
    m[10:20, 10:20] = 255
    m[10:20, 25:35] = 255  # 5 px gap
    m[80:90, 180:190] = 255
    assert len(mask_to_boxes(m, "merged", merge_gap=0)) == 3
    assert len(mask_to_boxes(m, "merged", merge_gap=6)) == 2
    (u,) = mask_to_boxes(m, "union")
    assert (u.x1, u.y1, u.x2, u.y2) == (10, 10, 190, 90)


def test_merge_is_transitive():
    bs = [Box(0, 0, 10, 10, 1), Box(12, 0, 22, 10, 1), Box(24, 0, 34, 10, 1)]
    (m,) = merge_boxes(bs, gap=3)
    assert (m.x1, m.x2) == (0, 34)


@pytest.mark.parametrize("vals", [(1.1, .5, .1, .1), (.5, .5, 0, .1), (.02, .5, .1, .1)])
def test_invalid_yolo_rejected(vals):
    with pytest.raises(ValueError):
        validate_yolo_line(0, *vals)
