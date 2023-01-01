from comic_ocr.typing import Rectangle
from comic_ocr.models import localization


def test_match_with_near_perfect_fits():
    rect_baseline_a = Rectangle.of_xywh(10, 10, 100, 10)
    rect_a = Rectangle.of_xywh(10, 10, 100, 10)
    assert rect_a.can_represent(rect_baseline_a)

    rect_baseline_b = Rectangle.of_xywh(85, 100, 100, 10)
    rect_b = Rectangle.of_xywh(85, 100, 101, 9)
    assert rect_b.can_represent(rect_baseline_b)

    rect_baseline_c = Rectangle.of_xywh(10, 50, 100, 10)
    rect_c = Rectangle.of_xywh(11, 51, 101, 9)
    assert rect_c.can_represent(rect_baseline_c)

    tp, fp, fn = localization.match_location_rectangles_with_baseline(
        [rect_a, rect_b, rect_c], [rect_baseline_a, rect_baseline_b, rect_baseline_c])

    assert len(tp) == 3
    assert len(fp) == 0
    assert len(fn) == 0

    tp, fp, fn = localization.match_location_rectangles_with_baseline(
        [rect_a, rect_c, rect_b], [rect_baseline_a, rect_baseline_b, rect_baseline_c])
    assert len(tp) == 3
    assert len(fp) == 0
    assert len(fn) == 0

    tp, fp, fn = localization.match_location_rectangles_with_baseline(
        [rect_b, rect_a, rect_c], [rect_baseline_a, rect_baseline_b, rect_baseline_c])
    assert len(tp) == 3
    assert len(fp) == 0
    assert len(fn) == 0


def test_match_with_miss_matches():
    rect_baseline_a = Rectangle.of_xywh(10, 10, 100, 10)
    rect_a = Rectangle.of_xywh(10, 10, 100, 10)
    assert rect_a.can_represent(rect_baseline_a)

    rect_baseline_b = Rectangle.of_xywh(85, 100, 100, 10)
    rect_b = Rectangle.of_xywh(120, 100, 101, 9)
    assert not rect_b.can_represent(rect_baseline_b)

    rect_baseline_c = Rectangle.of_xywh(10, 50, 100, 10)

    tp, fp, fn = localization.match_location_rectangles_with_baseline(
        [rect_a, rect_b], [rect_baseline_a, rect_baseline_b, rect_baseline_c])

    assert tp == [(rect_a, rect_baseline_a)]
    assert fp == [rect_b]
    assert fn == [rect_baseline_b, rect_baseline_c]
