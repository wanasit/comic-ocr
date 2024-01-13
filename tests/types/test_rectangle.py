from comic_ocr.types import Rectangle


def test_rect_basic():
    rect = Rectangle.of_xywh(x=10, y=12, w=5, h=2)
    assert rect == (10, 12, 15, 14)

    rect = Rectangle.of_size((10, 20))
    assert rect == (0, 0, 10, 20)
    assert rect.center == (5, 10)
    assert rect.tl == (0, 0)
    assert rect.br == (10, 20)
    assert rect.size == (10, 20)


def test_rect_expand():
    rect = Rectangle.of_xywh(x=10, y=12, w=5, h=2)
    assert rect == (10, 12, 15, 14)

    expanded_rect = rect.expand(2)
    assert expanded_rect == (8, 10, 17, 16)

    expanded_rect = rect.expand((1, 2))
    assert expanded_rect == (9, 10, 16, 16)


def test_rect_contains():
    rect = Rectangle.of_size((10, 20))
    assert (0, 0) in rect
    assert (-1, 0) not in rect
    assert (0, 21) not in rect

    assert (0, 0, 5, 5) in rect


def test_rect_is_overlap():
    rect_a = Rectangle.of_size((10, 20))
    assert Rectangle.is_overlap(rect_a, rect_a)

    rect_b = Rectangle.of_size((5, 5), at=(10, 20))
    assert not Rectangle.is_overlap(rect_a, rect_b)
    assert not Rectangle.is_overlap(rect_b, rect_a)

    rect_c = Rectangle.of_size((5, 5), at=(5, 10))
    assert Rectangle.is_overlap(rect_a, rect_c)
    assert Rectangle.is_overlap(rect_c, rect_a)


def test_rect_jaccard_similarity():
    rect_a = Rectangle.of_size((10, 20))
    assert Rectangle.jaccard_similarity(rect_a, rect_a) == 1.0

    rect_b = Rectangle.of_size((10, 20), at=(30, 40))
    assert Rectangle.jaccard_similarity(rect_a, rect_b) == 0.0
    assert Rectangle.jaccard_similarity(rect_b, rect_a) == 0.0

    rect_c = Rectangle.of_size((10, 10))
    assert Rectangle.jaccard_similarity(rect_a, rect_c) == 0.5
    assert Rectangle.jaccard_similarity(rect_c, rect_a) == 0.5


def test_can_represent():
    rect_larger = Rectangle.of_size((63, 18), at=(379, 705))
    rect_smaller_inside = Rectangle.of_size((55, 13), at=(382, 707))
    assert rect_larger.can_represent(rect_smaller_inside)

    rect_larger = Rectangle.of_size((61, 18), at=(379, 719))
    rect_smaller_inside = Rectangle.of_size((53, 16), at=(384, 720))
    assert rect_larger.can_represent(rect_smaller_inside)


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

    tp, fp, fn = Rectangle.match(
        [rect_a, rect_b, rect_c], [rect_baseline_a, rect_baseline_b, rect_baseline_c])

    assert len(tp) == 3
    assert len(fp) == 0
    assert len(fn) == 0

    tp, fp, fn = Rectangle.match(
        [rect_a, rect_c, rect_b], [rect_baseline_a, rect_baseline_b, rect_baseline_c])
    assert len(tp) == 3
    assert len(fp) == 0
    assert len(fn) == 0

    tp, fp, fn = Rectangle.match(
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

    tp, fp, fn = Rectangle.match(
        [rect_a, rect_b], [rect_baseline_a, rect_baseline_b, rect_baseline_c])

    assert tp == [(0, 0)]
    assert fp == [1]
    assert fn == [1, 2]
