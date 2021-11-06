from manga_ocr.typing import Rectangle


def test_rect_basic():

    rect = Rectangle.of_xywh(x=10, y=12, w=5, h=2)
    assert rect == (10, 12, 15, 14)

    rect = Rectangle.of_size((10, 20))
    assert rect == (0, 0, 10, 20)
    assert rect.center == (5, 10)
    assert rect.tl == (0, 0)
    assert rect.br == (10, 20)
    assert rect.size == (10, 20)


def test_rect_contains():

    rect = Rectangle.of_size((10, 20))
    assert (0, 0) in rect
    assert (-1, 0) not in rect
    assert (0, 21) not in rect

    assert (0, 0, 5, 5) in rect