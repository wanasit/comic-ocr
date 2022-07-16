from manga_ocr.typing import Line, Paragraph, Rectangle


def test_line_basic():
    line = Line.of('Hello', at=Rectangle.of_xywh(0, 0, 30, 8))
    assert line.text == 'Hello'
    assert line.location == (0, 0, 30, 8)


def test_paragraph_basic():
    line = Line.of('Hello', at=Rectangle.of_xywh(0, 0, 30, 8))
    paragraph = Paragraph(lines=[line])

    assert paragraph.text == 'Hello'
    assert paragraph.location == (0, 0, 30, 8)
    assert paragraph.lines == [line]
