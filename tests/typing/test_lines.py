from manga_ocr.typing import Line, Paragraph


def test_paragraphs_save_read(tmpdir):
    paragraphs = [
        Paragraph([
            Line.of('test', [0, 0, 50, 20]),
            Line.of('test', [0, 25, 50, 45])
        ]),
        Paragraph([
            Line.of('test', [0, 100, 50, 20]),
        ])
    ]

    filename = tmpdir.join("test.json")
    Paragraph.save_paragraphs_to_file(filename, paragraphs)

    saved_paragraphs = Paragraph.read_paragraphs_from_file(filename)
    assert saved_paragraphs == paragraphs
