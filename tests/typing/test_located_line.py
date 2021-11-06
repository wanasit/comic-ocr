from manga_ocr.types import Line


def test_located_line_save_read(tmpdir):

    lines = [
        Line.of('test', [0, 0, 50, 20]),
        Line.of('test', [0, 25, 50, 45])
    ]

    filename = tmpdir.join("test.txt")
    Line.save_lines_to_file(filename, lines)

    saved_lines = Line.read_lines_from_file(filename)
    assert saved_lines == lines

