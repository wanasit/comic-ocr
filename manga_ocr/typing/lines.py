from __future__ import annotations

import json
from typing import Tuple, List
from manga_ocr.types import Rectangle, RectangleLike


class Line(tuple):

    def __new__(cls, located_line: Tuple[str, RectangleLike]):
        return tuple.__new__(Line, (located_line[0], Rectangle(located_line[1])))

    @staticmethod
    def of(text: str, at: RectangleLike) -> Line:
        return Line((text, at))

    @property
    def text(self) -> str:
        return self[0]

    @property
    def location(self) -> Rectangle:
        return self[1]

    @staticmethod
    def read_lines_from_file(filename) -> List[Line]:
        results = []
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                entry = json.loads(line)
                entry = Line(entry)
                results.append(entry)
        return results

    @staticmethod
    def save_lines_to_file(filename, lines: List[Line]):
        with open(filename, 'w') as f:
            for e in lines:
                f.write(json.dumps(e))
                f.write('\n')
