from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Tuple, List, Union
from manga_ocr.typing import Rectangle, RectangleLike

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


class Paragraph(tuple):
    def __new__(cls, lines: Union[Tuple[Line], List[Line]]):
        return tuple.__new__(Line, lines)

    @staticmethod
    def of(lines: Union[Tuple[Tuple], List[Tuple]]) -> Paragraph:
        return Paragraph([Line(line) for line in lines])

    @property
    def lines(self) -> Tuple[Line]:
        return self

    @property
    def text(self) -> str:
        return self[0]

    @staticmethod
    def read_paragraphs_from_file(filename) -> List[Paragraph]:
        with open(filename, 'r') as f:
            json_paragraphs = json.load(f)
            return [Paragraph.of(p) for p in json_paragraphs]

    @staticmethod
    def save_paragraphs_to_file(filename, paragraphs: List[Paragraph]):
        with open(filename, 'w') as f:
            f.write(json.dumps(paragraphs))
            f.write('\n')