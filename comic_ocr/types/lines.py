from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Tuple, List, Union, Optional
from comic_ocr.types import Rectangle, RectangleLike


class Line(tuple):

    def __new__(cls, located_line: Tuple[str, RectangleLike]):
        assert len(located_line) >= 2
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


class Paragraph:
    def __init__(self,
                 lines: Union[Tuple[Line], List[Line]],
                 location: Optional[Rectangle] = None
                 ):
        self._lines = lines
        self._location = location if location else Rectangle.union_bounding_rect([l.location for l in lines])

    @property
    def location(self) -> Rectangle:
        return self._location

    @property
    def lines(self) -> List[Line]:
        return self._lines

    @property
    def text(self) -> str:
        return ' '.join((l.text for l in self._lines))

    def __repr__(self):
        return f'Paragraph("{self.text}", at={self._location})'