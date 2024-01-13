# Note: Line-types depends on Shape-types
import dataclasses

from comic_ocr.types.shapes import *
from comic_ocr.types.lines import *

# Note: Image-types depends on File-types
from comic_ocr.types.files import *
from comic_ocr.types.images import *


@dataclasses.dataclass
class Percentage(float):
    value: int
    max_value: int

    def __new__(cls, value: int, max_value: int):
        return float.__new__(cls, value / max_value)

    @staticmethod
    def of(x: int, to: int = 100) -> Union['Percentage', None]:
        if to <= 0:
            return None
        return Percentage(x, to)

    def __str__(self):
        return f'{(self.value / self.max_value * 100):.2f}% ({self.value}/{self.max_value})'

    def __repr__(self):
        return f'Percentage({self.value}, {self.max_value})'
