import math
from dataclasses import dataclass
from typing import List, Union

from PIL import Image
from manga_ocr.typing import Rectangle, Size


def divine_rect_into_overlapping_tiles(rect: Union[Rectangle, Size], tile_size: Size, min_overlap_x: int, min_overlap_y: int):

    tile_count_x = math.ceil((rect.width - min_overlap_x) / (tile_size.width - min_overlap_x))
    overlap_size_x = math.ceil((tile_count_x * tile_size.width - rect.width) / (tile_count_x - 1)) if tile_count_x > 1 else 0

    tile_count_y = math.ceil((rect.height - min_overlap_y) / (tile_size.height - min_overlap_y))
    overlap_size_y = math.ceil((tile_count_y * tile_size.height - rect.height) / (tile_count_y - 1)) if tile_count_y > 1 else 0

    offset_x = rect[0] if len(rect) == 4 else 0
    offset_y = rect[1] if len(rect) == 4 else 0

    for i in range(tile_count_y):
        for j in range(tile_count_x):
            y = offset_y + i * (tile_size.height - overlap_size_y)
            x = offset_x + j * (tile_size.width - overlap_size_x)
            yield Rectangle.of_size(tile_size, at=(x, y))


