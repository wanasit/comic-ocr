from __future__ import annotations

from typing import Tuple, Union, List, Iterable


class Point(tuple):
    """
    A tuple of [x, y] with helper functions
    """

    def __new__(cls, point: PointLike):
        return tuple.__new__(Point, point)

    @staticmethod
    def of(x: int, y: int) -> Point:
        return Point((x, y))

    @property
    def x(self) -> int:
        return self[0]

    @property
    def y(self) -> int:
        return self[1]

    def move(self, diff_x: int, diff_y: int) -> Point:
        return Point.of(self.x + diff_x, self.y + diff_y)


class Size(tuple):
    """
    A tuple of [w, h] with helper functions
    """

    def __new__(cls, size: SizeLike):
        return tuple.__new__(Size, size)

    @staticmethod
    def of(w: int, h: int) -> Size:
        return Size((w, h))

    @property
    def width(self) -> int:
        return self[0]

    @property
    def height(self) -> int:
        return self[1]

    @property
    def value(self) -> int:
        return self[0] * self[1]


class Rectangle(tuple):
    """
    A tuple of [x0, y0, x1, y1] with helper functions to manipulate the shape.
    It is compatible with Pillow function that require rectangle or box.
    e.g. draw.rectangle(rect) or image.crop(rect)
    """

    def __new__(cls, rect: RectangleLike) -> Rectangle:
        return tuple.__new__(Rectangle, rect)

    @staticmethod
    def of_xywh(x: int, y: int, w: int, h: int) -> Rectangle:
        return Rectangle([x, y, x + w, y + h])

    @staticmethod
    def of_size(size: Tuple[int, int], at: Tuple[int, int] = (0, 0)) -> Rectangle:
        return Rectangle([at[0], at[1], at[0] + size[0], at[1] + size[1]])

    @staticmethod
    def of_tl_br(tl: Tuple[int, int], br: Tuple[int, int] = (0, 0)) -> Rectangle:
        return Rectangle([tl[0], tl[1], br[0], br[1]])

    @staticmethod
    def intersect_bounding_rect(rectangles: Iterable[RectangleLike]):
        top, left, bottom, right = -float('inf'), -float('inf'), float('inf'), float('inf')
        for rect in rectangles:
            top, left = max(top, rect[1]), max(left, rect[0])
            bottom, right = min(bottom, rect[3]), min(right, rect[2])

        if bottom <= top or right <= left:
            return None

        return Rectangle.of_tl_br(
            tl=(left, top),
            br=(right, bottom)
        )

    @staticmethod
    def union_bounding_rect(rectangles: Iterable[RectangleLike]):
        top, left, bottom, right = float('inf'), float('inf'), -float('inf'), -float('inf')
        for rect in rectangles:
            top, left = min(top, rect[1]), min(left, rect[0])
            bottom, right = max(bottom, rect[3]), max(right, rect[2])

        return Rectangle.of_tl_br(
            tl=(left, top),
            br=(right, bottom)
        )

    @staticmethod
    def is_overlap(rect_a: RectangleLike, rect_b: RectangleLike):
        overlap_x = rect_a[0] <= rect_b[0] < rect_a[2] or \
                    rect_b[0] <= rect_a[0] < rect_b[2]
        overlap_y = rect_a[1] <= rect_b[1] < rect_a[3] or \
                    rect_b[1] <= rect_a[1] < rect_b[3]

        return overlap_x and overlap_y

    @staticmethod
    def jaccard_similarity(rect_a: RectangleLike, rect_b: RectangleLike) -> float:
        if not Rectangle.is_overlap(rect_a, rect_b):
            return 0.0

        rect_a = Rectangle(rect_a)
        rect_b = Rectangle(rect_b)
        rect_intersect = Rectangle.intersect_bounding_rect((rect_a, rect_b))

        intersect_size = rect_intersect.size.value
        union_size = rect_a.size.value + rect_b.size.value - intersect_size
        return intersect_size / union_size

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom

    @property
    def left(self) -> int:
        return self[0]

    @property
    def top(self) -> int:
        return self[1]

    @property
    def right(self) -> int:
        return self[2]

    @property
    def bottom(self) -> int:
        return self[3]

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def size(self) -> Size:
        return Size((self.width, self.height))

    @property
    def center(self) -> Point:
        return Point(((self.left + self.right) // 2, (self.top + self.bottom) // 2))

    @property
    def tl(self) -> Point:
        return Point((self[0], self[1]))

    @property
    def br(self) -> Point:
        return Point((self[2], self[3]))

    def move(self, unit_x: int, unit_y: int) -> Rectangle:
        return Rectangle((self[0] + unit_x, self[1] + unit_y, self[2] + unit_x, self[3] + unit_y))

    def expand(self, unit: Union[int, Tuple]) -> Rectangle:
        unit_x = unit if isinstance(unit, int) else unit[0]
        unit_y = unit if isinstance(unit, int) else unit[1]
        return Rectangle((self[0] - unit_x, self[1] - unit_y, self[2] + unit_x, self[3] + unit_y))

    def close_to(self, rect: RectangleLike, threshold=0.7) -> bool:
        similarity = Rectangle.jaccard_similarity(self, rect)
        return similarity >= threshold

    def can_represent(self, rect: RectangleLike, threshold_precision=0.7, threshold_recall=0.8) -> bool:
        intersect_rect = Rectangle.intersect_bounding_rect((self, rect))
        if not intersect_rect:
            return False

        volume_precision = intersect_rect.size.value / self.size.value
        volume_recall = intersect_rect.size.value / rect.size.value
        return volume_precision >= threshold_precision and volume_recall >= threshold_recall

    def __contains__(self, item: Union[RectangleLike, PointLike]) -> bool:

        if isinstance(item, tuple) or isinstance(item, list):
            if len(item) == 2:  # rect contains point
                x, y = item
                return (self.left <= x <= self.right) and (self.top <= y <= self.bottom)

            if len(item) == 4:  # rect contains rect
                x1, y1, x2, y2 = item
                return (self.left <= x1 <= x2 <= self.right) and (self.top <= y1 <= y2 <= self.bottom)

        return tuple.__contains__(self, item)

    def __repr__(self):
        return f'Rect(size={self.width, self.height}, at={self[0], self[1]})'


PointLike = Union[
    Tuple[int, int],
    List[int],
    Point
]

SizeLike = Union[
    Tuple[int, int],
    List[int],
    Size
]

RectangleLike = Union[
    Tuple[int, int, int, int],
    List[int],
    Rectangle
]
