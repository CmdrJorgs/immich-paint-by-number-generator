"""A4 page geometry, in millimetres, converted to points only at the edge."""

from __future__ import annotations

from dataclasses import dataclass

MM_PER_INCH = 25.4
POINTS_PER_INCH = 72.0

A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0


def mm_to_points(mm: float) -> float:
    return mm / MM_PER_INCH * POINTS_PER_INCH


def mm_to_pixels(mm: float, dpi: float) -> int:
    return max(1, int(round(mm / MM_PER_INCH * dpi)))


@dataclass(frozen=True)
class PageLayout:
    """Where the drawing, the legend and the caption sit on one A4 sheet."""

    landscape: bool = False
    margin_mm: float = 12.0
    legend_height_mm: float = 0.0
    caption_height_mm: float = 6.0

    @property
    def width_mm(self) -> float:
        return A4_HEIGHT_MM if self.landscape else A4_WIDTH_MM

    @property
    def height_mm(self) -> float:
        return A4_WIDTH_MM if self.landscape else A4_HEIGHT_MM

    @property
    def size_points(self) -> tuple[float, float]:
        return (mm_to_points(self.width_mm), mm_to_points(self.height_mm))

    @property
    def drawing_width_mm(self) -> float:
        return self.width_mm - 2 * self.margin_mm

    @property
    def drawing_height_mm(self) -> float:
        return (
            self.height_mm
            - 2 * self.margin_mm
            - self.legend_height_mm
            - self.caption_height_mm
        )

    @property
    def drawing_origin_mm(self) -> tuple[float, float]:
        """Bottom-left of the drawing area, in PDF coordinates (y grows up)."""
        return (self.margin_mm, self.margin_mm + self.caption_height_mm + self.legend_height_mm)

    def fitted_drawing_mm(self, aspect: float) -> tuple[float, float]:
        """Largest width x height with the given aspect that fits the drawing area."""
        available_w, available_h = self.drawing_width_mm, self.drawing_height_mm
        if available_w <= 0 or available_h <= 0:
            raise ValueError(
                "no room left for the drawing; reduce the margin or the legend height"
            )
        width = available_w
        height = width / aspect
        if height > available_h:
            height = available_h
            width = height * aspect
        return (width, height)

    def orientation_for(self, aspect: float) -> PageLayout:
        """Pick the page orientation that wastes the least paper."""
        portrait = PageLayout(False, self.margin_mm, self.legend_height_mm, self.caption_height_mm)
        landscape = PageLayout(True, self.margin_mm, self.legend_height_mm, self.caption_height_mm)
        p_w, p_h = portrait.fitted_drawing_mm(aspect)
        l_w, l_h = landscape.fitted_drawing_mm(aspect)
        return landscape if (l_w * l_h) > (p_w * p_h) else portrait


def legend_height_for(colour_count: int, columns: int, row_height_mm: float = 9.0) -> float:
    """Height of an inline legend strip holding ``colour_count`` swatches."""
    if colour_count <= 0:
        return 0.0
    rows = -(-colour_count // max(1, columns))
    return rows * row_height_mm + 4.0  # a little breathing room above the strip
