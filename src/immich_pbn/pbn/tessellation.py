"""Fixed geometric tilings, as an alternative to tracing colour fields.

The default page follows the picture: every region is a colour field with an
organic outline. These tilings ignore the picture's shapes entirely and impose
a lattice on it -- squares for a pixel-art mosaic, triangles for the low-poly
look, hexagons for something between the two. Each cell then takes the average
colour of whatever falls inside it.

Every tiling here is a *pure function of normalised position*. That matters
more than it sounds: the same tiling is evaluated twice per page, once at
working resolution to decide colours and place numbers, and once at print
resolution to draw the edges. Because cell identity comes from lattice
arithmetic rather than from labelling connected pixels, both evaluations agree
exactly, and the printed edges are the true geometry rather than an enlarged
copy of a small drawing.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

SQRT3 = float(np.sqrt(3.0))


class TessellationError(ValueError):
    pass


class Tessellation:
    """Assigns every pixel to a cell of a lattice.

    ``cell_fraction`` is the cell's characteristic size as a fraction of the
    image width. Expressing it that way, rather than in pixels, is what keeps
    the tiling identical at any resolution.
    """

    name = "base"
    #: Human-readable, for --help and the page caption.
    summary = ""
    #: Radius of the largest circle inscribed in a cell, as a fraction of the
    #: cell size. This, not the cell size, is the room a number actually has.
    inradius_factor = 0.5
    #: Cell size in printed mm that makes this tiling look right by default.
    #: Not shared between tilings on purpose: an 8 mm triangle has barely half
    #: the elbow room of an 8 mm square, so the same number would be a default
    #: that works for one shape and crowds another.
    default_cell_mm = 8.0

    def cell_map(
        self,
        width: int,
        height: int,
        cell_fraction: float,
        aspect: float | None = None,
    ) -> np.ndarray:
        raise NotImplementedError  # pragma: no cover - abstract

    def period(self, side: float) -> tuple[float, float]:
        """Horizontal and vertical repeat distance of the lattice."""
        return (side, side)

    @staticmethod
    def snap(cell_fraction: float) -> float:
        """Round the cell size so a whole number of them spans the width.

        Without this the right-hand edge is a sliver of a cell -- a quarter of a
        square, hanging off the frame with no room for its number. It reads as a
        printing mistake rather than a design, and the fix costs at most a few
        percent on a cell size nobody specified to the millimetre anyway.
        """
        return 1.0 / max(1, round(1.0 / cell_fraction))

    def counts(self, cell_fraction: float, aspect: float) -> tuple[int, int]:
        """Cells across and down, from the page's shape rather than its pixels.

        This is the single most important function in the module, and the
        reason it takes an aspect ratio instead of a width and a height.

        The lattice is evaluated twice per page -- once on the working raster
        to choose colours and place numbers, once on the print raster to draw
        the edges. Those two rasters are the same rectangle in different pixel
        counts, so their aspect ratios disagree in the fourth decimal. Round a
        row count out of each independently and a page whose ideal answer is
        34.5 rows gets 34 from one and 35 from the other: the print raster then
        draws a row of cells the colour pass never knew about, outlined and
        unnumbered along the bottom edge.

        Deriving both counts from the millimetre geometry -- which is what the
        page actually is -- removes the disagreement rather than narrowing it.
        No tolerance to tune, no knife-edge to land on.
        """
        _, period_y = self.period(1.0)
        columns = max(1, round(1.0 / cell_fraction))
        rows = max(1, round(aspect / (cell_fraction * period_y)))
        return columns, rows

    def _grid(
        self,
        width: int,
        height: int,
        cell_fraction: float,
        aspect: float | None,
    ) -> tuple[np.ndarray, np.ndarray, float, tuple[int, int]]:
        if width < 1 or height < 1:
            raise TessellationError("image must have a positive size")
        if not 0.0 < cell_fraction <= 1.0:
            raise TessellationError(
                f"cell size must be between 0 and the image width, got {cell_fraction:.3f}"
            )
        aspect = (height / width) if aspect is None else aspect
        side = cell_fraction * width
        period_x, period_y = self.period(side)
        columns, rows = self.counts(cell_fraction, aspect)

        # Centre the lattice on the frame: whatever does not divide evenly is
        # split between the two edges instead of piling up on the right and the
        # bottom, which is the difference between a border that looks intended
        # and one that looks cropped.
        offset_x = (columns * period_x - width) / 2.0
        offset_y = (rows * period_y - height) / 2.0

        # Pixel centres, so a cell boundary falling exactly on a pixel edge does
        # not land ambiguously on one side or the other.
        xs = np.arange(width, dtype=np.float64) + 0.5 + offset_x
        ys = np.arange(height, dtype=np.float64) + 0.5 + offset_y
        return xs[None, :], ys[:, None], side, (columns, rows)


class SquareGrid(Tessellation):
    """Plain square cells -- the pixel-art mosaic."""

    name = "square"
    summary = "square cells (a pixel mosaic)"

    def cell_map(
        self, width: int, height: int, cell_fraction: float, aspect: float | None = None
    ) -> np.ndarray:
        xs, ys, side, (column_count, row_count) = self._grid(
            width, height, cell_fraction, aspect
        )
        columns = np.clip(np.floor(xs / side), 0, column_count - 1).astype(np.int64)
        rows = np.clip(np.floor(ys / side), 0, row_count - 1).astype(np.int64)
        return (rows * (column_count + 2) + columns + 1).astype(np.int32)


class TriangleGrid(Tessellation):
    """Equilateral triangles -- the low-poly look.

    Worked in the lattice's own skewed basis: u = (side, 0) and
    v = (side/2, side*sqrt(3)/2). Every unit cell of that basis is a rhombus of
    two equilateral triangles, and which of the two a point falls in is decided
    by a single comparison against the rhombus's short diagonal. Doing it this
    way avoids a pile of per-row parity special cases, and gives true
    equilateral triangles rather than the right triangles you get from simply
    cutting squares in half.
    """

    name = "triangle"
    summary = "equilateral triangles (low-poly)"
    inradius_factor = 1.0 / (2.0 * SQRT3)
    default_cell_mm = 14.0

    def period(self, side: float) -> tuple[float, float]:
        return (side, side * SQRT3 / 2.0)

    def cell_map(
        self, width: int, height: int, cell_fraction: float, aspect: float | None = None
    ) -> np.ndarray:
        xs, ys, side, (columns, rows) = self._grid(width, height, cell_fraction, aspect)
        row_height = side * SQRT3 / 2.0

        b = ys / row_height
        a = xs / side - ys / (2.0 * row_height)

        ia = np.floor(a).astype(np.int64)
        ib = np.clip(np.floor(b), 0, rows - 1).astype(np.int64)
        ia = np.clip(ia, -(rows + 2), columns + 1)
        upper = (a - ia) + (b - ib) >= 1.0

        # ia runs negative towards the bottom-left because of the shear, so it
        # is rebased before being folded into an id. The base is taken from
        # rounded cell *counts*, never from the pixel dimensions.
        base = -(rows + 4)
        stride = columns + rows + 8
        rhombus = (ib * stride) + (ia - base)
        return (rhombus * 2 + upper.astype(np.int64) + 1).astype(np.int32)


class HexGrid(Tessellation):
    """Pointy-top hexagons.

    Pixel to fractional axial coordinates, then the standard cube rounding:
    round all three cube coordinates, then correct whichever one moved
    furthest, which is the only way to keep the constraint x + y + z = 0 and so
    the only way to land on a real hexagon rather than between two.
    """

    name = "hex"
    summary = "hexagons"
    default_cell_mm = 9.0

    def period(self, side: float) -> tuple[float, float]:
        # Flat-to-flat horizontally; rows nest at 3/4 of the full height.
        return (side, 1.5 * side / SQRT3)

    def cell_map(
        self, width: int, height: int, cell_fraction: float, aspect: float | None = None
    ) -> np.ndarray:
        xs, ys, side, (column_count, row_count) = self._grid(
            width, height, cell_fraction, aspect
        )
        # `side` is the flat-to-flat width for hexagons, so that --cell-mm means
        # roughly the same visual size whichever tiling is chosen.
        radius = side / SQRT3

        q = (SQRT3 / 3.0 * xs - ys / 3.0) / radius
        r = (2.0 / 3.0 * ys) / radius
        q, r = np.broadcast_arrays(q, r)

        cube_x, cube_z = q, r
        cube_y = -cube_x - cube_z
        rx, ry, rz = np.round(cube_x), np.round(cube_y), np.round(cube_z)
        dx, dy, dz = np.abs(rx - cube_x), np.abs(ry - cube_y), np.abs(rz - cube_z)

        fix_x = (dx > dy) & (dx > dz)
        fix_z = ~fix_x & (dz > dy)
        rx = np.where(fix_x, -ry - rz, rx)
        rz = np.where(fix_z, -rx - ry, rz)

        rz = np.clip(rz, 0, row_count - 1)
        rx = np.clip(rx, -(row_count + 2), column_count + 1)
        base = row_count + column_count + 8
        stride = 2 * base
        return (
            (rz.astype(np.int64) + base) * stride + (rx.astype(np.int64) + base) + 1
        ).astype(np.int32)


TessellationFactory = Callable[[], Tessellation]

_REGISTRY: dict[str, TessellationFactory] = {}


def register_tessellation(name: str, factory: TessellationFactory) -> None:
    _REGISTRY[name] = factory


def available_tessellations() -> list[str]:
    return sorted(_REGISTRY)


def build_tessellation(name: str) -> Tessellation:
    try:
        return _REGISTRY[name]()
    except KeyError:
        known = ", ".join(available_tessellations())
        raise TessellationError(
            f"unknown tessellation {name!r}; available: {known}"
        ) from None


register_tessellation("square", SquareGrid)
register_tessellation("triangle", TriangleGrid)
register_tessellation("hex", HexGrid)
