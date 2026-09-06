"""Where to print each region's number.

The centroid is the obvious answer and the wrong one: the centroid of a
crescent-shaped shadow lands outside the shadow, and the centroid of an
L-shaped wall lands in the corner it does not occupy. What is wanted is the
point furthest from any edge -- the centre of the largest circle that fits
inside the region, sometimes called the pole of inaccessibility -- because that
is both guaranteed inside the shape and the roomiest spot in it.

That same distance also says how big the number may be, and whether the region
has room for one at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .regions import RegionMap


@dataclass(frozen=True)
class NumberPlacement:
    """One printed number, in working-image pixel coordinates."""

    region_id: int
    colour_index: int
    x: float
    y: float
    #: Radius of the largest circle inscribed in the region, in pixels.
    room: float
    area: int


def place_numbers(regions: RegionMap) -> list[NumberPlacement]:
    """Find the roomiest interior point of every region, in one pass.

    Distance is measured to the nearest region boundary *anywhere* in the
    image, not per region. Because boundaries are shared, a pixel's distance to
    the nearest boundary is already its distance to its own region's edge, so
    one distance transform serves every region instead of one per region.
    """
    boundary = regions.boundary_mask(include_frame=True)
    distance = ndimage.distance_transform_edt(~boundary)

    ids = regions.region_ids
    if ids.size == 0:
        return []

    positions = ndimage.maximum_position(distance, regions.labels, index=ids)
    maxima = ndimage.maximum(distance, regions.labels, index=ids)

    placements = []
    # strict: scipy returns a list per index array, but a silent length
    # mismatch here would put numbers in the wrong regions.
    for region_id, position, room in zip(
        ids, positions, np.atleast_1d(maxima), strict=True
    ):
        # A region whose every pixel touches a boundary (a one-pixel filament)
        # has no interior; maximum_position returns None for an empty label.
        if position is None:
            continue
        y, x = position
        placements.append(
            NumberPlacement(
                region_id=int(region_id),
                colour_index=int(regions.colour_of_region[region_id]),
                x=float(x) + 0.5,
                y=float(y) + 0.5,
                room=float(room),
                area=int(regions.areas[region_id]),
            )
        )
    return placements


def fits(placement: NumberPlacement, text: str, font_size: float, scale: float) -> bool:
    """Does ``text`` at ``font_size`` fit inside this region's inscribed circle?

    Unit-agnostic: ``scale`` converts working-image pixels into whatever unit
    ``font_size`` is in, so the caller can reason in printed millimetres and
    never think about the working image's pixel grid at all.

    The text box is approximated as 0.62 x size wide per digit -- close enough
    for the digits of a Helvetica-class face, whose figures are tabular and
    narrower than its letters -- and 0.72 x size tall for cap height. It has to
    fit inside the inscribed circle, hence the diagonal.
    """
    width = 0.62 * font_size * len(text)
    height = 0.72 * font_size
    half_diagonal = 0.5 * float(np.hypot(width, height))
    return half_diagonal <= placement.room * scale * 0.92


def choose_font_size(
    placement: NumberPlacement,
    text: str,
    scale: float,
    *,
    preferred: float,
    minimum: float,
    step: float = 0.05,
) -> float | None:
    """Largest legible size that fits, or ``None`` if the region is too tight.

    Regions with no room are simply left unnumbered. That is what commercial
    paint-by-number kits do too: a sliver between two shapes is painted by
    inference from its neighbours, and a number crammed into it would be
    unreadable and would foul the outline.
    """
    size = preferred
    while size >= minimum:
        if fits(placement, text, size, scale):
            return size
        size -= step
    return None
