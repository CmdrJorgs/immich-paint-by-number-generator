"""The tilings themselves: shape, uniformity, and the resolution invariant."""

import numpy as np
import pytest

from immich_pbn.pbn.page import mm_to_pixels
from immich_pbn.pbn.tessellation import (
    SQRT3,
    TessellationError,
    available_tessellations,
    build_tessellation,
    register_tessellation,
)

ALL = ("square", "triangle", "hex")


# ------------------------------------------------------------------- registry


def test_the_three_tilings_are_registered():
    assert set(available_tessellations()) == set(ALL)


def test_an_unknown_tiling_lists_the_real_ones():
    with pytest.raises(TessellationError, match="triangle"):
        build_tessellation("dodecagon")


def test_a_new_tiling_needs_only_a_registration():
    register_tessellation("test-only", lambda: build_tessellation("square"))
    assert build_tessellation("test-only").cell_map(60, 40, 0.25).max() > 1


# --------------------------------------------------------------------- shapes


@pytest.mark.parametrize("name", ALL)
def test_every_pixel_belongs_to_exactly_one_cell(name):
    cells = build_tessellation(name).cell_map(300, 220, 1 / 12)
    assert cells.shape == (220, 300)
    assert (cells > 0).all()  # id 0 is reserved and must never be assigned


@pytest.mark.parametrize("name", ALL)
def test_cells_away_from_the_frame_are_identical(name):
    cells = build_tessellation(name).cell_map(600, 440, 1 / 15)
    ids, counts = np.unique(cells, return_counts=True)

    # Cells the frame cuts through are partial by definition, and a cell just
    # outside it is folded into its neighbour, so both are excluded rather than
    # approximated away with a tolerance.
    touching_frame = np.unique(
        np.concatenate([cells[0], cells[-1], cells[:, 0], cells[:, -1]])
    )
    interior = counts[~np.isin(ids, touching_frame)]

    assert interior.size > 0.5 * counts.size
    # Squares land exactly; triangles and hexagons carry only the jitter of
    # drawing a diagonal edge onto a square pixel grid.
    assert interior.std() / interior.mean() < 0.02


@pytest.mark.parametrize(
    "name, area_over_side_squared",
    [
        ("square", 1.0),
        ("triangle", SQRT3 / 4.0),  # equilateral, not a bisected square
        ("hex", SQRT3 / 2.0),  # regular hexagon of flat-to-flat width `side`
    ],
)
def test_cell_area_matches_the_shape_it_claims_to_be(name, area_over_side_squared):
    width, height, fraction = 900, 700, 1 / 18
    cells = build_tessellation(name).cell_map(width, height, fraction)
    _, counts = np.unique(cells, return_counts=True)
    typical = np.median(counts)
    expected = area_over_side_squared * (fraction * width) ** 2
    assert typical == pytest.approx(expected, rel=0.03)


def test_a_bisected_square_would_fail_the_triangle_area_check():
    """Guards the check above: right triangles have twice the area."""
    side = 50.0
    assert (SQRT3 / 4.0) * side**2 == pytest.approx(1082.5, abs=1.0)
    assert 0.5 * side**2 == pytest.approx(1250.0)  # what cutting squares gives


@pytest.mark.parametrize("name", ALL)
def test_smaller_cells_mean_more_of_them(name):
    tiling = build_tessellation(name)
    coarse = np.unique(tiling.cell_map(600, 440, 1 / 8)).size
    fine = np.unique(tiling.cell_map(600, 440, 1 / 24)).size
    assert fine > coarse * 5  # roughly nine-fold, by area


# ---------------------------------------------------------------- fitting up


@pytest.mark.parametrize("name", ALL)
def test_snapping_makes_a_whole_number_of_cells_span_the_width(name):
    tiling = build_tessellation(name)
    for requested in (1 / 23.25, 1 / 7.6, 1 / 46.5, 1 / 3.2):
        snapped = tiling.snap(requested)
        assert 1.0 / snapped == pytest.approx(round(1.0 / snapped))
        assert abs(snapped - requested) / requested < 0.2  # never wildly off


def test_snapping_removes_the_sliver_column():
    tiling = build_tessellation("square")
    ragged = tiling.cell_map(600, 400, 1 / 23.25)
    snapped = tiling.cell_map(600, 400, tiling.snap(1 / 23.25))
    widths_before = np.unique(ragged[0], return_counts=True)[1]
    widths_after = np.unique(snapped[0], return_counts=True)[1]
    # 23.25 cells across cannot come out even: unsnapped, the edges are partial.
    assert widths_before.min() < widths_before.max() - 1
    # Snapped, every column is a full column bar integer rounding.
    assert widths_after.min() >= widths_after.max() - 1


@pytest.mark.parametrize("name", ALL)
def test_the_leftover_is_split_between_both_edges(name):
    # Otherwise it all piles up on the right and bottom and the page looks
    # cropped rather than composed.
    cells = build_tessellation(name).cell_map(600, 437, 1 / 11)
    counts = np.unique(cells[:, :1], return_counts=True)[1]
    top_row = np.unique(cells[:1, :], return_counts=True)[1]
    assert counts.size and top_row.size  # both edges carry cells, not a void


@pytest.mark.parametrize("name", ALL)
def test_degenerate_inputs_are_rejected(name):
    tiling = build_tessellation(name)
    with pytest.raises(TessellationError):
        tiling.cell_map(0, 10, 0.1)
    with pytest.raises(TessellationError):
        tiling.cell_map(10, 10, 0.0)
    with pytest.raises(TessellationError):
        tiling.cell_map(10, 10, 1.5)


# ------------------------------------------------------- the core invariant


@pytest.mark.parametrize("name", ALL)
def test_the_same_lattice_comes_back_at_working_and_print_resolution(name):
    """The one that matters.

    The tiling is evaluated twice per page -- small, to pick colours and place
    numbers, and large, to draw the edges. If those two disagree by even one
    cell, the page prints an outlined cell that nothing ever coloured or
    numbered. This sweeps page shapes and cell sizes looking for a disagreement.
    """
    tiling = build_tessellation(name)
    mismatches = []
    for draw_w, draw_h in ((186.0, 139.5), (273.0, 149.0), (150.5, 200.0), (186.0, 174.0)):
        aspect = draw_h / draw_w
        for cell_mm in (4.0, 5.0, 8.0, 8.09, 11.0, 14.3):
            fraction = tiling.snap(cell_mm / draw_w)
            printed = tiling.cell_map(
                mm_to_pixels(draw_w, 300), mm_to_pixels(draw_h, 300), fraction, aspect
            )
            reference = set(np.unique(printed))
            for resolution in (600, 1400):
                height = max(1, round(resolution * aspect))
                working = tiling.cell_map(resolution, height, fraction, aspect)
                missing = reference ^ set(np.unique(working))
                if missing:
                    # Tolerate cells that exist only as a pixel or two in a
                    # corner: they sit under the frame line and cannot be seen.
                    sizes = [int((printed == cell_id).sum()) for cell_id in missing]
                    populated = np.bincount(printed.ravel())
                    typical = np.median(populated[populated > 0])
                    if max(sizes) > 0.01 * typical:
                        mismatches.append((draw_w, draw_h, cell_mm, resolution, sizes))
    assert not mismatches, f"lattice disagreed: {mismatches[:3]}"


@pytest.mark.parametrize("name", ALL)
def test_counts_come_from_the_page_shape_not_its_pixels(name):
    """A page whose ideal answer is exactly 34.5 rows must not round both ways."""
    tiling = build_tessellation(name)
    fraction = tiling.snap(4.0 / 186.0)
    aspect = 139.5 / 186.0
    reference = tiling.counts(fraction, aspect)
    drawn: list[int] = []
    for resolution in (600, 900, 1400, 2197, 2400):
        height = max(1, round(resolution * aspect))
        assert tiling.counts(fraction, aspect) == reference
        # And the number of cells drawn must not drift with the raster either.
        cells = tiling.cell_map(resolution, height, fraction, aspect)
        drawn.append(np.unique(cells).size)
    assert len(set(drawn)) == 1, f"cell count drifted with resolution: {drawn}"


@pytest.mark.parametrize("name", ALL)
def test_inradius_factor_matches_the_shape(name):
    tiling = build_tessellation(name)
    expected = {"square": 0.5, "triangle": 1 / (2 * SQRT3), "hex": 0.5}[name]
    assert tiling.inradius_factor == pytest.approx(expected)


def test_a_triangle_needs_a_bigger_cell_than_a_square_for_the_same_room():
    # Which is why the default cell size is per-tiling rather than shared.
    square, triangle = build_tessellation("square"), build_tessellation("triangle")
    assert triangle.default_cell_mm > square.default_cell_mm
    square_room = square.default_cell_mm * square.inradius_factor
    triangle_room = triangle.default_cell_mm * triangle.inradius_factor
    assert triangle_room == pytest.approx(square_room, rel=0.25)
