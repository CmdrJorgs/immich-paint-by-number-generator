import numpy as np
import pytest

from immich_pbn.pbn.colour import (
    INK_CROSSOVER_LUMINANCE,
    hex_code,
    lab_to_rgb_bytes,
    relative_luminance,
    rgb_bytes_to_lab,
)
from immich_pbn.pbn.numbering import choose_font_size, fits, place_numbers
from immich_pbn.pbn.page import PageLayout, legend_height_for, mm_to_points
from immich_pbn.pbn.quantize import quantize
from immich_pbn.pbn.regions import _label_by_colour, area_for_min_width, build_regions

# ------------------------------------------------------------------- colour


def test_lab_round_trip_is_lossless_for_8_bit_colour():
    values = np.arange(0, 256, 15)
    grid = np.stack(np.meshgrid(values, values, values, indexing="ij"), axis=-1)
    grid = grid.reshape(-1, 3).astype(np.uint8)
    assert np.array_equal(lab_to_rgb_bytes(rgb_bytes_to_lab(grid)), grid)


@pytest.mark.parametrize(
    "rgb, lab",
    [
        ([255, 255, 255], (100.0, 0.0, 0.0)),
        ([0, 0, 0], (0.0, 0.0, 0.0)),
        ([255, 0, 0], (53.24, 80.09, 67.20)),
        ([0, 0, 255], (32.30, 79.19, -107.86)),
    ],
)
def test_lab_matches_published_reference_values(rgb, lab):
    got = rgb_bytes_to_lab(np.array([rgb], dtype=np.uint8))[0]
    assert got == pytest.approx(np.array(lab), abs=0.01)


def test_ink_crossover_is_where_white_and_black_contrast_are_equal():
    luminance = INK_CROSSOVER_LUMINANCE
    on_white = 1.05 / (luminance + 0.05)
    on_black = (luminance + 0.05) / 0.05
    assert on_white == pytest.approx(on_black, rel=1e-3)


def test_mid_olive_gets_black_ink_not_white():
    # The bug this pins: a "luminance < 0.45" rule paints white digits onto
    # #A29159, which no one can read.
    olive = np.array([0xA2, 0x91, 0x59])
    assert relative_luminance(olive) > INK_CROSSOVER_LUMINANCE


def test_hex_code_formatting():
    assert hex_code(np.array([18, 52, 86])) == "#123456"


# ----------------------------------------------------------------- quantize


def _blobs(colours, shape=(120, 160), noise=6.0, seed=0):
    rng = np.random.default_rng(seed)
    palette = np.array(colours, dtype=np.uint8)
    picks = rng.integers(0, len(colours), (shape[0] // 8, shape[1] // 8))
    image = np.repeat(np.repeat(palette[picks], 8, axis=0), 8, axis=1)
    return np.clip(image + rng.normal(0, noise, image.shape), 0, 255).astype(np.uint8)


SIX = [[220, 40, 40], [40, 120, 220], [250, 220, 80], [30, 140, 60], [240, 240, 235], [20, 20, 25]]


def test_the_same_seed_gives_the_same_palette():
    lab = rgb_bytes_to_lab(_blobs(SIX))
    first, palette_a = quantize(lab, 6, seed=42)
    second, palette_b = quantize(lab, 6, seed=42)
    assert np.array_equal(first, second)
    assert palette_a.hex_codes == palette_b.hex_codes


def test_clearly_separated_colours_are_recovered_whatever_the_seed():
    lab = rgb_bytes_to_lab(_blobs(SIX))
    runs = [quantize(lab, 6, seed=s)[1].rgb.astype(int) for s in (1, 7, 99)]
    for other in runs[1:]:
        assert np.abs(runs[0] - other).max() <= 4


def test_palette_is_ordered_dark_to_light():
    _, palette = quantize(rgb_bytes_to_lab(_blobs(SIX)), 6, seed=1)
    assert list(palette.lab[:, 0]) == sorted(palette.lab[:, 0])


def test_indistinguishable_colours_do_not_each_claim_a_number():
    # Six near-identical blues must not become six legend entries the painter
    # cannot tell apart or mix differently.
    almost_one_colour = [[110, 150, 210], [111, 151, 211], [109, 149, 209]]
    _, palette = quantize(rgb_bytes_to_lab(_blobs(almost_one_colour, noise=1.0)), 8, seed=1)
    assert len(palette) < 8


def test_asking_for_more_colours_than_exist_is_not_an_error():
    _, palette = quantize(rgb_bytes_to_lab(np.full((40, 40, 3), 128, np.uint8)), 8, seed=1)
    assert len(palette) == 1


def test_fewer_than_two_colours_is_rejected():
    with pytest.raises(ValueError):
        quantize(rgb_bytes_to_lab(_blobs(SIX)), 1)


# ------------------------------------------------------------------ regions


def test_min_area_is_derived_from_printed_size_not_pixels():
    # The same 3 mm rule must mean different pixel counts at different
    # working resolutions, or the dial drifts when you change --resolution.
    small = area_for_min_width(3.0, 190.0, 600)
    large = area_for_min_width(3.0, 190.0, 1800)
    assert large == pytest.approx(small * 9, rel=0.02)


def test_zero_width_is_rejected():
    with pytest.raises(ValueError):
        area_for_min_width(3.0, 0.0, 600)


def test_merging_removes_every_region_below_the_threshold():
    rng = np.random.default_rng(3)
    noisy = rng.integers(0, 8, (150, 200)).astype(np.int16)
    palette = rng.normal(50, 25, (8, 3))
    regions = build_regions(noisy, palette, min_area=60)
    assert regions.areas[1:].min() >= 60


def test_merging_shrinks_a_speckled_image_by_orders_of_magnitude():
    rng = np.random.default_rng(4)
    noisy = rng.integers(0, 10, (150, 200)).astype(np.int16)
    before = _label_by_colour(noisy)[0].max()
    after = build_regions(noisy, rng.normal(50, 25, (10, 3)), min_area=80).count
    assert after < before / 50


def test_merging_leaves_a_clean_image_untouched():
    clean = np.zeros((100, 100), dtype=np.int16)
    clean[:, 50:] = 1
    regions = build_regions(clean, np.array([[20.0, 0, 0], [80.0, 0, 0]]), min_area=50)
    assert regions.count == 2
    assert regions.merge_passes == 0


def test_a_speck_joins_the_neighbour_it_shares_the_most_edge_with():
    image = np.zeros((60, 60), dtype=np.int16)
    image[:, 30:] = 1
    image[28:31, 28:31] = 2  # a speck sitting mostly on the left half
    image[28:31, 30:31] = 2
    palette = np.array([[50.0, 0, 0], [55.0, 0, 0], [90.0, 20, 20]])
    regions = build_regions(image, palette, min_area=40)
    assert 2 not in np.unique(regions.indices)


def test_diagonal_touches_do_not_join_regions():
    # Two squares meeting at a corner read as two shapes to someone painting.
    board = np.zeros((40, 40), dtype=np.int16)
    board[:20, :20] = 1
    board[20:, 20:] = 1
    labels, _, _ = _label_by_colour(board)
    assert labels.max() == 4


def test_boundary_mask_frames_the_drawing():
    regions = build_regions(
        np.zeros((30, 30), dtype=np.int16), np.array([[50.0, 0, 0]]), min_area=1
    )
    edge = regions.boundary_mask(include_frame=True)
    assert edge[0].all() and edge[-1].all() and edge[:, 0].all() and edge[:, -1].all()


# ---------------------------------------------------------------- numbering


def _region_map(mask):
    image = mask.astype(np.int16)
    return build_regions(image, np.array([[20.0, 0, 0], [80.0, 0, 0]]), min_area=4)


def test_every_number_lands_inside_its_own_region():
    rng = np.random.default_rng(5)
    blocks = rng.integers(0, 5, (30, 44)).astype(np.int16)
    image = np.repeat(np.repeat(blocks, 5, axis=0), 5, axis=1)
    regions = build_regions(image, rng.normal(50, 25, (5, 3)), min_area=40)
    for placement in place_numbers(regions):
        assert regions.labels[int(placement.y), int(placement.x)] == placement.region_id


def test_placement_beats_the_centroid_on_a_crescent():
    height = width = 200
    yy, xx = np.mgrid[0:height, 0:width]
    crescent = ((xx - 100) ** 2 + (yy - 100) ** 2 < 80**2) & (
        (xx - 125) ** 2 + (yy - 100) ** 2 >= 78**2
    )
    centre_y, centre_x = np.array(np.nonzero(crescent)).mean(axis=1)
    assert not crescent[int(centre_y), int(centre_x)]  # the naive answer fails

    regions = _region_map(crescent)
    placement = next(p for p in place_numbers(regions) if p.colour_index == 1)
    assert crescent[int(placement.y), int(placement.x)]


def test_placement_beats_the_centroid_on_an_l_shape():
    shape = np.zeros((200, 200), dtype=bool)
    shape[20:180, 20:70] = True
    shape[130:180, 20:180] = True
    centre_y, centre_x = np.array(np.nonzero(shape)).mean(axis=1)
    assert not shape[int(centre_y), int(centre_x)]

    regions = _region_map(shape)
    placement = next(p for p in place_numbers(regions) if p.colour_index == 1)
    assert shape[int(placement.y), int(placement.x)]


def test_room_matches_the_largest_inscribed_circle():
    disc = np.zeros((120, 120), dtype=bool)
    yy, xx = np.mgrid[0:120, 0:120]
    disc[(xx - 60) ** 2 + (yy - 60) ** 2 < 30**2] = True
    regions = _region_map(disc)
    placement = next(p for p in place_numbers(regions) if p.colour_index == 1)
    assert placement.room == pytest.approx(30, abs=1.5)
    assert (placement.x, placement.y) == pytest.approx((60, 60), abs=2)


def test_a_number_too_big_for_its_region_is_dropped_rather_than_overflowing():
    regions = _region_map(np.ones((8, 8), dtype=bool))
    placement = place_numbers(regions)[0]
    assert choose_font_size(placement, "12", 1.0, preferred=40, minimum=30) is None


def test_sizing_shrinks_to_fit_before_giving_up():
    disc = np.zeros((120, 120), dtype=bool)
    yy, xx = np.mgrid[0:120, 0:120]
    disc[(xx - 60) ** 2 + (yy - 60) ** 2 < 30**2] = True
    placement = next(p for p in place_numbers(_region_map(disc)) if p.colour_index == 1)
    chosen = choose_font_size(placement, "18", 1.0, preferred=100, minimum=4)
    assert chosen is not None and chosen < 100
    assert fits(placement, "18", chosen, 1.0)


def test_two_digits_need_more_room_than_one():
    disc = np.zeros((80, 80), dtype=bool)
    yy, xx = np.mgrid[0:80, 0:80]
    disc[(xx - 40) ** 2 + (yy - 40) ** 2 < 12**2] = True
    placement = next(p for p in place_numbers(_region_map(disc)) if p.colour_index == 1)
    one = choose_font_size(placement, "7", 1.0, preferred=40, minimum=1)
    two = choose_font_size(placement, "17", 1.0, preferred=40, minimum=1)
    assert one > two


# --------------------------------------------------------------------- page


def test_a4_dimensions_in_points():
    assert mm_to_points(210.0) == pytest.approx(595.28, abs=0.01)
    assert mm_to_points(297.0) == pytest.approx(841.89, abs=0.01)


def test_the_drawing_always_fits_its_area():
    layout = PageLayout(margin_mm=12, legend_height_mm=30, caption_height_mm=6)
    for aspect in (0.5, 0.75, 1.0, 1.5, 3.0):
        width, height = layout.fitted_drawing_mm(aspect)
        assert width <= layout.drawing_width_mm + 1e-9
        assert height <= layout.drawing_height_mm + 1e-9
        assert width / height == pytest.approx(aspect)


def test_orientation_follows_the_photo():
    layout = PageLayout(margin_mm=12, legend_height_mm=30, caption_height_mm=6)
    assert layout.orientation_for(16 / 9).landscape is True
    assert layout.orientation_for(2 / 3).landscape is False


def test_orientation_choice_is_the_one_that_prints_bigger():
    layout = PageLayout(margin_mm=12, legend_height_mm=20, caption_height_mm=6)
    for aspect in (0.6, 1.0, 1.4, 2.2):
        chosen = layout.orientation_for(aspect)
        other = PageLayout(not chosen.landscape, 12, 20, 6)
        chosen_area = np.prod(chosen.fitted_drawing_mm(aspect))
        assert chosen_area >= np.prod(other.fitted_drawing_mm(aspect)) - 1e-9


def test_a_legend_that_swallows_the_page_is_an_error_not_a_negative_drawing():
    layout = PageLayout(margin_mm=12, legend_height_mm=400, caption_height_mm=6)
    with pytest.raises(ValueError, match="no room"):
        layout.fitted_drawing_mm(1.0)


def test_legend_height_grows_by_the_row():
    assert legend_height_for(8, 8) < legend_height_for(9, 8)
    assert legend_height_for(0, 8) == 0.0
