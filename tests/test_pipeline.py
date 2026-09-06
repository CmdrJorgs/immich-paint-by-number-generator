"""End-to-end: a photo goes in, a valid A4 PDF comes out."""

import numpy as np
import pytest

from immich_pbn.pbn.numbering import choose_font_size
from immich_pbn.pbn.page import PageLayout, legend_height_for
from immich_pbn.pbn.pipeline import PbnOptions, PipelineError, generate, load_image
from immich_pbn.pbn.render import RenderOptions, render_pdf, save_preview_png

A4_POINTS = (595.28, 841.89)


def _layout(colours=12):
    return PageLayout(
        margin_mm=12.0,
        legend_height_mm=legend_height_for(colours, 8),
        caption_height_mm=6.0,
    )


def _options(**kwargs):
    base = dict(colours=12, resolution=420, seed=11)
    base.update(kwargs)
    return PbnOptions(**base)


def test_a_photo_becomes_regions_a_palette_and_an_outline(photo_bytes):
    result = generate(photo_bytes, _options(), _layout())
    assert 10 < result.stats["regions"] < 2000
    assert 2 <= len(result.palette) <= 12
    assert result.outline.any()
    assert result.placements


def test_every_region_is_at_least_the_printable_minimum(photo_bytes):
    result = generate(photo_bytes, _options(min_region_mm=4.0), _layout())
    assert result.regions.areas[1:].min() >= result.stats["min_region_px"]


def test_a_bigger_minimum_yields_fewer_regions(photo_bytes):
    coarse = generate(photo_bytes, _options(min_region_mm=6.0), _layout())
    fine = generate(photo_bytes, _options(min_region_mm=2.0), _layout())
    assert coarse.stats["regions"] < fine.stats["regions"]


def test_more_colours_yields_more_regions(photo_bytes):
    few = generate(photo_bytes, _options(colours=6), _layout())
    many = generate(photo_bytes, _options(colours=24), _layout())
    assert many.stats["regions"] > few.stats["regions"]


def test_resolution_only_downscales_and_says_so(photo_bytes):
    result = generate(photo_bytes, _options(resolution=9000), _layout())
    assert result.stats["resolution_capped"] is True
    assert max(result.working_size) == max(result.source_size)


def test_the_same_seed_reproduces_the_page(photo_bytes):
    first = generate(photo_bytes, _options(), _layout())
    second = generate(photo_bytes, _options(), _layout())
    assert first.palette.hex_codes == second.palette.hex_codes
    assert np.array_equal(first.outline, second.outline)


def test_orientation_can_be_forced(photo_bytes):
    assert generate(photo_bytes, _options(orientation="portrait"), _layout()).layout.landscape is False
    assert generate(photo_bytes, _options(orientation="landscape"), _layout()).layout.landscape is True


def test_a_landscape_photo_gets_a_landscape_page(photo_bytes):
    # The fixture is 320x240, so "auto" should choose landscape unprompted.
    assert generate(photo_bytes, _options(), _layout()).layout.landscape is True


def test_the_drawing_never_overflows_the_page(photo_bytes):
    result = generate(photo_bytes, _options(), _layout())
    width, height = result.drawing_size_mm
    assert width <= result.layout.drawing_width_mm + 1e-9
    assert height <= result.layout.drawing_height_mm + 1e-9


def test_a_single_colour_photo_is_refused_with_a_useful_message():
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (200, 200), (128, 128, 128)).save(buffer, format="PNG")
    with pytest.raises(PipelineError, match="single colour"):
        generate(buffer.getvalue(), _options(), _layout())


def test_a_non_image_is_refused_clearly():
    with pytest.raises(PipelineError, match="could not decode"):
        generate(b"this is not a JPEG", _options(), _layout())


def test_greyscale_and_palette_images_are_accepted():
    import io

    from PIL import Image

    for mode in ("L", "P"):
        image = Image.effect_mandelbrot((300, 220), (-2, -1.2, 0.6, 1.2), 40).convert(mode)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        result = generate(buffer.getvalue(), _options(colours=6), _layout())
        assert result.stats["regions"] > 1


def test_load_image_honours_exif_orientation():
    import io

    from PIL import Image

    portrait = Image.new("RGB", (100, 200), (10, 20, 30))
    buffer = io.BytesIO()
    # Orientation 6 means "rotate 90 CW to display", so W and H swap.
    exif = Image.Exif()
    exif[0x0112] = 6
    portrait.save(buffer, format="JPEG", exif=exif)
    assert load_image(buffer.getvalue(), max_edge=400).size == (200, 100)


# ------------------------------------------------------------------ rendering


def test_the_pdf_is_a4_and_has_the_pages_asked_for(photo_bytes, tmp_path):
    pypdfium2 = pytest.importorskip("pypdfium2")
    result = generate(photo_bytes, _options(), _layout())
    path = tmp_path / "page.pdf"
    with path.open("wb") as handle:
        stats = render_pdf(result, RenderOptions(legend="inline", reference_page=True), handle)

    document = pypdfium2.PdfDocument(str(path))
    assert len(document) == 2  # drawing + reference
    width, height = document[0].get_size()
    assert sorted((round(width, 1), round(height, 1))) == sorted(
        (round(A4_POINTS[0], 1), round(A4_POINTS[1], 1))
    )
    assert stats["numbered"] > 0


def test_legend_on_its_own_page_adds_a_page(photo_bytes, tmp_path):
    pypdfium2 = pytest.importorskip("pypdfium2")
    result = generate(photo_bytes, _options(), _layout())
    path = tmp_path / "page.pdf"
    with path.open("wb") as handle:
        render_pdf(result, RenderOptions(legend="page", reference_page=True), handle)
    assert len(pypdfium2.PdfDocument(str(path))) == 3


def test_a_bare_drawing_is_a_single_page(photo_bytes, tmp_path):
    pypdfium2 = pytest.importorskip("pypdfium2")
    result = generate(photo_bytes, _options(), _layout())
    path = tmp_path / "page.pdf"
    with path.open("wb") as handle:
        render_pdf(result, RenderOptions(legend="none", reference_page=False), handle)
    assert len(pypdfium2.PdfDocument(str(path))) == 1


def test_numbers_are_real_text_not_pixels(photo_bytes, tmp_path):
    """Vector text is the whole reason for the hybrid renderer; prove it is there."""
    pypdfium2 = pytest.importorskip("pypdfium2")
    result = generate(photo_bytes, _options(), _layout())
    path = tmp_path / "page.pdf"
    with path.open("wb") as handle:
        render_pdf(result, RenderOptions(legend="inline"), handle)
    text = pypdfium2.PdfDocument(str(path))[0].get_textpage().get_text_range()
    assert any(str(i) in text for i in range(1, len(result.palette) + 1))


def test_the_unnumbered_share_is_reported_by_area_not_region_count(photo_bytes, tmp_path):
    result = generate(photo_bytes, _options(), _layout())
    with (tmp_path / "p.pdf").open("wb") as handle:
        stats = render_pdf(result, RenderOptions(), handle)
    # Slivers are many but tiny; a page can be 40% unnumbered by count and
    # still cover almost all of its surface with numbered regions.
    assert 0 <= stats["unnumbered_area_percent"] <= 25


def test_every_drawn_number_fits_the_region_it_sits_in(photo_bytes):
    result = generate(photo_bytes, _options(), _layout())
    mm_per_px = result.drawing_size_mm[0] / result.working_size[0]
    for placement in result.placements:
        label = str(placement.colour_index + 1)
        size = choose_font_size(placement, label, mm_per_px, preferred=2.6, minimum=1.5)
        if size is None:
            continue
        half_diagonal = 0.5 * np.hypot(0.62 * size * len(label), 0.72 * size)
        assert half_diagonal <= placement.room * mm_per_px


def test_preview_png_is_written(photo_bytes, tmp_path):
    from PIL import Image

    result = generate(photo_bytes, _options(), _layout())
    path = tmp_path / "preview.png"
    save_preview_png(result, str(path))
    assert Image.open(path).size == result.working_size


# ------------------------------------------------------------- tiled styles

TILED = ("square", "triangle", "hex")


@pytest.mark.parametrize("style", TILED)
def test_every_tiled_style_produces_a_page(style, photo_bytes):
    result = generate(photo_bytes, _options(style=style, cell_mm=10.0), _layout())
    assert result.stats["style"] == style
    assert result.stats["regions"] > 10
    assert result.outline.any()
    assert len(result.placements) == result.stats["regions"]


@pytest.mark.parametrize("style", TILED)
def test_the_outline_is_drawn_from_the_same_lattice_that_was_coloured(style, photo_bytes):
    """The invariant a tiled page stands on.

    The colours and numbers come from the lattice evaluated on the working
    raster; the edges come from it evaluated on the print raster. If the two
    disagree, the page grows cells that are outlined but never coloured or
    numbered -- which is exactly what a half-height row along the bottom edge
    looked like before the counts were pinned to the page's millimetres.
    """
    for cell_mm in (5.0, 8.0, 11.0, 14.3):
        result = generate(photo_bytes, _options(style=style, cell_mm=cell_mm), _layout())
        assert result.stats["outline_cells"] == result.stats["regions"], (
            f"{style} at {cell_mm}mm: {result.stats['outline_cells']} cells outlined "
            f"but {result.stats['regions']} coloured"
        )


@pytest.mark.parametrize("style", TILED)
def test_a_tiled_page_merges_nothing(style, photo_bytes):
    # A mosaic's whole point is the visible grid; absorbing a cell into its
    # same-coloured neighbour would grow back the organic blobs it avoids.
    result = generate(photo_bytes, _options(style=style), _layout())
    assert result.regions.merge_passes == 0
    labels = result.regions.labels
    assert result.stats["regions"] == np.unique(labels).size


@pytest.mark.parametrize("style", TILED)
def test_every_cell_gets_a_colour_from_the_palette(style, photo_bytes):
    result = generate(photo_bytes, _options(style=style), _layout())
    used = np.unique(result.regions.indices)
    assert used.min() >= 0
    assert used.max() < len(result.palette)


@pytest.mark.parametrize("style", TILED)
def test_the_cell_size_is_snapped_and_reported(style, photo_bytes):
    result = generate(photo_bytes, _options(style=style, cell_mm=8.0), _layout())
    reported = result.stats["cell_mm"]
    # Snapped to a whole number of cells across, so never exactly what was asked.
    assert abs(reported - 8.0) < 1.0
    assert result.drawing_size_mm[0] / reported == pytest.approx(
        round(result.drawing_size_mm[0] / reported), abs=0.02
    )


@pytest.mark.parametrize("style", TILED)
def test_smaller_cells_mean_more_of_them(style, photo_bytes):
    coarse = generate(photo_bytes, _options(style=style, cell_mm=16.0), _layout())
    fine = generate(photo_bytes, _options(style=style, cell_mm=6.0), _layout())
    assert fine.stats["regions"] > coarse.stats["regions"] * 3


@pytest.mark.parametrize("style", TILED)
def test_digits_grow_with_the_room_a_cell_actually_has(style, photo_bytes):
    small = generate(photo_bytes, _options(style=style, cell_mm=8.0), _layout())
    large = generate(photo_bytes, _options(style=style, cell_mm=22.0), _layout())
    assert large.number_size_mm > small.number_size_mm
    # Sized off the inscribed circle, not the nominal cell.
    assert large.number_size_mm <= large.stats["inradius_mm"] * 1.5


def test_a_triangle_cell_has_less_room_than_a_square_of_the_same_size(photo_bytes):
    square = generate(photo_bytes, _options(style="square", cell_mm=14.0), _layout())
    triangle = generate(photo_bytes, _options(style="triangle", cell_mm=14.0), _layout())
    assert triangle.stats["inradius_mm"] < square.stats["inradius_mm"]


def test_contour_remains_the_default(photo_bytes):
    assert generate(photo_bytes, _options(), _layout()).stats["style"] == "contour"


def test_an_unknown_style_is_rejected(photo_bytes):
    from immich_pbn.pbn.tessellation import TessellationError

    with pytest.raises(TessellationError, match="hex"):
        generate(photo_bytes, _options(style="pentagon"), _layout())


@pytest.mark.parametrize("style", TILED)
def test_a_tiled_page_still_renders_to_a4(style, photo_bytes, tmp_path):
    pypdfium2 = pytest.importorskip("pypdfium2")
    result = generate(photo_bytes, _options(style=style), _layout())
    path = tmp_path / f"{style}.pdf"
    with path.open("wb") as handle:
        stats = render_pdf(result, RenderOptions(legend="inline"), handle)
    document = pypdfium2.PdfDocument(str(path))
    width, height = document[0].get_size()
    assert sorted((round(width), round(height))) == [595, 842]
    # Numbers are what makes it paintable; a tiled page should number nearly all.
    assert stats["numbered"] > 0.95 * result.stats["regions"]
