"""Draw the finished page(s) to a PDF.

Hybrid on purpose. The outlines go down as a one-bit raster computed at the
printer's own resolution, and the numbers go down as real vector text. Tracing
every region boundary into a Bezier path would produce a smaller file and no
visible improvement -- at 300 dpi a pixel is 0.085 mm, thinner than the pencil
line anyone will draw over it -- whereas vector numbers stay sharp at any zoom
and, more usefully, stay selectable and searchable in the PDF.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image
from reportlab.lib.colors import Color
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as pdfcanvas

from .colour import hex_code
from .numbering import choose_font_size
from .page import mm_to_points
from .pipeline import PbnResult

NUMBER_FONT = "Helvetica"
CAPTION_FONT = "Helvetica"
LEGEND_FONT = "Helvetica-Bold"

#: Fraction of font size from baseline to the optical centre of a row of digits.
_DIGIT_CENTRE = 0.36

INK = Color(0.09, 0.09, 0.10)
MUTED = Color(0.45, 0.45, 0.48)


@dataclass(frozen=True)
class RenderOptions:
    title: str = ""
    caption: str = ""
    legend: str = "inline"  # inline | page | none
    legend_columns: int = 8
    reference_page: bool = True
    number_size_mm: float = 2.6
    min_number_size_mm: float = 1.5
    show_hex: bool = True


def render_pdf(result: PbnResult, options: RenderOptions, destination) -> dict[str, int]:
    """Write the PDF and report what actually made it onto the page."""
    layout = result.layout
    canvas = pdfcanvas.Canvas(destination, pagesize=layout.size_points)
    canvas.setTitle(options.title or "Paint by number")
    canvas.setAuthor("immich-pbn")

    stats = _draw_drawing_page(canvas, result, options)

    if options.legend == "page":
        canvas.showPage()
        _draw_legend_page(canvas, result, options)
    if options.reference_page:
        canvas.showPage()
        _draw_reference_page(canvas, result, options)

    canvas.save()
    return stats


def _draw_drawing_page(
    canvas: pdfcanvas.Canvas, result: PbnResult, options: RenderOptions
) -> dict[str, int]:
    layout = result.layout
    draw_w_mm, draw_h_mm = result.drawing_size_mm
    origin_x_mm, origin_y_mm = layout.drawing_origin_mm
    # Centre the drawing in whatever space the aspect ratio leaves over. The
    # legend stays pinned to the foot of the page like a footer; only the
    # drawing floats, so a near-square photo on a portrait sheet gets even
    # bands of white above and below instead of one fat one at the top.
    origin_x_mm += (layout.drawing_width_mm - draw_w_mm) / 2.0
    origin_y_mm += (layout.drawing_height_mm - draw_h_mm) / 2.0

    canvas.drawImage(
        _outline_reader(result.outline),
        mm_to_points(origin_x_mm),
        mm_to_points(origin_y_mm),
        width=mm_to_points(draw_w_mm),
        height=mm_to_points(draw_h_mm),
        mask=[255, 255, 255, 255, 255, 255],  # white becomes transparent
    )

    stats = _draw_numbers(
        canvas, result, options, origin_x_mm, origin_y_mm, draw_w_mm, draw_h_mm
    )

    if options.legend == "inline" and layout.legend_height_mm > 0:
        _draw_legend_strip(
            canvas,
            result,
            options,
            x_mm=layout.margin_mm,
            y_mm=layout.margin_mm + layout.caption_height_mm,
            width_mm=layout.drawing_width_mm,
            height_mm=layout.legend_height_mm,
        )

    if options.caption:
        canvas.setFont(CAPTION_FONT, 7.0)
        canvas.setFillColor(MUTED)
        canvas.drawString(
            mm_to_points(layout.margin_mm),
            mm_to_points(layout.margin_mm * 0.55),
            options.caption[:180],
        )
    return stats


def _draw_numbers(
    canvas: pdfcanvas.Canvas,
    result: PbnResult,
    options: RenderOptions,
    origin_x_mm: float,
    origin_y_mm: float,
    draw_w_mm: float,
    draw_h_mm: float,
) -> dict[str, int]:
    working_w, working_h = result.working_size
    mm_per_px = draw_w_mm / working_w
    canvas.setFillColor(INK)

    numbered = skipped = 0
    skipped_area = 0
    for placement in result.placements:
        label = str(placement.colour_index + 1)
        size_mm = choose_font_size(
            placement,
            label,
            mm_per_px,
            preferred=options.number_size_mm,
            minimum=options.min_number_size_mm,
        )
        if size_mm is None:
            skipped += 1
            skipped_area += placement.area
            continue
        x_mm = origin_x_mm + (placement.x / working_w) * draw_w_mm
        y_mm = origin_y_mm + draw_h_mm - (placement.y / working_h) * draw_h_mm
        size_pt = mm_to_points(size_mm)
        canvas.setFont(NUMBER_FONT, size_pt)
        canvas.drawCentredString(
            mm_to_points(x_mm), mm_to_points(y_mm) - size_pt * _DIGIT_CENTRE, label
        )
        numbered += 1

    # Region count overstates the problem badly: a page can be 40% unnumbered
    # by count and 4% by area, because what goes unnumbered is always slivers.
    # Report the share of painted surface, which is what a painter feels.
    total_area = max(1, working_w * working_h)
    return {
        "numbered": numbered,
        "unnumbered": skipped,
        "unnumbered_area_percent": int(round(100.0 * skipped_area / total_area)),
    }


def _draw_legend_strip(
    canvas: pdfcanvas.Canvas,
    result: PbnResult,
    options: RenderOptions,
    *,
    x_mm: float,
    y_mm: float,
    width_mm: float,
    height_mm: float,
    max_swatch_mm: float = 6.4,
    hex_font_pt: float = 5.4,
) -> None:
    """Swatches in a grid, each carrying its own number in contrasting ink."""
    palette = result.palette
    used = sorted(int(i) for i in np.unique(result.regions.indices))
    if not used:
        return
    columns = max(1, min(options.legend_columns, len(used)))
    rows = -(-len(used) // columns)
    cell_w = width_mm / columns
    cell_h = max(6.0, (height_mm - 3.0) / rows)
    # Leave room for the hex code beside the swatch, not just under it.
    swatch = max(4.0, min(cell_h - 1.4, max_swatch_mm, cell_w * 0.45))
    light_ink = palette.ink_is_light()

    canvas.setFont(CAPTION_FONT, 6.0)
    canvas.setFillColor(MUTED)
    canvas.drawString(
        mm_to_points(x_mm),
        mm_to_points(y_mm + rows * cell_h + 0.8),
        f"{len(used)} colours - number, swatch, sRGB hex",
    )

    for position, colour_index in enumerate(used):
        column, row = position % columns, position // columns
        cx = x_mm + column * cell_w
        # Rows run downwards from the top of the strip.
        cy = y_mm + (rows - 1 - row) * cell_h
        rgb = palette.rgb[colour_index]

        canvas.setFillColorRGB(*(float(v) / 255.0 for v in rgb))
        canvas.setStrokeColor(MUTED)
        canvas.setLineWidth(0.3)
        canvas.rect(
            mm_to_points(cx), mm_to_points(cy), mm_to_points(swatch),
            mm_to_points(swatch), stroke=1, fill=1,
        )

        canvas.setFillColorRGB(*((1, 1, 1) if light_ink[colour_index] else (0, 0, 0)))
        canvas.setFont(LEGEND_FONT, mm_to_points(swatch * 0.5))
        canvas.drawCentredString(
            mm_to_points(cx + swatch / 2),
            mm_to_points(cy + swatch / 2) - mm_to_points(swatch * 0.5) * _DIGIT_CENTRE,
            str(colour_index + 1),
        )

        if options.show_hex:
            canvas.setFillColor(MUTED)
            canvas.setFont(CAPTION_FONT, hex_font_pt)
            canvas.drawString(
                mm_to_points(cx + swatch + 1.2),
                mm_to_points(cy + swatch / 2) - 1.9,
                hex_code(rgb),
            )


def _draw_legend_page(
    canvas: pdfcanvas.Canvas, result: PbnResult, options: RenderOptions
) -> None:
    layout = result.layout
    canvas.setFont(LEGEND_FONT, 13)
    canvas.setFillColor(INK)
    canvas.drawString(
        mm_to_points(layout.margin_mm),
        mm_to_points(layout.height_mm - layout.margin_mm - 5),
        "Colour key",
    )
    used_count = max(1, int(np.unique(result.regions.indices).size))
    columns = 3 if used_count <= 18 else 4
    rows = -(-used_count // columns)
    # A key on its own sheet should use the sheet. Fill the height rather than
    # printing postage stamps in the top corner.
    available = layout.height_mm - 2 * layout.margin_mm - 18
    row_height = min(24.0, available / rows)
    strip_height = rows * row_height
    _draw_legend_strip(
        canvas,
        result,
        RenderOptions(**{**options.__dict__, "legend_columns": columns}),
        x_mm=layout.margin_mm,
        y_mm=layout.height_mm - layout.margin_mm - 16 - strip_height,
        width_mm=layout.width_mm - 2 * layout.margin_mm,
        height_mm=strip_height,
        max_swatch_mm=min(18.0, row_height - 2.0),
        hex_font_pt=9.0,
    )


def _draw_reference_page(
    canvas: pdfcanvas.Canvas, result: PbnResult, options: RenderOptions
) -> None:
    """The posterised photo, so the painter can see what they are aiming at."""
    layout = result.layout
    preview = Image.fromarray(result.preview_rgb.astype(np.uint8), mode="RGB")
    aspect = preview.width / preview.height
    available_w = layout.width_mm - 2 * layout.margin_mm
    available_h = layout.height_mm - 2 * layout.margin_mm - 12
    width = available_w
    height = width / aspect
    if height > available_h:
        height, width = available_h, available_h * aspect

    canvas.setFont(LEGEND_FONT, 13)
    canvas.setFillColor(INK)
    canvas.drawString(
        mm_to_points(layout.margin_mm),
        mm_to_points(layout.height_mm - layout.margin_mm - 5),
        "Reference",
    )
    canvas.setFont(CAPTION_FONT, 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(
        mm_to_points(layout.margin_mm),
        mm_to_points(layout.height_mm - layout.margin_mm - 10),
        f"The photo reduced to {len(result.palette)} colours - what the finished page should look like.",
    )

    buffer = io.BytesIO()
    preview.save(buffer, format="PNG")
    buffer.seek(0)
    canvas.drawImage(
        ImageReader(buffer),
        mm_to_points(layout.margin_mm + (available_w - width) / 2),
        mm_to_points(layout.height_mm - layout.margin_mm - 14 - height),
        width=mm_to_points(width),
        height=mm_to_points(height),
    )


def _outline_reader(outline: np.ndarray) -> ImageReader:
    """Black lines on white, as a 1-bit PNG."""
    image = Image.fromarray(np.where(outline, 0, 255).astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    image.convert("1").save(buffer, format="PNG", optimize=True)
    buffer.seek(0)
    return ImageReader(buffer)


def save_preview_png(result: PbnResult, path: str, *, with_outline: bool = True) -> None:
    """A quick on-screen check of the same drawing, without opening a PDF."""
    preview = Image.fromarray(result.preview_rgb.astype(np.uint8), mode="RGB")
    if with_outline:
        outline = Image.fromarray(
            np.where(result.outline, 0, 255).astype(np.uint8), mode="L"
        ).resize(preview.size, Image.LANCZOS)
        preview = Image.composite(
            Image.new("RGB", preview.size, (0, 0, 0)), preview, Image.eval(outline, lambda v: 255 - v)
        )
    preview.save(path)
