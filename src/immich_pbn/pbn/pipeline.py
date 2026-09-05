"""Photo in, paint-by-number page data out."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage

from .colour import rgb_bytes_to_lab
from .numbering import NumberPlacement, place_numbers
from .page import PageLayout, mm_to_pixels
from .quantize import DEFAULT_MIN_SEPARATION, Palette, quantize
from .regions import RegionMap, area_for_min_width, build_regions

log = logging.getLogger(__name__)

#: Guard against someone passing --resolution 20000 and swapping to death.
MAX_WORKING_EDGE = 4000


class PipelineError(RuntimeError):
    pass


@dataclass
class PbnOptions:
    colours: int = 20
    #: Long edge of the working image in pixels -- the fidelity dial.
    #:
    #: Counter-intuitively, raising it yields *fewer* regions, not more
    #: (measured: 404 regions at 250 px against 304 at 752 px on the same
    #: photo). A hard downscale aliases fine texture into blocky noise that
    #: survives as dozens of separate blobs; sample the picture properly and
    #: the same area resolves into one clean shape with a truthful edge. The
    #: number of regions is governed by ``min_region_mm``, which is pinned to
    #: printed millimetres and so does not drift when this changes.
    #:
    #: Only ever downscales. Set it above the source photo's own long edge and
    #: it does nothing, because upscaling would invent detail that is not there.
    resolution: int = 1400
    #: Output raster density for the printed outline.
    dpi: int = 300
    seed: int | None = None
    #: Nothing narrower than this survives on paper, in printed millimetres.
    #: 3 mm is about the narrowest band a person can fill with a brush without
    #: swearing; below roughly 2 mm the page turns into confetti.
    min_region_mm: float = 3.0
    #: Median-filter radius applied before quantization; kills JPEG mosquito
    #: noise and film grain that would otherwise become thousands of specks.
    smoothing: int = 2
    line_width_mm: float = 0.28
    min_colour_separation: float = DEFAULT_MIN_SEPARATION
    number_size_mm: float = 2.6
    min_number_size_mm: float = 1.5
    #: "auto" picks whichever orientation wastes less paper for this photo.
    orientation: str = "auto"


@dataclass
class PbnResult:
    palette: Palette
    regions: RegionMap
    placements: list[NumberPlacement]
    #: Boolean outline at print resolution; True where ink goes.
    outline: np.ndarray
    #: The quantized image at working resolution, for the reference page.
    preview_rgb: np.ndarray
    layout: PageLayout
    drawing_size_mm: tuple[float, float]
    working_size: tuple[int, int]
    source_size: tuple[int, int]
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def numbered_count(self) -> int:
        return int(self.stats.get("numbered", 0))


def decode_image(data: bytes | str, *, max_edge: int) -> tuple[Image.Image, tuple[int, int]]:
    """Decode once, honour EXIF rotation, downscale, and report the original size.

    Returns the working image and the size the photo had before downscaling.
    Decoding once matters: the source may be a 45-megapixel original, and doing
    it twice just to learn its dimensions costs more than the rest of the
    pipeline put together.
    """
    source = io.BytesIO(data) if isinstance(data, bytes) else data
    try:
        image = Image.open(source)
        image.load()
    except Exception as exc:
        raise PipelineError(f"could not decode the image: {exc}") from exc

    image = ImageOps.exif_transpose(image)
    source_size = image.size
    if image.mode != "RGB":
        image = image.convert("RGB")

    max_edge = max(200, min(int(max_edge), MAX_WORKING_EDGE))
    longest = max(image.size)
    if longest > max_edge:
        scale = max_edge / longest
        new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(new_size, Image.LANCZOS)
    return image, source_size


def load_image(data: bytes | str, *, max_edge: int) -> Image.Image:
    """Just the working image, for callers that do not care about the original size."""
    return decode_image(data, max_edge=max_edge)[0]


def generate(
    image_data: bytes | str | Image.Image,
    options: PbnOptions,
    layout: PageLayout,
) -> PbnResult:
    """Run the whole pipeline for one photo."""
    if isinstance(image_data, Image.Image):
        image = image_data if image_data.mode == "RGB" else image_data.convert("RGB")
        source_size = image.size
    else:
        image, source_size = decode_image(image_data, max_edge=options.resolution)

    if max(source_size) < options.resolution:
        log.info(
            "source photo is %dx%d, smaller than --resolution %d; using the "
            "photo's own size (upscaling would only invent detail)",
            source_size[0], source_size[1], options.resolution,
        )

    aspect = image.width / image.height
    if options.orientation == "auto":
        layout = layout.orientation_for(aspect)
    else:
        layout = replace(layout, landscape=options.orientation == "landscape")
    drawing_w_mm, drawing_h_mm = layout.fitted_drawing_mm(aspect)

    smoothed = _smooth(image, options.smoothing)
    rgb = np.asarray(smoothed, dtype=np.uint8)

    log.debug("quantizing %sx%s to %s colours", image.width, image.height, options.colours)
    indices, palette = quantize(
        rgb_bytes_to_lab(rgb),
        options.colours,
        seed=options.seed,
        min_separation=options.min_colour_separation,
    )
    if len(palette) < 2:
        raise PipelineError(
            "this photo reduces to a single colour -- there is nothing to paint. "
            "Try a different image, or raise --colours if you lowered it."
        )

    min_area = area_for_min_width(options.min_region_mm, drawing_w_mm, image.width)
    regions = build_regions(indices, palette.lab, min_area=min_area)
    placements = place_numbers(regions)

    out_w = mm_to_pixels(drawing_w_mm, options.dpi)
    out_h = mm_to_pixels(drawing_h_mm, options.dpi)
    outline = _render_outline(regions.labels, (out_w, out_h), options, drawing_w_mm)

    preview_rgb = palette.rgb[np.clip(regions.indices, 0, len(palette) - 1)]

    used = np.unique(regions.indices)
    stats = {
        "regions": regions.count,
        "merge_passes": regions.merge_passes,
        "colours_used": int(used.size),
        "min_region_px": min_area,
        "outline_pixels": int(outline.sum()),
        "resolution_capped": bool(max(source_size) < options.resolution),
    }
    return PbnResult(
        palette=palette,
        regions=regions,
        placements=placements,
        outline=outline,
        preview_rgb=preview_rgb,
        layout=layout,
        drawing_size_mm=(drawing_w_mm, drawing_h_mm),
        working_size=(image.width, image.height),
        source_size=source_size,
        stats=stats,
    )


def _smooth(image: Image.Image, radius: int) -> Image.Image:
    """Median first, then a whisper of blur.

    Median is the right tool here because it removes speckle without dragging
    edges around the way a Gaussian does -- and edges are exactly what becomes
    the outline. The trailing half-pixel blur only takes the hard corner off
    the median's own output so quantization does not re-introduce stair steps.
    """
    if radius <= 0:
        return image
    size = 2 * int(radius) + 1
    out = image.filter(ImageFilter.MedianFilter(size=size))
    return out.filter(ImageFilter.GaussianBlur(radius=0.5))


def _render_outline(
    labels: np.ndarray,
    output_size: tuple[int, int],
    options: PbnOptions,
    drawing_width_mm: float,
) -> np.ndarray:
    """Upscale the label map to print size and draw its edges at a printable weight.

    Upscaling the *labels* rather than an already-drawn outline is what keeps
    the lines crisp: a line drawn small and then enlarged is a blurry, uneven
    line, while a boundary recomputed after enlargement is exactly one pixel
    wide at the target density before it is deliberately thickened.
    """
    out_w, out_h = output_size
    scaled = np.asarray(
        Image.fromarray(labels.astype(np.int32), mode="I").resize(
            (out_w, out_h), Image.NEAREST
        ),
        dtype=np.int32,
    )

    edge = np.zeros(scaled.shape, dtype=bool)
    edge[:, :-1] |= scaled[:, :-1] != scaled[:, 1:]
    edge[:-1, :] |= scaled[:-1, :] != scaled[1:, :]

    target_px = options.line_width_mm / drawing_width_mm * out_w
    extra = int(round((target_px - 1.0) / 2.0))
    if extra > 0:
        edge = ndimage.binary_dilation(edge, structure=_disk(extra))

    # A border round the whole drawing, so the page has a frame to paint up to.
    edge[:1, :] = edge[-1:, :] = True
    edge[:, :1] = edge[:, -1:] = True
    if extra > 0:
        edge[:extra, :] = edge[-extra:, :] = True
        edge[:, :extra] = edge[:, -extra:] = True
    return edge


def _disk(radius: int) -> np.ndarray:
    span = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(span, span, indexing="ij")
    return (yy**2 + xx**2) <= radius**2 + 0.5
