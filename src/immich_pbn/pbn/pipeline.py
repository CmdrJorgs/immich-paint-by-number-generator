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
from .quantize import DEFAULT_MIN_SEPARATION, Palette, quantize, quantize_cells
from .regions import RegionMap, area_for_min_width, build_regions
from .tessellation import Tessellation, build_tessellation

log = logging.getLogger(__name__)

#: Guard against someone passing --resolution 20000 and swapping to death.
MAX_WORKING_EDGE = 4000


class PipelineError(RuntimeError):
    pass


#: "contour" traces the picture's own colour fields. The rest impose a lattice
#: on it and let each cell take the average colour underneath.
CONTOUR_STYLE = "contour"


@dataclass
class PbnOptions:
    colours: int = 20
    #: "contour", or any name from the tessellation registry.
    style: str = CONTOUR_STYLE
    #: Cell size for a tiled style, in printed millimetres. Ignored by
    #: "contour", which takes its shapes from the photograph instead.
    cell_mm: float = 8.0
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
    #: Digit height actually to be used, in mm. On a tiled page this grows with
    #: the cell: 2.6 mm digits look right in an 8 mm square and ridiculous in a
    #: 20 mm one, and the cell size is the only thing that knows which it is.
    number_size_mm: float = 2.6
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
    lab = rgb_bytes_to_lab(rgb)

    out_w = mm_to_pixels(drawing_w_mm, options.dpi)
    out_h = mm_to_pixels(drawing_h_mm, options.dpi)

    if options.style == CONTOUR_STYLE:
        regions, palette, extra = _contour_regions(lab, image, options, drawing_w_mm)
        outline_labels = _upscale_labels(regions.labels, (out_w, out_h))
    else:
        tiling = build_tessellation(options.style)
        cell_fraction = tiling.snap(options.cell_mm / drawing_w_mm)
        # The page's true aspect, in millimetres, handed to both evaluations.
        # Letting each infer it from its own pixel dimensions is what puts an
        # unnumbered row of cells along the bottom edge -- see Tessellation.counts.
        aspect = drawing_h_mm / drawing_w_mm
        regions, palette, extra = _tiled_regions(
            lab, tiling, cell_fraction, aspect, options
        )
        extra["cell_mm"] = round(cell_fraction * drawing_w_mm, 2)
        extra["inradius_mm"] = round(
            extra["cell_mm"] * tiling.inradius_factor, 2
        )
        # Re-evaluated rather than enlarged: the tiling is a pure function of
        # position, so asking it for the print grid directly gives exact edges
        # instead of a magnified copy of the small one.
        outline_labels = tiling.cell_map(out_w, out_h, cell_fraction, aspect)
        # The invariant the tiled styles stand on: the lattice the edges are
        # drawn from must be the same lattice the colours and numbers came
        # from, or the page grows cells that are outlined but never coloured.
        # Pinning the counts to the page's millimetres gets this right to
        # within a couple of corner pixels, which this makes exact.
        outline_labels = _fold_strays(outline_labels, regions.region_ids)
        extra["outline_cells"] = int(np.unique(outline_labels).size)

    placements = place_numbers(regions)
    outline = _edges_from_labels(outline_labels, options, drawing_w_mm)

    preview_rgb = palette.rgb[np.clip(regions.indices, 0, len(palette) - 1)]

    used = np.unique(regions.indices)
    stats = {
        "style": options.style,
        "regions": regions.count,
        "merge_passes": regions.merge_passes,
        "colours_used": int(used.size),
        "outline_pixels": int(outline.sum()),
        "resolution_capped": bool(max(source_size) < options.resolution),
    }
    stats.update(extra)
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
        # Scaled off the inscribed radius rather than the nominal cell size:
        # a 14 mm triangle and a 14 mm square are not remotely the same amount
        # of room for a digit, and the circle that fits inside is what decides.
        number_size_mm=max(
            options.number_size_mm, 0.62 * float(stats.get("inradius_mm", 0.0))
        ),
        stats=stats,
    )


def _contour_regions(
    lab: np.ndarray,
    image: Image.Image,
    options: PbnOptions,
    drawing_w_mm: float,
) -> tuple[RegionMap, Palette, dict[str, Any]]:
    """The original route: cluster pixels, trace colour fields, dissolve specks."""
    log.debug("quantizing %sx%s to %s colours", image.width, image.height, options.colours)
    indices, palette = quantize(
        lab,
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
    return regions, palette, {"min_region_px": min_area}


def _tiled_regions(
    lab: np.ndarray,
    tiling: Tessellation,
    cell_fraction: float,
    aspect: float,
    options: PbnOptions,
) -> tuple[RegionMap, Palette, dict[str, Any]]:
    """A lattice route: fixed cells, each painted its own average colour.

    No merging happens here, and that is the point rather than an omission. A
    mosaic's whole appeal is the visible grid; absorbing a cell into its
    same-coloured neighbour would grow exactly the organic blobs the tiled
    styles exist to avoid. Every cell stays its own region and gets its own
    number, even when four in a row share a paint.
    """
    height, width = lab.shape[:2]
    cells = tiling.cell_map(width, height, cell_fraction, aspect)
    colour_of_cell, per_pixel, palette = quantize_cells(
        lab,
        cells,
        options.colours,
        seed=options.seed,
        min_separation=options.min_colour_separation,
    )
    if len(palette) < 2:
        raise PipelineError(
            "this photo reduces to a single colour -- there is nothing to paint. "
            "Try a different image, or raise --colours if you lowered it."
        )
    areas = np.bincount(cells.ravel(), minlength=colour_of_cell.size).astype(np.int64)
    regions = RegionMap(
        labels=cells,
        colour_of_region=colour_of_cell,
        areas=areas,
        indices=per_pixel,
    )
    return regions, palette, {}


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


def _fold_strays(labels: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Absorb any cell the colour pass never saw into its nearest real neighbour.

    A cell can exist in the print raster and not the working one when it is
    only a pixel or two across -- the corners of a sheared triangle lattice do
    this. Such a cell is smaller than the frame line drawn over it, so nobody
    would ever see it, but leaving it in means the page contains an outlined
    region with no colour and no number, and "nobody would see it" is a much
    weaker guarantee than "it is not there".
    """
    stray = ~np.isin(labels, known)
    if not stray.any():
        return labels
    # Indices of the nearest non-stray pixel, which is the cell to join.
    _, (rows, columns) = ndimage.distance_transform_edt(stray, return_indices=True)
    return labels[rows, columns]


def _upscale_labels(labels: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour, so boundaries stay boundaries.

    Upscaling the *labels* rather than an already-drawn outline is what keeps
    the lines crisp: a line drawn small and then enlarged is a blurry, uneven
    line, while a boundary recomputed after enlargement is exactly one pixel
    wide at the target density before it is deliberately thickened.
    """
    out_w, out_h = output_size
    return np.asarray(
        Image.fromarray(labels.astype(np.int32), mode="I").resize(
            (out_w, out_h), Image.NEAREST
        ),
        dtype=np.int32,
    )


def _edges_from_labels(
    labels: np.ndarray,
    options: PbnOptions,
    drawing_width_mm: float,
) -> np.ndarray:
    """Draw the boundaries of a label map at a printable line weight."""
    out_h, out_w = labels.shape

    edge = np.zeros(labels.shape, dtype=bool)
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    edge[:-1, :] |= labels[:-1, :] != labels[1:, :]

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
