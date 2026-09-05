"""Turn a quantized image into paintable regions.

Quantization alone does not give you a paint-by-number page. It gives you a
posterised photo whose "regions" include ten thousand two-pixel specks along
every edge -- unpaintable, unnumberable, and they turn the printed page into
grey mush. Making those specks go away, without dissolving the shapes that
carry the picture, is most of the work in here.

The rule used: a region smaller than the paintable minimum is absorbed into
whichever neighbour it shares the most border with, nudged towards neighbours
of a similar colour so a dark eyelash does not get swallowed by a bright cheek
when a closer mid-tone was available. Absorption changes colours, which can
merge previously separate regions, so it runs to a fixed point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

#: 4-connectivity. Diagonal touches deliberately do not join regions: two
#: squares meeting at a corner read as two shapes to a person with a brush.
_CONNECTIVITY = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

#: How fast the colour penalty falls off, in Lab dE. Around 15 dE is "clearly a
#: different colour but the same family", which is the right scale for deciding
#: that a similar neighbour is worth a shorter shared border.
_COLOUR_SCALE = 15.0


@dataclass
class RegionMap:
    """Final regions, one integer label per pixel, labels numbered from 1.

    Labels need not be contiguous. Connected-component labelling produces a
    dense run, but a geometric tiling derives each cell's id from lattice
    arithmetic, so ids outside the picture simply never appear. Everything here
    is therefore driven by which ids actually have pixels rather than by the
    highest id present -- assuming density would silently number the wrong
    regions on a tiled page.
    """

    labels: np.ndarray  # (H, W) int32
    colour_of_region: np.ndarray  # (max_label + 1,) int16, index 0 unused
    areas: np.ndarray  # (max_label + 1,) int64
    indices: np.ndarray  # (H, W) int16 palette index per pixel
    merge_passes: int = 0

    @property
    def count(self) -> int:
        """How many regions actually carry pixels."""
        return int(np.count_nonzero(self.areas[1:]))

    @property
    def region_ids(self) -> np.ndarray:
        return np.flatnonzero(self.areas[1:]) + 1

    def boundary_mask(self, *, include_frame: bool = True) -> np.ndarray:
        """Pixels that sit on the edge between two regions."""
        labels = self.labels
        edge = np.zeros(labels.shape, dtype=bool)
        edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
        edge[:, 1:] |= labels[:, :-1] != labels[:, 1:]
        edge[:-1, :] |= labels[:-1, :] != labels[1:, :]
        edge[1:, :] |= labels[:-1, :] != labels[1:, :]
        if include_frame:
            edge[0, :] = edge[-1, :] = True
            edge[:, 0] = edge[:, -1] = True
        return edge


def build_regions(
    indices: np.ndarray,
    palette_lab: np.ndarray,
    *,
    min_area: int,
    max_passes: int = 8,
) -> RegionMap:
    """Label ``indices`` into connected regions, absorbing anything under ``min_area``."""
    working = indices.astype(np.int16, copy=True)
    passes = 0
    for passes in range(1, max_passes + 1):
        labels, colour_of, areas = _label_by_colour(working)
        small = np.flatnonzero(areas[1:] < min_area) + 1
        if small.size == 0:
            return RegionMap(labels, colour_of, areas, working, passes - 1)
        merged = _absorb(working, labels, colour_of, areas, small, palette_lab)
        if not merged:
            # Nothing could be absorbed -- every small region is isolated, which
            # in practice means the whole image is one region. Stop rather than
            # spin.
            return RegionMap(labels, colour_of, areas, working, passes)

    labels, colour_of, areas = _label_by_colour(working)
    return RegionMap(labels, colour_of, areas, working, passes)


def _label_by_colour(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Connected components, where connectivity requires the same palette index."""
    labels = np.zeros(indices.shape, dtype=np.int32)
    colour_of = [np.int16(-1)]  # slot 0 is unused so labels start at 1
    offset = 0
    for colour in np.unique(indices):
        mask = indices == colour
        component, found = ndimage.label(mask, structure=_CONNECTIVITY)
        if not found:
            continue
        labels[mask] = component[mask] + offset
        colour_of.extend([np.int16(colour)] * found)
        offset += found
    areas = np.bincount(labels.ravel(), minlength=offset + 1).astype(np.int64)
    return labels, np.asarray(colour_of, dtype=np.int16), areas


def _adjacency(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every touching pair of regions and how many pixel-edges they share."""
    pairs = []
    for a, b in (
        (labels[:, :-1].ravel(), labels[:, 1:].ravel()),
        (labels[:-1, :].ravel(), labels[1:, :].ravel()),
    ):
        differs = a != b
        if differs.any():
            pairs.append(np.stack([a[differs], b[differs]], axis=1))
    if not pairs:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    stacked = np.concatenate(pairs, axis=0).astype(np.int64)
    low = np.minimum(stacked[:, 0], stacked[:, 1])
    high = np.maximum(stacked[:, 0], stacked[:, 1])
    span = int(labels.max()) + 1
    keys, counts = np.unique(low * span + high, return_counts=True)
    return keys // span, keys % span, counts


def _absorb(
    working: np.ndarray,
    labels: np.ndarray,
    colour_of: np.ndarray,
    areas: np.ndarray,
    small: np.ndarray,
    palette_lab: np.ndarray,
) -> bool:
    """Recolour each too-small region to its best neighbour. Mutates ``working``.

    Merges are recorded as parent pointers and resolved at the end rather than
    applied one at a time. Two reasons, one of correctness and one of speed:
    a speck whose chosen neighbour is itself a speck must end up with whatever
    colour that chain finally settles on (applying eagerly strands it with a
    colour nobody else has, which is how a "converged" pass leaves single-pixel
    regions behind), and rewriting the image once at the end turns a per-region
    full-array scan into a single lookup.
    """
    left, right, shared = _adjacency(labels)
    if left.size == 0:
        return False

    # Neighbour lists as a CSR-style structure indexed by region id.
    both_self = np.concatenate([left, right])
    both_other = np.concatenate([right, left])
    both_shared = np.concatenate([shared, shared])
    order = np.argsort(both_self, kind="stable")
    all_self = both_self[order]
    all_other = both_other[order]
    all_shared = both_shared[order].astype(np.float64)
    starts = np.searchsorted(all_self, np.arange(len(areas) + 1))

    parent = np.arange(len(areas), dtype=np.int64)

    def root(region: int) -> int:
        while parent[region] != region:
            parent[region] = parent[parent[region]]  # path halving
            region = int(parent[region])
        return region

    # Smallest first: a speck should land in a real shape, not in another speck.
    small_sorted = small[np.argsort(areas[small], kind="stable")]
    merged_area = areas.astype(np.int64).copy()
    changed = False

    for region in small_sorted:
        region = int(region)
        if root(region) != region:
            continue  # already absorbed by an earlier merge in this pass
        begin, end = starts[region], starts[region + 1]
        if begin == end:
            continue
        neighbours = all_other[begin:end]
        borders = all_shared[begin:end]

        roots = np.array([root(int(n)) for n in neighbours], dtype=np.int64)
        usable = roots != region  # a neighbour that already resolves to us is a cycle
        if not usable.any():
            continue
        neighbours, borders, roots = neighbours[usable], borders[usable], roots[usable]

        delta_e = np.linalg.norm(
            palette_lab[colour_of[roots]] - palette_lab[colour_of[region]], axis=1
        )
        # Longest shared border wins, discounted when the neighbour is a very
        # different colour, and nudged towards neighbours large enough to
        # survive this pass themselves.
        score = borders / (1.0 + delta_e / _COLOUR_SCALE)
        score *= 1.0 + 0.15 * (merged_area[roots] >= merged_area[region])
        winner = int(roots[int(np.argmax(score))])

        parent[region] = winner
        merged_area[winner] += merged_area[region]
        changed = True

    if not changed:
        return False

    # Resolve every region to its final root, then rewrite the image in one pass.
    final_colour = colour_of.copy()
    for region in range(1, len(areas)):
        final_colour[region] = colour_of[root(region)]
    working[...] = final_colour[labels]
    return True


def area_for_min_width(
    min_width_mm: float, printed_width_mm: float, pixel_width: int
) -> int:
    """Convert "no blob narrower than N mm" into a pixel-area threshold.

    Pixels are the wrong unit for this decision -- the same 40-pixel speck is
    fine on a 600px working image and invisible on a 3000px one. What actually
    matters is how big the shape lands on the paper, so the threshold is
    specified in millimetres of printed width and converted here.

    The area of a square of that width is used as the yardstick; real regions
    are rarely square, but it is the right order of magnitude and it errs
    towards keeping thin-but-long shapes like a branch or a horizon line.
    """
    if printed_width_mm <= 0 or pixel_width <= 0:
        raise ValueError("printed width and pixel width must both be positive")
    px_per_mm = pixel_width / printed_width_mm
    return max(4, int(round((min_width_mm * px_per_mm) ** 2)))
