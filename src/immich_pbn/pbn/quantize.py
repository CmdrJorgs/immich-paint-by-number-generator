"""Palette reduction by k-means in Lab space.

Rolled by hand rather than pulled from scikit-learn: the algorithm is forty
lines, the dependency is eighty megabytes, and doing it here means the seed
genuinely controls the result end to end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .colour import (
    INK_CROSSOVER_LUMINANCE,
    hex_code,
    lab_to_rgb_bytes,
    relative_luminance,
)

#: Pixels sampled for the clustering itself. Beyond this the centroids stop
#: moving in any way a painter could see.
DEFAULT_SAMPLE = 60_000
#: Pixels assigned per chunk, to keep the distance matrix off the heap.
_ASSIGN_CHUNK = 250_000
#: Two centroids closer than this in Lab are the same paint. The just-noticeable
#: difference is around 2.3 dE, and a painter mixing by eye will do worse than
#: that, so near-duplicates only waste a legend slot and a number.
DEFAULT_MIN_SEPARATION = 2.5
#: Independent restarts, best-of kept by inertia. k-means lands in local
#: minima, and on a palette the failure is visible rather than statistical:
#: one unlucky start spent two of six slots on near-identical greens and lost
#: white from the picture entirely. Restarting is the standard remedy and
#: costs little, because only the subsample is clustered.
DEFAULT_RESTARTS = 4


@dataclass(frozen=True)
class Palette:
    """The chosen colours, ordered dark to light."""

    lab: np.ndarray  # (k, 3)
    rgb: np.ndarray  # (k, 3) uint8

    def __len__(self) -> int:
        return int(self.lab.shape[0])

    @property
    def hex_codes(self) -> list[str]:
        return [hex_code(c) for c in self.rgb]

    def ink_is_light(self) -> np.ndarray:
        """True where a swatch is dark enough that white text beats black."""
        return relative_luminance(self.rgb) < INK_CROSSOVER_LUMINANCE


def quantize(
    lab_image: np.ndarray,
    k: int,
    *,
    seed: int | None = None,
    sample_size: int = DEFAULT_SAMPLE,
    max_iter: int = 60,
    tolerance: float = 0.02,
    min_separation: float = DEFAULT_MIN_SEPARATION,
    restarts: int = DEFAULT_RESTARTS,
) -> tuple[np.ndarray, Palette]:
    """Reduce ``lab_image`` (H, W, 3) to ``k`` colours.

    Returns the per-pixel palette index and the palette itself.
    """
    height, width = lab_image.shape[:2]
    indices, palette = quantize_points(
        lab_image.reshape(-1, 3),
        k,
        seed=seed,
        sample_size=sample_size,
        max_iter=max_iter,
        tolerance=tolerance,
        min_separation=min_separation,
        restarts=restarts,
    )
    return indices.reshape(height, width), palette


def quantize_points(
    points: np.ndarray,
    k: int,
    *,
    seed: int | None = None,
    sample_size: int = DEFAULT_SAMPLE,
    max_iter: int = 60,
    tolerance: float = 0.02,
    min_separation: float = DEFAULT_MIN_SEPARATION,
    restarts: int = DEFAULT_RESTARTS,
) -> tuple[np.ndarray, Palette]:
    """Cluster an (N, 3) cloud of Lab colours into ``k`` paints.

    The unit of clustering is deliberately a parameter of the caller rather
    than baked in: for a traced page the points are pixels, and for a tiled one
    they are the mean colour of each cell. Clustering the cells directly gives
    a palette chosen to represent the things that will actually be painted,
    instead of one chosen for pixels and then voted on.
    """
    if k < 2:
        raise ValueError("a paint-by-number page needs at least 2 colours")
    flat = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if flat.shape[0] == 0:
        raise ValueError("nothing to quantize")

    rng = np.random.default_rng(seed)
    sample = _subsample(flat, sample_size, rng)

    unique_count = len(np.unique(np.round(sample, 2), axis=0))
    k = min(k, max(1, unique_count))

    centroids = None
    best_inertia = np.inf
    for _ in range(max(1, restarts)):
        candidate, inertia = _lloyd(sample, k, rng, max_iter=max_iter, tolerance=tolerance)
        if inertia < best_inertia:
            centroids, best_inertia = candidate, inertia
    assert centroids is not None

    centroids = _merge_indistinct(centroids, min_separation)
    order = np.argsort(centroids[:, 0])  # by lightness: 1 is the darkest colour
    centroids = centroids[order]

    indices = np.empty(flat.shape[0], dtype=np.int16)
    for start in range(0, flat.shape[0], _ASSIGN_CHUNK):
        block = flat[start : start + _ASSIGN_CHUNK]
        indices[start : start + _ASSIGN_CHUNK] = _assign(block, centroids)

    return indices, Palette(lab=centroids, rgb=lab_to_rgb_bytes(centroids))


def cell_mean_lab(
    lab_image: np.ndarray, cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Average colour of every cell, plus how many pixels each holds.

    Averaged in Lab rather than in sRGB. Averaging gamma-encoded sRGB darkens
    the result wherever a cell straddles an edge, which on a mosaic shows up as
    a grubby outline shadowing every real contour in the photo.

    Returns arrays indexed by cell id; ids with no pixels are left at zero, and
    the count array is how the caller tells those apart from genuine black.
    """
    flat = cells.ravel()
    size = int(flat.max()) + 1
    counts = np.bincount(flat, minlength=size)
    channels = [
        np.bincount(flat, weights=lab_image[..., channel].ravel().astype(np.float64), minlength=size)
        for channel in range(3)
    ]
    sums = np.stack(channels, axis=1)
    means = np.zeros((size, 3), dtype=np.float64)
    present = counts > 0
    means[present] = sums[present] / counts[present, None]
    return means, counts


def quantize_cells(
    lab_image: np.ndarray,
    cells: np.ndarray,
    k: int,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, Palette]:
    """Give every cell of a tiling one paint colour.

    Returns the palette index per cell id, the per-pixel index map, and the
    palette.
    """
    means, counts = cell_mean_lab(lab_image, cells)
    present = np.flatnonzero(counts)
    if present.size == 0:
        raise ValueError("the tiling produced no cells")

    assigned, palette = quantize_points(means[present], k, **kwargs)
    colour_of_cell = np.zeros(means.shape[0], dtype=np.int16)
    colour_of_cell[present] = assigned
    return colour_of_cell, colour_of_cell[cells], palette


def _subsample(flat: np.ndarray, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    if flat.shape[0] <= sample_size:
        return flat
    picks = rng.choice(flat.shape[0], size=sample_size, replace=False)
    return flat[picks]


def _lloyd(
    points: np.ndarray,
    k: int,
    rng: np.random.Generator,
    *,
    max_iter: int,
    tolerance: float,
) -> tuple[np.ndarray, float]:
    """One k-means run. Returns the centroids and their within-cluster error."""
    centroids = _kmeans_plus_plus(points, k, rng)
    labels = _assign(points, centroids)
    for _ in range(max_iter):
        moved = _update(points, labels, centroids, rng)
        shift = float(np.abs(moved - centroids).max())
        centroids = moved
        labels = _assign(points, centroids)
        if shift < tolerance:
            break
    inertia = float(_sq_dist(points, centroids[labels]).sum())
    return centroids, inertia


def _kmeans_plus_plus(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Seed centroids far apart, so the palette does not open with two greys."""
    centroids = np.empty((k, points.shape[1]), dtype=np.float64)
    centroids[0] = points[rng.integers(points.shape[0])]
    closest = _sq_dist(points, centroids[0])
    for i in range(1, k):
        total = float(closest.sum())
        if total <= 0:
            centroids[i] = points[rng.integers(points.shape[0])]
        else:
            centroids[i] = points[rng.choice(points.shape[0], p=closest / total)]
        closest = np.minimum(closest, _sq_dist(points, centroids[i]))
    return centroids


def _sq_dist(points: np.ndarray, centre: np.ndarray) -> np.ndarray:
    delta = points - centre
    return np.einsum("ij,ij->i", delta, delta)


def _assign(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2; the ||x||^2 term is constant per
    # row so it drops out of the argmin.
    dots = points @ centroids.T
    centroid_norms = np.einsum("ij,ij->i", centroids, centroids)
    return np.argmin(centroid_norms[None, :] - 2.0 * dots, axis=1).astype(np.int16)


def _update(
    points: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    k = centroids.shape[0]
    counts = np.bincount(labels, minlength=k)
    sums = np.zeros_like(centroids)
    np.add.at(sums, labels, points)

    updated = centroids.copy()
    populated = counts > 0
    updated[populated] = sums[populated] / counts[populated, None]

    # An empty cluster is a wasted colour. Restart it on whichever pixel is
    # currently worst served, which is where an extra colour helps most.
    empty = np.flatnonzero(~populated)
    if empty.size:
        residual = _sq_dist(points, updated[labels]) if points.size else None
        for idx in empty:
            if residual is None or not residual.size:
                updated[idx] = points[rng.integers(points.shape[0])]
                continue
            worst = int(np.argmax(residual))
            updated[idx] = points[worst]
            residual[worst] = 0.0
    return updated


def _merge_indistinct(centroids: np.ndarray, min_separation: float) -> np.ndarray:
    """Collapse centroids the eye cannot tell apart.

    A photo of an overcast sky will happily eat six palette slots on six blues
    that all mix to the same grey. Greedy single-link merging, cheap at these
    sizes (k is a couple of dozen at most).
    """
    if min_separation <= 0 or centroids.shape[0] < 2:
        return centroids
    kept: list[np.ndarray] = []
    for centre in centroids:
        if any(float(np.linalg.norm(centre - k)) < min_separation for k in kept):
            continue
        kept.append(centre)
    if len(kept) == centroids.shape[0]:
        return centroids
    # One survivor means the photo is effectively a single flat colour. Say so
    # with a real one-entry palette rather than faking a second identical paint;
    # the pipeline turns that into a clear error further up.
    return np.stack(kept)
