"""sRGB <-> CIELAB, because clustering in RGB gives ugly palettes.

RGB distance does not match how different two colours look. Cluster a sunset in
RGB and you get four nearly identical oranges and one blue; cluster it in Lab
and the oranges separate the way the eye separates them. Everything downstream
that measures "how close are these two colours" works in Lab for that reason.
"""

from __future__ import annotations

import numpy as np

# D65 reference white, the one sRGB is defined against.
_WHITE = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)

_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)

_DELTA = 6.0 / 29.0
_DELTA_CUBED = _DELTA**3
_THREE_DELTA_SQ = 3.0 * _DELTA**2


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0)
    return np.where(
        linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055
    )


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """``rgb`` in 0..1 with colour last; returns L in 0..100, a/b roughly -128..127."""
    linear = srgb_to_linear(rgb)
    xyz = linear @ _RGB_TO_XYZ.T / _WHITE
    f = np.where(
        xyz > _DELTA_CUBED, np.cbrt(xyz), xyz / _THREE_DELTA_SQ + 4.0 / 29.0
    )
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rgb_to_lab`, clipped into gamut."""
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f > _DELTA, f**3, (f - 4.0 / 29.0) * _THREE_DELTA_SQ) * _WHITE
    return np.clip(linear_to_srgb(xyz @ _XYZ_TO_RGB.T), 0.0, 1.0)


def rgb_bytes_to_lab(image: np.ndarray) -> np.ndarray:
    return rgb_to_lab(np.asarray(image, dtype=np.float64) / 255.0)


def lab_to_rgb_bytes(lab: np.ndarray) -> np.ndarray:
    return np.round(lab_to_rgb(lab) * 255.0).astype(np.uint8)


#: Background luminance at which white and black ink have equal WCAG contrast.
#: Solve (1.05)/(L+0.05) = (L+0.05)/0.05 for L: the crossover is 0.1791, not
#: the "about half" that eyeballing suggests. Guessing high here paints white
#: digits onto mid-olive swatches nobody can read.
INK_CROSSOVER_LUMINANCE = 0.1791


def relative_luminance(rgb_bytes: np.ndarray) -> np.ndarray:
    """WCAG relative luminance, used to decide black-or-white ink on a swatch."""
    linear = srgb_to_linear(np.asarray(rgb_bytes, dtype=np.float64) / 255.0)
    return linear @ np.array([0.2126, 0.7152, 0.0722])


def hex_code(rgb_bytes: np.ndarray) -> str:
    r, g, b = (int(v) for v in np.asarray(rgb_bytes).reshape(3))
    return f"#{r:02X}{g:02X}{b:02X}"
