import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def photo_like() -> Image.Image:
    """A synthetic image with real structure: smooth regions, edges, and grain.

    Flat colour blocks would let a broken pipeline look healthy, so this has
    gradients (which quantization must band), hard edges (which must survive
    smoothing) and noise (which must not become ten thousand specks).
    """
    rng = np.random.default_rng(20240501)
    height, width = 240, 320
    yy, xx = np.mgrid[0:height, 0:width]

    sky = np.stack([80 + yy * 0.35, 130 + yy * 0.3, 200 - yy * 0.1], axis=-1)
    ground = np.stack([np.full_like(yy, 90), np.full_like(yy, 120), np.full_like(yy, 60)], axis=-1)
    image = np.where((yy > 150)[..., None], ground, sky).astype(float)

    disc = (xx - 210) ** 2 + (yy - 70) ** 2 < 45**2
    image[disc] = (250, 215, 90)
    trunk = (abs(xx - 90) < 9) & (yy > 95) & (yy < 200)
    image[trunk] = (70, 45, 30)

    image += rng.normal(0, 6, image.shape)
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB")


@pytest.fixture
def photo_bytes(photo_like: Image.Image) -> bytes:
    import io

    buffer = io.BytesIO()
    photo_like.save(buffer, format="JPEG", quality=88)
    return buffer.getvalue()
