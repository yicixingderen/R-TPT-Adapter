"""Deterministic digital lighting shifts for NEU-CLS evaluation."""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image, ImageEnhance


def add_local_glare(image: Image.Image, reference: Image.Image) -> Image.Image:
    """Blend a white elliptical highlight with a small saturated center."""
    width, height = image.size
    digest = hashlib.blake2b(reference.tobytes(), digest_size=8).digest()
    horizontal = int.from_bytes(digest[:4], "little") / 2 ** 32
    vertical = int.from_bytes(digest[4:], "little") / 2 ** 32
    center_x = width * (0.30 + 0.40 * horizontal)
    center_y = height * (0.30 + 0.40 * vertical)
    yy, xx = np.ogrid[:height, :width]
    distance = (((xx - center_x) / (0.16 * width)) ** 2
                + ((yy - center_y) / (0.11 * height)) ** 2)
    opacity = np.clip(1.5 * np.exp(-0.5 * distance), 0.0, 1.0)
    pixels = np.asarray(image, dtype=np.float32)
    blended = pixels * (1.0 - opacity[..., None]) + 255.0 * opacity[..., None]
    return Image.fromarray(np.uint8(np.rint(blended)))


def apply_condition(image: Image.Image, name: str) -> Image.Image:
    """Apply one fixed lighting proxy to an image."""
    if name == "dark_strong":
        return ImageEnhance.Brightness(image).enhance(0.55)
    if name == "bright_strong":
        return ImageEnhance.Brightness(image).enhance(1.55)
    if name == "glare":
        return add_local_glare(image, image)
    if name == "dark_glare":
        return add_local_glare(
            ImageEnhance.Brightness(image).enhance(0.55), image)
    if name == "bright_glare":
        return add_local_glare(
            ImageEnhance.Brightness(image).enhance(1.55), image)
    raise ValueError(name)
