"""Shared preprocessing — the single definition used by both dataset assembly
and the live worker, so training and inference inputs can never drift."""

from __future__ import annotations

import numpy as np


def stretch(data: np.ndarray, a: float = 0.01) -> np.ndarray:
    """Per-frame asinh stretch to uint8 (matches AsinhStretch(0.01))."""
    top = np.percentile(data, 99.9)
    x = np.clip(data, 0, top) / max(top, 1e-6)
    y = np.arcsinh(x / a) / np.arcsinh(1 / a)
    return (y * 255).astype(np.uint8)


def downsample_mean(img: np.ndarray) -> np.ndarray:
    """1024 -> 512 by 2x2 block mean."""
    h, w = img.shape
    return img.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def model_input(fits_data_by_wavelength: dict[int, np.ndarray]) -> np.ndarray:
    """(512, 512, 3) uint8 stack, channels = 171/193/304, from raw FITS arrays."""
    return np.stack(
        [stretch(downsample_mean(
            np.nan_to_num(fits_data_by_wavelength[wl].astype(np.float32))))
         for wl in (171, 193, 304)],
        axis=-1)
