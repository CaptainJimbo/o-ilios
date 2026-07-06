"""Dataset loading shared by the baseline and the U-Net."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

DATASET_ROOT = Path("data/dataset")

# Channel order in the stored PNGs (see pipeline/dataset.py).
CHANNELS = {171: 0, 193: 1, 304: 2}


@dataclass
class Sample:
    id: str
    image: np.ndarray  # (512, 512, 3) uint8, R/G/B = 171/193/304
    mask: np.ndarray   # (512, 512) uint8, 0 bg / 1 CH / 2 AR
    disk: np.ndarray   # (512, 512) bool, True inside the solar disk


def _disk_mask(cx: float, cy: float, r: float,
               shape: tuple[int, int]) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2


def load_split(split: str) -> list[Sample]:
    root = DATASET_ROOT / split
    samples = []
    for line in (root / "meta.jsonl").open():
        meta = json.loads(line)
        image = np.asarray(Image.open(root / "images" / f"{meta['id']}.png"))
        mask = np.asarray(Image.open(root / "masks" / f"{meta['id']}.png"))
        samples.append(Sample(
            id=meta["id"], image=image, mask=mask,
            disk=_disk_mask(meta["disk_cx"], meta["disk_cy"], meta["disk_r"],
                            mask.shape),
        ))
    return samples
