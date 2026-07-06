"""The Living Sun worker. Runs on a schedule (GitHub Actions cron now,
Lambda at step 5); no server anywhere.

Fetches JSOC's near-real-time synoptic AIA FITS (updated ~every 15 min),
runs the ONNX U-Net on CPU, and writes static artifacts for the frontend:

    web/public/live/sun_171.png     colorized suns (official AIA palettes),
    web/public/live/sun_193.png     1024px, north up — the hero imagery and
    web/public/live/sun_304.png     the wavelength-morph endpoints
    web/public/live/mask.png        512px RGBA overlay (CH cyan / AR amber),
                                    same source frame => pixel-aligned
    web/public/live/meta.json       observation time + coverage stats

Rendering our own suns from the same FITS the mask came from (instead of
NASA's browse JPEGs) guarantees overlay alignment and identical freshness.

Usage:
    python -m worker.run
"""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import onnxruntime as ort
import requests
from astropy.io import fits
from PIL import Image

from pipeline.preprocess import model_input, stretch

log = logging.getLogger(__name__)

MOSTRECENT = "https://jsoc1.stanford.edu/data/aia/synoptic/mostrecent"
WAVELENGTHS = (171, 193, 304)
ONNX_PATH = Path("checkpoints/unet.onnx")
OUT_DIR = Path("web/public/live")

# Class id -> RGBA, the Living Sun palette (CH cyan, AR amber).
MASK_COLORS = {1: (0, 230, 255, 110), 2: (255, 190, 0, 110)}


def fetch_latest() -> tuple[dict[int, np.ndarray], str]:
    """Latest synoptic FITS arrays by wavelength + observation ISO time."""
    arrays: dict[int, np.ndarray] = {}
    obs_time = ""
    for wl in WAVELENGTHS:
        resp = requests.get(f"{MOSTRECENT}/AIAsynoptic{wl:04d}.fits", timeout=60)
        resp.raise_for_status()
        with fits.open(io.BytesIO(resp.content)) as hdul:
            hdu = hdul[-1]
            arrays[wl] = np.asarray(hdu.data, dtype=np.float32)
            obs_time = obs_time or str(hdu.header.get("DATE-OBS", ""))
    log.info("fetched %s wavelengths, obs time %s", len(arrays), obs_time)
    return arrays, obs_time


def render_suns(arrays: dict[int, np.ndarray], out_dir: Path) -> None:
    """1024px colorized PNGs with the official AIA palettes, north up."""
    import sunpy.visualization.colormaps  # registers sdoaia* cmaps  # noqa: F401

    for wl, data in arrays.items():
        cmap = matplotlib.colormaps[f"sdoaia{wl}"]
        shade = stretch(np.nan_to_num(data)).astype(np.float32) / 255.0
        rgb = (cmap(shade)[..., :3] * 255).astype(np.uint8)
        # FITS row 0 is south; PNG row 0 is top — flip to put north up.
        Image.fromarray(np.flipud(rgb)).save(out_dir / f"sun_{wl}.png")


def render_mask(mask: np.ndarray, out_dir: Path) -> None:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    for class_id, color in MASK_COLORS.items():
        rgba[mask == class_id] = color
    Image.fromarray(np.flipud(rgba)).save(out_dir / "mask.png")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    arrays, obs_time = fetch_latest()
    x = model_input(arrays).astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]

    session = ort.InferenceSession(ONNX_PATH,
                                   providers=["CPUExecutionProvider"])
    logits = session.run(None, {"image": x})[0]
    mask = logits.argmax(1)[0].astype(np.uint8)

    render_suns(arrays, OUT_DIR)
    render_mask(mask, OUT_DIR)
    meta = {
        "observation_time": obs_time,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coronal_hole_pct": round(float(np.mean(mask == 1)) * 100, 2),
        "active_region_pct": round(float(np.mean(mask == 2)) * 100, 2),
        "source": "SDO/AIA via JSOC synoptic (NASA); segmentation: o-ilios U-Net",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("artifacts -> %s | %s", OUT_DIR, meta)


if __name__ == "__main__":
    main()
