"""Generate the time-lapse sequence for the frontend scrubber.

Curated story rather than a rolling window (the definitive synoptic archive
lags ~a week, so "last 10 days" isn't fetchable): AR 13664 crossing the disk,
2024-05-04 -> 2024-05-14 — the active region behind the Gannon superstorm,
the strongest geomagnetic storm in 20 years. Every frame is segmented by the
U-Net, so the scrubber shows the model tracking the region across rotation.

Outputs (512px, sized for the web):
    web/public/timelapse/sun_<ts>.jpg    193 A display render
    web/public/timelapse/mask_<ts>.png   RGBA overlay
    web/public/timelapse/index.json      frame list + stats

Usage:
    python -m pipeline.timelapse
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
import numpy as np
import onnxruntime as ort
import sunpy.map
from PIL import Image

from pipeline.fetch import WAVELENGTHS, fetch_day
from pipeline.preprocess import model_input
from worker.run import MASK_COLORS, ONNX_PATH, display_shade

log = logging.getLogger(__name__)

START = datetime(2024, 5, 4)
DAYS = 11
EVERY_HOURS = 6
OUT_DIR = Path("web/public/timelapse")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    import sunpy.visualization.colormaps  # registers sdoaia193  # noqa: F401

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(ONNX_PATH,
                                   providers=["CPUExecutionProvider"])
    cmap = matplotlib.colormaps["sdoaia193"]

    frames = []
    for d in range(DAYS):
        day = START + timedelta(days=d)
        day_frames = fetch_day(day, Path("data/raw") / day.date().isoformat(),
                               every_hours=EVERY_HOURS)
        for t, paths in sorted(day_frames.items()):
            arrays = {wl: np.asarray(sunpy.map.Map(paths[wl]).data,
                                     dtype=np.float32)
                      for wl in WAVELENGTHS}
            x = model_input(arrays).astype(np.float32) / 255.0
            logits = session.run(None, {"image": x.transpose(2, 0, 1)[None]})[0]
            mask = logits.argmax(1)[0].astype(np.uint8)

            ts = f"{t:%Y%m%dT%H%M}"
            shade = display_shade(arrays[193])[::2, ::2]
            rgb = (cmap(shade)[..., :3] * 255).astype(np.uint8)
            Image.fromarray(np.flipud(rgb)).save(
                OUT_DIR / f"sun_{ts}.jpg", quality=82)

            rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
            for class_id, color in MASK_COLORS.items():
                rgba[mask == class_id] = color
            Image.fromarray(np.flipud(rgba)).save(OUT_DIR / f"mask_{ts}.png")

            frames.append({
                "id": ts,
                "time": t.isoformat(),
                "coronal_hole_pct": round(float(np.mean(mask == 1)) * 100, 2),
                "active_region_pct": round(float(np.mean(mask == 2)) * 100, 2),
            })
            log.info("frame %s (CH %.1f%% AR %.1f%%)", ts,
                     frames[-1]["coronal_hole_pct"],
                     frames[-1]["active_region_pct"])

    (OUT_DIR / "index.json").write_text(json.dumps({
        "title": "AR 13664 — the Gannon storm region",
        "subtitle": "2024-05-04 to 2024-05-14 · U-Net segmentation per frame",
        "frames": frames,
    }, indent=2))
    log.info("%d frames -> %s", len(frames), OUT_DIR)


if __name__ == "__main__":
    main()
