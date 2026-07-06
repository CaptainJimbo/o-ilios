"""Build-spine step 1: fetch a day of AIA imagery, overlay SPoCA labels,
save side-by-side sanity PNGs for visual audit.

Usage:
    python -m pipeline.step1_sanity --date 2024-05-10 --every 4
Outputs land in data/sanity/<date>/.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
from astropy.visualization import AsinhStretch, ImageNormalize

from pipeline.fetch import fetch_day
from pipeline.labels import events_for_frame, query_spoca_events, rasterize

log = logging.getLogger(__name__)

# Overlay palette (mask class id -> RGBA), matching the Living Sun spec:
# coronal holes cyan, active regions amber.
OVERLAY = {1: (0.0, 0.9, 1.0, 0.35), 2: (1.0, 0.75, 0.0, 0.35)}
EDGE = {1: (0.0, 0.9, 1.0, 1.0), 2: (1.0, 0.75, 0.0, 1.0)}


def overlay_rgba(mask: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*mask.shape, 4))
    for class_id, color in OVERLAY.items():
        rgba[mask == class_id] = color
    return rgba


def render(smap, mask: np.ndarray, out_path: Path, title: str) -> None:
    norm = ImageNormalize(
        smap.data, vmin=0, vmax=np.percentile(smap.data, 99.9),
        stretch=AsinhStretch(0.01),
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor="black")
    for ax in axes:
        ax.imshow(smap.data, cmap="sdoaia193", norm=norm, origin="lower")
        ax.set_axis_off()
    axes[1].imshow(overlay_rgba(mask), origin="lower")
    for class_id, color in EDGE.items():
        axes[1].contour(mask == class_id, levels=[0.5], colors=[color],
                        linewidths=0.8)
    axes[0].set_title(f"{title} — raw", color="white")
    axes[1].set_title(f"{title} — SPoCA overlay (CH cyan / AR amber)",
                      color="white")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True,
                        help="UTC day to audit, YYYY-MM-DD")
    parser.add_argument("--every", type=int, default=4,
                        help="hours between frames (default 4)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    day = datetime.fromisoformat(args.date)
    raw_dir = Path("data/raw") / args.date
    out_dir = Path("data/sanity") / args.date
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = fetch_day(day, raw_dir, every_hours=args.every)
    log.info("complete frames: %d", len(frames))

    # One HEK query for the whole day (pad so early/late frames find a run).
    events = query_spoca_events(day - timedelta(hours=6),
                                day + timedelta(hours=30))

    for t, paths in sorted(frames.items()):
        # Labels are rasterized on the 193 A frame's WCS; all synoptic
        # wavelengths share the grid, so one mask serves the stack.
        smap = sunpy.map.Map(paths[193])
        frame_events = events_for_frame(events, t)
        mask = rasterize(frame_events, smap)
        ch_pct = 100 * np.mean(mask == 1)
        ar_pct = 100 * np.mean(mask == 2)
        log.info("%s: %d events, CH %.1f%% of pixels, AR %.1f%%",
                 t.isoformat(), len(frame_events), ch_pct, ar_pct)
        render(smap, mask, out_dir / f"sanity_{t:%Y%m%dT%H%M}.png",
               f"AIA 193 A {t:%Y-%m-%d %H:%M} UT")

    log.info("done — inspect %s/", out_dir)


if __name__ == "__main__":
    main()
