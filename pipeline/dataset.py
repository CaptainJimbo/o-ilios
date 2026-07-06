"""Assemble the segmentation dataset: multi-day AIA frames + SPoCA masks,
downsampled to 512x512, written as inspectable PNG pairs.

Layout (data/ is gitignored — everything here re-fetches from public archives):
    data/dataset/<split>/images/<ts>.png   RGB = AIA 171 / 193 / 304 (asinh-stretched)
    data/dataset/<split>/masks/<ts>.png    class ids: 0 bg, 1 CH, 2 AR
    data/dataset/<split>/meta.jsonl        per-sample disk geometry (cx, cy, r at 512px)

Split design — temporal holdout, never random: adjacent frames are
near-duplicates, so random splits leak. Train on 2023–2024, validate on
early 2025, test on late 2025 (with a gap so val doesn't kiss test).

Usage:
    python -m pipeline.dataset --split all
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import astropy.units as u
import numpy as np
import sunpy.map
from astropy.coordinates import SkyCoord
from PIL import Image

from pipeline.fetch import WAVELENGTHS, fetch_day
from pipeline.labels import events_for_frame, query_spoca_events, rasterize
from pipeline.preprocess import model_input

log = logging.getLogger(__name__)

DATA_ROOT = Path("data")

# Two days per month keeps solar-rotation diversity (a 27-day period means
# 1st + 15th sample opposite hemispheres of the rotation).
def _days(year_months: list[tuple[int, int]]) -> list[date]:
    return [date(y, m, d) for y, m in year_months for d in (1, 15)]


SPLITS: dict[str, list[date]] = {
    "train": _days([(2023, m) for m in range(1, 13)]
                   + [(2024, m) for m in range(1, 13)]),   # 48 days
    "val": _days([(2025, m) for m in (1, 2, 3)]),          # 6 days
    "test": _days([(2025, m) for m in (8, 9, 10)]),        # 6 days, gap after val
}

EVERY_HOURS = 6  # 4 frames/day; tighter cadence just adds near-duplicates


def assemble_split(split: str) -> None:
    out = DATA_ROOT / "dataset" / split
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.jsonl"
    done_ids = set()
    if meta_path.exists():
        done_ids = {json.loads(line)["id"] for line in meta_path.open()}

    events_cache: dict[date, list] = {}
    with meta_path.open("a") as meta_file:
        for day in SPLITS[split]:
            day_dt = datetime(day.year, day.month, day.day)
            frames = fetch_day(day_dt, DATA_ROOT / "raw" / day.isoformat(),
                               every_hours=EVERY_HOURS)
            if not frames:
                log.warning("no frames for %s", day)
                continue

            # Per-day HEK queries: a whole-month query takes ~6 min against
            # HEK's paginated API; a padded day is ~30 s and covers every
            # frame we actually use.
            if day not in events_cache:
                events_cache[day] = query_spoca_events(
                    day_dt - timedelta(hours=6), day_dt + timedelta(hours=30))
            events = events_cache[day]

            for t, paths in sorted(frames.items()):
                sample_id = f"{t:%Y%m%dT%H%M}"
                if sample_id in done_ids:
                    continue
                maps = {wl: sunpy.map.Map(paths[wl]) for wl in WAVELENGTHS}
                ref = maps[193]
                if ref.data.shape != (1024, 1024):
                    log.warning("%s: unexpected shape %s — skipped",
                                sample_id, ref.data.shape)
                    continue

                frame_events = events_for_frame(events, t)
                if not frame_events:
                    log.warning("%s: no SPoCA run nearby — skipped", sample_id)
                    continue
                mask = rasterize(frame_events, ref)[::2, ::2]
                rgb = model_input({wl: maps[wl].data for wl in WAVELENGTHS})

                Image.fromarray(rgb).save(out / "images" / f"{sample_id}.png")
                Image.fromarray(mask).save(out / "masks" / f"{sample_id}.png")

                disk_center = SkyCoord(0 * u.arcsec, 0 * u.arcsec,
                                       frame=ref.coordinate_frame)
                cx, cy = ref.world_to_pixel(disk_center)
                r_pix = (ref.rsun_obs / ref.scale[0]).to_value()
                meta_file.write(json.dumps({
                    "id": sample_id,
                    "time": t.isoformat(),
                    "disk_cx": float(cx.value) / 2,
                    "disk_cy": float(cy.value) / 2,
                    "disk_r": r_pix / 2,
                }) + "\n")
                meta_file.flush()
                log.info("%s/%s: CH %.1f%% AR %.1f%%", split, sample_id,
                         100 * np.mean(mask == 1), 100 * np.mean(mask == 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="all",
                        choices=[*SPLITS, "all"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for split in SPLITS if args.split == "all" else [args.split]:
        log.info("=== assembling %s (%d days) ===", split, len(SPLITS[split]))
        assemble_split(split)


if __name__ == "__main__":
    main()
