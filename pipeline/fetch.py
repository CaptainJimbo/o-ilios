"""Fetch AIA imagery from the JSOC synoptic archive.

The synoptic archive serves 1024x1024 FITS at 2-minute cadence over plain
HTTP with no auth — much lighter than full 4k JSOC/VSO exports and plenty
for 512px training masks.
URL pattern:
  https://jsoc.stanford.edu/data/aia/synoptic/YYYY/MM/DD/H%H00/AIA%Y%m%d_%H%M_WWWW.fits
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import requests

log = logging.getLogger(__name__)

# jsoc.stanford.edu redirects here anyway, and its own HTTPS endpoint has a
# legacy TLS config that modern OpenSSL refuses — go straight to the mirror.
SYNOPTIC_BASE = "https://jsoc1.stanford.edu/data/aia/synoptic"
WAVELENGTHS = (171, 193, 304)


def synoptic_url(t: datetime, wavelength: int) -> str:
    return (
        f"{SYNOPTIC_BASE}/{t:%Y/%m/%d}/H{t:%H}00/"
        f"AIA{t:%Y%m%d}_{t:%H%M}_{wavelength:04d}.fits"
    )


def fetch_frame(t: datetime, wavelength: int, out_dir: Path) -> Path | None:
    """Download one synoptic FITS; returns local path or None if unavailable."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"aia_{t:%Y%m%dT%H%M}_{wavelength:04d}.fits"
    if dest.exists():
        log.info("cached: %s", dest.name)
        return dest
    url = synoptic_url(t, wavelength)
    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        log.warning("missing (%s): %s", resp.status_code, url)
        return None
    dest.write_bytes(resp.content)
    log.info("fetched: %s (%.0f kB)", dest.name, len(resp.content) / 1024)
    return dest


def fetch_day(
    date: datetime,
    out_dir: Path,
    every_hours: int = 4,
    wavelengths: tuple[int, ...] = WAVELENGTHS,
) -> dict[datetime, dict[int, Path]]:
    """Fetch a day of AIA frames at the given cadence.

    Returns {timestamp: {wavelength: path}} with only complete
    (all-wavelengths-present) timestamps kept.
    """
    frames: dict[datetime, dict[int, Path]] = {}
    t = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = t + timedelta(days=1)
    while t < end:
        got = {}
        for wl in wavelengths:
            path = fetch_frame(t, wl, out_dir)
            if path is not None:
                got[wl] = path
        if len(got) == len(wavelengths):
            frames[t] = got
        else:
            log.warning("skipping %s — incomplete wavelength set", t.isoformat())
        t += timedelta(hours=every_hours)
    return frames
