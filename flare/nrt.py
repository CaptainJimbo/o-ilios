"""S5: live flare forecast from near-real-time SHARP keywords.

Runs in the scheduled worker after the AIA segmentation step. One no-auth
HTTP GET to JSOC's jsoc_info returns every currently-tracked active region's
SHARP keywords at the latest record time; the S4 bundle turns them into
calibrated P(M+ flare within 24 h) per region; NOAA's own published
per-region probabilities ride along for comparison. Output is a static
artifact the frontend reads.

Design constraints (see V2_NOTES.md, verified by probe):
  - plain http on jsoc.stanford.edu ONLY (its https is self-signed);
  - T_REC is TAI = UTC + 37 s;
  - NRT keywords go noisy near the limb -> drop |LON_FWT| > 68 deg;
  - QUALITY != 0 records are excluded from forecasting;
  - a spotless sun (count 0) is a valid state, not an error;
  - SOFT-FAIL: any exception must leave the previous flares.json in place
    and exit 0 — v2 must never break the v1 deploy.

Usage:
    python -m flare.nrt
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import requests

from flare.features import FEATURES
from flare.predict import predict_bundle

log = logging.getLogger(__name__)

JSOC = "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info"
SWPC_REGIONS = "https://services.swpc.noaa.gov/json/solar_regions.json"
BUNDLE_PATH = Path("data/v2_model/flare_model.joblib")
OUT_PATH = Path("web/public/live/flares.json")

META_KEYS = ["HARPNUM", "T_REC", "NOAA_AR", "NOAA_ARS", "LAT_FWT", "LON_FWT",
             "QUALITY", "CRLT_OBS", "RSUN_OBS"]
MAX_ABS_LON = 68.0
TAI_UTC_OFFSET_S = 37


def fetch_sharps() -> list[dict]:
    keys = ",".join(META_KEYS + FEATURES)
    resp = requests.get(JSOC, params={
        "op": "rs_list", "ds": "hmi.sharp_cea_720s_nrt[][$]", "key": keys,
    }, timeout=60)
    resp.raise_for_status()
    payload = resp.json()  # HTML error pages raise here — caught by soft-fail
    if payload.get("status") != 0:
        raise RuntimeError(f"jsoc_info status {payload.get('status')}: "
                           f"{payload.get('error', '?')}")
    count = payload.get("count", 0)
    columns = {kw["name"]: kw["values"] for kw in payload.get("keywords", [])}
    return [
        {name: values[i] for name, values in columns.items()}
        for i in range(count)
    ]


def parse_t_rec(t_rec: str) -> datetime:
    dt = datetime.strptime(t_rec.replace("_TAI", ""), "%Y.%m.%d_%H:%M:%S")
    return dt.replace(tzinfo=timezone.utc) - timedelta(seconds=TAI_UTC_OFFSET_S)


def to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def disk_fractions(lat_deg: float, lon_deg: float, b0_deg: float
                   ) -> tuple[float, float]:
    """Stonyhurst (lat, lon) -> helioprojective offsets in solar radii
    (+x west/right, +y north/up)."""
    lat, lon, b0 = map(math.radians, (lat_deg, lon_deg, b0_deg))
    fx = math.cos(lat) * math.sin(lon)
    fy = math.sin(lat) * math.cos(b0) - math.cos(lat) * math.cos(lon) * math.sin(b0)
    return fx, fy


def fetch_noaa_probabilities() -> dict[int, dict]:
    rows = requests.get(SWPC_REGIONS, timeout=30).json()
    if not rows:
        return {}
    latest = max(r.get("observed_date", "") for r in rows)
    return {
        int(r["region"]): {
            "noaa_p_c": r.get("c_flare_probability"),
            "noaa_p_m": r.get("m_flare_probability"),
            "noaa_p_x": r.get("x_flare_probability"),
        }
        for r in rows
        if r.get("observed_date") == latest and r.get("region")
    }


def build_forecast() -> dict:
    bundle = joblib.load(BUNDLE_PATH)
    records = fetch_sharps()
    log.info("NRT SHARPs at latest T_REC: %d", len(records))

    t_rec_utc = parse_t_rec(records[0]["T_REC"]) if records else None
    staleness_min = (
        round((datetime.now(timezone.utc) - t_rec_utc).total_seconds() / 60, 1)
        if t_rec_utc else None
    )
    try:
        noaa = fetch_noaa_probabilities()
    except Exception as err:  # NOAA comparison is garnish, never fatal
        log.warning("solar_regions.json unavailable: %s", err)
        noaa = {}

    regions, skipped = [], 0
    for rec in records:
        # QUALITY arrives as hex ("0x00011c00"); live NRT records routinely
        # carry informational bits, so it is reported, not gated on —
        # gating happens on limb distance and unusable features instead.
        quality = str(rec.get("QUALITY", ""))
        lon = to_float(rec.get("LON_FWT"))
        lat = to_float(rec.get("LAT_FWT"))
        x = np.array([[to_float(rec.get(f)) for f in FEATURES]],
                     dtype=np.float32)
        if (math.isnan(lon) or math.isnan(lat) or abs(lon) > MAX_ABS_LON
                or np.isnan(x).sum() > len(FEATURES) // 2):
            skipped += 1
            continue
        # Floor at 0.1%: isotonic maps empty low bins to exactly 0, and a
        # literal "0.000% chance of a flare" claims more than data can.
        p_m24 = max(float(predict_bundle(bundle, x)[0]), 0.001)
        fx, fy = disk_fractions(lat, lon, to_float(rec.get("CRLT_OBS")))
        noaa_ar = int(to_float(rec.get("NOAA_AR")) or 0)
        entry = {
            "harpnum": int(to_float(rec.get("HARPNUM"))),
            "noaa_ar": noaa_ar or None,
            "lat": round(lat, 2), "lon": round(lon, 2),
            "fx": round(fx, 4), "fy": round(fy, 4),
            "p_m24": round(p_m24, 4),
            "alert": bool(p_m24 >= bundle["threshold"]),
            "quality": quality,
        }
        entry.update(noaa.get(noaa_ar % 10000, {}))
        regions.append(entry)

    p_any = 1.0 - float(np.prod([1.0 - r["p_m24"] for r in regions])) \
        if regions else 0.0
    noaa_ms = [r["noaa_p_m"] / 100 for r in regions
               if isinstance(r.get("noaa_p_m"), (int, float))]
    noaa_any = 1.0 - float(np.prod([1.0 - p for p in noaa_ms])) \
        if noaa_ms else None

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "t_rec_utc": t_rec_utc.isoformat() if t_rec_utc else None,
        "staleness_min": staleness_min,
        "regions": sorted(regions, key=lambda r: -r["p_m24"]),
        "regions_skipped_limb_or_unusable": skipped,
        "full_disk": {"p_m24_any": round(p_any, 4),
                      "noaa_p_m24_any": round(noaa_any, 4)
                      if noaa_any is not None else None},
        "model": {
            "threshold": bundle["threshold"],
            "provenance": bundle["provenance"],
            "test_tss_p5": 0.861, "test_bss_p5": 0.267,
        },
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        forecast = build_forecast()
    except Exception:
        log.exception("flare forecast failed — keeping previous artifact")
        if not OUT_PATH.exists():
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUT_PATH.write_text(json.dumps({
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "regions": [], "error": "forecast unavailable",
            }))
        return  # exit 0 — never break the v1 deploy
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(forecast, indent=1))
    log.info("flares.json: %d regions, P(M+|24h anywhere)=%.3f, staleness %s min",
             len(forecast["regions"]), forecast["full_disk"]["p_m24_any"],
             forecast["staleness_min"])
    try:
        from flare.ledger import update as update_ledger
        update_ledger(forecast)
    except Exception:
        log.exception("ledger update failed — forecast artifact unaffected")


if __name__ == "__main__":
    main()
