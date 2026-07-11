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
    """Each currently-tracked HARP's newest record from the last 2 h.

    The obvious query `[][$]` (all HARPs at the single latest T_REC) is
    unstable: HARPs whose NRT processing lags a slot vanish from it, so the
    region set — and the full-disk probability — jittered run to run
    (observed 9 -> 2 regions in 20 minutes). A window + per-HARP newest
    is stable against processing lag."""
    now = datetime.now(timezone.utc) + timedelta(seconds=TAI_UTC_OFFSET_S)
    start = now - timedelta(hours=2)
    window = (f"{start:%Y.%m.%d_%H:%M:%S}_TAI-{now:%Y.%m.%d_%H:%M:%S}_TAI")
    keys = ",".join(META_KEYS + FEATURES)
    resp = requests.get(JSOC, params={
        "op": "rs_list", "ds": f"hmi.sharp_cea_720s_nrt[][{window}]",
        "key": keys,
    }, timeout=60)
    resp.raise_for_status()
    payload = resp.json()  # HTML error pages raise here — caught by soft-fail
    if payload.get("status") != 0:
        raise RuntimeError(f"jsoc_info status {payload.get('status')}: "
                           f"{payload.get('error', '?')}")
    count = payload.get("count", 0)
    columns = {kw["name"]: kw["values"] for kw in payload.get("keywords", [])}
    rows = [{name: values[i] for name, values in columns.items()}
            for i in range(count)]
    newest: dict[str, dict] = {}
    for row in rows:
        harp = row.get("HARPNUM")
        if harp is None:
            continue
        if harp not in newest or str(row.get("T_REC", "")) > \
                str(newest[harp].get("T_REC", "")):
            newest[harp] = row
    return list(newest.values())


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

    t_rec_utc = (max(parse_t_rec(r["T_REC"]) for r in records)
                 if records else None)
    staleness_min = (
        round((datetime.now(timezone.utc) - t_rec_utc).total_seconds() / 60, 1)
        if t_rec_utc else None
    )
    try:
        noaa = fetch_noaa_probabilities()
    except Exception as err:  # NOAA comparison is garnish, never fatal
        log.warning("solar_regions.json unavailable: %s", err)
        noaa = {}

    regions, raw_ps, skipped = [], [], 0
    for rec in records:
        # Per-row isolation: one malformed record must degrade ONE badge,
        # not abort the whole forecast into the preserve-stale path.
        try:
            # QUALITY arrives as hex ("0x00011c00"); live NRT records
            # routinely carry informational bits, so it is reported, not
            # gated on — gating is limb distance + unusable features.
            quality = str(rec.get("QUALITY", ""))
            lon = to_float(rec.get("LON_FWT"))
            lat = to_float(rec.get("LAT_FWT"))
            x = np.array([[to_float(rec.get(f)) for f in FEATURES]],
                         dtype=np.float32)
            if (math.isnan(lon) or math.isnan(lat) or abs(lon) > MAX_ABS_LON
                    or np.isnan(x).sum() > len(FEATURES) // 2):
                skipped += 1
                continue
            p_raw = float(predict_bundle(bundle, x)[0])
            # Floor at 0.1% for DISPLAY: isotonic maps empty low bins to
            # exactly 0, and "0.000% chance" claims more than data can.
            # The full-disk product uses the raw values — flooring every
            # term would build a ~1% quiet-day floor out of thin air.
            p_m24 = max(p_raw, 0.001)
            fx, fy = disk_fractions(lat, lon, to_float(rec.get("CRLT_OBS")))
            # NB: NaN is truthy — `to_float(...) or 0` stays NaN. Check it.
            noaa_ar_f = to_float(rec.get("NOAA_AR"))
            noaa_ar = 0 if math.isnan(noaa_ar_f) else int(noaa_ar_f)
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
        except Exception:
            log.exception("skipping malformed SHARP record: %s",
                          {k: rec.get(k) for k in ("HARPNUM", "T_REC")})
            skipped += 1
            continue
        regions.append(entry)
        raw_ps.append(p_raw)

    p_any = max(1.0 - float(np.prod([1.0 - p for p in raw_ps])), 0.001) \
        if raw_ps else 0.0
    # NOAA's number aggregates over EVERY region NOAA forecasts on disk —
    # including ones our limb filter drops — otherwise "NOAA says X%"
    # understates them on limb-active days.
    noaa_ms = [row["noaa_p_m"] / 100 for row in noaa.values()
               if isinstance(row.get("noaa_p_m"), (int, float))]
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


def _preserve_live_artifacts() -> None:
    """Failure path. CI checkouts are fresh, so 'keep the previous file'
    means fetching what the site currently serves and re-emitting it —
    otherwise a failed run REPLACES good artifacts with nothing (observed
    2026-07-07: a schema-incomplete stub crashed the frontend, and the
    ledger vanished from the deploy)."""
    live = "https://captainjimbo.github.io/o-ilios/live"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for name, path in (("flares.json", OUT_PATH),
                       ("ledger.json", OUT_PATH.parent / "ledger.json")):
        try:
            data = requests.get(f"{live}/{name}", timeout=30).json()
            if name == "flares.json" and "full_disk" not in data:
                raise ValueError("live copy is schema-incomplete")
            if name == "flares.json":
                data["stale"] = True
            path.write_text(json.dumps(data, indent=1))
            log.info("preserved live %s", name)
        except Exception:
            log.warning("could not preserve %s from live", name)
            if not path.exists() and name == "flares.json":
                # Last resort: schema-COMPLETE empty forecast.
                path.write_text(json.dumps({
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "t_rec_utc": None, "staleness_min": None,
                    "regions": [], "regions_skipped_limb_or_unusable": 0,
                    "full_disk": {"p_m24_any": 0.0, "noaa_p_m24_any": None},
                    "model": {"threshold": 0.0, "provenance": "unavailable",
                              "test_tss_p5": 0.861, "test_bss_p5": 0.267},
                    "error": "forecast unavailable",
                }))


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        forecast = build_forecast()
    except Exception:
        log.exception("flare forecast failed — preserving live artifacts")
        _preserve_live_artifacts()
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
