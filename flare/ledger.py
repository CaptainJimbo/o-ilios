"""S7: the forecast ledger — a rolling public record of what the model
predicted and what the sun actually did.

Persistence trick: Pages deploys are ephemeral, so each worker run fetches
the currently-deployed ledger from the live site, appends/updates, and
writes it back into the artifact set. One entry per UTC day:

    {date, p_any, noaa_p_any,            <- recorded on the day's first run
     outcome_m_plus, strongest, flares}  <- filled in on later days from
                                            SWPC's edited events (XRA rows)

Verification data: https://services.swpc.noaa.gov/json/edited_events.json —
official, AR-attributed, covers the recent week; M/X rows dated by begin
time. Duplicate re-issues deduped on (begin, region).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

log = logging.getLogger(__name__)

LIVE_LEDGER = "https://captainjimbo.github.io/o-ilios/live/ledger.json"
EDITED_EVENTS = "https://services.swpc.noaa.gov/json/edited_events.json"
OUT_PATH = Path("web/public/live/ledger.json")
MAX_DAYS = 400


def fetch_existing() -> list[dict]:
    try:
        rows = requests.get(LIVE_LEDGER, timeout=30).json()
        if isinstance(rows, list) and rows:
            return rows
    except Exception:
        pass
    # Live copy missing (first run, or a failed deploy dropped it) — fall
    # back to the checkout's seed so history survives interruptions.
    try:
        rows = json.loads(OUT_PATH.read_text())
        log.info("using local ledger seed (%d rows)", len(rows))
        return rows if isinstance(rows, list) else []
    except Exception:
        log.info("no existing ledger (first run?)")
        return []


def m_plus_flares_by_day() -> dict[str, list[str]]:
    """UTC day -> list of M/X GOES classes that began that day."""
    rows = requests.get(EDITED_EVENTS, timeout=30).json()
    seen, by_day = set(), {}
    for r in rows:
        if r.get("type") != "XRA":
            continue
        cls = str(r.get("particulars1", ""))
        if not cls or cls[0] not in "MX":
            continue
        begin = str(r.get("begin_datetime", ""))[:10]
        key = (begin, str(r.get("begin_datetime")), r.get("region"))
        if not begin or key in seen:
            continue
        seen.add(key)
        by_day.setdefault(begin, []).append(cls)
    return by_day


def update(forecast: dict) -> None:
    today = datetime.now(timezone.utc).date().isoformat()
    ledger = fetch_existing()
    by_date = {row["date"]: row for row in ledger}

    # Record today's forecast once (first worker run of the UTC day wins,
    # so the forecast is a true ~24 h-ahead statement).
    if today not in by_date:
        by_date[today] = {
            "date": today,
            "p_any": forecast["full_disk"]["p_m24_any"],
            "noaa_p_any": forecast["full_disk"]["noaa_p_m24_any"],
            "n_regions": len(forecast["regions"]),
        }

    # Fill outcomes for recent days from the official event list.
    try:
        flares = m_plus_flares_by_day()
    except Exception as err:
        log.warning("edited_events unavailable: %s", err)
        flares = {}
    # Ledger day D forecasts the window [D 00:00, D+1 00:00) UTC; a flare
    # counts if its begin time falls on D. Outcomes are only written once
    # the window has closed AND the event feed actually covers D (it spans
    # ~a week) — otherwise the day stays open rather than defaulting to
    # "no flare", which would silently bias the record quiet.
    feed_covers_from = min(flares) if flares else None
    for date_str, row in by_date.items():
        if "outcome_m_plus" in row or date_str >= today:
            continue
        if feed_covers_from is None or date_str <= feed_covers_from:
            continue  # strictly after the feed's earliest day (it may be cut)
        day_flares = flares.get(date_str, [])
        row["outcome_m_plus"] = bool(day_flares)
        row["strongest"] = max(day_flares, default=None,
                               key=lambda c: ("MX".index(c[0]), float(c[1:] or 0)))
        row["flares"] = sorted(day_flares)

    rows = sorted(by_date.values(), key=lambda r: r["date"])[-MAX_DAYS:]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(rows, indent=1))
    log.info("ledger: %d days, latest %s", len(rows), rows[-1])
