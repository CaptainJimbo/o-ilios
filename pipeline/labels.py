"""SPoCA labels from HEK: query, polygon parsing, rasterization to masks.

SPoCA reports coronal-hole (CH) and active-region (AR) detections to HEK with
boundary polygons in helioprojective (arcsec) coordinates. We rasterize those
onto each AIA frame's pixel grid via its WCS.

Class ids: 0 = background/quiet sun, 1 = coronal hole, 2 = active region.

Known simplification (fine for the sanity check, revisit for training):
events are matched to frames by nearest detection time within a tolerance,
without compensating solar rotation (~0.55 deg/hr — a few pixels at 1024px
for tolerances of a couple of hours).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from skimage.draw import polygon as draw_polygon
from sunpy.net import Fido
from sunpy.net import attrs as a

log = logging.getLogger(__name__)

CLASS_IDS = {"CH": 1, "AR": 2}


@dataclass
class HekEvent:
    event_type: str  # "CH" | "AR"
    time: datetime
    boundary_arcsec: np.ndarray  # (N, 2) Tx, Ty in arcsec


def query_spoca_events(start: datetime, end: datetime) -> list[HekEvent]:
    """All SPoCA CH + AR detections with usable boundaries in [start, end].

    sunpy's HEK client parses hpc_boundcc into a SkyCoord polygon already;
    we keep the raw (Tx, Ty) arcsec vertices.
    """
    events: list[HekEvent] = []
    for ev_type, hek_attr in (("CH", a.hek.CH), ("AR", a.hek.AR)):
        result = Fido.search(
            a.Time(start, end), hek_attr, a.hek.FRM.Name == "SPoCA"
        )["hek"]
        log.info("HEK returned %d %s rows", len(result), ev_type)
        for row in result:
            coord = row["hpc_boundcc"]
            if coord is None or coord.size < 3:
                continue
            events.append(
                HekEvent(
                    event_type=ev_type,
                    time=row["event_starttime"].datetime,
                    boundary_arcsec=np.column_stack(
                        [coord.Tx.to_value(u.arcsec), coord.Ty.to_value(u.arcsec)]
                    ),
                )
            )
    log.info("parsed %d events with boundaries", len(events))
    return events


def events_for_frame(
    events: list[HekEvent],
    frame_time: datetime,
    tolerance: timedelta = timedelta(hours=3),
) -> list[HekEvent]:
    """Events from the detection run nearest to frame_time, per event type.

    SPoCA runs periodically and emits all regions it sees at each run; picking
    the nearest run (not every event in the window) avoids stacking duplicate
    boundaries of the same region from consecutive runs.
    """
    picked: list[HekEvent] = []
    for ev_type in CLASS_IDS:
        typed = [e for e in events if e.event_type == ev_type]
        if not typed:
            continue
        nearest_run = min(typed, key=lambda e: abs(e.time - frame_time)).time
        if abs(nearest_run - frame_time) > tolerance:
            log.warning(
                "no %s run within %s of %s", ev_type, tolerance, frame_time
            )
            continue
        picked.extend(e for e in typed if e.time == nearest_run)
    return picked


def rasterize(events: list[HekEvent], smap) -> np.ndarray:
    """Burn event polygons into a class-id mask on smap's pixel grid.

    ARs are burned after CHs so they win overlapping pixels (rare, but ARs
    are the rarer, brighter class — don't let a sprawling CH polygon eat them).
    """
    mask = np.zeros(smap.data.shape, dtype=np.uint8)
    for ev_type in ("CH", "AR"):
        for ev in (e for e in events if e.event_type == ev_type):
            coord = SkyCoord(
                ev.boundary_arcsec[:, 0] * u.arcsec,
                ev.boundary_arcsec[:, 1] * u.arcsec,
                frame=smap.coordinate_frame,
            )
            x, y = smap.world_to_pixel(coord)
            rr, cc = draw_polygon(
                np.clip(y.value, 0, mask.shape[0] - 1),
                np.clip(x.value, 0, mask.shape[1] - 1),
                shape=mask.shape,
            )
            mask[rr, cc] = CLASS_IDS[ev_type]
    return mask
