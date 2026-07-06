"""SWAN-SF -> compact tabular features.

Each SWAN-SF instance is a 60-step x 55-column MVTS (12 h observation window);
the label is the strongest GOES flare in the following 24 h, encoded in the
file path (FL/ vs NF/ + class prefix). We reduce each instance to last-value
features — the LPVV variant, TSS ~0.64 with plain models in the literature —
restricted to the keywords that also exist in JSOC's NRT SHARP series, so the
trained model is deployable live without feature drift.

GWILL (in the NRT series) is absent from SWAN-SF, so the shared set is 17.

Usage:
    python -m flare.data          # process all partition tars found
"""

from __future__ import annotations

import io
import json
import logging
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SWANSF_DIR = Path("data/swansf")
OUT_DIR = Path("data/swansf/tabular")

# SHARP keywords present BOTH in SWAN-SF and in hmi.sharp_cea_720s_nrt.
FEATURES = [
    "TOTUSJH", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP", "USFLUX",
    "MEANPOT", "R_VALUE", "MEANSHR", "SHRGT45", "MEANGAM", "MEANGBT",
    "MEANGBZ", "MEANGBH", "MEANJZH", "MEANJZD", "MEANALP",
]

CLASS_ORDER = {"FQ": 0, "B": 1, "C": 2, "M": 3, "X": 4}
_NAME_RE = re.compile(r"([A-Z]+)[\d.]*(?:@\d+)?[:_]")


def parse_label(member_name: str) -> tuple[str, int]:
    """(class letter, binary XM label) from an instance file name."""
    base = member_name.rsplit("/", 1)[-1]
    m = _NAME_RE.match(base)
    cls = m.group(1) if m else "FQ"
    if cls not in CLASS_ORDER:
        cls = "FQ"
    return cls, int(cls in ("M", "X"))


def extract_partition(tar_path: Path) -> dict[str, np.ndarray]:
    """Stream a partition tarball into last-value feature arrays."""
    rows, labels, classes, ars, starts = [], [], [], [], []
    with tarfile.open(tar_path) as tf:
        for member in tf:
            if not member.name.endswith(".csv"):
                continue
            handle = tf.extractfile(member)
            if handle is None:
                continue
            # NB: usecols selects in FILE order — reindex to FEATURES order
            # explicitly or downstream indices silently misalign.
            df = pd.read_csv(io.BytesIO(handle.read()), sep="\t",
                             usecols=FEATURES)[FEATURES]
            # Last valid observation per keyword within the window (the
            # window's final row can carry nulls; ~8% of timestamps do).
            last = df.ffill().iloc[-1].to_numpy(dtype=np.float32)
            cls, y = parse_label(member.name)
            ar = re.search(r"_ar(\d+)_", member.name)
            t0 = re.search(r"_s([\dT:-]+)_e", member.name)
            rows.append(last)
            labels.append(y)
            classes.append(cls)
            ars.append(int(ar.group(1)) if ar else -1)
            starts.append(t0.group(1) if t0 else "")
    order = np.argsort(np.array(starts))
    return {
        "X": np.stack(rows)[order],
        "y": np.array(labels, dtype=np.uint8)[order],
        "cls": np.array(classes)[order],
        "ar": np.array(ars, dtype=np.int32)[order],
        "start": np.array(starts)[order],
    }


def load_partition(number: int) -> dict[str, np.ndarray]:
    with np.load(OUT_DIR / f"partition{number}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for tar_path in sorted(SWANSF_DIR.glob("partition*.tar.gz")):
        number = int(tar_path.stem.split(".")[0].removeprefix("partition"))
        out = OUT_DIR / f"partition{number}.npz"
        if out.exists():
            log.info("cached: %s", out.name)
            data = load_partition(number)
        else:
            log.info("extracting %s ...", tar_path.name)
            data = extract_partition(tar_path)
            np.savez_compressed(out, **data)
        counts = {c: int(np.sum(data["cls"] == c)) for c in CLASS_ORDER}
        counts["nan_rows"] = int(np.isnan(data["X"]).any(axis=1).sum())
        summary[f"P{number}"] = counts
        log.info("P%d: %s", number, counts)
    (OUT_DIR / "summary.json").write_text(json.dumps(
        {"features": FEATURES, "counts": summary}, indent=2))


if __name__ == "__main__":
    main()
