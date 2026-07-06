"""Step 2 baseline: classical intensity thresholding.

Physics: coronal holes are dark and active regions bright in 193 A. So the
baseline predicts, inside the solar disk, on the stored (asinh-stretched)
193 A channel:

    CH: intensity < ch_frac * disk_median
    AR: intensity > ar_frac * disk_median

followed by morphological cleanup. The two fractions are tuned by grid
search on the val split; results are reported on test. The U-Net's job is
to beat these numbers honestly.

Usage:
    python -m model.baseline            # tune on val, evaluate on test
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path

import numpy as np
from skimage.morphology import closing, disk, opening, remove_small_objects

from model.data import CHANNELS, Sample, load_split
from model.metrics import iou_per_class

log = logging.getLogger(__name__)

CH_GRID = np.arange(0.30, 0.95, 0.05)
AR_GRID = np.arange(1.10, 2.60, 0.10)
MIN_BLOB_PX = 64  # ~0.02% of the disk; kills salt noise, keeps small ARs


def predict(sample: Sample, ch_frac: float, ar_frac: float) -> np.ndarray:
    i193 = sample.image[..., CHANNELS[193]].astype(np.float32)
    median = np.median(i193[sample.disk])
    pred = np.zeros_like(sample.mask)
    for class_id, region in (
        (1, i193 < ch_frac * median),
        (2, i193 > ar_frac * median),
    ):
        region &= sample.disk
        region = opening(region, disk(2))
        region = closing(region, disk(4))
        region = remove_small_objects(region, max_size=MIN_BLOB_PX)
        pred[region] = class_id
    return pred


def evaluate(samples: list[Sample], ch_frac: float, ar_frac: float) -> dict:
    preds = [predict(s, ch_frac, ar_frac) for s in samples]
    return iou_per_class(preds, [s.mask for s in samples],
                         [s.disk for s in samples])


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    val = load_split("val")
    log.info("tuning on %d val samples", len(val))
    # The two thresholds are independent (disjoint intensity ranges), so
    # tune each class separately instead of a full 2-D grid.
    best = {}
    for name, grid, param in (
        ("coronal_hole", CH_GRID, "ch_frac"),
        ("active_region", AR_GRID, "ar_frac"),
    ):
        scores = {}
        for value in grid:
            kwargs = {"ch_frac": 0.0, "ar_frac": np.inf} | {param: value}
            scores[round(float(value), 2)] = evaluate(val, **kwargs)[name]
        best[param] = max(scores, key=scores.get)
        log.info("%s: best %s=%.2f (val IoU %.3f) | grid %s",
                 name, param, best[param], scores[best[param]],
                 {k: round(v, 3) for k, v in scores.items()})

    test = load_split("test")
    result = evaluate(test, **best)
    log.info("TEST (%d samples): %s", len(test),
             {k: round(v, 3) for k, v in result.items()})

    out = Path("data/eval")
    out.mkdir(parents=True, exist_ok=True)
    report = {"params": best, "test_iou": result,
              "n_val": len(val), "n_test": len(test)}
    (out / "baseline.json").write_text(json.dumps(report, indent=2))
    log.info("report -> %s", out / "baseline.json")


if __name__ == "__main__":
    main()
