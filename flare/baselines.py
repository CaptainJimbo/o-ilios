"""S2: the baseline ladder every fancier model must beat, honestly.

  1. Climatology — constant base-rate probability (TSS 0 by construction;
     it is the BSS reference, not a TSS competitor).
  2. R_VALUE threshold — Schrijver 2007's single-feature forecaster.
  3. Logistic regression on all 17 keywords.

Protocol: train on partitions 1-3 (thresholds, scalers, imputation medians
all fit there), test on partitions 4 and 5 separately. No test data touches
any fitted quantity.

Usage:
    python -m flare.baselines
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from flare.data import FEATURES, load_partition
from flare.metrics import best_threshold, classification_report

log = logging.getLogger(__name__)

EVAL_DIR = Path("data/v2_eval")
TRAIN_PARTS = (1, 2, 3)
TEST_PARTS = (4, 5)
R_IDX = FEATURES.index("R_VALUE")


def load_xy(parts) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for p in parts:
        d = load_partition(p)
        xs.append(d["X"])
        ys.append(d["y"])
    return np.concatenate(xs), np.concatenate(ys)


def impute_median(train_x, *others):
    med = np.nanmedian(train_x, axis=0)
    out = []
    for x in (train_x, *others):
        x = x.copy()
        idx = np.where(np.isnan(x))
        x[idx] = np.take(med, idx[1])
        out.append(x)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    x_tr, y_tr = load_xy(TRAIN_PARTS)
    tests = {p: load_xy((p,)) for p in TEST_PARTS}
    x_tr, *x_tes = impute_median(x_tr, *(tests[p][0] for p in TEST_PARTS))
    y_tes = {p: tests[p][1] for p in TEST_PARTS}
    base_rate = float(y_tr.mean())
    log.info("train n=%d, base rate %.4f", len(y_tr), base_rate)

    report: dict = {"protocol": {
        "train_partitions": TRAIN_PARTS, "test_partitions": TEST_PARTS,
        "features": FEATURES, "train_base_rate": round(base_rate, 5),
        "note": "thresholds/scalers/medians fit on train partitions only",
    }, "models": {}}

    # 1. Climatology: constant probability. No threshold sweep is meaningful.
    report["models"]["climatology"] = {
        f"P{p}": classification_report(
            y_tes[p], np.full(len(y_tes[p]), base_rate),
            threshold=0.5, climatology=base_rate)
        for p in TEST_PARTS
    }

    # 2. Schrijver R_VALUE threshold (single feature, threshold from train).
    r_train = x_tr[:, R_IDX]
    thr_r = best_threshold(y_tr, r_train)
    log.info("R_VALUE threshold (train, max TSS): %.3f", thr_r)
    report["models"]["r_value_threshold"] = {
        f"P{p}": {
            k: v for k, v in classification_report(
                y_tes[p], x_te[:, R_IDX], threshold=thr_r,
                climatology=base_rate).items()
            # A raw threshold model emits no probabilities — Brier/BSS on the
            # feature value itself would be meaningless.
            if k not in ("brier", "bss_vs_climatology")
        }
        for p, x_te in zip(TEST_PARTS, x_tes)
    }
    report["models"]["r_value_threshold"]["train_threshold"] = round(thr_r, 3)

    # 3. Logistic regression, all features, class-balanced.
    scaler = StandardScaler().fit(x_tr)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=4)
    lr.fit(scaler.transform(x_tr), y_tr)
    p_tr = lr.predict_proba(scaler.transform(x_tr))[:, 1]
    thr_lr = best_threshold(y_tr, p_tr)
    log.info("LR threshold (train, max TSS): %.4f", thr_lr)
    report["models"]["logistic_regression"] = {
        f"P{p}": classification_report(
            y_tes[p], lr.predict_proba(scaler.transform(x_te))[:, 1],
            threshold=thr_lr, climatology=base_rate)
        for p, x_te in zip(TEST_PARTS, x_tes)
    }

    (EVAL_DIR / "baselines.json").write_text(json.dumps(report, indent=2))
    for name, res in report["models"].items():
        scores = {k: v["tss"] for k, v in res.items() if k.startswith("P")}
        log.info("%s: TSS %s", name, scores)
    log.info("-> %s", EVAL_DIR / "baselines.json")


if __name__ == "__main__":
    main()
