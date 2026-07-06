"""S4: build the deployable model bundle.

Uses more of the data than S3's analysis model while preserving one honest,
untouched test partition:
    fit        P1 + P2 + P3   (early stopping on P4 AUC)
    calibrate  P4             (isotonic + max-TSS threshold)
    test       P5             (one shot — this is THE number the dashboard
                               and EVALUATION-V2.md report for the live model)

Bundle = LightGBM booster + isotonic calibrator + imputation medians +
feature list + threshold + provenance, saved with joblib, plus a frozen
sample batch for a load-and-predict round-trip test.

Usage:
    python -m flare.bundle          # build + round-trip test
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression

from flare.baselines import load_xy
from flare.data import FEATURES
from flare.metrics import best_threshold, classification_report
from flare.predict import predict_bundle
from flare.train import PARAMS, reliability_plot

log = logging.getLogger(__name__)

MODEL_DIR = Path("data/v2_model")
EVAL_DIR = Path("data/v2_eval")
BUNDLE = MODEL_DIR / "flare_model.joblib"


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    x_fit, y_fit = load_xy((1, 2, 3))
    x_cal, y_cal = load_xy((4,))
    x_test, y_test = load_xy((5,))
    medians = np.nanmedian(x_fit, axis=0)
    for x in (x_fit, x_cal, x_test):
        idx = np.where(np.isnan(x))
        x[idx] = np.take(medians, idx[1])
    base_rate = float(np.concatenate([y_fit, y_cal]).mean())

    params = dict(PARAMS)
    params["scale_pos_weight"] = float((y_fit == 0).sum() / y_fit.sum())
    booster = lgb.train(
        params, lgb.Dataset(x_fit, y_fit, feature_name=FEATURES),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(x_cal, y_cal)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    raw_cal = booster.predict(x_cal)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0,
                                    y_max=1.0).fit(raw_cal, y_cal)
    threshold = best_threshold(y_cal, calibrator.predict(raw_cal))

    bundle = {
        "booster": booster,
        "calibrator": calibrator,
        "impute_medians": medians,
        "features": FEATURES,
        "threshold": float(threshold),
        "train_base_rate": base_rate,
        "provenance": "SWAN-SF fit P1-P3, calibrated P4, tested P5; "
                      "17 NRT-available SHARP keywords, last-value features",
    }

    p_test = predict_bundle(bundle, x_test)
    result = classification_report(y_test, p_test, threshold=threshold,
                                   climatology=base_rate)
    log.info("P5 (final, one-shot): %s", result)
    reliability_plot(y_test, p_test, EVAL_DIR / "reliability_deploy.png",
                     "Deployable model — reliability on untouched P5")

    joblib.dump(bundle, BUNDLE, compress=3)
    sample = {"x": x_test[:256], "p": p_test[:256]}
    joblib.dump(sample, MODEL_DIR / "roundtrip_sample.joblib")

    # Round-trip: load fresh and reproduce predictions bit-for-bit.
    loaded = joblib.load(BUNDLE)
    p_again = predict_bundle(loaded, sample["x"])
    ok = bool(np.allclose(p_again, sample["p"], atol=1e-9))
    log.info("bundle %.1f MB · round-trip identical: %s",
             BUNDLE.stat().st_size / 1e6, ok)
    (MODEL_DIR / "deploy_report.json").write_text(json.dumps(
        {"p5_final": result, "threshold": float(threshold),
         "best_iteration": booster.best_iteration,
         "roundtrip_ok": ok}, indent=2))
    if not ok:
        raise SystemExit("round-trip FAILED")


if __name__ == "__main__":
    main()
