"""S3: LightGBM forecaster with honest probabilities.

Split discipline inside the training partitions:
    fit      P1 + P2   (model, with early stopping on P3 AUC)
    calibrate P3       (isotonic on raw scores; threshold = max TSS there)
    test     P4, P5    (frozen model + calibrator + threshold, one shot)

The S2 lesson drives the design: the LR baseline hit TSS 0.87 by crying wolf
(FAR 0.83, BSS -3). Isotonic calibration is monotone, so it cannot change the
ranking (TSS at the optimal threshold is preserved) — but it turns scores
into probabilities you can put on a public dashboard next to NOAA's.

Feature attributions use LightGBM's native pred_contrib (TreeSHAP).

Usage:
    python -m flare.train
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression

from flare.baselines import impute_median, load_xy
from flare.data import FEATURES
from flare.metrics import best_threshold, classification_report

log = logging.getLogger(__name__)

EVAL_DIR = Path("data/v2_eval")
MODEL_DIR = Path("data/v2_model")

# Bobra & Couvidat 2015's top predictors, restricted to our 17 NRT-safe set.
BOBRA_CANON = {"TOTUSJH", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP",
               "USFLUX", "MEANPOT", "R_VALUE"}

# Shallow + heavily regularized: selected on P3 AUC (0.9586) against the
# LR-under-identical-protocol (0.9558) and two other configs — deeper trees
# overfit P1+P2 and lost to plain logistic regression on ranking.
PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 1000,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "scale_pos_weight": 1.0,  # set from data below
    "metric": "auc",
    "num_threads": 4,
    "verbosity": -1,
    "seed": 7,
}


def fit_model(x_fit, y_fit, x_cal, y_cal) -> lgb.Booster:
    params = dict(PARAMS)
    params["scale_pos_weight"] = float((y_fit == 0).sum() / y_fit.sum())
    model = lgb.train(
        params,
        lgb.Dataset(x_fit, y_fit, feature_name=FEATURES),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(x_cal, y_cal)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    log.info("best iteration: %d", model.best_iteration)
    return model


def reliability_plot(y, p, path: Path, title: str) -> None:
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(p, bins) - 1
    observed, predicted, counts = [], [], []
    for b in range(10):
        sel = idx == b
        if sel.sum() < 20:
            continue
        observed.append(y[sel].mean())
        predicted.append(p[sel].mean())
        counts.append(int(sel.sum()))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="#9a938e", label="perfect")
    ax.plot(predicted, observed, "o-", color="#ffb84d", label="model")
    for x_, y_, n in zip(predicted, observed, counts):
        ax.annotate(f"{n:,}", (x_, y_), fontsize=7,
                    textcoords="offset points", xytext=(6, -4))
    ax.set_xlabel("forecast probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def shap_importance(model: lgb.Booster, x: np.ndarray) -> dict[str, float]:
    sample = x[np.random.default_rng(7).choice(len(x), min(20000, len(x)),
                                               replace=False)]
    contrib = model.predict(sample, pred_contrib=True)[:, :-1]  # drop bias
    imp = np.abs(contrib).mean(axis=0)
    order = np.argsort(imp)[::-1]
    return {FEATURES[i]: float(imp[i]) for i in order}


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    x_fit, y_fit = load_xy((1, 2))
    x_cal, y_cal = load_xy((3,))
    x_p4, y_p4 = load_xy((4,))
    x_p5, y_p5 = load_xy((5,))
    x_fit, x_cal, x_p4, x_p5 = impute_median(x_fit, x_cal, x_p4, x_p5)
    base_rate = float(np.concatenate([y_fit, y_cal]).mean())

    model = fit_model(x_fit, y_fit, x_cal, y_cal)

    raw_cal = model.predict(x_cal)
    calibrator = IsotonicRegression(out_of_bounds="clip",
                                    y_min=0.0, y_max=1.0).fit(raw_cal, y_cal)
    p_cal = calibrator.predict(raw_cal)
    threshold = best_threshold(y_cal, p_cal)
    log.info("threshold (max TSS on calibrated P3): %.5f", threshold)

    report = {"protocol": {
        "fit": [1, 2], "calibrate_and_threshold": [3], "test": [4, 5],
        "train_base_rate": round(base_rate, 5),
        "best_iteration": model.best_iteration,
    }, "results": {}}
    for name, x_te, y_te in (("P4", x_p4, y_p4), ("P5", x_p5, y_p5)):
        p = calibrator.predict(model.predict(x_te))
        report["results"][name] = classification_report(
            y_te, p, threshold=threshold, climatology=base_rate)
        log.info("%s: %s", name, report["results"][name])

    y_all = np.concatenate([y_p4, y_p5])
    p_all = calibrator.predict(model.predict(np.concatenate([x_p4, x_p5])))
    reliability_plot(y_all, p_all, EVAL_DIR / "reliability.png",
                     "LightGBM + isotonic — reliability on P4+P5")

    importances = shap_importance(model, x_cal)
    report["shap_importance"] = {k: round(v, 5) for k, v in importances.items()}
    top5 = list(importances)[:5]
    overlap = len(set(top5) & BOBRA_CANON)
    report["shap_top5"] = top5
    report["bobra_top5_overlap"] = overlap
    log.info("SHAP top5: %s (Bobra overlap %d/5)", top5, overlap)

    (EVAL_DIR / "lightgbm.json").write_text(json.dumps(report, indent=2))
    model.save_model(MODEL_DIR / "lgbm_s3.txt")

    ok = (all(report["results"][p]["bss_vs_climatology"] > 0 for p in ("P4", "P5"))
          and overlap >= 3)
    log.info("S3 acceptance (BSS>0 both, SHAP overlap>=3): %s", ok)


if __name__ == "__main__":
    main()
