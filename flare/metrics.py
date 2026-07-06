"""Forecast-verification metrics. TSS is the headline (base-rate independent);
HSS and FAR expose the false-alarm cost TSS ignores; Brier/BSS + reliability
assess the probabilities themselves. Accuracy is deliberately absent — at
~1-2% base rate it is meaningless.
"""

from __future__ import annotations

import numpy as np


def confusion(y: np.ndarray, yhat: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((yhat == 1) & (y == 1)))
    fp = int(np.sum((yhat == 1) & (y == 0)))
    fn = int(np.sum((yhat == 0) & (y == 1)))
    tn = int(np.sum((yhat == 0) & (y == 0)))
    return tp, fp, fn, tn


def tss(y: np.ndarray, yhat: np.ndarray) -> float:
    tp, fp, fn, tn = confusion(y, yhat)
    pod = tp / (tp + fn) if tp + fn else 0.0   # probability of detection
    pofd = fp / (fp + tn) if fp + tn else 0.0  # prob. of false detection
    return pod - pofd


def hss(y: np.ndarray, yhat: np.ndarray) -> float:
    tp, fp, fn, tn = confusion(y, yhat)
    num = 2 * (tp * tn - fp * fn)
    den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return num / den if den else 0.0


def far(y: np.ndarray, yhat: np.ndarray) -> float:
    """False-alarm ratio: fraction of positive forecasts that were wrong."""
    tp, fp, _, _ = confusion(y, yhat)
    return fp / (tp + fp) if tp + fp else 0.0


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Probability threshold maximizing TSS. Choose on TRAIN data only and
    report it — an unstated sweep on test is silent cherry-picking."""
    candidates = np.unique(np.quantile(p, np.linspace(0.01, 0.999, 200)))
    scores = [tss(y, (p >= t).astype(int)) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def bss(y: np.ndarray, p: np.ndarray, climatology: float) -> float:
    """Brier Skill Score vs a constant-climatology forecast (>0 = skill)."""
    ref = brier(y, np.full_like(p, climatology, dtype=float))
    return 1.0 - brier(y, p) / ref if ref else 0.0


def classification_report(y: np.ndarray, p: np.ndarray,
                          threshold: float, climatology: float) -> dict:
    yhat = (p >= threshold).astype(int)
    return {
        "tss": round(tss(y, yhat), 4),
        "hss": round(hss(y, yhat), 4),
        "far": round(far(y, yhat), 4),
        "brier": round(brier(y, p), 6),
        "bss_vs_climatology": round(bss(y, p, climatology), 4),
        "threshold": round(threshold, 5),
        "positives": int(y.sum()),
        "n": int(len(y)),
    }
