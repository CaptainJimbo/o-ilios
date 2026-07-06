"""Bundle inference — the single entry point shared by evaluation and the
live worker. Deliberately light imports (numpy only at module level; the
bundle itself carries the lightgbm/sklearn objects via joblib)."""

from __future__ import annotations

import numpy as np


def predict_bundle(bundle: dict, x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32).copy()
    idx = np.where(np.isnan(x))
    x[idx] = np.take(np.asarray(bundle["impute_medians"]), idx[1])
    return bundle["calibrator"].predict(bundle["booster"].predict(x))
