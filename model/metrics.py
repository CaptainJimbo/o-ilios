"""Segmentation metrics. All computed inside the solar disk only — off-disk
pixels are trivially background and would inflate every score."""

from __future__ import annotations

import numpy as np

CLASS_NAMES = {1: "coronal_hole", 2: "active_region"}


def iou_per_class(
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    disks: list[np.ndarray],
) -> dict[str, float]:
    """Micro-averaged IoU per class (sum of intersections / sum of unions
    across the whole split — robust to samples where a class is absent)."""
    inter = dict.fromkeys(CLASS_NAMES, 0)
    union = dict.fromkeys(CLASS_NAMES, 0)
    for pred, target, disk in zip(predictions, targets, disks):
        for class_id in CLASS_NAMES:
            p = (pred == class_id) & disk
            t = (target == class_id) & disk
            inter[class_id] += np.sum(p & t)
            union[class_id] += np.sum(p | t)
    return {
        CLASS_NAMES[c]: float(inter[c] / union[c]) if union[c] else float("nan")
        for c in CLASS_NAMES
    }
