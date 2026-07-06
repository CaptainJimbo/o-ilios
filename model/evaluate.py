"""Step 3: the evaluation layer. Scores the best U-Net checkpoint on the
held-out test split with the same inside-disk protocol as the baseline,
adds a disk-center vs near-limb breakdown (projection effects make the limb
the honest failure zone), and renders a gallery of best/worst predictions.

Outputs:
    data/eval/unet.json      test IoU + limb breakdown + baseline comparison
    data/eval/gallery/*.png  image | ground truth | prediction triptychs

Usage:
    python -m model.evaluate
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model.data import Sample, load_split
from model.metrics import CLASS_NAMES, iou_per_class
from model.train import CHECKPOINT_DIR, build_model

log = logging.getLogger(__name__)

EVAL_DIR = Path("data/eval")
INNER_FRAC = 0.7  # disk-center zone: r < 0.7 R_sun; limb zone: 0.7 R..1.0 R
COLORS = {1: np.array([0, 230, 255]), 2: np.array([255, 190, 0])}


def predict_split(samples: list[Sample], device) -> list[np.ndarray]:
    checkpoint = torch.load(CHECKPOINT_DIR / "unet_best.pt",
                            map_location=device, weights_only=True)
    model = build_model(checkpoint["encoder"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    log.info("checkpoint from epoch %d (val IoU %s)",
             checkpoint["epoch"], checkpoint["val_iou"])
    preds = []
    with torch.no_grad():
        for s in samples:
            x = torch.from_numpy(s.image.astype(np.float32) / 255.0)
            x = x.permute(2, 0, 1)[None].to(device)
            preds.append(model(x).argmax(1)[0].cpu().numpy().astype(np.uint8))
    return preds


def _zone_disks(sample: Sample) -> tuple[np.ndarray, np.ndarray]:
    """(inner, limb) boolean masks splitting the disk at INNER_FRAC."""
    # Erode the disk radius: pixels inside a shrunken disk are "inner".
    yy, xx = np.mgrid[: sample.disk.shape[0], : sample.disk.shape[1]]
    # Recover center/radius from the stored disk mask extents.
    ys, xs = np.where(sample.disk)
    cy, cx = ys.mean(), xs.mean()
    r = (xs.max() - xs.min() + 1) / 2
    inner = (xx - cx) ** 2 + (yy - cy) ** 2 <= (INNER_FRAC * r) ** 2
    return inner & sample.disk, sample.disk & ~inner


def _overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = image.copy()
    for class_id, color in COLORS.items():
        sel = mask == class_id
        out[sel] = (out[sel] * 0.45 + color * 0.55).astype(np.uint8)
    return out


def render_gallery(samples, preds, per_sample_iou, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    order = np.argsort(per_sample_iou)
    picks = {"worst": order[:2], "best": order[-2:]}
    for tag, indices in picks.items():
        for i in indices:
            s, p = samples[i], preds[i]
            trip = np.concatenate(
                [s.image, _overlay(s.image, s.mask), _overlay(s.image, p)],
                axis=1)
            name = f"{tag}_{s.id}_iou{per_sample_iou[i]:.2f}.png"
            Image.fromarray(trip).save(out_dir / name)
            log.info("gallery: %s (raw | ground truth | prediction)", name)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    samples = load_split("test")
    preds = predict_split(samples, device)
    targets = [s.mask for s in samples]

    report = {"test_iou": iou_per_class(preds, targets,
                                        [s.disk for s in samples])}
    zones = [_zone_disks(s) for s in samples]
    report["test_iou_disk_center"] = iou_per_class(
        preds, targets, [z[0] for z in zones])
    report["test_iou_near_limb"] = iou_per_class(
        preds, targets, [z[1] for z in zones])

    baseline_path = EVAL_DIR / "baseline.json"
    if baseline_path.exists():
        report["baseline_test_iou"] = json.loads(
            baseline_path.read_text())["test_iou"]

    # Per-sample mean IoU over present classes, for gallery pick.
    per_sample = []
    for p, s in zip(preds, samples):
        vals = [v for v in iou_per_class([p], [s.mask], [s.disk]).values()
                if not np.isnan(v)]
        per_sample.append(float(np.mean(vals)) if vals else 0.0)
    render_gallery(samples, preds, np.array(per_sample), EVAL_DIR / "gallery")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "unet.json").write_text(json.dumps(report, indent=2))
    log.info("report:\n%s", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
