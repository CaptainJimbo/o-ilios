# Evaluation — Solar Feature Segmenter (v1)

The signature layer of this project: numbers you can trust, failure modes stated
plainly. Protocol code: `model/metrics.py`, `model/baseline.py`, `model/evaluate.py`.

## Protocol

- **Metric:** micro-averaged per-class IoU (sum of intersections / sum of unions
  across the split), computed **inside the solar disk only** — off-disk pixels are
  trivially background and would inflate every score.
- **Split: temporal holdout, never random.** Adjacent solar frames are
  near-duplicates; a random split leaks and inflates IoU. Train = 2023–2024
  (192 samples), val = Jan–Mar 2025 (23), test = Aug–Oct 2025 (23), with a
  months-long gap between val and test.
- **Tuning discipline:** baseline thresholds and U-Net checkpoint selection both
  use val only; test is touched once, by the final models.
- **Labels:** SPoCA detections from HEK. These are algorithmic, not human,
  ground truth — see label-noise notes below.

## Results (test split, Aug–Oct 2025)

| Model | Coronal hole IoU | Active region IoU |
|---|---|---|
| 193 Å threshold baseline (tuned on val) | 0.531 | 0.238 |
| U-Net (ResNet18 encoder, 171+193+304 Å) | **0.603** | **0.413** |

Zone breakdown for the U-Net (disk split at 0.7 R_sun):

| Zone | Coronal hole IoU | Active region IoU |
|---|---|---|
| Disk center (r < 0.7 R) | 0.606 | 0.312 |
| Near limb (0.7–1.0 R) | 0.597 | 0.481 |

## Honest failure notes

1. **Thin filamentary coronal holes are under-segmented.** The model finds
   compact CH cores reliably but clips the thin, sprawling extensions
   (see `data/eval/gallery/worst_*`). Plausible fixes for v2: higher input
   resolution, boundary-aware loss.
2. **The ceiling is partly label noise, not model error.** SPoCA leaves
   limb-hugging CH arcs (limb-brightening artifacts) in the "ground truth";
   the model sensibly refuses to predict them and pays IoU for it. Cleaning
   or down-weighting limb labels is likely worth more than architecture work.
3. **AR center-vs-limb result is counterintuitive** (better near limb, 0.481
   vs 0.312) — the opposite of the usual projection-degradation story. Not yet
   diagnosed; candidate explanations: few AR pixels at center in this test
   window (small-denominator effect), or SPoCA's AR boundaries being tighter
   at center where the model predicts generously. Flagged for investigation,
   not explained away.
4. **Absolute numbers are modest by design honesty:** ~0.5–0.6 IoU against an
   algorithmic labeler is agreement-with-SPoCA, not agreement-with-truth. The
   right reading is "U-Net reproduces the operational detector at 512 px and
   beats a physics-motivated threshold by +0.07 CH / +0.18 AR IoU."
