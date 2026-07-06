# Evaluation — Flare Forecaster (v2)

Per-active-region probability of an M-class-or-stronger GOES flare within
24 h, from SHARP magnetic-complexity keywords. Protocol code: `flare/`.
Companion to [EVALUATION.md](EVALUATION.md) (v1 segmentation).

## Task & data

- **Unit of prediction:** one active region at one moment → P(≥M within 24 h).
- **Training data:** [SWAN-SF](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EBCFKM)
  (Angryk et al. 2020): ~330k sliced instances over solar cycle 24, five
  temporally disjoint partitions engineered for comparable X/M counts.
  Labels are GOES flare classes; base rate for M+ is ~2% of instances.
- **Features:** last-value vectors of the **17 SHARP keywords available in
  both SWAN-SF and JSOC's near-real-time series** — the live model must never
  see a feature distribution it wasn't trained on. (This excludes the
  Lorentz-force keywords used in some papers: they don't exist in NRT.)

## Protocol — leakage is the whole game here

Random splits on SWAN-SF reach TSS ≈ 0.9 by memorization (adjacent sliced
instances share 92% of their data — Ahmadzadeh et al. 2021). Everything
below is therefore **cross-partition**: no partition contributes to both
sides of any fit/evaluate boundary, scalers and imputation medians are fit
on training partitions only, and probability thresholds are chosen on a
partition the model never trained on.

- Analysis protocol: fit P1+P2 → calibrate + threshold on P3 → test P4, P5.
- Deployed model: fit P1–P3 → calibrate + threshold on P4 → **P5 touched
  exactly once** for the final reported numbers.
- Sanity gate before any modeling: P(M+|24 h) vs R_VALUE reproduces
  Schrijver's monotonic relationship (0 → 40% across bins, n = 331,185).
- We additionally verified **zero active-region overlap** between partitions.

## Results

Baseline ladder (fit P1–P3, test P4/P5; threshold chosen on train):

| Model | TSS (P4 / P5) | HSS | FAR | BSS vs climatology |
|---|---|---|---|---|
| Climatology (constant 2%) | 0.00 / 0.00 | 0.00 | — | 0 (reference) |
| R_VALUE threshold (Schrijver 2007) | 0.79 / 0.86 | 0.21 / 0.18 | 0.86 / 0.89 | n/a (no probabilities) |
| Logistic regression (balanced) | 0.87 / 0.88 | 0.29 / 0.27 | 0.81 / 0.83 | **−2.4 / −3.0** |
| **LightGBM + isotonic** (fit P1+P2, cal P3) | 0.78 / 0.86 | 0.27 / 0.29 | 0.82 / 0.81 | **+0.34 / +0.29** |

**Deployed model** (fit P1–P3, calibrated P4), final one-shot on P5:
**TSS 0.861 · HSS 0.195 · FAR 0.879 · BSS +0.267**, threshold 0.029.
Reliability diagrams: `data/v2_eval/` (regenerable from the pipeline).

## How to read these numbers honestly

1. **TSS ~0.85 is not the achievement it appears to be.** The
   logistic-regression row proves it: TSS 0.88 alongside a false-alarm ratio
   of 0.83 and *negative* Brier skill. TSS is blind to false-alarm cost, and
   a max-TSS threshold at a 2% base rate happily fires constantly. The
   number that separates a forecaster from a classifier is **BSS: our
   deployed model's probabilities carry +0.27 skill over climatology**,
   where the wolf-crier scores −3.
2. **These are instance-level scores against SWAN-SF's slicing**, the
   community benchmark convention — not operational event-level skill. The
   published caution stands: offline TSS ≈ 0.8 systems have collapsed to
   ~0.24 in live operations (DeepFlareNet). Our own live record accrues in
   the public [forecast ledger](https://captainjimbo.github.io/o-ilios/live/ledger.json).
3. **The ceiling is real physics, not engineering debt.** FLARECAST's
   209-predictor study concluded flare onset is intrinsically stochastic;
   the All-Clear workshops found no method clearly beating climatology.
   Claims far above this band on SWAN-SF usually indicate leakage.
4. **Model selection was boring on purpose** — and evaluated cleanly:
   default-depth LightGBM *lost* to logistic regression on ranking
   (AUC 0.961 vs 0.977); a shallow regularized configuration won on the
   selection partition (P3 AUC 0.9586 vs 0.9558) and only then was tested.
   SHAP attributions of the final model rank MEANPOT, TOTUSJH, R_VALUE,
   SHRGT45, TOTUSJZ on top — 4/5 overlap with Bobra & Couvidat's canonical
   predictor ranking, as physics would demand.

## Live-deployment deviations (documented, deliberate)

- **Limb gate |lon| > 68°:** NRT keywords blow up with projection near the
  limb. Validated on day one: a limb region reported R_VALUE 5.4 (implying
  ~40% flare probability) that the gate correctly excluded.
- **QUALITY reported, not gated:** live NRT records routinely carry
  informational quality bits (0x00011c00); a strict ==0 gate would blank
  every forecast.
- **Probability floor 0.1%:** isotonic calibration maps empty low bins to
  exactly 0, and "0.000% chance of a flare" claims more than the data can.
- **No X-class head:** at ~800:1 imbalance a dedicated X model was judged
  untrainable from this data; we report P(≥M) only rather than fabricate.

## The ledger

Every UTC day, the worker records the model's full-disk P(≥M) and NOAA
SWPC's own forecaster probability *before the outcome is known*, then fills
in what the sun actually did from the official GOES event list. It is an
append-only, public, self-updating verification record — the part of this
project we could not have faked even if we wanted to.

## Sources

Angryk et al. 2020 (SWAN-SF); Ahmadzadeh et al. 2021 (leakage protocol);
Bobra & Couvidat 2015 (SHARP features); Georgoulis et al. 2021 (FLARECAST);
Barnes et al. 2016 / Leka et al. 2019 (All-Clear); Camporeale & Berger 2025
(SWPC verification); Schrijver 2007 (R_VALUE). Full URLs in the research
notes and code comments.
