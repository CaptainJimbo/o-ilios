# Ο Ήλιος (o-ilios) — Project Spec

**The Living Sun** — AI segmentation of solar features on live SDO imagery + a
real-time space-weather strip. **Live** at captainjimbo.github.io/o-ilios/
(GH Actions cron refreshes every 20 min).

## What this is

1. **Solar Feature Segmenter (core ML, v1)** — full-disk SDO/AIA EUV images →
   pixel-precise masks for **coronal holes, active regions** (sunspots deferred —
   they need HMI continuum, a second data source + catalog). U-Net, PyTorch.
2. **The Living Sun (front-end)** — full-screen live sun, wavelength morph
   slider (171Å→193Å→304Å), luminous mask overlays with raw↔annotated wipe,
   time-lapse of AR 13664 (May 2024 Gannon storm) crossing the disk.
3. **"Conditions now" strip** — live solar wind, Kp index, storm indicator
   (NOAA SWPC real-time feeds, fetched client-side).
4. **Flare Watch (v2)** — per-active-region P(M+ flare in 24 h) from NRT SHARP
   keywords (LightGBM + isotonic calibration, SWAN-SF cross-partition
   protocol), badges on the disk, NOAA comparison, public forecast ledger.
5. **Evaluation layers (the signature)** — EVALUATION.md (per-class IoU,
   temporal holdout, failure modes) and EVALUATION-V2.md (TSS/HSS/BSS,
   calibration, honest-ceiling framing).

Architecture decisions and rationale: `ARCHITECTURE.md`.
Growth path (later, same dashboard): CME detection; sunspots via HMI.

## Data sources (all free)

- **SDO/AIA imagery:** JSOC synoptic archive (1024px FITS) for training;
  `mostrecent/` for the live view. SunPy + HEK for SPoCA labels.
- **Flares:** SWAN-SF benchmark (Harvard Dataverse) for training; JSOC
  `hmi.sharp_cea_720s_nrt` keywords live; GOES/SWPC event lists for outcomes.
- **Space weather now:** NOAA SWPC JSON feeds — public, no auth, CORS-open.

## Working conventions

- **Ask before deploying** anything with a slow feedback loop.
- Disable, don't delete (`published: false` pattern where applicable).
- Batch edits, then one commit; don't panic on CDN/cache lag.
- Ship-not-sprawl: each build step must end in a working artifact.
- v1 must never break: v2+ workers are additive and soft-fail.

## Plugins used in this repo

frontend-design, fiftyone, huggingface-skills, pyright-lsp (project scope);
playwright, chrome-devtools, context7, firecrawl (user scope).
Later, at AWS migration: aws-serverless.
