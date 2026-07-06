# Ο Ήλιος (o-ilios) — Project Spec

**The Living Sun** — AI segmentation of solar features on live SDO imagery + a
real-time space-weather strip. Private repo while under construction.

## Why this project exists

Portfolio flagship for Dimitris's pivot toward science/mission-driven ML work
(see `polish-my-profile` repo: `CAREER_DIRECTION.md`). Chosen over the wildfire
Burned-Area Mapper because: astrophysics pull + space weather is *funded &
operational* heliophysics (grids, aviation, satellites, GPS) + highest visual
ceiling of all candidate projects.

**Identity it sells:** physicist + MSc-level computer vision, ships AND rigorously
evaluates models, and can build a stunning front-end around them.

## What we're building (decided)

Project "B + a slice of C" from the solar options. Architecture decisions locked
6/7/2026 — see `ARCHITECTURE.md` (React+WebGL frontend, local MPS training,
scheduled-inference-to-static-artifacts serving, local → GH Pages → AWS hosting).

1. **Solar Feature Segmenter (core ML)** — full-disk SDO/AIA EUV images →
   pixel-precise masks for **coronal holes, active regions** (sunspots deferred
   to v2 — they need HMI continuum, a second data source + catalog).
   U-Net family, PyTorch. This is the thesis-skillset showcase (segmentation).
2. **The Living Sun (front-end, the stunner)** —
   - Full-screen glowing SDO sun as hero, updated from NASA's feed.
   - **Wavelength slider** morphing 171Å → 193Å → 304Å.
   - Segmentation masks as **luminous overlays** (coronal holes cyan, active
     regions amber), toggleable, with a raw ↔ annotated **wipe slider**.
   - **Time-lapse scrubber**: active region tracked across ~10 days of rotation.
3. **"Conditions now" strip (slice of project C)** — live solar wind
   (NOAA SWPC / DSCOVR), current **Kp index**, "aurora / storm chance" indicator.
4. **Evaluation layer (the signature)** — per-class IoU, honest failure modes,
   documented methodology. This is the differentiator vs. notebook-only work.

**Growth path (later, same dashboard):** flare-probability panel (project A —
SHARP magnetograms + SWAN-SF), CME detection (project D).

## Data sources (all free)

- **SDO/AIA imagery:** SDOML (curated ML-ready archive), SunPy + JSOC/VSO for
  fresh imagery; NASA SDO near-real-time browse images for the live view.
- **Labels for segmentation:** SPoCA (coronal holes / active regions, HEK),
  sunspot catalogs (Debrecen / NOAA SRS). Verify label quality early — this is
  the main project risk.
- **Space weather now:** NOAA SWPC JSON feeds (solar wind from DSCOVR, planetary
  Kp) — public, no auth.

## Build spine (ship-not-sprawl, each step a working artifact)

1. **Data pipeline first:** fetch one day of AIA imagery (2–3 wavelengths) +
   overlay SPoCA/HEK labels → sanity-check visually. Proves data + labels.
2. **Baseline segmentation:** classical thresholding baseline (coronal holes are
   dark in 193Å) → then U-Net beats it. Mirrors dNBR→U-Net logic from the EO plan.
3. **Evaluation layer:** per-class IoU vs. held-out labels, spatial CV, failure notes.
4. **Front-end + live feed:** The Living Sun page + conditions strip.
5. **Deploy on AWS** (+ scheduled refresh of imagery/conditions).

## Plugins to set up in this repo (decided 4/7/2026, not yet installed)

```
claude plugin enable frontend-design            # already on machine; THE plugin for The Living Sun UI
claude plugin install fiftyone@claude-plugins-official           # CV dataset viz — de-risks label quality (step 1)
claude plugin install huggingface-skills@claude-plugins-official # SDOML & solar datasets/backbones live on HF
claude plugin install pyright-lsp@claude-plugins-official        # Python type checking QoL
```

Later, at deploy (step 5): `claude plugin enable aws-serverless`.
Already enabled at user scope: playwright, chrome-devtools, context7, firecrawl.

## Working conventions (carried from other repos)

- **Ask before deploying** anything with a slow feedback loop.
- Disable, don't delete (`published: false` pattern where applicable).
- Batch edits, then one commit; don't panic on CDN/cache lag.
- New ideas → parking lot (`polish-my-profile/PORTFOLIO_PLAN.md`), not this repo's
  scope. Risk is breadth without shipping.

## Related repos / docs

- `polish-my-profile` — career docs (`CAREER_DIRECTION.md`, `EO_RESEARCH_FINDINGS.md`,
  `PORTFOLIO_PLAN.md` — the master project queue).
- `CaptainJimbo.github.io` — portfolio site; this project gets a card + eventually
  hosts/links The Living Sun demo.
