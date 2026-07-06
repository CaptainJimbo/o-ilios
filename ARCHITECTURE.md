# Architecture — Ο Ήλιος

Decisions locked 2026-07-06 after brainstorm. This is the how; CLAUDE.md is the what/why.

## System shape

```
                    UNDER THE HOOD                          OVER THE HOOD
┌─────────────────────────────────────────┐   ┌──────────────────────────────────┐
│ pipeline/  historical AIA (SunPy/JSOC)  │   │ worker/  cron (GH Actions →      │
│            + SPoCA labels via HEK       │   │          Lambda at step 5):      │
│            → 512×512 images + masks     │   │   fetch latest SDO browse JPEGs  │
│                                         │   │   → ONNX U-Net on CPU (~1–2 s)   │
│ model/     threshold baseline →         │   │   → static artifacts:            │
│            U-Net (smp, pretrained enc.) │   │     imagery ×3λ, mask PNGs,      │
│            train local (MPS)            │   │     metadata.json                │
│            eval: per-class IoU,         │   └────────────────┬─────────────────┘
│            temporal holdout,            │                    │ static hosting
│            center-vs-limb breakdown     │                    ▼
│            → export ONNX                │   ┌──────────────────────────────────┐
└─────────────────────────────────────────┘   │ web/  static SPA (React + Vite)  │
                                              │   WebGL canvas: sun, λ-morph,    │
              NOAA SWPC JSON feeds ─────────► │   glow, mask overlays, wipe,     │
              (fetched client-side, live)     │   time-lapse scrubber            │
                                              │   + "conditions now" strip       │
                                              └──────────────────────────────────┘
```

Key insight: **no inference API.** The sun updates every ~15 min, so inference runs on a
schedule and publishes static artifacts. The frontend is a fully static site. Zero
servers, near-zero cost, nothing to fall over during a demo.

## Decisions & rationale

| Decision | Choice | Why |
|---|---|---|
| Frontend | React + Vite, WebGL canvas for the sun | Hiring signal + true shader-blended wavelength morph (not a JPEG crossfade) |
| Training compute | Local Mac (MPS) first | 512² U-Net on a few-thousand images is minutes/epoch; iteration cost is in data/labels, not FLOPs. Escalate to HF Jobs only if needed |
| Classes | Coronal holes + active regions only (v1) | Sunspots need HMI continuum — different instrument, different catalog, second pipeline. Deferred to v2 (ship-not-sprawl) |
| Model input | 3-channel stack: AIA 171+193+304 Å | Mirrors false-color composites; more signal than 193 alone |
| Architecture | U-Net, pretrained encoder (segmentation_models_pytorch) | Thesis-skillset showcase; proven for this exact task |
| Baseline | 193 Å intensity threshold inside disk + morphological cleanup | CHs are dark in 193 Å; U-Net must honestly beat this number |
| Eval split | **Temporal holdout, never random** | Adjacent solar frames are near-duplicates; random split leaks and inflates IoU |
| Eval extras | Per-class IoU + disk-center vs near-limb breakdown, failure gallery | Limb degradation is the honest failure mode to showcase |
| Serving | Scheduled job → static artifacts (no model server) | See key insight above. ONNX CPU inference, ~1–2 s per frame |
| Conditions strip | Client-side fetch of NOAA SWPC JSON (CORS-enabled) | Always live even if the ML worker hiccups; nothing to host |
| Hosting path | Local dev (repo private) → GitHub Pages at `captainjimbo.github.io/o-ilios/` (repo flipped public) → AWS S3/CloudFront + Lambda at step 5 | Pages on a private repo needs Pro; the demo must be public to be a portfolio piece anyway. `actions/deploy-pages` publishes artifacts without git-history churn |
| License | All rights reserved (explicit LICENSE added before flipping public) | Portfolio piece: viewable, not reusable |

## Data

- **Training imagery:** AIA 171/193/304 Å via SunPy (VSO/JSOC), downsampled to 512×512,
  ~1 h cadence sampled across different activity levels. SDOML as fallback source.
- **Labels:** SPoCA coronal-hole / active-region detections from HEK (boundary polygons
  via SunPy), rasterized onto the image grid. **Main project risk** — visually audit with
  FiftyOne before any training (build-spine step 1).
- **Live imagery (production):** NASA SDO near-real-time browse JPEGs
  (`sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_XXXX.jpg`), ~15 min cadence, no auth.
  Fully decoupled from the training path.
- **Space weather:** NOAA SWPC JSON (planetary Kp, DSCOVR solar wind), public, CORS.

## Repo layout

```
pipeline/   data fetch, HEK label build, dataset assembly, label QA
model/      baseline, U-Net training, evaluation, ONNX export
worker/     scheduled inference job (produces web-ready artifacts)
web/        The Living Sun frontend (React + Vite + WebGL)
```

## Deferred (growth path — parking lot, not v1)

Sunspots (HMI continuum + Debrecen/NOAA SRS), flare-probability panel (SHARP + SWAN-SF),
CME detection.
