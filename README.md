# Ο Ήλιος — The Living Sun ☀️

AI segmentation of solar features on live NASA SDO imagery, wrapped in a real-time
space-weather dashboard.

> Point it at the Sun — right now, today — and it draws what it sees: coronal holes
> outlined in cyan, active regions pulsing amber, sunspots pinned to the disk. Plus a
> live "conditions now" strip: solar wind, Kp index, aurora chance tonight.

## What it does

1. **Pulls the latest solar imagery** from NASA's Solar Dynamics Observatory (SDO) —
   full-disk EUV images in multiple wavelengths (171Å, 193Å, 304Å, ...).
2. **Segments solar features** with a deep-learning model (U-Net family):
   coronal holes, active regions, sunspots — pixel-precise masks.
3. **Renders "The Living Sun"** — an interactive front-end: wavelength morph slider,
   raw ↔ AI-annotated wipe, time-lapse scrubber across the solar rotation.
4. **Shows space weather now** — live solar-wind data (NOAA SWPC / DSCOVR), current
   Kp index, and a simple aurora/storm indicator.
5. **Reports honest accuracy** — evaluation against curated annotations, with the
   same rigor applied to any production model (per-class IoU, confusion behavior,
   failure modes documented).

## Status

🚧 Early scaffold — private while under construction.

## Stack (planned)

- **ML:** Python, PyTorch, segmentation (U-Net / DeepLab family)
- **Solar data:** SDO/AIA & HMI via SDOML / SunPy / JSOC; NOAA SWPC real-time feeds
- **Front-end:** modern web (visual-first — the Sun is the hero)
- **Deploy:** AWS

---

*Built by [Dimitris Kogias](https://captainjimbo.github.io) — physicist & AI/ML systems engineer.*
