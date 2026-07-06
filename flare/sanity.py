"""S1 acceptance: reproduce the known monotonic relationship between
Schrijver's R_VALUE (log10 unsigned flux near polarity-inversion lines)
and 24 h M+ flare rate. If this plot isn't monotonic, the pipeline is
mislabeled or mis-parsed and nothing downstream can be trusted.

Usage:
    python -m flare.sanity
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flare.data import FEATURES, load_partition

log = logging.getLogger(__name__)

EVAL_DIR = Path("data/v2_eval")
R_IDX = FEATURES.index("R_VALUE")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    r_all, y_all = [], []
    for p in range(1, 6):
        d = load_partition(p)
        ok = ~np.isnan(d["X"][:, R_IDX])
        r_all.append(d["X"][ok, R_IDX])
        y_all.append(d["y"][ok])
    r = np.concatenate(r_all)
    y = np.concatenate(y_all)

    # R_VALUE = 0 means "no strong PIL detected" — its own regime; then bins.
    edges = [0.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.5]
    labels, rates, ns = [], [], []
    zero = r <= 0
    labels.append("0")
    rates.append(float(y[zero].mean()))
    ns.append(int(zero.sum()))
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (r > lo) & (r <= hi) & ~zero
        if sel.sum() < 50:
            continue
        labels.append(f"{lo}–{hi}")
        rates.append(float(y[sel].mean()))
        ns.append(int(sel.sum()))

    if len(rates) < 4 or sum(ns) < 0.9 * len(r):
        raise SystemExit(f"R_VALUE sanity VACUOUS — {len(rates)} bins cover "
                         f"{sum(ns):,}/{len(r):,} rows; distribution is not "
                         "what this test assumes. Investigate before trusting.")
    # Non-decreasing within 2 binomial standard errors: near-zero bins jitter
    # at the 1e-4 level, which is noise, not a physics violation.
    def se(p: float, n: int) -> float:
        return (max(p, 1e-6) * (1 - max(p, 1e-6)) / n) ** 0.5

    increasing = all(
        b >= a - 2 * (se(a, na) + se(b, nb))
        for (a, na), (b, nb) in zip(zip(rates[:-1], ns[:-1]),
                                    zip(rates[1:], ns[1:]))
    )
    log.info("bins: %s", list(zip(labels, [round(x, 4) for x in rates], ns)))
    log.info("monotonic non-decreasing: %s", increasing)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(rates)), rates, color="#ffb84d")
    ax.set_xticks(range(len(rates)), labels, rotation=45, ha="right")
    ax.set_xlabel("R_VALUE (log10 Mx near PIL)")
    ax.set_ylabel("P(M+ flare within 24 h)")
    ax.set_title(f"SWAN-SF sanity: flare rate vs R_VALUE "
                 f"(monotonic={increasing}, n={len(r):,})")
    for i, n in enumerate(ns):
        ax.annotate(f"n={n:,}", (i, rates[i]), ha="center", va="bottom",
                    fontsize=7)
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "sanity_rvalue.png", dpi=120)

    (EVAL_DIR / "sanity.json").write_text(json.dumps({
        "monotonic_nondecreasing": increasing,
        "bins": [{"bin": b, "rate": r_, "n": n}
                 for b, r_, n in zip(labels, rates, ns)],
    }, indent=2))
    if not increasing:
        raise SystemExit("R_VALUE sanity FAILED — do not proceed to baselines")
    log.info("S1 sanity PASSED -> %s", EVAL_DIR / "sanity_rvalue.png")


if __name__ == "__main__":
    main()
