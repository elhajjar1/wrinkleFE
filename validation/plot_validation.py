#!/usr/bin/env python3
"""Plot WrinkleFE predictions against the experimental dataset.

Consumes the per-case records produced by :mod:`validate_dataset` and
renders two figures:

- ``validation/fig_validation_scatter.png`` -- predicted vs experimental
  knockdown for every single-wrinkle case, coloured by dataset, with the
  +/-20 % pass corridor shaded around the 1:1 line.
- ``validation/fig_validation_kd_vs_dt.png`` -- knockdown vs severity
  ratio ``D/T`` is not reconstructed here (D/T is not carried on the
  records); instead the second panel shows the per-dataset MAE bars so
  the scorecard is legible at a glance.

Run::

    python validation/plot_validation.py
"""

from __future__ import annotations

import os
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from validate_dataset import TOL, run  # noqa: E402

# One colour/marker per dataset family.
_STYLE = OrderedDict([
    ("A Elhajjar comp", ("#1f77b4", "o")),
    ("B Elhajjar tens", ("#17becf", "o")),
    ("C Mukhop comp", ("#2ca02c", "s")),
    ("C Mukhop tens-ult", ("#98df8a", "s")),
    ("C Mukhop tens-onset", ("#d62728", "s")),
    ("D Wang convex", ("#ff7f0e", "^")),
    ("D Wang concave", ("#8c564b", "v")),
    ("E Li2024 comp", ("#9467bd", "D")),
    ("F Li2025 comp", ("#e377c2", "*")),
])


def _scatter(records, ax):
    ax.plot([0, 1.1], [0, 1.1], "k-", lw=1, label="1:1")
    ax.fill_between([0, 1.1], [0, 1.1 * (1 - TOL)], [0, 1.1 * (1 + TOL)],
                    color="grey", alpha=0.15, label=f"+/-{int(TOL*100)} %")
    for ds, (c, m) in _STYLE.items():
        rs = [r for r in records if r.dataset == ds]
        if not rs:
            continue
        ax.scatter([r.kd_exp for r in rs], [r.kd_pred for r in rs],
                   c=c, marker=m, s=55, edgecolor="k", linewidth=0.4,
                   label=ds, zorder=3)
    ax.set_xlim(0.2, 1.1)
    ax.set_ylim(0.2, 1.1)
    ax.set_xlabel("Experimental KD")
    ax.set_ylabel("Predicted KD (analytical)")
    ax.set_title("WrinkleFE vs experiment -- single-wrinkle cases")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")


def _mae_bars(records, ax):
    groups = OrderedDict()
    for r in records:
        groups.setdefault(r.dataset, []).append(r)
    labels = list(groups)
    maes = [sum(r.err for r in rs) / len(rs) * 100 for rs in groups.values()]
    colors = [_STYLE.get(d, ("#777", "o"))[0] for d in labels]
    ax.barh(range(len(labels)), maes, color=colors, edgecolor="k")
    ax.axvline(TOL * 100, color="r", ls="--", lw=1, label=f"{int(TOL*100)} % tol")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Per-dataset MAE (%)")
    ax.set_title("Mean absolute error by dataset")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="x")


def main() -> None:
    records = run()
    here = os.path.dirname(os.path.abspath(__file__))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    _scatter(records, ax1)
    _mae_bars(records, ax2)
    fig.tight_layout()
    out = os.path.join(here, "fig_validation_scatter.png")
    fig.savefig(out, dpi=200)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
