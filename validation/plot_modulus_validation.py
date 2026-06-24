#!/usr/bin/env python3
"""Stiffness-only validation chart: predicted vs experimental modulus knockdown.

The visual companion to ``validate_modulus.py`` (and the stiffness counterpart
of ``plot_all_validation.py``, which charts *strength*). It reuses that
driver's case definitions and predictors so the figure reproduces exactly the
numbers it reports, for every UD dataset that records a *measured modulus* —
**E** (Li 2024, S-glass, indicative), **F** (Li 2025, S-glass, measured) and
**G** (Hsiao & Daniel 1996, carbon, measured) — against the two predictors:

  * **analytical** — the closed-form CLT series-average of the off-axis lamina
    modulus over the wrinkle profile (no FE solve), and
  * **FE** — ``modulus_retention`` from the linear static solve.

Two panels: ``KD_modulus`` vs peak misalignment angle (the carbon G data
extends the validated range down to ~0.52), and a predicted-vs-experimental
parity plot with a +/-10 % corridor. Dataset G is a periodic-in-x RVE; E/F are
single localized wrinkles.

Run::

    python validation/plot_modulus_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import validate_modulus as vm  # sibling driver: single source of truth


def _rows():
    """Compute ``(dataset, alpha_deg, kd_exp, kd_analytical, kd_fe)`` per case,
    using the driver's predictors so the chart matches its reported MAEs."""
    rows = []

    matf = vm.ML.get(vm.F_MAT)
    for _cid, app, lam, alpha, _dt, zf, _pos, kd_exp in vm.F_CASES:
        a = app / 2.0
        an = vm.analytical_modulus_kd(matf, amplitude=a, wavelength=lam,
                                      n_plies=vm.F_NPLY, t_ply=vm.F_TPLY,
                                      z_frac=zf)
        fe = vm.fe_modulus_kd(matf, amplitude=a, wavelength=lam,
                              n_plies=vm.F_NPLY, t_ply=vm.F_TPLY, z_frac=zf)
        rows.append(("F", alpha, kd_exp, an, fe))

    mate = vm.ML.get(vm.E_MAT)
    for _cid, npl, a1, lam, alpha, _dt, e_test in vm.E_CASES:
        a = a1 / 2.0
        an = vm.analytical_modulus_kd(mate, amplitude=a, wavelength=lam,
                                      n_plies=npl, t_ply=vm.E_TPLY)
        fe = vm.fe_modulus_kd(mate, amplitude=a, wavelength=lam,
                              n_plies=npl, t_ply=vm.E_TPLY)
        rows.append(("E", alpha, e_test / vm.E_E1_CARD, an, fe))

    matg = vm.ML.get(vm.G_MAT)
    for _cid, morph, amp, lam, npl, tp, kd_exp in vm.G_CASES:
        alpha = float(np.degrees(np.arctan(2.0 * np.pi * amp / lam)))
        decay = "linear" if morph == "graded" else "gaussian"
        an = vm.analytical_modulus_kd(matg, amplitude=amp, wavelength=lam,
                                      n_plies=npl, t_ply=tp, morphology=morph,
                                      periodic=True, decay=decay)
        fe = vm.fe_modulus_kd(matg, amplitude=amp, wavelength=lam, n_plies=npl,
                              t_ply=tp, morphology=morph, periodic=True)
        rows.append(("G", alpha, kd_exp, an, fe))

    return rows


# dataset -> (marker, label, measured?)
DS = {
    "E": ("D", "E Li2024 (S-glass, indic.)", False),
    "F": ("o", "F Li2025 (S-glass, meas.)", True),
    "G": ("*", "G Hsiao-Daniel (carbon, meas.)", True),
}
C_EXP, C_AN, C_FE = "#111111", "#1f77b4", "#d62728"


def main():
    rows = _rows()

    # Per-dataset / per-predictor MAE table.
    print(f"{'Dataset':<10} {'N':>3} {'analytical MAE':>15} {'FE MAE':>9}")
    print("-" * 40)
    for ds in ("F", "E", "G"):
        sub = [r for r in rows if r[0] == ds]
        an_e = np.mean([abs(an - ex) / ex for _, _, ex, an, _ in sub]) * 100
        fe_e = np.mean([abs(fe - ex) / ex for _, _, ex, _, fe in sub]) * 100
        print(f"{ds:<10} {len(sub):>3} {an_e:>13.1f} % {fe_e:>7.1f} %")

    fig, (ax_a, ax_p) = plt.subplots(1, 2, figsize=(13.0, 5.8))

    # Panel (a): KD_modulus vs misalignment angle.
    for ds, (mk, lab, meas) in DS.items():
        sub = [r for r in rows if r[0] == ds]
        a = [r[1] for r in sub]
        ax_a.scatter(a, [r[2] for r in sub], marker=mk, s=95,
                     facecolors=(C_EXP if meas else "none"), edgecolors=C_EXP,
                     linewidths=1.5, zorder=6, label=f"exp · {lab}")
        ax_a.scatter(a, [r[3] for r in sub], marker=mk, s=55,
                     facecolors="none", edgecolors=C_AN, linewidths=1.6)
        ax_a.scatter(a, [r[4] for r in sub], marker=mk, s=55,
                     facecolors="none", edgecolors=C_FE, linewidths=1.6)
    ax_a.scatter([], [], marker="s", facecolors=C_EXP, edgecolors=C_EXP, s=70,
                 label="— experimental (filled=meas, open=indic.)")
    ax_a.scatter([], [], marker="s", facecolors="none", edgecolors=C_AN, s=70,
                 label="— analytical (CLT)")
    ax_a.scatter([], [], marker="s", facecolors="none", edgecolors=C_FE, s=70,
                 label="— FE (modulus_retention)")
    ax_a.set_xlabel(r"Peak misalignment angle  $\theta_{max}$ (deg)")
    ax_a.set_ylabel(r"Modulus knockdown  $E_x/E_{x0}$")
    ax_a.set_title("(a) Stiffness knockdown vs angle — all modulus datasets")
    ax_a.set_ylim(0.45, 1.04)
    ax_a.grid(alpha=0.3)
    ax_a.legend(fontsize=7.6, loc="lower left")

    # Panel (b): predicted vs experimental parity, +/-10 % corridor.
    xs = np.linspace(0.4, 1.05, 50)
    ax_p.plot(xs, xs, "k-", lw=1.0, zorder=1)
    ax_p.fill_between(xs, 0.9 * xs, 1.1 * xs, color="0.85", alpha=0.6,
                      zorder=0, label="±10 % corridor")
    for ds, (mk, _lab, _meas) in DS.items():
        sub = [r for r in rows if r[0] == ds]
        ax_p.scatter([r[2] for r in sub], [r[3] for r in sub], marker=mk,
                     s=80, facecolors=C_AN, edgecolors="k", linewidths=0.5,
                     zorder=3, label=f"analytical · {ds}")
        ax_p.scatter([r[2] for r in sub], [r[4] for r in sub], marker=mk,
                     s=80, facecolors=C_FE, edgecolors="k", linewidths=0.5,
                     zorder=3, label=f"FE · {ds}")
    ax_p.set_xlim(0.45, 1.04)
    ax_p.set_ylim(0.45, 1.04)
    ax_p.set_aspect("equal")
    ax_p.set_xlabel("Experimental modulus knockdown")
    ax_p.set_ylabel("Predicted modulus knockdown")
    ax_p.set_title("(b) Predicted vs experimental")
    ax_p.grid(alpha=0.3)
    ax_p.legend(fontsize=7.5, loc="upper left", ncol=2)

    fig.suptitle("WrinkleFE stiffness (modulus) validation — analytical & FE "
                 "vs measured", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = REPO / "validation" / "fig_modulus_validation.png"
    fig.savefig(out, dpi=200)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
