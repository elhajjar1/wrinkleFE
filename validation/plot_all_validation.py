#!/usr/bin/env python3
"""Single combined validation chart: predicted vs experimental knockdown.

Plots every *single-wrinkle* experimental case in the WrinkleFE validation
database (Datasets A-F of VALIDATION_DATA) on one parity axes, with the
+/-20 % pass corridor around the y = x diagonal.

The point of putting them on one chart is to show, at a glance, that we do
**not** use one method everywhere: each dataset is predicted with the model
that physically applies to it (encoded by marker shape), while colour
encodes the dataset:

  * Multidirectional laminates (A, B, C, D)  -> the angle-based analytical
    models run through ``WrinkleAnalysis`` (Budiansky-Fleck kink-band for
    compression, the three-mechanism min() for tension, with the
    morphology factor for Wang's concave/convex cases).  These are
    scale-invariant in D/T.
  * Unidirectional laminates (E, F)          -> the two-parameter
    penetration gate ``KD = 1 - (1 - KD_angle(theta)) * S(D/T) * P(z)``,
    which the angle-only models cannot reproduce (the Li grids vary
    knockdown at *fixed* angle).  E uses the moulded preset, F the
    vacuum-bag preset with the through-thickness position factor.

A parity plot is used (not KD-vs-D/T) precisely because E and F are
normalised on different pristine strengths (different material
realizations) and cannot share an absolute axis; (KD_exp, KD_pred) pairs
are comparable regardless.

Run::

    python validation/plot_all_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary
from wrinklefe.core.penetration_gate import (
    GATE_LI2024_MOULDED,
    GATE_LI2025_VACBAG,
    penetration_gate_kd,
)

ML = MaterialLibrary()

# --- Layups (VALIDATION_DATA section 2) -------------------------------
ELHAJJAR = [0, 45, 90, -45, 0, 45, -45, 0]
ELHAJJAR = ELHAJJAR + ELHAJJAR[::-1]                       # [...]_s, 16 plies
MUKHO = [45, 45, 90, 90, -45, -45, 0, 0] * 3
MUKHO = MUKHO + MUKHO[::-1]                                # [...]_3s, 48 plies
WANG = [45, 0, -45, 90, 45, 0, -45, 0, 45, 0]
WANG = WANG + WANG[::-1]                                   # [...]_s, 20 plies


def _analytical_kd(A, lam, *, material, angles, t_ply, loading,
                   morphology="uniform", onset=False):
    """Run the analytical pipeline and return the predicted knockdown."""
    cfg = AnalysisConfig(
        amplitude=A, wavelength=lam, width=0.75 * lam,
        morphology=morphology, loading=loading,
        material=ML.get(material), angles=angles, ply_thickness=t_ply,
        applied_strain=(+0.01 if loading == "tension" else -0.01),
        analytical_only=True,
    )
    res = WrinkleAnalysis(cfg).run()
    if onset and res.analytical_onset_knockdown is not None:
        return float(res.analytical_onset_knockdown)
    return float(res.analytical_knockdown)


# ----------------------------------------------------------------------
# Experimental cases.  Each entry yields (KD_exp, KD_pred).
# ----------------------------------------------------------------------
def dataset_A():
    """Elhajjar (2025) compression -- BF kink-band, T700/2510."""
    rows = [  # (A_mm, KD_exp)
        (0.0073, 1.02), (0.0122, 1.00), (0.0194, 0.95), (0.0243, 0.90),
        (0.0486, 0.80), (0.0729, 0.72), (0.1215, 0.62), (0.1944, 0.52),
        (0.2430, 0.47), (0.3645, 0.40), (0.4860, 0.37), (0.6075, 0.35),
        (0.7290, 0.32),
    ]
    out = []
    for A, kd in rows:
        lam = max(19.9 * A, 8.2)
        out.append((kd, _analytical_kd(A, lam, material="T700_2510",
                                       angles=ELHAJJAR, t_ply=0.152,
                                       loading="compression")))
    return out


def dataset_B():
    """Elhajjar (2025) tension -- three-mechanism, T700/2510."""
    rows = [
        (0.0073, 1.00), (0.0122, 0.95), (0.0243, 0.90), (0.1215, 0.77),
        (0.2430, 0.65), (0.4860, 0.55), (0.7290, 0.47),
    ]
    out = []
    for A, kd in rows:
        lam = max(19.9 * A, 8.2)
        out.append((kd, _analytical_kd(A, lam, material="T700_2510",
                                       angles=ELHAJJAR, t_ply=0.152,
                                       loading="tension")))
    return out


def dataset_C_comp():
    """Mukhopadhyay (2015) compression -- BF kink-band, IM7/8552."""
    rows = [(0.168, 0.82), (0.372, 0.68), (0.492, 0.67)]
    out = []
    for A, kd in rows:
        lam = max(22.0 * A, 10.0)
        out.append((kd, _analytical_kd(A, lam, material="IM7_8552",
                                       angles=MUKHO, t_ply=0.125,
                                       loading="compression",
                                       morphology="graded")))
    return out


def dataset_C_tens():
    """Mukhopadhyay (2015) tension ultimate -- three-mechanism."""
    rows = [(0.168, 0.94), (0.372, 0.83), (0.492, 0.77)]
    out = []
    for A, kd in rows:
        lam = max(22.0 * A, 10.0)
        out.append((kd, _analytical_kd(A, lam, material="IM7_8552",
                                       angles=MUKHO, t_ply=0.125,
                                       loading="tension",
                                       morphology="graded")))
    return out


def dataset_C_onset():
    """Mukhopadhyay (2015) delamination onset -- KD_oop mechanism."""
    rows = [(0.372, 0.70), (0.492, 0.67), (0.570, 0.51)]
    out = []
    for A, kd in rows:
        lam = max(22.0 * A, 10.0)
        out.append((kd, _analytical_kd(A, lam, material="IM7_8552",
                                       angles=MUKHO, t_ply=0.125,
                                       loading="tension", onset=True,
                                       morphology="graded")))
    return out


def dataset_D():
    """Wang (2021) compression -- BF with morphology factor.

    Wang's T800/epoxy alias is not in the library; T800S_M21 is the
    closest built-in card (the morphology asymmetry, not the exact
    modulus, is the feature under test here).
    """
    rows = [  # (A_mm, morphology, KD_exp)
        (0.38, "convex", 0.729), (0.76, "convex", 0.677),
        (0.38, "concave", 0.635), (0.76, "concave", 0.419),
    ]
    out = []
    for A, morph, kd in rows:
        out.append((kd, _analytical_kd(A, 24.0, material="T800S_M21",
                                       angles=WANG, t_ply=0.19,
                                       loading="compression",
                                       morphology=morph)))
    return out


def dataset_E():
    """Li (2024) UD compression -- penetration gate (moulded), z = mid."""
    # (theta_deg, D/T, KD_exp) from VALIDATION_DATA section 2.7.
    grid = [
        (4.9, 0.025, 0.907), (10.6, 0.026, 0.823), (16.0, 0.026, 0.758),
        (16.7, 0.056, 0.612), (15.8, 0.079, 0.523), (16.5, 0.083, 0.545),
        (14.2, 0.105, 0.506), (16.6, 0.042, 0.657), (15.9, 0.059, 0.558),
    ]
    return [(kd, penetration_gate_kd(th, dt, GATE_LI2024_MOULDED,
                                     z_position=0.5))
            for th, dt, kd in grid]


def dataset_F():
    """Li (2025) UD compression -- penetration gate (vacuum-bag) + z."""
    # (theta_deg, D/T, z, KD_exp); S-A-2 is the near-surface case.
    grid = [
        (10.3, 0.122, 0.5, 0.891), (20.1, 0.122, 0.5, 0.629),
        (30.2, 0.122, 0.5, 0.472), (20.1, 0.081, 0.5, 0.943),
        (20.1, 0.041, 0.5, 1.000), (20.1, 0.122, 10.0 / 14.0, 0.981),
    ]
    return [(kd, penetration_gate_kd(th, dt, GATE_LI2025_VACBAG,
                                     z_position=z))
            for th, dt, z, kd in grid]


# Dataset -> (cases, colour, marker, method label, method family).
DATASETS = {
    "A Elhajjar comp": (dataset_A, "#1f77b4", "o", "BF kink-band"),
    "B Elhajjar tens": (dataset_B, "#17becf", "s", "3-mechanism"),
    "C Mukhopadhyay comp": (dataset_C_comp, "#2ca02c", "o", "BF kink-band"),
    "C Mukhopadhyay tens": (dataset_C_tens, "#98df8a", "s", "3-mechanism"),
    "C Mukhopadhyay onset": (dataset_C_onset, "#bcbd22", "P", "3-mech onset"),
    "D Wang conc/conv": (dataset_D, "#9467bd", "^", "BF + morphology"),
    "E Li2024 UD": (dataset_E, "#ff7f0e", "D", "penetration gate"),
    "F Li2025 UD": (dataset_F, "#d62728", "*", "penetration gate (+pos)"),
}


def main():
    fig, ax = plt.subplots(figsize=(8.2, 8.0))

    # +/-20 % corridor around the parity diagonal.
    xs = np.linspace(0.0, 1.15, 50)
    ax.plot(xs, xs, "k-", lw=1.0, zorder=1, label="parity (y = x)")
    ax.fill_between(xs, 0.8 * xs, 1.2 * xs, color="0.85", alpha=0.6,
                    zorder=0, label="+/-20 % corridor")

    print(f"{'Dataset':<24} {'N':>3} {'MAE':>7} {'PASS':>7}")
    print("-" * 46)
    all_err = []
    for label, (fn, colour, marker, method) in DATASETS.items():
        pairs = fn()
        exp = np.array([p[0] for p in pairs])
        pred = np.array([p[1] for p in pairs])
        err = np.abs(pred - exp) / exp
        all_err.extend(err.tolist())
        n_pass = int((err <= 0.20).sum())
        print(f"{label:<24} {len(pairs):>3} {err.mean()*100:>6.1f}% "
              f"{n_pass:>3}/{len(pairs)}")
        ax.scatter(exp, pred, c=colour, marker=marker, s=90,
                   edgecolor="k", linewidth=0.5, zorder=3,
                   label=f"{label}  [{method}]")
    print("-" * 46)
    print(f"{'OVERALL':<24} {len(all_err):>3} "
          f"{np.mean(all_err)*100:>6.1f}% "
          f"{int(np.sum(np.array(all_err) <= 0.20)):>3}/{len(all_err)}")

    ax.set_xlim(0.0, 1.15)
    ax.set_ylim(0.0, 1.15)
    ax.set_aspect("equal")
    ax.set_xlabel("Experimental knockdown  $KD_{exp}$")
    ax.set_ylabel("Predicted knockdown  $KD_{pred}$")
    ax.set_title("WrinkleFE validation -- all single-wrinkle cases (A-F)\n"
                 "marker = method, colour = dataset, band = +/-20 %")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    out = REPO / "validation" / "fig_all_validation_parity.png"
    fig.savefig(out, dpi=300)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
