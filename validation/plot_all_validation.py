#!/usr/bin/env python3
"""Single combined validation chart: predicted vs experimental knockdown.

Plots every *single-wrinkle* experimental case in the WrinkleFE validation
database (Datasets A-F of VALIDATION_DATA, plus the ledger's Datasets H
to K; I is a distributed seven-wave tension case) on one parity axes,
with the +/-20 % pass corridor around the y = x diagonal.

The point of putting them on one chart is to show, at a glance, that we do
**not** use one method everywhere: each dataset is predicted with the model
that physically applies to it (encoded by marker shape), while colour
encodes the dataset:

  * Multidirectional laminates (A, B, C, D)  -> the angle-based analytical
    models run through ``WrinkleAnalysis`` (Budiansky-Fleck kink-band for
    compression, the three-mechanism min() for tension, with the
    morphology factor for Wang's concave/convex cases).  These are
    scale-invariant in D/T.
  * Dataset H (Shi 2025, UD + multidirectional) -> shown TWICE: the
    angle-based analytical prediction (hollow-side markers, like A-D;
    no penetration-gate preset is calibrated for this carbon material)
    AND the first-ply FE LaRC05 retention (X markers), which is the
    measured-checked path for the multidirectional half. The ledger
    recipe drives both, via scripts/validate.py case_config.
  * Datasets J (Thor 2021, quasi-isotropic + UD IM7/8552) and K
    (Pilato 2022, near-UD industrial wrinkles) -> shown twice the same
    way. Both are whole-thickness waves, which the FE mesh reproduces;
    no gate preset is calibrated for either material.
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


def case_config(A, lam, *, material, angles, t_ply, loading,
                morphology="uniform"):
    """The A-D recipe: one ``AnalysisConfig`` per experimental case.

    Shared with ``strength_error_summary.py``, which runs the same
    configs through the FE so the two paths see identical inputs.
    """
    return AnalysisConfig(
        amplitude=A, wavelength=lam, width=0.75 * lam,
        morphology=morphology, loading=loading,
        material=ML.get(material), angles=angles, ply_thickness=t_ply,
        analytical_only=True,
    )


def _analytical_kd(A, lam, *, material, angles, t_ply, loading,
                   morphology="uniform", onset=False):
    """Run the analytical pipeline and return the predicted knockdown."""
    cfg = case_config(A, lam, material=material, angles=angles,
                      t_ply=t_ply, loading=loading, morphology=morphology)
    res = WrinkleAnalysis(cfg).run()
    if onset and res.analytical_onset_knockdown is not None:
        return float(res.analytical_onset_knockdown)
    return float(res.analytical_knockdown)


# ----------------------------------------------------------------------
# Experimental cases.  Each entry yields (KD_exp, KD_pred).
# ----------------------------------------------------------------------
A_ROWS = [  # Elhajjar (2025) compression: (A_mm, KD_exp)
    (0.0073, 1.02), (0.0122, 1.00), (0.0194, 0.95), (0.0243, 0.90),
    (0.0486, 0.80), (0.0729, 0.72), (0.1215, 0.62), (0.1944, 0.52),
    (0.2430, 0.47), (0.3645, 0.40), (0.4860, 0.37), (0.6075, 0.35),
    (0.7290, 0.32),
]
B_ROWS = [  # Elhajjar (2025) tension: (A_mm, KD_exp)
    (0.0073, 1.00), (0.0122, 0.95), (0.0243, 0.90), (0.1215, 0.77),
    (0.2430, 0.65), (0.4860, 0.55), (0.7290, 0.47),
]
C_COMP_ROWS = [(0.168, 0.82), (0.372, 0.68), (0.492, 0.67)]
C_TENS_ROWS = [(0.168, 0.94), (0.372, 0.83), (0.492, 0.77)]
C_ONSET_ROWS = [(0.372, 0.70), (0.492, 0.67), (0.570, 0.51)]
D_ROWS = [  # Wang (2021): (A_mm, morphology, KD_exp)
    (0.38, "convex", 0.729), (0.76, "convex", 0.677),
    (0.38, "concave", 0.635), (0.76, "concave", 0.419),
]
E_GRID = [  # Li (2024), VALIDATION_DATA section 2.7: (theta_deg, D/T, KD_exp)
    (4.9, 0.025, 0.907), (10.6, 0.026, 0.823), (16.0, 0.026, 0.758),
    (16.7, 0.056, 0.612), (15.8, 0.079, 0.523), (16.5, 0.083, 0.545),
    (14.2, 0.105, 0.506), (16.6, 0.042, 0.657), (15.9, 0.059, 0.558),
]

F_GRID = [  # Li (2025), ledger case order: (theta_deg, D/T, z, KD_exp);
    # S-A-2 is the near-surface case.
    (10.3, 0.122, 0.5, 0.891), (20.1, 0.122, 0.5, 0.629),
    (30.2, 0.122, 0.5, 0.472), (20.1, 0.081, 0.5, 0.943),
    (20.1, 0.041, 0.5, 1.000), (20.1, 0.122, 10.0 / 14.0, 0.981),
]


def elhajjar_wavelength(A):
    return max(19.9 * A, 8.2)


def mukhopadhyay_wavelength(A):
    return max(22.0 * A, 10.0)


def dataset_A():
    """Elhajjar (2025) compression -- BF kink-band, T700/2510."""
    out = []
    for A, kd in A_ROWS:
        lam = elhajjar_wavelength(A)
        out.append((kd, _analytical_kd(A, lam, material="T700_2510",
                                       angles=ELHAJJAR, t_ply=0.152,
                                       loading="compression")))
    return out


def dataset_B():
    """Elhajjar (2025) tension -- three-mechanism, T700/2510."""
    out = []
    for A, kd in B_ROWS:
        lam = elhajjar_wavelength(A)
        out.append((kd, _analytical_kd(A, lam, material="T700_2510",
                                       angles=ELHAJJAR, t_ply=0.152,
                                       loading="tension")))
    return out


def dataset_C_comp():
    """Mukhopadhyay (2015) compression -- BF kink-band, IM7/8552."""
    out = []
    for A, kd in C_COMP_ROWS:
        lam = mukhopadhyay_wavelength(A)
        out.append((kd, _analytical_kd(A, lam, material="IM7_8552",
                                       angles=MUKHO, t_ply=0.125,
                                       loading="compression",
                                       morphology="graded")))
    return out


def dataset_C_tens():
    """Mukhopadhyay (2015) tension ultimate -- three-mechanism."""
    out = []
    for A, kd in C_TENS_ROWS:
        lam = mukhopadhyay_wavelength(A)
        out.append((kd, _analytical_kd(A, lam, material="IM7_8552",
                                       angles=MUKHO, t_ply=0.125,
                                       loading="tension",
                                       morphology="graded")))
    return out


def dataset_C_onset():
    """Mukhopadhyay (2015) delamination onset -- KD_oop mechanism."""
    out = []
    for A, kd in C_ONSET_ROWS:
        lam = mukhopadhyay_wavelength(A)
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
    out = []
    for A, morph, kd in D_ROWS:
        out.append((kd, _analytical_kd(A, 24.0, material="T800S_M21",
                                       angles=WANG, t_ply=0.19,
                                       loading="compression",
                                       morphology=morph)))
    return out


def dataset_E():
    """Li (2024) UD compression -- penetration gate (moulded), z = mid."""
    return [(kd, penetration_gate_kd(th, dt, GATE_LI2024_MOULDED,
                                     z_position=0.5))
            for th, dt, kd in E_GRID]


def dataset_F():
    """Li (2025) UD compression -- penetration gate (vacuum-bag) + z."""
    return [(kd, penetration_gate_kd(th, dt, GATE_LI2025_VACBAG,
                                     z_position=z))
            for th, dt, z, kd in F_GRID]


def _ledger_dataset(name_prefix: str):
    import json

    ledger = json.loads(
        (REPO / "tests" / "test_validation" / "ledger.json").read_text()
    )
    return next(d for d in ledger["datasets"]
                if d["name"].startswith(name_prefix))


def _dataset_H_analytical(half: str):
    """Shi (2025) Dataset H, analytical path, recipe-exact via the ledger."""
    sys.path.insert(0, str(REPO / "scripts"))
    from validate import case_config

    ds = _ledger_dataset(f"shi_2025_{half}")
    out = []
    for case in ds["cases"]:
        res = WrinkleAnalysis(case_config(ds, case)).run(analytical_only=True)
        out.append((float(case["measured_kd"]),
                    float(res.analytical_knockdown)))
    return out


def dataset_H_ud():
    """Shi (2025) UD CFRP -- plain BF (no gate preset for this carbon)."""
    return _dataset_H_analytical("ud")


def dataset_H_md():
    """Shi (2025) multidirectional CFRP -- angle-based analytical."""
    return _dataset_H_analytical("md")


_H_FE_CACHE: dict[str, list] = {}


def _dataset_H_fe(prefix: str):
    """Shi (2025) first-ply FE LaRC05 retention (one 6-solve pass, cached)."""
    if not _H_FE_CACHE:
        sys.path.insert(0, str(REPO / "validation"))
        from strength_error_summary import fe_larc05_errors_shi

        for case, (kd, meas, _e) in fe_larc05_errors_shi().items():
            _H_FE_CACHE.setdefault(case[:4], []).append((meas, kd))
    return _H_FE_CACHE[prefix]


def dataset_I():
    """Calvo (2023) MD CFRP tension, 7 distributed waves -- 3-mechanism."""
    sys.path.insert(0, str(REPO / "scripts"))
    from validate import case_config

    ds = _ledger_dataset("calvo_2023")
    out = []
    for case in ds["cases"]:
        res = WrinkleAnalysis(case_config(ds, case)).run(analytical_only=True)
        out.append((float(case["measured_kd"]),
                    float(res.analytical_knockdown)))
    return out


def dataset_I_fe():
    """Calvo (2023) first-ply FE LaRC05 retention (tension)."""
    sys.path.insert(0, str(REPO / "validation"))
    from strength_error_summary import fe_larc05_errors_calvo

    return [(meas, kd) for kd, meas, _e in fe_larc05_errors_calvo().values()]


def _ledger_analytical(prefix: str):
    """A ledger dataset's analytical prediction, recipe-exact."""
    sys.path.insert(0, str(REPO / "scripts"))
    from validate import case_config

    ds = _ledger_dataset(prefix)
    out = []
    for case in ds["cases"]:
        res = WrinkleAnalysis(case_config(ds, case)).run(analytical_only=True)
        out.append((float(case["measured_kd"]),
                    float(res.analytical_knockdown)))
    return out


def _ledger_fe(prefix: str):
    """A ledger dataset's first-ply FE LaRC05 retention."""
    sys.path.insert(0, str(REPO / "validation"))
    from strength_error_summary import fe_larc05_errors_ledger

    return [(meas, kd)
            for kd, meas, _e in fe_larc05_errors_ledger(prefix).values()]


def dataset_J_qi():
    """Thor (2021) quasi-isotropic IM7/8552, whole-thickness waves -- BF."""
    return _ledger_analytical("thor_2021_qi")


def dataset_J_ud():
    """Thor (2021) UD IM7/8552 -- plain BF (no gate preset for IM7/8552)."""
    return _ledger_analytical("thor_2021_ud")


def dataset_K():
    """Pilato (2022) near-UD industrial wrinkles -- plain BF."""
    return _ledger_analytical("pilato_2022")


def dataset_J_qi_fe():
    return _ledger_fe("thor_2021_qi")


def dataset_J_ud_fe():
    return _ledger_fe("thor_2021_ud")


def dataset_K_fe():
    return _ledger_fe("pilato_2022")


def dataset_H_ud_fe():
    return _dataset_H_fe("H-UD")


def dataset_H_md_fe():
    return _dataset_H_fe("H-MD")


# Dataset -> (cases, colour, marker, method label, method family).
DATASETS = {
    "A Elhajjar comp": (dataset_A, "#1f77b4", "o", "BF kink-band"),
    "B Elhajjar tens": (dataset_B, "#17becf", "s", "3-mechanism"),
    "C Mukhopadhyay comp": (dataset_C_comp, "#2ca02c", "o", "BF kink-band"),
    "C Mukhopadhyay tens": (dataset_C_tens, "#98df8a", "s", "3-mechanism"),
    "C Mukhopadhyay onset": (dataset_C_onset, "#bcbd22", "P", "3-mech onset"),
    "D Wang conc/conv": (dataset_D, "#9467bd", "^", "BF + morphology"),
    "E Li2024 UD comp": (dataset_E, "#ff7f0e", "D", "penetration gate"),
    "F Li2025 UD comp": (dataset_F, "#d62728", "*", "penetration gate (+pos)"),
    "H Shi2025 UD comp": (dataset_H_ud, "#8c564b", "v", "BF kink-band"),
    "H Shi2025 MD comp": (dataset_H_md, "#e377c2", "v", "BF kink-band"),
    "H Shi2025 UD comp (FE)": (dataset_H_ud_fe, "#8c564b", "X",
                          "FE LaRC05 retention"),
    "H Shi2025 MD comp (FE)": (dataset_H_md_fe, "#e377c2", "X",
                          "FE LaRC05 retention"),
    "I Calvo2023 MD tens": (dataset_I, "#7f7f7f", "h", "3-mechanism"),
    "I Calvo2023 MD tens (FE)": (dataset_I_fe, "#7f7f7f", "X",
                                 "FE LaRC05 retention"),
    "J Thor2021 QI comp": (dataset_J_qi, "#393b79", "p", "BF kink-band"),
    "J Thor2021 QI comp (FE)": (dataset_J_qi_fe, "#393b79", "X",
                                "FE LaRC05 retention"),
    "J Thor2021 UD comp": (dataset_J_ud, "#6b6ecf", "p", "BF kink-band"),
    "J Thor2021 UD comp (FE)": (dataset_J_ud_fe, "#6b6ecf", "X",
                                "FE LaRC05 retention"),
    "K Pilato2022 UD comp": (dataset_K, "#e7ba52", "<", "BF kink-band"),
    "K Pilato2022 UD comp (FE)": (dataset_K_fe, "#e7ba52", "X",
                                  "FE LaRC05 retention"),
}


def main():
    fig, ax = plt.subplots(figsize=(12.5, 8.0))

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
    ax.set_title("WrinkleFE validation -- all cases (A-K)\n"
                 "marker = method, colour = dataset, band = +/-20 %")
    ax.grid(alpha=0.3)
    # 20 series: the legend sits outside the axes so it hides no points.
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              framealpha=0.95)
    fig.tight_layout()
    out = REPO / "validation" / "fig_all_validation_parity.png"
    fig.savefig(out, dpi=300)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
