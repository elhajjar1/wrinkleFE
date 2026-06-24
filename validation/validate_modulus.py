#!/usr/bin/env python3
"""Stiffness-only validation: WrinkleFE modulus knockdown vs experiment.

Compares two WrinkleFE predictors of the **axial Young's modulus** knockdown
(``E_x,wrinkled / E_x,pristine``) against the measured modulus knockdown in the
validation database, for every UD dataset that reports a modulus:

* **Analytical (CLT series-average).** WrinkleFE's shipped *analytical* path has
  no stiffness model (it returns ``modulus_retention = 1.0``), so this driver
  computes a closed-form Classical-Lamination-Theory estimate from the same
  primitives: the local fibre angle ``theta(x, z)`` from the wrinkle slope and
  the through-thickness decay, the off-axis lamina axial modulus ``Ex(theta)``,
  thickness-averaged to a section modulus and then series-averaged along the
  load direction.  This is the same off-axis-compliance integration used by
  Hsiao & Daniel (1996).
* **FE.** ``WrinkleAnalysis(...).run().modulus_retention`` from the linear
  static solve (mean fibre-direction stress / applied strain, wrinkled vs
  pristine).

Datasets:
  F  Li et al. (2025)      AC318 S-glass/epoxy (vac-bag)   measured KD_modulus
  E  Li, X. et al. (2024)  AC318 S-glass/epoxy (moulded)   indicative (E_Test/58 GPa)
  G  Hsiao & Daniel (1996) IM6G/3501-6 carbon/epoxy        measured (uniform + graded)

Usage::

    python validation/validate_modulus.py                # all datasets -> CSV
    python validation/validate_modulus.py --csv out.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis  # noqa: E402
from wrinklefe.core.material import MaterialLibrary, OrthotropicMaterial  # noqa: E402
from wrinklefe.core.wrinkle import GaussianSinusoidal  # noqa: E402

ML = MaterialLibrary()
NX, NY, NZ = 32, 2, 2

# --- Dataset F (Li 2025) single wrinkle; measured KD_modulus -----------
# (case, A_pp, L, alpha_deg, D/T, z_frac, position, KD_mod_exp)
F_CASES = [
    ("S-M-1", 1.5, 26.0, 10.3, 0.122, 0.5,       "Middle", 0.931),
    ("S-M-2", 1.5, 12.9, 20.1, 0.122, 0.5,       "Middle", 0.864),
    ("S-M-3", 1.5,  8.1, 30.2, 0.122, 0.5,       "Middle", 0.811),
    ("S-M-4", 1.0,  8.6, 20.1, 0.081, 0.5,       "Middle", 0.967),
    ("S-M-5", 0.5,  4.3, 20.1, 0.041, 0.5,       "Middle", 0.975),
    ("S-A-2", 1.5, 12.9, 20.1, 0.122, 10.0 / 14, "Above",  0.923),
]
F_MAT, F_NPLY, F_TPLY = "AC318_S6C10_vacbag", 14, 0.44

# --- Dataset E (Li 2024) single wrinkle; indicative KD = E_Test/58 GPa --
# (case, n_plies, A1, L, alpha_deg, D/T, E_Test_GPa)
E_CASES = [
    ("6.3-s-1", 15, 0.314, 11.4,  4.9, 0.025, 54.19),
    ("6.3-s-2", 15, 0.332,  5.6, 10.6, 0.026, 55.59),
    ("6.3-s-3", 15, 0.328,  3.6, 16.0, 0.026, 54.16),
    ("6.3-s-4", 15, 0.708,  7.4, 16.7, 0.056, 51.54),
    ("6.3-s-5", 15, 0.992, 11.0, 15.8, 0.079, 44.03),
    ("4.2-s-4", 10, 0.696,  7.4, 16.5, 0.083, 49.60),
    ("4.2-s-5", 10, 0.886, 11.0, 14.2, 0.105, 48.17),
    ("8.4-s-4", 20, 0.702,  7.4, 16.6, 0.042, 56.30),
    ("8.4-s-5", 20, 0.997, 11.0, 15.9, 0.059, 54.03),
]
E_MAT, E_TPLY, E_E1_CARD = "AC318_S6C10", 0.42, 58.0

# --- Dataset G (Hsiao & Daniel 1996) periodic-RVE; measured ------------
# (case, morphology, A, L, n_plies, t_ply, KD_mod_exp)
G_CASES = [
    ("HD-uniform", "uniform", 1.19, 27.9, 20, 0.95,  0.571),
    ("HD-graded",  "graded",  0.29, 14.5, 24, 0.381, 0.941),
]
G_MAT = "IM6G_3501_6"


def analytical_modulus_kd(mat: OrthotropicMaterial, *, amplitude, wavelength,
                          n_plies, t_ply, z_frac=0.5, morphology="graded",
                          decay="gaussian", periodic=False, nxs=600):
    """CLT series-average axial-modulus knockdown ``E_eff / E1``.

    ``morphology='uniform'`` applies the full amplitude through the thickness
    (no decay). ``'graded'`` tapers the amplitude through the thickness with
    either WrinkleFE's Gaussian decay ``decay='gaussian'`` (σ = max(λ/2, A),
    the package's ``graded`` morphology default — used for the Li machined-
    insert wrinkles) or a linear taper ``decay='linear'`` (the profile stated
    by Hsiao & Daniel 1996 for their graded specimen). ``periodic=True``
    represents a periodic-in-x RVE (one wavelength, wide envelope) rather than
    a single localized wrinkle.
    """
    E1, E2, G12, nu12 = mat.E1, mat.E2, mat.G12, mat.nu12
    T = n_plies * t_ply
    Lx = wavelength if periodic else max(3.0 * wavelength, 10.0)
    width = 8.0 * wavelength if periodic else wavelength / 2.0
    prof = GaussianSinusoidal(amplitude=amplitude, wavelength=wavelength,
                              width=width, center=Lx / 2.0)
    x = np.linspace(0.0, Lx, nxs)
    dzdx = np.asarray(prof.slope(x))
    z_p = -T / 2.0 + (np.arange(n_plies) + 0.5) * t_ply
    z_c = (z_frac - 0.5) * T
    if morphology == "uniform":
        phi = np.ones(n_plies)
    elif decay == "linear":  # linear taper, 1 at the wrinkle plane -> 0 at surfaces
        phi = np.clip(1.0 - np.abs(z_p - z_c) / (T / 2.0), 0.0, 1.0)
    else:  # WrinkleFE 'graded' default: Gaussian through-thickness decay
        sigma = max(wavelength / 2.0, amplitude)
        phi = np.exp(-((z_p - z_c) ** 2) / (2.0 * sigma ** 2))
    theta = np.arctan(np.outer(dzdx, phi))
    c, s = np.cos(theta), np.sin(theta)
    inv_Ex = (c**4) / E1 + (1.0 / G12 - 2.0 * nu12 / E1) * (c**2) * (s**2) \
        + (s**4) / E2
    E_sec = (1.0 / inv_Ex).mean(axis=1)          # thickness-average at each x
    E_eff = 1.0 / np.mean(1.0 / E_sec)           # series-average along x
    return float(E_eff / E1)


def fe_modulus_kd(mat, *, amplitude, wavelength, n_plies, t_ply, z_frac=0.5,
                  morphology="graded", periodic=False):
    """FE ``modulus_retention`` from the linear static solve."""
    L = wavelength
    Lx = L if periodic else max(3.0 * L, 10.0)
    width = 8.0 * L if periodic else L / 2.0
    cfg = AnalysisConfig(
        amplitude=amplitude, wavelength=L, width=width,
        morphology=morphology, loading="compression",
        material=mat, angles=[0.0] * n_plies, ply_thickness=t_ply,
        nx=NX, ny=NY, nz_per_ply=NZ, domain_length=Lx, domain_width=10.0,
        applied_strain=-0.01, wrinkle_z_position=z_frac,
    )
    return float(WrinkleAnalysis(cfg).run().modulus_retention)


def _row(dataset, case, alpha, kd_exp, kd_an, kd_fe, basis):
    e_an = 100.0 * (kd_an - kd_exp) / kd_exp
    e_fe = 100.0 * (kd_fe - kd_exp) / kd_exp
    print(f"  {case:9s} a={alpha:4.1f}  exp={kd_exp:.3f}  "
          f"an={kd_an:.3f} ({e_an:+5.1f}%)  fe={kd_fe:.3f} ({e_fe:+5.1f}%)")
    return dict(dataset=dataset, case=case, alpha_deg=alpha, kd_exp=kd_exp,
                kd_basis=basis, kd_analytical=round(kd_an, 4),
                kd_fe=round(kd_fe, 4), err_an_pct=round(e_an, 1),
                err_fe_pct=round(e_fe, 1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(Path(__file__).with_name(
        "modulus_validation.csv")))
    args = ap.parse_args(argv)

    rows = []

    matf = ML.get(F_MAT)
    print("=== Dataset F (Li 2025) — measured KD_modulus ===")
    for cid, App, L, a, _dt, zf, _pos, kd_exp in F_CASES:
        A = App / 2.0
        kd_an = analytical_modulus_kd(matf, amplitude=A, wavelength=L,
                                      n_plies=F_NPLY, t_ply=F_TPLY, z_frac=zf)
        kd_fe = fe_modulus_kd(matf, amplitude=A, wavelength=L, n_plies=F_NPLY,
                              t_ply=F_TPLY, z_frac=zf)
        rows.append(_row("F", cid, a, kd_exp, kd_an, kd_fe, "measured"))

    mate = ML.get(E_MAT)
    print("=== Dataset E (Li 2024) — indicative KD = E_Test/58 GPa ===")
    for cid, npl, A1, L, a, _dt, e_test in E_CASES:
        A = A1 / 2.0
        kd_exp = e_test / E_E1_CARD
        kd_an = analytical_modulus_kd(mate, amplitude=A, wavelength=L,
                                      n_plies=npl, t_ply=E_TPLY)
        kd_fe = fe_modulus_kd(mate, amplitude=A, wavelength=L, n_plies=npl,
                              t_ply=E_TPLY)
        rows.append(_row("E", cid, a, kd_exp, kd_an, kd_fe, "indicative/58GPa"))

    matg = ML.get(G_MAT)
    print("=== Dataset G (Hsiao & Daniel 1996) — measured, periodic RVE ===")
    for cid, morph, A, L, npl, tp, kd_exp in G_CASES:
        alpha = np.degrees(np.arctan(2 * np.pi * A / L))
        kd_an = analytical_modulus_kd(
            matg, amplitude=A, wavelength=L, n_plies=npl, t_ply=tp,
            morphology=morph, periodic=True,
            decay=("linear" if morph == "graded" else "gaussian"))
        kd_fe = fe_modulus_kd(matg, amplitude=A, wavelength=L, n_plies=npl,
                              t_ply=tp, morphology=morph, periodic=True)
        rows.append(_row("G", cid, alpha, kd_exp, kd_an, kd_fe, "measured"))

    for ds in ("F", "E", "G"):
        sub = [r for r in rows if r["dataset"] == ds]
        mae_an = np.mean([abs(r["err_an_pct"]) for r in sub])
        mae_fe = np.mean([abs(r["err_fe_pct"]) for r in sub])
        print(f"Dataset {ds}: analytical MAE {mae_an:.1f}%  |  FE MAE {mae_fe:.1f}%")

    out = Path(args.csv)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
