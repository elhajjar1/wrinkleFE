#!/usr/bin/env python3
"""Optimize decay_floor parameter for graded morphology.

Sweeps decay_floor values from 0.0 to 1.0 and computes mean absolute
error of Budiansky-Fleck knockdown vs Elhajjar (2025) experimental data.

Usage:
    python validation/optimize_decay_floor.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary

# ======================================================================
# Constants (same as validate_elhajjar2025.py)
# ======================================================================
MATERIAL_NAME = "T700_2510_Elhajjar2014"
PLY_THICKNESS = 0.152  # mm
N_PLIES = 16
LAM_THICKNESS = N_PLIES * PLY_THICKNESS  # 2.43 mm
LAYUP = [0, 45, 90, -45, 0, 45, -45, 0, 0, -45, 45, 0, -45, 90, 45, 0]
K_LAMBDA = 19.9
LAMBDA_MIN = 8.2  # mm
NX, NY, NZ = 12, 4, 1  # coarse mesh for fast sweeps

# Reference data from Elhajjar (2025) Fig 5b
REF_DATA = [
    (0.003, 1.02), (0.005, 1.00), (0.008, 0.95), (0.010, 0.90),
    (0.020, 0.80), (0.030, 0.72), (0.050, 0.62), (0.080, 0.52),
    (0.100, 0.47), (0.150, 0.40), (0.200, 0.37), (0.250, 0.35),
    (0.300, 0.32),
]


def run_case(dt_ratio: float, morphology: str, decay_floor: float) -> float:
    """Run one FE case, return Budiansky-Fleck knockdown."""
    amplitude = dt_ratio * LAM_THICKNESS
    wavelength = max(K_LAMBDA * amplitude, LAMBDA_MIN)
    width = 0.75 * wavelength
    domain_length = max(3.0 * wavelength, 10.0)

    mat = MaterialLibrary().get(MATERIAL_NAME)

    config = AnalysisConfig(
        amplitude=amplitude,
        wavelength=wavelength,
        width=width,
        morphology=morphology,
        decay_floor=decay_floor,
        loading="compression",
        material=mat,
        angles=LAYUP,
        ply_thickness=PLY_THICKNESS,
        nx=NX,
        ny=NY,
        nz_per_ply=NZ,
        domain_length=domain_length,
        domain_width=10.0,
        applied_strain=-0.01,
        solver="direct",
        verbose=False,
    )

    result = WrinkleAnalysis(config).run()
    # Use mesh-based max angle (accounts for decay mode) instead of
    # analytical effective_angle_rad (which ignores through-thickness decay)
    theta_mesh = result.mesh_max_angle_rad
    gamma_Y = mat.gamma_Y
    bf_kd = 1.0 / (1.0 + theta_mesh / gamma_Y)
    return bf_kd


def evaluate_floor(decay_floor: float, morphology: str = "graded") -> dict:
    """Evaluate a single decay_floor value across all D/T ratios."""
    errors = []
    abs_errors = []
    n_pass = 0

    for dt, ref in REF_DATA:
        bf_kd = run_case(dt, morphology, decay_floor)
        err = (bf_kd - ref) / ref * 100
        errors.append(err)
        abs_errors.append(abs(err))
        if abs(err) <= 20.0:
            n_pass += 1

    return {
        "decay_floor": decay_floor,
        "mean_abs_err": np.mean(abs_errors),
        "max_abs_err": np.max(abs_errors),
        "n_pass": n_pass,
        "errors": errors,
    }


def main():
    print("=" * 70)
    print("  Decay Floor Optimization for Graded Morphology")
    print("  Elhajjar (2025) Validation — Compression")
    print("=" * 70)
    print()

    # Also run stack (baseline) and uniform for comparison
    morphologies = [
        ("stack", [0.0]),
        ("uniform", [0.0]),
        ("graded", np.arange(0.0, 1.05, 0.1).tolist()),
    ]

    all_results = []
    t0 = time.monotonic()

    for morph, floors in morphologies:
        for floor in floors:
            label = f"{morph}" if morph != "graded" else f"graded(floor={floor:.1f})"
            print(f"  Evaluating {label}...", end="", flush=True)
            t1 = time.monotonic()
            res = evaluate_floor(floor, morph)
            elapsed = time.monotonic() - t1
            res["morphology"] = morph
            res["label"] = label
            all_results.append(res)
            print(f" MAE={res['mean_abs_err']:.1f}%  MaxE={res['max_abs_err']:.1f}%  "
                  f"Pass={res['n_pass']}/13  ({elapsed:.0f}s)")

    total = time.monotonic() - t0
    print(f"\n  Total: {total:.0f}s")

    # Find optimal
    graded_results = [r for r in all_results if r["morphology"] == "graded"]
    best = min(graded_results, key=lambda r: r["mean_abs_err"])

    print()
    print("=" * 70)
    print("  OPTIMIZATION RESULTS")
    print("=" * 70)
    print()
    print(f"  {'Configuration':<25s} {'MAE%':>6s} {'MaxE%':>6s} {'Pass':>6s}")
    print("  " + "-" * 50)
    for r in all_results:
        marker = " <-- BEST" if r is best else ""
        print(f"  {r['label']:<25s} {r['mean_abs_err']:>5.1f}% {r['max_abs_err']:>5.1f}% "
              f"{r['n_pass']:>3d}/13{marker}")

    print()
    print(f"  Optimal decay_floor = {best['decay_floor']:.2f}")
    print(f"  Mean absolute error = {best['mean_abs_err']:.1f}%")
    print(f"  Max absolute error  = {best['max_abs_err']:.1f}%")
    print(f"  Points within ±20%  = {best['n_pass']}/13")
    print("=" * 70)


if __name__ == "__main__":
    main()
