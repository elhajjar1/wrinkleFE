#!/usr/bin/env python3
"""Analytical tension knockdown model based on LaRC04/Pinho et al. (2005).

Physics-based approach using stress transformation into the fiber
coordinate frame of a misaligned ply under far-field tension:

  σ₁₁ = σ·cos²θ   (fiber-direction)
  σ₂₂ = σ·sin²θ   (transverse tension)
  τ₁₂ = σ·sinθ·cosθ  (in-plane shear)

Two competing failure modes:
  1. Fiber tension:  FI_F = σ₁₁ / Xt  → knockdown = cos²θ  (mild)
  2. Matrix tension: Hashin-type interaction  → knockdown from σ₂₂, τ₁₂

The laminate fails at whichever mode gives the lower retention.

For a multidirectional laminate, the 0° plies carry most of the tension
load. The waviness reduces their contribution. The effective laminate
knockdown accounts for the fraction of load carried by the 0° plies.

References:
  Pinho et al. (2005) NASA-TM-2005-213530, Eq. 40, 82
  Elhajjar (2025) Scientific Reports 15:25977

Usage:
    python validation/validate_tension_analytical.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wrinklefe.core.material import MaterialLibrary

# ======================================================================
# Constants
# ======================================================================
MATERIAL_NAME = "T700_2510_Elhajjar2014"
PLY_THICKNESS = 0.152  # mm
N_PLIES = 16
LAM_THICKNESS = N_PLIES * PLY_THICKNESS  # 2.43 mm

# Layup: [0/45/90/-45/0/45/-45/0]s
LAYUP = [0, 45, 90, -45, 0, 45, -45, 0, 0, -45, 45, 0, -45, 90, 45, 0]

# Wavelength model
K_LAMBDA = 19.9
LAMBDA_MIN = 8.2  # mm

# Pass criterion
TOLERANCE = 0.20

# Experimental tension data from Elhajjar (2025) Fig 5b
REF_TENSION = [
    (0.003, 1.00),
    (0.005, 0.95),
    (0.010, 0.90),
    (0.050, 0.77),
    (0.100, 0.65),
    (0.200, 0.55),
    (0.300, 0.47),
]

# Experimental compression data (for comparison)
REF_COMPRESSION = [
    (0.003, 1.02), (0.005, 1.00), (0.008, 0.95), (0.010, 0.90),
    (0.020, 0.80), (0.030, 0.72), (0.050, 0.62), (0.080, 0.52),
    (0.100, 0.47), (0.150, 0.40), (0.200, 0.37), (0.250, 0.35),
    (0.300, 0.32),
]


def theta_from_dt(dt_ratio: float) -> float:
    """Convert D/T ratio to maximum fiber misalignment angle [rad]."""
    amplitude = dt_ratio * LAM_THICKNESS
    wavelength = max(K_LAMBDA * amplitude, LAMBDA_MIN)
    return np.arctan(2 * np.pi * amplitude / wavelength)


def tension_knockdown_fiber(theta: float) -> float:
    """Fiber tension knockdown: cos²θ (load projection).

    From LaRC04 #3 (Pinho et al. Eq. 82): FI_F = σ₁₁/Xt
    Under far-field tension σ, local σ₁₁ = σ·cos²θ
    So failure at σ = Xt/cos²θ → knockdown = cos²θ
    """
    return np.cos(theta) ** 2


def tension_knockdown_matrix(theta: float, Yt: float, S12: float) -> float:
    """Matrix tension knockdown: Hashin-type interaction.

    From LaRC04 #1 (Pinho et al. Eq. 40, simplified for plane stress):
    FI_M = (σ₂₂/Yt_is)² + (τ₁₂/S_is)² = 1 at failure

    Under far-field tension σ:
      σ₂₂ = σ·sin²θ
      τ₁₂ = σ·sinθ·cosθ

    Setting FI_M = 1:
      σ² [ (sin²θ/Yt)² + (sinθ·cosθ/S12)² ] = 1
      σ_fail = 1 / sqrt[ (sin²θ/Yt)² + (sinθ·cosθ/S12)² ]

    Knockdown = σ_fail / σ_pristine where σ_pristine is the 0° ply
    fiber tension strength Xt. But for the matrix mode, we normalize
    by the pristine laminate strength (which is fiber-dominated).

    For the pristine laminate, θ=0 → σ₂₂=0, τ₁₂=0, so matrix mode
    never triggers. The knockdown is the ratio of the matrix-limited
    stress to the pristine fiber-limited stress.
    """
    if theta < 1e-10:
        return 1.0  # no waviness → no matrix knockdown

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    # σ_fail from matrix criterion
    term1 = (sin_t ** 2 / Yt) ** 2
    term2 = (sin_t * cos_t / S12) ** 2
    sigma_fail_matrix = 1.0 / np.sqrt(term1 + term2)

    return sigma_fail_matrix


def _max_consecutive_zero_plies(layup: list, tol: float = 5.0) -> int:
    """Find maximum number of consecutive 0-degree plies in layup."""
    max_count = 0
    count = 0
    for angle in layup:
        if abs(angle) < tol:
            count += 1
            max_count = max(max_count, count)
        else:
            count = 0
    return max(max_count, 1)  # at least 1


def tension_knockdown_oop(
    dt_ratio: float, mat, layup: list, ply_thickness: float = PLY_THICKNESS,
) -> dict:
    """Out-of-plane stress knockdown from wrinkle curvature (sigma_33 + tau_13).

    Curved fibers under tension generate interlaminar stresses via the
    curved-beam effect (Timoshenko & Gere, 1961):

    At the wrinkle CREST (max curvature, kappa_max):
        sigma_33 = sigma_11 * h_eff * kappa_max

    At the wrinkle INFLECTION POINT (max curvature gradient, |dkappa/dx|_max):
        tau_13   = sigma_11 * h_eff * |dkappa/dx|_max

    For a sinusoidal wrinkle z = A sin(2*pi*x/lambda):
        kappa_max      = (2*pi/lambda)^2 * A         [at crest]
        |dkappa/dx|_max = (2*pi/lambda)^3 * A         [at inflection]

    These two interlaminar stresses peak at DIFFERENT spatial locations
    (crest vs. inflection), separated by lambda/4. The delamination
    failure index envelope is:

        FI_crest      = (sigma_33 / Yt)^2             [mode I opening]
        FI_inflection = (tau_13   / S13)^2             [mode II shear]
        FI_max        = max(FI_crest, FI_inflection)

    Smooth progressive-damage interaction:
        KD_oop = 1 / sqrt(1 + FI_max)

    Physical basis:
        - Curved-beam theory (Timoshenko & Gere, 1961)
        - sigma_33: interlaminar normal (mode I delamination driver)
        - tau_13:   interlaminar shear  (mode II delamination driver)
        - h_eff = n_adj * t_ply  (adjacent 0-deg ply group at midplane)
        - Validated against 3D FE stress ratios

    Returns dict with sigma_33, tau_13, FI components, and KD_oop.
    """
    lam_thickness = len(layup) * ply_thickness
    amplitude = dt_ratio * lam_thickness
    wavelength = max(K_LAMBDA * amplitude, LAMBDA_MIN)

    if amplitude < 1e-12 or wavelength < 1e-12:
        return {"kd_oop": 1.0, "sigma33": 0.0, "tau13": 0.0,
                "FI_s33": 0.0, "FI_t13": 0.0, "controlling": "none"}

    # Peak curvature at crest: kappa = (2*pi/lambda)^2 * A
    kappa_max = (2.0 * np.pi / wavelength) ** 2 * amplitude

    # Max curvature gradient at inflection: |dkappa/dx| = (2*pi/lambda)^3 * A
    dkappa_dx_max = (2.0 * np.pi / wavelength) ** 3 * amplitude

    # Effective thickness: max consecutive 0-degree plies
    n_adj = _max_consecutive_zero_plies(layup)
    h_eff = n_adj * ply_thickness

    # Interlaminar stresses at pristine fiber failure (sigma_11 = Xt)
    Xt = mat.Xt
    Yt = mat.Yt if mat.Yt else 49.0
    S13 = mat.S13 if hasattr(mat, "S13") and mat.S13 else 85.0

    # sigma_33 at wrinkle crest (mode I — opening)
    sigma33 = Xt * h_eff * kappa_max  # MPa

    # tau_13 at wrinkle inflection (mode II — shear)
    tau13 = Xt * h_eff * dkappa_dx_max  # MPa

    # Failure indices (peak at different spatial locations)
    FI_s33 = (sigma33 / Yt) ** 2       # at crest
    FI_t13 = (tau13 / S13) ** 2        # at inflection point

    # Delamination envelope: max FI along wrinkle
    FI_max = max(FI_s33, FI_t13)
    controlling = "sigma_33" if FI_s33 >= FI_t13 else "tau_13"

    # Smooth interaction: progressive delamination damage
    kd_oop = 1.0 / np.sqrt(1.0 + FI_max)

    return {
        "kd_oop": min(kd_oop, 1.0),
        "sigma33": sigma33,
        "tau13": tau13,
        "FI_s33": FI_s33,
        "FI_t13": FI_t13,
        "controlling": controlling,
    }


def tension_knockdown_laminate(
    theta: float, mat, layup: list, dt_ratio: float = 0.0,
) -> dict:
    """Combined laminate tension knockdown accounting for ply mix.

    Three competing failure mechanisms for 0-degree plies:
      1. Fiber tension:  cos^2(theta) — load projection (mild knockdown)
      2. Matrix tension: Hashin-type sigma_22/tau_12 interaction (high angles)
      3. Out-of-plane:   Curved-beam sigma_33 delamination (moderate D/T)

    In a multidirectional laminate under tension:
    - 0-degree plies: affected by waviness, knockdown from above mechanisms
    - Off-axis plies (45/90/-45): less affected, provide load redistribution

    The laminate knockdown is a CLT-weighted average based on axial stiffness
    contribution of each ply group.

    Returns dict with all components for diagnostic output.
    """
    E11 = mat.E1
    E22 = mat.E2
    G12 = mat.G12
    Xt = mat.Xt
    Yt = mat.Yt if mat.Yt is not None else 50.0
    S12 = mat.S12 if mat.S12 is not None else 85.0

    # Use in-situ strengths (boost for embedded plies)
    # Thick ply: Yt_is = 1.12·√2·Yt ≈ 1.584·Yt (Pinho Eq. 47)
    # S12_is = √2·S12 (Pinho Eq. 57, linear shear)
    Yt_is = 1.12 * np.sqrt(2) * Yt
    S12_is = np.sqrt(2) * S12

    # --- Mechanism 1: Fiber tension (load projection) ---
    kd_fiber = tension_knockdown_fiber(theta)

    # --- Mechanism 2: Matrix tension (Hashin interaction) ---
    sigma_matrix = tension_knockdown_matrix(theta, Yt_is, S12_is)
    kd_matrix = min(sigma_matrix / Xt, 1.0)

    # --- Mechanism 3: Out-of-plane stress (curved-beam delamination) ---
    if dt_ratio > 0:
        oop = tension_knockdown_oop(dt_ratio, mat, layup)
        kd_oop = oop["kd_oop"]
    else:
        oop = {"kd_oop": 1.0, "sigma33": 0.0, "tau13": 0.0,
               "FI_s33": 0.0, "FI_t13": 0.0, "controlling": "none"}
        kd_oop = 1.0

    # 0° plies fail at the minimum of all three modes
    kd_0 = min(kd_fiber, kd_matrix, kd_oop)

    # Determine controlling mode
    if kd_oop <= kd_fiber and kd_oop <= kd_matrix:
        mode = "OOP"
    elif kd_matrix <= kd_fiber:
        mode = "matrix"
    else:
        mode = "fiber"

    # Count ply types
    n_0 = sum(1 for a in layup if abs(a) < 5)
    n_45 = sum(1 for a in layup if 40 < abs(a) < 50)
    n_90 = sum(1 for a in layup if abs(a) > 85)

    # Axial stiffness fractions (simplified CLT)
    Q11_0 = E11
    Q11_45 = E11 / 4 + E22 / 4 + G12 / 2  # approximate
    Q11_90 = E22

    total_stiffness = n_0 * Q11_0 + n_45 * Q11_45 + n_90 * Q11_90
    f_0 = n_0 * Q11_0 / total_stiffness

    # Off-axis plies: waviness has reduced effect
    kd_lam = f_0 * kd_0 + (1.0 - f_0) * 1.0

    return {
        "kd_fiber": kd_fiber,
        "kd_matrix": kd_matrix,
        "kd_oop": kd_oop,
        "kd_0": kd_0,
        "kd_lam": kd_lam,
        "mode": mode,
        "f_0": f_0,
        "oop_detail": oop,
    }


def main():
    script_dir = Path(__file__).resolve().parent
    mat = MaterialLibrary().get(MATERIAL_NAME)

    n_adj = _max_consecutive_zero_plies(LAYUP)

    print("=" * 80)
    print("  Analytical Tension Knockdown — Three-Mechanism Model")
    print("  1. Fiber tension: cos²θ (LaRC04 #3, Pinho Eq. 82)")
    print("  2. Matrix tension: Hashin σ₂₂/τ₁₂ interaction (LaRC04 #1, Pinho Eq. 40)")
    print("  3. Out-of-plane σ₃₃: Curved-beam delamination (Timoshenko)")
    print("=" * 80)
    print()
    print(f"  Material: {MATERIAL_NAME}")
    print(f"  Xt = {mat.Xt:.0f} MPa, Yt = {mat.Yt:.0f} MPa, S12 = {mat.S12:.0f} MPa")

    Yt_is = 1.12 * np.sqrt(2) * mat.Yt
    S12_is = np.sqrt(2) * mat.S12
    print(f"  In-situ: Yt_is = {Yt_is:.0f} MPa, S12_is = {S12_is:.0f} MPa")
    print(f"  Layup: [0/45/90/-45/0/45/-45/0]s ({N_PLIES} plies)")
    print(f"  Max adjacent 0° plies: {n_adj} (h_eff = {n_adj * PLY_THICKNESS:.3f} mm)")
    print(f"  Lambda model: lam = {K_LAMBDA:.1f}*A, min {LAMBDA_MIN:.1f} mm")
    print(f"  Tolerance: +/-{TOLERANCE*100:.0f}%")
    print()

    # Header
    print(f"  {'D/T':>6s} {'theta':>6s} {'kappa':>7s} {'KD_fib':>7s} {'KD_mat':>7s} "
          f"{'KD_oop':>7s} {'KD_lam':>7s} {'Ref':>6s} {'Err%':>7s} {'Mode':>6s} {'':>6s}")
    print("  " + "-" * 82)

    results = []
    for dt, ref in REF_TENSION:
        theta = theta_from_dt(dt)
        theta_deg = np.degrees(theta)

        # Compute curvature for display
        amplitude = dt * LAM_THICKNESS
        wavelength = max(K_LAMBDA * amplitude, LAMBDA_MIN)
        kappa = (2 * np.pi / wavelength) ** 2 * amplitude

        # Full laminate knockdown with all three mechanisms
        lam = tension_knockdown_laminate(theta, mat, LAYUP, dt_ratio=dt)

        err = (lam["kd_lam"] - ref) / ref * 100
        passed = abs(err) <= TOLERANCE * 100
        status = "PASS" if passed else "FAIL"

        results.append({
            "dt": dt, "theta_deg": theta_deg, "kappa": kappa,
            "kd_fiber": lam["kd_fiber"],
            "kd_matrix": lam["kd_matrix"],
            "kd_oop": lam["kd_oop"],
            "kd_lam": lam["kd_lam"],
            "ref": ref, "err": err,
            "mode": lam["mode"], "passed": passed,
            "f_0": lam["f_0"],
            "oop_detail": lam.get("oop_detail", {}),
        })

        print(f"  {dt:>6.3f} {theta_deg:>5.1f}d {kappa:>7.4f} {lam['kd_fiber']:>7.3f} "
              f"{lam['kd_matrix']:>7.3f} {lam['kd_oop']:>7.3f} {lam['kd_lam']:>7.3f} "
              f"{ref:>6.2f} {err:>+6.1f}% {lam['mode']:>6s} {status:>6s}")

    # Summary
    n_pass = sum(1 for r in results if r["passed"])
    mae = np.mean([abs(r["err"]) for r in results])
    max_err = max(abs(r["err"]) for r in results)

    print()
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"  Pass: {n_pass}/{len(results)} (+/-{TOLERANCE*100:.0f}%)")
    print(f"  Mean abs error: {mae:.1f}%")
    print(f"  Max abs error:  {max_err:.1f}%")
    print(f"  f_0 (0° axial stiffness fraction): {results[0]['f_0']:.3f}")
    print()

    # Mode breakdown
    mode_counts = {}
    for r in results:
        mode_counts[r["mode"]] = mode_counts.get(r["mode"], 0) + 1
    print("  Controlling modes:")
    for m, c in sorted(mode_counts.items()):
        print(f"    {m}: {c} points")

    # Comparison: cos²θ only
    print()
    print("  --- Fiber tension only (cos²θ) ---")
    for r in results:
        err_fib = (r["kd_fiber"] - r["ref"]) / r["ref"] * 100
        print(f"    D/T={r['dt']:.3f}: cos²θ={r['kd_fiber']:.3f} vs ref={r['ref']:.2f} "
              f"err={err_fib:+.1f}%")

    # Interlaminar stress details
    print()
    S13_val = mat.S13 if hasattr(mat, "S13") and mat.S13 else 85.0
    print(f"  --- Interlaminar stresses at pristine failure (σ₁₁=Xt={mat.Xt:.0f} MPa) ---")
    print(f"      Yt = {mat.Yt:.0f} MPa (mode I),  S₁₃ = {S13_val:.0f} MPa (mode II)")
    print(f"      h_eff = {n_adj} × {PLY_THICKNESS:.3f} = {n_adj*PLY_THICKNESS:.3f} mm")
    print()
    print(f"  {'D/T':>6s} {'κ':>7s} {'σ₃₃':>7s} {'σ₃₃/Yt':>7s} {'τ₁₃':>7s} "
          f"{'τ₁₃/S₁₃':>8s} {'FI_σ₃₃':>7s} {'FI_τ₁₃':>7s} {'Controls':>10s}")
    print("  " + "-" * 78)
    for r in results:
        oop = r.get("oop_detail", {})
        s33 = oop.get("sigma33", 0.0)
        t13 = oop.get("tau13", 0.0)
        fi_s = oop.get("FI_s33", 0.0)
        fi_t = oop.get("FI_t13", 0.0)
        ctrl = oop.get("controlling", "none")
        print(f"  {r['dt']:>6.3f} {r['kappa']:>7.4f} {s33:>6.1f}  {s33/mat.Yt:>6.2f}  "
              f"{t13:>6.1f}  {t13/S13_val:>7.2f}  {fi_s:>7.3f} {fi_t:>7.3f} {ctrl:>10s}")

    # Generate plots
    plot_results(results, mat, script_dir)
    plot_combined(results, mat, script_dir)

    print()
    print("=" * 80)


def plot_results(results: list, mat, output_dir: Path) -> None:
    """Plot tension knockdown components with three-mechanism model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: knockdown vs D/T
    dt_vals = [r["dt"] for r in results]
    refs = [r["ref"] for r in results]
    kd_lam = [r["kd_lam"] for r in results]
    kd_fib = [r["kd_fiber"] for r in results]
    kd_mat = [r["kd_matrix"] for r in results]
    kd_oop = [r["kd_oop"] for r in results]

    ax1.scatter(dt_vals, refs, s=80, color="#1f77b4", marker="s", zorder=5,
                edgecolors="black", linewidths=0.5,
                label="Experimental (Elhajjar 2025)")
    ax1.plot(dt_vals, kd_lam, "o-", color="#ff7f0e", linewidth=2.5, markersize=7,
             zorder=4, label="Laminate knockdown (3-mechanism)")
    ax1.plot(dt_vals, kd_fib, "^--", color="#2ca02c", linewidth=1.5, markersize=6,
             alpha=0.6, label=r"Fiber tension ($\cos^2\theta$)")
    ax1.plot(dt_vals, kd_mat, "v--", color="#d62728", linewidth=1.5, markersize=6,
             alpha=0.6, label="Matrix tension (Hashin)")
    ax1.plot(dt_vals, kd_oop, "D--", color="#9467bd", linewidth=1.5, markersize=6,
             alpha=0.6, label=r"OOP $\sigma_{33}$ (curved beam)")

    ax1.set_xlabel("D/T Ratio", fontsize=12)
    ax1.set_ylabel("Normalized Tension Strength", fontsize=12)
    ax1.set_title("Tension Knockdown — Three Mechanisms", fontsize=13)
    ax1.set_xlim(0, 0.35)
    ax1.set_ylim(0, 1.15)
    ax1.legend(fontsize=8, loc="lower left")
    ax1.grid(True, alpha=0.3)

    mae = np.mean([abs(r["err"]) for r in results])
    n_pass = sum(1 for r in results if r["passed"])
    ax1.text(0.98, 0.98,
             f"MAE: {mae:.1f}%\nPass: {n_pass}/{len(results)}\n"
             f"OOP = curved-beam $\\sigma_{{33}}$\n"
             f"$h_{{eff}}$ = {_max_consecutive_zero_plies(LAYUP)}$\\times${PLY_THICKNESS:.3f} mm",
             transform=ax1.transAxes, fontsize=8, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="0.7"))

    # Right: knockdown vs D/T showing mechanism dominance regions
    dt_smooth = np.linspace(0.001, 0.35, 300)

    kd_f_s = []
    kd_m_s = []
    kd_o_s = []
    kd_lam_s = []
    for d in dt_smooth:
        t = theta_from_dt(d)
        lam = tension_knockdown_laminate(t, mat, LAYUP, dt_ratio=d)
        kd_f_s.append(lam["kd_fiber"])
        kd_m_s.append(lam["kd_matrix"])
        kd_o_s.append(lam["kd_oop"])
        kd_lam_s.append(lam["kd_lam"])

    ax2.plot(dt_smooth, kd_f_s, "-", color="#2ca02c", linewidth=1.5, alpha=0.5,
             label=r"Fiber ($\cos^2\theta$)")
    ax2.plot(dt_smooth, kd_m_s, "-", color="#d62728", linewidth=1.5, alpha=0.5,
             label="Matrix (Hashin)")
    ax2.plot(dt_smooth, kd_o_s, "-", color="#9467bd", linewidth=1.5, alpha=0.5,
             label=r"OOP ($\sigma_{33}$)")
    ax2.plot(dt_smooth, kd_lam_s, "-", color="#ff7f0e", linewidth=2.5,
             label="Laminate (CLT-weighted)")

    ax2.scatter(dt_vals, refs, s=70, color="#1f77b4", marker="s", zorder=5,
                edgecolors="black", linewidths=0.5,
                label="Expt. (Elhajjar 2025)")

    ax2.set_xlabel("D/T Ratio", fontsize=12)
    ax2.set_ylabel("Knockdown Factor (0° ply)", fontsize=12)
    ax2.set_title("Mechanism Dominance Regions", fontsize=13)
    ax2.set_xlim(0, 0.35)
    ax2.set_ylim(0, 1.15)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Annotate dominance regions
    ax2.annotate("OOP\ndominates", xy=(0.05, 0.75), fontsize=8, color="#9467bd",
                 ha="center", style="italic")
    ax2.annotate("Matrix\ndominates", xy=(0.25, 0.3), fontsize=8, color="#d62728",
                 ha="center", style="italic")

    fig.tight_layout()
    out = output_dir / "fig_tension_analytical_model.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_combined(results: list, mat, output_dir: Path) -> None:
    """Combined tension + compression comparison plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Experimental data
    t_dt = [r["dt"] for r in results]
    t_ref = [r["ref"] for r in results]
    ax.scatter(t_dt, t_ref, s=70, color="#1f77b4", marker="s", zorder=5,
               edgecolors="black", linewidths=0.5,
               label="Expt. Tension (Elhajjar 2025)")

    c_dt = [d[0] for d in REF_COMPRESSION]
    c_ref = [d[1] for d in REF_COMPRESSION]
    ax.scatter(c_dt, c_ref, s=70, color="#d62728", marker="o", zorder=5,
               edgecolors="black", linewidths=0.5,
               label="Expt. Compression (Elhajjar 2025)")

    # Smooth curves
    dt_smooth = np.linspace(0.001, 0.35, 300)

    # Tension: three-mechanism analytical model
    kd_t = [tension_knockdown_laminate(theta_from_dt(d), mat, LAYUP, dt_ratio=d)["kd_lam"]
            for d in dt_smooth]
    ax.plot(dt_smooth, kd_t, "-", color="#1f77b4", linewidth=2.5,
            label="Tension — 3-Mechanism (LaRC04 + $\\sigma_{33}$)")

    # Compression: Budiansky-Fleck
    gamma_Y = mat.gamma_Y
    kd_c = [1.0 / (1.0 + theta_from_dt(d) / gamma_Y) for d in dt_smooth]
    ax.plot(dt_smooth, kd_c, "-", color="#d62728", linewidth=2.5,
            label=r"Compression — Budiansky-Fleck ($\gamma_Y$=0.162)")

    ax.set_xlabel("D/T Ratio (Defect Severity)", fontsize=12)
    ax.set_ylabel("Normalized Strength", fontsize=12)
    ax.set_title("Tension and Compression Knockdown vs. Fiber Waviness\n"
                 "Physics-Based Models vs. Elhajjar (2025) Sci. Rep. 15:25977",
                 fontsize=13)
    ax.set_xlim(0, 0.35)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax.text(0.02, 0.05,
            "Compression: Budiansky-Fleck kink-band instability\n"
            "Tension: $\\cos^2\\theta$ (fiber) + Hashin (matrix) + $\\sigma_{33}$ (OOP)\n"
            "All physics-based — no empirical curve fitting",
            transform=ax.transAxes, fontsize=8, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="0.7"))

    fig.tight_layout()
    out = output_dir / "fig_tension_compression_physics.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
