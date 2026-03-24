#!/usr/bin/env python3
"""Validation of WrinkleFE against Jin et al. (2026).

Thin-Walled Structures 219:114237
"Interlaminar damage analysis of dual-wrinkled composite laminates
under multidirectional static loading"

Usage:
    python validation/validate_jin2026.py

Outputs:
    Console pass/fail tables
    validation/fig_damage_vs_amplitude.png
    validation/fig_damage_vs_angle.png
    validation/fig_compression_predictions.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure wrinklefe is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis

# ======================================================================
# Constants
# ======================================================================
PLY_THICKNESS = 0.183  # mm
WAVELENGTH = 16.0      # mm
WIDTH = 12.0           # mm
TOLERANCE = 0.30       # +/-30% pass criterion
NX, NY, NZ = 20, 6, 3

# Paper layup: [45/0/-45/0/45/0/-45/0/45/0/-45/0]_S
LAYUP = [45, 0, -45, 0, 45, 0, -45, 0, 45, 0, -45, 0,
         -45, 0, 45, 0, -45, 0, 45, 0, -45, 0, 45, 0]


def wavelength_from_angle(angle_deg: float, amplitude: float) -> float:
    """Back-calculate wavelength from target wrinkle angle at fixed amplitude.

    theta ~ arctan(2*pi*A/lambda)  =>  lambda = 2*pi*A / tan(theta)
    """
    theta_rad = np.radians(angle_deg)
    return 2.0 * np.pi * amplitude / np.tan(theta_rad)


def run_case(
    morphology: str,
    amplitude: float,
    loading: str = "tension",
) -> dict:
    """Run a single WrinkleFE analysis and return key results."""
    strain = 0.01 if loading == "tension" else -0.01
    domain_length = 3.0 * WAVELENGTH

    config = AnalysisConfig(
        amplitude=amplitude,
        wavelength=WAVELENGTH,
        width=WIDTH,
        morphology=morphology,
        loading=loading,
        angles=LAYUP,
        ply_thickness=PLY_THICKNESS,
        nx=NX,
        ny=NY,
        nz_per_ply=NZ,
        domain_length=domain_length,
        domain_width=30.0,
        applied_strain=strain,
        solver="direct",
        verbose=False,
    )

    t0 = time.monotonic()
    analysis = WrinkleAnalysis(config)
    result = analysis.run()
    elapsed = time.monotonic() - t0

    return {
        "damage_index": result.damage_index,
        "analytical_knockdown": result.analytical_knockdown,
        "analytical_strength_MPa": result.analytical_strength_MPa,
        "morphology_factor": result.morphology_factor,
        "max_angle_deg": np.degrees(result.max_angle_rad),
        "effective_angle_deg": np.degrees(result.effective_angle_rad),
        "retention_factors": result.retention_factors,
        "elapsed_s": elapsed,
    }


# ======================================================================
# Group runners
# ======================================================================

def run_group_a(ref_data: dict) -> list[dict]:
    """Group A: Tensile damage vs amplitude at 20 deg angle."""
    group = ref_data["group_a"]
    results = []

    for case in group["cases"]:
        amp_factor = case["amplitude_factor"]
        amplitude = amp_factor * PLY_THICKNESS

        print(f"  Running {case['id']}: {case['morphology']} {amp_factor}A tension...",
              end=" ", flush=True)
        r = run_case(case["morphology"], amplitude, "tension")

        predicted = r["damage_index"]
        ref = case["ref_damage"]
        err_pct = (predicted - ref) / ref * 100 if ref > 0 else 0
        passed = abs(predicted - ref) / max(ref, 1e-6) <= TOLERANCE

        results.append({
            **case,
            "predicted": predicted,
            "error_pct": err_pct,
            "passed": passed,
            **r,
        })
        status = "PASS" if passed else "FAIL"
        print(f"D={predicted:.3f} (ref={ref:.2f}, err={err_pct:+.0f}%) {status}")

    return results


def run_group_b(ref_data: dict) -> list[dict]:
    """Group B: Tensile damage vs wrinkle angle at 2A.

    Holds amplitude fixed at 2A (0.366mm) and varies wavelength
    to achieve the target wrinkle angle, matching the paper's approach.
    """
    group = ref_data["group_b"]
    amp_factor = group["amplitude_factor"]
    amplitude = amp_factor * PLY_THICKNESS  # 2A = 0.366mm
    results = []

    for case in group["cases"]:
        angle_deg = case["angle_deg"]
        wl = wavelength_from_angle(angle_deg, amplitude)
        width = 0.75 * wl  # scale Gaussian width with wavelength

        print(f"  Running {case['id']}: {case['morphology']} {angle_deg}deg "
              f"(lam={wl:.1f}mm) tension...", end=" ", flush=True)

        # Override wavelength/width for this case
        strain = 0.01
        domain_length = 3.0 * wl
        config = AnalysisConfig(
            amplitude=amplitude,
            wavelength=wl,
            width=width,
            morphology=case["morphology"],
            loading="tension",
            angles=LAYUP,
            ply_thickness=PLY_THICKNESS,
            nx=NX,
            ny=NY,
            nz_per_ply=NZ,
            domain_length=domain_length,
            domain_width=30.0,
            applied_strain=strain,
            solver="direct",
            verbose=False,
        )

        t0 = time.monotonic()
        analysis = WrinkleAnalysis(config)
        result = analysis.run()
        elapsed = time.monotonic() - t0

        r = {
            "damage_index": result.damage_index,
            "analytical_knockdown": result.analytical_knockdown,
            "max_angle_deg": np.degrees(result.max_angle_rad),
            "elapsed_s": elapsed,
        }

        predicted = r["damage_index"]
        ref = case["ref_damage"]
        err_pct = (predicted - ref) / ref * 100 if ref > 0 else 0
        passed = abs(predicted - ref) / max(ref, 1e-6) <= TOLERANCE

        results.append({
            **case,
            "amplitude_mm": amplitude,
            "wavelength_mm": wl,
            "predicted": predicted,
            "error_pct": err_pct,
            "passed": passed,
            **r,
        })
        status = "PASS" if passed else "FAIL"
        print(f"D={predicted:.3f} (ref={ref:.2f}, err={err_pct:+.0f}%) {status}")

    return results


def run_group_c(results_a: list, results_b: list) -> list[dict]:
    """Group C: Verify morphology ranking from Groups A and B."""
    rankings = []

    # Check ranking for each amplitude in Group A
    for amp_factor in [1, 2, 3]:
        cases = [r for r in results_a if r["amplitude_factor"] == amp_factor]
        by_morph = {r["morphology"]: r["predicted"] for r in cases}
        # Expected: stack > convex ~ concave
        stack_worst = by_morph.get("stack", 0) >= by_morph.get("convex", 0)
        stack_worst &= by_morph.get("stack", 0) >= by_morph.get("concave", 0)
        order = sorted(by_morph.items(), key=lambda x: -x[1])
        order_str = " > ".join(f"{m}({v:.3f})" for m, v in order)
        rankings.append({
            "group": f"A ({amp_factor}A)",
            "order": order_str,
            "stack_worst": stack_worst,
            "passed": stack_worst,
        })

    # Check ranking for each angle in Group B
    for angle in [10, 15, 20]:
        cases = [r for r in results_b if r["angle_deg"] == angle]
        by_morph = {r["morphology"]: r["predicted"] for r in cases}
        stack_worst = by_morph.get("stack", 0) >= by_morph.get("convex", 0)
        stack_worst &= by_morph.get("stack", 0) >= by_morph.get("concave", 0)
        order = sorted(by_morph.items(), key=lambda x: -x[1])
        order_str = " > ".join(f"{m}({v:.3f})" for m, v in order)
        rankings.append({
            "group": f"B ({angle}deg)",
            "order": order_str,
            "stack_worst": stack_worst,
            "passed": stack_worst,
        })

    return rankings


def run_group_d(ref_data: dict) -> tuple[list[dict], list[dict]]:
    """Group D: Compression predictions (no reference data)."""
    results = []

    for amp_factor in [1, 2, 3]:
        amplitude = amp_factor * PLY_THICKNESS
        for morphology in ["stack", "convex", "concave"]:
            case_id = f"D-{morphology}-{amp_factor}A"
            print(f"  Running {case_id}: compression...", end=" ", flush=True)
            r = run_case(morphology, amplitude, "compression")
            results.append({
                "id": case_id,
                "morphology": morphology,
                "amplitude_factor": amp_factor,
                "damage": r["damage_index"],
                "knockdown": r["analytical_knockdown"],
                "strength_MPa": r["analytical_strength_MPa"],
                "retention": r.get("retention_factors", {}),
                **r,
            })
            print(f"KD={r['analytical_knockdown']:.3f} D={r['damage_index']:.3f}")

    # Check ranking: concave worst, convex best
    ranking_results = []
    for amp_factor in [1, 2, 3]:
        cases = [r for r in results if r["amplitude_factor"] == amp_factor]
        by_morph = {r["morphology"]: r["knockdown"] for r in cases}
        # Expected: convex > stack > concave (higher KD = stronger)
        correct = (by_morph.get("convex", 0) >= by_morph.get("stack", 0)
                   >= by_morph.get("concave", 0))
        ranking_results.append({
            "amplitude_factor": amp_factor,
            "ranking_correct": correct,
            "by_morph": by_morph,
        })

    return results, ranking_results


# ======================================================================
# Plotting
# ======================================================================

def plot_damage_vs_amplitude(results_a: list, output_dir: Path) -> None:
    """Group A plot: damage vs amplitude for each morphology."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"stack": "#1f77b4", "convex": "#2ca02c", "concave": "#d62728"}
    markers_pred = {"stack": "o", "convex": "s", "concave": "^"}

    for morph in ["stack", "convex", "concave"]:
        cases = [r for r in results_a if r["morphology"] == morph]
        amps = [c["amplitude_factor"] for c in cases]
        predicted = [c["predicted"] for c in cases]
        refs = [c["ref_damage"] for c in cases]

        ax.plot(amps, predicted, marker=markers_pred[morph], color=colors[morph],
                linewidth=1.5, label=f"{morph} (WrinkleFE)")
        ax.plot(amps, refs, marker="x", color=colors[morph],
                linewidth=1.5, linestyle="--", label=f"{morph} (Jin et al.)")

    ax.set_xlabel("Amplitude (x ply thickness)")
    ax.set_ylabel("Peak Interlaminar Damage Index")
    ax.set_title("Tensile Damage vs. Amplitude - WrinkleFE vs. Jin et al. (2026)")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["1A", "2A", "3A"])
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "fig_damage_vs_amplitude.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'fig_damage_vs_amplitude.png'}")


def plot_damage_vs_angle(results_b: list, output_dir: Path) -> None:
    """Group B plot: damage vs angle for each morphology."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"stack": "#1f77b4", "convex": "#2ca02c", "concave": "#d62728"}

    for morph in ["stack", "convex", "concave"]:
        cases = sorted(
            [r for r in results_b if r["morphology"] == morph],
            key=lambda x: x["angle_deg"],
        )
        angles = [c["angle_deg"] for c in cases]
        predicted = [c["predicted"] for c in cases]
        refs = [c["ref_damage"] for c in cases]

        ax.plot(angles, predicted, marker="o", color=colors[morph],
                linewidth=1.5, label=f"{morph} (WrinkleFE)")
        ax.plot(angles, refs, marker="x", color=colors[morph],
                linewidth=1.5, linestyle="--", label=f"{morph} (Jin et al.)")

    ax.set_xlabel("Wrinkle Angle (degrees)")
    ax.set_ylabel("Peak Interlaminar Damage Index")
    ax.set_title("Tensile Damage vs. Wrinkle Angle - WrinkleFE vs. Jin et al. (2026)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "fig_damage_vs_angle.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'fig_damage_vs_angle.png'}")


def plot_compression_predictions(results_d: list, output_dir: Path) -> None:
    """Group D plot: compression knockdown predictions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"stack": "#1f77b4", "convex": "#2ca02c", "concave": "#d62728"}

    for morph in ["stack", "convex", "concave"]:
        cases = sorted(
            [r for r in results_d if r["morphology"] == morph],
            key=lambda x: x["amplitude_factor"],
        )
        amps = [c["amplitude_factor"] for c in cases]
        kd = [c["knockdown"] for c in cases]
        dmg = [c["damage"] for c in cases]

        ax1.plot(amps, kd, marker="o", color=colors[morph],
                 linewidth=1.5, label=morph)
        ax2.plot(amps, dmg, marker="s", color=colors[morph],
                 linewidth=1.5, label=morph)

    ax1.set_xlabel("Amplitude (x ply thickness)")
    ax1.set_ylabel("Knockdown Factor")
    ax1.set_title("Compression Knockdown (PREDICTIONS)")
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(["1A", "2A", "3A"])
    ax1.legend()
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Amplitude (x ply thickness)")
    ax2.set_ylabel("Damage Index")
    ax2.set_title("Compression Damage (PREDICTIONS)")
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(["1A", "2A", "3A"])
    ax2.legend()
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "WrinkleFE Compression Predictions - No Reference Data (Gap Analysis)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "fig_compression_predictions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'fig_compression_predictions.png'}")


# ======================================================================
# Summary
# ======================================================================

def print_summary(results_a, results_b, rankings_c, results_d, rankings_d):
    """Print final validation summary."""
    n_pass_a = sum(1 for r in results_a if r["passed"])
    n_pass_b = sum(1 for r in results_b if r["passed"])
    n_pass_c = sum(1 for r in rankings_c if r["passed"])
    n_pass_d = sum(1 for r in rankings_d if r["ranking_correct"])

    n_total_a = len(results_a)
    n_total_b = len(results_b)
    n_total_c = len(rankings_c)
    n_total_d = len(rankings_d)

    total_pass = n_pass_a + n_pass_b + n_pass_c + n_pass_d
    total_cases = n_total_a + n_total_b + n_total_c + n_total_d
    n_predictions = len(results_d)

    print()
    print("=" * 60)
    print("  VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Group A (damage vs amplitude):  {n_pass_a}/{n_total_a} PASS")
    print(f"  Group B (damage vs angle):      {n_pass_b}/{n_total_b} PASS")
    print(f"  Group C (morphology ranking):   {n_pass_c}/{n_total_c} PASS")
    print(f"  Group D (compression ranking):  {n_pass_d}/{n_total_d} PASS")
    print(f"  Compression predictions:        {n_predictions} (no reference)")
    print()
    print(f"  TOTAL: {total_pass}/{total_cases} PASS, "
          f"{total_cases - total_pass} FAIL, "
          f"{n_predictions} PREDICTIONS")
    print("=" * 60)


# ======================================================================
# Main
# ======================================================================

def main():
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir

    # Load reference data
    with open(script_dir / "reference_data.json") as f:
        ref_data = json.load(f)

    print("=" * 60)
    print("  WrinkleFE Validation - Jin et al. (2026)")
    print("  Thin-Walled Structures 219:114237")
    print("=" * 60)
    print()

    # Group A
    print("Group A: Tensile Damage vs. Amplitude (Fig. 18)")
    t0 = time.monotonic()
    results_a = run_group_a(ref_data)
    print(f"  Group A complete in {time.monotonic() - t0:.0f}s")
    print()

    # Group B
    print("Group B: Tensile Damage vs. Wrinkle Angle (Fig. 19)")
    t0 = time.monotonic()
    results_b = run_group_b(ref_data)
    print(f"  Group B complete in {time.monotonic() - t0:.0f}s")
    print()

    # Group C
    print("Group C: Morphology Ranking Verification")
    rankings_c = run_group_c(results_a, results_b)
    for r in rankings_c:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {r['group']:12s}  {r['order']}  {status}")
    print()

    # Group D
    print("Group D: Compression Predictions (no reference data)")
    results_d, rankings_d = run_group_d(ref_data)
    print()
    for r in rankings_d:
        status = "PASS" if r["ranking_correct"] else "FAIL"
        by = r["by_morph"]
        print(f"  {r['amplitude_factor']}A: convex={by.get('convex', 0):.3f} "
              f"stack={by.get('stack', 0):.3f} "
              f"concave={by.get('concave', 0):.3f}  ranking {status}")
    print()

    # Plots
    print("Generating plots...")
    plot_damage_vs_amplitude(results_a, output_dir)
    plot_damage_vs_angle(results_b, output_dir)
    plot_compression_predictions(results_d, output_dir)

    # Summary
    print_summary(results_a, results_b, rankings_c, results_d, rankings_d)


if __name__ == "__main__":
    main()
