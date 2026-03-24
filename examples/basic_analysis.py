#!/usr/bin/env python3
"""Basic WrinkleFE usage example.

Demonstrates:
1. Creating an AnalysisConfig with the default CYCOM X850/T800 material.
2. Running an analytical-only analysis (no FE mesh or solve).
3. Printing the results summary.
4. Comparing all three morphologies (stack, convex, concave).

Usage
-----
::

    python examples/basic_analysis.py
"""

from __future__ import annotations

import numpy as np

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Single analysis with default material
    # ------------------------------------------------------------------
    print("=" * 65)
    print("  Example 1: Single Analysis (concave, 2A amplitude)")
    print("=" * 65)

    config = AnalysisConfig(
        amplitude=0.366,       # 2A = 2 ply thicknesses
        wavelength=16.0,       # mm
        width=12.0,            # mm
        morphology="concave",  # worst case for compression
        loading="compression",
    )

    analysis = WrinkleAnalysis(config)
    result = analysis.run_analytical_only()

    print(result.summary())
    print()

    # ------------------------------------------------------------------
    # 2. Compare all three morphologies
    # ------------------------------------------------------------------
    print("=" * 65)
    print("  Example 2: Morphology Comparison")
    print("=" * 65)
    print()

    base_config = AnalysisConfig(
        amplitude=0.366,
        wavelength=16.0,
        width=12.0,
        loading="compression",
    )

    all_results = WrinkleAnalysis.compare_morphologies(
        base_config,
        morphologies=("stack", "convex", "concave"),
        analytical_only=True,
    )

    # Print comparison table
    header = (
        f"  {'Morphology':<12} {'M_f':>8} {'theta_max':>10} "
        f"{'theta_eff':>10} {'Damage':>8} {'Strength':>12}"
    )
    units = (
        f"  {'':.<12} {'':>8} {'(deg)':>10} "
        f"{'(deg)':>10} {'D':>8} {'(MPa)':>12}"
    )
    print(header)
    print(units)
    print("-" * 65)

    for morph in ("stack", "convex", "concave"):
        r = all_results[morph]
        print(
            f"  {morph:<12} {r.morphology_factor:>8.4f} "
            f"{np.degrees(r.max_angle_rad):>10.2f} "
            f"{np.degrees(r.effective_angle_rad):>10.2f} "
            f"{r.damage_index:>8.4f} "
            f"{r.analytical_strength_MPa:>12.1f}"
        )

    print()

    # Rank by strength
    ranked = sorted(
        all_results.items(),
        key=lambda x: x[1].analytical_strength_MPa,
        reverse=True,
    )
    print("  Ranking (strongest to weakest):")
    for i, (morph, r) in enumerate(ranked, 1):
        print(f"    {i}. {morph:<10} {r.analytical_strength_MPa:.1f} MPa")

    print()
    print("  Key insight: convex morphology provides the highest compression")
    print("  strength because the outward bulge stabilises against kink-band")
    print("  formation, while concave (inward pinch) amplifies it.")
    print()


if __name__ == "__main__":
    main()
