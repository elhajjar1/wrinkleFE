#!/usr/bin/env python3
"""Monte Carlo simulation and Jensen gap analysis example for WrinkleFE.

Demonstrates:
1. Running a Monte Carlo simulation to generate strength distributions.
2. Computing the Jensen gap (overestimate from using mean defect parameters).
3. Printing detailed statistics and per-morphology breakdowns.
4. Plotting strength distributions and Jensen gap visualisation.

Usage
-----
::

    python examples/montecarlo_analysis.py
"""

from __future__ import annotations

import numpy as np

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Configure analysis with Monte Carlo enabled
    # ------------------------------------------------------------------
    print("=" * 70)
    print("  WrinkleFE Monte Carlo + Jensen Gap Analysis")
    print("=" * 70)
    print()

    config = AnalysisConfig(
        amplitude=0.366,        # 2A = 2 ply thicknesses
        wavelength=16.0,        # mm
        width=12.0,             # mm
        morphology="stack",     # baseline morphology
        loading="compression",
        run_montecarlo=True,
        mc_samples=5000,
        mc_seed=42,             # reproducible results
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 2. Run analytical-only analysis (includes MC + Jensen gap)
    # ------------------------------------------------------------------
    print("  Running Monte Carlo simulation with 5000 samples...")
    print()

    analysis = WrinkleAnalysis(config)
    result = analysis.run_analytical_only()

    # ------------------------------------------------------------------
    # 3. Print results summary
    # ------------------------------------------------------------------
    print(result.summary())
    print()

    # ------------------------------------------------------------------
    # 4. Detailed Monte Carlo statistics
    # ------------------------------------------------------------------
    if result.mc_results is not None:
        mc = result.mc_results

        print("=" * 70)
        print("  Detailed Monte Carlo Statistics")
        print("=" * 70)
        print()
        print(f"  Number of samples:    {mc.n_samples}")
        print(f"  Mean strength:        {mc.mean_strength:.1f} MPa")
        print(f"  Std deviation:        {mc.std_strength:.1f} MPa")
        print(f"  CoV:                  {mc.cov_strength:.4f}")
        print(f"  Minimum strength:     {mc.min_strength:.1f} MPa")
        print(f"  5th percentile:       {mc.percentile_5:.1f} MPa")
        print(f"  1st percentile:       {mc.percentile_1:.1f} MPa")
        print()

        # Per-morphology breakdown (if available)
        if hasattr(mc, "morphologies") and mc.morphologies is not None:
            print("  Per-morphology breakdown:")
            print(f"  {'Morphology':<12} {'Count':>8} {'Mean (MPa)':>12} "
                  f"{'Std (MPa)':>12} {'Min (MPa)':>12}")
            print("-" * 70)

            for morph in ("stack", "convex", "concave"):
                mask = mc.morphologies == morph
                if np.any(mask):
                    s = mc.strengths[mask]
                    print(
                        f"  {morph:<12} {np.sum(mask):>8d} {s.mean():>12.1f} "
                        f"{s.std():>12.1f} {s.min():>12.1f}"
                    )
            print()

    # ------------------------------------------------------------------
    # 5. Jensen gap analysis
    # ------------------------------------------------------------------
    if result.jensen_gap is not None:
        jg = result.jensen_gap

        print("=" * 70)
        print("  Jensen Gap Analysis")
        print("=" * 70)
        print()
        print(f"  Strength at mean defect:   {jg.strength_at_mean:.1f} MPa")
        print(f"  Mean of strengths:         {jg.mean_of_strengths:.1f} MPa")
        print(f"  Jensen gap:                {jg.jensen_gap:.1f} MPa "
              f"({jg.jensen_gap_percent:.1f}%)")
        print()
        print("  Interpretation:")
        print("  The Jensen gap quantifies how much mean-based design approaches")
        print("  overestimate strength due to the concave (sublinear) relationship")
        print("  between defect severity and compression strength.")
        print(f"  Using mean defect parameters overestimates strength by "
              f"~{jg.jensen_gap_percent:.1f}%.")
        print()

        # Per-morphology Jensen gap (if available)
        if jg.gap_by_morphology:
            print("  Per-morphology Jensen gaps:")
            for morph, gap in sorted(jg.gap_by_morphology.items()):
                print(f"    {morph:<12} {gap:.1f} MPa")
            print()

    # ------------------------------------------------------------------
    # 6. Plot (optional -- requires matplotlib)
    # ------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt

        if result.mc_results is not None:
            mc = result.mc_results

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Strength distribution histogram
            ax1 = axes[0]
            ax1.hist(
                mc.strengths, bins=60, density=True,
                color="#2196F3", alpha=0.7, edgecolor="white", linewidth=0.3,
            )
            ax1.axvline(
                mc.mean_strength, color="k", linestyle="--", linewidth=1.5,
                label=f"Mean = {mc.mean_strength:.0f} MPa",
            )
            ax1.axvline(
                mc.percentile_5, color="red", linestyle=":", linewidth=1.5,
                label=f"5th pctl = {mc.percentile_5:.0f} MPa",
            )
            ax1.set_xlabel("Predicted Strength (MPa)")
            ax1.set_ylabel("Probability Density")
            ax1.set_title("Monte Carlo Strength Distribution")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Jensen gap visualisation
            if result.jensen_gap is not None:
                jg = result.jensen_gap
                ax2 = axes[1]
                labels = ["Strength at\nmean defect", "Mean of\nstrengths"]
                values = [jg.strength_at_mean, jg.mean_of_strengths]
                colors = ["#4CAF50", "#F44336"]
                bars = ax2.bar(labels, values, color=colors, edgecolor="white",
                               width=0.5)

                # Annotate gap
                gap_y = max(values) * 1.02
                ax2.annotate(
                    f"Jensen Gap = {jg.jensen_gap:.1f} MPa\n"
                    f"({jg.jensen_gap_percent:.1f}% overestimate)",
                    xy=(0.5, gap_y), fontsize=10, ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow",
                              alpha=0.8),
                )

                ax2.set_ylabel("Strength (MPa)")
                ax2.set_title("Jensen Gap: Mean-Based Overestimate")
                ax2.grid(True, alpha=0.3, axis="y")

            fig.tight_layout()
            fig.savefig(
                "montecarlo_jensen_gap.png", dpi=150, bbox_inches="tight"
            )
            print("  Plot saved to: montecarlo_jensen_gap.png")
            plt.show()

    except ImportError:
        print("  matplotlib not available -- skipping plot generation.")
        print("  Install with: pip install matplotlib")

    print()


if __name__ == "__main__":
    main()
