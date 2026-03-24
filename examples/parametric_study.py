#!/usr/bin/env python3
"""Parametric study example for WrinkleFE.

Demonstrates:
1. Sweeping amplitude from 1A to 3A (0.183 mm to 0.549 mm).
2. Running analytical predictions for each morphology at each amplitude.
3. Printing a results table.
4. Plotting strength vs. amplitude for all three morphologies.

Usage
-----
::

    python examples/parametric_study.py
"""

from __future__ import annotations

import numpy as np

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis


def main() -> None:
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    ply_thickness = 0.183  # mm
    amplitudes = np.linspace(1.0 * ply_thickness, 3.0 * ply_thickness, 10)
    morphologies = ("stack", "convex", "concave")

    # Storage for results
    results_table: dict[str, list[float]] = {m: [] for m in morphologies}
    knockdowns_table: dict[str, list[float]] = {m: [] for m in morphologies}

    # ------------------------------------------------------------------
    # Run sweeps
    # ------------------------------------------------------------------
    print("=" * 70)
    print("  WrinkleFE Parametric Study: Amplitude Sweep")
    print("=" * 70)
    print()

    for morph in morphologies:
        base_config = AnalysisConfig(
            wavelength=16.0,
            width=12.0,
            morphology=morph,
            loading="compression",
        )

        sweep_results = WrinkleAnalysis.parametric_sweep(
            base_config,
            parameter="amplitude",
            values=amplitudes,
            analytical_only=True,
        )

        for r in sweep_results:
            results_table[morph].append(r.analytical_strength_MPa)
            knockdowns_table[morph].append(r.analytical_knockdown)

    # ------------------------------------------------------------------
    # Print results table
    # ------------------------------------------------------------------
    print(f"  {'Amplitude':>10} {'A/t':>6}", end="")
    for morph in morphologies:
        print(f" {morph + ' (MPa)':>16}", end="")
    print()
    print("-" * 70)

    for i, A in enumerate(amplitudes):
        print(f"  {A:>10.3f} {A / ply_thickness:>6.1f}", end="")
        for morph in morphologies:
            print(f" {results_table[morph][i]:>16.1f}", end="")
        print()

    print()
    print("  Knockdown factors:")
    print(f"  {'Amplitude':>10} {'A/t':>6}", end="")
    for morph in morphologies:
        print(f" {morph:>16}", end="")
    print()
    print("-" * 70)

    for i, A in enumerate(amplitudes):
        print(f"  {A:>10.3f} {A / ply_thickness:>6.1f}", end="")
        for morph in morphologies:
            print(f" {knockdowns_table[morph][i]:>16.4f}", end="")
        print()

    print()

    # ------------------------------------------------------------------
    # Plot (optional -- requires matplotlib)
    # ------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        amp_over_t = amplitudes / ply_thickness

        colors = {"stack": "#2196F3", "convex": "#4CAF50", "concave": "#F44336"}
        markers = {"stack": "s", "convex": "^", "concave": "v"}

        for morph in morphologies:
            ax1.plot(
                amp_over_t, results_table[morph],
                marker=markers[morph], color=colors[morph],
                label=morph.capitalize(), linewidth=2, markersize=6,
            )
            ax2.plot(
                amp_over_t, knockdowns_table[morph],
                marker=markers[morph], color=colors[morph],
                label=morph.capitalize(), linewidth=2, markersize=6,
            )

        ax1.set_xlabel("Amplitude / Ply Thickness (A/t)")
        ax1.set_ylabel("Predicted Strength (MPa)")
        ax1.set_title("Compression Strength vs. Wrinkle Amplitude")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("Amplitude / Ply Thickness (A/t)")
        ax2.set_ylabel("Knockdown Factor")
        ax2.set_title("Knockdown Factor vs. Wrinkle Amplitude")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig("parametric_amplitude_sweep.png", dpi=150, bbox_inches="tight")
        print("  Plot saved to: parametric_amplitude_sweep.png")

        plt.show()

    except ImportError:
        print("  matplotlib not available -- skipping plot generation.")
        print("  Install with: pip install matplotlib")

    print()


if __name__ == "__main__":
    main()
