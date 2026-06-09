"""Parametric sweep: knockdown curve vs wrinkle amplitude.

Sweeps the wrinkle half-amplitude A while holding wavelength fixed and
plots the analytical knockdown curve. ``WrinkleAnalysis.parametric_sweep``
can sweep any numeric ``AnalysisConfig`` field the same way (e.g.
``'wavelength'``, ``'width'``, ``'applied_strain'``).

Expected runtime: ~1 s (analytical-only, 9 points).
Expected output:  printed table of amplitude vs knockdown (monotonically
                  decreasing from 1.0) and ``02_knockdown_curve.png``.
"""

import matplotlib

matplotlib.use("Agg")  # headless-safe; remove to use an interactive backend
import matplotlib.pyplot as plt
import numpy as np

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis

base = AnalysisConfig(
    amplitude=0.366, wavelength=16.0, width=12.0,
    morphology="stack", loading="compression",
)

amplitudes = np.linspace(0.0, 0.8, 9)
results = WrinkleAnalysis.parametric_sweep(
    base, "amplitude", amplitudes, analytical_only=True
)

print(f"{'A (mm)':>8} {'knockdown':>10} {'strength (MPa)':>15}")
for A, r in zip(amplitudes, results):
    print(
        f"{A:8.3f} {r.analytical_knockdown:10.4f} "
        f"{r.analytical_strength_MPa:15.1f}"
    )

knockdowns = [r.analytical_knockdown for r in results]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(amplitudes, knockdowns, "o-")
ax.set_xlabel("Half-amplitude A (mm)")
ax.set_ylabel("Knockdown factor")
ax.set_title(f"Knockdown vs amplitude ($\\lambda$={base.wavelength} mm)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("02_knockdown_curve.png", dpi=150)
print("Saved: 02_knockdown_curve.png")
