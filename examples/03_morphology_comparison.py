"""Morphology comparison: the 5 named morphologies on one laminate.

Runs the same wrinkle geometry through every named morphology
(``stack``, ``convex``, ``concave``, ``uniform``, ``graded``) using
``WrinkleAnalysis.compare_morphologies`` and tabulates the analytical
predictions side by side.

Expected runtime: ~1 s (analytical-only).
Expected output:  a 5-row table; ``concave`` is the weakest dual-wrinkle
                  morphology (M_f < 1), ``convex`` the strongest.
"""

import numpy as np

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis

base = AnalysisConfig(
    amplitude=0.366, wavelength=16.0, width=12.0,
    morphology="stack", loading="compression",
)

morphologies = ("stack", "convex", "concave", "uniform", "graded")
results = WrinkleAnalysis.compare_morphologies(
    base, morphologies=morphologies, analytical_only=True
)

print(
    f"{'morphology':<12} {'M_f':>8} {'theta_max (deg)':>16} "
    f"{'knockdown':>10} {'strength (MPa)':>15}"
)
for name in morphologies:
    r = results[name]
    print(
        f"{name:<12} {r.morphology_factor:8.4f} "
        f"{np.degrees(r.max_angle_rad):16.2f} "
        f"{r.analytical_knockdown:10.4f} "
        f"{r.analytical_strength_MPa:15.1f}"
    )
