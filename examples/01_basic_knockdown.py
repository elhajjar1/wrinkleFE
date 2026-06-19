"""Basic knockdown analysis: pristine vs wrinkled strength.

Runs the README quick-start case through both pathways — the analytical
Budiansky-Fleck knockdown and the 3-D FE solve with LaRC05 failure
evaluation — and saves a knockdown summary plus a wrinkle-profile figure.

Expected runtime: ~10 s (one small FE solve).
Expected output:  printed summary (analytical knockdown ~0.67, FE
                  modulus retention and per-criterion strength retention)
                  and ``01_wrinkle_profile.png`` in the working directory.
"""

import matplotlib

matplotlib.use("Agg")  # headless-safe; remove to use an interactive backend
import matplotlib.pyplot as plt
import numpy as np

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis

# README quick-start geometry: half-amplitude A=0.366 mm (2 ply
# thicknesses), wavelength 16 mm, Gaussian envelope width 12 mm.
config = AnalysisConfig(
    amplitude=0.366, wavelength=16.0, width=12.0,
    morphology="stack", loading="compression",
)
result = WrinkleAnalysis(config).run()
print(result.summary())

# The same wrinkle, plotted from the profile stored on the results.
profile = result.wrinkle_config.wrinkles[0].profile
x = np.linspace(0.0, config.domain_length, 400)
z = profile.displacement(x)

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(x, z, lw=2)
ax.set_xlabel("x (mm)")
ax.set_ylabel("z(x) (mm)")
ax.set_title(
    f"Wrinkle profile: A={config.amplitude} mm, "
    f"$\\lambda$={config.wavelength} mm "
    f"(knockdown {result.analytical_knockdown:.3f})"
)
fig.tight_layout()
fig.savefig("01_wrinkle_profile.png", dpi=150)
print("Saved: 01_wrinkle_profile.png")
