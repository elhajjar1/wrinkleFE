"""Localized (through-width) vs uniform wrinkle knockdown (issue #300).

Real manufacturing wrinkles (resin infusion, AFP, tape winding) are
usually *localized* — high amplitude at mid-width, fading toward the
specimen edges — whereas the default analysis assumes the wrinkle is
uniform across the full width. This overstates the defect volume and
biases the predicted knockdown conservative.

This script runs the *same* crest geometry two ways through the FE path:

* ``transverse_mode="uniform"``   — the x-only baseline (default).
* ``transverse_mode="gaussian_decay"`` — a ``WrinkleSurface3D`` whose
  amplitude decays toward the edges (localized mid-width defect).

The crest fibre angle at the mid-width centreline is identical (both
surfaces are full-amplitude there), so ``mesh_max_angle_rad`` matches;
the difference shows up in the width-averaged FE knockdown — the
localized wrinkle retains more stiffness and strength.

Expected runtime: ~20 s (two small FE solves).
Expected output:  a printed retention comparison and
                  ``transverse_wrinkle_knockdown.png`` in the working
                  directory.
"""

import matplotlib

matplotlib.use("Agg")  # headless-safe; remove to use an interactive backend
import matplotlib.pyplot as plt
import numpy as np

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis


def run(mode: str) -> "object":
    """Run the shared geometry with the given transverse mode (FE path)."""
    cfg = AnalysisConfig(
        amplitude=0.366,
        wavelength=16.0,
        width=12.0,
        morphology="stack",
        loading="compression",
        transverse_mode=mode,
    )
    return WrinkleAnalysis(cfg).run(analytical_only=False)


def strength_retention(result: "object") -> float:
    """Weakest-criterion FE strength retention factor (higher = milder)."""
    factors = result.retention_factors
    return min(factors.values()) if factors else float("nan")


uniform = run("uniform")
localized = run("gaussian_decay")

u_mod = uniform.modulus_retention_global
l_mod = localized.modulus_retention_global
u_str = strength_retention(uniform)
l_str = strength_retention(localized)

print("Through-width wrinkle knockdown (same crest amplitude A=0.366 mm)")
print("-" * 62)
print(f"{'metric':<34}{'uniform':>12}{'gaussian_decay':>16}")
print(f"{'mesh max fibre angle (deg)':<34}"
      f"{np.degrees(uniform.mesh_max_angle_rad):>12.3f}"
      f"{np.degrees(localized.mesh_max_angle_rad):>16.3f}")
print(f"{'global modulus retention':<34}{u_mod:>12.4f}{l_mod:>16.4f}")
print(f"{'strength retention (weakest)':<34}{u_str:>12.4f}{l_str:>16.4f}")
print("-" * 62)
print(
    "Localized wrinkle knockdown is MILDER: strength retention "
    f"{u_str:.3f} -> {l_str:.3f}, modulus retention {u_mod:.4f} -> "
    f"{l_mod:.4f} (crest angle unchanged)."
)

# --- Figure: the transverse envelope and the retention comparison ------
localized_profile = localized.wrinkle_config.wrinkles[0].profile
span_y = localized_profile.span_y
y = np.linspace(0.0, span_y, 400)
# f(y) at the crest (x = wrinkle centre) normalised to the crest value.
x_crest = np.full_like(y, localized_profile.profile.center)
envelope = localized_profile.displacement(x_crest, y)
crest = localized_profile.profile.displacement(
    np.array([localized_profile.profile.center])
)[0]
envelope = envelope / crest if crest else envelope

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))

ax0.axhline(1.0, color="tab:blue", lw=2, label="uniform")
ax0.plot(y, envelope, color="tab:orange", lw=2, label="gaussian_decay")
ax0.set_xlabel("y across specimen width (mm)")
ax0.set_ylabel("normalised crest amplitude f(y)")
ax0.set_title("Through-width amplitude envelope")
ax0.set_ylim(0.0, 1.1)
ax0.legend()

labels = ["modulus\nretention", "strength\nretention"]
uniform_vals = [u_mod, u_str]
localized_vals = [l_mod, l_str]
xpos = np.arange(len(labels))
bar_w = 0.35
ax1.bar(xpos - bar_w / 2, uniform_vals, bar_w, label="uniform",
        color="tab:blue")
ax1.bar(xpos + bar_w / 2, localized_vals, bar_w, label="gaussian_decay",
        color="tab:orange")
ax1.set_xticks(xpos)
ax1.set_xticklabels(labels)
ax1.set_ylabel("retention (1 = no knockdown)")
ax1.set_ylim(0.0, 1.05)
ax1.set_title("Localized wrinkle -> milder knockdown")
ax1.legend()

fig.tight_layout()
fig.savefig("transverse_wrinkle_knockdown.png", dpi=150)
print("Saved: transverse_wrinkle_knockdown.png")
