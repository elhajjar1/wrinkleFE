"""Progressive damage: carrying a UD compression run past first-ply failure.

The linear FE path reports a first-ply-failure index, and for pristine
unidirectional compression the LaRC05 index never activates — so the
linear route returns no usable knockdown at all for exactly the case UD
wrinkle work cares about. ``ProgressiveDamageSolver`` fixes that: it
ramps the applied strain in increments, re-solves, degrades newly-failed
elements by failure-mode family (ply discount), and reports the peak
carried nominal stress over the history as the ultimate strength. The
knockdown is that peak for the wrinkled coupon over the same peak for a
pristine flat baseline.

This script runs one 12-ply UD coupon on a deliberately coarse mesh with
``enable_progressive_damage=True``, prints the load-increment table so
the peak and the post-peak load drop are visible, and contrasts the
ultimate-strength knockdown with the first-ply-failure retention factors
the linear path produces for the same run.

Expected runtime: ~15 s (two load-stepped FE solves: wrinkled + pristine).
Expected output:  the load-increment table, the FPF-vs-ultimate
                  comparison, and the progressive knockdown.
"""

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary

N_INCREMENTS = 18

config = AnalysisConfig(
    amplitude=0.7, wavelength=15.0, width=10.0,
    morphology="graded", loading="compression",
    material=MaterialLibrary().get("AC318_S6C10_vacbag"),
    angles=[0.0] * 12, ply_thickness=0.183,
    # Coarse on purpose: the point is the load-stepping, not mesh
    # convergence. Use 07_mesh_convergence.py for the latter.
    domain_length=30.0, domain_width=4.0,
    nx=16, ny=2, nz_per_ply=1,
    analytical_only=False,
    enable_progressive_damage=True,
    progressive_n_increments=N_INCREMENTS,
)

result = WrinkleAnalysis(config).run()

print("=" * 72)
print("Progressive damage — 12-ply UD, A = 0.7 mm, "
      f"{N_INCREMENTS} increments")
print("=" * 72)

# The load history: applied nominal strain vs the post-equilibrium
# carried nominal stress at each increment.
history = result.progressive_history or []
running_peak = 0.0
peak_increment = 0
print(f"\n{'incr':>5}{'applied strain':>17}{'carried stress (MPa)':>23}   note")
print("-" * 72)
for i, (strain, stress) in enumerate(history, start=1):
    note = ""
    if stress > running_peak:
        running_peak = stress
        peak_increment = i
    elif i == peak_increment + 1:
        note = "<- load drop (damage localises)"
    print(f"{i:>5}{strain:>17.5f}{stress:>23.2f}   {note}".rstrip())

print(f"\nLargest sampled carried stress: {running_peak:.2f} MPa "
      f"at increment {peak_increment}.")
print(f"Reported ultimate:              "
      f"{result.progressive_strength_MPa:.2f} MPa")
print("The reported ultimate is increment-robust: it is the larger of the")
print("first-failure load (the elastic carried stress interpolated to the")
print("strain where the global failure index first reaches 1.0) and the")
print("largest post-equilibrium stress above, so it does not swing with the")
print("increment count and can exceed any single sampled row.")

# First-ply failure vs ultimate: the reason this path exists.
print()
print("=" * 72)
print("First-ply failure vs ultimate strength")
print("=" * 72)
print(f"{'quantity':<42}{'value':>14}")
print("-" * 72)
for name, value in sorted((result.retention_factors or {}).items()):
    print(f"{'FE first-ply-failure retention (' + name + ')':<42}{value:>14.4e}")
print(f"{'analytical knockdown (Budiansky-Fleck)':<42}"
      f"{result.analytical_knockdown:>14.4f}")
print(f"{'progressive ultimate, wrinkled (MPa)':<42}"
      f"{result.progressive_strength_MPa:>14.2f}")
print(f"{'progressive ultimate, pristine (MPa)':<42}"
      f"{result.progressive_pristine_strength_MPa:>14.2f}")
print(f"{'progressive knockdown':<42}"
      f"{result.progressive_knockdown:>14.4f}")

print()
print("The first-ply-failure retention is degenerate here: a pristine UD "
      "coupon in\ncompression never activates the linear LaRC05 index, so "
      "the ratio it is formed\nfrom is meaningless. The progressive "
      "knockdown is the FE number to quote for\nthis case.")
print()
print("The analytical number above is the angle-only Budiansky-Fleck "
      "knockdown, which\nis scale-invariant and does not know how deep "
      "this wrinkle penetrates — see\n11_penetration_gate.py for the "
      "closed-form model that does.")
print()
print("Nearest CLI form (the domain size has no analyze flag, so the "
      "mesh differs;\nfor an exact reproduction save this config and "
      "pass --config):")
print("  wrinklefe analyze --progressive --increments 18 \\")
print("      --layup '[0]_12' --amplitude 0.7 --wavelength 15 --width 10 \\")
print("      --morphology graded --material AC318_S6C10_vacbag "
      "--nx 16 --ny 2")
