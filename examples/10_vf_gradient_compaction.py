"""Compaction Vf gradient: two caul plates squeeze resin out of the crest.

A wrinkle cured between rigid tooling does not keep a constant ply
thickness. The resin is the mobile phase: it is squeezed out of the
compacted regions and pools where the geometry opens up, so the local
fibre volume fraction rises on the crest side and falls in the trough.
``enable_vf_gradient`` models that directly — per element,

    Vf_local = vf_nominal * h0 / h

(fibre content conserved, the thickness change absorbing or expelling
resin) — and gives each element a material obtained by scaling the preset
card with the micromechanics ``Vf`` ratio. Poisson ratios and all
strengths stay at the preset values: no mixing rule predicts a strength
from ``Vf``.

This script runs the same 24-ply UD tool-flat coupon three ways — plain
wrinkle, the binary surface-resin-pocket tag, and the continuous Vf
gradient — and prints the resulting modulus retention plus the Vf field
statistics.

Expected runtime: ~30 s (three FE solves).
Expected output:  a three-row comparison table and the Vf-field summary.
"""

import numpy as np

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.compaction import VfGradientSpec, compute_vf_field
from wrinklefe.core.material import MaterialLibrary

MATERIAL = "IM7_8552"
AMPLITUDE = 0.25          # mm; below the tool_flat inversion bound (0.293)


def build(**overrides) -> AnalysisConfig:
    """A 24-ply UD coupon wrinkled between two caul plates."""
    config = AnalysisConfig(
        morphology="tool_flat",
        amplitude=AMPLITUDE, wavelength=16.0, width=12.0,
        surface_pocket_side="both",       # two caul plates: top AND bottom
        surface_transition_plies=2,
        material=MaterialLibrary().get(MATERIAL),
        angles=[0.0] * 24, ply_thickness=0.183,
        domain_length=40.0, domain_width=4.0,
        nx=40, ny=2, nz_per_ply=1,
        analytical_only=False,
        **overrides,
    )
    return config


print("=" * 72)
print("Compaction Vf gradient — 24-ply UD, tool_flat, two caul plates")
print("=" * 72)

# 1. Plain wrinkle: no trough treatment at all.
plain = build()
plain.enable_surface_resin_pockets = False       # tool_flat auto-enables them
plain_result = WrinkleAnalysis(plain).run()

# 2. The binary surface resin pocket (the pre-#379 model).
binary_result = WrinkleAnalysis(build()).run()

# 3. The continuous Vf / ply-thickness gradient.
gradient_config = build(enable_vf_gradient=True)
gradient_result = WrinkleAnalysis(gradient_config).run()

rows = (
    ("no trough treatment", plain_result),
    ("binary surface pockets", binary_result),
    ("compaction Vf gradient", gradient_result),
)
print(f"{'model':<26}{'modulus_retention_global':>26}")
print("-" * 72)
for label, result in rows:
    print(f"{label:<26}{result.modulus_retention_global:>26.6f}")

delta = (
    gradient_result.modulus_retention_global
    - binary_result.modulus_retention_global
)
print()
print(f"gradient - binary pockets: {delta:+.6f} "
      f"({100.0 * delta / binary_result.modulus_retention_global:+.3f} %)")

# The Vf field itself, recomputed on the solved mesh.
spec = VfGradientSpec.for_material(MATERIAL)
vf = compute_vf_field(gradient_result.mesh, spec)
changed = vf != spec.vf_nominal
print()
print(f"Vf field: nominal {spec.vf_nominal:.3f}, "
      f"range {vf.min():.3f}-{vf.max():.3f}")
print(f"  resin-rich (Vf < nominal): {int(np.count_nonzero(vf < spec.vf_nominal))} elements")
print(f"  compacted  (Vf > nominal): {int(np.count_nonzero(vf > spec.vf_nominal))} elements")
print(f"  saturated at vf_max={spec.vf_max:.2f}: "
      f"{int(np.count_nonzero(vf >= spec.vf_max))} elements")
print(f"  re-materialised: {len(gradient_result.mesh.resin_blend_materials)} "
      f"of {gradient_result.mesh.n_elements} elements, sharing "
      f"{len({id(m) for m in gradient_result.mesh.resin_blend_materials.values()})} "
      f"material objects")
print(f"  (changed from nominal before quantization: "
      f"{int(np.count_nonzero(changed))} elements)")
