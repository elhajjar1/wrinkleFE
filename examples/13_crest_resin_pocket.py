"""Crest resin lens: the soft, fibre-free inclusion a machined wrinkle leaves.

The cosine insert that creates the wrinkle in the Li UD glass/epoxy
datasets is co-cured bulk epoxy, so what sits at the crest is not
homogenised composite at all — it is neat, fibre-free, isotropic matrix.
``enable_resin_pocket`` models that: ``compute_resin_mask`` flags the hex
elements whose centroids fall inside a cosine lens at the crest, and
``compute_resin_blend`` grades the material from neat resin at the lens
centre to the host ply at its boundary.

The lens is ``height_scale * A`` in crest half-height and
``length_scale * lambda / 2`` in longitudinal half-extent, tapering to
zero half-height at its longitudinal edges. The defect is counted once:
where the resin blends in, the fibre-misalignment angle is scaled by
``(1 - weight)`` (``MeshData.resin_angle_scale``) so the resin path and
the kink-band path do not double-count the same wrinkle.

This script runs one 14-ply UD coupon with the lens off, graded on, and
binary on, then sweeps the lens height. The graded transition is the
default for a reason, and the binary run shows why.

Expected runtime: ~20 s (six FE solves on a coarse mesh).
Expected output:  the off/graded/binary comparison and a height-scale
                  sweep with the tagged-element counts.
"""

import numpy as np

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary

AMPLITUDE = 0.5     # mm
WAVELENGTH = 15.0   # mm
N_PLIES = 14
PLY_THICKNESS = 0.183


def build(**overrides) -> AnalysisConfig:
    """A 14-ply UD AC318/S6C10 coupon with a mid-plane wrinkle."""
    return AnalysisConfig(
        amplitude=AMPLITUDE, wavelength=WAVELENGTH, width=10.0,
        morphology="graded", loading="compression",
        material=MaterialLibrary().get("AC318_S6C10_vacbag"),
        angles=[0.0] * N_PLIES, ply_thickness=PLY_THICKNESS,
        # Coarse on purpose; see 07_mesh_convergence.py for convergence.
        domain_length=30.0, domain_width=4.0,
        nx=24, ny=2, nz_per_ply=1,
        analytical_only=False,
        **overrides,
    )


def tagged(result) -> int:
    """How many elements the lens actually touched, graded or binary."""
    mesh = result.mesh
    if mesh.resin_blend is not None:
        return int(np.count_nonzero(mesh.resin_blend > 0.0))
    if mesh.resin_mask is not None:
        return int(np.count_nonzero(mesh.resin_mask))
    return 0


print("=" * 72)
print(f"Crest resin lens — {N_PLIES}-ply UD, A = {AMPLITUDE} mm, "
      f"lambda = {WAVELENGTH} mm")
print("=" * 72)

no_lens = WrinkleAnalysis(build()).run()
graded = WrinkleAnalysis(build(enable_resin_pocket=True)).run()
binary = WrinkleAnalysis(
    build(enable_resin_pocket=True, resin_pocket_graded=False)
).run()

print(f"\n{'lens':<26}{'E_x/E_x0 (global)':>20}{'elements tagged':>18}")
print("-" * 72)
for label, result in (
    ("off", no_lens),
    ("on, graded (default)", graded),
    ("on, binary", binary),
):
    print(f"{label:<26}{result.modulus_retention_global:>20.6f}"
          f"{tagged(result):>18}")
print(f"{'':<26}{'':>20}{'of ' + str(no_lens.mesh.n_elements):>18}")

print()
print("The binary lens over-weakens: a hard fibre/resin jump introduces a")
print("spurious stress concentration on top of the misaligned-fibre crest")
print("knockdown the mesh already carries, double-counting the defect. The")
print("graded blend scales the misalignment angle by (1 - weight) as the")
print("resin blends in, so each mechanism is counted once. Prefer the")
print("default (resin_pocket_graded=True).")

# How big is the lens? Sweep its crest half-height.
print()
print("=" * 72)
print("Lens height sweep (half-extent fixed at lambda/2 = "
      f"{WAVELENGTH / 2:.1f} mm)")
print("=" * 72)
print(f"\n{'height_scale':>13}{'crest half-height (mm)':>25}"
      f"{'E_x/E_x0':>12}{'tagged':>9}")
print("-" * 72)
for height_scale in (0.5, 1.0, 1.5):
    result = WrinkleAnalysis(
        build(enable_resin_pocket=True,
              resin_pocket_height_scale=height_scale)
    ).run()
    print(f"{height_scale:>13.1f}{height_scale * AMPLITUDE:>25.3f}"
          f"{result.modulus_retention_global:>12.6f}{tagged(result):>9}")

print("\nThe lens crest half-height is height_scale * A and its "
      "longitudinal\nhalf-extent is length_scale * lambda/2, so a deeper "
      "wrinkle leaves a thicker\npocket. Both scales are "
      "AnalysisConfig knobs "
      "(resin_pocket_height_scale,\nresin_pocket_length_scale).")
print()
print("The lens is an FE-path feature: it changes the mesh materials, not "
      "the\nclosed-form knockdown, so it has no effect under "
      "analytical_only=True.")
print()
print("Nearest CLI form (the domain size has no analyze flag, so the "
      "mesh differs;\nfor an exact reproduction save this config and "
      "pass --config):")
print("  wrinklefe analyze --resin-pocket --layup '[0]_14' \\")
print("      --amplitude 0.5 --wavelength 15 --width 10 "
      "--morphology graded \\")
print("      --material AC318_S6C10_vacbag --nx 24 --ny 2")
