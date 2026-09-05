"""Cure residual stress: what a cool-down from cure does to a wrinkle.

A laminate is stress-free at its cure temperature, not at room
temperature. Cooling ``[0/90]s`` IM7/8552 from a 177 degC cure to 22 degC
service is ``delta_T = -155``, and because the fibre direction barely
shrinks while the transverse direction shrinks a lot, each ply ends up
holding its neighbours stretched across the fibres: the matrix direction
of *every* ply goes into tension before any mechanical load is applied.

``AnalysisConfig.delta_T`` drives both solution paths (issue #273). The
CLT path adds the thermal resultants to the ABD solve; the FE path
assembles the element thermal initial-strain load vector

    f_th = integral( B^T C_bar eps_th dV ),      eps_th = alpha_glob * delta_T

and subtracts the thermal strain during stress recovery, so
``sigma = C_bar (B u - eps_th)``. Because the wrinkle rotates the fibre
frame, the CTE mismatch concentrates in exactly the elements where the
failure criteria are evaluated — which is the reason to carry the term
into the FE path at all rather than leaving it at laminate level.

The residual is a *load-independent* offset, so its effect on failure is
signed, and this script shows both signs: in tension the residual adds to
the mechanically-driven matrix tension and the failure index rises; in
compression it relieves the matrix compression and the index falls.
Treating cure residual stress as a blanket penalty would be wrong.

SIGN CONVENTION: ``delta_T = T_service - T_stress_free``. A cure
cool-down is NEGATIVE. Inverting it flips the residual matrix stress
from tension to compression, which is the difference between predicting
cure microcracking and missing it.

Expected runtime: ~15 s (four FE solves on a small mesh).
Expected output:  a residual-stress table and a signed failure-index
                  comparison for tension and compression.
"""

import numpy as np

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.laminate import Laminate, LoadState
from wrinklefe.core.material import MaterialLibrary

MATERIAL = "IM7_8552"
CURE_COOLDOWN_DT = -155.0    # 177 degC cure -> 22 degC service
LAYUP = [0.0, 90.0, 90.0, 0.0]
PLY_T = 0.125


def build(delta_T: float, applied_strain: float) -> AnalysisConfig:
    """A small wrinkled [0/90]s coupon at a given temperature and load."""
    return AnalysisConfig(
        amplitude=0.15, wavelength=12.0, width=8.0,
        morphology="graded",
        material=MaterialLibrary().get(MATERIAL),
        angles=list(LAYUP), ply_thickness=PLY_T,
        loading="tension" if applied_strain > 0 else "compression",
        applied_strain=applied_strain,
        domain_length=16.0, domain_width=8.0,
        nx=8, ny=3, nz_per_ply=1,
        analytical_only=False,
        delta_T=delta_T,
    )


print("=" * 72)
print("Cure residual stress — [0/90]s IM7/8552, delta_T = "
      f"{CURE_COOLDOWN_DT:+.0f} degC")
print("=" * 72)

# ---------------------------------------------------------------------- #
# 1. The closed-form CLT answer, for reference.
# ---------------------------------------------------------------------- #
laminate = Laminate.from_angles(
    LAYUP, MaterialLibrary().get(MATERIAL), ply_thickness=PLY_T
)
cold = LoadState(delta_T=CURE_COOLDOWN_DT)
print()
print("CLT ply stresses under the cool-down alone (no mechanical load):")
print(f"  {'ply':<6}{'angle':>8}{'sigma_1 [MPa]':>16}{'sigma_2 [MPa]':>16}")
for k, angle in enumerate(LAYUP):
    s = laminate.ply_stresses_local(cold, k, "mid")
    print(f"  {k:<6}{angle:>8.0f}{s[0]:>16.2f}{s[1]:>16.2f}")
print("  Every ply's matrix direction (sigma_2) is in TENSION — the")
print("  cure-microcracking driver.")

# ---------------------------------------------------------------------- #
# 2. The same load through the FE path, wrinkled, under both load signs.
# ---------------------------------------------------------------------- #
runs = {
    (strain, delta_T): WrinkleAnalysis(build(delta_T, strain)).run()
    for strain in (-0.005, +0.005)
    for delta_T in (CURE_COOLDOWN_DT, 0.0)
}

print()
print("FE residual matrix stress (mean element-frame sigma_2, MPa):")
print(f"  {'loading':<14}{'delta_T=0':>12}{'cooled':>12}{'shift':>12}")
shifts = {}
for strain in (-0.005, +0.005):
    label = "tension" if strain > 0 else "compression"
    neutral = runs[strain, 0.0].field_results.stress_local[:, :, 1].mean()
    cooled = runs[strain, CURE_COOLDOWN_DT].field_results.stress_local[
        :, :, 1
    ].mean()
    shifts[label] = cooled - neutral
    print(f"  {label:<14}{neutral:>12.3f}{cooled:>12.3f}"
          f"{cooled - neutral:>+12.3f}")
print("  The shift is the SAME under both load signs: a thermal initial")
print("  strain is a constant right-hand-side term, not a stiffness change.")

print()
print("FE LaRC05 maximum failure index:")
print(f"  {'loading':<14}{'delta_T=0':>12}{'cooled':>12}{'change':>12}")
for strain in (-0.005, +0.005):
    label = "tension" if strain > 0 else "compression"
    neutral = float(np.max(runs[strain, 0.0].failure_indices["larc05"]))
    cooled = float(
        np.max(runs[strain, CURE_COOLDOWN_DT].failure_indices["larc05"])
    )
    print(f"  {label:<14}{neutral:>12.4f}{cooled:>12.4f}"
          f"{100.0 * (cooled / neutral - 1.0):>+11.1f}%")
print("  Signed, not a blanket penalty: residual matrix TENSION adds to a")
print("  tensile matrix state and relieves a compressive one.")

# ---------------------------------------------------------------------- #
# 3. The deliberate asymmetry in the reported metrics.
# ---------------------------------------------------------------------- #
print()
print("Reported stiffness is deliberately left alone:")
cold_E = runs[-0.005, CURE_COOLDOWN_DT].modulus_retention_global
warm_E = runs[-0.005, 0.0].modulus_retention_global
print(f"  modulus_retention_global  cooled {cold_E:.6f}  "
      f"delta_T=0 {warm_E:.6f}")
print("  The global modulus solve is pinned at delta_T = 0: a residual")
print("  load adds a strain-INDEPENDENT offset to the reaction force, and")
print("  reporting that as a stiffness change would be a wrong number.")
print("  The pristine retention baseline, by contrast, IS solved at the")
print("  same delta_T, so retention factors compare like with like.")
