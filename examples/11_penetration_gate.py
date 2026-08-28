"""Penetration gate: why a UD knockdown needs more than the angle.

The angle-based Budiansky-Fleck knockdown is scale-invariant: hold the
peak fibre misalignment ``theta_max = arctan(2*pi*A/lambda)`` fixed and
it returns the same number whether the wrinkle penetrates one ply or the
whole laminate. Measured UD compressive strength does not behave that
way. The penetration gate adds the missing axis,

    KD = 1 - (1 - KD_angle(theta)) * S(D/T) * P(z)

where ``S(D/T) = min(1, (D/T / dt0)**p)`` gates the angle deficit by the
through-thickness penetration ``D/T = A/T`` and ``P(z)`` by the wrinkle's
through-thickness position. ``GATE_LI2024_MOULDED`` and
``GATE_LI2025_VACBAG`` are the calibrated AC318/S6C10 presets.

This script holds ``theta_max`` fixed while varying the laminate
thickness, so the angle-only model is flat by construction and every
difference is the gate; then it sweeps the wrinkle position; then it runs
the same UD coupon through ``AnalysisConfig`` with and without
``penetration_gate`` set.

The gate is UD-scoped -- do not apply it to multidirectional or blocked
laminates.

Expected runtime: ~1 s (analytical-only).
Expected output:  a fixed-angle thickness table, a position table, and
                  the gated vs ungated AnalysisConfig comparison.
"""

import math

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary
from wrinklefe.core.penetration_gate import (
    GATE_LI2024_MOULDED,
    GATE_LI2025_VACBAG,
    penetration_gate_kd,
    predict_from_geometry,
)

PLY_THICKNESS = 0.183  # mm
WAVELENGTH = 15.0      # mm

print("=" * 72)
print("Penetration gate — the same angle, different penetrations")
print("=" * 72)

# 1. Fixed angle, varying thickness. A/lambda is held constant so
#    theta_max is identical on every row; only D/T = A/T moves.
AMPLITUDE = 0.5  # mm
theta_deg = math.degrees(math.atan(2.0 * math.pi * AMPLITUDE / WAVELENGTH))
print(f"\nA = {AMPLITUDE} mm, lambda = {WAVELENGTH} mm "
      f"-> theta_max = {theta_deg:.2f} deg (fixed on every row)")
print()
print(f"{'n_plies':>8}{'T (mm)':>9}{'D/T':>8}"
      f"{'KD moulded':>13}{'KD vacbag':>12}")
print("-" * 72)
for n_plies in (8, 14, 24, 40, 64):
    thickness = n_plies * PLY_THICKNESS
    dt = AMPLITUDE / thickness
    kd_moulded = predict_from_geometry(
        amplitude=AMPLITUDE, wavelength=WAVELENGTH,
        n_plies=n_plies, ply_thickness=PLY_THICKNESS,
        params=GATE_LI2024_MOULDED,
    )
    kd_vacbag = predict_from_geometry(
        amplitude=AMPLITUDE, wavelength=WAVELENGTH,
        n_plies=n_plies, ply_thickness=PLY_THICKNESS,
        params=GATE_LI2025_VACBAG,
    )
    print(f"{n_plies:>8}{thickness:>9.3f}{dt:>8.3f}"
          f"{kd_moulded:>13.4f}{kd_vacbag:>12.4f}")

span = [
    predict_from_geometry(
        amplitude=AMPLITUDE, wavelength=WAVELENGTH,
        n_plies=n, ply_thickness=PLY_THICKNESS,
        params=GATE_LI2025_VACBAG,
    )
    for n in (8, 64)
]
print("\nThe angle is constant, so an angle-only model gives one number "
      "for the whole\ncolumn; the gate spans "
      f"{span[0]:.3f}-{span[1]:.3f} on the vacuum-bag preset.")
print("The thickest-wrinkle rows repeat because S(D/T) saturates at 1 "
      "once D/T >= dt0:\nthe wrinkle already penetrates far enough for "
      "the full angle deficit to apply.")

# 2. Position factor P(z). Only GATE_LI2025_VACBAG carries a calibrated
#    position exponent; the moulded preset leaves P(z) = 1.
N_PLIES = 14
DT = AMPLITUDE / (N_PLIES * PLY_THICKNESS)
print()
print("=" * 72)
print(f"Through-thickness position, {N_PLIES}-ply laminate "
      f"(D/T = {DT:.3f}, theta = {theta_deg:.2f} deg)")
print("=" * 72)
print(f"{'z_position':>12}{'KD moulded':>14}{'KD vacbag':>13}   note")
print("-" * 72)
for z_position, note in (
    (0.20, "near the bottom surface"),
    (0.35, ""),
    (0.50, "mid-plane"),
    (0.71, "Li 2025 'Above' (z = 10/14)"),
    (0.85, "near the top surface"),
):
    kd_moulded = penetration_gate_kd(
        theta_deg=theta_deg, dt=DT, params=GATE_LI2024_MOULDED,
        z_position=z_position,
    )
    kd_vacbag = penetration_gate_kd(
        theta_deg=theta_deg, dt=DT, params=GATE_LI2025_VACBAG,
        z_position=z_position,
    )
    print(f"{z_position:>12.2f}{kd_moulded:>14.4f}{kd_vacbag:>13.4f}   {note}")
print("\nP(z) is calibrated only on the vacuum-bag preset (a single "
      "Above/Middle pair,\nso q ~ 5.3 is steep and indicative); the "
      "moulded preset leaves P(z) = 1.")

# 3. The same coupon through the full pipeline, gated and ungated.
print()
print("=" * 72)
print("AnalysisConfig: gated vs ungated (14-ply UD, analytical path)")
print("=" * 72)


def build(**overrides) -> AnalysisConfig:
    """A 14-ply UD AC318/S6C10 coupon (vacuum-bag card) with a mid-plane wrinkle."""
    return AnalysisConfig(
        amplitude=AMPLITUDE, wavelength=WAVELENGTH, width=12.0,
        morphology="graded", loading="compression",
        material=MaterialLibrary().get("AC318_S6C10_vacbag"),
        angles=[0.0] * N_PLIES, ply_thickness=PLY_THICKNESS,
        analytical_only=True,
        **overrides,
    )


ungated = WrinkleAnalysis(build()).run()
gated = WrinkleAnalysis(build(penetration_gate=GATE_LI2025_VACBAG)).run()

print(f"{'model':<34}{'knockdown':>12}{'strength (MPa)':>18}")
print("-" * 72)
print(f"{'Budiansky-Fleck (angle only)':<34}"
      f"{ungated.analytical_knockdown:>12.4f}"
      f"{ungated.analytical_strength_MPa:>18.1f}")
print(f"{'penetration gate (li2025-vacbag)':<34}"
      f"{gated.analytical_knockdown:>12.4f}"
      f"{gated.analytical_strength_MPa:>18.1f}")
print()
print(f"Gate inputs: theta_max = "
      f"{math.degrees(gated.max_angle_rad):.2f} deg, "
      f"D/T = {DT:.3f}, z = {build().wrinkle_z_position:.2f}")
print("Equivalent CLI: wrinklefe analyze --gate li2025-vacbag "
      "--layup '[0]_14' \\")
print("                    --amplitude 0.5 --wavelength 15 --width 12 "
      "--morphology graded \\")
print("                    --material AC318_S6C10_vacbag "
      "--ply-thickness 0.183 --analytical-only")
