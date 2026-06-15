#!/usr/bin/env python3
"""D.3: two-parameter (theta, D/T) analytical knockdown model.

The angle-only Budiansky-Fleck kink-band knockdown is scale-invariant:
at a fixed peak misalignment angle it cannot reproduce the strong
dependence of knockdown on the through-thickness penetration D/T that the
Li (2024/2025) controlled grids show (VALIDATION_DATA section 2.7). This
implements and calibrates the Bazant-type size-effect form proposed
there:

    KD(theta, D/T) = KD_floor(theta)
                     + (1 - KD_floor(theta)) / sqrt(1 + (D/T) / dt0)

with the angle floor (large-D/T, toughness-controlled asymptote)

    KD_floor(theta) = 1 / (1 + theta_rad / gamma_Y).

Asymptotes: D/T -> 0 gives KD -> 1 (a shallow wrinkle realises none of
its angle-driven knockdown); D/T -> inf gives KD -> KD_floor(theta) (a
deep wrinkle realises all of it). Two physical parameters: the matrix
yield strain gamma_Y (angle response) and the transitional penetration
dt0 (the size-effect length). UD-scoped (not for multidirectional/blocked
laminates whose low-D/T knockdown is delamination-driven).

Calibrated on the Li single-wrinkle (theta, D/T, KD) grid (section 2.7);
F is the measured-KD anchor, E (indicative /830) cross-checked.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# (case, theta_deg, D/T, KD_exp) from VALIDATION_DATA section 2.7.
E_GRID = [
    ("6.3-s-1", 4.9, 0.025, 0.907), ("6.3-s-2", 10.6, 0.026, 0.823),
    ("6.3-s-3", 16.0, 0.026, 0.758), ("6.3-s-4", 16.7, 0.056, 0.612),
    ("6.3-s-5", 15.8, 0.079, 0.523), ("4.2-s-4", 16.5, 0.083, 0.545),
    ("4.2-s-5", 14.2, 0.105, 0.506), ("8.4-s-4", 16.6, 0.042, 0.657),
    ("8.4-s-5", 15.9, 0.059, 0.558),
]
# F: S-A-2 excluded from the fit (near-surface position axis, not (theta,D/T)).
F_GRID = [
    ("S-M-1", 10.3, 0.122, 0.891), ("S-M-2", 20.1, 0.122, 0.629),
    ("S-M-3", 30.2, 0.122, 0.472), ("S-M-4", 20.1, 0.081, 0.943),
    ("S-M-5", 20.1, 0.041, 1.000),
]
S_A_2 = ("S-A-2", 20.1, 0.122, 0.981)  # position axis, reported separately


def kd_floor(theta_deg, gamma_Y):
    return 1.0 / (1.0 + np.radians(theta_deg) / gamma_Y)


def kd_model(theta_deg, dt, gamma_Y, dt0):
    floor = kd_floor(theta_deg, gamma_Y)
    return floor + (1.0 - floor) / np.sqrt(1.0 + dt / dt0)


def angle_only(theta_deg, gamma_Y):
    """Baseline: angle-only BF knockdown (no D/T term)."""
    return kd_floor(theta_deg, gamma_Y)


def kd_gate(theta_deg, dt, gamma_Y, dt0, p):
    """Penetration-gate form (VALIDATION_DATA 2.7 candidate 1):

        KD = 1 - (1 - KD_angle(theta)) * S(D/T),
        S(D/T) = min(1, (D/T / dt0) ** p).

    A shallow wrinkle (D/T -> 0, S -> 0) realises none of its angle-driven
    knockdown (KD -> 1); a deep one (S -> 1) realises all of it
    (KD -> KD_angle). The steep power p gives the sharp D/T transition the
    Li constant-angle triple shows (1.00 -> 0.94 -> 0.63), which the
    gradual sqrt size-effect form cannot.
    """
    s = np.minimum(1.0, (dt / dt0) ** p)
    ka = kd_floor(theta_deg, gamma_Y)
    return 1.0 - (1.0 - ka) * s


def fit_gate(grid):
    th = np.array([g[1] for g in grid])
    dt = np.array([g[2] for g in grid])
    kd = np.array([g[3] for g in grid])

    def resid(ppar):
        return kd_gate(th, dt, ppar[0], ppar[1], ppar[2]) - kd

    sol = least_squares(resid, [0.1, 0.1, 4.0],
                        bounds=([1e-3, 1e-3, 0.5], [2.0, 1.0, 12.0]))
    return sol.x  # gamma_Y, dt0, p


def fit(grid, two_param=True):
    th = np.array([g[1] for g in grid])
    dt = np.array([g[2] for g in grid])
    kd = np.array([g[3] for g in grid])
    if two_param:
        def resid(p):
            return kd_model(th, dt, p[0], p[1]) - kd
        sol = least_squares(resid, [0.1, 0.1],
                            bounds=([1e-3, 1e-3], [2.0, 5.0]))
        return sol.x  # gamma_Y, dt0
    else:
        def resid(p):
            return angle_only(th, p[0]) - kd
        sol = least_squares(resid, [0.1], bounds=([1e-3], [2.0]))
        return sol.x[0]


def report(name, grid, predfn):
    print(f"\n=== {name} ===")
    errs = []
    for case, th, dt, kd in grid:
        pred = predfn(th, dt)
        e = abs(pred - kd) / kd
        errs.append(e)
        flag = "" if e <= 0.20 else "  <-- >20%"
        print(f"  {case:8s} theta={th:5.1f} D/T={dt:.3f} "
              f"exp={kd:.3f} pred={pred:.3f} err={100*e:+5.1f}%{flag}")
    n_pass = sum(1 for e in errs if e <= 0.20)
    print(f"  MAE={100*np.mean(errs):.1f}%  PASS={n_pass}/{len(grid)}")
    return np.mean(errs), n_pass


if __name__ == "__main__":
    gY_a = fit(F_GRID, two_param=False)
    gY_s, dt0_s = fit(F_GRID, two_param=True)
    gY_g, dt0_g, p_g = fit_gate(F_GRID)
    print(f"Angle-only fit on F:      gamma_Y={gY_a:.4f}")
    print(f"Bazant-sqrt fit on F:     gamma_Y={gY_s:.4f}, (D/T)0={dt0_s:.4f}")
    print(f"Penetration-gate fit on F: gamma_Y={gY_g:.4f}, "
          f"(D/T)0={dt0_g:.4f}, p={p_g:.2f}")

    print("\n--- ANGLE-ONLY BASELINE ---")
    report("F", F_GRID, lambda th, dt: angle_only(th, gY_a))
    print("\n--- BAZANT SQRT SIZE-EFFECT ---")
    report("F", F_GRID, lambda th, dt: kd_model(th, dt, gY_s, dt0_s))
    print("\n--- PENETRATION GATE (steep) ---")
    report("F (calibration, measured)", F_GRID,
           lambda th, dt: kd_gate(th, dt, gY_g, dt0_g, p_g))
    print(f"\n  S-A-2 (position axis, excluded from fit): "
          f"pred={kd_gate(S_A_2[1], S_A_2[2], gY_g, dt0_g, p_g):.3f} "
          f"exp={S_A_2[3]}")
    report("E (cross-check with F's params -- shape only)", E_GRID,
           lambda th, dt: kd_gate(th, dt, gY_g, dt0_g, p_g))

    # Same gate FORM, calibrated on E's own grid (E is a different
    # material realization with its own gamma_Y / dt0 -- VALIDATION_DATA
    # 2.7).  If E fits well with its own constants, the (theta, D/T)
    # penetration-gate law is universal in form, material-specific in
    # parameters.
    gYe, dt0e, pe = fit_gate(E_GRID)
    print(f"\nPenetration-gate fit on E (own params): gamma_Y={gYe:.4f}, "
          f"(D/T)0={dt0e:.4f}, p={pe:.2f}")
    report("E (self-calibrated gate)", E_GRID,
           lambda th, dt: kd_gate(th, dt, gYe, dt0e, pe))
