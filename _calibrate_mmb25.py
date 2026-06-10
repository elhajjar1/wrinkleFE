"""Calibration -- NX=200, fine increments around r=100.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "tests/integration")
import test_mmb_25_experimental_validation as t  # type: ignore

t.NX = 200

# Test different (N_INC, DELTA_II_MAX) -- finer steps catch earlier failure.
configs = [
    (100, 0.2),
    (200, 0.2),
    (400, 0.2),
    (200, 0.15),
]

coh_props = t._build_cohesive_properties()

r = 100.0
print(f"r={r}, NX={t.NX}")
print(
    f"{'N_INC':>6} | {'D_MAX':>6} | {'step':>7} | {'best_mr':>8} | "
    f"{'P_ff':>8} | {'dII_ff':>7} | {'n_failed':>8}"
)
print("-" * 80)

for N_INC, DELTA_II_MAX in configs:
    mesh, cohesive_elements, is_bonded = t._build_mesh(coh_props)
    res = t._drive_mmb_fixed(
        mesh, cohesive_elements, is_bonded,
        delta_II_max=DELTA_II_MAX,
        delta_ratio_opening=r,
        n_increments=N_INC,
        verbose=False,
    )
    cmr = res["mode_ratio"]
    mr_history = cmr[cmr >= 0.0]
    if mr_history.size > 0:
        best_mr = float(mr_history[np.argmin(np.abs(mr_history - 0.25))])
    else:
        best_mr = float("nan")
    i_ff = int(res["i_first_full_fail"])
    P_ff = float(res["P"][i_ff]) if i_ff > 0 else float("nan")
    dII_ff = float(res["delta_II"][i_ff]) if i_ff > 0 else float("nan")
    n_failed = int(res["n_failed_elements"])
    step = DELTA_II_MAX / N_INC
    print(
        f"{N_INC:>6d} | {DELTA_II_MAX:>6.3f} | {step:>7.4f} | "
        f"{best_mr:>+8.3f} | {P_ff:>8.1f} | {dII_ff:>7.4f} | {n_failed:>8d}"
    )
