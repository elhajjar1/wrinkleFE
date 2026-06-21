#!/usr/bin/env python3
"""Diagnostic: corrected (crest-located) resin pocket, on vs off.

Re-checks the E (moulded) and F (vacuum-bag) representative cases after
the through-thickness placement fix, with the resin pocket on and off, to
see the corrected pocket's effect and whether re-calibration is needed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from wrinklefe.core.laminate import Laminate
from wrinklefe.core.material import MaterialLibrary
from wrinklefe.core.mesh import WrinkleMesh
from wrinklefe.core.morphology import WrinkleConfiguration
from wrinklefe.core.resin_pocket import ResinPocketSpec, compute_resin_blend
from wrinklefe.core.wrinkle import GaussianSinusoidal
from wrinklefe.solver.progressive_damage import ProgressiveDamageSolver

ML = MaterialLibrary()
RESIN = ML.get("EPOXY_S6C10")


def run(matname, n, t, A, L, pocket=True):
    mat = ML.get(matname)
    lam = Laminate.from_angles([0.0] * n, mat, ply_thickness=t)
    Lx = max(3 * L, 10)
    cx = Lx / 2
    pr = GaussianSinusoidal(amplitude=A, wavelength=L, width=L / 2, center=cx)
    wc = WrinkleConfiguration.from_morphology_name(
        "graded", pr, interface1=n // 2 - 1, interface2=n // 2, decay_floor=0.0)
    m = WrinkleMesh(laminate=lam, wrinkle_config=wc, Lx=Lx, Ly=10.0,
                    nx=16, ny=2, nz_per_ply=2).generate()
    nr = 0
    if pocket:
        zlo, zhi = m.nodes[:, 2].min(), m.nodes[:, 2].max()
        zc = zlo + 0.5 * (zhi - zlo)
        w = compute_resin_blend(m, ResinPocketSpec.from_wrinkle(
            amplitude=A, wavelength=L, center_x=cx, z_center=zc))
        m.resin_material = RESIN
        m.resin_blend = w
        m.resin_blend_materials = {
            int(e): mat.blend(RESIN, float(w[e])) for e in np.flatnonzero(w > 0)}
        nr = int((w > 0).sum())
    sw = ProgressiveDamageSolver(
        m, lam, applied_strain=-1.8 * mat.Xc / mat.E1,
        n_increments=15, residual_factor=0.1).solve().peak_stress
    return sw / mat.Xc, nr


CASES = [
    ("E s-1", "AC318_S6C10", 15, 0.42, 0.157, 11.4, 0.907),
    ("E s-4", "AC318_S6C10", 15, 0.42, 0.354, 7.4, 0.612),
    ("E s-5", "AC318_S6C10", 15, 0.42, 0.496, 11.0, 0.523),
    ("F S-M-2", "AC318_S6C10_vacbag", 14, 0.44, 0.75, 12.9, 0.629),
    ("F S-M-3", "AC318_S6C10_vacbag", 14, 0.44, 0.75, 8.1, 0.472),
]

if __name__ == "__main__":
    for name, mat, n, t, A, L, kd in CASES:
        t0 = time.time()
        k, nr = run(mat, n, t, A, L, pocket=True)
        ko, _ = run(mat, n, t, A, L, pocket=False)
        print(f"{name:9s} exp={kd:.3f} pocket={k:.3f}(nr={nr}) "
              f"nopocket={ko:.3f} err={100 * (k - kd) / kd:+.1f}% "
              f"[{time.time() - t0:.0f}s]", flush=True)
