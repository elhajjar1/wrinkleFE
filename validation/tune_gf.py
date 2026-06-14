#!/usr/bin/env python3
"""Tune the fibre-kink fracture energy Gf for E (and F) crack-band runs."""
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


def kd(matname, n, t, A, L, Gf, nx=16):
    mat = ML.get(matname)
    lam = Laminate.from_angles([0.0] * n, mat, ply_thickness=t)
    Lx = max(3 * L, 10); cx = Lx / 2
    pr = GaussianSinusoidal(amplitude=A, wavelength=L, width=L / 2, center=cx)
    wc = WrinkleConfiguration.from_morphology_name(
        "graded", pr, interface1=n // 2 - 1, interface2=n // 2, decay_floor=0.0)
    m = WrinkleMesh(laminate=lam, wrinkle_config=wc, Lx=Lx, Ly=10.0,
                    nx=nx, ny=2, nz_per_ply=2).generate()
    zlo, zhi = m.nodes[:, 2].min(), m.nodes[:, 2].max()
    w = compute_resin_blend(m, ResinPocketSpec.from_wrinkle(
        amplitude=A, wavelength=L, center_x=cx, z_center=zlo + 0.5 * (zhi - zlo)))
    m.resin_material = RESIN; m.resin_blend = w
    m.resin_blend_materials = {
        int(e): mat.blend(RESIN, float(w[e])) for e in np.flatnonzero(w > 0)}
    sw = ProgressiveDamageSolver(
        m, lam, applied_strain=-1.8 * mat.Xc / mat.E1, n_increments=15,
        crack_band=True, Gc_fiber=Gf).solve().peak_stress
    return sw / mat.Xc


# E representative cases (moulded)
E_CASES = [("s-1", 15, 0.42, 0.157, 11.4, 0.907),
           ("s-4", 15, 0.42, 0.354, 7.4, 0.612),
           ("s-5", 15, 0.42, 0.496, 11.0, 0.523)]
# F representative cases (vacbag)
F_CASES = [("S-M-1", 14, 0.44, 0.75, 26.0, 0.891),
           ("S-M-2", 14, 0.44, 0.75, 12.9, 0.629),
           ("S-M-3", 14, 0.44, 0.75, 8.1, 0.472)]

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "E"
    gfs = [float(x) for x in sys.argv[2:]] or [5.0, 12.0, 25.0]
    mat = "AC318_S6C10" if which == "E" else "AC318_S6C10_vacbag"
    cases = E_CASES if which == "E" else F_CASES
    for Gf in gfs:
        errs = []
        row = []
        for name, n, t, A, L, exp in cases:
            t0 = time.time()
            k = kd(mat, n, t, A, L, Gf)
            errs.append(abs(k - exp) / exp)
            row.append(f"{name}={k:.3f}(exp{exp:.3f})")
        print(f"Gf={Gf:5.1f}: " + "  ".join(row)
              + f"  MAE={100*np.mean(errs):.1f}%", flush=True)
