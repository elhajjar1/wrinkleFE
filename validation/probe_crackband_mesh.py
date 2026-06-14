#!/usr/bin/env python3
"""D.1 crack-band mesh-objectivity probe.

Runs E 6.3-s-4 (measured KD 0.612) through the crack-band progressive
solver at several longitudinal mesh densities with a FIXED fracture
energy Gf. If crack-band regularization works, the predicted KD is
~mesh-independent (vs the ~38% swing the un-regularized ply-discount
showed between nx=16 and nx=24).
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


def kd(matname, n, t, A, L, nx, Gf, crack_band=True, pocket=True):
    mat = ML.get(matname)
    lam = Laminate.from_angles([0.0] * n, mat, ply_thickness=t)
    Lx = max(3 * L, 10); cx = Lx / 2
    pr = GaussianSinusoidal(amplitude=A, wavelength=L, width=L / 2, center=cx)
    wc = WrinkleConfiguration.from_morphology_name(
        "graded", pr, interface1=n // 2 - 1, interface2=n // 2, decay_floor=0.0)
    m = WrinkleMesh(laminate=lam, wrinkle_config=wc, Lx=Lx, Ly=10.0,
                    nx=nx, ny=2, nz_per_ply=2).generate()
    if pocket:
        zlo, zhi = m.nodes[:, 2].min(), m.nodes[:, 2].max()
        w = compute_resin_blend(m, ResinPocketSpec.from_wrinkle(
            amplitude=A, wavelength=L, center_x=cx,
            z_center=zlo + 0.5 * (zhi - zlo)))
        m.resin_material = RESIN; m.resin_blend = w
        m.resin_blend_materials = {
            int(e): mat.blend(RESIN, float(w[e])) for e in np.flatnonzero(w > 0)}
    sw = ProgressiveDamageSolver(
        m, lam, applied_strain=-1.8 * mat.Xc / mat.E1, n_increments=15,
        residual_factor=0.1, crack_band=crack_band, Gc_fiber=Gf,
    ).solve().peak_stress
    return sw / mat.Xc


if __name__ == "__main__":
    Gf = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0
    print(f"E 6.3-s-4 (exp 0.612), crack-band Gf={Gf} N/mm — mesh objectivity:")
    for nx in (12, 16, 24):
        t0 = time.time()
        k = kd("AC318_S6C10", 15, 0.42, 0.354, 7.4, nx, Gf)
        print(f"  nx={nx:2d}: KD={k:.3f}  [{time.time()-t0:.0f}s]", flush=True)
