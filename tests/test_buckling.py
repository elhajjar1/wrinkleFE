"""Tests for the geometric stiffness and linearized buckling solver (D.4).

The geometric stiffness ``K_geo`` is verified against Euler column theory
(critical stress ~ 1/L^2).  The wrinkle-knockdown behaviour is checked
qualitatively (a wrinkle lowers the buckling factor); the *quantitative*
over-prediction of the knockdown by linear buckling is a documented
physical limitation (imperfection sensitivity + the homogenised continuum
not resolving fibre-scale kinking), not tested as an accuracy target.
"""

from __future__ import annotations

import numpy as np
import pytest

from wrinklefe.core.laminate import Laminate
from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.core.morphology import WrinkleConfiguration
from wrinklefe.core.mesh import WrinkleMesh
from wrinklefe.core.wrinkle import GaussianSinusoidal
from wrinklefe.solver.buckling import LinearBucklingSolver


def _bar(Lx, amplitude=1e-5, material=None, n_plies=8, t_ply=0.5):
    mat = material or OrthotropicMaterial.isotropic(10_000.0, 0.3, name="iso")
    lam = Laminate.from_angles([0.0] * n_plies, mat, ply_thickness=t_ply)
    pr = GaussianSinusoidal(amplitude=amplitude, wavelength=Lx,
                            width=Lx, center=Lx / 2)
    wc = WrinkleConfiguration.from_morphology_name(
        "graded", pr, interface1=n_plies // 2 - 1, interface2=n_plies // 2,
        decay_floor=0.0)
    mesh = WrinkleMesh(laminate=lam, wrinkle_config=wc, Lx=Lx, Ly=4.0,
                       nx=20, ny=3, nz_per_ply=1).generate()
    return mesh, lam


class TestGeometricStiffness:
    def test_symmetric(self):
        from wrinklefe.elements.hex8 import Hex8Element
        # Unit cube, arbitrary displacement -> K_geo symmetric.
        coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                           [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
                          dtype=float)
        el = Hex8Element(node_coords=coords,
                         material=OrthotropicMaterial.isotropic(1e4, 0.3),
                         ply_angle=0.0, wrinkle_angles=np.zeros(8))
        u = np.linspace(-0.01, 0.01, 24)
        Kg = el.geometric_stiffness_matrix(u)
        assert Kg.shape == (24, 24)
        assert np.allclose(Kg, Kg.T, atol=1e-9)

    def test_euler_slenderness_scaling(self):
        # Critical stress must fall with length roughly as 1/L^2 (Euler).
        sig = {}
        for Lx in (20.0, 40.0, 80.0):
            mesh, lam = _bar(Lx)
            lc = LinearBucklingSolver(
                mesh, lam, applied_strain=-0.001).solve().critical_load_factor
            sig[Lx] = lc * 0.001 * 10_000.0  # sigma_cr (MPa)
        # Monotonic decrease, and each doubling drops sigma by ~3-4x.
        assert sig[20.0] > sig[40.0] > sig[80.0]
        assert 2.5 < sig[20.0] / sig[40.0] < 5.0
        assert 2.5 < sig[40.0] / sig[80.0] < 5.0


class TestBucklingKnockdown:
    def test_wrinkle_lowers_buckling_factor(self):
        # Stocky coupon (Li regime): the local wrinkle mode is the lowest,
        # so a wrinkle imperfection lowers the buckling factor (and in fact
        # over-lowers it -- the documented imperfection-sensitivity limit).
        from wrinklefe.core.material import MaterialLibrary
        mat = MaterialLibrary().get("AC318_S6C10_vacbag")
        pm, lam = _bar(20.0, amplitude=1e-4, material=mat,
                       n_plies=14, t_ply=0.44)
        wm, _ = _bar(20.0, amplitude=0.6, material=mat,
                     n_plies=14, t_ply=0.44)
        lam0 = LinearBucklingSolver(
            pm, lam, applied_strain=-0.005).solve().critical_load_factor
        lamw = LinearBucklingSolver(
            wm, lam, applied_strain=-0.005).solve().critical_load_factor
        assert lamw < lam0  # imperfection lowers the buckling load

    def test_positive_finite_factor(self):
        mesh, lam = _bar(40.0)
        res = LinearBucklingSolver(mesh, lam, applied_strain=-0.001).solve()
        assert np.isfinite(res.critical_load_factor)
        assert res.critical_load_factor > 0
