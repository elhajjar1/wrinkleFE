"""Tests for the progressive-damage load-stepping solver.

These use a deliberately coarse UD glass/epoxy mesh (fast) and assert the
robust, mesh-insensitive properties of the solver rather than absolute
strengths:

* the pristine UD baseline fails near its compressive allowable ``Xc``
  (the MaxStress fibre-compression check that LaRC05 alone misses),
* a wrinkle lowers the predicted ultimate strength below pristine,
* the API surface (history, peak, failed-element count) is populated.
"""

from __future__ import annotations

import pytest

from wrinklefe.core.laminate import Laminate
from wrinklefe.core.material import MaterialLibrary
from wrinklefe.core.mesh import WrinkleMesh
from wrinklefe.core.morphology import WrinkleConfiguration
from wrinklefe.core.wrinkle import GaussianSinusoidal
from wrinklefe.solver.progressive_damage import (
    ProgressiveDamageSolver,
    _mode_family,
)

ML = MaterialLibrary()
MAT = ML.get("AC318_S6C10")
N_PLIES = 15
T_PLY = 0.42
WAVELENGTH = 7.4
LX = 22.2


def _build_mesh(amplitude: float):
    lam = Laminate.from_angles([0.0] * N_PLIES, MAT, ply_thickness=T_PLY)
    profile = GaussianSinusoidal(
        amplitude=amplitude, wavelength=WAVELENGTH,
        width=WAVELENGTH / 2.0, center=LX / 2.0,
    )
    wc = WrinkleConfiguration.from_morphology_name(
        "graded", profile,
        interface1=N_PLIES // 2 - 1, interface2=N_PLIES // 2,
        decay_floor=0.0,
    )
    mesh = WrinkleMesh(
        laminate=lam, wrinkle_config=wc, Lx=LX, Ly=10.0,
        nx=12, ny=2, nz_per_ply=1,
    ).generate()
    return mesh, lam


def _solve(amplitude: float):
    mesh, lam = _build_mesh(amplitude)
    return ProgressiveDamageSolver(
        mesh, lam, applied_strain=-0.03, n_increments=12,
        residual_factor=0.1,
    ).solve()


def test_mode_family_maps_kinking_to_fiber():
    assert _mode_family("fiber_kinking") == "fiber_compression"
    assert _mode_family("fiber_tension") == "fiber_tension"
    assert _mode_family("matrix_compression") == "matrix_compression"


class TestProgressiveDamage:
    def test_pristine_equals_Xc(self):
        res = _solve(0.0)
        # Pristine UD fibre compression fails exactly at Xc: the
        # first-failure (FI=1) interpolation is increment-robust and a
        # uniform laminate has no stress concentration, so the peak is the
        # compressive allowable independent of the load-step count.
        assert res.peak_stress == pytest.approx(MAT.Xc, rel=1e-3)
        assert res.n_failed_elements > 0

    def test_wrinkle_knocks_down_strength(self):
        pristine = _solve(0.0).peak_stress
        wrinkled = _solve(0.354).peak_stress
        assert wrinkled < pristine
        # Knockdown is physically meaningful (not a rounding wobble).
        assert wrinkled / pristine < 0.95

    def test_history_and_api(self):
        res = _solve(0.354)
        assert len(res.history) == 12
        strains = [e for e, _s in res.history]
        # Monotonically increasing |strain| over the increments.
        assert all(abs(strains[i]) < abs(strains[i + 1])
                   for i in range(len(strains) - 1))
        # Peak is the larger of the interpolated first-failure load and
        # the redistributed post-equilibrium maximum, so it is at least
        # the discrete history max.
        assert res.peak_stress >= max(s for _e, s in res.history) - 1e-6
        assert res.n_failed_elements > 0

    def test_rejects_bad_inputs(self):
        mesh, lam = _build_mesh(0.0)
        with pytest.raises(ValueError):
            ProgressiveDamageSolver(mesh, lam, applied_strain=0.0)
        with pytest.raises(ValueError):
            ProgressiveDamageSolver(
                mesh, lam, applied_strain=-0.02, n_increments=0
            )
