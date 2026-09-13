"""Independent checks on the ``FieldResults`` reporting surface.

``equivalent_resultants`` and ``interlaminar_stresses`` are both public,
both documented as the bridge to CLT and the delamination driver
respectively — and both had **zero tests**.  A mutation audit confirmed
what that costs: swapping the Voigt slots that feed ``tau_13``/``tau_23``,
and dropping the lever arm from the moment integrand (turning a moment
resultant into a force resultant), each left all 2167 tests green.

Both were also wrong, and the tests below are written against independent
facts rather than against the implementations:

* a membrane load state applied to a flat laminate must come back out of
  the through-thickness integral — equilibrium, not a property of how the
  integral is coded;
* the reported interlaminar peak must equal the actual peak in the
  adjoining elements, which is a statement about the field, not the
  reduction.
"""

from __future__ import annotations

import numpy as np
import pytest

from wrinklefe.core.laminate import Laminate, LoadState
from wrinklefe.core.material import MaterialLibrary
from wrinklefe.core.mesh import WrinkleMesh
from wrinklefe.core.morphology import WrinkleConfiguration
from wrinklefe.core.wrinkle import GaussianSinusoidal
from wrinklefe.solver.boundary import BoundaryHandler
from wrinklefe.solver.static import StaticSolver

PLY_T = 0.183
QI = [0.0, 45.0, -45.0, 90.0, 90.0, -45.0, 45.0, 0.0]
CROSS = [0.0, 90.0, 90.0, 0.0]


def _material():
    return MaterialLibrary().get("IM7_8552")


def _flat(angles, nz_per_ply):
    lam = Laminate.from_angles(angles, _material(), ply_thickness=PLY_T)
    mesh = WrinkleMesh(
        laminate=lam, wrinkle_config=None,
        Lx=20.0, Ly=10.0, nx=10, ny=4, nz_per_ply=nz_per_ply,
    ).generate()
    return mesh, lam


def _solve_load_state(mesh, lam, load):
    return StaticSolver(mesh, lam).solve(
        BoundaryHandler.load_state_to_bcs(load, mesh), solver="direct",
    )


# --------------------------------------------------------------------------- #
# equivalent_resultants
# --------------------------------------------------------------------------- #


class TestEquivalentResultants:
    """What goes in as a resultant must come back out of the integral.

    The previous implementation ran a trapezoid over element *centroids*,
    so it integrated over ``h - t`` instead of ``h`` and dropped the outer
    half-element at each surface — and it smoothed across the stress jump
    at every ply boundary.  For ``[0, 90, 90, 0]`` the exact sum
    ``s1+s2+s3+s4`` became ``s1/2+s2+s3+s4/2``, losing half of each stiff
    surface ply.  Measured recovery of an applied ``Nx = -100``:

    ======  ===========  ==========
    nz      old          new
    ======  ===========  ==========
    1       -52.5        -99.97
    2       -76.2        -99.97
    4       -92.4        -100.1
    ======  ===========  ==========

    The point of parametrising over ``nz_per_ply`` is that the old form
    converged only as ``O(1/nz)``; a mesh-independent result is the
    signature of a correct quadrature.
    """

    TOL = 0.03   # 3 % — FE/CLT discretisation, not integration error

    @pytest.mark.parametrize("nz", [1, 2, 4])
    @pytest.mark.parametrize("angles", [CROSS, QI], ids=["cross", "qi"])
    def test_membrane_resultant_is_recovered_at_any_mesh_density(
        self, nz, angles
    ):
        mesh, lam = _flat(angles, nz)
        load = LoadState(Nx=-100.0)
        res = _solve_load_state(mesh, lam, load)
        N, _M = res.equivalent_resultants()
        assert N[0] == pytest.approx(load.Nx, rel=self.TOL)

    def test_biaxial_state_recovers_both_components(self):
        mesh, lam = _flat(CROSS, 2)
        load = LoadState(Nx=-100.0, Ny=-40.0)
        res = _solve_load_state(mesh, lam, load)
        N, _M = res.equivalent_resultants()
        assert N[0] == pytest.approx(load.Nx, rel=self.TOL)
        assert N[1] == pytest.approx(load.Ny, rel=self.TOL)

    def test_symmetric_laminate_under_membrane_load_has_no_moment(self):
        """The lever-arm guard.

        Dropping ``z`` from the moment integrand turns ``M`` into a second
        copy of ``N`` — dimensionally wrong and silently plausible.  A
        symmetric laminate under pure membrane load has zero moment, so a
        moment that has quietly become a force is glaring.
        """
        mesh, lam = _flat(CROSS, 2)
        load = LoadState(Nx=-100.0)
        res = _solve_load_state(mesh, lam, load)
        N, M = res.equivalent_resultants()
        h = len(CROSS) * PLY_T
        # |M| must be far below the |N|*h/2 it would reach if the lever arm
        # were missing or mis-scaled.
        assert abs(M[0]) < 0.02 * abs(N[0]) * h

    def test_scales_linearly_with_the_applied_load(self):
        mesh, lam = _flat(CROSS, 2)
        one = _solve_load_state(mesh, lam, LoadState(Nx=-100.0))
        two = _solve_load_state(mesh, lam, LoadState(Nx=-200.0))
        assert two.equivalent_resultants()[0][0] == pytest.approx(
            2.0 * one.equivalent_resultants()[0][0], rel=1e-6
        )


# --------------------------------------------------------------------------- #
# interlaminar_stresses
# --------------------------------------------------------------------------- #


class TestInterlaminarStresses:
    """The reported value must be the peak that is actually in the field.

    The previous implementation averaged over *all* Gauss points of *all*
    elements in the two adjoining plies — a domain average of a field
    whose mean is ~0 by equilibrium.  It reported exactly 0.0000 MPa on a
    flat laminate whose free-edge peak is ~25 MPa, and under-reported by
    26x on a wrinkled one.
    """

    @staticmethod
    def _case(wrinkled):
        lam = Laminate.from_angles(QI, _material(), ply_thickness=PLY_T)
        wc = None
        if wrinkled:
            wc = WrinkleConfiguration.from_morphology_name(
                "graded",
                GaussianSinusoidal(
                    amplitude=0.366, wavelength=10.0, width=8.0, center=24.0,
                ),
                interface1=3, interface2=4,
            )
        mesh = WrinkleMesh(
            laminate=lam, wrinkle_config=wc,
            Lx=48.0, Ly=6.0, nx=48, ny=3, nz_per_ply=1,
        ).generate()
        res = StaticSolver(mesh, lam).solve(
            BoundaryHandler.compression_bcs(mesh, applied_strain=-0.01),
            solver="direct",
        )
        return mesh, res

    @staticmethod
    def _field_values(mesh, res, interface, component):
        """Every value of one component in the two adjoining ply layers."""
        below = mesh.elements_in_ply(interface)
        above = mesh.elements_in_ply(interface + 1)
        return np.concatenate([
            res.stress_global[below][:, :, component].ravel(),
            res.stress_global[above][:, :, component].ravel(),
        ])

    @pytest.mark.parametrize("wrinkled", [False, True],
                             ids=["flat", "wrinkled"])
    def test_reported_value_is_peak_scale_and_real(self, wrinkled):
        """Two independent facts, neither restating the reduction.

        The reported number must (a) actually occur in the field — not be
        an artefact of averaging — and (b) be of peak magnitude, not of
        plane-average magnitude.  The old implementation failed (b) by
        26x on the wrinkled case and by infinity on the flat one, where
        it returned exactly zero.
        """
        mesh, res = self._case(wrinkled)
        s33, _t13, _t23 = res.interlaminar_stresses()
        k = 3
        field = self._field_values(mesh, res, k, 2)
        peak = np.abs(field).max()
        assert peak > 1.0, "case must develop a real interlaminar stress"

        # (a) it is a value the field actually takes
        assert np.isclose(np.abs(field - s33[k]).min(), 0.0, atol=1e-9)
        # (b) peak-scale, not plane-average-scale
        assert abs(s33[k]) > 0.5 * peak
        assert abs(s33[k]) > 10.0 * abs(field.mean())

    def test_flat_laminate_is_not_reported_as_zero(self):
        """The specific symptom of averaging a self-equilibrating field."""
        _mesh, res = self._case(wrinkled=False)
        s33, _t13, _t23 = res.interlaminar_stresses()
        assert np.abs(s33).max() > 1.0

    def test_shear_components_read_their_own_voigt_slots(self):
        """Swapping the ``tau_13``/``tau_23`` slots was undetectable.

        Rather than recompute the reduction, this asserts that each
        reported value is drawn from *its own* component's field.  Under a
        swap the reported ``tau_13`` would be a ``tau_23`` value, which
        generically does not occur in the ``tau_13`` field at all.
        """
        mesh, res = self._case(wrinkled=True)
        _s33, t13, t23 = res.interlaminar_stresses()
        k = 3
        f13 = self._field_values(mesh, res, k, 4)   # Voigt slot 4 = tau_13
        f23 = self._field_values(mesh, res, k, 3)   # Voigt slot 3 = tau_23
        assert abs(t13[k] - t23[k]) > 1e-6, "components must be distinct here"
        assert np.abs(f13 - t13[k]).min() < 1e-9
        assert np.abs(f23 - t23[k]).min() < 1e-9
        # ...and each is peak-scale within its own component.
        assert abs(t13[k]) > 0.5 * np.abs(f13).max()
        assert abs(t23[k]) > 0.5 * np.abs(f23).max()

    def test_shapes_are_per_interface(self):
        _mesh, res = self._case(wrinkled=True)
        n_interfaces = len(QI) - 1
        for arr in res.interlaminar_stresses():
            assert arr.shape == (n_interfaces,)
            assert np.all(np.isfinite(arr))
