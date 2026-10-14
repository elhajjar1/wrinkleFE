"""Per-load-increment CZM damage record (issue #453).

The CZM path used to keep only the final interface state, so the load at
which a delamination started could not be read back. These tests pin the
record itself on the smallest model that exercises it -- two stiff cubes
with one cohesive element between them, opened in pure mode I -- where the
answers are known in closed form:

* initiation happens when the interface traction reaches ``sigma_max``,
  so on a 1 mm^2 interface the carried stress at onset is ``sigma_max``;
* the point separates (``d = 1``) once the opening passes
  ``delta_f = 2 GIc / sigma_max``, after which it carries nothing.

Plus the solver hook that feeds the record (``increment_callback``) and the
pure onset logic of :class:`CzmDamageHistory`.
"""

from __future__ import annotations

import numpy as np
import pytest

from wrinklefe.core.laminate import Laminate, Ply
from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.core.mesh import MeshData
from wrinklefe.elements.cohesive8 import (
    Cohesive8Element,
    CohesiveProperties,
    CohesiveState,
)
from wrinklefe.solver.assembler import GlobalAssembler
from wrinklefe.solver.boundary import BoundaryCondition, BoundaryHandler
from wrinklefe.solver.czm_history import (
    CzmDamageHistory,
    CzmDamageRecorder,
)
from wrinklefe.solver.nonlinear import NewtonRaphsonSolver

_PROPS = CohesiveProperties(
    K=1.0e5, sigma_max=50.0, tau_max=80.0,
    GIc=0.5, GIIc=1.5, eta_BK=1.45, beta=1.0,
)
_TOP = np.array([12, 13, 14, 15])
_BOTTOM = np.array([0, 1, 2, 3])


def _sandwich() -> tuple[GlobalAssembler, MeshData]:
    """Two 1 mm stiff cubes joined by one cohesive element at z = 1."""
    quad = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    nodes = np.vstack([
        np.column_stack([quad, np.full(4, z)]) for z in (0.0, 1.0, 1.0, 2.0)
    ])
    mesh = MeshData(
        nodes=nodes,
        elements=np.array([[0, 1, 2, 3, 4, 5, 6, 7],
                           [8, 9, 10, 11, 12, 13, 14, 15]], dtype=np.intp),
        ply_ids=np.zeros(2, dtype=np.intp),
        fiber_angles=np.zeros(16),
        ply_angles=np.zeros(2),
        nx=1, ny=1, nz=2,
    )
    mat = OrthotropicMaterial(
        E1=1.0e8, E2=1.0e8, E3=1.0e8, G12=4.0e7, G13=4.0e7, G23=4.0e7,
        nu12=0.25, nu13=0.25, nu23=0.25, name="iso-stiff",
    )
    ids = np.arange(4, 12, dtype=np.intp)
    coh = Cohesive8Element(mesh.nodes[ids], _PROPS, node_ids=ids, elem_id=0)
    assembler = GlobalAssembler(
        mesh, Laminate([Ply(material=mat, angle=0.0, thickness=1.0)]),
        cohesive_elements=[(0, coh)],
    )
    return assembler, mesh


def _mode_i_bcs(uz: float) -> list[BoundaryCondition]:
    return [
        BoundaryCondition(bc_type="fixed", node_ids=_BOTTOM, dofs=[0, 1, 2]),
        BoundaryCondition(bc_type="fixed", node_ids=_TOP, dofs=[0, 1]),
        BoundaryCondition(bc_type="displacement", node_ids=_TOP, dofs=[2],
                          value=float(uz)),
    ]


def _run_mode_i(n_inc: int, uz: float, callback=None):
    assembler, mesh = _sandwich()
    rec = CzmDamageRecorder(
        assembler, applied_strain=uz / 2.0, reaction_dofs=3 * _TOP + 2,
        area=1.0, n_increments=n_inc,
    )

    def both(inc, lam, u):
        rec(inc, lam, u)
        return callback(inc, lam, u) if callback is not None else None

    solver = NewtonRaphsonSolver(
        assembler, BoundaryHandler(mesh), _mode_i_bcs(uz),
        n_increments=n_inc, max_newton_iter=40, tol_residual=1e-8,
        tol_displacement=1e-10, increment_callback=both,
    )
    out = solver.solve()
    hist = rec.finish(completed=out["converged"] and not out["stopped_early"])
    return out, hist, rec


# ----------------------------------------------------------------------
# The solver hook
# ----------------------------------------------------------------------

class TestIncrementCallback:
    def test_called_once_per_converged_increment_with_the_load_fraction(self):
        seen = []
        out, _h, _r = _run_mode_i(
            8, 0.0004, callback=lambda i, lam, u: seen.append((i, lam)),
        )
        assert out["converged"] and not out["stopped_early"]
        assert seen == [(i, i / 8) for i in range(1, 9)]

    def test_a_truthy_return_stops_the_ramp(self):
        out, hist, _r = _run_mode_i(10, 0.0004, callback=lambda i, lam, u: i == 3)
        assert out["stopped_early"] is True
        assert out["converged"] is True
        assert out["increments_completed"] == 3
        assert hist.n_recorded == 3 and hist.completed is False

    def test_observing_leaves_the_solution_bit_identical(self):
        """The record must not perturb the solve it observes."""
        assembler, mesh = _sandwich()
        plain = NewtonRaphsonSolver(
            assembler, BoundaryHandler(mesh), _mode_i_bcs(0.03),
            n_increments=12, max_newton_iter=40,
        ).solve()
        assert plain["stopped_early"] is False
        assembler2, mesh2 = _sandwich()
        rec = CzmDamageRecorder(
            assembler2, applied_strain=0.015, reaction_dofs=3 * _TOP + 2,
            area=1.0, n_increments=12,
        )
        with_rec = NewtonRaphsonSolver(
            assembler2, BoundaryHandler(mesh2), _mode_i_bcs(0.03),
            n_increments=12, max_newton_iter=40, increment_callback=rec,
        ).solve()
        assert plain["converged"] and with_rec["converged"]
        assert np.array_equal(plain["displacement"], with_rec["displacement"])
        assert plain["iteration_counts"] == with_rec["iteration_counts"]
        assert rec.finish(completed=True).n_recorded == 12


# ----------------------------------------------------------------------
# The record on a closed-form mode-I problem
# ----------------------------------------------------------------------

_DELTA_0 = _PROPS.sigma_max / _PROPS.K
_DELTA_F = 2.0 * _PROPS.GIc / _PROPS.sigma_max


@pytest.fixture(scope="module")
def ramp():
    # 1.5 delta_f in 60 steps: initiation in the first increment
    # (delta_0 << delta_f / 40), separation at ~2/3 of the ramp.
    return _run_mode_i(60, 1.5 * _DELTA_F)


class TestModeIRecord:
    delta_0 = _DELTA_0
    delta_f = _DELTA_F

    def test_one_row_per_increment(self, ramp):
        out, hist, _r = ramp
        assert out["converged"]
        assert hist.n_recorded == 60 and hist.completed
        assert hist.total_area == pytest.approx(1.0)
        assert np.all(np.diff(hist.load_fraction) > 0)

    def test_damage_measures_are_monotone_and_bounded(self, ramp):
        _o, hist, _r = ramp
        assert np.all(np.diff(hist.max_damage) >= 0.0)
        assert np.all((hist.max_damage >= 0.0) & (hist.max_damage <= 1.0))
        assert np.all(hist.mean_damage <= hist.max_damage + 1e-15)
        assert np.all(hist.separated_area <= hist.damaged_area + 1e-12)
        assert hist.damaged_fraction[-1] == pytest.approx(1.0)

    def test_initiation_load_is_the_interface_strength(self):
        """Elastic ramp below initiation: the onset is extrapolated
        exactly, and on a 1 mm^2 interface it is sigma_max."""
        _o, hist, _r = _run_mode_i(3, 0.5 * self.delta_0)
        onset = hist.initiation()
        assert onset is not None and onset.extrapolated and onset.exact
        assert onset.nominal_stress == pytest.approx(_PROPS.sigma_max, rel=1e-6)

    def test_initiation_does_not_depend_on_the_increment_size(self):
        coarse = _run_mode_i(4, 3.0 * self.delta_0)[1].initiation()
        fine = _run_mode_i(23, 3.0 * self.delta_0)[1].initiation()
        assert coarse is not None and fine is not None
        assert coarse.exact and fine.exact and not coarse.extrapolated
        assert coarse.nominal_stress == pytest.approx(fine.nominal_stress,
                                                      rel=1e-9)
        assert coarse.nominal_stress == pytest.approx(_PROPS.sigma_max,
                                                      rel=1e-6)

    def test_separation_at_the_critical_opening(self, ramp):
        _o, hist, _r = ramp
        sep = hist.separation()
        assert sep is not None and not sep.exact
        # Stiff cubes: the opening is the applied top displacement, so the
        # first separated increment is the first with uz >= delta_f.
        uz = hist.load_fraction * 1.5 * self.delta_f
        expected = int(np.flatnonzero(uz >= self.delta_f * (1 - 1e-6))[0])
        assert sep.increment == expected + 1
        # A separated interface is traction-free.
        assert sep.nominal_stress == pytest.approx(0.0, abs=1e-3)
        assert hist.separated_area[expected - 1] == 0.0

    def test_carried_stress_peaks_at_the_strength(self, ramp):
        _o, hist, _r = ramp
        assert hist.nominal_stress.max() <= _PROPS.sigma_max * (1 + 1e-6)


# ----------------------------------------------------------------------
# The recorder's initiation index mirrors the cohesive law
# ----------------------------------------------------------------------

def test_initiation_index_crosses_one_exactly_where_the_law_damages():
    """Random mixed-mode openings on a virgin point: the recorder's
    ``delta_eff / delta_0`` exceeds 1 exactly when the element's own law
    starts to damage, so the onset it reports is the law's onset."""
    assembler, _mesh = _sandwich()
    rec = CzmDamageRecorder(
        assembler, applied_strain=1.0, reaction_dofs=3 * _TOP + 2,
        area=1.0, n_increments=1,
    )
    _gid, coh = assembler.cohesive_elements[0]
    rng = np.random.default_rng(453)
    n_gp = coh.n_gp
    checked = 0
    for _ in range(300):
        # A uniform jump of the top face, scaled around the initiation
        # opening so both sides of the threshold are sampled.
        jump = rng.normal(size=3) * rng.uniform(0.2, 2.0) * 1.2e-3
        u = np.zeros(assembler.mesh.n_dof)
        for nid in range(8, 16):
            u[3 * nid:3 * nid + 3] = jump
        index = rec.initiation_index(u, np.zeros(n_gp), -np.ones(n_gp))
        for g in range(n_gp):
            delta_local = coh._R_gp[g] @ coh._B_jump(g) @ u[
                GlobalAssembler._cohesive_dof_indices(coh)
            ]
            _t, _d, st = coh._law_local(delta_local, CohesiveState())
            if abs(index[g] - 1.0) < 1e-9:
                continue  # on the threshold: either answer is right
            assert (index[g] > 1.0) == (st.d > 0.0), (index[g], st.d)
            checked += 1
    assert checked > 1000


# ----------------------------------------------------------------------
# Onset logic on hand-built records
# ----------------------------------------------------------------------

def _hist(index, damage, separated=None, thermal=False) -> CzmDamageHistory:
    n = len(index)
    lam = np.arange(1, n + 1) / n
    return CzmDamageHistory(
        load_fraction=lam,
        applied_strain=-0.01 * lam,
        nominal_stress=500.0 * lam,
        max_damage=np.asarray(damage, dtype=float),
        mean_damage=np.asarray(damage, dtype=float) / 2,
        damaged_area=(np.asarray(damage) > 0).astype(float),
        separated_area=np.zeros(n) if separated is None
        else np.asarray(separated, dtype=float),
        initiation_index=np.asarray(index, dtype=float),
        total_area=1.0, n_increments=n, completed=True, thermal=thermal,
    )


class TestOnsetLogic:
    def test_scales_the_last_undamaged_increment(self):
        h = _hist([0.25, 0.5, 0.75, 1.1], [0, 0, 0, 0.1])
        on = h.initiation()
        assert on is not None
        assert on.increment == 4 and on.exact and not on.extrapolated
        # Increment 3 (lambda 0.75, 375 MPa) at index 0.75 -> 500 MPa.
        assert on.nominal_stress == pytest.approx(500.0)
        assert on.load_fraction == pytest.approx(1.0)
        assert on.applied_strain == pytest.approx(-0.01)

    def test_damaged_first_increment_is_flagged_inexact(self):
        h = _hist([2.0, 4.0], [0.3, 0.6])
        on = h.initiation()
        assert on is not None and on.increment == 1 and not on.exact
        assert on.nominal_stress == pytest.approx(125.0)

    def test_no_damage_extrapolates(self):
        h = _hist([0.1, 0.2], [0, 0])
        on = h.initiation()
        assert on is not None and on.extrapolated and on.increment is None
        assert on.nominal_stress == pytest.approx(2500.0)

    def test_a_thermal_load_makes_the_scaling_inexact(self):
        on = _hist([0.5, 1.5], [0, 0.1], thermal=True).initiation()
        assert on is not None and not on.exact

    def test_never_opened_and_empty_give_none(self):
        assert _hist([0.0, 0.0], [0, 0]).initiation() is None
        assert CzmDamageHistory().initiation() is None
        assert CzmDamageHistory().separation() is None

    def test_separation_is_the_first_separated_increment(self):
        h = _hist([2, 3, 4, 5], [0.2, 0.9, 1.0, 1.0], separated=[0, 0, 0.5, 1])
        sep = h.separation()
        assert sep is not None and sep.increment == 3
        assert sep.nominal_stress == pytest.approx(375.0)
        assert _hist([2, 3], [0.2, 0.9]).separation() is None

    def test_to_dict_is_json_ready(self):
        import json

        d = _hist([0.5, 1.5], [0, 0.1]).to_dict()
        assert json.loads(json.dumps(d))["max_damage"] == [0.0, 0.1]
