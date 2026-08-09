"""Convergence-failure diagnostics for :class:`NewtonRaphsonSolver` (#262).

A failed nonlinear (CZM) solve used to surface only ``converged: False``.
The solver now records a ``failure_diagnostics`` dict for the first failing
increment and derives a single actionable ``failure_hint`` string.  These
tests cover:

1. The hint mapping (``_failure_hint``) and the residual-trend classifier
   (``_classify_residual_trend``) — pure, fast, deterministic unit tests
   that an ``iteration_cap`` diagnosis differs from a ``diverged`` one and
   that each names the right tuning knob.
2. A converged solve is unchanged: ``failure_diagnostics`` /
   ``failure_hint`` are ``None`` and the displacement is bit-identical
   across repeated runs.
3. A deliberately under-incremented DCB (Mode-I) case fails and produces a
   populated ``failure_diagnostics`` with an increment index and a
   ``failure_reason`` from the enum (slow / integration).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from wrinklefe.solver.nonlinear import NewtonRaphsonSolver

# ======================================================================
# 1. Hint mapping + residual-trend classifier (fast unit tests)
# ======================================================================

_ITERATION_CAP_DIAG = {
    "increment": 12,
    "n_increments": 20,
    "load_fraction": 0.6,
    "n_iterations": 25,
    "final_phys_norm": 3.2e-3,
    "final_bc_violation": 1e-9,
    "final_du_norm": 4e-4,
    "residual_history": [1.0, 0.5, 0.2, 0.08, 0.03],
    "line_search_exhausted": False,
    "last_alpha": 1.0,
    "tangent_singular": False,
    "failure_reason": "iteration_cap",
}

_DIVERGED_DIAG = {
    "increment": 3,
    "n_increments": 20,
    "load_fraction": 0.15,
    "n_iterations": 7,
    "final_phys_norm": 9.9e3,
    "final_bc_violation": 2e-3,
    "final_du_norm": None,
    "residual_history": [1.0, 3.0, 12.0, 90.0, 9900.0],
    "line_search_exhausted": True,
    "last_alpha": 0.03125,
    "tangent_singular": False,
    "failure_reason": "diverged",
}


def test_hint_differs_between_iteration_cap_and_diverged():
    """Acceptance #2: the hint string must change between an
    iteration-cap failure and a divergence failure, and each must name the
    knob appropriate to that failure mode."""
    cap_hint = NewtonRaphsonSolver._failure_hint(_ITERATION_CAP_DIAG)
    div_hint = NewtonRaphsonSolver._failure_hint(_DIVERGED_DIAG)

    assert cap_hint != div_hint

    # Iteration-cap: point the user at the iteration cap / tolerance.
    assert "max_newton_iter" in cap_hint
    assert "still decreasing" in cap_hint
    # It should NOT tell them to reduce the applied strain (that is the
    # divergence remedy, not the cap remedy).
    assert "reduce the applied strain" not in cap_hint

    # Divergence: point the user at smaller steps / applied strain.
    assert "diverged" in div_hint.lower()
    assert "czm_n_load_increments" in div_hint
    assert "applied strain" in div_hint

    # Both should locate the failing increment (index / total / % load).
    assert "12/20" in cap_hint
    assert "3/20" in div_hint


def test_hint_tangent_singular_points_at_load_increments_and_roadmap():
    diag = dict(_DIVERGED_DIAG, failure_reason="tangent_singular",
                tangent_singular=True, increment=19, load_fraction=0.95)
    hint = NewtonRaphsonSolver._failure_hint(diag)
    assert "singular" in hint.lower()
    assert "czm_n_load_increments" in hint
    assert "arc-length" in hint
    assert "19/20" in hint


def test_hint_stagnated_points_at_load_increments_and_tol():
    diag = dict(_ITERATION_CAP_DIAG, failure_reason="stagnated")
    hint = NewtonRaphsonSolver._failure_hint(diag)
    assert "stagnated" in hint.lower()
    assert "czm_n_load_increments" in hint
    assert "czm_newton_tol" in hint


def test_hint_none_diag_is_generic_but_actionable():
    hint = NewtonRaphsonSolver._failure_hint(None)
    assert "czm_n_load_increments" in hint
    assert "czm_newton_tol" in hint


@pytest.mark.parametrize(
    ("history", "expected"),
    [
        ([1.0, 3.0, 9.0, 27.0], "diverged"),        # strictly increasing
        ([1.0, 0.99, 0.98, 0.97], "stagnated"),      # ~flat (>90% of start)
        ([1.0, 0.5, 0.2, 0.05], "iteration_cap"),    # still decreasing fast
        ([0.5], "iteration_cap"),                    # too few samples
        ([0.0, 0.0], "iteration_cap"),               # degenerate zero start
    ],
)
def test_classify_residual_trend(history, expected):
    assert NewtonRaphsonSolver._classify_residual_trend(history) == expected


# ======================================================================
# 2. A converged solve is unchanged (acceptance #3)
# ======================================================================

class _Linear1DOFAssembler:
    """Minimal linear spring assembler: F_int = K * u, constant tangent."""

    def __init__(self, K: float) -> None:
        self.K = float(K)

        class _Mesh:
            n_dof = 1

        self.mesh = _Mesh()

    def assemble_stiffness(self):
        from scipy import sparse

        return sparse.csc_matrix(np.array([[self.K]]))

    def assemble_internal_force(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float).reshape(-1)
        return np.array([self.K * float(u[0])])


class _Single1DOFBCHandler:
    def __init__(self, u_target: float) -> None:
        self.u_target = float(u_target)

        class _Mesh:
            n_dof = 1

        self.mesh = _Mesh()

    def get_constrained_dofs(self, bcs):
        return {0: self.u_target}

    def get_force_dofs(self, bcs):
        return np.zeros(1)


def _make_converging_solver() -> NewtonRaphsonSolver:
    return NewtonRaphsonSolver(
        assembler=_Linear1DOFAssembler(K=1000.0),
        bc_handler=_Single1DOFBCHandler(u_target=0.25),
        n_increments=5,
        max_newton_iter=25,
        tol_residual=1e-8,
        tol_displacement=1e-10,
    )


def test_converged_run_has_no_failure_record_and_is_bit_identical():
    out_a = _make_converging_solver().solve()
    out_b = _make_converging_solver().solve()

    assert out_a["converged"] is True
    assert out_a["failure_diagnostics"] is None
    assert out_a["failure_hint"] is None
    # The two new keys are additive; the converged displacement is
    # unaffected by the diagnostics plumbing (bit-identical run to run).
    assert out_a["displacement"].tolist() == out_b["displacement"].tolist()
    assert float(out_a["displacement"][0]) == pytest.approx(0.25, rel=1e-6)


# ======================================================================
# 3. Under-incremented DCB fails with populated diagnostics (acceptance #1)
# ======================================================================

_VALID_REASONS = {"tangent_singular", "diverged", "stagnated", "iteration_cap"}


def _load_dcb_module():
    """Load the DCB integration module by path (the ``tests`` tree is not
    an importable package, so a normal import will not resolve it)."""
    dcb_path = (
        Path(__file__).resolve().parents[1]
        / "integration"
        / "test_dcb_monotonic.py"
    )
    spec = importlib.util.spec_from_file_location("_dcb_monotonic", dcb_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
@pytest.mark.integration
def test_under_incremented_dcb_populates_failure_diagnostics():
    """A DCB Mode-I opening driven to the full 20 mm ramp in just two
    coarse increments, with a two-iteration Newton cap and a residual
    tolerance below the penalty method's reachable floor, cannot
    converge.  (Displacement control makes this benchmark otherwise very
    hard to break with load-step coarsening alone — at a large opening
    the cohesive elements fully fail and the residual settles — so we
    also starve the Newton loop of iterations.)  The solve must report
    ``converged=False`` plus a populated ``failure_diagnostics`` carrying
    the failing increment index and a classified reason, and a non-empty
    ``failure_hint``."""
    from wrinklefe.solver.boundary import BoundaryHandler

    dcb = _load_dcb_module()
    mesh, cohesive_elements = dcb._build_dcb_mesh()
    assembler = dcb._build_assembler(mesh, cohesive_elements)
    bc_handler = BoundaryHandler(mesh)

    solver = NewtonRaphsonSolver(
        assembler=assembler,
        bc_handler=bc_handler,
        # Full 20 mm opening in TWO steps — a 10 mm/step jump straight
        # through the cohesive law's softening branch.
        boundary_conditions=dcb._build_bcs(mesh, dcb.DELTA_MAX),
        n_increments=2,
        max_newton_iter=2,
        tol_residual=1e-30,
        tol_absolute=1e-30,
        tol_displacement=1e-30,
        line_search=False,
    )

    out = solver.solve()

    assert out["converged"] is False
    diag = out["failure_diagnostics"]
    assert diag is not None, "expected a populated failure_diagnostics record"
    assert diag["failure_reason"] in _VALID_REASONS
    assert 1 <= diag["increment"] <= 2
    assert diag["n_increments"] == 2
    assert 0.0 < diag["load_fraction"] <= 1.0
    assert isinstance(diag["residual_history"], list)
    assert diag["final_phys_norm"] >= 0.0

    hint = out["failure_hint"]
    assert isinstance(hint, str) and hint.strip()
    # The hint locates the failing increment and names a real knob.
    assert f"{diag['increment']}/2" in hint
