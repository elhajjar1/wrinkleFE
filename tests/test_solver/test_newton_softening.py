"""Tests for the Newton-Raphson nonlinear solver on a softening 1-DOF problem.

The test problem is a single spring with a piecewise-linear traction-separation
law (bilinear damage model) — analogous to a mode-I cohesive element under
displacement control:

    T(d) = K * d                                       (0 <= d < d0)
    T(d) = sigma_max * (df - d) / (df - d0)            (d0 <= d <= df)
    T(d) = 0                                           (d >  df)

The softening branch has negative slope; a naive Newton-Raphson step can
overshoot, and a backtracking line search is the simplest way to recover.
"""

from __future__ import annotations

import numpy as np
import pytest

from wrinklefe.solver.nonlinear import NewtonRaphsonSolver

# ======================================================================
# 1-DOF mock assembler + boundary handler
# ======================================================================

class _Softening1DOFAssembler:
    """Minimal assembler-shaped object for a single-DOF softening spring.

    Mimics the public API the solver actually uses:
    - ``mesh.n_dof``  (via the ``mesh`` shim attribute)
    - ``assemble_stiffness()`` -> sparse 1x1 tangent
    - ``assemble_internal_force(u)`` -> 1-element internal force
    """

    def __init__(self, K: float, sigma_max: float, d0: float, df: float) -> None:
        self.K = K
        self.sigma_max = sigma_max
        self.d0 = d0
        self.df = df

        class _Mesh:
            n_dof = 1

        self.mesh = _Mesh()

    def _T_and_dT(self, d: float) -> tuple[float, float]:
        if d <= 0.0:
            return self.K * d, self.K
        if d < self.d0:
            return self.K * d, self.K
        if d <= self.df:
            slope = -self.sigma_max / (self.df - self.d0)
            T = self.sigma_max * (self.df - d) / (self.df - self.d0)
            return T, slope
        return 0.0, 0.0

    def assemble_stiffness(self):
        # Will be evaluated at the current u inside _newton_step via the
        # implicit assumption that the assembler tracks u; we expose
        # ``current_u`` for that purpose so the tangent matches F_int.
        from scipy import sparse

        d = float(self._current_u)
        _, dT = self._T_and_dT(d)
        return sparse.csc_matrix(np.array([[dT]]))

    def assemble_internal_force(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float).reshape(-1)
        self._current_u = float(u[0])
        T, _ = self._T_and_dT(self._current_u)
        return np.array([T])


class _Single1DOFBCHandler:
    """Applies a single prescribed displacement on DOF 0."""

    def __init__(self, u_target: float) -> None:
        self.u_target = float(u_target)

        class _Mesh:
            n_dof = 1

        self.mesh = _Mesh()

    def get_constrained_dofs(self, bcs):
        return {0: self.u_target}

    def get_force_dofs(self, bcs):
        return np.zeros(1)


# ======================================================================
# Tests
# ======================================================================

@pytest.fixture
def softening_params():
    return dict(K=1000.0, sigma_max=50.0, d0=0.05, df=0.5)


def test_softening_spring_converges(softening_params):
    """Drive the spring all the way through the softening branch in 20
    increments; Newton must converge at every increment."""
    asm = _Softening1DOFAssembler(**softening_params)
    bc = _Single1DOFBCHandler(u_target=2.0 * softening_params["df"])

    solver = NewtonRaphsonSolver(
        assembler=asm,
        bc_handler=bc,
        n_increments=20,
        max_newton_iter=30,
        tol_residual=1e-6,
        tol_displacement=1e-9,
        line_search=True,
    )

    out = solver.solve()
    assert out["converged"], (
        f"Newton failed: completed {out['increments_completed']}/20 "
        f"iterations={out['iteration_counts']}"
    )
    assert out["increments_completed"] == 20
    # Final displacement should hit the prescribed value to ~6 digits.
    assert np.isclose(
        out["displacement"][0], 2.0 * softening_params["df"], rtol=1e-4
    )


def test_line_search_helps_softening(softening_params):
    """Line search either reduces total Newton iterations OR succeeds on a
    case where the bare Newton diverges.  We pick increment counts that
    force a stiff jump straight onto the softening branch."""
    # Aggressive increment that lands on softening on iteration 1.
    n_inc = 4
    u_target = 1.2 * softening_params["df"]

    def _run(line_search: bool):
        asm = _Softening1DOFAssembler(**softening_params)
        bc = _Single1DOFBCHandler(u_target=u_target)
        solver = NewtonRaphsonSolver(
            assembler=asm,
            bc_handler=bc,
            n_increments=n_inc,
            max_newton_iter=40,
            tol_residual=1e-6,
            tol_displacement=1e-9,
            line_search=line_search,
        )
        return solver.solve()

    out_ls = _run(line_search=True)
    out_no = _run(line_search=False)

    assert out_ls["converged"], (
        "Line search variant must converge: "
        f"iters={out_ls['iteration_counts']}"
    )

    iters_ls = sum(out_ls["iteration_counts"])
    iters_no = sum(out_no["iteration_counts"])

    helped = (
        (not out_no["converged"])
        or (iters_ls <= iters_no)
    )
    assert helped, (
        f"Line search did not help: "
        f"ls converged={out_ls['converged']} iters={iters_ls}; "
        f"no-ls converged={out_no['converged']} iters={iters_no}"
    )


class _CombinedOnlySoftening1DOFAssembler:
    """Variant of :class:`_Softening1DOFAssembler` that ONLY exposes the
    combined ``assemble_residual_and_tangent`` API.

    A user-written assembler may reasonably skip the legacy two-method
    interface (``assemble_stiffness`` + ``assemble_internal_force``).
    The solver's line-search and displacement-verify paths must work in
    that case too — they should route through the combined API rather
    than calling :meth:`assemble_internal_force` directly.
    """

    def __init__(
        self, K: float, sigma_max: float, d0: float, df: float,
    ) -> None:
        self.K = K
        self.sigma_max = sigma_max
        self.d0 = d0
        self.df = df

        class _Mesh:
            n_dof = 1

        self.mesh = _Mesh()

    def _T_and_dT(self, d: float) -> tuple[float, float]:
        if d <= 0.0:
            return self.K * d, self.K
        if d < self.d0:
            return self.K * d, self.K
        if d <= self.df:
            slope = -self.sigma_max / (self.df - self.d0)
            T = self.sigma_max * (self.df - d) / (self.df - self.d0)
            return T, slope
        return 0.0, 0.0

    def assemble_residual_and_tangent(self, u):
        from scipy import sparse

        u = np.asarray(u, dtype=float).reshape(-1)
        d = float(u[0])
        T, dT = self._T_and_dT(d)
        K_t = sparse.csc_matrix(np.array([[dT]]))
        F_int = np.array([T])
        return K_t, F_int


def test_solver_works_with_combined_api_only_assembler(softening_params):
    """A user-written assembler implementing only
    ``assemble_residual_and_tangent`` (no separate
    ``assemble_internal_force``) must still drive the line search and
    the displacement-verify check.

    Without the fix the line search calls
    ``self.assembler.assemble_internal_force(...)`` directly and raises
    ``AttributeError`` on this assembler.
    """
    asm = _CombinedOnlySoftening1DOFAssembler(**softening_params)
    bc = _Single1DOFBCHandler(u_target=2.0 * softening_params["df"])

    solver = NewtonRaphsonSolver(
        assembler=asm,
        bc_handler=bc,
        n_increments=20,
        max_newton_iter=30,
        tol_residual=1e-6,
        tol_displacement=1e-9,
        line_search=True,
    )

    out = solver.solve()
    assert out["converged"], (
        f"Newton failed with combined-API-only assembler: "
        f"completed {out['increments_completed']}/20 "
        f"iters={out['iteration_counts']}"
    )
    assert out["increments_completed"] == 20
    assert np.isclose(
        out["displacement"][0], 2.0 * softening_params["df"], rtol=1e-4
    )
