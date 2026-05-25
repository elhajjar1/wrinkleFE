"""1-DOF snap-back regression for :class:`ArcLengthSolver`.

Newton-Raphson under load control can't traverse a limit point.  The
arc-length method picks up the burden by adding the load factor as an
unknown and imposing an arc-length constraint, so the solver can run
"backwards" in load past the snap-through.

Test problem
------------
A single-DOF system with internal force

    f(u) = u + 0.5 u^2 - 0.1 u^3

and reference external load ``F_ref = 1`` so the equilibrium is

    f(u) = lambda * F_ref.

``f`` rises monotonically until ``u ~= 4.14`` (limit point) then
decreases.  Standard Newton under load control above ``lambda_max =
f(4.14)`` has no solution; under displacement control it works (because
``u -> lambda(u)`` is single-valued), but it can't tell you "what's the
solution at a given lambda > lambda_max".

Arc length under this problem should produce a ``lambda`` history that
rises, peaks, and decays — non-monotone — confirming traversal of the
limit point.
"""

from __future__ import annotations

import numpy as np

from wrinklefe.solver.arclength import ArcLengthSolver


# ----------------------------------------------------------------------
# Toy single-DOF assembler / BC handler
# ----------------------------------------------------------------------


class _OneDOFMesh:
    """Mesh shim exposing only ``n_dof = 1`` so the arc-length solver
    can derive its size without a full MeshData object."""

    n_dof = 1


class _OneDOFAssembler:
    """Stand-in for :class:`GlobalAssembler` with one nonlinear DOF.

    ``F_int(u) = u + 0.5 u^2 - 0.1 u^3``;
    ``K_t(u)   = 1 + u - 0.3 u^2``.
    """

    def __init__(self) -> None:
        self.mesh = _OneDOFMesh()
        # No history state on this problem — make commit/revert no-ops
        # (the arc-length solver only calls them through getattr).

    def assemble_residual_and_tangent(self, u):
        from scipy import sparse
        u_val = float(np.asarray(u).reshape(-1)[0])
        F = np.array(
            [u_val + 0.5 * u_val * u_val - 0.1 * u_val ** 3],
            dtype=float,
        )
        K_val = 1.0 + u_val - 0.3 * u_val * u_val
        # Guard the trivial degenerate case: if K hits zero the sparse
        # spsolve will fail with a singular-matrix error which the
        # solver translates into a step failure.  Add a tiny floor so
        # the predictor still has a finite direction near limit points.
        K = sparse.csc_matrix([[K_val]])
        return K, F

    def assemble_tangent(self, u):
        K, _ = self.assemble_residual_and_tangent(u)
        return K

    def assemble_internal_force(self, u):
        _, F = self.assemble_residual_and_tangent(u)
        return F


class _OneDOFBCHandler:
    """Reference load vector of length 1 with F_ref = 1.  No
    displacement constraints (pure load-controlled-style problem)."""

    def __init__(self) -> None:
        self.mesh = _OneDOFMesh()

    def get_force_dofs(self, bcs):  # noqa: ARG002 — BC list ignored
        return np.array([1.0], dtype=float)

    def get_constrained_dofs(self, bcs):  # noqa: ARG002 — BC list ignored
        return {}


# ----------------------------------------------------------------------
# Test
# ----------------------------------------------------------------------


def test_arclength_traverses_snapback():
    """Drive the 1-DOF cubic snap-through with ArcLengthSolver and
    assert the load-factor history is non-monotone."""
    assembler = _OneDOFAssembler()
    bc_handler = _OneDOFBCHandler()

    solver = ArcLengthSolver(
        assembler=assembler,
        bc_handler=bc_handler,
        boundary_conditions=[],
        n_arc_steps=30,
        arc_length=0.4,        # smaller than the limit-point span
        max_newton_iter=40,
        tol_residual=1e-8,
        tol_absolute=1e-12,
    )

    result = solver.solve(verbose=False)
    assert result["converged"], (
        f"arc length did not converge: steps_completed="
        f"{result['steps_completed']}/30, "
        f"final lambda={result['load_factor']}"
    )
    lam_hist = result["load_factor_history"]
    # Limit-point lambda for f(u) = u + 0.5 u^2 - 0.1 u^3 is at the
    # zero of f'(u) = 1 + u - 0.3 u^2:  u_lp = (1 + sqrt(2.2)) / 0.6
    # = 4.137; lambda_lp = f(u_lp) ~= 5.687.
    lam_max = float(lam_hist.max())
    assert lam_max > 5.0, (
        f"Did not climb close to the limit point: lam_max={lam_max:.3f}"
    )
    # Non-monotone => traversed the limit point and started descending.
    assert lam_hist[-1] < lam_max - 1e-3, (
        f"Lambda did not descend after peak: lam_max={lam_max:.4f}, "
        f"lam_final={lam_hist[-1]:.4f}; history={lam_hist.tolist()}"
    )
    # The classical sanity: the second half of the history should have
    # smaller mean lambda than the first half.
    half = len(lam_hist) // 2
    first_mean = float(np.mean(lam_hist[1:half + 1]))
    second_mean = float(np.mean(lam_hist[half + 1:]))
    assert second_mean < first_mean, (
        f"second-half mean lambda {second_mean:.4f} not below first-half "
        f"{first_mean:.4f}"
    )
