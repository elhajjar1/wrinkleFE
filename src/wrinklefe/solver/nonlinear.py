"""Newton-Raphson nonlinear static solver with displacement-controlled loading.

Solves the nonlinear equilibrium equation

.. math::

    R(u) = F_{int}(u) - F_{ext} = 0

via an outer load-increment loop and an inner Newton iteration that
linearises ``R`` about the current displacement ``u`` and solves

.. math::

    K_t(u) \\, \\Delta u = -R(u)

at each iteration.  Displacement boundary conditions are applied via the
same penalty method used by :class:`~wrinklefe.solver.static.StaticSolver`
(so that the tangent and residual stay consistent at every increment).
A simple backtracking line search controls the Newton step in the
softening regime where a full step can overshoot the descent direction.

This is intentionally a slim siblng to :class:`StaticSolver` — no
arc-length, no adaptive stepping, no callbacks.  It exists to drive
intrinsic-CZM problems where the global tangent loses positive
definiteness once damage starts accumulating.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

if TYPE_CHECKING:
    from wrinklefe.solver.assembler import GlobalAssembler
    from wrinklefe.solver.boundary import BoundaryCondition, BoundaryHandler


class NewtonRaphsonSolver:
    """Incremental Newton-Raphson solver for nonlinear static problems.

    Parameters
    ----------
    assembler : GlobalAssembler
        Provides ``assemble_stiffness()`` (consistent tangent) and
        ``assemble_internal_force(u)`` (internal force vector) for the
        current state.  The assembler may carry internal history state
        (e.g. cohesive-element damage) that the solver does not own; the
        solver only calls ``commit_state()`` / ``revert_state()`` if those
        methods exist on the assembler.
    bc_handler : BoundaryHandler
        Owns the boundary-condition list.  The full prescribed values are
        scaled by the current load factor; force/pressure BCs are scaled
        the same way.
    n_increments : int, optional
        Number of equal load increments from 0 to 1.  Default 10.
    max_newton_iter : int, optional
        Maximum Newton iterations per increment.  Default 25.
    tol_residual : float, optional
        Relative residual tolerance ``||R|| / max(||F_ext||, eps)``.
        Default 1e-4.
    tol_displacement : float, optional
        Relative displacement-increment tolerance
        ``||du|| / max(||u||, eps)``.  Default 1e-6.
    line_search : bool, optional
        Enable backtracking line search on the Newton step.  Default True.
    """

    def __init__(
        self,
        assembler: "GlobalAssembler",
        bc_handler: "BoundaryHandler",
        boundary_conditions: "list[BoundaryCondition] | None" = None,
        n_increments: int = 10,
        max_newton_iter: int = 25,
        tol_residual: float = 1e-4,
        tol_displacement: float = 1e-6,
        line_search: bool = True,
    ) -> None:
        self.assembler = assembler
        self.bc_handler = bc_handler
        self.boundary_conditions = boundary_conditions or []
        self.n_increments = int(n_increments)
        self.max_newton_iter = int(max_newton_iter)
        self.tol_residual = float(tol_residual)
        self.tol_displacement = float(tol_displacement)
        self.line_search = bool(line_search)

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def solve(self, verbose: bool = False) -> dict:
        """Run the incremental Newton-Raphson loop.

        Returns
        -------
        dict
            ``displacement`` : final converged displacement vector
            (shape ``(n_dof,)``).
            ``load_displacement`` : array of shape
            ``(n_increments_completed, 2)`` with columns ``[lambda, ||u||]``
            for plotting load-displacement curves.
            ``converged`` : ``True`` if every increment converged.
            ``increments_completed`` : number of increments completed.
            ``iteration_counts`` : list of Newton iterations used per
            increment (helpful for line-search comparison tests).
        """
        n_dof = self._n_dof()
        u = np.zeros(n_dof)
        load_displacement = []
        iteration_counts: list[int] = []
        converged_all = True
        completed = 0

        constrained_full = self.bc_handler.get_constrained_dofs(
            self.boundary_conditions
        )
        F_ext_full = self.bc_handler.get_force_dofs(self.boundary_conditions)

        for inc in range(1, self.n_increments + 1):
            lam = inc / self.n_increments
            constrained_inc = {
                dof: lam * val for dof, val in constrained_full.items()
            }
            F_ext_inc = lam * F_ext_full

            u_new, n_iter, ok = self._newton_step(
                u, F_ext_inc, constrained_inc, verbose=verbose, inc=inc
            )
            iteration_counts.append(n_iter)

            if not ok:
                converged_all = False
                if verbose:
                    print(f"  Increment {inc}: Newton failed to converge.")
                break

            u = u_new
            completed = inc
            load_displacement.append([lam, float(np.linalg.norm(u))])

            self._commit_state()

        return {
            "displacement": u,
            "load_displacement": np.asarray(load_displacement, dtype=float),
            "converged": converged_all,
            "increments_completed": completed,
            "iteration_counts": iteration_counts,
        }

    # ------------------------------------------------------------------
    # Inner Newton loop
    # ------------------------------------------------------------------

    def _newton_step(
        self,
        u_prev: np.ndarray,
        F_ext: np.ndarray,
        constrained_dofs: dict[int, float],
        verbose: bool,
        inc: int,
    ) -> tuple[np.ndarray, int, bool]:
        u = u_prev.copy()
        F_ext_norm = max(float(np.linalg.norm(F_ext)), 1.0)

        for it in range(1, self.max_newton_iter + 1):
            # Internal force is evaluated first so the assembler can cache
            # per-element trial state (e.g. damage in cohesive elements)
            # that ``assemble_stiffness`` then linearises about.  For
            # linear elements this is harmless: F_int = K @ u and the
            # tangent is identical to the linear stiffness.
            F_int = self.assembler.assemble_internal_force(u)
            K_t = self.assembler.assemble_stiffness()
            R = F_int - F_ext

            K_bc, R_bc = self._apply_bcs_to_system(
                K_t, R.copy(), u, constrained_dofs
            )

            res_norm = float(np.linalg.norm(R_bc))
            if verbose:
                print(
                    f"  inc {inc} iter {it}: |R|={res_norm:.3e} "
                    f"|F|={F_ext_norm:.3e}"
                )
            if res_norm / F_ext_norm < self.tol_residual and it > 1:
                return u, it - 1, True

            try:
                du = spla.spsolve(K_bc, -R_bc)
            except RuntimeError:
                return u, it, False

            if not np.all(np.isfinite(du)):
                return u, it, False

            alpha = 1.0
            if self.line_search:
                alpha = self._backtracking_line_search(
                    u, du, F_ext, constrained_dofs, res_norm
                )

            u = u + alpha * du

            du_norm = float(np.linalg.norm(alpha * du))
            u_norm = max(float(np.linalg.norm(u)), 1.0)
            if du_norm / u_norm < self.tol_displacement and it > 1:
                # Verify the residual at the new state.
                F_int_chk = self.assembler.assemble_internal_force(u)
                R_chk = F_int_chk - F_ext
                _, R_chk_bc = self._apply_bcs_to_system(
                    None, R_chk.copy(), u, constrained_dofs,
                    only_residual=True,
                )
                if (
                    float(np.linalg.norm(R_chk_bc)) / F_ext_norm
                    < self.tol_residual
                ):
                    return u, it, True

        return u, self.max_newton_iter, False

    # ------------------------------------------------------------------
    # Line search
    # ------------------------------------------------------------------

    def _backtracking_line_search(
        self,
        u: np.ndarray,
        du: np.ndarray,
        F_ext: np.ndarray,
        constrained_dofs: dict[int, float],
        res_norm: float,
        max_trials: int = 6,
        shrink: float = 0.5,
    ) -> float:
        """Simple backtracking: pick the largest ``alpha in {1, 0.5, ...}``
        that strictly reduces the residual norm at constrained-rows-zeroed
        free DOFs.  Falls back to ``alpha = shrink^max_trials`` if no trial
        improves on the current residual.
        """
        alpha = 1.0
        best_alpha = alpha
        best_norm = res_norm
        for _ in range(max_trials):
            u_trial = u + alpha * du
            F_int_trial = self.assembler.assemble_internal_force(u_trial)
            R_trial = F_int_trial - F_ext
            _, R_trial_bc = self._apply_bcs_to_system(
                None, R_trial.copy(), u_trial, constrained_dofs,
                only_residual=True,
            )
            trial_norm = float(np.linalg.norm(R_trial_bc))
            if trial_norm < best_norm:
                best_norm = trial_norm
                best_alpha = alpha
                # Accept the first (largest) alpha that improves.
                return best_alpha
            alpha *= shrink
        return best_alpha

    # ------------------------------------------------------------------
    # BC application — penalty method on tangent and residual rows
    # ------------------------------------------------------------------

    def _apply_bcs_to_system(
        self,
        K: sparse.csc_matrix | None,
        R: np.ndarray,
        u_current: np.ndarray,
        constrained_dofs: dict[int, float],
        only_residual: bool = False,
    ) -> tuple[sparse.csc_matrix | None, np.ndarray]:
        """Apply displacement BCs to the Newton system.

        For each constrained DOF *i* with target ``u_i``:

        - Tangent:  add penalty ``alpha`` to ``K[i, i]``.
        - Residual: set ``R[i] = alpha * (u_current[i] - u_i)`` so that the
          Newton update ``du[i]`` drives ``u[i]`` toward ``u_i``.
        """
        from wrinklefe.solver.boundary import _PENALTY_SCALE

        if not constrained_dofs:
            return K, R

        if K is not None:
            diag_max = float(np.abs(K.diagonal()).max()) if K.nnz else 1.0
        else:
            diag_max = 1.0
        alpha = _PENALTY_SCALE * max(diag_max, 1.0)

        dofs = np.fromiter(constrained_dofs.keys(), dtype=np.intp)
        vals = np.fromiter(constrained_dofs.values(), dtype=float)

        R[dofs] = alpha * (u_current[dofs] - vals)

        if K is not None and not only_residual:
            K = K.tolil()
            for d in dofs:
                K[d, d] = K[d, d] + alpha
            K = K.tocsc()

        return K, R

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _n_dof(self) -> int:
        if hasattr(self.assembler, "mesh"):
            return int(self.assembler.mesh.n_dof)
        if hasattr(self.bc_handler, "mesh"):
            return int(self.bc_handler.mesh.n_dof)
        raise RuntimeError("Cannot determine number of DOFs from inputs.")

    def _commit_state(self) -> None:
        commit = getattr(self.assembler, "commit_state", None)
        if callable(commit):
            commit()
