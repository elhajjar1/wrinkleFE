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

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

if TYPE_CHECKING:
    from wrinklefe.solver.assembler import GlobalAssembler
    from wrinklefe.solver.boundary import BoundaryCondition, BoundaryHandler


logger = logging.getLogger(__name__)


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
        Relative residual tolerance ``||R_phys|| / ||R_phys,0||`` on the
        free-DOF (non-constrained) part of the physical residual, where
        ``R_phys,0`` is the iter-1 physical residual norm.  Default 1e-4.
    tol_absolute : float, optional
        Absolute floor for the physical residual: an increment with
        ``||R_phys||`` below this value is considered already converged.
        Also bounds the relative test from below to handle the case of
        a vanishingly small ``R_phys,0``.  Default 1e-10.
    tol_displacement : float, optional
        Relative displacement-increment tolerance
        ``||du|| / max(||u||, 1)``, combined with a BC-violation check
        ``max(|u[d] - v|) < tol_displacement * max(|v|, 1.0)`` on
        constrained DOFs.  Default 1e-6.
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
        tol_absolute: float = 1e-10,
        tol_displacement: float = 1e-6,
        line_search: bool = True,
    ) -> None:
        self.assembler = assembler
        self.bc_handler = bc_handler
        self.boundary_conditions = boundary_conditions or []
        self.n_increments = int(n_increments)
        self.max_newton_iter = int(max_newton_iter)
        self.tol_residual = float(tol_residual)
        self.tol_absolute = float(tol_absolute)
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

        if len(constrained_full) == 0:
            raise ValueError(
                "NewtonRaphsonSolver requires at least one displacement "
                "boundary condition (constrained DOF) to avoid rigid-body "
                "singularity. "
                f"Got {len(self.boundary_conditions)} BCs with no "
                "constraints."
            )

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
                logger.warning(
                    "Increment %d: Newton failed to converge.", inc
                )
                break

            u = u_new
            completed = inc
            load_displacement.append([lam, float(np.linalg.norm(u))])

            self._commit_state()

        logger.info(
            "Newton-Raphson solve: %d/%d increments completed, converged=%s",
            completed, self.n_increments, converged_all,
        )

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
        from wrinklefe.solver.boundary import _PENALTY_SCALE

        u = u_prev.copy()
        phys_0: float | None = None
        phys_ref: float = self.tol_absolute

        # Build the constrained-DOF index/value arrays once per increment.
        # Used both for BC application and for the BC-violation half of
        # the convergence test.
        if constrained_dofs:
            dofs_arr = np.fromiter(
                constrained_dofs.keys(), dtype=np.intp,
                count=len(constrained_dofs),
            )
            vals_arr = np.fromiter(
                constrained_dofs.values(), dtype=float,
                count=len(constrained_dofs),
            )
            val_scale = float(max(np.max(np.abs(vals_arr)), 1.0))
        else:
            dofs_arr = np.empty(0, dtype=np.intp)
            vals_arr = np.empty(0, dtype=float)
            val_scale = 1.0

        # BC-violation tolerance: respect the user's tol_displacement but
        # not below the inherent penalty-method precision floor
        # ``~ 10 / _PENALTY_SCALE`` (the residual on a constrained DOF
        # decays as ~ K_phys / alpha = K_phys / (_PENALTY_SCALE * diag_max),
        # so absolute u-error is ~ val / _PENALTY_SCALE).
        penalty_floor = 10.0 / float(_PENALTY_SCALE)
        bc_tol = max(self.tol_displacement, penalty_floor) * val_scale

        diag_max: float = 1.0

        for it in range(1, self.max_newton_iter + 1):
            # One combined pass: assembler returns (K_t, F_int) computed
            # from the same trial-state evaluation of the cohesive law.
            K_t, F_int = self._assemble_residual_and_tangent(u)
            R = F_int - F_ext

            # Penalty scale derived once per increment from the first
            # iteration's tangent so trial-residual checks (line search
            # and displacement verify) see the same alpha as the Newton
            # solve itself.
            if it == 1:
                diag_max = max(
                    float(np.abs(K_t.diagonal()).max()) if K_t.nnz else 1.0,
                    1.0,
                )

            K_bc, R_bc = self._apply_bcs_to_system(
                K_t, R.copy(), u, constrained_dofs, diag_max,
            )

            # Physical residual: ignore the constrained rows so the
            # penalty term does not dominate the convergence check
            # for displacement-controlled problems.
            R_phys = R.copy()
            R_phys[dofs_arr] = 0.0
            phys_norm = float(np.linalg.norm(R_phys))
            # BC violation: how far the current u is from prescribed
            # values on constrained DOFs.
            if dofs_arr.size:
                bc_violation = float(np.max(np.abs(u[dofs_arr] - vals_arr)))
            else:
                bc_violation = 0.0
            res_norm = float(np.linalg.norm(R_bc))
            logger.debug(
                "inc %d iter %d: |R_phys|=%.3e |R_bc|=%.3e bc_viol=%.3e",
                inc, it, phys_norm, res_norm, bc_violation,
            )

            if it == 1:
                # Choosing the reference scale ``phys_ref`` for the
                # relative residual test is subtle under mixed loading:
                #
                # - The applied-force vector ``F_ext`` (on free DOFs)
                #   gives a clean physical scale for the imbalance the
                #   solver must drive to zero.  When present, it is the
                #   most physically meaningful reference.
                # - For pure displacement control there is no F_ext on
                #   free DOFs, so we back out the load scale from the
                #   penalty-augmented residual (res_norm ~ alpha *
                #   ||lam * val||; dividing by ``_PENALTY_SCALE``
                #   recovers diag_max * ||lam * val|| — the physical
                #   force scale of the displacement loading).
                #
                # Always taking the max of these two unconditionally
                # would spuriously loosen the tolerance under mixed
                # loading where the applied force is small (say 1e-3 N)
                # but the displacement-derived load scale is large
                # (say 1e6 N), making the relative tolerance dominated
                # by a quantity unrelated to the imbalance the user
                # cares about.  Using ``F_ext_free_norm`` as the
                # discriminator avoids that pitfall while remaining
                # robust to numerical noise in ``phys_0`` accumulated
                # across continuation increments.
                phys_0 = phys_norm
                F_ext_phys = F_ext.copy()
                F_ext_phys[dofs_arr] = 0.0
                F_ext_free_norm = float(np.linalg.norm(F_ext_phys))
                if F_ext_free_norm > self.tol_absolute:
                    phys_ref = max(F_ext_free_norm, self.tol_absolute)
                else:
                    load_scale = res_norm / float(_PENALTY_SCALE)
                    phys_ref = max(load_scale, self.tol_absolute)
                if (
                    phys_norm < self.tol_absolute
                    and bc_violation < bc_tol
                ):
                    return u, it, True
            else:
                assert phys_0 is not None
                if (
                    (
                        phys_norm < self.tol_residual * phys_ref
                        or phys_norm < self.tol_absolute
                    )
                    and bc_violation < bc_tol
                ):
                    return u, it, True

            try:
                du = spla.spsolve(K_bc, -R_bc)
            except RuntimeError:
                self._revert_state()
                return u, it, False

            if not np.all(np.isfinite(du)):
                self._revert_state()
                return u, it, False

            alpha = 1.0
            if self.line_search:
                alpha = self._backtracking_line_search(
                    u, du, F_ext, constrained_dofs, res_norm, diag_max,
                )

            u = u + alpha * du

            du_norm = float(np.linalg.norm(alpha * du))
            u_norm = max(float(np.linalg.norm(u)), 1.0)
            if du_norm / u_norm < self.tol_displacement:
                # Verify physical residual + BC violation at the new state.
                F_int_chk = self._assemble_internal_force(u)
                R_chk = F_int_chk - F_ext
                R_chk_phys = R_chk.copy()
                R_chk_phys[dofs_arr] = 0.0
                chk_phys_norm = float(np.linalg.norm(R_chk_phys))
                if dofs_arr.size:
                    chk_bc_viol = float(
                        np.max(np.abs(u[dofs_arr] - vals_arr))
                    )
                else:
                    chk_bc_viol = 0.0
                assert phys_0 is not None
                if (
                    (
                        chk_phys_norm < self.tol_residual * phys_ref
                        or chk_phys_norm < self.tol_absolute
                    )
                    and chk_bc_viol < bc_tol
                ):
                    return u, it, True

        self._revert_state()
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
        diag_max: float,
        max_trials: int = 6,
        shrink: float = 0.5,
    ) -> float:
        """Simple backtracking: pick the largest ``alpha in {1, 0.5, ...}``
        that strictly reduces the residual norm.  Falls back to the
        smallest TRIED alpha (``shrink ** (max_trials - 1)``) if no trial
        improves on the current residual — a tiny step is preferred over
        a full overshoot in the softening regime.
        """
        alpha = 1.0
        for i in range(max_trials):
            u_trial = u + alpha * du
            F_int_trial = self._assemble_internal_force(u_trial)
            R_trial = F_int_trial - F_ext
            _, R_trial_bc = self._apply_bcs_to_system(
                None, R_trial.copy(), u_trial, constrained_dofs,
                diag_max, only_residual=True,
            )
            trial_norm = float(np.linalg.norm(R_trial_bc))
            if trial_norm < res_norm:
                return alpha
            if i < max_trials - 1:
                alpha *= shrink
        # No trial reduced the residual; return the last (smallest)
        # tried alpha rather than an untested ``alpha * shrink``.
        return alpha

    # ------------------------------------------------------------------
    # BC application — penalty method on tangent and residual rows
    # ------------------------------------------------------------------

    def _apply_bcs_to_system(
        self,
        K: sparse.csc_matrix | None,
        R: np.ndarray,
        u_current: np.ndarray,
        constrained_dofs: dict[int, float],
        diag_max: float,
        only_residual: bool = False,
    ) -> tuple[sparse.csc_matrix | None, np.ndarray]:
        """Apply displacement BCs to the Newton system via the penalty
        method, matching :func:`wrinklefe.solver.boundary.apply_penalty_bcs`.

        For each constrained DOF *i* with target ``u_i``:

        - Tangent:  add penalty ``alpha = _PENALTY_SCALE * max(diag_max, 1)``
          to ``K[i, i]``.
        - Residual: add ``alpha * (u_current[i] - u_i)`` to ``R[i]`` so that
          the Newton update ``du[i]`` drives ``u[i]`` toward ``u_i`` without
          discarding the physical internal-force contribution at the
          constrained row.
        """
        from wrinklefe.solver.boundary import _PENALTY_SCALE

        if not constrained_dofs:
            return K, R

        alpha = _PENALTY_SCALE * max(diag_max, 1.0)

        dofs = np.fromiter(constrained_dofs.keys(), dtype=np.intp)
        vals = np.fromiter(constrained_dofs.values(), dtype=float)

        R[dofs] += alpha * (u_current[dofs] - vals)

        if K is not None and not only_residual:
            n_dof = K.shape[0]
            diag_data = np.zeros(n_dof, dtype=np.float64)
            diag_data[dofs] = alpha
            K = (K + sparse.diags(diag_data, 0, format="csc")).tocsc()

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

    def _revert_state(self) -> None:
        revert = getattr(self.assembler, "revert_state", None)
        if callable(revert):
            revert()

    def _assemble_tangent(self, u: np.ndarray) -> sparse.csc_matrix:
        tan = getattr(self.assembler, "assemble_tangent", None)
        if callable(tan):
            return tan(u)
        # Fall back to linear stiffness only when the assembler has no
        # cohesive elements registered; otherwise the linear path is
        # silently wrong for nonlinear CZM problems.
        cohesive = getattr(self.assembler, "cohesive_elements", None)
        if cohesive:
            raise RuntimeError(
                "Assembler has cohesive elements registered but does not "
                "implement assemble_tangent(u); the linear "
                "assemble_stiffness() is not appropriate for nonlinear "
                "CZM problems."
            )
        return self.assembler.assemble_stiffness()

    def _assemble_residual_and_tangent(
        self, u: np.ndarray
    ) -> tuple[sparse.csc_matrix, np.ndarray]:
        """Single-pass tangent + internal force.

        Prefers the assembler's combined ``assemble_residual_and_tangent``
        when available; falls back to two separate calls for assemblers
        that only expose the older two-method API.  As with
        :meth:`_assemble_tangent`, the linear-stiffness fall-back path
        is rejected for assemblers that have cohesive elements.
        """
        combined = getattr(
            self.assembler, "assemble_residual_and_tangent", None
        )
        if callable(combined):
            return combined(u)
        # This IS the fallback for assemblers without the combined API
        # — calling through :meth:`_assemble_internal_force` here would
        # infinitely recurse (that shim routes back through the combined
        # API when available), so we must call the assembler directly.
        F_int = self.assembler.assemble_internal_force(u)
        K_t = self._assemble_tangent(u)
        return K_t, F_int

    def _assemble_internal_force(self, u: np.ndarray) -> np.ndarray:
        """Shim: prefer direct API, fall back to combined-then-discard.

        Preserves performance for the built-in ``GlobalAssembler`` (which
        provides ``assemble_internal_force`` directly) while supporting
        user-written assemblers that only expose the combined
        ``assemble_residual_and_tangent`` API.
        """
        direct = getattr(self.assembler, "assemble_internal_force", None)
        if callable(direct):
            return direct(u)
        combined = getattr(
            self.assembler, "assemble_residual_and_tangent", None
        )
        if callable(combined):
            _K, F_int = combined(u)
            return F_int
        raise AttributeError(
            "Assembler must implement either assemble_internal_force(u) "
            "or assemble_residual_and_tangent(u); neither found."
        )
