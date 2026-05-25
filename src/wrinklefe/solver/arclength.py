"""Crisfield-style cylindrical arc-length continuation solver.

Solves the nonlinear equilibrium equation

.. math::

    R(u, \\lambda) = F_{int}(u) - \\lambda \\, F_{ext}^{ref} = 0

by treating the load factor ``lambda`` as an additional unknown and
imposing the cylindrical arc-length constraint

.. math::

    \\|\\Delta u\\|^2 = \\Delta s^2

at each arc step.  The constraint allows the solver to traverse limit
points (snap-through) and snap-back regions where displacement-
controlled or load-controlled Newton both fail.

This is a slim sibling to :class:`NewtonRaphsonSolver` — it owns the
same incremental loop structure but parametrises by arc length instead
of by the load factor.  Implementation choices follow Crisfield
(1981) for the spherical method specialised to psi = 0 (cylindrical):

.. warning::
    **Known limitation — not yet suitable for DCB-style snap-back
    problems with displacement BCs.**  The cylindrical norm
    ``||Delta u||`` is dominated by the penalty BC contributions at
    constrained DOFs, which makes the arc-length constraint behave
    essentially as ordinary displacement control — defeating the
    purpose on systems where snap-back must be traversed.

    On the canonical 1-DOF softening-spring snap-through problem
    (see ``tests/test_solver/test_arclength_snapback.py``) the solver
    works correctly and traverses the limit point.  On a real DCB
    with displacement-controlled BCs the solver stagnates after the
    first cohesive element fully damages.

    The proper fix is *indirect displacement control*: parametrise the
    arc by a single interior DOF (typically the crack-mouth opening
    displacement, CMOD) rather than the global ``||Delta u||``.  This
    is roughly a 300-LOC addition and not currently on the roadmap —
    wrinkleFE's wrinkle-knockdown use case (compression kink-band,
    tension curved-beam delamination) is captured by peak load under
    displacement control via :class:`NewtonRaphsonSolver`; snap-back
    is a post-peak numerical feature, not a knockdown driver.

- Inner Newton iteration uses the standard update u <- u + Delta u,
  lambda <- lambda + Delta lambda.
- The constraint reduces to a scalar quadratic in Delta lambda,
  solved by the closest-root strategy (Crisfield's "minimum residual
  angle" criterion: pick the root that keeps Delta u aligned with the
  predictor direction).
- If no real root exists for the quadratic (rare; usually means the
  arc length is too large), the step is reported as failed.

References
----------
Crisfield, M.A. (1981).  "A fast incremental/iterative solution
procedure that handles 'snap-through'."  Computers & Structures
13(1-3): 55-62.
Riks, E. (1979).  "An incremental approach to the solution of snapping
and buckling problems."  Int. J. Solids Struct. 15: 529-551.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

if TYPE_CHECKING:
    from wrinklefe.solver.assembler import GlobalAssembler
    from wrinklefe.solver.boundary import BoundaryCondition, BoundaryHandler


class ArcLengthSolver:
    """Cylindrical arc-length continuation solver.

    Parameters
    ----------
    assembler : GlobalAssembler
        Owns the global tangent + internal force.  May carry path-
        dependent state (e.g. cohesive damage); the solver calls
        ``commit_state()`` / ``revert_state()`` if those methods exist.
    bc_handler : BoundaryHandler
        Owns the boundary conditions.  Force/pressure BCs build the
        reference load vector ``F_ext_ref``; displacement BCs are
        applied via the same penalty method used by
        :class:`NewtonRaphsonSolver`.
    boundary_conditions : list[BoundaryCondition]
        The full BC list.  At each arc step the displacement BC values
        are scaled by the current load factor ``lambda``, exactly as
        the Newton solver does.
    n_arc_steps : int, optional
        Maximum number of arc steps.  Default 50.
    arc_length : float, optional
        Initial arc-length increment Delta s.  Default 0.1.
    max_newton_iter : int, optional
        Maximum Newton iterations per arc step.  Default 25.
    tol_residual : float, optional
        Relative residual tolerance on the physical (non-constrained)
        part of the out-of-balance vector.  Default 1e-4.
    tol_absolute : float, optional
        Absolute floor for the physical residual.  Default 1e-10.
    """

    def __init__(
        self,
        assembler: "GlobalAssembler",
        bc_handler: "BoundaryHandler",
        boundary_conditions: "list[BoundaryCondition] | None" = None,
        n_arc_steps: int = 50,
        arc_length: float = 0.1,
        max_newton_iter: int = 25,
        tol_residual: float = 1e-4,
        tol_absolute: float = 1e-10,
        adaptive: bool = True,
        max_halvings_per_step: int = 6,
    ) -> None:
        self.assembler = assembler
        self.bc_handler = bc_handler
        self.boundary_conditions = boundary_conditions or []
        self.n_arc_steps = int(n_arc_steps)
        self.arc_length = float(arc_length)
        self.max_newton_iter = int(max_newton_iter)
        self.tol_residual = float(tol_residual)
        self.tol_absolute = float(tol_absolute)
        self.adaptive = bool(adaptive)
        self.max_halvings_per_step = int(max_halvings_per_step)

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def solve(self, verbose: bool = False) -> dict:
        """Run the arc-length continuation loop.

        Returns
        -------
        dict
            ``displacement`` : final converged displacement vector.
            ``load_factor`` : final load factor ``lambda``.
            ``displacement_history`` : list of ``u`` at each arc step.
            ``load_factor_history`` : 1-D array of ``lambda`` values.
            ``converged`` : ``True`` if every arc step converged.
            ``steps_completed`` : number of arc steps completed.
        """
        n_dof = self._n_dof()
        u = np.zeros(n_dof)
        lam = 0.0
        # Reset the "previous total step" cache on every public solve()
        # entry — important when a solver instance is re-used.
        self._last_delta_u = None

        F_ext_ref = self.bc_handler.get_force_dofs(
            self.boundary_conditions
        )
        constrained_full = self.bc_handler.get_constrained_dofs(
            self.boundary_conditions
        )

        # Stash the constrained DOFs once — used both for the BC
        # penalty terms and for masking the physical residual.
        if constrained_full:
            dofs_arr = np.fromiter(
                constrained_full.keys(), dtype=np.intp,
                count=len(constrained_full),
            )
            vals_arr = np.fromiter(
                constrained_full.values(), dtype=float,
                count=len(constrained_full),
            )
        else:
            dofs_arr = np.empty(0, dtype=np.intp)
            vals_arr = np.empty(0, dtype=float)

        # Free-DOF mask: arc-length norm is computed over the free
        # DOFs only, so the prescribed-displacement BC contributions
        # don't dominate ``||Delta u||``.  Without this, the predictor
        # under penalty BCs has ``Delta u[constrained] = vals * lam``
        # which scales with the prescribed displacement magnitude, not
        # the physical structural response — turning the arc-length
        # constraint into a thin proxy for load control.
        free_mask = np.ones(n_dof, dtype=bool)
        if dofs_arr.size:
            free_mask[dofs_arr] = False
        self._free_mask = free_mask

        lam_history: list[float] = [lam]
        u_history: list[np.ndarray] = [u.copy()]
        converged_all = True
        steps_completed = 0

        # First-step tangent direction.  Negative-determinant detection
        # is needed to flip the predictor sign past limit points, so we
        # carry the previous "sign(Delta lambda * det)" between steps.
        prev_sign = 1.0

        ds = self.arc_length

        step = 0
        while step < self.n_arc_steps:
            step += 1
            tried_ds = ds
            n_halvings = 0
            attempt_ok = False
            while n_halvings <= self.max_halvings_per_step:
                # Revert state in case prior attempt updated it
                self._revert_state()
                try:
                    u_new, lam_new, n_iter, ok, sign_new = self._arc_step(
                        u, lam, F_ext_ref, dofs_arr, vals_arr,
                        constrained_full, tried_ds, prev_sign,
                        verbose, step,
                    )
                except RuntimeError:
                    ok = False
                if ok:
                    attempt_ok = True
                    prev_sign = sign_new
                    break
                if not self.adaptive:
                    break
                tried_ds *= 0.5
                n_halvings += 1

            if not attempt_ok:
                converged_all = False
                if verbose:
                    print(
                        f"  arc step {step}: bailed out after "
                        f"{n_halvings} halvings (last ds={tried_ds:.3e})"
                    )
                break

            u = u_new
            lam = lam_new
            steps_completed = step
            lam_history.append(lam)
            u_history.append(u.copy())

            self._commit_state()

            if verbose:
                print(
                    f"  arc step {step}: lambda={lam:+.4e} "
                    f"|u|={float(np.linalg.norm(u)):.3e} "
                    f"ds={tried_ds:.3e}"
                )

            # Re-arm step size for the next attempt: grow modestly on
            # success, but never above the user-supplied initial ds.
            if self.adaptive:
                if tried_ds < ds and n_halvings > 0:
                    # Still in the halved region — keep the smaller ds.
                    ds = tried_ds
                else:
                    ds = min(ds * 1.2, self.arc_length)

        return {
            "displacement": u,
            "load_factor": lam,
            "displacement_history": u_history,
            "load_factor_history": np.asarray(lam_history, dtype=float),
            "converged": converged_all,
            "steps_completed": steps_completed,
        }

    # ------------------------------------------------------------------
    # Per-step solve
    # ------------------------------------------------------------------

    def _arc_step(
        self,
        u_prev: np.ndarray,
        lam_prev: float,
        F_ext_ref: np.ndarray,
        dofs_arr: np.ndarray,
        vals_arr: np.ndarray,
        constrained_full: dict[int, float],
        ds: float,
        prev_sign: float,
        verbose: bool,
        step_idx: int,
    ) -> tuple[np.ndarray, float, int, bool, float]:
        """One arc step.

        Predictor: solves K_t du_F = F_ext_ref at (u_prev, lam_prev)
        and scales by ``Delta lambda_0 = +/- ds / ||du_F||``, with the
        sign chosen by ``prev_sign`` (tangent stiffness sign rule).

        Corrector: for each iteration, solve K_t du_R = -R and K_t du_F
        = F_ext_ref, then pick ``Delta lambda`` from the quadratic
        constraint ``||du + Delta lambda du_F + du_R||^2 = ds^2``.

        Returns ``(u_new, lam_new, n_iter, ok, prev_sign_new)``.
        """
        from wrinklefe.solver.boundary import _PENALTY_SCALE

        u = u_prev.copy()
        lam = float(lam_prev)

        # ----- predictor -----
        K_t = self._assemble_tangent(u)
        diag_max = max(
            float(np.abs(K_t.diagonal()).max()) if K_t.nnz else 1.0,
            1.0,
        )
        alpha = _PENALTY_SCALE * max(diag_max, 1.0)
        # Apply BC penalty to the predictor tangent.
        K_bc = self._apply_penalty_to_K(K_t, dofs_arr, alpha)
        F_eff = F_ext_ref.copy()
        if dofs_arr.size:
            F_eff[dofs_arr] += alpha * vals_arr
        try:
            du_F_pred = spla.spsolve(K_bc, F_eff)
        except RuntimeError as e:
            raise RuntimeError(f"predictor solve failed: {e}") from e
        if not np.all(np.isfinite(du_F_pred)):
            raise RuntimeError("predictor solve produced non-finite values")

        # Arc-length norm: free DOFs only.
        fm = self._free_mask
        if fm.any():
            norm_du_F = float(np.linalg.norm(du_F_pred[fm]))
        else:
            norm_du_F = float(np.linalg.norm(du_F_pred))
        if norm_du_F <= 0.0:
            raise RuntimeError(
                "predictor du_F has zero norm; reference load may be empty"
            )
        # Sign rule (Crisfield): pick the sign of the predictor that
        # keeps moving in the direction of the previous arc step.  This
        # is the dot-product test
        #
        #   sign(Delta lambda_0) = sign( Delta u_prev . du_F )
        #
        # (Bergan et al. 1978 "current stiffness parameter").  When the
        # tangent K_t changes sign past a limit point, du_F flips sign
        # too, so the dot product naturally reverses Delta lambda_0 and
        # we slide off the back of the load curve instead of bouncing.
        # On the very first arc step we have no Delta u_prev — fall
        # back to ``prev_sign`` (default +1 from the caller).
        delta_u_prev_step = getattr(self, "_last_delta_u", None)
        if delta_u_prev_step is None:
            sign_choice = prev_sign
        else:
            if fm.any():
                dotp = float(
                    np.dot(delta_u_prev_step[fm], du_F_pred[fm])
                )
            else:
                dotp = float(np.dot(delta_u_prev_step, du_F_pred))
            if abs(dotp) < 1.0e-30:
                sign_choice = prev_sign
            else:
                sign_choice = 1.0 if dotp > 0 else -1.0
        dlam_pred = sign_choice * ds / norm_du_F
        delta_u = dlam_pred * du_F_pred

        u = u + delta_u
        lam = lam + dlam_pred

        # ----- corrector -----
        phys_ref = max(
            float(np.linalg.norm(F_ext_ref)) * abs(lam),
            self.tol_absolute,
        )

        for it in range(1, self.max_newton_iter + 1):
            K_t, F_int = self._assemble_residual_and_tangent(u)
            # Residual w.r.t. F_ext = lambda * F_ext_ref.
            R = F_int - lam * F_ext_ref

            # Physical residual norm (mask constrained DOFs so the
            # penalty term doesn't dominate the convergence check).
            R_phys = R.copy()
            if dofs_arr.size:
                R_phys[dofs_arr] = 0.0
            phys_norm = float(np.linalg.norm(R_phys))

            if dofs_arr.size:
                bc_viol = float(np.max(np.abs(u[dofs_arr] - lam * vals_arr)))
            else:
                bc_viol = 0.0

            if verbose:
                print(
                    f"  arc {step_idx} it {it}: lam={lam:+.4e} "
                    f"|R_phys|={phys_norm:.3e} bc_viol={bc_viol:.3e}"
                )

            # Convergence: physical residual is small AND BC violation
            # is small (penalty term is finite, so the constrained
            # DOFs are never satisfied to round-off).
            if (
                phys_norm < self.tol_residual * phys_ref
                or phys_norm < self.tol_absolute
            ):
                # Take care to also check BC compliance — under arc
                # length the constrained values scale with lambda.
                penalty_floor = 10.0 / float(_PENALTY_SCALE)
                bc_tol = max(self.tol_residual, penalty_floor) * (
                    max(float(np.max(np.abs(lam * vals_arr))), 1.0)
                    if dofs_arr.size else 1.0
                )
                if bc_viol < bc_tol:
                    # Stash the total Delta u for the predictor sign
                    # rule on the next arc step.
                    self._last_delta_u = (u - u_prev).copy()
                    new_sign = (
                        1.0 if (lam - lam_prev) >= 0.0 else -1.0
                    )
                    return u, lam, it, True, new_sign

            # Apply BC penalty to the iteration tangent + residual.
            K_bc = self._apply_penalty_to_K(K_t, dofs_arr, alpha)
            R_bc = R.copy()
            if dofs_arr.size:
                R_bc[dofs_arr] += alpha * (u[dofs_arr] - lam * vals_arr)
            # Reference load vector with the BC penalty contribution.
            # The corrector solves K du_F = F_ext_ref + alpha * v (with
            # the user's vals_arr held fixed; constraint enforcement
            # happens via the BC residual).
            F_eff = F_ext_ref.copy()
            if dofs_arr.size:
                F_eff[dofs_arr] += alpha * vals_arr

            try:
                du_R = spla.spsolve(K_bc, -R_bc)
                du_F = spla.spsolve(K_bc, F_eff)
            except RuntimeError:
                return u, lam, it, False, prev_sign
            if not (np.all(np.isfinite(du_R))
                    and np.all(np.isfinite(du_F))):
                return u, lam, it, False, prev_sign

            # Cylindrical-arc constraint: ||Delta u + dlam * du_F||^2
            # = ds^2, where ``Delta u`` is the total displacement
            # increment from the start of this arc step.  Inner
            # products take the free-DOF mask (see :meth:`solve`).
            delta_u_cur = u - u_prev
            v = delta_u_cur + du_R
            if fm.any():
                v_f = v[fm]
                du_F_f = du_F[fm]
                a_coef = float(np.dot(du_F_f, du_F_f))
                b_coef = 2.0 * float(np.dot(v_f, du_F_f))
                c_coef = float(np.dot(v_f, v_f)) - ds * ds
            else:
                a_coef = float(np.dot(du_F, du_F))
                b_coef = 2.0 * float(np.dot(v, du_F))
                c_coef = float(np.dot(v, v)) - ds * ds
            disc = b_coef * b_coef - 4.0 * a_coef * c_coef
            if a_coef <= 0.0 or disc < 0.0:
                # No real intersection — arc length too large for the
                # current geometry.  Bail out so the caller can shrink.
                return u, lam, it, False, prev_sign

            sqrt_disc = float(np.sqrt(disc))
            dlam1 = (-b_coef + sqrt_disc) / (2.0 * a_coef)
            dlam2 = (-b_coef - sqrt_disc) / (2.0 * a_coef)

            # Crisfield's closest-root strategy: pick whichever Delta
            # lambda makes the resulting Delta u most aligned with the
            # current direction Delta u_cur.  Use the masked inner
            # product so the prescribed-displacement BC terms don't
            # dominate the cosine.
            u1 = v + dlam1 * du_F
            u2 = v + dlam2 * du_F
            if fm.any():
                cos1 = float(np.dot(u1[fm], delta_u_cur[fm]))
                cos2 = float(np.dot(u2[fm], delta_u_cur[fm]))
            else:
                cos1 = float(np.dot(u1, delta_u_cur))
                cos2 = float(np.dot(u2, delta_u_cur))
            if cos1 >= cos2:
                dlam_use = dlam1
                du_use = u1
            else:
                dlam_use = dlam2
                du_use = u2

            # Re-decompose: ``du_use = (u - u_prev) + d_u_step``, so
            # the actual update for ``u`` is ``du_use - (u - u_prev)``.
            d_u_step = du_use - (u - u_prev)
            u = u + d_u_step
            lam = lam + dlam_use

        # Out of Newton iterations.
        return u, lam, self.max_newton_iter, False, prev_sign

    # ------------------------------------------------------------------
    # Internal helpers
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
        cohesive = getattr(self.assembler, "cohesive_elements", None)
        if cohesive:
            raise RuntimeError(
                "Assembler has cohesive elements registered but does "
                "not implement assemble_tangent(u)."
            )
        return self.assembler.assemble_stiffness()

    def _assemble_residual_and_tangent(
        self, u: np.ndarray
    ) -> tuple[sparse.csc_matrix, np.ndarray]:
        combined = getattr(
            self.assembler, "assemble_residual_and_tangent", None
        )
        if callable(combined):
            return combined(u)
        F_int = self.assembler.assemble_internal_force(u)
        K_t = self._assemble_tangent(u)
        return K_t, F_int

    @staticmethod
    def _apply_penalty_to_K(
        K: sparse.csc_matrix,
        dofs_arr: np.ndarray,
        alpha: float,
    ) -> sparse.csc_matrix:
        """Return ``K`` with ``alpha`` added to the diagonal at the
        constrained DOFs, preserving CSC sparsity.
        """
        if dofs_arr.size == 0:
            return K
        n_dof = K.shape[0]
        diag_data = np.zeros(n_dof, dtype=np.float64)
        diag_data[dofs_arr] = alpha
        return (K + sparse.diags(diag_data, 0, format="csc")).tocsc()
