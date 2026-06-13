"""Progressive-damage load-stepping solver for wrinkled laminates.

The linear FE path reports a *first-ply* failure index at a single
applied strain — it cannot follow damage past initiation, so for
unidirectional compression (where the pristine LaRC05 index never
activates) it returns no knockdown at all.  This module carries the
solve to **ultimate load** by ramping the applied strain in increments
and, at each increment, iterating an equilibrium loop that:

1. assembles and solves the linear system with the *current* (possibly
   degraded) per-element materials,
2. recovers the element stresses and evaluates the LaRC05 failure index,
3. degrades every newly-failed element (ply-discount stiffness
   reduction, by failure-mode family) and re-solves,

until no further elements fail at that strain.  The nominal carried
stress ``sigma = R / A`` (reaction on the loaded face over its area) is
recorded per increment; its **peak over the loading history is the
predicted ultimate strength**.  The wrinkle/resin-pocket knockdown is
then ``sigma_peak(wrinkled) / sigma_peak(pristine)``.

This reuses :class:`~wrinklefe.solver.static.StaticSolver` for each
linear solve and the per-element material override on
:class:`~wrinklefe.core.mesh.MeshData` (so resin-pocket and damaged
materials compose) and degrades via
:class:`~wrinklefe.failure.progressive.PlyDiscount`.

References
----------
- Li, X. et al. (2024). Composites Science and Technology 256, 110762
  (3-D progressive damage of wrinkled UD glass/epoxy).
- Lapczyk, I. & Hurtado, J.A. (2007). Composites Part A 38(11), 2333.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wrinklefe.core.laminate import Laminate
from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.core.mesh import MeshData
from wrinklefe.failure.base import FailureResult
from wrinklefe.failure.evaluator import FailureEvaluator
from wrinklefe.failure.progressive import PlyDiscount
from wrinklefe.solver.boundary import BoundaryHandler
from wrinklefe.solver.static import StaticSolver


def _mode_family(mode: str) -> str:
    """Collapse a LaRC05 mode label to a degradation family.

    LaRC05 emits ``fiber_kinking`` for compressive fibre failure, which
    :class:`PlyDiscount` does not recognise; map any fibre/kink label to
    ``fiber_compression`` (degrades E1) and pass matrix/shear labels
    through unchanged.
    """
    m = mode.lower()
    if "fiber" in m or "kink" in m:
        # Preserve the tension/compression split where present, else
        # default to the compression family (degrades E1 either way).
        return "fiber_tension" if "tension" in m else "fiber_compression"
    return m


@dataclass
class ProgressiveDamageResult:
    """Outcome of a progressive-damage run.

    Attributes
    ----------
    peak_stress : float
        Predicted ultimate nominal stress (MPa), the peak of the carried
        stress over the loading history (always >= 0).
    history : list[tuple[float, float]]
        ``(applied_strain, nominal_stress)`` samples, one per increment.
    n_failed_elements : int
        Number of elements that received any degradation.
    failed_at_increment : int or None
        Index (1-based) of the increment at which the carried stress
        first dropped below the running peak (damage localisation); None
        if the stress rose monotonically (no peak reached in range).
    converged : bool
        Whether every increment's equilibrium loop reached a damage-free
        state within ``max_equilibrium_iters``.
    """

    peak_stress: float
    history: list[tuple[float, float]] = field(default_factory=list)
    n_failed_elements: int = 0
    failed_at_increment: int | None = None
    converged: bool = True


class ProgressiveDamageSolver:
    """Load-stepping ply-discount progressive-damage solver.

    Parameters
    ----------
    mesh : MeshData
        The (wrinkled, optionally resin-pocketed) hex8 mesh.  The solver
        mutates ``mesh.element_material_override`` as elements fail.
    laminate : Laminate
        Laminate definition (ply materials / angles).
    applied_strain : float
        Target nominal strain (negative for compression).  Should be
        large enough in magnitude to drive the laminate past its peak
        load (e.g. ``-0.025`` for UD glass with epsilon_f ~ 1.4 %).
    n_increments : int, optional
        Number of equal load steps from 0 to ``applied_strain``.
        Default 12.
    residual_factor : float, optional
        Ply-discount residual stiffness fraction for a failed element.
        Default 0.1.
    max_equilibrium_iters : int, optional
        Maximum re-solve iterations per increment.  Default 10.
    fi_threshold : float, optional
        Failure-index threshold for degradation.  Default 1.0.
    solver : str, optional
        Linear solver backend (``"direct"`` / ``"iterative"``).
        Default ``"direct"``.
    verbose : bool, optional
        Print per-increment progress.  Default False.
    """

    def __init__(
        self,
        mesh: MeshData,
        laminate: Laminate,
        *,
        applied_strain: float,
        n_increments: int = 12,
        residual_factor: float = 0.1,
        max_equilibrium_iters: int = 10,
        fi_threshold: float = 1.0,
        solver: str = "direct",
        verbose: bool = False,
    ) -> None:
        self.mesh = mesh
        self.laminate = laminate
        self.applied_strain = float(applied_strain)
        self.n_increments = int(n_increments)
        self.max_equilibrium_iters = int(max_equilibrium_iters)
        self.fi_threshold = float(fi_threshold)
        self.solver = solver
        self.verbose = verbose
        self._discount = PlyDiscount(residual_factor=residual_factor)
        # Combined criteria: LaRC05 captures fibre kinking and matrix /
        # inter-fibre failure at the wrinkle (misalignment-driven), while
        # MaxStress supplies the plain fibre-compression check |sigma_11|
        # >= Xc that LaRC05's kinking criterion does NOT trigger at zero
        # initial misalignment — without it the pristine UD baseline never
        # fails and the knockdown is undefined.
        from wrinklefe.failure.larc05 import LaRC05Criterion
        from wrinklefe.failure.max_stress import MaxStressCriterion
        self._evaluator = FailureEvaluator(
            [MaxStressCriterion(), LaRC05Criterion()]
        )

        if self.n_increments < 1:
            raise ValueError("n_increments must be >= 1")
        if self.applied_strain == 0.0:
            raise ValueError("applied_strain must be non-zero")

    # ------------------------------------------------------------------
    def solve(self) -> ProgressiveDamageResult:
        """Run the load-stepping damage analysis."""
        mesh = self.mesh
        if mesh.element_material_override is None:
            mesh.element_material_override = {}
        override = mesh.element_material_override

        # One StaticSolver instance; its assembler caches per-element Ke
        # which we refresh via update_element after each degradation.
        static = StaticSolver(mesh, self.laminate)

        # Loaded-face (x_max) x-DOFs and cross-section area for the
        # reaction-force / nominal-stress computation.
        xmax_nodes = mesh.nodes_on_face("x_max")
        xmax_dofs = 3 * xmax_nodes  # ux DOFs
        _Lx, Ly, Lz = mesh.domain_size
        area = Ly * Lz

        degraded_families: dict[int, set[str]] = {}
        history: list[tuple[float, float]] = []
        converged = True

        for i in range(1, self.n_increments + 1):
            eps = self.applied_strain * (i / self.n_increments)
            bcs = BoundaryHandler.compression_bcs(mesh, applied_strain=eps)

            field = None
            for _it in range(self.max_equilibrium_iters):
                field = static.solve(
                    bcs, solver=self.solver, keep_stiffness=True,
                    verbose=False,
                )
                new_failures = self._degrade_failed_elements(
                    field.stress_local, override, degraded_families, static,
                )
                if not new_failures:
                    break
            else:
                converged = False

            # Nominal carried stress = reaction on x_max / area.
            assert field is not None
            u = field.displacement.ravel()
            K = static._K  # bare stiffness (keep_stiffness=True)
            assert K is not None
            f_int = K @ u
            reaction = float(np.sum(f_int[xmax_dofs]))
            sigma = abs(reaction) / area if area > 0 else 0.0
            history.append((eps, sigma))

            if self.verbose:
                print(f"  [inc {i}/{self.n_increments}] eps={eps:.4f} "
                      f"sigma={sigma:.1f} MPa  "
                      f"n_failed={len(degraded_families)}")

        stresses = [s for _e, s in history]
        peak = max(stresses) if stresses else 0.0
        peak_idx = int(np.argmax(stresses)) if stresses else -1
        failed_at = (
            peak_idx + 1 if 0 <= peak_idx < len(stresses) - 1 else None
        )

        return ProgressiveDamageResult(
            peak_stress=peak,
            history=history,
            n_failed_elements=len(degraded_families),
            failed_at_increment=failed_at,
            converged=converged,
        )

    # ------------------------------------------------------------------
    def _degrade_failed_elements(
        self,
        stress_local: np.ndarray,
        override: dict[int, OrthotropicMaterial],
        degraded_families: dict[int, set[str]],
        static: StaticSolver,
    ) -> int:
        """Detect and degrade newly-failed elements; return the count.

        Evaluates LaRC05 on the current stress field using each element's
        *current* material (override -> resin -> ply), then for every
        element whose peak index reaches ``fi_threshold`` in a not-yet-
        degraded mode family applies a ply-discount degradation and
        refreshes that element's cached stiffness.
        """
        mesh = self.mesh
        materials, eval_ply_ids, fiber_angles = self._effective_materials()

        # Degraded materials can push the LaRC05 in-situ-strength formula
        # (sqrt of a toughness/stiffness ratio) negative on already-failed
        # elements; the resulting NaN is harmless (those elements have
        # already degraded) but noisy — silence it here.
        with np.errstate(invalid="ignore", divide="ignore"):
            fi_fields, mode_fields = self._evaluator.evaluate_field(
                stress_local, materials, eval_ply_ids,
                fiber_angles=fiber_angles,
            )
        # Element-and-GP-wise worst index across all criteria, with the
        # mode label taken from whichever criterion governs.
        n_elem, n_gp = next(iter(fi_fields.values())).shape
        fi_stack = np.stack(list(fi_fields.values()), axis=0)   # (n_crit, E, G)
        mode_stack = np.stack(list(mode_fields.values()), axis=0)
        crit_arg = np.argmax(fi_stack, axis=0)                  # (E, G)
        ee, gg = np.indices((n_elem, n_gp))
        fi = fi_stack[crit_arg, ee, gg]                         # (E, G)
        modes = mode_stack[crit_arg, ee, gg]                    # (E, G)

        # Per-element worst Gauss point.
        gp_max = np.argmax(fi, axis=1)
        elem_max_fi = fi[np.arange(n_elem), gp_max]
        candidates = np.flatnonzero(elem_max_fi >= self.fi_threshold)

        n_new = 0
        for e in candidates:
            e = int(e)
            family = _mode_family(str(modes[e, gp_max[e]]))
            seen = degraded_families.setdefault(e, set())
            if family in seen:
                continue
            seen.add(family)

            current = mesh.element_material(e, self._ply_material(e))
            degraded = self._discount.degrade(
                current,
                FailureResult(
                    index=float(elem_max_fi[e]), mode=family,
                    reserve_factor=1.0 / max(float(elem_max_fi[e]), 1e-12),
                    criterion_name="progressive",
                ),
            )
            override[e] = degraded
            static.assembler.update_element(e)
            n_new += 1

        return n_new

    # ------------------------------------------------------------------
    def _ply_material(self, elem_idx: int) -> OrthotropicMaterial:
        ply_idx = int(self.mesh.ply_ids[elem_idx])
        return self.laminate.plies[ply_idx].material

    def _effective_materials(
        self,
    ) -> tuple[list[OrthotropicMaterial], np.ndarray, np.ndarray]:
        """Build the per-element material list + ply-id index + fibre angles.

        Honours, per element, the degradation override, then the resin
        pocket, then the host ply.  Resin elements carry a zeroed fibre
        angle (no fibres to kink).  Returns a deduplicated material list
        and an index array suitable for ``FailureEvaluator.evaluate_field``.
        """
        mesh = self.mesh
        n_elem = mesh.n_elements
        override = mesh.element_material_override or {}

        elem_fiber = mesh.element_fiber_angles_array().copy()
        if mesh.resin_mask is not None:
            elem_fiber[mesh.resin_mask] = 0.0

        materials: list[OrthotropicMaterial] = []
        index_of: dict[int, int] = {}
        eval_ply_ids = np.empty(n_elem, dtype=int)
        for e in range(n_elem):
            mat = override.get(e)
            if mat is None:
                mat = mesh.element_material(e, self._ply_material(e))
            key = id(mat)
            idx = index_of.get(key)
            if idx is None:
                idx = len(materials)
                index_of[key] = idx
                materials.append(mat)
            eval_ply_ids[e] = idx
        return materials, eval_ply_ids, elem_fiber
