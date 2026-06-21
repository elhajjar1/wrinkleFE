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

import logging
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

logger = logging.getLogger(__name__)


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
        crack_band: bool = False,
        Gc_fiber: float = 50.0,
        solver: str = "direct",
        verbose: bool = False,
    ) -> None:
        self.mesh = mesh
        self.laminate = laminate
        self.applied_strain = float(applied_strain)
        self.n_increments = int(n_increments)
        self.max_equilibrium_iters = int(max_equilibrium_iters)
        self.fi_threshold = float(fi_threshold)
        # Crack-band regularization (D.1): when enabled, the dominant
        # fibre-compression (kink) mode softens gradually with a slope
        # scaled by the element size so the energy dissipated per unit
        # crack area equals the fibre-kink fracture energy ``Gc_fiber``
        # (N/mm) regardless of mesh — making the predicted strength
        # mesh-objective and replacing the arbitrary ``residual_factor`` +
        # ``nx`` choices with one physical material parameter (Bažant-Oh).
        self.crack_band = bool(crack_band)
        self.Gc_fiber = float(Gc_fiber)
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

        # Crack-band characteristic length per element: the kink crack for
        # fibre compression is a band roughly normal to the fibre (x) axis,
        # so it advances along x and the fracture energy is smeared over the
        # element's x-extent.  Damage state is an irreversible per-element
        # fibre-mode scalar.
        char_len = (
            self._element_x_extent() if self.crack_band else None
        )
        # Monotonic max-failure-index history per element (the crack-band
        # loading variable); 1.0 = elastic, > 1 = damaging.
        r_hist = np.ones(mesh.n_elements) if self.crack_band else None
        # Per-element matrix/shear modes already ply-discounted (crack-band
        # path only regularises the fibre mode).
        self._matrix_degraded: dict[int, set[str]] = {}

        degraded_families: dict[int, set[str]] = {}
        history: list[tuple[float, float]] = []
        # Per-increment pre-degradation (elastic) state for the
        # increment-robust peak: (|strain|, elastic carried stress, global
        # max failure index of the first solve at that strain).
        elastic: list[tuple[float, float, float]] = []
        converged = True

        def _sigma(field) -> float:
            u = field.displacement.ravel()
            K = static._K  # bare stiffness (keep_stiffness=True)
            assert K is not None
            reaction = float(np.sum((K @ u)[xmax_dofs]))
            return abs(reaction) / area if area > 0 else 0.0

        for i in range(1, self.n_increments + 1):
            eps = self.applied_strain * (i / self.n_increments)
            bcs = BoundaryHandler.compression_bcs(mesh, applied_strain=eps)

            field = None
            first_sigma = 0.0
            first_fi = 0.0
            for it in range(self.max_equilibrium_iters):
                field = static.solve(
                    bcs, solver=self.solver, keep_stiffness=True,
                    verbose=False,
                )
                if self.crack_band:
                    # Both arrays are allocated whenever crack_band is set.
                    assert r_hist is not None and char_len is not None
                    new_failures, max_fi = self._crack_band_update(
                        field, override, r_hist, char_len, static,
                    )
                else:
                    new_failures, max_fi = self._degrade_failed_elements(
                        field.stress_local, override, degraded_families,
                        static,
                    )
                if it == 0:
                    # Pre-(new-)degradation elastic state at this strain.
                    first_sigma = _sigma(field)
                    first_fi = max_fi
                if not new_failures:
                    break
            else:
                converged = False

            assert field is not None
            sigma = _sigma(field)
            history.append((eps, sigma))
            elastic.append((abs(eps), first_sigma, first_fi))

            if self.verbose:
                logger.info(
                    "  [inc %d/%d] eps=%.4f sigma=%.1f MPa  fi=%.3f  "
                    "n_failed=%d",
                    i, self.n_increments, eps, sigma, first_fi,
                    len(degraded_families),
                )

        peak, failed_at = self._resolve_peak(elastic, history)

        if self.crack_band:
            assert r_hist is not None
            n_failed = int((r_hist > 1.0).sum())
        else:
            n_failed = len(degraded_families)
        return ProgressiveDamageResult(
            peak_stress=peak,
            history=history,
            n_failed_elements=n_failed,
            failed_at_increment=failed_at,
            converged=converged,
        )

    # ------------------------------------------------------------------
    def _element_x_extent(self) -> np.ndarray:
        """Per-element extent in x (mm) — the crack-band length for the
        fibre-compression kink mode (band normal to the fibre/x axis)."""
        xe = self.mesh.nodes[self.mesh.elements][:, :, 0]  # (n_elem, 8)
        extent: np.ndarray = xe.max(axis=1) - xe.min(axis=1)
        return extent

    def _crack_band_update(
        self,
        field,
        override: dict[int, OrthotropicMaterial],
        r_hist: np.ndarray,
        char_len: np.ndarray,
        static: StaticSolver,
    ) -> tuple[int, float]:
        """Crack-band fibre-failure damage update (Bažant-Oh, FI-driven).

        Initiation and growth are driven by the **failure index** of the
        combined MaxStress + LaRC05 criteria, so the misalignment-shear
        coupling that triggers kinking at the wrinkle is included (driving
        by the axial strain alone would miss it — the wrinkle reduces the
        local fibre strain).  The monotonic loading variable is
        ``r = max history of FI`` (1 at initiation).  Damage follows a
        linear-softening law whose end point ``r_f`` is set by the
        *energy* per unit crack area:

            r_f = eps_f / eps_0 = 2 Gc_fiber E1 / (Xc^2 h),
            d(r) = 1 - (r_f - r) / (r (r_f - 1)),   1 <= r <= r_f,

        with ``h`` the element x-extent.  Because ``r_f`` scales as
        ``1/h``, a finer mesh softens over a longer ``r`` range so the
        dissipated energy stays ``Gc_fiber`` — i.e. mesh-objective.  Only
        the dominant fibre mode is crack-band-regularised; matrix/shear
        modes fall back to the (minor, for UD) ply-discount.  Returns
        ``(n_newly_damaged, global max FI)`` for the peak detection.
        """
        from dataclasses import replace as _replace

        mesh = self.mesh
        materials, eval_ply_ids, fiber_angles = self._effective_materials()
        with np.errstate(invalid="ignore", divide="ignore"):
            fi_fields, mode_fields = self._evaluator.evaluate_field(
                field.stress_local, materials, eval_ply_ids,
                fiber_angles=fiber_angles,
            )
        n_elem, n_gp = next(iter(fi_fields.values())).shape
        fi_stack = np.stack(list(fi_fields.values()), axis=0)
        mode_stack = np.stack(list(mode_fields.values()), axis=0)
        crit_arg = np.argmax(fi_stack, axis=0)
        ee, gg = np.indices((n_elem, n_gp))
        fi = fi_stack[crit_arg, ee, gg]
        modes = mode_stack[crit_arg, ee, gg]
        gp_max = np.argmax(fi, axis=1)
        elem_fi = fi[np.arange(n_elem), gp_max]

        n_new = 0
        for e in range(n_elem):
            fie = float(elem_fi[e])
            if fie <= r_hist[e]:
                continue  # no new loading beyond the stored maximum
            family = _mode_family(str(modes[e, gp_max[e]]))
            if "fiber" not in family:
                # Matrix/shear: keep the instantaneous ply-discount.
                seen = self._matrix_degraded.setdefault(e, set())
                if family in seen:
                    continue
                seen.add(family)
                base = mesh.element_material(e, self._ply_material(e))
                override[e] = self._discount.degrade(
                    base, FailureResult(index=fie, mode=family,
                                        reserve_factor=1.0 / max(fie, 1e-12),
                                        criterion_name="progressive"),
                )
                static.assembler.update_element(e)
                n_new += 1
                continue

            # Fibre mode: crack-band softening driven by r = max FI.
            r_hist[e] = fie
            ply_mat = self._ply_material(e)
            Xc, E1 = ply_mat.Xc, ply_mat.E1
            h = float(char_len[e])
            r_f = 2.0 * self.Gc_fiber * E1 / (Xc * Xc * h)
            r_f = max(r_f, 1.001)  # snap-back guard (h too large for Gf/Xc)
            if fie >= r_f:
                d = 0.999
            else:
                d = 1.0 - (r_f - fie) / (fie * (r_f - 1.0))
                d = min(max(d, 0.0), 0.999)
            base = mesh.element_material(e, ply_mat)
            override[e] = _replace(
                base,
                E1=max((1.0 - d) * base.E1, 1.0e-3 * base.E1),
                nu12=(1.0 - d) * base.nu12,
                nu13=(1.0 - d) * base.nu13,
                name=f"{base.name}_cb{d:.3f}",
            )
            static.assembler.update_element(e)
            n_new += 1

        return n_new, float(elem_fi.max()) if n_elem else 0.0

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_peak(
        elastic: list[tuple[float, float, float]],
        history: list[tuple[float, float]],
    ) -> tuple[float, int | None]:
        """Increment-robust ultimate strength.

        Combines two estimates and takes the larger:

        1. **First-failure load** — interpolating the elastic carried
           stress (``elastic`` = ``(|strain|, pre-degradation carried
           stress, global max failure index)`` per increment) to the
           strain where the global failure index first reaches 1.0.  For a
           uniform (pristine) UD coupon this is the exact compressive
           allowable ``Xc`` and is *independent of the increment count*,
           removing the load-step-alignment artefact that otherwise makes
           the pristine baseline — and hence every knockdown — swing with
           the number of increments.

        2. **Redistributed ultimate** — the largest *post-equilibrium*
           carried stress over the history.  A wrinkled coupon develops a
           stress concentration so one element reaches FI=1 early, but
           progressive degradation redistributes load and the laminate
           carries more before ultimate; this term captures that reserve.

        For pristine UD the first-failure term (``Xc``) dominates; for a
        wrinkled coupon the redistributed ultimate dominates.  The pair is
        consistent across mesh / increment changes because the pristine
        anchor is exact.

        Returns ``(peak_stress, failed_at_increment)``.
        """
        # First-failure load via FI=1 interpolation.
        first_failure = 0.0
        failed_at: int | None = None
        prev_fi = 0.0
        prev_sigma = 0.0
        for k, (_eps, sigma, fi) in enumerate(elastic):
            if fi >= 1.0:
                if k == 0 or fi == prev_fi:
                    first_failure = sigma
                else:
                    frac = (1.0 - prev_fi) / (fi - prev_fi)
                    first_failure = prev_sigma + frac * (sigma - prev_sigma)
                failed_at = k + 1
                break
            prev_fi, prev_sigma = fi, sigma

        # Redistributed ultimate = largest post-equilibrium carried stress.
        redistributed = max((s for _e, s in history), default=0.0)

        return max(first_failure, redistributed), failed_at

    # ------------------------------------------------------------------
    def _degrade_failed_elements(
        self,
        stress_local: np.ndarray,
        override: dict[int, OrthotropicMaterial],
        degraded_families: dict[int, set[str]],
        static: StaticSolver,
    ) -> tuple[int, float]:
        """Detect and degrade newly-failed elements.

        Returns the number of newly-degraded elements and the global
        maximum failure index of the current stress field.

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
        for e_np in candidates:
            e = int(e_np)
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

        global_max_fi = float(elem_max_fi.max()) if n_elem else 0.0
        return n_new, global_max_fi

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
        # Scale the fibre angle by the resin retention factor (0 at a
        # fibre-free resin centre, 1 in the bulk) so the kink-band path is
        # not double-counted in the pocket.
        if mesh.resin_blend is not None:
            elem_fiber *= (1.0 - mesh.resin_blend)
        elif mesh.resin_mask is not None:
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
