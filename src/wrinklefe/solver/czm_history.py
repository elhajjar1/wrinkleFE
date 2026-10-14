"""Per-load-increment record of cohesive-zone (CZM) damage (issue #453).

The CZM path (:meth:`wrinklefe.analysis.WrinkleAnalysis._run_czm_path`)
ramps the applied strain through :class:`~wrinklefe.solver.nonlinear.NewtonRaphsonSolver`
but used to keep only the *final* interface state, so the load at which a
delamination started at the wrinkle could not be read back.
:class:`CzmDamageRecorder` is an increment callback for that solver: after
every converged increment it stores the carried nominal stress and a few
scalar damage measures in a :class:`CzmDamageHistory`.

Cost: one sparse mat-vec for the loaded-face reaction and one for the
interface openings, plus a pass over the committed Gauss-point states. That
is small next to a single Newton iteration (which evaluates the cohesive law
at every Gauss point and factorises the tangent), so the record is always
taken; it changes no numerical result.

Two onset definitions are offered:

* :meth:`CzmDamageHistory.initiation` -- the first Gauss point reaching its
  mixed-mode interlaminar strength (the bilinear law's own damage-initiation
  condition, ``delta_eff >= delta_0`` with the Benzeggagh-Kenane mode ratio).
  Below it the interface and the bulk are both linear, so the onset load is
  obtained *exactly* by scaling the last undamaged increment (no dependence
  on the increment size) when there is no thermal load.
* :meth:`CzmDamageHistory.separation` -- the first Gauss point that is fully
  separated (``d = 1``, traction-free): a delamination crack exists. Its load
  is resolved only to one increment and depends on how well the mesh
  resolves the cohesive zone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    from wrinklefe.solver.assembler import GlobalAssembler

logger = logging.getLogger(__name__)

#: Damage at or above this counts as a fully separated (traction-free)
#: point. The bilinear law sets ``d = 1.0`` exactly once the effective
#: opening passes ``delta_f``; the margin only absorbs rounding.
SEPARATED_DAMAGE = 1.0 - 1.0e-9


@dataclass(frozen=True)
class CzmOnset:
    """Load at which a CZM damage threshold was first met.

    Attributes
    ----------
    kind : str
        ``"initiation"`` or ``"separation"``.
    increment : int or None
        1-based index of the first recorded increment at or past the
        threshold; ``None`` when the threshold was not reached in the ramp
        and the value is an elastic extrapolation (``extrapolated``).
    load_fraction : float
        Fraction of the full applied strain at onset.
    applied_strain : float
        Signed nominal strain at onset.
    nominal_stress : float
        Carried nominal stress at onset (MPa, ``>= 0``).
    exact : bool
        ``True`` when the value follows from the linear pre-damage response
        (initiation scaled from an undamaged increment with no thermal
        load); ``False`` when it is resolved only to the increment size.
    extrapolated : bool
        ``True`` when no recorded increment reached the threshold and the
        value was extrapolated from the (linear, undamaged) last increment.
    """

    kind: str
    increment: int | None
    load_fraction: float
    applied_strain: float
    nominal_stress: float
    exact: bool
    extrapolated: bool = False


@dataclass
class CzmDamageHistory:
    """Per-increment cohesive damage record of one CZM solve.

    One entry per converged load increment, in load order; every array has
    that length. Areas are Gauss-point-weighted mid-surface areas of the
    cohesive interface(s), in mm^2.

    Attributes
    ----------
    load_fraction : np.ndarray
        Fraction ``lambda`` of the full applied strain.
    applied_strain : np.ndarray
        Signed nominal strain at the increment.
    nominal_stress : np.ndarray
        ``|reaction| / A`` on the loaded ``x_max`` face (MPa), the same
        measure the progressive-damage solver reports.
    max_damage : np.ndarray
        Largest Gauss-point damage ``d``.
    mean_damage : np.ndarray
        Area-weighted mean damage over all cohesive interfaces.
    damaged_area : np.ndarray
        Interface area with ``d > 0`` (past initiation).
    separated_area : np.ndarray
        Interface area with ``d = 1`` (traction-free).
    initiation_index : np.ndarray
        Largest ``delta_eff / delta_0`` over all Gauss points: the
        interface analogue of a failure index (``>= 1`` once some point has
        reached its mixed-mode interlaminar strength). Homogeneous of degree
        one in the load while no point is damaged.
    total_area : float
        Total cohesive interface area (mm^2).
    n_increments : int
        Increments the solve was asked for (the ramp length).
    completed : bool
        Whether every requested increment converged (``False`` when the
        Newton solve stopped early on a non-converged increment, or a
        callback stopped it).
    thermal : bool
        Whether a thermal load was present (then the response is affine,
        not linear, and :meth:`initiation` is no longer exact).
    """

    load_fraction: np.ndarray = field(default_factory=lambda: np.empty(0))
    applied_strain: np.ndarray = field(default_factory=lambda: np.empty(0))
    nominal_stress: np.ndarray = field(default_factory=lambda: np.empty(0))
    max_damage: np.ndarray = field(default_factory=lambda: np.empty(0))
    mean_damage: np.ndarray = field(default_factory=lambda: np.empty(0))
    damaged_area: np.ndarray = field(default_factory=lambda: np.empty(0))
    separated_area: np.ndarray = field(default_factory=lambda: np.empty(0))
    initiation_index: np.ndarray = field(default_factory=lambda: np.empty(0))
    total_area: float = 0.0
    n_increments: int = 0
    completed: bool = False
    thermal: bool = False

    # ------------------------------------------------------------------
    @property
    def n_recorded(self) -> int:
        """Number of converged increments recorded."""
        return int(self.load_fraction.size)

    @property
    def damaged_fraction(self) -> np.ndarray:
        """``damaged_area / total_area`` per increment."""
        if self.total_area <= 0.0:
            return np.zeros_like(self.damaged_area)
        return np.asarray(self.damaged_area / self.total_area)

    @property
    def separated_fraction(self) -> np.ndarray:
        """``separated_area / total_area`` per increment."""
        if self.total_area <= 0.0:
            return np.zeros_like(self.separated_area)
        return np.asarray(self.separated_area / self.total_area)

    # ------------------------------------------------------------------
    def initiation(self) -> CzmOnset | None:
        """Load at which the first interface point reaches its strength.

        While no Gauss point is damaged the whole model (linear bulk,
        undamaged penalty interface) responds linearly, and the initiation
        index is proportional to the load. The onset is therefore the last
        undamaged increment scaled by ``1 / initiation_index`` -- exact,
        whatever the increment size, when there is no thermal load. When
        the very first increment is already damaged the same scaling is
        applied to it (``exact=False``: damage there has already softened
        the response, so the value errs low). When no increment reached
        initiation the last one is extrapolated (``extrapolated=True``).

        Returns ``None`` when nothing was recorded or the interface never
        opened (``initiation_index`` zero throughout).
        """
        n = self.n_recorded
        if n == 0:
            return None
        idx = self.initiation_index
        reached = np.flatnonzero(self.max_damage > 0.0)
        first = int(reached[0]) if reached.size else None
        if first is None:
            base, exact, extrapolated = n - 1, not self.thermal, True
        elif first == 0:
            base, exact, extrapolated = 0, False, False
        else:
            base, exact, extrapolated = first - 1, not self.thermal, False
        scale_index = float(idx[base])
        if not np.isfinite(scale_index) or scale_index <= 0.0:
            return None
        s = 1.0 / scale_index
        if first == 0:
            # Already past initiation at the first increment: scaling down
            # from a damaged state only ever moves the estimate lower.
            s = min(s, 1.0)
        return CzmOnset(
            kind="initiation",
            increment=None if first is None else first + 1,
            load_fraction=float(self.load_fraction[base] * s),
            applied_strain=float(self.applied_strain[base] * s),
            nominal_stress=float(self.nominal_stress[base] * s),
            exact=exact,
            extrapolated=extrapolated,
        )

    def separation(self) -> CzmOnset | None:
        """Load at the first increment with a fully separated point.

        Resolved to one increment (the true onset lies between this
        increment and the previous one, so the value errs high by at most
        one load step). ``None`` when no point separated in the recorded
        ramp -- including when the Newton solve stopped first.
        """
        hit = np.flatnonzero(self.separated_area > 0.0)
        if hit.size == 0:
            return None
        k = int(hit[0])
        return CzmOnset(
            kind="separation",
            increment=k + 1,
            load_fraction=float(self.load_fraction[k]),
            applied_strain=float(self.applied_strain[k]),
            nominal_stress=float(self.nominal_stress[k]),
            exact=False,
        )

    def to_dict(self) -> dict:
        """JSON-friendly copy (arrays as lists)."""
        out: dict = {}
        for name in ("load_fraction", "applied_strain", "nominal_stress",
                     "max_damage", "mean_damage", "damaged_area",
                     "separated_area", "initiation_index"):
            out[name] = [float(v) for v in getattr(self, name)]
        out.update(total_area=float(self.total_area),
                   n_increments=int(self.n_increments),
                   completed=bool(self.completed),
                   thermal=bool(self.thermal))
        return out


class CzmDamageRecorder:
    """Increment callback that fills a :class:`CzmDamageHistory`.

    Pass an instance as ``increment_callback`` to
    :class:`~wrinklefe.solver.nonlinear.NewtonRaphsonSolver`; it reads the
    assembler's *committed* cohesive state after each converged increment.
    Call :meth:`finish` after the solve to stamp the completion flag.

    Parameters
    ----------
    assembler : GlobalAssembler
        The assembler driving the Newton solve (cohesive elements
        registered).
    applied_strain : float
        Full (``lambda = 1``) signed nominal strain of the ramp.
    reaction_dofs : np.ndarray
        Global DOF indices whose internal forces sum to the reaction on the
        loaded face (the ``x``-DOFs of the ``x_max`` face nodes). Cohesive
        node pairs on that face carry equal and opposite interface forces,
        so only the hex8 stiffness (and any thermal load) enters the sum.
    area : float
        Loaded-face area (mm^2) for the nominal stress.
    n_increments : int
        Length of the ramp, recorded on the history.
    """

    def __init__(
        self,
        assembler: GlobalAssembler,
        *,
        applied_strain: float,
        reaction_dofs: np.ndarray,
        area: float,
        n_increments: int,
    ) -> None:
        self._assembler = assembler
        self._applied_strain = float(applied_strain)
        self._area = float(area)
        dofs = np.asarray(reaction_dofs, dtype=np.intp)
        # Rows of the (cached, hex8-only) stiffness on the loaded face: the
        # reaction is their product with u, summed.
        K = assembler._assemble_hex8_stiffness().tocsr()
        self._K_face = K[dofs]
        self._thermal_face = 0.0
        self._thermal = assembler.delta_T != 0.0
        if self._thermal:
            self._thermal_face = float(
                np.sum(assembler.assemble_thermal_force()[dofs])
            )

        # Interface kinematics, stacked over every (element, Gauss point):
        # the local (n, s, t) jump is ``J @ u`` with J built once from each
        # element's jump operator and frame.
        coh = list(assembler.cohesive_elements)
        self._elem_ids = [gid for gid, _ in coh]
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        gp_area: list[float] = []
        props: list[tuple[float, float, float, float, float]] = []
        r = 0
        for _gid, c_elem in coh:
            edofs = assembler._cohesive_dof_indices(c_elem)
            p = c_elem.properties
            for g in range(c_elem.n_gp):
                op = c_elem._R_gp[g] @ c_elem._B_jump(g)  # (3, 24)
                ii, jj = np.nonzero(op)
                rows.append(r + ii)
                cols.append(edofs[jj])
                vals.append(op[ii, jj])
                gp_area.append(
                    float(c_elem._detJ_gp[g] * c_elem._gp_weights[g])
                )
                props.append((p.K, p.sigma_max, p.tau_max, p.eta_BK, p.beta))
                r += 3
        n_dof = assembler.mesh.n_dof
        if rows:
            self._J = sparse.csr_matrix(
                (np.concatenate(vals),
                 (np.concatenate(rows), np.concatenate(cols))),
                shape=(r, n_dof),
            )
        else:
            self._J = sparse.csr_matrix((0, n_dof))
        self._gp_area = np.asarray(gp_area, dtype=float)
        prop_arr = np.asarray(props, dtype=float).reshape(-1, 5)
        self._K, self._sig, self._tau, self._eta, self._beta = prop_arr.T

        self._rows: list[tuple[float, ...]] = []
        self.history = CzmDamageHistory(
            total_area=float(self._gp_area.sum()),
            n_increments=int(n_increments),
            thermal=self._thermal,
        )

    # ------------------------------------------------------------------
    def initiation_index(self, u: np.ndarray, d: np.ndarray,
                         mode_ratio_init: np.ndarray) -> np.ndarray:
        """Per-Gauss-point ``delta_eff / delta_0`` at displacement ``u``.

        Mirrors the initiation test of
        :meth:`~wrinklefe.elements.cohesive8.Cohesive8Element._law_local`:
        Macaulay-bracketed normal opening, Benzeggagh-Kenane mixed-mode
        strength, and the mode ratio frozen at initiation where a point is
        already damaged (``mode_ratio_init >= 0``). Exceeds one exactly
        where the law starts (or continues) to damage.
        """
        delta = (self._J @ u).reshape(-1, 3)
        d_n_pos = np.maximum(delta[:, 0], 0.0)
        shear_sq = delta[:, 1] ** 2 + delta[:, 2] ** 2
        beta_sq = self._beta * self._beta
        delta_eff = np.sqrt(d_n_pos * d_n_pos + beta_sq * shear_sq)
        g_i = 0.5 * self._K * d_n_pos * d_n_pos
        g_ii = 0.5 * self._K * shear_sq * beta_sq
        g_tot = g_i + g_ii
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(g_tot > 0.0, g_ii / g_tot, 0.0)
        ratio = np.where(mode_ratio_init >= 0.0, mode_ratio_init, ratio)
        sig_sq = self._sig ** 2 + (self._tau ** 2 - self._sig ** 2) * (
            ratio ** self._eta
        )
        delta_0 = np.sqrt(np.maximum(sig_sq, 1.0e-30)) / self._K
        out: np.ndarray = delta_eff / delta_0
        return out

    def __call__(self, inc: int, load_fraction: float, u: np.ndarray) -> bool:
        """Record one converged increment. Never asks the solver to stop."""
        state = self._assembler.cohesive_state
        d = np.fromiter(
            (s.d for gid in self._elem_ids for s in state[gid]),
            dtype=float, count=self._gp_area.size,
        )
        mri = np.fromiter(
            (s.mode_ratio_init for gid in self._elem_ids for s in state[gid]),
            dtype=float, count=self._gp_area.size,
        )
        reaction = float(np.sum(self._K_face @ u)) - self._thermal_face
        sigma = abs(reaction) / self._area if self._area > 0.0 else 0.0
        if d.size:
            index = self.initiation_index(u, d, mri)
            total = float(self._gp_area.sum())
            row = (
                float(load_fraction),
                float(load_fraction) * self._applied_strain,
                sigma,
                float(d.max()),
                float(np.dot(d, self._gp_area) / total) if total > 0 else 0.0,
                float(self._gp_area[d > 0.0].sum()),
                float(self._gp_area[d >= SEPARATED_DAMAGE].sum()),
                float(index.max()),
            )
        else:
            row = (float(load_fraction),
                   float(load_fraction) * self._applied_strain,
                   sigma, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._rows.append(row)
        logger.debug(
            "CZM inc %d: lambda=%.4f sigma=%.2f MPa max_d=%.4f index=%.4f",
            inc, row[0], row[2], row[3], row[7],
        )
        return False

    def finish(self, *, completed: bool) -> CzmDamageHistory:
        """Freeze the recorded rows into :attr:`history` and return it."""
        cols = (np.asarray(self._rows, dtype=float).reshape(-1, 8).T
                if self._rows else np.empty((8, 0)))
        h = self.history
        (h.load_fraction, h.applied_strain, h.nominal_stress, h.max_damage,
         h.mean_damage, h.damaged_area, h.separated_area,
         h.initiation_index) = (np.ascontiguousarray(c) for c in cols)
        h.completed = bool(completed)
        return h
