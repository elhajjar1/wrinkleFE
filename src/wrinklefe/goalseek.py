"""Inverse goal-seek: the largest defect that still meets a criterion.

The whole analysis surface is forward-only — given a wrinkle, predict a
knockdown.  The question a stress engineer is actually asked is the
inverse: *"manufacturing found waviness here; what is the largest
amplitude we can accept and still meet the allowable?"*  That number
goes straight into an inspection criterion or an NCR disposition, so
:func:`find_critical_value` automates the manual bisection loop it used
to take — and, critically, **refuses to answer** when the answer would
not be unique.

The search is deliberately conservative.  ``brentq`` converges *to* a
root, not to the side of it that satisfies the inequality, so the raw
root routinely returns a value whose knockdown sits a hair *below* the
target.  For an acceptance limit that is the wrong sign of wrong, so
the search backs the answer off to the safe side and verifies it with a
real forward evaluation: the reported ``critical_value`` satisfies
``objective >= target`` under an actual solve, not merely to within the
root tolerance.

Neither monotonicity nor uniqueness is assumed.  The search direction is
*measured* from a log-spaced scan of the resolved range, and when that
scan shows more than one sign change, a direction reversal, or a curve
too flat to resolve the target, the search returns a refusal status
(``non_monotonic`` / ``flat`` / ``no_crossing`` / ``target_unreachable``)
with an actionable, measurement-quoting message rather than an arbitrary
root.  Knockdown *is* monotonically decreasing in amplitude, but it is
**not** monotonic in wavelength for ``morphology='graded'`` — that curve
is U-shaped with an interior minimum, and a naive root-find there would
report a safe-sounding limit for a catastrophically failing geometry.

Example
-------
>>> from wrinklefe.analysis import AnalysisConfig
>>> from wrinklefe.goalseek import find_critical_value
>>> result = find_critical_value(
...     AnalysisConfig(amplitude=0.366, wavelength=16.0, width=12.0),
...     parameter="amplitude", target_knockdown=0.85,
... )                                                   # doctest: +SKIP
>>> print(result.summary())                             # doctest: +SKIP
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field, fields

import numpy as np
from scipy.optimize import brentq

from wrinklefe.analysis import (
    AnalysisConfig,
    AnalysisResults,
    WrinkleAnalysis,
    _replace_swept,
)
from wrinklefe.core.mesh import mesh_shear_diagnostics

logger = logging.getLogger(__name__)

__all__ = [
    "GoalSeekEvaluation",
    "CriticalValueResult",
    "find_critical_value",
]

#: Number of scan points when *scan_points* is left at its default.
DEFAULT_SCAN_POINTS: int = 9

#: Log-spacing is used when the bound ratio exceeds this.
LOG_SPACING_RATIO: float = 100.0

#: SciPy's hard floor on ``rtol`` (``4 * eps``); validation clamps to it.
MIN_RTOL: float = 4.0 * float(np.finfo(float).eps)

#: Absolute floor on the ``brentq`` parameter tolerance.
MIN_XTOL: float = 1e-12

#: Maximum geometric back-off budget, as a fraction of the root.
BACKOFF_BUDGET_RTOL_MULT: float = 1.0

#: Config features that only exist on the FE path.  Each one either
#: silently no-ops or raises under ``analytical_only=True``, so the
#: solve-path resolution ladder treats them as FE-forcing (issue #346
#: applies the same rule to ``wrinklefe analyze``).
FE_ONLY_FEATURES: tuple[str, ...] = (
    "enable_czm",
    "enable_resin_pocket",
    "enable_surface_resin_pockets",
    "enable_progressive_damage",
    "transverse_mode",
)

_EPS: float = float(np.finfo(float).eps)


# ======================================================================
# Objectives
# ======================================================================

def _obj_knockdown(results: AnalysisResults) -> float:
    """Combined analytical strength knockdown (every solve path)."""
    return float(results.analytical_knockdown)


def _obj_strength_MPa(results: AnalysisResults) -> float:
    """Predicted wrinkled strength in MPa (every solve path)."""
    return float(results.analytical_strength_MPa)


def _obj_modulus_knockdown(results: AnalysisResults) -> float:
    """Analytical stiffness knockdown (every solve path)."""
    return float(results.analytical_modulus_knockdown)


def _obj_onset_knockdown(results: AnalysisResults) -> float:
    """Delamination-onset knockdown; tension configurations only."""
    value = results.analytical_onset_knockdown
    if value is None:
        raise ValueError(
            "results.analytical_onset_knockdown is None — the "
            "delamination-onset knockdown is computed on the tension "
            "path only, and only when the material defines GIc and "
            "GIIc. Use loading='tension' with a material that carries "
            "both toughnesses, or pick objective='knockdown'."
        )
    return float(value)


def _obj_modulus_retention_global(results: AnalysisResults) -> float:
    """Global reaction-based modulus retention; FE path only."""
    if results.mesh is None:
        raise ValueError(
            "results.modulus_retention_global requires the FE path (it "
            "is measured from the global reaction response); this run "
            "was analytical-only, where the field keeps its 1.0 "
            "default. Pass analytical_only=False, or pick "
            "objective='modulus_knockdown' for the analytical "
            "stiffness knockdown."
        )
    if results.modulus_retention_global_failed:
        raise ValueError(
            "results.modulus_retention_global_failed is True — the "
            "pristine reference solve did not converge, so the field "
            "was forced to its 1.0 fallback and is not a usable "
            "objective. Fix the solve (see the logged reason) or pick "
            "objective='modulus_knockdown'."
        )
    return float(results.modulus_retention_global)


_OBJECTIVES: dict[str, Callable[[AnalysisResults], float]] = {
    "knockdown": _obj_knockdown,
    "strength_MPa": _obj_strength_MPa,
    "modulus_knockdown": _obj_modulus_knockdown,
    "onset_knockdown": _obj_onset_knockdown,
    "modulus_retention_global": _obj_modulus_retention_global,
}

#: Objectives that are deliberately not root-findable, with the reason.
_REFUSED_OBJECTIVES: dict[str, str] = {
    "modulus_retention": (
        "'modulus_retention' is the wrinkle-region-averaged strain "
        "measure whose own docstring records that it over-predicts "
        "stiffness loss; use 'modulus_retention_global' (FE path) or "
        "'modulus_knockdown' (analytical)."
    ),
    "retention_factors": (
        "'retention_factors' is hard-clamped to 1.0 by the failure "
        "evaluation, so as a function of a defect size it is a step, "
        "not a continuous curve, and has no root to find."
    ),
    "progressive_knockdown": (
        "'progressive_knockdown' is a constant 1.0 unless "
        "enable_progressive_damage=True, and with it enabled each "
        "forward evaluation costs ~30 load-stepped solves. Run "
        "WrinkleAnalysis.parametric_sweep over an explicit value list "
        "instead."
    ),
}

#: Objectives that are only defined on the FE path.
_FE_ONLY_OBJECTIVES: frozenset[str] = frozenset({"modulus_retention_global"})


# ======================================================================
# Result containers
# ======================================================================

@dataclass
class GoalSeekEvaluation:
    """One forward solve performed during the search."""

    index: int
    value: float
    """Swept-parameter value at this evaluation."""
    objective: float
    """Objective (knockdown, strength, ...) returned by the forward model."""
    knockdown: float
    """``analytical_knockdown`` at this point, recorded regardless of
    which objective drives the search."""
    strength_MPa: float
    residual: float
    """``objective - target``, in the objective's own units."""
    phase: str
    """``'scan'`` | ``'root'`` | ``'verify'`` | ``'backoff'``."""
    runtime_s: float


@dataclass
class CriticalValueResult:
    """Result of :func:`find_critical_value`."""

    parameter: str
    objective_name: str
    target: float
    """Target in the objective's units (knockdown factor, or MPa when
    *target_strength_MPa* was given)."""
    target_is_strength: bool
    direction: str
    """``'decreasing'`` or ``'increasing'`` — measured from the scan
    endpoints, never assumed from the parameter name."""
    status: str
    """``'converged'`` | ``'no_crossing'`` | ``'target_unreachable'`` |
    ``'non_monotonic'`` | ``'flat'``."""
    message: str
    """One-paragraph, actionable explanation. Always populated; it is the
    text the CLI prints for every non-``'converged'`` status."""

    critical_value: float | None
    """The **conservative** answer: for a decreasing objective, the
    largest *parameter* value that still satisfies ``objective >=
    target`` under a real forward evaluation; for an increasing one, the
    smallest such value. ``None`` unless *status* is ``'converged'``."""
    critical_value_root: float | None
    """The raw ``brentq`` root, before the safe-side back-off. Reported
    for transparency; ``critical_value`` is what a user must quote."""
    achieved_objective: float | None
    achieved_knockdown: float | None
    achieved_strength_MPa: float | None
    critical_config: AnalysisConfig | None
    """``base_config`` cloned at *critical_value*, with the resolved
    ``analytical_only`` baked in — ready to hand to
    :class:`~wrinklefe.analysis.WrinkleAnalysis`. ``None`` when no
    root."""

    bounds: tuple[float, float]
    """The resolved ``[lo, cap]`` actually scanned."""
    bounds_source: tuple[str, str]
    """Which rule set each bound: ``'user_bracket'`` | ``'zero'`` |
    ``'scaled_base'`` | ``'laminate_thickness'`` | ``'tool_flat_bound'``
    | ``'mesh_amplitude_safe'`` | ``'user_max_value'`` — so a refusal
    message can point at the knob that has to move."""
    scan_values: list[float]
    scan_objectives: list[float]
    """The scan grid and its measured objectives — the curve context, and
    the evidence behind every refusal message."""
    sign_changes: int
    """Number of sign changes of ``objective - target`` across the scan.
    ``1`` is the only value that admits a unique root."""
    log_spaced: bool
    """Whether the scan grid was log-spaced (drives the plot x-scale)."""

    pristine_reference_MPa: float | None
    """``achieved_strength_MPa / achieved_knockdown`` — the divisor that
    expresses a MPa target as a knockdown factor. ``None`` when no
    root."""

    evaluations: list[GoalSeekEvaluation]
    n_evaluations: int
    rtol: float
    analytical_only: bool
    """The **resolved** value actually used, not the argument."""
    runtime_s: float
    results: list[AnalysisResults] | None = field(default=None, repr=False)
    """Full per-evaluation results, only when ``keep_results=True`` (they
    carry mesh and field arrays on the FE path)."""

    def summary(self) -> str:
        """Human-readable search report (the caller prints it)."""
        unit = "MPa" if self.target_is_strength else ""
        lines = [
            f"Critical-value search: {self.parameter} "
            f"({self.objective_name} >= {self.target:.6g}{unit})",
            f"Direction: {self.direction}  |  range "
            f"[{self.bounds[0]:.6g}, {self.bounds[1]:.6g}] "
            f"(lower: {self.bounds_source[0]}, "
            f"upper: {self.bounds_source[1]})",
            f"Scan: {len(self.scan_values)} points, "
            f"{'log' if self.log_spaced else 'linear'}-spaced, "
            f"{self.sign_changes} sign change"
            f"{'' if self.sign_changes == 1 else 's'}",
            "",
            f"{'value':>16} {'objective':>14} {'knockdown':>12} "
            f"{'strength (MPa)':>16}",
        ]
        by_value = {ev.value: ev for ev in self.evaluations}
        for value, obj in zip(self.scan_values, self.scan_objectives):
            ev = by_value.get(value)
            kd = ev.knockdown if ev is not None else float("nan")
            mpa = ev.strength_MPa if ev is not None else float("nan")
            lines.append(
                f"{value:>16.6g} {obj:>14.6g} {kd:>12.4f} {mpa:>16.1f}"
            )
        lines.append("")
        if self.status == "converged":
            assert self.critical_value is not None
            assert self.achieved_objective is not None
            extreme = (
                "largest" if self.direction == "decreasing" else "smallest"
            )
            lines.append(
                f"{extreme.capitalize()} acceptable {self.parameter}: "
                f"{self.critical_value:.6g}"
            )
            lines.append(
                f"  raw root {self.critical_value_root:.6g}, backed off to "
                f"the safe side"
            )
            lines.append(
                f"Achieved {self.objective_name}: "
                f"{self.achieved_objective:.6f} "
                f"(target {self.target:.6f}, rtol {self.rtol:.1e})"
            )
            lines.append(
                f"Criterion satisfied: {self.objective_name} "
                f"{self.achieved_objective:.6f} >= {self.target:.6f}"
            )
            if self.target_is_strength and self.pristine_reference_MPa:
                lines.append(
                    f"Equivalent knockdown target: "
                    f"{self.target / self.pristine_reference_MPa:.6f} "
                    f"(pristine reference "
                    f"{self.pristine_reference_MPa:.1f} MPa)"
                )
        else:
            lines.append(f"Status: {self.status}")
            lines.append(self.message)
        lines.append(
            f"Forward-model evaluations: {self.n_evaluations} "
            f"({self.runtime_s:.2f} s, "
            f"{'analytical' if self.analytical_only else 'FE'} path)"
        )
        return "\n".join(lines)

    def plot(self, ax=None):
        """Objective vs parameter with the target and the answer marked.

        Returns the matplotlib Axes.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self.scan_values, self.scan_objectives, "o-", label="scan")
        ax.axhline(
            self.target, color="crimson", ls="--",
            label=f"target {self.target:.4g}",
        )
        if self.critical_value is not None:
            ax.axvline(
                self.critical_value, color="seagreen", ls=":",
                label=f"critical {self.critical_value:.4g}",
            )
        if self.log_spaced and min(self.scan_values) > 0.0:
            ax.set_xscale("log")
        ax.set_xlabel(self.parameter)
        ax.set_ylabel(self.objective_name)
        ax.set_title(f"Critical-value search ({self.status})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        return ax


# ======================================================================
# Private helpers
# ======================================================================

def _float_field_names(cfg: AnalysisConfig) -> list[str]:
    """Names of the continuously-searchable (``float``) config fields."""
    return sorted(
        f.name
        for f in fields(cfg)
        if type(getattr(cfg, f.name)) is float
    )


def _fe_forcing_features(cfg: AnalysisConfig) -> list[str]:
    """FE-only features enabled on *cfg*, in declaration order."""
    active: list[str] = []
    for name in FE_ONLY_FEATURES:
        value = getattr(cfg, name)
        if name == "transverse_mode":
            if value != "uniform":
                active.append(f"transverse_mode={value!r}")
        elif bool(value):
            active.append(name)
    return active


def _resolve_analytical_only(
    cfg: AnalysisConfig, analytical_only: bool | None
) -> bool:
    """Resolve the solve path, refusing combinations that cannot run.

    ``None`` means "analytical unless the config demands FE"; an
    explicit ``True`` alongside an FE-only feature is an error rather
    than a silent no-op.
    """
    if cfg.enable_progressive_damage:
        raise ValueError(
            "enable_progressive_damage=True is not supported by the "
            "critical-value search: each forward evaluation costs ~30 "
            "load-stepped solves, so a ~14-evaluation search runs for "
            "hours. Run WrinkleAnalysis.parametric_sweep over an "
            "explicit amplitude list instead, or disable progressive "
            "damage for the search."
        )
    active = _fe_forcing_features(cfg)
    if active:
        listed = ", ".join(active)
        if analytical_only is True:
            raise ValueError(
                f"analytical_only=True conflicts with FE-only config "
                f"feature(s): {listed}. They have no analytical-path "
                f"effect and would either silently no-op or raise "
                f"mid-search. Either drop the feature(s) from the base "
                f"config, or pass analytical_only=False (or leave it "
                f"None) to run the search on the FE path."
            )
        if analytical_only is None:
            logger.warning(
                "FE-only config feature(s) %s force the FE path; the "
                "search will run about 14 forward evaluations, each "
                "performing 4 linear solves",
                listed,
            )
        return False
    resolved = True if analytical_only is None else bool(analytical_only)
    if resolved and cfg.morphology == "tool_flat":
        raise ValueError(
            "morphology='tool_flat' cannot be searched on the "
            "analytical path: the analytical knockdown for tool_flat "
            "equals the 'uniform' result, and the FE path auto-enables "
            "surface resin pockets, which AnalysisConfig rejects under "
            "analytical_only=True. Pass analytical_only=False to search "
            "tool_flat, or pick a different morphology."
        )
    return resolved


def _scan_grid(lo: float, cap: float, n: int) -> tuple[list[float], bool]:
    """Return *n* scan abscissae over ``[lo, cap]`` and the spacing flag.

    Log spacing is used whenever the range spans more than
    :data:`LOG_SPACING_RATIO`, because a wide linear grid can walk
    straight past an interior extremum — the graded-morphology
    wavelength curve is the measured case.  A zero lower bound (the
    amplitude default) keeps the exact ``0.0`` endpoint and log-spaces
    the rest over three decades.

    Examples
    --------
    >>> values, log_spaced = _scan_grid(0.0, 4.0, 5)
    >>> [round(v, 6) for v in values]
    [0.0, 0.004, 0.04, 0.4, 4.0]
    >>> log_spaced
    True
    >>> values, log_spaced = _scan_grid(1.0, 5.0, 5)
    >>> [round(v, 6) for v in values]
    [1.0, 2.0, 3.0, 4.0, 5.0]
    >>> log_spaced
    False
    """
    if lo > 0.0 and cap / lo > LOG_SPACING_RATIO:
        grid = np.logspace(math.log10(lo), math.log10(cap), n)
        return [float(v) for v in grid], True
    if lo == 0.0 and cap > 0.0:
        rest = np.logspace(math.log10(cap) - 3.0, math.log10(cap), n - 1)
        return [0.0] + [float(v) for v in rest], True
    grid = np.linspace(lo, cap, n)
    return [float(v) for v in grid], False


def _sign(x: float) -> int:
    """Sign of *x* as ``-1`` / ``0`` / ``+1``."""
    if x > 0.0:
        return 1
    if x < 0.0:
        return -1
    return 0


def _count_reversals(f: list[float], tol: float) -> int:
    """Direction reversals in *f*, ignoring moves no larger than *tol*.

    A bit-exact plateau (or any move inside the tolerance) is a tie, not
    a reversal — otherwise a saturating-but-well-posed curve would be
    refused.
    """
    reversals = 0
    previous = 0
    for a, b in zip(f, f[1:]):
        delta = b - a
        if abs(delta) <= tol:
            continue
        current = 1 if delta > 0.0 else -1
        if previous != 0 and current != previous:
            reversals += 1
        previous = current
    return reversals


def _classify_scan(
    f: list[float], g: list[float], target: float, rtol: float
) -> tuple[str, int, tuple[int, int] | None]:
    """Choose the terminal status from the measured scan.

    This is the **only** place a terminal status is decided, and it runs
    before any "no crossing" exit — otherwise a scan whose interior sits
    far below the target could be reported as "nothing in range fails
    your criterion".

    Returns ``(status, sign_changes, bracket_indices)`` where *status*
    is ``'ok'`` when exactly one sign change and no direction reversal
    were found (the only case that admits a unique root).
    """
    tol = max(rtol * abs(target), MIN_RTOL)

    # (a) Flat, relative to the target — an inert parameter for this
    #     configuration cannot resolve the target at all.
    if max(f) - min(f) < tol:
        return "flat", 0, None

    # (b) A bit-exact plateau sitting on the target value.
    for a, b in zip(f, f[1:]):
        if a == b and abs(a - target) <= tol:
            return "flat", 0, None

    # (c) Sign changes over the non-zero residuals (an exact zero
    #     matches either neighbour), plus tie-tolerant reversals.
    nonzero = [(i, _sign(gi)) for i, gi in enumerate(g) if _sign(gi) != 0]
    sign_changes = 0
    bracket: tuple[int, int] | None = None
    for (i0, s0), (i1, s1) in zip(nonzero, nonzero[1:]):
        if s0 != s1:
            sign_changes += 1
            if bracket is None:
                bracket = (i0, i1)
    reversals = _count_reversals(f, tol)

    if sign_changes > 1 or (reversals > 0 and sign_changes >= 1):
        return "non_monotonic", sign_changes, bracket
    if sign_changes == 0:
        # (d) Direction-independent joint table: everything above the
        #     target means nothing in range violates the criterion;
        #     everything below means nothing in range meets it.
        if all(gi > 0.0 for gi in g):
            return "no_crossing", 0, None
        return "target_unreachable", 0, None
    return "ok", sign_changes, bracket


def _clip(x: float, lo: float, hi: float) -> float:
    """Clamp *x* into ``[lo, hi]``."""
    return min(max(x, lo), hi)


def _resolve_bounds(
    cfg: AnalysisConfig,
    parameter: str,
    bracket: tuple[float, float] | None,
    max_value: float | None,
    analytical_only: bool,
) -> tuple[float, float, tuple[str, str]]:
    """Resolve the ``[lo, cap]`` search range and record its provenance.

    Both bounds are sign-aware (``applied_strain`` is negative for
    compression, so a scaled range must stay on the negative half-line),
    and the upper bound is clamped by every validation limit that
    applies so the search cannot die mid-run on a legal-looking probe.
    """
    p0 = float(getattr(cfg, parameter))
    source_lo = "user_bracket"
    source_hi = "user_bracket"

    if bracket is not None:
        lo, cap = float(bracket[0]), float(bracket[1])
    else:
        if max_value is not None:
            cap, source_hi = float(max_value), "user_max_value"
        elif parameter == "amplitude":
            n_plies = max(len(cfg.angles or ()), 1)
            cap = float(n_plies * cfg.ply_thickness)
            source_hi = "laminate_thickness"
        elif p0 > 0.0:
            cap, source_hi = 256.0 * p0, "scaled_base"
        elif p0 < 0.0:
            cap, source_hi = p0 / 256.0, "scaled_base"
        else:
            raise ValueError(
                f"cannot derive a search range from "
                f"AnalysisConfig.{parameter} = 0.0 — a scaled range "
                f"around zero is degenerate. Pass an explicit "
                f"bracket=(lo, hi) or max_value."
            )
        if parameter == "amplitude":
            lo, source_lo = 0.0, "zero"
        elif p0 > 0.0:
            lo, source_lo = p0 / 256.0, "scaled_base"
        elif p0 < 0.0:
            lo, source_lo = 256.0 * p0, "scaled_base"
        else:
            lo, source_lo = 0.0, "zero"

    if parameter == "amplitude":
        if cfg.morphology == "tool_flat":
            bound = (
                0.999
                * 0.8
                * cfg.surface_transition_plies
                * cfg.ply_thickness
                / cfg.nz_per_ply
            )
            if bound < cap:
                if bracket is not None:
                    logger.warning(
                        "requested upper bound %.6g exceeds the "
                        "tool_flat element-inversion limit %.6g mm; "
                        "clamping",
                        cap, bound,
                    )
                cap, source_hi = float(bound), "tool_flat_bound"
        if not analytical_only:
            diagnostics = mesh_shear_diagnostics(
                amplitude=(
                    cfg.amplitude if cfg.amplitude > 0.0 else cfg.ply_thickness
                ),
                wavelength=cfg.wavelength,
                ply_thickness=cfg.ply_thickness,
                nx=cfg.nx,
                nz_per_ply=cfg.nz_per_ply,
                Lx=cfg.domain_length,
            )
            bound = diagnostics.amplitude_safe
            if bound < cap:
                if bracket is not None:
                    logger.warning(
                        "requested upper bound %.6g exceeds the "
                        "mesh-inversion limit %.6g mm "
                        "(2.5 * ply_thickness / nz_per_ply); clamping",
                        cap, bound,
                    )
                cap, source_hi = float(bound), "mesh_amplitude_safe"

    if not lo < cap:
        raise ValueError(
            f"empty search range for '{parameter}': resolved bounds "
            f"[{lo:.6g}, {cap:.6g}] (lower: {source_lo}, upper: "
            f"{source_hi}). A validation clamp inverted the range — "
            f"lower the bracket, or relax nz_per_ply / "
            f"surface_transition_plies."
        )
    return lo, cap, (source_lo, source_hi)


def _bound_clause(
    source: str, cap: float, cfg: AnalysisConfig
) -> str:
    """One sentence naming the knob that set the upper bound."""
    if source == "mesh_amplitude_safe":
        return (
            f"The upper bound came from 'mesh_amplitude_safe' "
            f"({cap:.6g} mm = 2.5 * ply_thickness / nz_per_ply); above "
            f"it the hex mesh inverts. Reduce nz_per_ply, or run with "
            f"the analytical path."
        )
    if source == "tool_flat_bound":
        return (
            f"The upper bound came from 'tool_flat_bound' ({cap:.6g} "
            f"mm); above it the crest-side transition elements invert. "
            f"Increase surface_transition_plies or reduce nz_per_ply."
        )
    if source == "user_bracket":
        return "The upper bound came from the explicit bracket."
    if source == "user_max_value":
        return "The upper bound came from the explicit max_value."
    if source == "laminate_thickness":
        return (
            f"The upper bound came from 'laminate_thickness' "
            f"({cap:.6g} mm = len(angles) * ply_thickness). To search "
            f"further raise max_value or pass an explicit bracket."
        )
    return f"The upper bound came from '{source}'."


def _flat_message(
    parameter: str,
    objective_name: str,
    target: float,
    values: list[float],
    f: list[float],
    rtol: float,
) -> str:
    """Explain a curve too flat (or too plateaued) to resolve a target."""
    span = max(f) - min(f)
    i_min, i_max = f.index(min(f)), f.index(max(f))
    return (
        f"'{objective_name}' does not resolve the target over the "
        f"searched range of '{parameter}': across "
        f"{len(values)} scan points it spans only {span:.6g} "
        f"({f[i_min]:.10g} at {parameter}={values[i_min]:.6g} to "
        f"{f[i_max]:.10g} at {parameter}={values[i_max]:.6g}), which is "
        f"below the resolution rtol * |target| = "
        f"{max(rtol * abs(target), MIN_RTOL):.3g}, or it is bit-exactly "
        f"constant at the target value. The parameter is effectively "
        f"inert for this configuration (a saturated penetration gate "
        f"and a decayed-away wrinkle are the usual causes). Search a "
        f"different parameter, widen the bracket, or check that the "
        f"configuration actually responds to '{parameter}'."
    )


def _non_monotonic_message(
    parameter: str,
    objective_name: str,
    values: list[float],
    f: list[float],
    sign_changes: int,
    log_spaced: bool,
    cfg: AnalysisConfig,
) -> str:
    """Explain a refusal caused by a non-unique root."""
    i_min = f.index(min(f))
    message = (
        f"'{objective_name}' is not monotonic in '{parameter}' over "
        f"[{values[0]:.6g}, {values[-1]:.6g}]: the {len(values)}-point "
        f"{'log' if log_spaced else 'linear'}-spaced scan shows "
        f"{sign_changes} sign change"
        f"{'' if sign_changes == 1 else 's'} of (objective - target) "
        f"and a direction reversal — it reaches {f[i_min]:.6g} at "
        f"{parameter}={values[i_min]:.6g} between the endpoint values "
        f"{f[0]:.6g} at {values[0]:.6g} and {f[-1]:.6g} at "
        f"{values[-1]:.6g}. A root is not unique, so the search refuses "
        f"rather than returning an arbitrary one."
    )
    if cfg.morphology == "graded" and parameter == "wavelength":
        message += (
            " Known cause for this configuration: morphology='graded' "
            "widens the through-thickness Gaussian as wavelength grows "
            "(decay_scale_eff = max(wavelength/2, amplitude)), "
            "producing a U-shaped curve with an interior minimum near "
            "lambda ~ 3 mm. Pin through_thickness_decay_scale, or "
            "narrow the bracket to one side of the minimum."
        )
    message += (
        " Narrow the bracket to a monotone sub-range, or run "
        "WrinkleAnalysis.parametric_sweep over the bracket to see the "
        "whole curve first."
    )
    return message


def _no_crossing_message(
    parameter: str,
    objective_name: str,
    target: float,
    values: list[float],
    f: list[float],
    source_hi: str,
    cap: float,
    cfg: AnalysisConfig,
) -> str:
    """Explain that nothing in range violates the criterion."""
    i_min = f.index(min(f))
    return (
        f"no value of '{parameter}' in [{values[0]:.6g}, "
        f"{values[-1]:.6g}] violates the criterion '{objective_name} >= "
        f"{target:.6g}'. The objective was scanned at {len(values)} "
        f"points over that range; its minimum is {f[i_min]:.6g} at "
        f"{parameter} = {values[i_min]:.6g} — still above the target. "
        f"{_bound_clause(source_hi, cap, cfg)} Otherwise pick a target "
        f"above {f[i_min]:.6g}. This is not necessarily an error: it "
        f"means no defect size in range fails your criterion."
    )


def _unreachable_message(
    parameter: str,
    objective_name: str,
    target: float,
    values: list[float],
    f: list[float],
    source_hi: str,
    cap: float,
    cfg: AnalysisConfig,
) -> str:
    """Explain that nothing in range meets the criterion."""
    i_max = f.index(max(f))
    return (
        f"no value of '{parameter}' in [{values[0]:.6g}, "
        f"{values[-1]:.6g}] meets the criterion '{objective_name} >= "
        f"{target:.6g}'. The objective was scanned at {len(values)} "
        f"points over that range; even its maximum, {f[i_max]:.6g} at "
        f"{parameter} = {values[i_max]:.6g}, falls short. "
        f"{_bound_clause(source_hi, cap, cfg)} The configuration cannot "
        f"meet this target at any searched defect size — lower the "
        f"target, or change the laminate."
    )


def _validate_target(
    target_knockdown: float | None, target_strength_MPa: float | None
) -> tuple[float, bool]:
    """Validate the mutually exclusive target pair; return (target, is_MPa)."""
    if (target_knockdown is None) == (target_strength_MPa is None):
        raise ValueError(
            "exactly one of target_knockdown and target_strength_MPa "
            f"is required, got target_knockdown={target_knockdown!r} "
            f"and target_strength_MPa={target_strength_MPa!r}"
        )
    if target_knockdown is not None:
        value = float(target_knockdown)
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            if value == 1.0:
                raise ValueError(
                    "target_knockdown=1.0 asks for the largest defect "
                    "that causes no knockdown at all. That is "
                    "degenerate: the analytical knockdown is "
                    "bit-exactly 1.0 at amplitude 0 for both loadings, "
                    "and on the tension path it stays 1.0 over a finite "
                    "interval, so the answer is an interval, not a "
                    "value. Pick a target strictly below 1.0 (e.g. "
                    "0.99)."
                )
            raise ValueError(
                f"target_knockdown must be a finite value in (0.0, "
                f"1.0), got {target_knockdown!r}"
            )
        return value, False
    assert target_strength_MPa is not None
    value = float(target_strength_MPa)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"target_strength_MPa must be a finite value > 0, got "
            f"{target_strength_MPa!r}"
        )
    return value, True


def _validate_parameter(cfg: AnalysisConfig, parameter: str) -> None:
    """Refuse anything ``brentq`` cannot legally probe."""
    names = {f.name for f in fields(cfg)}
    if parameter not in names:
        raise ValueError(
            f"AnalysisConfig has no continuously-searchable field "
            f"{parameter!r} — expected one of "
            f"{_float_field_names(cfg)} (float fields only; integer "
            f"mesh/ply-count fields are not root-findable)"
        )
    value = getattr(cfg, parameter)
    if type(value) is float:
        return
    if type(value) is bool:
        raise ValueError(
            f"{parameter!r} is a boolean AnalysisConfig field; a "
            f"continuous root-find over it is meaningless. Search a "
            f"float field, or compare two explicit runs."
        )
    if type(value) is int:
        raise ValueError(
            f"{parameter!r} is an integer AnalysisConfig field. brentq "
            f"is a continuous solver and AnalysisConfig._validate "
            f"rejects non-integral values, so a goal-seek over it would "
            f"fail on the first fractional probe. Use "
            f"wrinklefe.convergence.mesh_convergence_study for "
            f"mesh-density questions, or "
            f"WrinkleAnalysis.parametric_sweep over an explicit integer "
            f"list."
        )
    if value is None:
        raise ValueError(
            f"AnalysisConfig.{parameter} is None on this config, so the "
            f"search has no value to scale a range from. Set it to a "
            f"float first, or pass an explicit bracket for a different "
            f"parameter."
        )
    raise ValueError(
        f"AnalysisConfig.{parameter} is a "
        f"{type(value).__name__}, not a float — expected one of "
        f"{_float_field_names(cfg)}"
    )


def _resolve_objective(
    objective: str | Callable[[AnalysisResults], float],
    target_is_strength: bool,
) -> tuple[Callable[[AnalysisResults], float], str]:
    """Resolve the objective extractor and its reported name."""
    if callable(objective):
        if target_is_strength:
            raise ValueError(
                "target_strength_MPa selects the built-in 'strength_MPa' "
                "objective; it cannot be combined with a callable "
                "objective. Use target_knockdown with the callable, and "
                "have the callable return the quantity being targeted."
            )
        return objective, getattr(objective, "__name__", "custom")
    if target_is_strength:
        if objective not in ("knockdown", "strength_MPa"):
            raise ValueError(
                f"target_strength_MPa root-finds the 'strength_MPa' "
                f"objective directly, so it cannot be combined with "
                f"objective={objective!r}. Drop the objective argument, "
                f"or express the target as a knockdown factor."
            )
        objective = "strength_MPa"
    if objective in _REFUSED_OBJECTIVES:
        raise ValueError(_REFUSED_OBJECTIVES[objective])
    if objective not in _OBJECTIVES:
        raise ValueError(
            f"unknown objective {objective!r}; expected one of "
            f"{sorted(_OBJECTIVES)} or a callable"
        )
    return _OBJECTIVES[objective], objective


def _check_objective_available(
    cfg: AnalysisConfig, objective_name: str, analytical_only: bool
) -> None:
    """Refuse an objective the resolved solve path cannot produce."""
    if objective_name in _FE_ONLY_OBJECTIVES and analytical_only:
        raise ValueError(
            f"objective={objective_name!r} is measured from the FE "
            f"global reaction response and keeps its 1.0 default on the "
            f"analytical path. Pass analytical_only=False, or pick "
            f"objective='modulus_knockdown' for the analytical "
            f"stiffness knockdown."
        )
    if objective_name == "onset_knockdown" and cfg.loading != "tension":
        raise ValueError(
            f"objective='onset_knockdown' is computed on the tension "
            f"path only; this config has loading={cfg.loading!r}. Use "
            f"loading='tension', or pick objective='knockdown'."
        )


def _backoff_to_safe_side(
    probe: Callable[[float, str], GoalSeekEvaluation],
    row: GoalSeekEvaluation,
    root: float,
    target: float,
    direction: str,
    xtol: float,
    rtol: float,
    lo: float,
    cap: float,
) -> GoalSeekEvaluation:
    """Walk off the root until a real forward solve meets the criterion.

    ``brentq`` converges to a root, not to the side of it that satisfies
    ``objective >= target``; on measured cases the raw root violates the
    criterion in three runs out of four.  The walk is geometric,
    direction-aware and budgeted by ``rtol * |root|``, so the returned
    value stays inside the tolerance contract.
    """
    step_sign = -1.0 if direction == "decreasing" else 1.0
    step0 = max(xtol, 8.0 * _EPS * abs(root))
    budget = max(BACKOFF_BUDGET_RTOL_MULT * rtol * abs(root), 16.0 * step0)
    offset = 0.0
    previous = root
    k = 0
    while row.objective < target and offset <= budget and k < 24:
        offset = (2.0**k) * step0
        candidate = _clip(root + step_sign * offset, lo, cap)
        k += 1
        if candidate == previous:
            break
        previous = candidate
        row = probe(candidate, "backoff")
    return row


# ======================================================================
# Public driver
# ======================================================================

def find_critical_value(
    base_config: AnalysisConfig,
    *,
    parameter: str = "amplitude",
    target_knockdown: float | None = None,
    target_strength_MPa: float | None = None,
    objective: str | Callable[[AnalysisResults], float] = "knockdown",
    bracket: tuple[float, float] | None = None,
    max_value: float | None = None,
    scan_points: int = DEFAULT_SCAN_POINTS,
    rtol: float = 1e-3,
    analytical_only: bool | None = None,
    keep_results: bool = False,
) -> CriticalValueResult:
    """Find the largest defect size that still meets a target criterion.

    The acceptable set is always ``{x : objective(x) >= target}`` (a
    larger objective is safer — true for knockdown, strength and modulus
    knockdown, and the contract a user-supplied callable must honour).
    For a decreasing objective the answer is the **largest** acceptable
    value; for an increasing one it is the **smallest**.  The returned
    ``critical_value`` is verified by a real forward evaluation, not
    merely by the root tolerance.

    Parameters
    ----------
    base_config : AnalysisConfig
        Configuration to clone at every trial value.
    parameter : str, optional
        Float :class:`~wrinklefe.analysis.AnalysisConfig` field to
        search. Default ``'amplitude'``. Integer mesh/ply-count fields
        are refused by name.
    target_knockdown : float, optional
        Target as a knockdown factor in ``(0, 1)``. Exactly one of
        *target_knockdown* and *target_strength_MPa* is required.
    target_strength_MPa : float, optional
        Target as an absolute strength in MPa; selects the
        ``'strength_MPa'`` objective and root-finds it directly.
    objective : str or callable, optional
        One of ``'knockdown'`` (default), ``'strength_MPa'``,
        ``'modulus_knockdown'``, ``'onset_knockdown'`` (tension only),
        ``'modulus_retention_global'`` (FE only) — or a callable
        ``f(AnalysisResults) -> float``.
    bracket : tuple of float, optional
        Explicit ``(lo, hi)`` search range. Still clamped by the
        tool_flat and mesh-inversion limits.
    max_value : float, optional
        Upper bound when *bracket* is not given.
    scan_points : int, optional
        Points in the bounded scan that measures direction and
        monotonicity (>= 3). Default 9.
    rtol : float, optional
        Relative tolerance on the parameter, passed to
        :func:`scipy.optimize.brentq`. Default ``1e-3``.
    analytical_only : bool or None, optional
        ``None`` (default) means "analytical unless the base config
        enables an FE-only feature". ``True`` with such a feature is an
        error rather than a silent no-op. Root-finding is inherently
        sequential, so there is no worker-count knob.
    keep_results : bool, optional
        Retain the full :class:`~wrinklefe.analysis.AnalysisResults` per
        evaluation (they carry mesh and field arrays on the FE path).

    Returns
    -------
    CriticalValueResult
        The critical value plus the scan curve, the evaluation ledger
        and — for every refusal — an actionable message.

    Raises
    ------
    ValueError
        On any invalid input: an unknown, integer or boolean
        *parameter*, a missing/ambiguous target, an unavailable
        objective, a degenerate range, or an FE-only config feature
        combined with ``analytical_only=True``. Search *outcomes* (no
        crossing, non-monotonic, flat) are **returned** with a status,
        not raised.

    Notes
    -----
    ``run()`` resolves ``analytical_only=None`` to ``cfg.analytical_only``;
    the ladder here overrides that, and the resolved value is baked into
    every cloned config so a saved result records the path it ran on.
    """
    t_start = time.perf_counter()

    target, target_is_strength = _validate_target(
        target_knockdown, target_strength_MPa
    )
    _validate_parameter(base_config, parameter)

    if not math.isfinite(rtol) or not MIN_RTOL <= rtol < 0.1:
        raise ValueError(
            f"rtol must be a finite value in [{MIN_RTOL:.6g}, 0.1) — "
            f"scipy's own floor is 4 * eps — got {rtol!r}"
        )
    if scan_points < 3:
        raise ValueError(
            f"scan_points must be >= 3 (both endpoints plus at least "
            f"one interior point), got {scan_points!r}"
        )
    if bracket is not None:
        if len(tuple(bracket)) != 2:
            raise ValueError(
                f"bracket must be a (lo, hi) pair, got {bracket!r}"
            )
        b_lo, b_hi = float(bracket[0]), float(bracket[1])
        if not (math.isfinite(b_lo) and math.isfinite(b_hi)):
            raise ValueError(f"bracket bounds must be finite, got {bracket!r}")
        if not b_lo < b_hi:
            raise ValueError(
                f"bracket must satisfy lo < hi, got lo={b_lo!r}, hi={b_hi!r}"
            )
        if b_lo < 0.0 and parameter in ("amplitude", "wavelength", "width"):
            raise ValueError(
                f"bracket lower bound must be >= 0 for {parameter!r}, "
                f"got {b_lo!r}"
            )
        bracket = (b_lo, b_hi)

    resolved_analytical = _resolve_analytical_only(base_config, analytical_only)
    extract, objective_name = _resolve_objective(objective, target_is_strength)
    if not callable(objective):
        _check_objective_available(
            base_config, objective_name, resolved_analytical
        )

    if base_config.wrinkles is not None and parameter in (
        "amplitude", "wavelength",
    ):
        raise ValueError(
            f"AnalysisConfig.{parameter} is ignored when 'wrinkles' is "
            f"set — both analytical pathways read only the per-spec "
            f"values. Goal-seek on wrinkles[i].{parameter} is not "
            f"supported; drop 'wrinkles', or search a different field."
        )

    if resolved_analytical is False and objective_name in (
        "knockdown", "strength_MPa", "modulus_knockdown", "onset_knockdown",
    ):
        logger.warning(
            "objective %r is computed analytically before the FE branch, "
            "so the FE path costs 4 linear solves per evaluation without "
            "changing the number being searched",
            objective_name,
        )
    if (
        parameter == "wavelength"
        and base_config.domain_length != 3.0 * base_config.wavelength
    ):
        logger.warning(
            "goal-seek on 'wavelength' with an explicitly pinned "
            "domain_length=%s: the domain is held fixed while the "
            "wavelength varies, so the number of wrinkle periods in the "
            "domain changes across the search. Set domain_length=0.0 to "
            "auto-derive 3 * wavelength if that is not intended",
            base_config.domain_length,
        )

    lo, cap, bounds_source = _resolve_bounds(
        base_config, parameter, bracket, max_value, resolved_analytical
    )

    evaluations: list[GoalSeekEvaluation] = []
    kept: list[AnalysisResults] = []

    def probe(value: float, phase: str) -> GoalSeekEvaluation:
        t0 = time.perf_counter()
        cfg = _replace_swept(
            base_config,
            parameter,
            float(value),
            analytical_only=resolved_analytical,
        )
        results = WrinkleAnalysis(cfg).run(analytical_only=resolved_analytical)
        runtime = time.perf_counter() - t0
        measured = float(extract(results))
        row = GoalSeekEvaluation(
            index=len(evaluations),
            value=float(value),
            objective=measured,
            knockdown=float(results.analytical_knockdown),
            strength_MPa=float(results.analytical_strength_MPa),
            residual=measured - target,
            phase=phase,
            runtime_s=runtime,
        )
        evaluations.append(row)
        if keep_results:
            kept.append(results)
        logger.info(
            "goal-seek %s: %s=%.6g -> %s=%.6g (residual %+.3e)",
            phase, parameter, row.value, objective_name, measured,
            row.residual,
        )
        return row

    grid, log_spaced = _scan_grid(lo, cap, scan_points)
    scan_rows = [probe(value, "scan") for value in grid]
    f = [row.objective for row in scan_rows]
    g = [row.residual for row in scan_rows]
    direction = "decreasing" if f[-1] < f[0] else "increasing"

    status, sign_changes, bracket_indices = _classify_scan(f, g, target, rtol)

    def _build(
        status: str,
        message: str,
        row: GoalSeekEvaluation | None = None,
        value: float | None = None,
        root: float | None = None,
    ) -> CriticalValueResult:
        reference: float | None = None
        if row is not None and row.knockdown > 0.0:
            reference = row.strength_MPa / row.knockdown
        result = CriticalValueResult(
            parameter=parameter,
            objective_name=objective_name,
            target=target,
            target_is_strength=target_is_strength,
            direction=direction,
            status=status,
            message=message,
            critical_value=value,
            critical_value_root=root,
            achieved_objective=None if row is None else row.objective,
            achieved_knockdown=None if row is None else row.knockdown,
            achieved_strength_MPa=None if row is None else row.strength_MPa,
            critical_config=(
                None
                if value is None
                else _replace_swept(
                    base_config, parameter, value,
                    analytical_only=resolved_analytical,
                )
            ),
            bounds=(lo, cap),
            bounds_source=bounds_source,
            scan_values=list(grid),
            scan_objectives=list(f),
            sign_changes=sign_changes,
            log_spaced=log_spaced,
            pristine_reference_MPa=reference,
            evaluations=evaluations,
            n_evaluations=len(evaluations),
            rtol=float(rtol),
            analytical_only=resolved_analytical,
            runtime_s=time.perf_counter() - t_start,
            results=kept if keep_results else None,
        )
        logger.info(
            "critical-value search finished: status=%s, %s=%s",
            status, parameter, value,
        )
        return result

    if status == "flat":
        return _build(
            "flat",
            _flat_message(
                parameter, objective_name, target, grid, f, rtol
            ),
        )
    if status == "non_monotonic":
        return _build(
            "non_monotonic",
            _non_monotonic_message(
                parameter, objective_name, grid, f, sign_changes,
                log_spaced, base_config,
            ),
        )
    if status == "no_crossing":
        return _build(
            "no_crossing",
            _no_crossing_message(
                parameter, objective_name, target, grid, f,
                bounds_source[1], cap, base_config,
            ),
        )
    if status == "target_unreachable":
        return _build(
            "target_unreachable",
            _unreachable_message(
                parameter, objective_name, target, grid, f,
                bounds_source[1], cap, base_config,
            ),
        )

    assert bracket_indices is not None  # guaranteed by _classify_scan
    i0, i1 = bracket_indices
    a, b = grid[i0], grid[i1]
    xtol = max(MIN_XTOL, 1e-6 * (b - a))

    def residual(x: float) -> float:
        return probe(float(x), "root").residual

    root = float(
        brentq(residual, a, b, xtol=xtol, rtol=rtol, maxiter=100)
    )
    row = probe(root, "verify")
    row = _backoff_to_safe_side(
        probe, row, root, target, direction, xtol, rtol, lo, cap
    )

    if row.objective >= target and abs(row.objective - target) <= rtol * abs(
        target
    ):
        return _build("converged", "", row, row.value, root)
    return _build(
        "flat",
        f"the safe-side back-off could not satisfy '{objective_name} >= "
        f"{target:.6g}' within rtol * |root| of the root "
        f"{root:.6g}: the closest real evaluation gave "
        f"{row.objective:.10g} at {parameter}={row.value:.6g}. The "
        f"objective is near-degenerate there (a near-vertical or "
        f"near-flat curve). Widen rtol, or narrow the bracket around "
        f"the crossing.",
        row,
        None,
        root,
    )
