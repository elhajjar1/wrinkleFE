"""Mesh-convergence study helper.

Every FE number the package produces is conditional on mesh density
(``nx`` / ``ny`` / ``nz_per_ply`` on :class:`~wrinklefe.analysis.AnalysisConfig`),
and the wrinkle region concentrates gradients, so "is my mesh fine
enough?" deserves a supported answer. :func:`mesh_convergence_study`
runs the analysis at successively refined meshes, records a quantity of
interest (QoI) per level alongside DOF count and wall time, and
recommends the coarsest level whose QoI agrees with the finest level
within a user tolerance.

Example
-------
>>> from wrinklefe.analysis import AnalysisConfig
>>> from wrinklefe.convergence import mesh_convergence_study
>>> study = mesh_convergence_study(
...     AnalysisConfig(amplitude=0.366, wavelength=16.0, width=12.0),
...     levels=3, qoi="max_fi", tolerance=0.01,
... )                                                   # doctest: +SKIP
>>> print(study.summary())                              # doctest: +SKIP
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from wrinklefe.analysis import AnalysisConfig, AnalysisResults, WrinkleAnalysis

logger = logging.getLogger(__name__)

__all__ = [
    "ConvergenceLevel",
    "ConvergenceStudy",
    "mesh_convergence_study",
]

#: Default geometric-ish refinement multipliers applied to the selected
#: mesh axes, level by level.
DEFAULT_FACTORS: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _qoi_max_fi(results: AnalysisResults) -> float:
    """Peak failure index over all criteria (linear and CZM paths)."""
    if not results.failure_indices:
        raise ValueError(
            "results.failure_indices is empty — the FE failure evaluation "
            "did not run (analytical_only config?)"
        )
    return max(
        float(np.max(arr)) for arr in results.failure_indices.values()
    )


def _qoi_modulus_retention(results: AnalysisResults) -> float:
    return float(results.modulus_retention)


def _qoi_strength_retention(results: AnalysisResults) -> float:
    """Min per-criterion strength retention (linear path only)."""
    if not results.retention_factors:
        raise ValueError(
            "results.retention_factors is empty — strength retention is "
            "only computed on the linear (non-CZM) FE path"
        )
    return min(float(v) for v in results.retention_factors.values())


def _qoi_max_damage(results: AnalysisResults) -> float:
    """Peak cohesive interface damage (CZM path only)."""
    if results.czm_damage is None or not results.czm_damage.size:
        raise ValueError(
            "results.czm_damage is empty — enable_czm must be True for "
            "the 'max_damage' QoI"
        )
    return float(np.max(results.czm_damage))


_QOI_EXTRACTORS = {
    "max_fi": _qoi_max_fi,
    "modulus_retention": _qoi_modulus_retention,
    "strength_retention": _qoi_strength_retention,
    "max_damage": _qoi_max_damage,
}


@dataclass
class ConvergenceLevel:
    """One row of a mesh-convergence table."""

    level: int
    nx: int
    ny: int
    nz_per_ply: int
    n_dof: int
    qoi: float
    delta_pct: float | None
    """Relative change vs the previous level, in percent
    (``100 * |q_k - q_{k-1}| / |q_k|``); ``None`` for the first level."""
    runtime_s: float


@dataclass
class ConvergenceStudy:
    """Result of :func:`mesh_convergence_study`."""

    levels: list[ConvergenceLevel]
    qoi_name: str
    tolerance: float
    recommended_config: AnalysisConfig | None
    """Coarsest level whose QoI agrees with the finest level within
    ``tolerance`` (relative). ``None`` if only the finest level
    satisfies the tolerance — refine further."""
    recommended_level: int | None
    observed_rate: float | None
    """Richardson-style observed convergence rate from the last three
    levels; ``None`` when the refinement ratio is not constant or the
    QoI differences do not allow it."""

    def summary(self) -> str:
        """Human-readable convergence table."""
        lines = [
            f"Mesh-convergence study (QoI: {self.qoi_name}, "
            f"tolerance {self.tolerance:.2%})",
            f"{'lvl':>3} {'nx':>5} {'ny':>4} {'nz/ply':>6} {'DOFs':>9} "
            f"{'QoI':>12} {'d%':>8} {'time (s)':>9}",
        ]
        for lv in self.levels:
            delta = f"{lv.delta_pct:8.3f}" if lv.delta_pct is not None else "       —"
            lines.append(
                f"{lv.level:>3} {lv.nx:>5} {lv.ny:>4} {lv.nz_per_ply:>6} "
                f"{lv.n_dof:>9} {lv.qoi:>12.6g} {delta} {lv.runtime_s:>9.2f}"
            )
        if self.observed_rate is not None:
            lines.append(f"Observed convergence rate: {self.observed_rate:.2f}")
        if self.recommended_level is not None:
            lines.append(
                f"Recommended level: {self.recommended_level} "
                f"(coarsest within {self.tolerance:.2%} of the finest level)"
            )
        else:
            lines.append(
                "No level within tolerance of the finest — refine further."
            )
        return "\n".join(lines)

    def plot(self, ax=None):
        """QoI vs DOF count (log-x). Returns the matplotlib Axes."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))
        dofs = [lv.n_dof for lv in self.levels]
        qois = [lv.qoi for lv in self.levels]
        ax.semilogx(dofs, qois, "o-")
        ax.set_xlabel("DOF count")
        ax.set_ylabel(self.qoi_name)
        ax.set_title("Mesh convergence")
        ax.grid(True, alpha=0.3)
        return ax


def _observed_rate(
    qois: Sequence[float], ratios: Sequence[float]
) -> float | None:
    """Richardson-style rate from the last three QoIs, if the sequence allows."""
    if len(qois) < 3 or len(ratios) < 2:
        return None
    r1, r2 = ratios[-2], ratios[-1]
    if not math.isclose(r1, r2, rel_tol=0.05):
        return None  # rate formula assumes a constant refinement ratio
    q1, q2, q3 = qois[-3], qois[-2], qois[-1]
    d12, d23 = abs(q2 - q1), abs(q3 - q2)
    if d23 <= 0.0 or d12 <= 0.0:
        return None
    return math.log(d12 / d23) / math.log(r2)


def mesh_convergence_study(
    base_config: AnalysisConfig,
    levels: int = 4,
    refine: Sequence[str] = ("nx", "nz_per_ply"),
    qoi: str | Callable[[AnalysisResults], float] = "max_fi",
    tolerance: float = 0.01,
    factors: Sequence[float] | None = None,
) -> ConvergenceStudy:
    """Run the analysis at successively refined meshes and tabulate a QoI.

    Parameters
    ----------
    base_config : AnalysisConfig
        Level-0 configuration. Must run the FE path (``analytical_only``
        is forced off for the study); works in both pure-FE and CZM
        (``enable_czm=True``) modes.
    levels : int, optional
        Number of refinement levels to run (>= 2). Default 4.
    refine : sequence of str, optional
        Mesh axes to refine: any of ``'nx'``, ``'ny'``, ``'nz_per_ply'``.
        Default ``('nx', 'nz_per_ply')``.
    qoi : str or callable, optional
        Quantity of interest per level. One of ``'max_fi'`` (default;
        peak failure index over all criteria), ``'modulus_retention'``,
        ``'strength_retention'`` (linear path only), ``'max_damage'``
        (CZM only) — or a callable ``f(AnalysisResults) -> float``.
    tolerance : float, optional
        Relative QoI tolerance for the recommendation. Default 0.01 (1%).
    factors : sequence of float, optional
        Refinement multipliers per level (first entry is level 0).
        Default geometric-ish ``(1, 1.5, 2, 3, 4, 6)`` truncated to
        *levels* entries.

    Returns
    -------
    ConvergenceStudy
        Per-level table, recommendation, and observed convergence rate.

    Raises
    ------
    ValueError
        On bad *levels*, *refine* axes, unknown *qoi* name, or a QoI
        that is unavailable for the config's solve path.
    """
    if levels < 2:
        raise ValueError(f"levels must be >= 2, got {levels}")
    valid_axes = {"nx", "ny", "nz_per_ply"}
    refine = tuple(refine)
    bad = set(refine) - valid_axes
    if bad or not refine:
        raise ValueError(
            f"refine must be a non-empty subset of {sorted(valid_axes)}, "
            f"got {refine}"
        )
    if callable(qoi):
        extract, qoi_name = qoi, getattr(qoi, "__name__", "custom")
    else:
        if qoi not in _QOI_EXTRACTORS:
            raise ValueError(
                f"unknown qoi {qoi!r}; expected one of "
                f"{sorted(_QOI_EXTRACTORS)} or a callable"
            )
        extract, qoi_name = _QOI_EXTRACTORS[qoi], qoi

    if factors is None:
        factors = DEFAULT_FACTORS
    if len(factors) < levels:
        raise ValueError(
            f"need at least {levels} refinement factors, got {len(factors)}"
        )
    factors = tuple(float(f) for f in factors[:levels])

    rows: list[ConvergenceLevel] = []
    qois: list[float] = []
    for k, f in enumerate(factors):
        overrides: dict[str, Any] = {
            axis: max(1, int(round(getattr(base_config, axis) * f)))
            for axis in refine
        }
        cfg = replace(base_config, analytical_only=False, **overrides)
        logger.info(
            "Convergence level %d/%d: %s", k, levels - 1, overrides
        )
        t0 = time.perf_counter()
        results = WrinkleAnalysis(cfg).run(analytical_only=False)
        runtime = time.perf_counter() - t0

        q = float(extract(results))
        delta = (
            100.0 * abs(q - qois[-1]) / max(abs(q), 1e-30) if qois else None
        )
        qois.append(q)
        rows.append(
            ConvergenceLevel(
                level=k,
                nx=cfg.nx, ny=cfg.ny, nz_per_ply=cfg.nz_per_ply,
                n_dof=(
                    int(results.mesh.n_dof) if results.mesh is not None else 0
                ),
                qoi=q,
                delta_pct=delta,
                runtime_s=runtime,
            )
        )

    # Recommendation: coarsest level within tolerance of the finest.
    q_ref = qois[-1]
    recommended_level: int | None = None
    for k in range(len(qois) - 1):
        if abs(qois[k] - q_ref) <= tolerance * max(abs(q_ref), 1e-30):
            recommended_level = k
            break
    recommended_config: AnalysisConfig | None = None
    if recommended_level is not None:
        lv = rows[recommended_level]
        recommended_config = replace(
            base_config,
            nx=lv.nx, ny=lv.ny, nz_per_ply=lv.nz_per_ply,
        )

    ratios = [factors[i + 1] / factors[i] for i in range(len(factors) - 1)]
    study = ConvergenceStudy(
        levels=rows,
        qoi_name=qoi_name,
        tolerance=float(tolerance),
        recommended_config=recommended_config,
        recommended_level=recommended_level,
        observed_rate=_observed_rate(qois, ratios),
    )
    logger.info(
        "Convergence study complete: recommended level %s",
        recommended_level,
    )
    return study
