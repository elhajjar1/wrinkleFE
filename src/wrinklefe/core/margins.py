"""Regime-conditional validation safety margins (issue #455).

How far a predicted knockdown can be trusted depends on the regime it was
predicted in: the model path (kink-band, tension three-mechanism,
penetration gate, FE), the load, the laminate (UD-dominated or
multidirectional) and whether the wrinkle bends the section
(:attr:`~wrinklefe.analysis.AnalysisResults.bending_dominated`). The
validation residuals are fat-tailed and differ between those regimes, so
one global error band would be too loose for some regimes and too tight
for others.

This module turns the residuals of each regime into a **margin**: the
factor ``m`` to multiply a predicted knockdown by so that a stated
fraction (:data:`COVERAGE`, 95%) of that regime's validation cases would
have been conservative::

    measured_kd >= m * predicted_kd   for 95% of the regime's cases

``m`` is the 5th percentile of ``measured / predicted``, capped at 1 so a
margin never raises a knockdown. The percentile comes from a
**two-component Gaussian mixture** fitted to ``ln(measured / predicted)``,
not from a normal assumption. A mixture is the model Elhajjar (2025) used
for the fat-tailed, multi-modal strength distributions that wrinkles
produce, and it reduces to a normal when the residuals are unimodal.

A regime with fewer than :data:`MIN_CASES` validation cases reports
*insufficient evidence* instead of a margin. Ten is two cases per free
parameter of the two-component mixture (two means, two standard
deviations and a weight). Below that the tail is not identified.

The validation ledger (``tests/test_validation/ledger.json``) is not
shipped with the package, so the margins are fitted once, by
``scripts/fit_margins.py``, and shipped as package data
(``validation_margins.json``, read by :func:`load_margin_table`). A test
re-fits them from the ledger and fails when the shipped table is stale.
After a model change, ``python scripts/validate.py --update`` re-pins the
ledger and regenerates the table in one step.

Reporting only: no knockdown changes. The margin is a point estimate from
a few published studies, not an A- or B-basis value, which would need a
statistical tolerance bound on test data.

References
----------
- Elhajjar, R. (2025). Fat-tailed failure strength distributions and
  manufacturing defects in advanced composites. Scientific Reports,
  15:25977 (Gaussian-mixture model of fat-tailed strength distributions).
- McLachlan, G. & Peel, D. (2000). *Finite Mixture Models*. Wiley (the EM
  fit).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import optimize, stats

if TYPE_CHECKING:
    from wrinklefe.analysis import AnalysisConfig, AnalysisResults

__all__ = [
    "COVERAGE",
    "MIN_CASES",
    "SIGMA_FLOOR",
    "N_COMPONENTS",
    "UD_DOMINATED_FRACTION",
    "MODEL_PATHS",
    "MARGIN_DEFINITION",
    "MarginRegime",
    "MarginResidual",
    "MixtureFit",
    "ValidationMargin",
    "fit_gaussian_mixture",
    "fit_margin",
    "build_margin_table",
    "zero_ply_fraction",
    "laminate_class",
    "analytical_path",
    "regime_for",
    "load_margin_table",
    "lookup_margin",
    "margin_for_results",
]

#: Fraction of a regime's validation cases the margined knockdown must be
#: conservative for.
COVERAGE = 0.95

#: Fewest validation cases a regime needs for a margin: two per free
#: parameter of the two-component mixture (two means, two standard
#: deviations and one weight). A regime with fewer reports insufficient
#: evidence.
MIN_CASES = 10

#: Floor on each mixture component's standard deviation, in units of
#: ``ln(measured / predicted)``. A measured knockdown carries the test
#: scatter of a coupon's strength, of the order of 5%, so a component
#: narrower than that is fitting noise. The floor also keeps the mixture
#: likelihood bounded: without it a component can collapse onto one case.
SIGMA_FLOOR = 0.05

#: Mixture components. Two is the smallest mixture that can carry a tail
#: or a second mode, and a fixed count keeps the fit from falling back to
#: a normal on the small samples the regimes have.
N_COMPONENTS = 2

#: A layup is UD-dominated when at least this fraction of its plies are
#: 0-degree (axial) plies. The validation layups fall either side of it
#: with room to spare: the UD datasets have 0.89 to 1.0, the
#: multidirectional ones 0.25 to 0.40.
UD_DOMINATED_FRACTION = 0.75

#: Angle tolerance (degrees) for a 0-degree ply, as
#: ``wrinklefe.analysis._is_unidirectional`` uses.
_AXIAL_TOL_DEG = 5.0

#: The model paths a margin can belong to, with their plain-language
#: names. The first three are the analytical knockdown's paths; the last
#: two are the FE strength outputs.
MODEL_PATHS: dict[str, str] = {
    "kink_band": "kink-band (analytical, Budiansky-Fleck)",
    "three_mechanism": "three-mechanism (analytical)",
    "penetration_gate": "penetration gate (analytical)",
    "fe_retention": "FE first-ply retention (LaRC05)",
    "progressive_damage": "progressive damage (crack-band FE)",
}

_LOADINGS = ("compression", "tension")
_LAMINATES = {"ud": "UD-dominated", "md": "multidirectional"}

STATUS_FITTED = "fitted"
STATUS_INSUFFICIENT = "insufficient_evidence"

#: The package-data file the fitted table ships in.
TABLE_RESOURCE = "validation_margins.json"

#: Version of the table's layout (not of its numbers).
TABLE_FORMAT_VERSION = 1

#: One-paragraph definition carried wherever a margin is reported, so the
#: NCR summary and the results JSON cannot disagree.
MARGIN_DEFINITION = (
    "The validation margin is the factor to multiply the predicted "
    f"knockdown by so that {COVERAGE:.0%} of the validation cases in the "
    "same regime (model path, load, laminate type, bending regime) would "
    f"have been conservative. It is the {1.0 - COVERAGE:.0%} quantile of "
    "measured / predicted knockdown from a two-component Gaussian-mixture "
    "(fat-tail) fit to the regime's residuals, capped at 1. It is a point "
    "estimate from a few published studies, not an A- or B-basis value, "
    "and it changes no number in this report."
)


# ----------------------------------------------------------------------
# Regimes
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MarginRegime:
    """One cell of the regime table.

    Attributes
    ----------
    path : str
        Model path, a key of :data:`MODEL_PATHS`.
    loading : str
        ``"compression"`` or ``"tension"``.
    laminate : str
        ``"ud"`` (UD-dominated) or ``"md"`` (multidirectional); see
        :func:`laminate_class`.
    bending_dominated : bool
        The load-path eccentricity flag of issue #439. Always ``False`` in
        tension, where the flag is not defined.
    """

    path: str
    loading: str
    laminate: str
    bending_dominated: bool

    def __post_init__(self) -> None:
        if self.path not in MODEL_PATHS:
            raise ValueError(
                f"unknown model path {self.path!r}; expected one of "
                f"{sorted(MODEL_PATHS)}"
            )
        if self.loading not in _LOADINGS:
            raise ValueError(
                f"loading must be 'compression' or 'tension', got "
                f"{self.loading!r}"
            )
        if self.laminate not in _LAMINATES:
            raise ValueError(
                f"laminate must be 'ud' or 'md', got {self.laminate!r}"
            )
        if self.loading == "tension" and self.bending_dominated:
            raise ValueError(
                "the bending-dominated flag is defined for compression only"
            )

    @property
    def key(self) -> str:
        """Stable identifier, e.g. ``kink_band/compression/ud/non-bending``."""
        bend = "bending" if self.bending_dominated else "non-bending"
        return f"{self.path}/{self.loading}/{self.laminate}/{bend}"

    @property
    def label(self) -> str:
        """Plain-language description, for reports."""
        bend = (
            "bending-dominated" if self.bending_dominated
            else "not bending-dominated"
        )
        return (
            f"{MODEL_PATHS[self.path]}; {self.loading}; "
            f"{_LAMINATES[self.laminate]}; {bend}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "path": self.path,
            "loading": self.loading,
            "laminate": self.laminate,
            "bending_dominated": self.bending_dominated,
        }


def zero_ply_fraction(angles: Sequence[float]) -> float:
    """Fraction of plies within 5 degrees of the load axis (0 or 180)."""
    if not angles:
        raise ValueError("a layup needs at least one ply")
    axial = 0
    for a in angles:
        off = abs(float(a)) % 180.0
        if min(off, 180.0 - off) <= _AXIAL_TOL_DEG:
            axial += 1
    return axial / len(angles)


def laminate_class(angles: Sequence[float]) -> str:
    """``"ud"`` for a UD-dominated layup, otherwise ``"md"``.

    UD-dominated means at least :data:`UD_DOMINATED_FRACTION` of the
    plies are 0-degree plies.
    """
    return "ud" if zero_ply_fraction(angles) >= UD_DOMINATED_FRACTION else "md"


def analytical_path(config: AnalysisConfig) -> str:
    """The model path that produced ``analytical_knockdown`` for a config.

    A configured penetration gate replaces the other analytical models;
    otherwise tension uses the three-mechanism model and compression the
    kink-band model.
    """
    if getattr(config, "penetration_gate", None) is not None:
        return "penetration_gate"
    if _normalise_loading(config.loading) == "tension":
        return "three_mechanism"
    return "kink_band"


def _normalise_loading(loading: Any) -> str:
    # Anything but tension is compression, as in the analysis itself.
    return "tension" if str(loading).lower().strip() == "tension" else "compression"


def regime_for(
    path: str,
    loading: Any,
    angles: Sequence[float],
    bending_dominated: bool,
) -> MarginRegime:
    """Classify a prediction into its :class:`MarginRegime`.

    ``bending_dominated`` is ignored in tension, where the flag is not
    defined (issue #439).
    """
    load = _normalise_loading(loading)
    return MarginRegime(
        path=path,
        loading=load,
        laminate=laminate_class(angles),
        bending_dominated=bool(bending_dominated) and load == "compression",
    )


# ----------------------------------------------------------------------
# Fat-tail fit
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MixtureFit:
    """A fitted one-dimensional Gaussian mixture."""

    weights: tuple[float, ...]
    means: tuple[float, ...]
    sds: tuple[float, ...]
    log_likelihood: float

    def cdf(self, x: float) -> float:
        """Mixture cumulative distribution at ``x``."""
        return float(sum(
            w * stats.norm.cdf(x, m, s)
            for w, m, s in zip(self.weights, self.means, self.sds, strict=True)
        ))

    def quantile(self, p: float) -> float:
        """Inverse of :meth:`cdf` (``0 < p < 1``)."""
        if not 0.0 < p < 1.0:
            raise ValueError(f"p must be in (0, 1), got {p}")
        lo = min(m - 12.0 * s for m, s in zip(self.means, self.sds, strict=True))
        hi = max(m + 12.0 * s for m, s in zip(self.means, self.sds, strict=True))
        root = optimize.brentq(
            lambda x: self.cdf(x) - p, lo, hi, xtol=1e-12, rtol=1e-12,
        )
        return float(root)


#: Most EM starts :func:`fit_gaussian_mixture` tries.
_MAX_STARTS = 48


def _em(
    y: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    sds: np.ndarray,
    sigma_floor: float,
    tol: float = 1e-10,
    max_iter: int = 2000,
) -> MixtureFit | None:
    """Expectation-maximisation from one start; ``None`` if a component
    empties."""
    n = y.size
    ll_old = -math.inf
    ll = -math.inf
    half_log_2pi = 0.5 * math.log(2.0 * math.pi)
    for _ in range(max_iter):
        z = (y[:, None] - means[None, :]) / sds[None, :]
        log_p = (
            np.log(weights)[None, :] - 0.5 * z * z
            - np.log(sds)[None, :] - half_log_2pi
        )
        top = log_p.max(axis=1, keepdims=True)
        log_norm = top[:, 0] + np.log(np.exp(log_p - top).sum(axis=1))
        ll = float(log_norm.sum())
        resp = np.exp(log_p - log_norm[:, None])
        nk = resp.sum(axis=0)
        if np.any(nk < 1e-9):
            return None
        weights = nk / n
        means = (resp * y[:, None]).sum(axis=0) / nk
        var = (resp * (y[:, None] - means[None, :]) ** 2).sum(axis=0) / nk
        sds = np.maximum(np.sqrt(var), sigma_floor)
        if abs(ll - ll_old) < tol:
            break
        ll_old = ll
    order = np.argsort(means, kind="stable")
    return MixtureFit(
        weights=tuple(float(v) for v in weights[order]),
        means=tuple(float(v) for v in means[order]),
        sds=tuple(float(v) for v in sds[order]),
        log_likelihood=ll,
    )


def fit_gaussian_mixture(
    samples: Sequence[float] | np.ndarray,
    n_components: int = N_COMPONENTS,
    sigma_floor: float = SIGMA_FLOOR,
) -> MixtureFit:
    """Maximum-likelihood Gaussian mixture by EM, deterministically.

    Each start splits the sorted sample into ``n_components`` contiguous
    groups of at least two points (for two components every such split,
    or 48 evenly spaced ones for a large sample; equal-count groups
    otherwise), and the highest-likelihood fit wins.
    No random number is drawn, so the same sample always gives the same
    fit. Component standard deviations are floored at ``sigma_floor``.

    Parameters
    ----------
    samples : sequence of float
        At least ``2 * n_components`` values.
    n_components : int
        Number of Gaussian components (default 2).
    sigma_floor : float
        Smallest allowed component standard deviation (> 0).

    Returns
    -------
    MixtureFit
        Components ordered by mean.
    """
    y = np.sort(np.asarray(samples, dtype=float).ravel())
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    if sigma_floor <= 0.0:
        raise ValueError(f"sigma_floor must be > 0, got {sigma_floor}")
    if y.size < 2 * n_components:
        raise ValueError(
            f"a {n_components}-component mixture needs at least "
            f"{2 * n_components} samples, got {y.size}"
        )
    if not np.all(np.isfinite(y)):
        raise ValueError("samples must be finite")

    if n_components == 2:
        # Every split for the small samples the regimes have; evenly
        # spaced ones for a large sample, where neighbours start alike.
        cuts: list[int] = list(range(2, y.size - 1))
        if len(cuts) > _MAX_STARTS:
            cuts = sorted({int(round(c)) for c in np.linspace(
                2, y.size - 2, _MAX_STARTS)})
        splits = [[k] for k in cuts]
    else:
        splits = [[round(i * y.size / n_components)
                   for i in range(1, n_components)]]

    best: MixtureFit | None = None
    for cut in splits:
        groups = np.split(y, cut)
        weights = np.array([g.size / y.size for g in groups])
        means = np.array([g.mean() for g in groups])
        sds = np.maximum(np.array([g.std() for g in groups]), sigma_floor)
        fit = _em(y, weights, means, sds, sigma_floor)
        if fit is None:
            continue
        if best is None or fit.log_likelihood > best.log_likelihood + 1e-9:
            best = fit
    if best is None:  # pragma: no cover - every start emptied a component
        raise RuntimeError("the Gaussian-mixture fit did not converge")
    return best


def _round(value: float) -> float:
    return round(float(value), 4)


def fit_margin(
    ratios: Sequence[float],
    coverage: float = COVERAGE,
    min_cases: int = MIN_CASES,
) -> dict[str, Any]:
    """Fit one regime's margin from its ``measured / predicted`` ratios.

    Returns the regime's table entry (without its regime fields):
    ``n``, ``status``, ``margin_factor`` (the ``1 - coverage`` quantile of
    the ratio, capped at 1), ``ratio_quantile`` (the uncapped quantile),
    ``normal_ratio_quantile`` (the same quantile under a normal assumption,
    for comparison only), ``min_ratio``, ``n_below_margin`` (cases still
    non-conservative after the margin) and the fitted ``mixture``. Below
    ``min_cases`` only ``n``, ``status`` and ``min_ratio`` are filled.
    """
    r = np.asarray(ratios, dtype=float).ravel()
    if r.size == 0:
        raise ValueError("a regime needs at least one ratio")
    if not np.all(np.isfinite(r)) or np.any(r <= 0.0):
        raise ValueError("ratios must be finite and positive")
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must be in (0, 1), got {coverage}")
    entry: dict[str, Any] = {
        "n": int(r.size),
        "status": STATUS_INSUFFICIENT,
        "margin_factor": None,
        "ratio_quantile": None,
        "normal_ratio_quantile": None,
        "min_ratio": _round(r.min()),
        "n_below_margin": None,
        "mixture": None,
    }
    if r.size < max(min_cases, 2 * N_COMPONENTS):
        return entry

    y = np.log(r)
    p = 1.0 - coverage
    mix = fit_gaussian_mixture(y)
    q = math.exp(mix.quantile(p))
    q_normal = math.exp(
        float(y.mean()) + float(stats.norm.ppf(p)) * float(y.std(ddof=1))
    )
    margin = min(1.0, q)
    entry.update(
        status=STATUS_FITTED,
        margin_factor=_round(margin),
        ratio_quantile=_round(q),
        normal_ratio_quantile=_round(q_normal),
        n_below_margin=int(np.sum(r < _round(margin))),
        mixture={
            "weights": [_round(v) for v in mix.weights],
            "means": [_round(v) for v in mix.means],
            "sds": [_round(v) for v in mix.sds],
        },
    )
    return entry


@dataclass(frozen=True)
class MarginResidual:
    """One validation case's ``measured / predicted`` ratio in a regime."""

    regime: MarginRegime
    ratio: float
    dataset: str
    case: str


def build_margin_table(
    residuals: Iterable[MarginResidual],
    *,
    source: str,
    excluded_datasets: Sequence[str] = (),
) -> dict[str, Any]:
    """Fit every regime that has residuals and assemble the shipped table.

    ``scripts/fit_margins.py`` calls this with the validation ledger's
    residuals and writes the result as :data:`TABLE_RESOURCE`. The table
    is deterministic: regimes are keyed and cases sorted, so the same
    residuals always give the same file.
    """
    by_key: dict[str, list[MarginResidual]] = {}
    regimes: dict[str, MarginRegime] = {}
    for res in residuals:
        by_key.setdefault(res.regime.key, []).append(res)
        regimes[res.regime.key] = res.regime

    table_regimes: dict[str, Any] = {}
    for key in sorted(by_key):
        cases = sorted(by_key[key], key=lambda c: (c.dataset, c.case))
        regime = regimes[key]
        entry: dict[str, Any] = {
            "path": regime.path,
            "loading": regime.loading,
            "laminate": regime.laminate,
            "bending_dominated": regime.bending_dominated,
        }
        entry.update(fit_margin([c.ratio for c in cases]))
        entry["datasets"] = sorted({c.dataset for c in cases})
        entry["cases"] = [
            {"dataset": c.dataset, "case": c.case, "ratio": _round(c.ratio)}
            for c in cases
        ]
        table_regimes[key] = entry

    return {
        "_comment": [
            "GENERATED by scripts/fit_margins.py from the validation ledger "
            "(issue #455). Do not edit by hand.",
            "After a model change: python scripts/validate.py --update "
            "(re-pins the ledger and regenerates this file), or "
            "python scripts/fit_margins.py --write on its own.",
            "ratio = measured / predicted knockdown; margin_factor = the "
            "(1 - coverage) quantile of ratio from a Gaussian-mixture fit "
            "to ln(ratio), capped at 1. See wrinklefe.core.margins.",
        ],
        "format_version": TABLE_FORMAT_VERSION,
        "source": source,
        "excluded_datasets": sorted(excluded_datasets),
        "coverage": COVERAGE,
        "min_cases": MIN_CASES,
        "n_components": N_COMPONENTS,
        "sigma_floor": SIGMA_FLOOR,
        "ud_dominated_fraction": UD_DOMINATED_FRACTION,
        "regimes": table_regimes,
    }


# ----------------------------------------------------------------------
# The shipped table and its lookups
# ----------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_table_text() -> str:
    return (
        resources.files("wrinklefe.core")
        .joinpath(TABLE_RESOURCE)
        .read_text(encoding="utf-8")
    )


def load_margin_table() -> dict[str, Any]:
    """The shipped margin table (a fresh copy on every call)."""
    table: dict[str, Any] = json.loads(_load_table_text())
    return table


@dataclass(frozen=True)
class ValidationMargin:
    """The margin that applies to one prediction.

    Attributes
    ----------
    regime : MarginRegime
        The prediction's regime.
    n : int
        Validation cases in that regime (0 when it has none).
    status : str
        ``"fitted"`` or ``"insufficient_evidence"``.
    margin_factor : float or None
        Factor to multiply the predicted knockdown by (``<= 1``); ``None``
        for insufficient evidence.
    ratio_quantile : float or None
        The fitted ``1 - coverage`` quantile of ``measured / predicted``
        before the cap at 1.
    coverage : float
        Fraction of the regime's cases the margin is fitted to cover.
    min_cases : int
        Fewest cases a regime needs for a margin.
    datasets : tuple of str
        Ledger datasets the regime's cases come from.
    """

    regime: MarginRegime
    n: int
    status: str
    margin_factor: float | None
    ratio_quantile: float | None
    coverage: float
    min_cases: int
    datasets: tuple[str, ...] = ()

    @property
    def fitted(self) -> bool:
        return self.status == STATUS_FITTED and self.margin_factor is not None

    def apply(self, knockdown: float) -> float | None:
        """``knockdown * margin_factor``, or ``None`` without a margin."""
        if not self.fitted or self.margin_factor is None:
            return None
        return float(knockdown) * self.margin_factor

    def describe(self) -> str:
        """One line for a report."""
        if self.fitted:
            return (
                f"margin {self.margin_factor:.2f} "
                f"({self.coverage:.0%} of {self.n} validation cases "
                f"conservative; {self.regime.label})"
            )
        return (
            f"insufficient evidence ({self.n} validation case"
            f"{'' if self.n == 1 else 's'}, at least {self.min_cases} "
            f"needed; {self.regime.label})"
        )

    def to_dict(self, knockdown: float | None = None) -> dict[str, Any]:
        """JSON-ready form; with ``knockdown``, also the margined value."""
        out: dict[str, Any] = {
            "regime": self.regime.to_dict(),
            "status": self.status,
            "n": self.n,
            "min_cases": self.min_cases,
            "coverage": self.coverage,
            "margin_factor": self.margin_factor,
            "ratio_quantile": self.ratio_quantile,
            "datasets": list(self.datasets),
            "definition": MARGIN_DEFINITION,
        }
        if knockdown is not None:
            margined = self.apply(knockdown)
            out["knockdown"] = float(knockdown)
            out["margined_knockdown"] = (
                round(margined, 6) if margined is not None else None
            )
        return out


def lookup_margin(
    regime: MarginRegime, table: dict[str, Any] | None = None,
) -> ValidationMargin:
    """The margin for ``regime`` from the shipped table (or ``table``).

    A regime the table does not list has no validation cases: it reports
    insufficient evidence with ``n = 0``.
    """
    if table is None:
        table = load_margin_table()
    coverage = float(table["coverage"])
    min_cases = int(table["min_cases"])
    entry = table["regimes"].get(regime.key)
    if entry is None:
        return ValidationMargin(
            regime=regime, n=0, status=STATUS_INSUFFICIENT,
            margin_factor=None, ratio_quantile=None,
            coverage=coverage, min_cases=min_cases,
        )
    factor = entry.get("margin_factor")
    quantile = entry.get("ratio_quantile")
    return ValidationMargin(
        regime=regime,
        n=int(entry["n"]),
        status=str(entry["status"]),
        margin_factor=None if factor is None else float(factor),
        ratio_quantile=None if quantile is None else float(quantile),
        coverage=coverage,
        min_cases=min_cases,
        datasets=tuple(entry.get("datasets", ())),
    )


def margin_for_results(results: AnalysisResults) -> ValidationMargin:
    """The margin that applies to ``results.analytical_knockdown``."""
    cfg = results.config
    regime = regime_for(
        analytical_path(cfg), cfg.loading, cfg.angles or [0.0],
        results.bending_dominated,
    )
    return lookup_margin(regime)
