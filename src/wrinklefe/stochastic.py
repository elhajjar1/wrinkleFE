"""Monte-Carlo / Latin-hypercube uncertainty propagation (issue #301).

Every deterministic WrinkleFE output answers "what is the knockdown for
*this* wrinkle geometry?".  In the package's core use case —
dispositioning a manufacturing defect — the geometry is uncertain:
amplitude and wavelength come from a measurement with error, and the
defect population inside one part scatters.  This module propagates
those input distributions through the analysis so the answer becomes
"the model says 0.74–0.82 given my measurement uncertainty" instead of
"the model says 0.78".

The propagation is doubly worthwhile here because the compressive
knockdown ``1 / (1 + theta_eff / gamma_Y)`` is **concave** in the
misalignment angle: by Jensen's inequality, evaluating the model at the
*mean* input overestimates the *mean* knockdown (the fat-tail argument
of Elhajjar 2025) — a bias only sampling can quantify.

.. warning:: **Not A-/B-basis values.**  The percentiles reported here
   are *model-input propagation* statistics: the distribution of the
   deterministic model's output when its geometric inputs are sampled
   from the distributions you supplied.  They are **not** CMH-17
   A-/B-basis allowables, which are one-sided statistical tolerance
   bounds computed from physical *test* data with prescribed confidence
   levels.  Do not present these percentiles as basis values in
   certification paperwork.

Example
-------
::

    from wrinklefe.analysis import AnalysisConfig
    from wrinklefe.stochastic import probabilistic_analysis

    base = AnalysisConfig(amplitude=0.4, wavelength=16.0, width=12.0)
    prob = probabilistic_analysis(
        base,
        {"amplitude": ("normal", 0.4, 0.05)},
        n_samples=1000, seed=42,
    )
    print(prob.summary())
    print(prob.knockdown_percentile(5.0))   # 5th-percentile knockdown

References
----------
- Elhajjar, R. (2025). Scientific Reports, 15:25977 (fat-tail
  statistics from concave knockdown laws).
- CMH-17-1G, Vol. 1, Ch. 8 (statistical basis values — what this
  module deliberately does *not* compute).
"""

from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, fields, replace
from typing import Any

import numpy as np
from scipy import stats
from scipy.stats import qmc

from wrinklefe.analysis import (
    AnalysisConfig,
    AnalysisResults,
    WrinkleAnalysis,
    _resolve_sweep_workers,
    _sweep_run_one,
)

__all__ = [
    "ProbabilisticResults",
    "probabilistic_analysis",
]


# ----------------------------------------------------------------------
# Distribution specs
# ----------------------------------------------------------------------


def _to_ppf(spec: Any, field_name: str):
    """Normalise a distribution spec to a ``ppf(u)`` callable.

    Accepted forms:

    * ``("normal", mean, std)`` — Gaussian; ``std == 0`` degenerates to
      the constant ``mean``.
    * ``("uniform", lo, hi)`` — uniform on ``[lo, hi]``.
    * ``("lognormal", mu, sigma)`` — log-normal where ``mu``/``sigma``
      are the mean and standard deviation of the *underlying normal*
      (the same convention as ``numpy.random.Generator.lognormal``).
    * any object with a ``.ppf`` method — e.g. a frozen
      ``scipy.stats`` distribution such as ``scipy.stats.norm(0.4,
      0.05)``.

    Sampling is implemented uniformly as ``ppf(u)`` with ``u`` drawn
    either i.i.d. uniform (plain Monte-Carlo) or from a Latin-hypercube
    (stratified), which keeps both methods reproducible from one seed.
    """
    if hasattr(spec, "ppf"):
        return spec.ppf
    if not (isinstance(spec, tuple) and len(spec) == 3):
        raise ValueError(
            f"distribution for {field_name!r} must be a 3-tuple "
            "('normal'|'uniform'|'lognormal', a, b) or an object with a "
            f".ppf method (e.g. a frozen scipy.stats distribution); got "
            f"{spec!r}"
        )
    kind, a, b = spec
    kind = str(kind).lower().strip()
    a = float(a)
    b = float(b)
    if kind == "normal":
        if b < 0:
            raise ValueError(
                f"{field_name!r}: normal std must be >= 0, got {b}"
            )
        if b == 0.0:
            return lambda u: np.full_like(np.asarray(u, dtype=float), a)
        return stats.norm(loc=a, scale=b).ppf
    if kind == "uniform":
        if b < a:
            raise ValueError(
                f"{field_name!r}: uniform needs lo <= hi, got ({a}, {b})"
            )
        if b == a:
            return lambda u: np.full_like(np.asarray(u, dtype=float), a)
        return stats.uniform(loc=a, scale=b - a).ppf
    if kind == "lognormal":
        if b < 0:
            raise ValueError(
                f"{field_name!r}: lognormal sigma must be >= 0, got {b}"
            )
        if b == 0.0:
            const = float(np.exp(a))
            return lambda u: np.full_like(
                np.asarray(u, dtype=float), const
            )
        # scipy lognorm(s=sigma, scale=exp(mu)) == numpy lognormal(mu, sigma)
        return stats.lognorm(s=b, scale=float(np.exp(a))).ppf
    raise ValueError(
        f"unknown distribution kind {kind!r} for {field_name!r}; "
        "expected 'normal', 'uniform' or 'lognormal'"
    )


# ----------------------------------------------------------------------
# Results container
# ----------------------------------------------------------------------


@dataclass
class ProbabilisticResults:
    """Sampled-input propagation of the WrinkleFE analysis.

    All arrays share the sample axis (length ``n_samples``).  The
    reported percentiles are **model-input propagation** statistics —
    see the module docstring for why they must not be presented as
    CMH-17 A-/B-basis values.
    """

    base_config: AnalysisConfig
    input_samples: dict[str, np.ndarray]
    knockdown: np.ndarray
    strength_MPa: np.ndarray
    modulus_knockdown: np.ndarray
    n_samples: int
    seed: int | None
    method: str
    analytical_only: bool = True
    results: list[AnalysisResults] | None = field(default=None, repr=False)

    # -- statistics ----------------------------------------------------

    def knockdown_percentile(self, q) -> float | np.ndarray:
        """Percentile(s) of the knockdown sample (``q`` in [0, 100])."""
        out = np.percentile(self.knockdown, q)
        return float(out) if np.isscalar(q) else np.asarray(out)

    def strength_percentile(self, q) -> float | np.ndarray:
        """Percentile(s) of the strength sample (``q`` in [0, 100])."""
        out = np.percentile(self.strength_MPa, q)
        return float(out) if np.isscalar(q) else np.asarray(out)

    @property
    def knockdown_mean(self) -> float:
        return float(np.mean(self.knockdown))

    @property
    def knockdown_std(self) -> float:
        return float(np.std(self.knockdown))

    @property
    def strength_mean(self) -> float:
        return float(np.mean(self.strength_MPa))

    @property
    def strength_std(self) -> float:
        return float(np.std(self.strength_MPa))

    def summary(self) -> str:
        """Multi-line text summary of the propagated statistics."""
        kd_p = np.percentile(self.knockdown, [5.0, 50.0, 95.0])
        s_p = np.percentile(self.strength_MPa, [5.0, 50.0, 95.0])
        lines = [
            "=" * 65,
            "  WrinkleFE Probabilistic Analysis (input propagation)",
            "=" * 65,
            "",
            f"  Samples:            {self.n_samples} "
            f"({self.method}, seed={self.seed})",
            f"  Path:               "
            f"{'analytical' if self.analytical_only else 'full FE'}",
            "  Sampled inputs:",
        ]
        for name, arr in self.input_samples.items():
            lines.append(
                f"    {name:<18} mean={np.mean(arr):.4g}  "
                f"std={np.std(arr):.4g}  "
                f"range=[{arr.min():.4g}, {arr.max():.4g}]"
            )
        lines += [
            "",
            f"  Knockdown:          mean={self.knockdown_mean:.4f}  "
            f"std={self.knockdown_std:.4f}",
            f"    percentiles       P5={kd_p[0]:.4f}  P50={kd_p[1]:.4f}  "
            f"P95={kd_p[2]:.4f}",
            f"  Strength [MPa]:     mean={self.strength_mean:.1f}  "
            f"std={self.strength_std:.1f}",
            f"    percentiles       P5={s_p[0]:.1f}  P50={s_p[1]:.1f}  "
            f"P95={s_p[2]:.1f}",
            "",
            "  NOTE: percentiles above are model-INPUT-propagation",
            "  statistics (the deterministic model driven by sampled",
            "  geometry), NOT CMH-17 A-/B-basis allowables, which are",
            "  tolerance bounds on physical test data. Do not present",
            "  them as basis values.",
            "=" * 65,
        ]
        return "\n".join(lines)

    # -- optional viz ----------------------------------------------------

    def plot(self, show_strength: bool = False):
        """Histogram of the knockdown (or strength) sample plus one
        scatter panel per sampled input for sensitivity screening.

        Returns the matplotlib ``Figure``.  Matplotlib is imported
        lazily so the stochastic module stays import-light.
        """
        import matplotlib.pyplot as plt

        y = self.strength_MPa if show_strength else self.knockdown
        y_label = (
            "strength [MPa]" if show_strength else "knockdown factor"
        )
        n_inputs = len(self.input_samples)
        fig, axes = plt.subplots(
            1, 1 + n_inputs, figsize=(4.0 * (1 + n_inputs), 3.4),
            squeeze=False,
        )
        ax0 = axes[0][0]
        ax0.hist(y, bins=min(40, max(10, self.n_samples // 25)),
                 color="#4878CF", edgecolor="white")
        for q, ls in ((5.0, ":"), (50.0, "--"), (95.0, ":")):
            ax0.axvline(np.percentile(y, q), color="k", linestyle=ls,
                        linewidth=1.0)
        ax0.set_xlabel(y_label)
        ax0.set_ylabel("count")
        ax0.set_title(f"{y_label} (P5/P50/P95 marked)")

        for k, (name, x) in enumerate(self.input_samples.items()):
            ax = axes[0][1 + k]
            ax.plot(x, y, ".", markersize=3, alpha=0.5, color="#4878CF")
            ax.set_xlabel(name)
            ax.set_ylabel(y_label)
            ax.set_title(f"{y_label} vs {name}")
        fig.tight_layout()
        return fig


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def probabilistic_analysis(
    base_config: AnalysisConfig,
    distributions: dict[str, Any],
    n_samples: int = 1000,
    seed: int | None = None,
    method: str = "lhs",
    analytical_only: bool = True,
    n_workers: int = 1,
    keep_results: bool = False,
) -> ProbabilisticResults:
    """Propagate input distributions through the wrinkle analysis.

    Parameters
    ----------
    base_config : AnalysisConfig
        Deterministic baseline; every non-sampled field is held at its
        value here.
    distributions : dict
        Maps :class:`AnalysisConfig` field names to distribution specs:
        ``("normal", mean, std)``, ``("uniform", lo, hi)``,
        ``("lognormal", mu, sigma)`` (parameters of the underlying
        normal), or any object with a ``.ppf`` method (e.g. a frozen
        ``scipy.stats`` distribution).
    n_samples : int
        Number of samples (>= 1).  The analytical path runs ~1000
        samples in seconds.
    seed : int, optional
        Seed for the sampler.  A fixed seed makes the whole analysis
        reproducible (same samples, same percentiles) for both methods.
    method : str
        ``"lhs"`` (default) — Latin-hypercube stratification, lower
        variance per sample; ``"mc"`` — plain Monte-Carlo.
    analytical_only : bool
        Run only the analytical path per sample (default; recommended).
        ``False`` runs the full FE pipeline per sample — prohibitive
        beyond small ``n_samples``; combine with ``n_workers``.
    n_workers : int
        Worker processes for the per-sample runs (``0`` = all cores,
        default ``1`` = in-process), reusing the sweep parallelism of
        issue #260.  Sampling itself always happens in the parent, so
        results are identical for any worker count.
    keep_results : bool
        Also retain the full per-sample :class:`AnalysisResults` list
        on the returned object (memory scales with ``n_samples``;
        mostly useful with ``analytical_only=False``).

    Returns
    -------
    ProbabilisticResults
        Sample arrays, summary statistics and percentile accessors.

    Raises
    ------
    ValueError
        For unknown config fields, malformed distribution specs, or
        sampled values that fail :class:`AnalysisConfig` validation
        (e.g. a normal amplitude distribution wide enough to produce
        negative draws) — fix the distribution rather than relying on
        silent clipping.
    """
    if not isinstance(n_samples, int) or isinstance(n_samples, bool):
        raise ValueError(f"n_samples must be an int >= 1, got {n_samples!r}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if method not in ("lhs", "mc"):
        raise ValueError(
            f"method must be 'lhs' or 'mc', got {method!r}"
        )
    if not distributions:
        raise ValueError(
            "distributions must map at least one AnalysisConfig field "
            "to a distribution spec"
        )
    valid_field_names = {f.name for f in fields(base_config)}
    for name in distributions:
        if name not in valid_field_names:
            raise ValueError(
                f"AnalysisConfig has no field {name!r} to sample"
            )
    n_workers = _resolve_sweep_workers(n_workers)

    names = sorted(distributions.keys())
    ppfs = {name: _to_ppf(distributions[name], name) for name in names}

    # Uniform hypercube samples: stratified (LHS) or i.i.d. (MC), then
    # mapped through each marginal ppf.  Inputs are sampled
    # independently — joint/correlated inputs can be expressed by
    # passing a frozen scipy distribution per field only if marginals
    # suffice; correlation support is out of scope here.
    d = len(names)
    if method == "lhs":
        sampler = qmc.LatinHypercube(d=d, seed=seed)
        u = sampler.random(n=n_samples)  # (n, d) in (0, 1)
    else:
        rng = np.random.default_rng(seed)
        u = rng.random((n_samples, d))
    # Guard the open interval so ppf never sees exactly 0 or 1.
    tiny = np.finfo(float).tiny
    u = np.clip(u, tiny, 1.0 - 1e-16)

    input_samples = {
        name: np.asarray(ppfs[name](u[:, j]), dtype=float)
        for j, name in enumerate(names)
    }

    # Build one config per sample; a validation failure points at the
    # offending sample and field values.
    configs: list[AnalysisConfig] = []
    for i in range(n_samples):
        overrides: dict[str, Any] = {
            name: float(input_samples[name][i]) for name in names
        }
        try:
            configs.append(replace(base_config, **overrides))
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"sample {i} ({overrides}) failed AnalysisConfig "
                f"validation: {exc}. Adjust the distribution (e.g. "
                "tighten the std or switch to a lognormal) so draws "
                "stay in the physical range."
            ) from exc

    # Run the samples — sequentially, or over the #260 process pool.
    if n_workers == 1:
        results = [
            WrinkleAnalysis(cfg).run(analytical_only=analytical_only)
            for cfg in configs
        ]
    else:
        executor = ProcessPoolExecutor(max_workers=n_workers)
        try:
            results = list(
                executor.map(
                    _sweep_run_one,
                    configs,
                    itertools.repeat(analytical_only),
                )
            )
        except BaseException:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

    knockdown = np.array(
        [r.analytical_knockdown for r in results], dtype=float
    )
    strength = np.array(
        [r.analytical_strength_MPa for r in results], dtype=float
    )
    modulus = np.array(
        [r.analytical_modulus_knockdown for r in results], dtype=float
    )

    return ProbabilisticResults(
        base_config=base_config,
        input_samples=input_samples,
        knockdown=knockdown,
        strength_MPa=strength,
        modulus_knockdown=modulus,
        n_samples=n_samples,
        seed=seed,
        method=method,
        analytical_only=analytical_only,
        results=list(results) if keep_results else None,
    )
