"""Stochastic knockdown: propagating measurement uncertainty, not guessing.

Every deterministic run answers "what is the knockdown for *this* wrinkle
geometry?". On a real NCR the geometry is uncertain: the amplitude and
wavelength come off a micrograph with measurement error.
``probabilistic_analysis`` samples the inputs (Latin hypercube by
default) and pushes each sample through the analysis, so the answer
becomes a distribution instead of a point.

It is also worth doing for a second reason: the knockdown law is
nonlinear in the geometry, so a single run at the *mean* measurement is
not the mean of the runs, and it is nowhere near the low tail a
conservative disposition actually reads. This script measures both gaps
rather than assuming their size or sign.

The percentiles here are model-INPUT-propagation statistics — the
deterministic model driven by sampled geometry — NOT CMH-17 A-/B-basis
allowables, which are tolerance bounds on physical test data. Do not
present them as basis values.

Expected runtime: ~15 s (analytical path, two 300-sample seeded runs).
Expected output:  the propagation summary, a percentile table, the
                  mean-vs-tail comparison, an LHS-vs-Monte-Carlo
                  comparison, and a rank-correlation sensitivity screen.
"""

import numpy as np

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.stochastic import probabilistic_analysis

N_SAMPLES = 300
SEED = 11

# The measurement: A = 0.30 +/- 0.04 mm and lambda = 16.0 +/- 1.5 mm,
# each quoted as a normal 1-sigma from the micrograph.
base = AnalysisConfig(
    amplitude=0.30, wavelength=16.0, width=12.0,
    morphology="stack", loading="compression",
)
distributions = {
    "amplitude": ("normal", 0.30, 0.04),
    "wavelength": ("normal", 16.0, 1.5),
}

prob = probabilistic_analysis(
    base, distributions, n_samples=N_SAMPLES, seed=SEED, method="lhs",
)
print(prob.summary())

# Percentiles, including the low tail a conservative disposition reads.
print()
print("=" * 65)
print("  Knockdown percentiles (input propagation, NOT basis values)")
print("=" * 65)
print(f"\n{'percentile':>12}{'knockdown':>13}{'strength (MPa)':>18}")
print("-" * 65)
for q in (1.0, 5.0, 10.0, 50.0, 90.0, 95.0, 99.0):
    print(f"{'P' + format(q, '.0f'):>12}"
          f"{prob.knockdown_percentile(q):>13.4f}"
          f"{prob.strength_percentile(q):>18.1f}")

# One run at the mean geometry vs the distribution it stands in for.
deterministic = WrinkleAnalysis(base).run(analytical_only=True)
point = deterministic.analytical_knockdown
p5 = prob.knockdown_percentile(5.0)
print()
print("=" * 65)
print("  What the single deterministic run misses")
print("=" * 65)
print(f"\n  knockdown at the mean geometry: {point:.4f}")
print(f"  mean of the sampled knockdowns: {prob.knockdown_mean:.4f}"
      f"   ({prob.knockdown_mean - point:+.4f})")
print(f"  P5 of the sampled knockdowns:   {p5:.4f}"
      f"   ({p5 - point:+.4f})")
print("\n  The model is nonlinear in the geometry, so evaluating it at the")
print("  mean measurement is not the same as averaging the evaluations —")
print("  the size and the sign of that gap depend on the curvature of the")
print("  knockdown law over your input range, which is what the sampling")
print("  measures. The gap that matters for a disposition is the second")
print("  one: the low tail sits well below the single-point answer, and a")
print("  deterministic run reports nothing about it at all.")

# Sampling method: LHS stratifies, plain Monte-Carlo does not. Same seed,
# same inputs — the difference is the sampler's variance, not the model.
mc = probabilistic_analysis(
    base, distributions, n_samples=N_SAMPLES, seed=SEED, method="mc",
)
print()
print("=" * 65)
print("  Latin hypercube vs plain Monte-Carlo (same seed, same n)")
print("=" * 65)
print(f"\n{'method':>10}{'mean KD':>12}{'std':>10}{'P5':>10}{'P95':>10}")
print("-" * 65)
for label, run in (("lhs", prob), ("mc", mc)):
    print(f"{label:>10}{run.knockdown_mean:>12.4f}{run.knockdown_std:>10.4f}"
          f"{run.knockdown_percentile(5.0):>10.4f}"
          f"{run.knockdown_percentile(95.0):>10.4f}")
print("\n  LHS stratifies each marginal, so it reaches the same statistics")
print("  with lower sampler variance per sample. Both are reproducible:")
print("  a fixed seed gives the same samples and the same percentiles.")

# Which input drives the scatter? A rank correlation is enough to screen.
print()
print("=" * 65)
print("  Input sensitivity screen (Spearman rank correlation vs KD)")
print("=" * 65)
print()
order = np.argsort(prob.knockdown)
ranks_kd = np.empty(N_SAMPLES)
ranks_kd[order] = np.arange(N_SAMPLES)
for name, samples in prob.input_samples.items():
    order_in = np.argsort(samples)
    ranks_in = np.empty(N_SAMPLES)
    ranks_in[order_in] = np.arange(N_SAMPLES)
    rho = float(np.corrcoef(ranks_in, ranks_kd)[0, 1])
    print(f"  {name:<14} rho = {rho:+.3f}")
print("\n  Amplitude and wavelength enter the angle as A/lambda, so they")
print("  push the knockdown in opposite directions.")

print()
print("Equivalent CLI: wrinklefe stochastic "
      "--distribution amplitude:normal:0.30:0.04 \\")
print("                    --distribution wavelength:normal:16.0:1.5 \\")
print("                    --n-samples 300 --seed 11")
