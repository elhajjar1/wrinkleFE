"""The sign and size of the Jensen gap for the strength laws.

Elhajjar (2025, Sci. Rep. 15:25977) measured strength as *concave* in the
wrinkle depth ratio D/T, so defect scatter lowers the mean strength
(``E[f(X)] <= f(E[X])``) and fattens the left tail. These tests pin what
WrinkleFE's own laws do under scatter, so the docstrings cannot drift
from it again:

* the Budiansky-Fleck kink-band law is **convex** in the misalignment
  angle: angle scatter raises the mean knockdown, by a small amount, and
  its risk shows in the low percentiles instead;
* the same holds through the full analytical pipeline;
* the penetration gate is **concave** in D/T below its threshold, the
  sign Elhajjar (2025) measured, so only a depth-aware law reproduces it.
"""

from __future__ import annotations

import numpy as np
import pytest

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary
from wrinklefe.core.penetration_gate import (
    GATE_LI2025_VACBAG,
    penetration_gate_kd,
)
from wrinklefe.failure.kinkband import BudianskyFleckKinkBand
from wrinklefe.stochastic import probabilistic_analysis

_CV = 0.20  # 20 % scatter on the sampled input
_GAMMA_Y = 0.02


def _kink_kd(theta_rad: np.ndarray) -> np.ndarray:
    return np.array([
        BudianskyFleckKinkBand(theta_eff=float(t)).knockdown(gamma_Y=_GAMMA_Y)
        for t in theta_rad
    ])


def _scatter(mean: float, n: int = 20_000, seed: int = 0) -> np.ndarray:
    s = np.random.default_rng(seed).normal(mean, _CV * mean, n)
    return s[s > 0.0]


def test_kink_band_law_is_convex_in_angle():
    theta = np.radians(np.linspace(0.5, 30.0, 60))
    assert np.all(np.diff(_kink_kd(theta), 2) > 0.0)


@pytest.mark.parametrize("theta_deg", [2.0, 5.0, 10.0, 17.5, 25.0])
def test_angle_scatter_raises_the_mean_slightly_and_drops_the_tail(theta_deg):
    """Measured: gap +0.002 to +0.006, P5 0.01-0.06 below the point value."""
    samples = _kink_kd(np.radians(_scatter(theta_deg)))
    point = _kink_kd(np.radians([theta_deg]))[0]
    gap = samples.mean() - point
    assert 0.0 < gap < 0.01
    assert np.percentile(samples, 5) < point - 0.005


@pytest.mark.parametrize("amplitude", [0.05, 0.24])
def test_pipeline_angle_scatter_has_a_small_positive_gap(amplitude):
    """The full analytical pipeline (Dataset A recipe, fixed wavelength):
    measured gap about +0.003 to +0.004, P5 0.03-0.05 below."""
    lam = max(19.9 * amplitude, 8.2)
    base = AnalysisConfig(
        amplitude=amplitude, wavelength=lam, width=0.75 * lam,
        morphology="uniform", loading="compression",
        material=MaterialLibrary().get("T700_2510"),
        angles=[0, 45, 90, -45, 0, 45, -45, 0, 0, -45, 45, 0, -45, 90, 45, 0],
        ply_thickness=0.152, analytical_only=True,
    )
    point = float(WrinkleAnalysis(base).run().analytical_knockdown)
    prob = probabilistic_analysis(
        base, {"amplitude": ("normal", amplitude, _CV * amplitude)},
        n_samples=200, seed=0,
    )
    gap = prob.knockdown_mean - point
    assert 0.0 < gap < 0.01
    assert prob.knockdown_percentile(5.0) < point - 0.02


@pytest.mark.parametrize("dt", [0.06, 0.08])
def test_gate_is_concave_below_its_threshold(dt):
    """Below dt0 the gate gives Elhajjar's sign: scatter LOWERS the mean
    (measured -0.017 at D/T = 0.08, theta = 20 deg)."""
    samples = penetration_gate_kd(20.0, _scatter(dt), GATE_LI2025_VACBAG)
    point = penetration_gate_kd(20.0, dt, GATE_LI2025_VACBAG)
    assert samples.mean() < point


def test_gate_floor_reverses_the_sign_above_threshold():
    """At and above dt0 the gate is clamped at the angle floor, so scatter
    can only raise the mean, and the lower tail collapses onto the floor."""
    dt0 = GATE_LI2025_VACBAG.dt0
    samples = penetration_gate_kd(20.0, _scatter(1.15 * dt0), GATE_LI2025_VACBAG)
    point = penetration_gate_kd(20.0, 1.15 * dt0, GATE_LI2025_VACBAG)
    assert samples.mean() > point
    assert np.percentile(samples, 5) == pytest.approx(point)
