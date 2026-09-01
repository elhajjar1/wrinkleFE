"""Monte-Carlo / LHS uncertainty propagation (issue #301).

``probabilistic_analysis`` samples AnalysisConfig fields from user
distributions, runs the analytical path per sample, and reports
percentile knockdowns/strengths — model-input propagation statistics,
deliberately *not* CMH-17 basis values (the summary carries the
disclaimer, pinned here).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.stochastic import probabilistic_analysis


def _base(**overrides) -> AnalysisConfig:
    cfg = dict(amplitude=0.4, wavelength=16.0, width=12.0,
               analytical_only=True)
    cfg.update(overrides)
    return AnalysisConfig(**cfg)


def _ud_gate_base() -> AnalysisConfig:
    from wrinklefe.core.penetration_gate import GATE_LI2025_VACBAG

    return AnalysisConfig(
        amplitude=0.75, wavelength=12.9, width=6.45,
        angles=[0.0] * 14, ply_thickness=0.44, morphology="graded",
        penetration_gate=GATE_LI2025_VACBAG, analytical_only=True,
    )


class TestReproducibilityAndDegeneracy:

    def test_fixed_seed_reproduces_everything(self):
        kwargs = dict(
            distributions={"amplitude": ("normal", 0.4, 0.05)},
            n_samples=64, seed=42,
        )
        a = probabilistic_analysis(_base(), **kwargs)
        b = probabilistic_analysis(_base(), **kwargs)
        np.testing.assert_array_equal(a.knockdown, b.knockdown)
        np.testing.assert_array_equal(a.strength_MPa, b.strength_MPa)
        np.testing.assert_array_equal(
            a.input_samples["amplitude"], b.input_samples["amplitude"]
        )

    @pytest.mark.parametrize("method", ["lhs", "mc"])
    @pytest.mark.parametrize(
        "dist",
        [("normal", 0.4, 0.0), ("uniform", 0.4, 0.4), ("lognormal",
                                                       float(np.log(0.4)),
                                                       0.0)],
    )
    def test_zero_variance_reproduces_deterministic(self, method, dist):
        """Degenerate distributions == the deterministic run, exactly."""
        prob = probabilistic_analysis(
            _base(), {"amplitude": dist}, n_samples=8, seed=1,
            method=method,
        )
        det = WrinkleAnalysis(
            _base(amplitude=0.4)
        ).run(analytical_only=True)
        # lognormal exp(log(0.4)) reproduces 0.4 to float rounding.
        rtol = 0.0 if dist[0] != "lognormal" else 1e-12
        np.testing.assert_allclose(
            prob.knockdown, det.analytical_knockdown, rtol=rtol
        )
        np.testing.assert_allclose(
            prob.strength_MPa, det.analytical_strength_MPa, rtol=rtol
        )


class TestStatistics:

    def test_percentiles_monotone_and_bracket_median(self):
        prob = probabilistic_analysis(
            _ud_gate_base(),
            {"amplitude": ("normal", 0.75, 0.08),
             "wavelength": ("normal", 12.9, 1.0)},
            n_samples=400, seed=7,
        )
        p5, p50, p95 = prob.knockdown_percentile([5.0, 50.0, 95.0])
        assert p5 < p50 < p95
        s5, s50, s95 = prob.strength_percentile([5.0, 50.0, 95.0])
        assert s5 < s50 < s95
        # Nonzero variance must actually spread the output.
        assert prob.knockdown_std > 0.01

    def test_amplitude_knockdown_anticorrelated(self):
        """Physical sanity: larger sampled amplitude -> lower knockdown."""
        prob = probabilistic_analysis(
            _ud_gate_base(), {"amplitude": ("normal", 0.75, 0.08)},
            n_samples=300, seed=11,
        )
        r = np.corrcoef(prob.input_samples["amplitude"], prob.knockdown)
        assert r[0, 1] < -0.9

    def test_summary_carries_basis_value_disclaimer(self):
        prob = probabilistic_analysis(
            _base(), {"amplitude": ("normal", 0.4, 0.03)},
            n_samples=16, seed=0,
        )
        text = prob.summary()
        assert "NOT CMH-17 A-/B-basis" in text
        assert "P5=" in text and "P95=" in text

    def test_scipy_frozen_distribution_accepted(self):
        frozen = stats.norm(loc=0.4, scale=0.05)
        a = probabilistic_analysis(
            _base(), {"amplitude": frozen}, n_samples=64, seed=42,
        )
        b = probabilistic_analysis(
            _base(), {"amplitude": ("normal", 0.4, 0.05)},
            n_samples=64, seed=42,
        )
        np.testing.assert_allclose(a.knockdown, b.knockdown, rtol=1e-12)

    def test_keep_results_returns_full_objects(self):
        prob = probabilistic_analysis(
            _base(), {"amplitude": ("normal", 0.4, 0.03)},
            n_samples=5, seed=0, keep_results=True,
        )
        assert prob.results is not None and len(prob.results) == 5
        assert prob.results[0].analytical_knockdown == prob.knockdown[0]


class TestParallelAndPlot:

    def test_parallel_identical_to_sequential(self):
        """Sampling happens in the parent, so worker count cannot change
        the numbers (real cross-process run)."""
        kwargs = dict(
            distributions={"amplitude": ("normal", 0.75, 0.08)},
            n_samples=60, seed=3,
        )
        seq = probabilistic_analysis(_ud_gate_base(), **kwargs, n_workers=1)
        par = probabilistic_analysis(_ud_gate_base(), **kwargs, n_workers=2)
        np.testing.assert_array_equal(seq.knockdown, par.knockdown)

    def test_plot_returns_figure(self):
        import matplotlib

        matplotlib.use("Agg")
        prob = probabilistic_analysis(
            _base(), {"amplitude": ("normal", 0.4, 0.03)},
            n_samples=32, seed=0,
        )
        fig = prob.plot()
        # Histogram panel + one scatter per sampled input.
        assert len(fig.axes) == 2
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestValidation:

    def test_unknown_field_rejected(self):
        with pytest.raises(ValueError, match="no field"):
            probabilistic_analysis(
                _base(), {"not_a_field": ("normal", 1.0, 0.1)},
                n_samples=4,
            )

    def test_malformed_spec_rejected(self):
        with pytest.raises(ValueError, match="3-tuple"):
            probabilistic_analysis(
                _base(), {"amplitude": ("normal", 0.4)}, n_samples=4,
            )
        with pytest.raises(ValueError, match="unknown distribution"):
            probabilistic_analysis(
                _base(), {"amplitude": ("weibull", 0.4, 0.1)}, n_samples=4,
            )

    def test_unphysical_sample_fails_loudly(self):
        """A distribution wide enough to draw an invalid value must
        raise a targeted error, not silently clip."""
        with pytest.raises(ValueError, match="failed AnalysisConfig"):
            probabilistic_analysis(
                _base(), {"wavelength": ("normal", 1.0, 5.0)},
                n_samples=64, seed=0,
            )

    @pytest.mark.parametrize("bad_n", [0, -1, 2.5])
    def test_bad_n_samples_rejected(self, bad_n):
        with pytest.raises(ValueError, match="n_samples"):
            probabilistic_analysis(
                _base(), {"amplitude": ("normal", 0.4, 0.01)},
                n_samples=bad_n,
            )

    def test_bad_method_rejected(self):
        with pytest.raises(ValueError, match="method"):
            probabilistic_analysis(
                _base(), {"amplitude": ("normal", 0.4, 0.01)},
                n_samples=4, method="sobol",
            )

    def test_empty_distributions_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            probabilistic_analysis(_base(), {}, n_samples=4)


class TestKnockdownCurvature:
    """Pin the curvature claims the module docstring now makes (#394).

    The docstring used to argue the compressive law is *concave*, so a
    run at the mean input would overestimate the mean knockdown. It is
    convex, and the measured gap has the opposite sign. These tests
    measure the curvature rather than restating it, and they also pin
    the two places where the sign is *not* global — which is why the
    docstring tells readers to use the percentiles, not the gap's sign.
    """

    @staticmethod
    def _knockdown_vs_amplitude(amplitudes, **cfg_overrides):
        return np.array([
            WrinkleAnalysis(
                AnalysisConfig(
                    amplitude=float(a), wavelength=16.0, width=12.0,
                    morphology="stack", **cfg_overrides,
                )
            ).run(analytical_only=True).analytical_knockdown
            for a in amplitudes
        ])

    def test_compressive_law_is_convex_over_realistic_amplitudes(self):
        """Second differences of KD(A) are strictly positive in compression."""
        amps = np.linspace(0.05, 1.0, 48)
        kd = self._knockdown_vs_amplitude(amps, loading="compression")
        second_diff = np.diff(kd, 2)
        assert np.all(second_diff > 0.0), (
            "compressive kink-band knockdown is documented as convex in "
            f"amplitude; found {(second_diff <= 0).sum()} non-convex "
            "sample(s)"
        )

    def test_tension_path_is_locally_concave_somewhere(self):
        """The convexity is path-specific: tension flips it at low amplitude."""
        amps = np.linspace(0.05, 1.0, 48)
        kd = self._knockdown_vs_amplitude(amps, loading="tension")
        second_diff = np.diff(kd, 2)
        assert np.any(second_diff < 0.0), (
            "tension knockdown is documented as concave over part of the "
            "range; no concave sample found"
        )

    def test_penetration_gate_flips_curvature_below_its_clamp(self):
        """The gate's D/T ramp is concave; above the clamp convexity returns."""
        from wrinklefe.core.penetration_gate import GATE_LI2025_VACBAG

        amps = np.linspace(0.05, 1.0, 48)
        kd = self._knockdown_vs_amplitude(
            amps, loading="compression", penetration_gate=GATE_LI2025_VACBAG,
        )
        second_diff = np.diff(kd, 2)
        assert np.any(second_diff < 0.0), (
            "gated knockdown is documented as concave below the D/T clamp"
        )
        assert second_diff[-1] > 0.0, (
            "gated knockdown is documented as convex again above the clamp"
        )

    def test_measured_mean_gap_sign_matches_the_docstring(self):
        """On the example-14 configuration the sampled mean sits above the
        deterministic point, and P5 sits well below it."""
        base = AnalysisConfig(
            amplitude=0.30, wavelength=16.0, width=12.0,
            morphology="stack", loading="compression",
        )
        prob = probabilistic_analysis(
            base,
            {"amplitude": ("normal", 0.30, 0.04),
             "wavelength": ("normal", 16.0, 1.5)},
            n_samples=300, seed=11, method="lhs",
        )
        point = WrinkleAnalysis(base).run(
            analytical_only=True
        ).analytical_knockdown

        mean_gap = prob.knockdown_mean - point
        assert mean_gap > 0.0, (
            "convex law + Jensen => sampled mean at or above the "
            f"deterministic point; got {mean_gap:+.5f}"
        )
        # Small, as the docstring says — the mean gap is not the point.
        assert mean_gap < 0.01

        # The gap that matters for a disposition, and the reason the
        # docstring anchors on percentiles rather than the mean.
        p5_gap = prob.knockdown_percentile(5.0) - point
        assert p5_gap < -0.01
        assert abs(p5_gap) > 10.0 * mean_gap


class TestModulusVectorizationEquivalence:
    """The batched `_laminate_modulus_knockdown` (the enabler for
    interactive sampling) must match the original per-(ply, x) loop."""

    def test_batched_matches_loop_reference(self):
        import math

        from wrinklefe.analysis import (
            _laminate_modulus_knockdown,
            _plane_stress_qbar_tilted,
        )

        rng = np.random.default_rng(5)
        n_w, n_x, n_p = 2, 40, 6
        slope_field = rng.uniform(-0.3, 0.3, (n_w, n_x))
        ply_decays = rng.uniform(0.0, 1.0, (n_p, n_w, n_x))
        angles = [0.0, 45.0, -45.0, 90.0, 30.0, 0.0]
        # A representative orthotropic 3D stiffness from the library.
        from wrinklefe.core.laminate import Laminate
        from wrinklefe.core.material import MaterialLibrary

        mat = MaterialLibrary().get("IM7_8552")
        lam = Laminate.from_angles(angles, mat, ply_thickness=0.183)
        C = mat.stiffness_matrix
        E_x0 = lam.Ex

        got = _laminate_modulus_knockdown(
            slope_field, ply_decays, angles, C, 0.183, E_x0
        )

        # Original loop implementation as the oracle.
        phis = [math.radians(a) for a in angles]
        inv_E = np.empty(n_x)
        for xi in range(n_x):
            a_mat = np.zeros((3, 3))
            for p, phi in enumerate(phis):
                slope_p = float(
                    np.dot(slope_field[:, xi], ply_decays[p, :, xi])
                )
                theta = math.atan(abs(slope_p))
                a_mat += _plane_stress_qbar_tilted(C, phi, theta) * 0.183
            inv_E[xi] = np.linalg.inv(a_mat)[0, 0] * (n_p * 0.183)
        expected = float(1.0 / np.mean(inv_E) / E_x0)

        assert got == pytest.approx(expected, rel=1e-10)
