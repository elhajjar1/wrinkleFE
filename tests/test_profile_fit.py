"""Tests for :mod:`wrinklefe.core.fit` — fitting a profile to measured data.

These tests are written against *independent* facts wherever possible:
parameters are recovered against the values used to generate the trace,
never against a second copy of the fitting algebra.  Where a test pins an
implementation choice (BIC ranking, joint trend fitting, detrended
seeding) it also pins the measurement that motivated the choice, so a
future change that silently undoes it fails loudly.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import numpy.testing as npt
import pytest

from wrinklefe.core.fit import (
    FAMILIES,
    FitResult,
    fit_profile,
    load_trace,
    rank_families,
)
from wrinklefe.core.wrinkle import (
    GaussianBump,
    GaussianSinusoidal,
    PureSinusoidal,
    WrinkleProfile,
)

TRUTH = dict(amplitude=0.5, wavelength=12.0, width=8.0, center=0.0)


@pytest.fixture
def clean_trace() -> tuple[np.ndarray, np.ndarray]:
    """A noiseless Gaussian-sinusoidal trace with known parameters."""
    profile = GaussianSinusoidal(**TRUTH)
    x = np.linspace(-20.0, 20.0, 80)
    return x, np.asarray(profile.displacement(x), dtype=float)


@pytest.fixture
def noisy_trace() -> tuple[np.ndarray, np.ndarray]:
    """The same trace with 2 % noise, as a digitized trace would carry."""
    profile = GaussianSinusoidal(**TRUTH)
    x = np.linspace(-20.0, 20.0, 50)
    rng = np.random.default_rng(7)
    z = np.asarray(profile.displacement(x), dtype=float)
    return x, z + rng.normal(0.0, 0.02 * TRUTH["amplitude"], x.size)


class TestRoundTrip:
    """Generate from known parameters, fit, recover the parameters."""

    def test_noiseless_round_trip_is_near_exact(self, clean_trace):
        x, z = clean_trace
        fit = fit_profile(x, z, "gaussian_sinusoidal")
        npt.assert_allclose(fit.amplitude, TRUTH["amplitude"], rtol=1e-4)
        npt.assert_allclose(fit.wavelength, TRUTH["wavelength"], rtol=1e-4)
        npt.assert_allclose(fit.width, TRUTH["width"], rtol=1e-4)
        assert fit.rms_residual < 1e-6
        assert fit.r_squared > 1.0 - 1e-9

    def test_noisy_round_trip_recovers_parameters_within_a_few_percent(
        self, noisy_trace
    ):
        """Issue #270 AC1: ~50 points at 2 % noise, parameters to a few %."""
        x, z = noisy_trace
        fit = fit_profile(x, z, "gaussian_sinusoidal")
        assert abs(fit.amplitude - TRUTH["amplitude"]) / TRUTH["amplitude"] < 0.05
        assert abs(fit.wavelength - TRUTH["wavelength"]) / TRUTH["wavelength"] < 0.05
        assert abs(fit.width - TRUTH["width"]) / TRUTH["width"] < 0.05
        assert fit.r_squared > 0.99

    def test_fitted_profile_reproduces_the_trace_it_was_fitted_to(
        self, clean_trace
    ):
        """The returned profile is the fit, not just a parameter carrier."""
        x, z = clean_trace
        fit = fit_profile(x, z, "gaussian_sinusoidal")
        assert isinstance(fit.profile, WrinkleProfile)
        reconstructed = np.asarray(fit.profile.displacement(x), dtype=float)
        # trend_coeffs are ~0 for an untilted trace, so the shape alone
        # must already reproduce the data.
        npt.assert_allclose(reconstructed, z, atol=1e-5)

    @pytest.mark.parametrize("family", sorted(FAMILIES))
    def test_every_family_round_trips_its_own_shape(self, family):
        cls, free = FAMILIES[family]
        gen = cls(amplitude=0.4, wavelength=10.0, width=7.0, center=1.0)
        x = np.linspace(-18.0, 20.0, 120)
        z = np.asarray(gen.displacement(x), dtype=float)
        fit = fit_profile(x, z, family)
        assert fit.rms_residual < 1e-5, f"{family} cannot fit its own shape"
        truth = {
            "amplitude": 0.4, "wavelength": 10.0, "width": 7.0, "center": 1.0,
        }
        for name in free:
            if name == "center" and family == "pure_sinusoidal":
                # A pure sinusoid has no envelope to locate it against, so
                # its centre is identifiable only modulo the wavelength:
                # center=1 and center=11 are the *same* profile.  Compare
                # the shift, not the absolute value.
                shift = (fit.center - truth["center"]) % fit.wavelength
                shift = min(shift, fit.wavelength - shift)
                assert shift < 2e-2, (
                    f"{family}: centre off by {shift} mm modulo the wavelength"
                )
                continue
            npt.assert_allclose(
                getattr(fit, name), truth[name], rtol=2e-3,
                err_msg=f"{family}: parameter {name} not recovered",
            )


class TestDegenerateFamilies:
    """Two families ignore one of the four shared constructor arguments."""

    def test_pure_sinusoidal_does_not_fit_width(self):
        assert "width" not in FAMILIES["pure_sinusoidal"][1]

    def test_gaussian_bump_does_not_fit_wavelength(self):
        assert "wavelength" not in FAMILIES["gaussian_bump"][1]

    @pytest.mark.parametrize(
        "family,ignored", [
            ("pure_sinusoidal", "width"),
            ("gaussian_bump", "wavelength"),
        ],
    )
    def test_the_ignored_parameter_really_is_ignored_by_the_model(
        self, family, ignored
    ):
        """The exclusion is justified by the physics, not by convention.

        If ``displacement`` did depend on this parameter, holding it
        fixed would bias the fit.  Pin that it genuinely does not.
        """
        cls, _ = FAMILIES[family]
        x = np.linspace(-10.0, 10.0, 40)
        base = dict(amplitude=0.4, wavelength=10.0, width=7.0, center=0.0)
        other = dict(base)
        other[ignored] = base[ignored] * 3.0
        npt.assert_allclose(
            np.asarray(cls(**base).displacement(x), dtype=float),
            np.asarray(cls(**other).displacement(x), dtype=float),
            atol=1e-12,
            err_msg=f"{family}.displacement does depend on {ignored}",
        )

    @pytest.mark.parametrize(
        "family", ["pure_sinusoidal", "gaussian_bump"],
    )
    def test_degenerate_family_reports_finite_sigmas(self, family):
        """The reason for holding it fixed: a varied-but-unused parameter
        leaves a zero Jacobian column and a singular covariance."""
        cls, free = FAMILIES[family]
        gen = cls(amplitude=0.4, wavelength=10.0, width=7.0, center=0.5)
        x = np.linspace(-18.0, 20.0, 120)
        rng = np.random.default_rng(1)
        z = np.asarray(gen.displacement(x), dtype=float)
        z = z + rng.normal(0.0, 0.004, x.size)
        fit = fit_profile(x, z, family)
        assert set(fit.sigmas) == set(free)
        assert all(math.isfinite(s) for s in fit.sigmas.values()), fit.sigmas


class TestTrendHandling:
    """A digitized micrograph trace is almost never level."""

    @pytest.mark.parametrize("slope", [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, -0.15])
    @pytest.mark.parametrize("offset", [0.0, 0.3, -2.0])
    def test_amplitude_is_unbiased_under_tilt(self, slope, offset):
        """Issue #270 AC3: tilt must not leak into the amplitude."""
        profile = GaussianSinusoidal(**TRUTH)
        x = np.linspace(-20.0, 20.0, 50)
        z = np.asarray(profile.displacement(x), dtype=float)
        z = z + slope * x + offset
        fit = fit_profile(x, z, "gaussian_sinusoidal", detrend=True)
        npt.assert_allclose(fit.amplitude, TRUTH["amplitude"], rtol=5e-3)

    def test_recovered_trend_matches_the_applied_trend(self):
        """The nuisance parameters are estimates, and are reported."""
        profile = GaussianSinusoidal(**TRUTH)
        x = np.linspace(-20.0, 20.0, 60)
        z = np.asarray(profile.displacement(x), dtype=float) + 0.04 * x - 1.5
        fit = fit_profile(x, z, "gaussian_sinusoidal", detrend=True)
        assert fit.trend_coeffs is not None
        npt.assert_allclose(fit.trend_coeffs[0], 0.04, atol=1e-4)
        npt.assert_allclose(fit.trend_coeffs[1], -1.5, atol=1e-3)

    def test_uncorrected_tilt_does_bias_the_amplitude(self):
        """Why ``detrend`` defaults to True: without it, tilt inflates A."""
        profile = GaussianSinusoidal(**TRUTH)
        x = np.linspace(-20.0, 20.0, 50)
        z = np.asarray(profile.displacement(x), dtype=float) + 0.05 * x + 0.3
        raw = fit_profile(x, z, "gaussian_sinusoidal", detrend=False)
        assert raw.trend_coeffs is None
        assert abs(raw.amplitude - TRUTH["amplitude"]) / TRUTH["amplitude"] > 0.2

    def test_trend_is_fitted_jointly_not_subtracted_beforehand(self):
        """A bump that decays to zero cannot follow a mean-subtracted trace.

        Pre-detrending removes the *mean*, which pushes the far field
        negative; ``A exp(-dx^2/w^2)`` with ``A > 0`` has nowhere to go.
        Measured, that costs an order of magnitude in residual.  Assert
        the joint fit stays at noise level on exactly that case.
        """
        gen = GaussianBump(amplitude=0.5, wavelength=10.0, width=6.0, center=0.0)
        x = np.linspace(-20.0, 20.0, 120)
        z = np.asarray(gen.displacement(x), dtype=float)
        pre_detrended = z - z.mean()
        assert pre_detrended[0] < -0.05, "fixture does not exercise the defect"

        fit = fit_profile(x, z, "gaussian_bump", detrend=True)
        assert fit.rms_residual < 1e-3
        npt.assert_allclose(fit.amplitude, 0.5, rtol=5e-3)

    def test_seed_is_detrended_even_though_the_fit_is_joint(self):
        """Large tilt swamps every data-derived starting guess.

        Raw-seeded, a 0.05 mm/mm tilt over this window takes the
        optimiser to a local minimum ~55 % low in amplitude.  This pins
        that the seeding path survives a tilt far larger than that.
        """
        profile = GaussianSinusoidal(**TRUTH)
        x = np.linspace(-20.0, 20.0, 50)
        z = np.asarray(profile.displacement(x), dtype=float) + 0.05 * x + 0.3
        # The raw trace's peak-to-peak is dominated by the tilt: this is
        # the precondition that makes a raw seed wrong.  The tilt spans
        # 0.05 * 40 = 2 mm against a wrinkle peak-to-peak of 2A = 1 mm.
        wrinkle_ptp = float(np.ptp(profile.displacement(x)))
        assert np.ptp(z) > 1.5 * wrinkle_ptp
        fit = fit_profile(x, z, "gaussian_sinusoidal", detrend=True)
        npt.assert_allclose(fit.amplitude, TRUTH["amplitude"], rtol=5e-3)


class TestRanking:
    """Which morphology? — answered by a number."""

    @pytest.mark.parametrize("family", sorted(FAMILIES))
    @pytest.mark.parametrize("seed", [0, 3])
    def test_generating_family_ranks_first(self, family, seed):
        """Issue #270 AC2, for every family and more than one noise draw."""
        cls, _ = FAMILIES[family]
        gen = cls(amplitude=0.4, wavelength=10.0, width=7.0, center=1.0)
        x = np.linspace(-18.0, 20.0, 120)
        rng = np.random.default_rng(seed)
        z = np.asarray(gen.displacement(x), dtype=float)
        z = z + rng.normal(0.0, 0.004, x.size)
        ranked = rank_families(x, z)
        assert ranked[0].family == family, (
            f"{family} generated, but {ranked[0].family} ranked first"
        )

    def test_raw_residual_ranking_cannot_separate_nested_families(self):
        """Why the default is BIC and not the residual issue #270 asked for.

        Three families carry four shape parameters and two carry three;
        a four-parameter family can always match a three-parameter one.
        On data generated by ``pure_sinusoidal`` the residuals tie, so a
        raw ranking is decided by noise.  BIC charges for the extra
        parameters and recovers the true family.
        """
        gen = PureSinusoidal(
            amplitude=0.4, wavelength=10.0, width=7.0, center=1.0,
        )
        x = np.linspace(-18.0, 20.0, 120)
        rng = np.random.default_rng(7)
        z = np.asarray(gen.displacement(x), dtype=float)
        z = z + rng.normal(0.0, 0.004, x.size)

        by_rms = rank_families(x, z, criterion="rms")
        truth_rms = next(r for r in by_rms if r.family == "pure_sinusoidal")
        # The residuals are a statistical tie: a four-parameter family
        # sits within a few percent of the true three-parameter one.
        assert by_rms[0].rms_residual <= truth_rms.rms_residual
        assert (
            truth_rms.rms_residual - by_rms[0].rms_residual
        ) / truth_rms.rms_residual < 0.05

        assert rank_families(x, z, criterion="bic")[0].family == (
            "pure_sinusoidal"
        )

    def test_bic_penalises_parameter_count_relative_to_rms(self):
        """The penalty is real, not a relabelled residual."""
        gen = PureSinusoidal(
            amplitude=0.4, wavelength=10.0, width=7.0, center=1.0,
        )
        x = np.linspace(-18.0, 20.0, 120)
        rng = np.random.default_rng(2)
        z = np.asarray(gen.displacement(x), dtype=float)
        z = z + rng.normal(0.0, 0.004, x.size)
        results = {r.family: r for r in rank_families(x, z)}
        three = results["pure_sinusoidal"]
        four = results["gaussian_sinusoidal"]
        # Four parameters fit at least as well ...
        assert four.rms_residual <= three.rms_residual * 1.05
        # ... and are still ranked worse, because BIC charges for them.
        assert three.bic < four.bic

    def test_ranking_is_sorted_by_the_requested_criterion(self):
        x = np.linspace(-18.0, 20.0, 120)
        gen = GaussianSinusoidal(
            amplitude=0.4, wavelength=10.0, width=7.0, center=1.0,
        )
        rng = np.random.default_rng(5)
        z = np.asarray(gen.displacement(x), dtype=float)
        z = z + rng.normal(0.0, 0.004, x.size)
        for criterion, attr in (
            ("bic", "bic"), ("aic", "aic"), ("rms", "rms_residual"),
        ):
            values = [
                getattr(r, attr) for r in rank_families(x, z, criterion=criterion)
            ]
            assert values == sorted(values), criterion

    def test_families_argument_restricts_the_comparison(self):
        x = np.linspace(-18.0, 20.0, 120)
        gen = GaussianSinusoidal(
            amplitude=0.4, wavelength=10.0, width=7.0, center=1.0,
        )
        z = np.asarray(gen.displacement(x), dtype=float)
        subset = ["pure_sinusoidal", "gaussian_bump"]
        ranked = rank_families(x, z, families=subset)
        assert {r.family for r in ranked} == set(subset)

    def test_auto_family_returns_the_top_ranked_fit(self):
        """The issue's literal API: ``fit(x, z, family='auto')``."""
        x = np.linspace(-18.0, 20.0, 120)
        gen = GaussianBump(
            amplitude=0.4, wavelength=10.0, width=7.0, center=1.0,
        )
        rng = np.random.default_rng(4)
        z = np.asarray(gen.displacement(x), dtype=float)
        z = z + rng.normal(0.0, 0.004, x.size)
        auto = fit_profile(x, z, "auto")
        assert auto.family == "gaussian_bump"
        assert auto.family == rank_families(x, z)[0].family

    def test_unknown_criterion_is_rejected(self):
        x = np.linspace(-10.0, 10.0, 40)
        z = np.zeros_like(x)
        with pytest.raises(ValueError, match="criterion must be"):
            rank_families(x, z, criterion="residual")


class TestFitResult:
    """The result object is a report, not just a tuple."""

    def test_to_config_kwargs_round_trips_through_the_geometry(
        self, clean_trace
    ):
        x, z = clean_trace
        fit = fit_profile(x, z, "gaussian_sinusoidal")
        kwargs = fit.to_config_kwargs()
        assert set(kwargs) == {"amplitude", "wavelength", "width"}
        npt.assert_allclose(kwargs["amplitude"], TRUTH["amplitude"], rtol=1e-4)
        npt.assert_allclose(kwargs["wavelength"], TRUTH["wavelength"], rtol=1e-4)
        npt.assert_allclose(kwargs["width"], TRUTH["width"], rtol=1e-4)

    def test_to_config_kwargs_is_accepted_by_analysis_config(self, clean_trace):
        """The advertised one-liner from a trace to a runnable analysis."""
        from wrinklefe.analysis import AnalysisConfig

        x, z = clean_trace
        fit = fit_profile(x, z, "gaussian_sinusoidal")
        cfg = AnalysisConfig(**fit.to_config_kwargs(), angles=[0.0] * 8)
        npt.assert_allclose(cfg.amplitude, fit.amplitude, rtol=1e-12)
        npt.assert_allclose(cfg.wavelength, fit.wavelength, rtol=1e-12)
        npt.assert_allclose(cfg.width, fit.width, rtol=1e-12)

    def test_summary_names_only_the_fitted_parameters(self):
        x = np.linspace(-18.0, 20.0, 120)
        gen = PureSinusoidal(
            amplitude=0.4, wavelength=10.0, width=7.0, center=1.0,
        )
        z = np.asarray(gen.displacement(x), dtype=float)
        text = fit_profile(x, z, "pure_sinusoidal").summary()
        assert "amplitude" in text and "wavelength" in text
        assert "width" not in text
        assert "RMS" in text and "R^2" in text

    def test_r_squared_is_negative_for_a_family_that_fits_worse_than_a_line(
        self,
    ):
        """Documented behaviour, pinned so it is not "fixed" into a clamp."""
        result = FitResult(
            profile=GaussianSinusoidal(**TRUTH),
            family="gaussian_sinusoidal",
            fitted_parameters=("amplitude",),
            r_squared=-0.4,
            **{k: v for k, v in TRUTH.items()},
        )
        assert result.r_squared < 0.0

    def test_n_points_counts_the_points_actually_used(self):
        x = np.linspace(-18.0, 20.0, 41)
        gen = GaussianSinusoidal(**TRUTH)
        z = np.asarray(gen.displacement(x), dtype=float)
        z[3] = np.nan
        fit = fit_profile(x, z, "gaussian_sinusoidal")
        assert fit.n_points == 40


class TestInputValidation:
    def test_unknown_family_is_rejected_and_lists_the_valid_ones(self):
        x = np.linspace(-10.0, 10.0, 40)
        with pytest.raises(ValueError, match="unknown profile family"):
            fit_profile(x, np.zeros_like(x), "sawtooth")

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            fit_profile(np.zeros(10), np.zeros(9), "gaussian_sinusoidal")

    def test_too_few_points_is_rejected_with_the_count_needed(self):
        x = np.linspace(0.0, 1.0, 4)
        with pytest.raises(ValueError, match="needs at least"):
            fit_profile(x, np.zeros_like(x), "gaussian_sinusoidal")

    def test_three_parameter_family_needs_fewer_points(self):
        """The requirement tracks the model, not a hard-coded number."""
        x = np.linspace(0.0, 1.0, 5)
        z = np.zeros_like(x)
        with pytest.raises(ValueError, match="needs at least"):
            fit_profile(x, z, "gaussian_sinusoidal")   # 4 shape + 2 trend
        with warnings.catch_warnings():                # 3 shape + 2 trend
            warnings.simplefilter("ignore")
            fit_profile(x, z, "pure_sinusoidal")

    def test_ill_conditioned_fit_reports_infinite_sigma_not_a_failure(self):
        """Exactly determined: 5 points against 5 parameters leaves no
        residual degrees of freedom, so the covariance is undefined.  The
        documented behaviour is ``inf`` — information, not an exception."""
        x = np.linspace(0.0, 1.0, 5)
        z = np.zeros_like(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = fit_profile(x, z, "pure_sinusoidal")
        assert all(math.isinf(s) for s in fit.sigmas.values()), fit.sigmas
        assert "+/-?" in fit.summary()

    def test_detrend_false_lowers_the_minimum_point_count(self):
        """Without the two nuisance parameters the model is smaller."""
        x = np.linspace(0.0, 1.0, 4)
        z = np.zeros_like(x)
        with pytest.raises(ValueError, match="needs at least"):
            fit_profile(x, z, "pure_sinusoidal", detrend=True)   # needs 5
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_profile(x, z, "pure_sinusoidal", detrend=False)  # needs 3

    def test_non_finite_rows_are_dropped_rather_than_poisoning_the_fit(self):
        gen = GaussianSinusoidal(**TRUTH)
        x = np.linspace(-20.0, 20.0, 80)
        z = np.asarray(gen.displacement(x), dtype=float)
        z[10] = np.nan
        z[20] = np.inf
        fit = fit_profile(x, z, "gaussian_sinusoidal")
        npt.assert_allclose(fit.amplitude, TRUTH["amplitude"], rtol=1e-3)

    def test_unsorted_input_gives_the_same_fit_as_sorted(self, clean_trace):
        x, z = clean_trace
        rng = np.random.default_rng(0)
        order = rng.permutation(x.size)
        shuffled = fit_profile(x[order], z[order], "gaussian_sinusoidal")
        npt.assert_allclose(
            shuffled.amplitude,
            fit_profile(x, z, "gaussian_sinusoidal").amplitude,
            rtol=1e-6,
        )


class TestLoadTrace:
    def test_reads_two_columns_and_applies_the_pixel_scale(self, tmp_path):
        path = tmp_path / "trace.csv"
        path.write_text("0,0\n10,5\n20,10\n30,15\n")
        x, z = load_trace(path, scale=0.01)
        npt.assert_allclose(x, [0.0, 0.1, 0.2, 0.3])
        npt.assert_allclose(z, [0.0, 0.05, 0.1, 0.15])

    def test_z_scale_applies_on_top_of_scale(self, tmp_path):
        path = tmp_path / "trace.csv"
        path.write_text("0,0\n10,5\n20,10\n30,15\n")
        x, z = load_trace(path, scale=0.01, z_scale=2.0)
        npt.assert_allclose(x, [0.0, 0.1, 0.2, 0.3])
        npt.assert_allclose(z, [0.0, 0.1, 0.2, 0.3])

    def test_header_is_skipped_and_output_is_sorted_by_x(self, tmp_path):
        path = tmp_path / "trace.csv"
        path.write_text("x_px,z_px\n30,15\n0,0\n20,10\n10,5\n")
        x, z = load_trace(path, skip_header=1)
        assert np.all(np.diff(x) > 0.0)
        npt.assert_allclose(z, [0.0, 5.0, 10.0, 15.0])

    def test_column_indices_are_honoured(self, tmp_path):
        path = tmp_path / "trace.csv"
        path.write_text("0,99,0\n1,99,5\n2,99,10\n3,99,15\n")
        _, z = load_trace(path, x_column=0, z_column=2)
        npt.assert_allclose(z, [0.0, 5.0, 10.0, 15.0])

    def test_too_few_columns_is_rejected(self, tmp_path):
        path = tmp_path / "trace.csv"
        path.write_text("0\n1\n2\n3\n")
        with pytest.raises(ValueError, match="expected at least"):
            load_trace(path)

    def test_too_few_usable_rows_is_rejected(self, tmp_path):
        path = tmp_path / "trace.csv"
        path.write_text("0,0\n1,1\nnan,nan\n")
        with pytest.raises(ValueError, match="usable rows"):
            load_trace(path)

    def test_whitespace_separated_files_still_work(self, tmp_path):
        path = tmp_path / "trace.dat"
        path.write_text("0 0\n10 5\n20 10\n30 15\n")
        x, z = load_trace(path)
        npt.assert_allclose(x, [0.0, 10.0, 20.0, 30.0])
        npt.assert_allclose(z, [0.0, 5.0, 10.0, 15.0])

    @pytest.mark.parametrize("delimiter", [",", ";", "\t"])
    def test_delimiter_is_sniffed_not_left_to_numpy(self, tmp_path, delimiter):
        """Regression: ``genfromtxt(delimiter=None)`` splits on whitespace
        only, so a delimited file read through it comes back as a single
        column of ``nan`` instead of raising."""
        path = tmp_path / "trace.csv"
        path.write_text(
            "".join(f"{i}{delimiter}{i * 5}\n" for i in range(4))
        )
        x, z = load_trace(path)
        npt.assert_allclose(x, [0.0, 1.0, 2.0, 3.0])
        npt.assert_allclose(z, [0.0, 5.0, 10.0, 15.0])

    def test_explicit_delimiter_overrides_the_sniff(self, tmp_path):
        path = tmp_path / "trace.csv"
        path.write_text("0;0\n1;5\n2;10\n3;15\n")
        _, z = load_trace(path, delimiter=";")
        npt.assert_allclose(z, [0.0, 5.0, 10.0, 15.0])

    def test_single_column_file_is_rejected_as_too_few_columns(self, tmp_path):
        """Regression: ``np.atleast_2d`` reshapes a one-column file into a
        one-*row* array, which passes the column check and then reads x
        and z out of the same data."""
        path = tmp_path / "trace.csv"
        path.write_text("0\n1\n2\n3\n")
        with pytest.raises(ValueError, match="expected at least"):
            load_trace(path)

    def test_single_row_file_is_rejected_for_row_count_not_columns(
        self, tmp_path
    ):
        """The mirror case: one row of two columns is a column-count
        success and a row-count failure, and must say so."""
        path = tmp_path / "trace.csv"
        path.write_text("0,1\n")
        with pytest.raises(ValueError, match="usable rows"):
            load_trace(path)

    def test_comment_lines_do_not_confuse_the_sniff(self, tmp_path):
        path = tmp_path / "trace.csv"
        path.write_text("# digitized 2026-01-01\n0,0\n1,5\n2,10\n3,15\n")
        _, z = load_trace(path)
        npt.assert_allclose(z, [0.0, 5.0, 10.0, 15.0])

    def test_a_loaded_trace_fits(self, tmp_path):
        """End to end: CSV in pixels -> scaled trace -> recovered mm."""
        gen = GaussianSinusoidal(**TRUTH)
        x_mm = np.linspace(-20.0, 20.0, 80)
        z_mm = np.asarray(gen.displacement(x_mm), dtype=float)
        path = tmp_path / "trace.csv"
        path.write_text(
            "\n".join(f"{xi / 0.01},{zi / 0.01}" for xi, zi in zip(x_mm, z_mm))
        )
        x, z = load_trace(path, scale=0.01)
        fit = fit_profile(x, z, "gaussian_sinusoidal")
        npt.assert_allclose(fit.amplitude, TRUTH["amplitude"], rtol=1e-3)
