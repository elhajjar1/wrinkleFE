"""Tests for wrinklefe.core.wrinkle module."""

import numpy as np
import numpy.testing as npt
import pytest
from scipy.optimize import minimize_scalar

from wrinklefe.core.wrinkle import (
    GaussianBump,
    GaussianSinusoidal,
    PureSinusoidal,
    RectangularSinusoidal,
    WrinkleSurface3D,
)


class TestGaussianSinusoidal:
    """Test the Jin et al. Gaussian-enveloped sinusoidal wrinkle."""

    def test_displacement_at_center_is_amplitude(self, gaussian_wrinkle):
        """At x = center, cos(0) = 1 and exp(0) = 1, so z = A."""
        z = gaussian_wrinkle.displacement(np.array([0.0]))
        npt.assert_allclose(z, [0.366], atol=1e-12)

    def test_displacement_far_from_center_decays(self, gaussian_wrinkle):
        """At x >> width, the Gaussian envelope should drive z to ~0."""
        x_far = np.array([100.0])
        z = gaussian_wrinkle.displacement(x_far)
        npt.assert_allclose(z, [0.0], atol=1e-10)

    def test_slope_at_center_is_zero(self, gaussian_wrinkle):
        """At x = center: derivative of cos at 0 is 0, derivative of Gaussian at 0 is 0.

        So total slope = A * (0 * 1 - k * 0) = 0.
        """
        s = gaussian_wrinkle.slope(np.array([0.0]))
        npt.assert_allclose(s, [0.0], atol=1e-12)

    def test_max_angle_approx(self, gaussian_wrinkle):
        """max_angle_approx = arctan(2*pi*A/lambda)."""
        expected = np.arctan(2.0 * np.pi * 0.366 / 16.0)
        actual = gaussian_wrinkle.max_angle_approx()
        npt.assert_allclose(actual, expected, rtol=1e-12)

    def test_max_angle_close_to_approx(self, gaussian_wrinkle):
        """Numerical max_angle should be reasonably close to the analytical approximation.

        For a Gaussian-enveloped sinusoid, the Gaussian modulation can reduce
        the effective peak slope compared to a pure sinusoid (the max slope
        occurs away from the center where the envelope is < 1). The two values
        should still be within a factor of ~2 of each other.
        """
        numerical = gaussian_wrinkle.max_angle()
        approx = gaussian_wrinkle.max_angle_approx()
        ratio = numerical / approx
        assert 0.3 < ratio < 2.0, (
            f"ratio={ratio:.3f}, numerical={numerical:.6f}, "
            f"approx={approx:.6f}"
        )

    def test_fiber_angle_shape(self, gaussian_wrinkle):
        x = np.linspace(-20, 20, 50)
        angles = gaussian_wrinkle.fiber_angle(x)
        assert angles.shape == x.shape

    def test_fiber_angle_at_center_is_zero(self, gaussian_wrinkle):
        """Fiber angle at center should be 0 since slope is 0."""
        angle = gaussian_wrinkle.fiber_angle(np.array([0.0]))
        npt.assert_allclose(angle, [0.0], atol=1e-12)

    def test_curvature_at_center(self, gaussian_wrinkle):
        """Curvature should be well-defined at the center."""
        kappa = gaussian_wrinkle.curvature(np.array([0.0]))
        assert np.isfinite(kappa[0])

    def test_validation_negative_amplitude(self):
        with pytest.raises(ValueError, match="amplitude"):
            GaussianSinusoidal(amplitude=-0.1, wavelength=16.0, width=12.0)

    def test_validation_zero_wavelength(self):
        with pytest.raises(ValueError, match="wavelength"):
            GaussianSinusoidal(amplitude=0.1, wavelength=0.0, width=12.0)

    def test_validation_negative_width(self):
        with pytest.raises(ValueError, match="width"):
            GaussianSinusoidal(amplitude=0.1, wavelength=16.0, width=-1.0)

    def test_domain(self, gaussian_wrinkle):
        lo, hi = gaussian_wrinkle.domain()
        assert lo < 0 < hi
        npt.assert_allclose(lo, -3.0 * 12.0, atol=1e-12)
        npt.assert_allclose(hi, 3.0 * 12.0, atol=1e-12)


class TestPureSinusoidal:
    """Test the pure sinusoidal wrinkle (no envelope)."""

    @pytest.fixture
    def pure_wrinkle(self):
        return PureSinusoidal(amplitude=0.366, wavelength=16.0, width=12.0)

    def test_displacement_at_center(self, pure_wrinkle):
        z = pure_wrinkle.displacement(np.array([0.0]))
        npt.assert_allclose(z, [0.366], atol=1e-12)

    def test_max_angle_exact(self, pure_wrinkle):
        """For pure sinusoid, max_angle = arctan(2*pi*A/lambda) exactly."""
        expected = np.arctan(2.0 * np.pi * 0.366 / 16.0)
        actual = pure_wrinkle.max_angle()
        npt.assert_allclose(actual, expected, rtol=1e-10)

    def test_max_angle_equals_approx(self, pure_wrinkle):
        """For pure sinusoid, max_angle and max_angle_approx should be identical."""
        npt.assert_allclose(
            pure_wrinkle.max_angle(),
            pure_wrinkle.max_angle_approx(),
            rtol=1e-10,
        )

    def test_no_decay(self, pure_wrinkle):
        """Pure sinusoid should not decay -- displacement at integer wavelengths = A."""
        # At x = lambda (one full wavelength from center)
        x = np.array([16.0])
        z = pure_wrinkle.displacement(x)
        npt.assert_allclose(z, [0.366], atol=1e-12)


class TestMaxAngleGlobalSearch:
    """Regression tests for issue #16: ``max_angle`` must find the GLOBAL peak.

    ``|dz/dx|`` of an enveloped multi-period wrinkle is multimodal (one peak
    per quarter-period).  The original implementation used a single bounded
    Brent search (``scipy.optimize.minimize_scalar(method="bounded")``), which
    assumes unimodality and routinely converges to a *local* maximum, thereby
    under-reporting the true peak fibre misalignment angle.  The fix performs
    a dense uniform grid sweep over the domain followed by a bracketed local
    polish, so these tests assert the GLOBAL maximum is returned.
    """

    @staticmethod
    def _reference_max_angle(profile, n=2_000_000):
        """Brute-force reference: max |slope| on a very dense grid."""
        xlo, xhi = profile.domain()
        xs = np.linspace(xlo, xhi, n)
        return float(np.arctan(np.max(np.abs(profile.slope(xs)))))

    @staticmethod
    def _old_bounded_brent_angle(profile):
        """Reproduce the buggy pre-fix algorithm: bounded Brent over the
        whole domain. Used to prove the new code beats the old behaviour."""
        xlo, xhi = profile.domain()

        def neg_abs_slope(xv):
            return -float(np.abs(profile.slope(np.atleast_1d(xv))[0]))

        res = minimize_scalar(
            neg_abs_slope, bounds=(xlo, xhi), method="bounded"
        )
        return float(np.arctan(-res.fun))

    def test_multimodal_gaussian_returns_global_not_local(self):
        """Multi-period Gaussian wrinkle: the global |slope| peak sits near
        the center, but bounded Brent latches a smaller off-center local peak.

        With A=0.5, lambda=1.5, w=8.0 the domain spans 32 wavelengths so
        |slope| has ~64 comparable peaks.  The analytical global maximum is
        arctan(2*pi*A/lambda) attenuated slightly by the Gaussian envelope
        (~64.4 deg).  Naive full-domain bounded Brent converges to a local
        peak near x~3.4 and reports only ~60.3 deg -- a >4 deg under-report
        that this test would have caught on the old code.
        """
        prof = GaussianSinusoidal(amplitude=0.5, wavelength=1.5, width=8.0)

        reference = self._reference_max_angle(prof)
        new_value = prof.max_angle()
        old_buggy = self._old_bounded_brent_angle(prof)

        # New implementation recovers the true global maximum.
        npt.assert_allclose(new_value, reference, rtol=1e-3)

        # And it is meaningfully better than the old bounded-Brent result:
        # the old algorithm under-reports the peak angle by several degrees.
        assert old_buggy < reference - np.radians(2.0), (
            f"expected old bounded-Brent to under-report; "
            f"old={np.degrees(old_buggy):.3f} deg, "
            f"true={np.degrees(reference):.3f} deg"
        )
        assert new_value > old_buggy + np.radians(2.0), (
            f"new max_angle should beat old bounded Brent by >2 deg; "
            f"new={np.degrees(new_value):.3f} deg, "
            f"old={np.degrees(old_buggy):.3f} deg"
        )

    def test_pure_sinusoid_analytic_max_slope(self):
        """For a pure cosine, max |dz/dx| = 2*pi*A/lambda at every
        quarter-wave point, so max_angle == arctan(2*pi*A/lambda) exactly.

        This pins the documented analytic value and the radian convention.
        """
        A, lam = 0.42, 9.0
        prof = PureSinusoidal(amplitude=A, wavelength=lam, width=7.0)
        expected = np.arctan(2.0 * np.pi * A / lam)
        npt.assert_allclose(prof.max_angle(), expected, rtol=1e-10)

    def test_multimodal_rectangular_window_global(self):
        """Flat-top (tanh) window over many periods: every interior peak has
        equal height, and the steepest slope is a specific one.  The grid
        sweep must still return the true global value (envelope <= 1, so the
        analytic ceiling is arctan(2*pi*A/lambda))."""
        A, lam, w = 0.4, 3.0, 30.0
        prof = RectangularSinusoidal(amplitude=A, wavelength=lam, width=w)

        reference = self._reference_max_angle(prof, n=4_000_000)
        new_value = prof.max_angle()
        ceiling = np.arctan(2.0 * np.pi * A / lam)

        npt.assert_allclose(new_value, reference, rtol=2e-3)
        assert new_value <= ceiling + 1e-9

    def test_max_angle_signature_unchanged(self):
        """Public contract: max_angle() takes no args and returns a float
        in radians (consistent with fiber_angle / max_angle_approx)."""
        prof = GaussianSinusoidal(amplitude=0.3, wavelength=2.0, width=20.0)
        value = prof.max_angle()
        assert isinstance(value, float)
        # Radians, not degrees: a wrinkle this steep is ~0.75 rad (~43 deg),
        # which would be a nonsensical >40 if it were degrees-as-radians.
        assert 0.0 < value < np.pi / 2.0
        npt.assert_allclose(
            value, prof.max_angle_approx(), rtol=5e-3
        )


class TestGaussianBump:
    """Test the Gaussian bump (no oscillation)."""

    @pytest.fixture
    def bump(self):
        return GaussianBump(amplitude=0.5, wavelength=16.0, width=10.0)

    def test_displacement_at_center(self, bump):
        z = bump.displacement(np.array([0.0]))
        npt.assert_allclose(z, [0.5], atol=1e-12)

    def test_slope_at_center_is_zero(self, bump):
        s = bump.slope(np.array([0.0]))
        npt.assert_allclose(s, [0.0], atol=1e-12)

    def test_displacement_decays(self, bump):
        z = bump.displacement(np.array([100.0]))
        npt.assert_allclose(z, [0.0], atol=1e-10)

    def test_max_angle_analytical(self, bump):
        """max|dz/dx| = 2A / (w * sqrt(2e)), so max_angle = arctan(that)."""
        A, w = 0.5, 10.0
        max_slope = 2.0 * A / (w * np.sqrt(2.0 * np.e))
        expected = np.arctan(max_slope)
        npt.assert_allclose(bump.max_angle(), expected, rtol=1e-10)


class TestWrinkleSurface3D:
    """Test 3D wrinkle surface extension."""

    @pytest.fixture
    def profile(self):
        return GaussianSinusoidal(amplitude=0.366, wavelength=16.0, width=12.0)

    def test_uniform_mode_matches_profile(self, profile):
        """In 'uniform' mode, displacement(x, y) should equal profile(x) for all y."""
        surface = WrinkleSurface3D(profile, transverse_mode="uniform")
        x = np.array([0.0, 2.0, 4.0])
        y = np.array([5.0, 10.0, 15.0])
        z_surface = surface.displacement(x, y)
        z_profile = profile.displacement(x)
        npt.assert_allclose(z_surface, z_profile, atol=1e-14)

    def test_gaussian_decay_centerline_matches(self, profile):
        """In 'gaussian_decay' mode, displacement at y = span_y/2 matches profile."""
        span_y = 20.0
        surface = WrinkleSurface3D(
            profile,
            transverse_mode="gaussian_decay",
            width_y=10.0,
            span_y=span_y,
        )
        x = np.array([0.0, 2.0])
        y_center = np.array([span_y / 2, span_y / 2])
        z_surface = surface.displacement(x, y_center)
        z_profile = profile.displacement(x)
        npt.assert_allclose(z_surface, z_profile, atol=1e-12)

    def test_gaussian_decay_edges_smaller(self, profile):
        """Displacement at y-edges should be smaller than at centerline."""
        span_y = 20.0
        surface = WrinkleSurface3D(
            profile,
            transverse_mode="gaussian_decay",
            width_y=5.0,
            span_y=span_y,
        )
        x = np.array([0.0])
        z_center = surface.displacement(x, np.array([span_y / 2]))
        z_edge = surface.displacement(x, np.array([0.0]))
        assert abs(z_center[0]) > abs(z_edge[0])

    def test_invalid_mode_raises(self, profile):
        with pytest.raises(ValueError, match="transverse_mode"):
            WrinkleSurface3D(profile, transverse_mode="invalid_mode")

    def test_negative_width_y_raises(self, profile):
        with pytest.raises(ValueError, match="width_y"):
            WrinkleSurface3D(profile, width_y=-1.0)

    def test_negative_span_y_raises(self, profile):
        with pytest.raises(ValueError, match="span_y"):
            WrinkleSurface3D(profile, span_y=-1.0)

    def test_gradient_returns_two_arrays(self, profile):
        surface = WrinkleSurface3D(profile, transverse_mode="uniform")
        x = np.array([0.0, 1.0])
        y = np.array([5.0, 10.0])
        dz_dx, dz_dy = surface.gradient(x, y)
        assert dz_dx.shape == x.shape
        assert dz_dy.shape == x.shape

    def test_uniform_dz_dy_is_zero(self, profile):
        """In uniform mode, dz/dy should be zero everywhere."""
        surface = WrinkleSurface3D(profile, transverse_mode="uniform")
        x = np.array([0.0, 2.0, 4.0])
        y = np.array([5.0, 10.0, 15.0])
        _, dz_dy = surface.gradient(x, y)
        npt.assert_allclose(dz_dy, 0.0, atol=1e-14)

    def test_fiber_angle_shape(self, profile):
        surface = WrinkleSurface3D(profile, transverse_mode="uniform")
        x = np.linspace(-20, 20, 10)
        y = np.full_like(x, 10.0)
        angles = surface.fiber_angle(x, y)
        assert angles.shape == x.shape
