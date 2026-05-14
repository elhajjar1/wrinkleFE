"""Regression tests for ``RectangularSinusoidal`` envelope derivatives.

Specifically guards against issue #15: the analytical second derivative
``RectangularSinusoidal._d2_envelope`` was once sign-flipped relative to
the true second derivative of ``env(x) = 0.5 * (tanh(arg) + 1)`` with
``arg = (w/2 - |x - x0|) / tw``. The expected closed form is

    d^2 env / dx^2 = - sech^2(arg) * tanh(arg) / tw^2  (for dx != 0).
"""

import numpy as np
import numpy.testing as npt
import pytest

from wrinklefe.core.wrinkle import RectangularSinusoidal


@pytest.fixture
def rect_profile():
    """Profile with width=20 (half-width 10) so taper_width tw = w/20 = 1.0."""
    return RectangularSinusoidal(
        amplitude=1.0, wavelength=10.0, width=20.0, center=0.0
    )


def _numerical_d2(profile, x, h=1e-4):
    """Central-difference second derivative of the envelope at scalar ``x``."""
    arr = np.array([x + h, x, x - h])
    e_plus, e_zero, e_minus = profile._envelope(arr)
    return (e_plus - 2.0 * e_zero + e_minus) / (h ** 2)


class TestD2EnvelopeSign:
    """Issue #15 regression: confirm sign and magnitude of ``_d2_envelope``."""

    def test_matches_numerical_inside_taper(self, rect_profile):
        """At x = 9.5 (just inside w/2 = 10), arg = 0.5 > 0 so tanh(arg) > 0.

        The correct second derivative is therefore strictly negative:
        the envelope is concave-down as it approaches the plateau from
        the taper (it is bending back toward 1).
        """
        x = 9.5
        analytical = rect_profile._d2_envelope(np.array([x]))[0]
        numerical = _numerical_d2(rect_profile, x)
        npt.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-8)
        assert analytical < 0.0, (
            "Inside the taper region (|dx| < w/2, arg > 0) the envelope "
            "is concave-down: d^2 env / dx^2 must be negative."
        )

    def test_matches_numerical_outside_taper(self, rect_profile):
        """At x = 10.5 (just outside w/2 = 10), arg = -0.5 < 0 so tanh(arg) < 0.

        The correct second derivative is therefore strictly positive:
        the envelope is concave-up as it decays toward zero.
        """
        x = 10.5
        analytical = rect_profile._d2_envelope(np.array([x]))[0]
        numerical = _numerical_d2(rect_profile, x)
        npt.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-8)
        assert analytical > 0.0, (
            "Outside the taper region (|dx| > w/2, arg < 0) the envelope "
            "is concave-up: d^2 env / dx^2 must be positive."
        )

    def test_symmetric_about_center(self, rect_profile):
        """The envelope is symmetric in |x - x0|, so _d2_envelope must be even."""
        xs = np.array([-11.0, -9.5, -8.0, 8.0, 9.5, 11.0])
        d2 = rect_profile._d2_envelope(xs)
        # Pair each x with its mirror image about the center (0.0).
        npt.assert_allclose(d2[:3][::-1], d2[3:], rtol=1e-12, atol=1e-14)

    def test_closed_form(self, rect_profile):
        """Spot-check against ``-sech^2(arg) * tanh(arg) / tw^2`` directly."""
        x = np.array([9.0, 9.5, 10.5, 11.0])
        tw = rect_profile.width / 20.0
        arg = (rect_profile.width / 2.0 - np.abs(x - rect_profile.center)) / tw
        expected = -1.0 / np.cosh(arg) ** 2 * np.tanh(arg) / tw ** 2
        npt.assert_allclose(rect_profile._d2_envelope(x), expected, rtol=1e-12)
