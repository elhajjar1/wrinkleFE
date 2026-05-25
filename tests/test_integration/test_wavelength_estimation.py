"""Unit tests for ``estimate_wavelength_from_amplitude``.

The helper supplies wavelengths to validation harnesses for datasets that
only report amplitude. It supports two scaling rules:

* ``"linear"`` (legacy) - ``lambda = K_lambda * A``
* ``"sqrt"``  (default) - ``lambda = K_lambda * sqrt(A * lambda_ref)``

These tests pin the numerics at the calibration anchor, demonstrate
sub-linear growth above the anchor, exercise the ``lambda_min`` clamp,
and confirm that bad ``scaling`` arguments are rejected loudly.
"""

from __future__ import annotations

import math

import pytest

from wrinklefe.analysis import estimate_wavelength_from_amplitude


# ----------------------------------------------------------------------
# Linear scaling
# ----------------------------------------------------------------------

def test_linear_scaling_matches_K_lambda_times_A():
    """Legacy linear rule: lambda = K_lambda * A above the clamp."""
    lam = estimate_wavelength_from_amplitude(
        0.5, K_lambda=19.9, scaling="linear"
    )
    assert lam == pytest.approx(9.95, rel=1e-12)


# ----------------------------------------------------------------------
# Sqrt scaling: anchor matches the legacy linear rule
# ----------------------------------------------------------------------

def test_sqrt_scaling_at_lambda_ref_matches_linear_rule():
    """At A == lambda_ref the sqrt rule equals K_lambda * lambda_ref.

    We pick lambda_ref large enough that the result clears the
    ``lambda_min`` clamp so the calibration-anchor identity is observed
    in the raw value rather than masked by the clamp.
    """
    K_lambda = 19.9
    lambda_ref = 0.6  # K_lambda * lambda_ref = 11.94 > lambda_min = 8.2
    lam = estimate_wavelength_from_amplitude(
        lambda_ref,
        K_lambda=K_lambda,
        lambda_min=8.2,
        lambda_ref=lambda_ref,
        scaling="sqrt",
    )
    assert lam == pytest.approx(K_lambda * lambda_ref, rel=1e-12)


# ----------------------------------------------------------------------
# Sqrt scaling: sub-linear growth above the anchor
# ----------------------------------------------------------------------

def test_sqrt_scaling_sub_linear_growth_above_anchor():
    """For A = 4 * lambda_ref the sqrt rule returns 2 * K_lambda * lambda_ref.

    Linear growth would give 4x the anchor wavelength; sqrt gives 2x,
    which is the half-power scaling that lets ``theta_max`` increase
    with amplitude instead of saturating.
    """
    K_lambda = 19.9
    lambda_ref = 0.6
    A = 4.0 * lambda_ref  # 2.4 mm

    lam = estimate_wavelength_from_amplitude(
        A,
        K_lambda=K_lambda,
        lambda_min=8.2,
        lambda_ref=lambda_ref,
        scaling="sqrt",
    )
    expected = 2.0 * K_lambda * lambda_ref  # sqrt(4) == 2
    assert lam == pytest.approx(expected, rel=1e-12)

    # And it is exactly half the linear result, demonstrating sub-linear growth.
    linear = estimate_wavelength_from_amplitude(
        A, K_lambda=K_lambda, lambda_min=0.0, scaling="linear"
    )
    assert lam == pytest.approx(0.5 * linear, rel=1e-12)

    # theta_max strictly increases relative to the anchor case, confirming
    # the plateau in the linear-rule prediction is gone.
    theta_anchor = math.atan2(2.0 * math.pi * lambda_ref, K_lambda * lambda_ref)
    theta_high = math.atan2(2.0 * math.pi * A, lam)
    assert theta_high > theta_anchor


# ----------------------------------------------------------------------
# lambda_min clamp
# ----------------------------------------------------------------------

def test_lambda_min_clamps_tiny_amplitudes():
    """Vanishingly small amplitudes return ``lambda_min``, not ~0."""
    tiny = 1e-6
    lam_sqrt = estimate_wavelength_from_amplitude(
        tiny, K_lambda=19.9, lambda_min=8.2, lambda_ref=0.5, scaling="sqrt"
    )
    assert lam_sqrt == pytest.approx(8.2, rel=1e-12)

    lam_linear = estimate_wavelength_from_amplitude(
        tiny, K_lambda=19.9, lambda_min=8.2, scaling="linear"
    )
    assert lam_linear == pytest.approx(8.2, rel=1e-12)


# ----------------------------------------------------------------------
# Argument validation
# ----------------------------------------------------------------------

def test_invalid_scaling_raises_value_error():
    """Unknown scaling rule names must fail loudly with ValueError."""
    with pytest.raises(ValueError, match="scaling"):
        estimate_wavelength_from_amplitude(0.5, scaling="cubic")
