"""Pin the canonical definition of the wrinkle ``amplitude`` parameter (issue #181).

The canonical contract documented in ``WrinkleProfile``,
``AnalysisConfig.amplitude``, the CLI ``--amplitude`` help, the
Streamlit ``Amplitude A`` slider, the README parameter table, and the
``WrinkleConfiguration`` class docstring is:

    ``amplitude`` is the half-amplitude *A* (mm): the peak displacement
    of the wrinkled mid-surface from the flat reference, so
    ``z(x) = A * cos(2*pi*(x - x0) / lambda)`` (modulated by the
    envelope) and the peak-to-trough height is ``2A``.

These tests fix that convention in the math itself: if the formula ever
switches to the full peak-to-trough convention
(``z = 0.5 * amplitude * cos(...)``) the assertions below will fail and
force the documentation to be updated in lockstep.
"""

import numpy as np
import pytest

from wrinklefe.core.wrinkle import (
    GaussianBump,
    GaussianSinusoidal,
    PureSinusoidal,
    RectangularSinusoidal,
    TriangularSinusoidal,
)

# Canonical test point: A = 1.0 mm, lambda = 10.0 mm.
# Under the half-amplitude convention the peak |z| of the displaced
# surface equals exactly 1.0 (the full peak-to-trough height is 2.0).
_AMPLITUDE = 1.0
_WAVELENGTH = 10.0
_WIDTH = 50.0  # >> wavelength so the envelope is essentially flat near x0


def test_pure_sinusoidal_peak_equals_half_amplitude() -> None:
    """``PureSinusoidal`` peak |z| equals the half-amplitude A exactly."""
    profile = PureSinusoidal(
        amplitude=_AMPLITUDE, wavelength=_WAVELENGTH, width=_WIDTH
    )
    # At the centre the cosine is at its maximum, so z(0) = A.
    assert profile.displacement(np.array([0.0]))[0] == pytest.approx(_AMPLITUDE)
    # Half a wavelength away the cosine is at its minimum, so z = -A.
    assert profile.displacement(np.array([_WAVELENGTH / 2.0]))[0] == pytest.approx(
        -_AMPLITUDE
    )
    # The peak-to-trough height is 2A.
    xs = np.linspace(-_WAVELENGTH, _WAVELENGTH, 4001)
    zs = profile.displacement(xs)
    assert (zs.max() - zs.min()) == pytest.approx(2.0 * _AMPLITUDE)


def test_pure_sinusoidal_with_amplitude_one_peak_is_one() -> None:
    """Behavioural pin for issue #181: peak |z| == 1.0 for A=1.0, lambda=10.0."""
    profile = PureSinusoidal(amplitude=1.0, wavelength=10.0, width=50.0)
    xs = np.linspace(-20.0, 20.0, 8001)
    peak = float(np.max(np.abs(profile.displacement(xs))))
    # If the formula were z = 0.5 * A * cos(...), this would be 0.5.
    assert peak == pytest.approx(1.0)


def test_gaussian_sinusoidal_peak_at_centre_equals_half_amplitude() -> None:
    """``GaussianSinusoidal`` at the crest equals A (envelope = 1 at centre)."""
    profile = GaussianSinusoidal(
        amplitude=_AMPLITUDE, wavelength=_WAVELENGTH, width=_WIDTH
    )
    # At x = center the envelope and cosine are both 1, so z = A exactly.
    assert profile.displacement(np.array([0.0]))[0] == pytest.approx(_AMPLITUDE)


def test_gaussian_bump_peak_equals_half_amplitude() -> None:
    """``GaussianBump`` peak (at x = centre) equals A."""
    profile = GaussianBump(
        amplitude=_AMPLITUDE, wavelength=_WAVELENGTH, width=_WIDTH
    )
    assert profile.displacement(np.array([0.0]))[0] == pytest.approx(_AMPLITUDE)


@pytest.mark.parametrize(
    "cls",
    [PureSinusoidal, GaussianSinusoidal, RectangularSinusoidal, TriangularSinusoidal],
)
def test_amplitude_scales_displacement_linearly(cls) -> None:
    """Doubling A doubles the peak displacement (linear-in-A contract)."""
    profile_a = cls(amplitude=1.0, wavelength=_WAVELENGTH, width=_WIDTH)
    profile_b = cls(amplitude=2.0, wavelength=_WAVELENGTH, width=_WIDTH)
    # At the centre x = 0 (= profile.center), both the envelope and the
    # cosine are at their maxima, so z(0) = A * 1 * 1 = A.
    z_a = profile_a.displacement(np.array([0.0]))[0]
    z_b = profile_b.displacement(np.array([0.0]))[0]
    assert z_a == pytest.approx(1.0)
    assert z_b == pytest.approx(2.0)
    assert z_b == pytest.approx(2.0 * z_a)
