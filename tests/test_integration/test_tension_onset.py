"""Tests for the delamination-onset tension knockdown prediction.

The three-mechanism tension model reports an *ultimate* fibre-failure
knockdown via :attr:`AnalysisResults.analytical_knockdown`.  Embedded
wrinkles also exhibit an *onset* (first-load-drop) event corresponding
to delamination initiation at the curved 0-block interface, reported
via :attr:`AnalysisResults.analytical_onset_knockdown`.  The onset
event is predicted from a Benzeggagh-Kenane mode-mixity criterion on
the interlaminar stresses already computed by the OOP block, using
``material.GIc`` and ``material.GIIc``.

References
----------
- Mukhopadhyay, S., Hallett, S. R., & Wisnom, M. R. (2015).
  Compressive failure of multidirectional carbon-fibre composites with
  off-axis fibre waviness.  *Composites Part A* 73, 132-142.
"""

from __future__ import annotations

import dataclasses

import pytest

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary

pytestmark = pytest.mark.integration

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _mukhopadhyay_angles() -> list[float]:
    """Mukhopadhyay (2015) layup ``[+45_2/90_2/-45_2/0_2]_3s`` (48 plies)."""
    half = [45, 45, 90, 90, -45, -45, 0, 0] * 3
    return half + list(reversed(half))


@pytest.fixture
def material():
    """IM7/8552 from the built-in library (provides GIc and GIIc)."""
    return MaterialLibrary().get("IM7_8552")


def _embedded_wrinkle_config(material, **overrides) -> AnalysisConfig:
    """Embedded-wrinkle tension config for the Mukhopadhyay layup.

    Default amplitude reproduces the **M-To1** specimen (A = 0.372 mm,
    D/T = 0.062 in a 6 mm laminate).
    """
    defaults = dict(
        amplitude=0.372,
        wavelength=16.0,
        width=12.0,
        morphology="stack",
        loading="tension",
        material=material,
        angles=_mukhopadhyay_angles(),
        ply_thickness=0.125,  # 48 plies x 0.125 mm = 6 mm total
        nx=4,
        ny=2,
        nz_per_ply=1,
        domain_width=10.0,
        applied_strain=0.005,
        analytical_only=True,
        verbose=False,
    )
    defaults.update(overrides)
    return AnalysisConfig(**defaults)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


class TestOnsetMukhopadhyay:
    """The onset KD is populated and consistent with the ultimate KD."""

    def test_onset_in_valid_range(self, material):
        """Onset KD is in (0.4, 1.0) for the M-To1 embedded wrinkle."""
        cfg = _embedded_wrinkle_config(material)
        result = WrinkleAnalysis(cfg).run()

        assert result.analytical_onset_knockdown is not None
        assert 0.4 < result.analytical_onset_knockdown < 1.0

    def test_onset_strictly_below_ultimate(self, material):
        """Onset KD is strictly less than the ultimate KD."""
        cfg = _embedded_wrinkle_config(material)
        result = WrinkleAnalysis(cfg).run()

        assert result.analytical_onset_knockdown is not None
        assert (
            result.analytical_onset_knockdown
            < result.analytical_knockdown
        )

    def test_onset_monotonic_in_amplitude(self, material):
        """Onset KD decreases monotonically as the wrinkle grows.

        Covers the three Mukhopadhyay validation amplitudes
        (M-To1, M-To2, M-To3 with A = 0.372, 0.492, 0.570 mm).
        """
        onsets = []
        for amp in [0.372, 0.492, 0.570]:
            cfg = _embedded_wrinkle_config(material, amplitude=amp)
            result = WrinkleAnalysis(cfg).run()
            assert result.analytical_onset_knockdown is not None
            onsets.append(result.analytical_onset_knockdown)

        assert onsets[0] > onsets[1] > onsets[2], (
            f"Onset KD must decrease with amplitude, got {onsets}"
        )

    def test_onset_in_mechanisms_dict(self, material):
        """The mechanisms dict carries a ``kd_onset`` entry."""
        cfg = _embedded_wrinkle_config(material)
        result = WrinkleAnalysis(cfg).run()

        assert result.tension_mechanisms is not None
        assert "kd_onset" in result.tension_mechanisms
        assert result.tension_mechanisms["kd_onset"] == pytest.approx(
            result.analytical_onset_knockdown
        )


class TestOnsetGuards:
    """Onset is ``None`` when a prerequisite is missing."""

    def test_onset_none_when_giclc_missing(self, material):
        """Onset is ``None`` when ``material.GIc`` is None."""
        stripped = dataclasses.replace(material, GIc=None, GIIc=None)
        cfg = _embedded_wrinkle_config(stripped)
        result = WrinkleAnalysis(cfg).run()

        assert result.analytical_onset_knockdown is None

    def test_onset_none_when_only_giicl_missing(self, material):
        """Onset is ``None`` when only ``GIIc`` is missing."""
        stripped = dataclasses.replace(material, GIIc=None)
        cfg = _embedded_wrinkle_config(stripped)
        result = WrinkleAnalysis(cfg).run()

        assert result.analytical_onset_knockdown is None

    def test_onset_none_for_compression(self, material):
        """Onset is ``None`` for compression loading."""
        cfg = _embedded_wrinkle_config(
            material, loading="compression", applied_strain=-0.005
        )
        result = WrinkleAnalysis(cfg).run()

        assert result.analytical_onset_knockdown is None
