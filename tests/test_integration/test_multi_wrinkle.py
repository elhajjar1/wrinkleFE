"""Integration tests for the multi-wrinkle AnalysisConfig override.

Tests the ``wrinkles`` field on :class:`AnalysisConfig` that allows
arbitrary N-wrinkle layouts (one :class:`WrinkleSpec` per placement) to
be analysed. This is the path used to model Dataset F (Li et al. 2025)
multi-wrinkle specimens (D-AB-2, D-A-2, D-M-2, T-M-2).
"""

from __future__ import annotations

import math

import pytest

from wrinklefe.analysis import (
    AnalysisConfig,
    AnalysisResults,
    WrinkleAnalysis,
    WrinkleSpec,
)
from wrinklefe.core.material import MaterialLibrary

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def ac318_material():
    """AC318 / S6C10-800 (S-glass / epoxy) from Li et al. (2025)."""
    return MaterialLibrary().get("AC318_S6C10")


@pytest.fixture
def ud14_angles():
    """UD [0]_14 stacking sequence."""
    return [0.0] * 14


def _build_cfg(material, angles, wrinkles, **overrides):
    """Helper to build a multi-wrinkle AnalysisConfig with sane defaults."""
    base = dict(
        amplitude=1.5,
        wavelength=12.9,
        width=12.9,
        morphology="stack",
        loading="compression",
        material=material,
        angles=list(angles),
        wrinkles=wrinkles,
        analytical_only=True,
    )
    base.update(overrides)
    return AnalysisConfig(**base)


# ---------------------------------------------------------------------
# Multi-wrinkle path: analytical KD runs and is in (0, 1)
# ---------------------------------------------------------------------


class TestMultiWrinkleAnalytical:
    """Multi-wrinkle analytical pathway."""

    def test_two_wrinkle_ud_runs(self, ac318_material, ud14_angles):
        """Two-wrinkle UD AC318 config returns a finite KD in (0, 1)."""
        wrinkles = [
            WrinkleSpec(amplitude=1.5, wavelength=12.9, width=12.9, ply_interface=4),
            WrinkleSpec(amplitude=1.5, wavelength=12.9, width=12.9, ply_interface=10),
        ]
        cfg = _build_cfg(ac318_material, ud14_angles, wrinkles)
        result = WrinkleAnalysis(cfg).run(analytical_only=True)

        assert isinstance(result, AnalysisResults)
        kd = result.analytical_knockdown
        assert math.isfinite(kd), f"Expected finite KD, got {kd}"
        assert 0.0 < kd < 1.0, f"Expected KD in (0, 1), got {kd}"
        # Multi-wrinkle config was stored, wrinkle_config has 2 wrinkles
        assert result.wrinkle_config is not None
        assert result.wrinkle_config.n_wrinkles() == 2

    def test_multi_wrinkle_monotonic(self, ac318_material, ud14_angles):
        """Adding a wrinkle should not increase predicted strength.

        Compare the two-wrinkle KD against the single-wrinkle equivalent
        (only one of the two specs). The peak angle theta_max is the
        same in both cases, but the second wrinkle introduces an extra
        pairwise interaction (M_f_agg may shift) -- in the "stack"
        (phi=0) baseline, M_f_agg stays at 1.0, so the KDs should match
        to within tight tolerance; in general the multi-wrinkle KD
        should be <= the single-wrinkle KD (monotonic).
        """
        single = [
            WrinkleSpec(amplitude=1.5, wavelength=12.9, width=12.9, ply_interface=4),
        ]
        multi = [
            WrinkleSpec(amplitude=1.5, wavelength=12.9, width=12.9, ply_interface=4),
            WrinkleSpec(amplitude=1.5, wavelength=12.9, width=12.9, ply_interface=10),
        ]
        kd_single = WrinkleAnalysis(
            _build_cfg(ac318_material, ud14_angles, single)
        ).run(analytical_only=True).analytical_knockdown
        kd_multi = WrinkleAnalysis(
            _build_cfg(ac318_material, ud14_angles, multi)
        ).run(analytical_only=True).analytical_knockdown

        # Multi-wrinkle never STRONGER than single (monotonic).
        assert kd_multi <= kd_single + 1e-9, (
            f"Adding a wrinkle increased predicted strength: "
            f"kd_single={kd_single}, kd_multi={kd_multi}"
        )


# ---------------------------------------------------------------------
# Validation & error paths
# ---------------------------------------------------------------------


class TestMultiWrinkleValidation:
    """Validation of the wrinkles= field and the FE-solve guard."""

    def test_empty_wrinkles_raises(self, ac318_material, ud14_angles):
        """An empty wrinkles list is ambiguous and rejected."""
        with pytest.raises(ValueError, match="at least one"):
            _build_cfg(ac318_material, ud14_angles, [])

    def test_negative_amplitude_raises(self, ac318_material, ud14_angles):
        """Specs with negative amplitude are rejected."""
        bad = [
            WrinkleSpec(amplitude=-0.5, wavelength=12.9, width=12.9, ply_interface=4),
        ]
        with pytest.raises(ValueError, match="amplitude"):
            _build_cfg(ac318_material, ud14_angles, bad)

    def test_zero_amplitude_raises(self, ac318_material, ud14_angles):
        """Zero amplitude is not a valid WrinkleSpec (the multi-wrinkle
        path requires strictly positive geometry per spec)."""
        bad = [
            WrinkleSpec(amplitude=0.0, wavelength=12.9, width=12.9, ply_interface=4),
        ]
        with pytest.raises(ValueError, match="amplitude"):
            _build_cfg(ac318_material, ud14_angles, bad)

    def test_negative_wavelength_raises(self, ac318_material, ud14_angles):
        bad = [
            WrinkleSpec(amplitude=1.0, wavelength=-12.9, width=12.9, ply_interface=4),
        ]
        with pytest.raises(ValueError, match="wavelength"):
            _build_cfg(ac318_material, ud14_angles, bad)

    def test_negative_width_raises(self, ac318_material, ud14_angles):
        bad = [
            WrinkleSpec(amplitude=1.0, wavelength=12.9, width=-1.0, ply_interface=4),
        ]
        with pytest.raises(ValueError, match="width"):
            _build_cfg(ac318_material, ud14_angles, bad)

    def test_ply_interface_out_of_range_raises(self, ac318_material, ud14_angles):
        # n_plies=14 → valid interfaces are 0..12 (n_plies - 2)
        bad = [
            WrinkleSpec(amplitude=1.0, wavelength=12.9, width=12.9, ply_interface=13),
        ]
        with pytest.raises(ValueError, match="ply_interface"):
            _build_cfg(ac318_material, ud14_angles, bad)

    # Multi-wrinkle + CZM is supported since issue #283 (cohesive
    # layers along the full length of every wrinkle-nominated
    # interface); the end-to-end coverage lives in
    # tests/test_integration/test_multi_wrinkle_czm.py.


# ---------------------------------------------------------------------
# Backwards-compatible default (wrinkles=None)
# ---------------------------------------------------------------------


class TestDefaultBehaviourUnchanged:
    """``wrinkles=None`` must reproduce the pre-change numerical output."""

    def test_default_single_wrinkle_kd_unchanged(self):
        """Stock AnalysisConfig (wrinkles=None) returns the legacy KD.

        This is a regression guard: when the multi-wrinkle override is
        not active, the analytical KD must exactly match the value
        produced by the pre-change code path. The expected value below
        was captured from the same default AnalysisConfig() prior to
        the multi-wrinkle change.
        """
        cfg = AnalysisConfig(analytical_only=True)
        result = WrinkleAnalysis(cfg).run(analytical_only=True)
        # Default config: amplitude=0.366, wavelength=16.0, width=12.0,
        # morphology="stack", loading="compression", IM7/8552,
        # angles=[0/45/-45/90]_3s (24 plies). Pre-change KD = 0.605573...
        assert result.analytical_knockdown == pytest.approx(
            0.605573420074152, rel=1e-9
        )
