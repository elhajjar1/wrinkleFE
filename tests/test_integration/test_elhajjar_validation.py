"""End-to-end regression tests for Elhajjar (2025) analytical predictions.

Validates that the full WrinkleAnalysis pipeline produces physically
consistent knockdown factors for representative defect configurations.

References
----------
- Elhajjar, R. (2025). Scientific Reports, 15:25977.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def material():
    """IM7/8552 material."""
    return MaterialLibrary().get("IM7_8552")


def _small_config(material, **overrides):
    """Build a minimal AnalysisConfig for fast tests."""
    defaults = dict(
        amplitude=0.366,
        wavelength=16.0,
        width=12.0,
        morphology="stack",
        loading="compression",
        material=material,
        angles=[0, 90, 0, 90, 0, 90, 90, 0, 90, 0, 90, 0],  # 12 plies
        interface_1=5,
        interface_2=6,
        nx=6,
        ny=4,
        nz_per_ply=1,
        domain_width=10.0,
        applied_strain=-0.005,
        verbose=False,
    )
    defaults.update(overrides)
    return AnalysisConfig(**defaults)


# ======================================================================
# Amplitude dependence
# ======================================================================

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestAmplitudeDependence:
    """Higher amplitude -> lower knockdown (concavity of BF model)."""

    def test_zero_amplitude_gives_full_strength(self, material):
        """Near-zero amplitude -> knockdown ~ 1.0."""
        cfg = _small_config(material, amplitude=1e-10)
        result = WrinkleAnalysis(cfg).run()
        assert result.analytical_knockdown == pytest.approx(1.0, abs=0.01)

    def test_1A_knockdown(self, material):
        """1A amplitude -> moderate knockdown."""
        cfg = _small_config(material, amplitude=0.183)
        result = WrinkleAnalysis(cfg).run()
        assert 0.5 < result.analytical_knockdown < 1.0

    def test_2A_knockdown(self, material):
        """2A amplitude -> noticeable knockdown."""
        cfg = _small_config(material, amplitude=0.366)
        result = WrinkleAnalysis(cfg).run()
        assert 0.3 < result.analytical_knockdown < 0.95

    def test_3A_knockdown(self, material):
        """3A amplitude -> significant knockdown (< 0.7 for compression)."""
        cfg = _small_config(material, amplitude=0.549)
        result = WrinkleAnalysis(cfg).run()
        assert result.analytical_knockdown < 0.7
        assert result.analytical_knockdown > 0.0

    def test_monotonic_decrease(self, material):
        """Knockdown decreases monotonically with increasing amplitude."""
        amplitudes = [0.183, 0.366, 0.549]
        knockdowns = []
        for amp in amplitudes:
            cfg = _small_config(material, amplitude=amp)
            result = WrinkleAnalysis(cfg).run()
            knockdowns.append(result.analytical_knockdown)
        assert knockdowns[0] > knockdowns[1] > knockdowns[2]


# ======================================================================
# Morphology ordering (compression)
# ======================================================================

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestMorphologyOrdering:
    """Convex > stack > concave for compression."""

    def test_compression_ordering(self, material):
        """Convex morphology gives highest compression strength."""
        results = {}
        for morph in ["convex", "stack", "concave"]:
            cfg = _small_config(material, morphology=morph)
            results[morph] = WrinkleAnalysis(cfg).run()

        s_convex = results["convex"].analytical_strength_MPa
        s_stack = results["stack"].analytical_strength_MPa
        s_concave = results["concave"].analytical_strength_MPa

        assert s_convex > s_stack > s_concave

    def test_morphology_factors(self, material):
        """Morphology factor: convex < 1 < concave, stack == 1."""
        results = {}
        for morph in ["convex", "stack", "concave"]:
            cfg = _small_config(material, morphology=morph)
            results[morph] = WrinkleAnalysis(cfg).run()

        assert results["stack"].morphology_factor == pytest.approx(1.0, abs=1e-6)
        assert results["convex"].morphology_factor < 1.0
        assert results["concave"].morphology_factor > 1.0


# ======================================================================
# Strength consistency
# ======================================================================

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestStrengthConsistency:
    """Analytical strength = Xc * knockdown for compression."""

    def test_strength_equals_xc_times_knockdown(self, material):
        """Predicted strength should be Xc * analytical_knockdown."""
        cfg = _small_config(material)
        result = WrinkleAnalysis(cfg).run()
        expected = material.Xc * result.analytical_knockdown
        assert result.analytical_strength_MPa == pytest.approx(expected, rel=1e-10)

    def test_positive_strength(self, material):
        """All results should have positive strength."""
        for morph in ["convex", "stack", "concave"]:
            cfg = _small_config(material, morphology=morph)
            result = WrinkleAnalysis(cfg).run()
            assert result.analytical_strength_MPa > 0

    def test_knockdown_in_valid_range(self, material):
        """Knockdown must be in (0, 1]."""
        for amp in [0.183, 0.366, 0.549]:
            cfg = _small_config(material, amplitude=amp)
            result = WrinkleAnalysis(cfg).run()
            assert 0.0 < result.analytical_knockdown <= 1.0
