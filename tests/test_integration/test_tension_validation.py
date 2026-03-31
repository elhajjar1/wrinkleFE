"""Tension model validation tests.

Tests the tension analysis pathway including the three-mechanism model
(fiber tension, matrix tension, out-of-plane delamination) and
CLT-weighted laminate knockdown.

References
----------
- Pinho et al. (2005) NASA-TM-2005-213530 (LaRC04)
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
        loading="tension",
        material=material,
        angles=[0, 90, 0, 90, 0, 90, 90, 0, 90, 0, 90, 0],  # 12 plies
        interface_1=5,
        interface_2=6,
        nx=6,
        ny=4,
        nz_per_ply=1,
        domain_width=10.0,
        applied_strain=0.005,
        verbose=False,
    )
    defaults.update(overrides)
    return AnalysisConfig(**defaults)


# ======================================================================
# Basic tension runs
# ======================================================================

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestTensionRuns:
    """Basic tension analysis completes without error."""

    def test_tension_runs(self, material):
        """Tension analysis completes and returns valid knockdown."""
        cfg = _small_config(material)
        result = WrinkleAnalysis(cfg).run()
        assert result.analytical_knockdown > 0
        assert result.analytical_strength_MPa > 0

    def test_tension_zero_amplitude(self, material):
        """Near-zero amplitude -> knockdown ~ 1.0 for tension."""
        cfg = _small_config(material, amplitude=1e-10)
        result = WrinkleAnalysis(cfg).run()
        assert result.analytical_knockdown == pytest.approx(1.0, abs=0.01)

    def test_tension_strength_uses_xt(self, material):
        """Tension strength = Xt * knockdown (not Xc)."""
        cfg = _small_config(material)
        result = WrinkleAnalysis(cfg).run()
        expected = material.Xt * result.analytical_knockdown
        assert result.analytical_strength_MPa == pytest.approx(expected, rel=1e-10)


# ======================================================================
# Tension vs compression severity
# ======================================================================

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestTensionVsCompression:
    """Tension knockdown typically less severe than compression."""

    def test_tension_knockdown_less_severe(self, material):
        """For the same defect, tension knockdown >= compression knockdown.

        This is because kink-band (compression) is highly sensitive to
        misalignment, while tension mechanisms are generally less severe.
        """
        cfg_t = _small_config(material, loading="tension")
        cfg_c = _small_config(material, loading="compression")

        result_t = WrinkleAnalysis(cfg_t).run()
        result_c = WrinkleAnalysis(cfg_c).run()

        # Tension knockdown should be less severe (higher value) than compression
        assert result_t.analytical_knockdown >= result_c.analytical_knockdown

    def test_both_decrease_with_amplitude(self, material):
        """Both tension and compression knockdown decrease with amplitude."""
        for loading in ["tension", "compression"]:
            kd_low = WrinkleAnalysis(
                _small_config(material, loading=loading, amplitude=0.183)
            ).run().analytical_knockdown
            kd_high = WrinkleAnalysis(
                _small_config(material, loading=loading, amplitude=0.549)
            ).run().analytical_knockdown
            assert kd_low > kd_high, f"{loading}: 1A should have higher KD than 3A"


# ======================================================================
# Tension morphology ordering
# ======================================================================

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestTensionMorphologyOrdering:
    """Tension morphology ordering differs from compression.

    In tension, the morphology effect is governed by stress concentration
    and delamination rather than kink-band, so the ordering may differ
    from compression (where concave is worst).
    """

    def test_tension_morphology_all_positive(self, material):
        """All morphologies produce positive tension knockdowns.

        Note: the tension three-mechanism model may produce identical
        knockdowns across morphologies when the controlling mechanism
        (e.g., fiber cos^2 or OOP) does not depend on morphology factor.
        This is physically correct -- morphology primarily affects
        compression kink-band, not tension fiber breakage.
        """
        results = {}
        for morph in ["convex", "stack", "concave"]:
            cfg = _small_config(material, morphology=morph)
            results[morph] = WrinkleAnalysis(cfg).run()

        for morph, res in results.items():
            assert res.analytical_knockdown > 0, f"{morph} knockdown should be positive"
            assert res.analytical_strength_MPa > 0, f"{morph} strength should be positive"

    def test_all_morphologies_positive(self, material):
        """All morphologies give positive tension strength."""
        for morph in ["convex", "stack", "concave"]:
            cfg = _small_config(material, morphology=morph)
            result = WrinkleAnalysis(cfg).run()
            assert result.analytical_strength_MPa > 0
            assert 0 < result.analytical_knockdown <= 1.0


# ======================================================================
# Tension mechanisms dict
# ======================================================================

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestTensionMechanisms:
    """The tension_mechanisms dict should be populated with expected keys."""

    def test_tension_mechanisms_populated(self, material):
        """Tension results should include mechanism decomposition."""
        cfg = _small_config(material)
        result = WrinkleAnalysis(cfg).run()
        assert result.tension_mechanisms is not None

    def test_tension_mechanisms_keys(self, material):
        """Mechanisms dict should have fiber, matrix, oop, and mode keys."""
        cfg = _small_config(material)
        result = WrinkleAnalysis(cfg).run()
        mechs = result.tension_mechanisms
        assert "kd_fiber" in mechs
        assert "kd_matrix" in mechs
        assert "kd_oop" in mechs
        assert "kd_0" in mechs
        assert "mode" in mechs

    def test_tension_mechanisms_valid_ranges(self, material):
        """All mechanism knockdowns should be in (0, 1]."""
        cfg = _small_config(material)
        result = WrinkleAnalysis(cfg).run()
        mechs = result.tension_mechanisms
        for key in ["kd_fiber", "kd_matrix", "kd_oop", "kd_0"]:
            assert 0.0 < mechs[key] <= 1.0, f"{key} = {mechs[key]} out of range"

    def test_kd_0_is_minimum(self, material):
        """kd_0 should be the minimum of the three mechanisms."""
        cfg = _small_config(material)
        result = WrinkleAnalysis(cfg).run()
        mechs = result.tension_mechanisms
        expected_min = min(mechs["kd_fiber"], mechs["kd_matrix"], mechs["kd_oop"])
        assert mechs["kd_0"] == pytest.approx(expected_min, rel=1e-10)

    def test_compression_has_no_tension_mechanisms(self, material):
        """Compression results should NOT have tension_mechanisms."""
        cfg = _small_config(material, loading="compression")
        result = WrinkleAnalysis(cfg).run()
        assert result.tension_mechanisms is None

    def test_mode_is_string(self, material):
        """The controlling mode should be a non-empty string."""
        cfg = _small_config(material)
        result = WrinkleAnalysis(cfg).run()
        mode = result.tension_mechanisms["mode"]
        assert isinstance(mode, str)
        assert len(mode) > 0
