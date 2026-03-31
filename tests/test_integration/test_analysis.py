"""Integration tests for the WrinkleAnalysis pipeline.

Tests the high-level analysis orchestrator that chains together
material, laminate, wrinkle, mesh, solver, failure, and statistics modules.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from wrinklefe.analysis import AnalysisConfig, AnalysisResults, WrinkleAnalysis
from wrinklefe.core.material import OrthotropicMaterial, MaterialLibrary


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def default_material():
    """IM7/8552 material."""
    return MaterialLibrary().get("IM7_8552")


@pytest.fixture
def small_config(default_material):
    """Minimal-size config for fast tests (coarse mesh)."""
    return AnalysisConfig(
        amplitude=0.366,
        wavelength=16.0,
        width=12.0,
        morphology="stack",
        loading="compression",
        material=default_material,
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


# ======================================================================
# AnalysisConfig tests
# ======================================================================

class TestAnalysisConfig:
    """Tests for AnalysisConfig dataclass."""

    def test_defaults(self):
        """Default config creates valid object."""
        cfg = AnalysisConfig()
        assert cfg.amplitude == 0.366
        assert cfg.wavelength == 16.0
        assert cfg.morphology == "stack"
        assert cfg.loading == "compression"
        assert cfg.material is not None
        assert cfg.angles is not None
        assert len(cfg.angles) == 24  # [0/45/-45/90]_3s
        assert cfg.domain_length == pytest.approx(48.0)  # 3 * 16

    def test_auto_domain_length(self):
        """domain_length auto-computed as 3 * wavelength."""
        cfg = AnalysisConfig(wavelength=20.0)
        assert cfg.domain_length == pytest.approx(60.0)

    def test_explicit_domain_length(self):
        """Explicit domain_length is preserved."""
        cfg = AnalysisConfig(wavelength=20.0, domain_length=100.0)
        assert cfg.domain_length == pytest.approx(100.0)

    def test_default_material(self):
        """Default material is CYCOM X850/T800."""
        cfg = AnalysisConfig()
        assert cfg.material.E1 > 100_000  # ~161 GPa

    def test_custom_material(self, default_material):
        """Custom material is preserved."""
        cfg = AnalysisConfig(material=default_material)
        assert cfg.material is default_material


# ======================================================================
# Analytical-only tests
# ======================================================================

class TestAnalyticalOnly:
    """Tests for the analytical prediction pathway (no FE)."""

    def test_stack_analytical(self, small_config):
        """Stack morphology has M_f = 1.0."""
        small_config.morphology = "stack"
        analysis = WrinkleAnalysis(small_config)
        result = analysis.run()

        assert result.morphology_factor == pytest.approx(1.0, abs=1e-6)
        assert result.max_angle_rad > 0
        assert result.effective_angle_rad == pytest.approx(
            result.max_angle_rad * result.morphology_factor, rel=1e-6
        )
        assert result.analytical_strength_MPa > 0
        assert result.analytical_knockdown < 1.0
        assert result.analytical_knockdown > 0.0

    def test_convex_stronger_than_concave(self, small_config):
        """Convex morphology gives higher strength than concave."""
        small_config.morphology = "convex"
        convex = WrinkleAnalysis(small_config).run()

        small_config.morphology = "concave"
        concave = WrinkleAnalysis(small_config).run()

        assert convex.analytical_strength_MPa > concave.analytical_strength_MPa
        assert convex.morphology_factor < 1.0
        assert concave.morphology_factor > 1.0

    def test_damage_index_valid(self, small_config):
        """Damage index is in [0, 1) range."""
        analysis = WrinkleAnalysis(small_config)
        result = analysis.run()

        assert 0.0 <= result.damage_index < 1.0

    def test_knockdown_decomposition(self, small_config):
        """Compression knockdown = CLT-weighted BF kink-band."""
        analysis = WrinkleAnalysis(small_config)
        result = analysis.run()

        # gamma_Y_eff is computed from layup confinement, not material
        gamma_Y_eff = result.gamma_Y_eff
        kd_bf = 1.0 / (1.0 + result.effective_angle_rad / gamma_Y_eff)

        # CLT weighting: KD_lam = f0 * KD_BF + (1 - f0)
        angles = small_config.angles if small_config.angles else [0, 45, -45, 90] * 6
        mat = small_config.material
        n_0 = sum(1 for a in angles if abs(a) < 5)
        n_45 = sum(1 for a in angles if 40 < abs(a) < 50)
        n_90 = sum(1 for a in angles if abs(a) > 85)
        Q0 = mat.E1
        Q45 = mat.E1 / 4 + mat.E2 / 4 + mat.G12 / 2
        Q90 = mat.E2
        total = n_0 * Q0 + n_45 * Q45 + n_90 * Q90
        f_0 = n_0 * Q0 / total if total > 0 else 1.0
        expected = f_0 * kd_bf + (1.0 - f_0)

        assert result.analytical_knockdown == pytest.approx(expected, rel=1e-10)

    def test_strength_equals_xc_times_knockdown(self, small_config):
        """Predicted strength = Xc * combined_knockdown."""
        analysis = WrinkleAnalysis(small_config)
        result = analysis.run()

        expected = small_config.material.Xc * result.analytical_knockdown
        assert result.analytical_strength_MPa == pytest.approx(expected, rel=1e-10)

    def test_zero_amplitude_gives_full_strength(self, small_config):
        """Zero amplitude → no knockdown → full Xc."""
        small_config.amplitude = 1e-10  # near zero
        analysis = WrinkleAnalysis(small_config)
        result = analysis.run()

        assert result.analytical_knockdown == pytest.approx(1.0, abs=0.01)
        assert result.analytical_strength_MPa == pytest.approx(
            small_config.material.Xc, rel=0.01
        )


# ======================================================================
# Full pipeline tests (FE)
# ======================================================================

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestFullPipeline:
    """Tests for the complete FE pipeline."""

    def test_full_run_produces_results(self, small_config):
        """Full run returns valid AnalysisResults with all fields."""
        analysis = WrinkleAnalysis(small_config)
        result = analysis.run()

        assert result.mesh is not None
        assert result.laminate is not None
        assert result.wrinkle_config is not None
        assert result.field_results is not None
        assert result.failure_indices is not None
        assert result.mesh.n_elements > 0
        assert result.mesh.n_nodes > 0

    def test_displacement_field(self, small_config):
        """Displacement field has correct shape and non-trivial values."""
        result = WrinkleAnalysis(small_config).run()
        fr = result.field_results

        assert fr.displacement.shape == (result.mesh.n_nodes, 3)
        # Under compression, there should be non-zero x-displacements
        max_disp, _ = fr.max_displacement()
        assert max_disp > 0

    def test_stress_field_shape(self, small_config):
        """Stress fields have correct shapes."""
        result = WrinkleAnalysis(small_config).run()
        fr = result.field_results

        n_elem = result.mesh.n_elements
        assert fr.stress_global.shape[0] == n_elem
        assert fr.stress_global.shape[2] == 6
        assert fr.stress_local.shape == fr.stress_global.shape

    def test_failure_indices_populated(self, small_config):
        """Failure index fields are populated for each criterion."""
        result = WrinkleAnalysis(small_config).run()

        assert result.failure_indices is not None
        assert len(result.failure_indices) > 0
        for name, fi_field in result.failure_indices.items():
            assert fi_field.shape[0] == result.mesh.n_elements
            # FI should have finite values (some criteria like Tsai-Wu can
            # produce negative FI values, so we only check for finiteness)
            finite_frac = np.sum(np.isfinite(fi_field)) / fi_field.size
            assert finite_frac > 0.9  # at least 90% finite

    def test_summary_string(self, small_config):
        """Summary produces a non-empty string."""
        result = WrinkleAnalysis(small_config).run()
        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 100
        assert "WrinkleFE" in summary


# ======================================================================
# Morphology comparison tests
# ======================================================================

class TestCompare:
    """Tests for compare_morphologies."""

    def test_three_morphologies(self, small_config):
        """Comparing 3 morphologies returns 3 results."""
        results = WrinkleAnalysis.compare_morphologies(small_config)
        assert len(results) == 3
        assert "stack" in results
        assert "convex" in results
        assert "concave" in results

    def test_ranking_order(self, small_config):
        """Convex > stack > concave in compression strength."""
        results = WrinkleAnalysis.compare_morphologies(small_config)

        s_convex = results["convex"].analytical_strength_MPa
        s_stack = results["stack"].analytical_strength_MPa
        s_concave = results["concave"].analytical_strength_MPa

        assert s_convex > s_stack > s_concave

    def test_all_have_analytical_results(self, small_config):
        """All morphology results have non-zero analytical strength."""
        results = WrinkleAnalysis.compare_morphologies(small_config)
        for morph, res in results.items():
            assert res.analytical_strength_MPa > 0
            assert 0 < res.analytical_knockdown < 1


# ======================================================================
# Parametric sweep tests
# ======================================================================

class TestParametricSweep:
    """Tests for parametric_sweep."""

    def test_amplitude_sweep(self, small_config):
        """Sweeping amplitude returns correct number of results."""
        amps = [0.183, 0.366, 0.549]
        results = WrinkleAnalysis.parametric_sweep(
            small_config, "amplitude", amps
        )
        assert len(results) == 3

    def test_strength_decreases_with_amplitude(self, small_config):
        """Higher amplitude → lower strength."""
        amps = [0.183, 0.366, 0.549]
        results = WrinkleAnalysis.parametric_sweep(
            small_config, "amplitude", amps
        )
        strengths = [r.analytical_strength_MPa for r in results]
        assert strengths[0] > strengths[1] > strengths[2]

    def test_invalid_parameter_raises(self, small_config):
        """Invalid parameter name raises AttributeError."""
        with pytest.raises(AttributeError, match="no field"):
            WrinkleAnalysis.parametric_sweep(
                small_config, "nonexistent_param", [1, 2, 3]
            )


# ======================================================================
# Edge cases
# ======================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_large_amplitude(self, small_config):
        """Large amplitude (3A) still produces valid results."""
        small_config.amplitude = 0.549  # 3 ply thicknesses
        result = WrinkleAnalysis(small_config).run()
        assert result.analytical_strength_MPa > 0
        assert result.analytical_knockdown < 0.7  # significant knockdown
