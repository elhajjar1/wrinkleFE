"""Tests for LaRC04/05 failure criterion in wrinklefe.failure."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.failure.base import FailureResult
from wrinklefe.failure.larc05 import LaRC05Criterion


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def x850_material():
    """Default CYCOM X850/T800 OrthotropicMaterial."""
    return OrthotropicMaterial()


@pytest.fixture
def zero_stress():
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def uniaxial_tension():
    return np.array([500.0, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def uniaxial_compression():
    return np.array([-500.0, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def transverse_tension():
    return np.array([0.0, 30.0, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def pure_shear():
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 50.0])


@pytest.fixture
def combined_stress():
    return np.array([500.0, 30.0, 5.0, 2.0, 3.0, 50.0])


# ======================================================================
# Common behaviour
# ======================================================================

class TestLaRC05Common:

    def test_zero_stress_fi_zero(self, x850_material, zero_stress):
        criterion = LaRC05Criterion()
        result = criterion.evaluate(zero_stress, x850_material)
        assert result.index == pytest.approx(0.0, abs=1e-10)

    def test_returns_failure_result(self, x850_material, uniaxial_tension):
        criterion = LaRC05Criterion()
        result = criterion.evaluate(uniaxial_tension, x850_material)
        assert isinstance(result, FailureResult)
        assert hasattr(result, "index")
        assert hasattr(result, "mode")
        assert hasattr(result, "reserve_factor")

    def test_fi_non_negative(self, x850_material, combined_stress):
        criterion = LaRC05Criterion()
        result = criterion.evaluate(combined_stress, x850_material)
        assert result.index >= 0

    def test_name_is_larc05(self, x850_material, zero_stress):
        criterion = LaRC05Criterion()
        result = criterion.evaluate(zero_stress, x850_material)
        assert result.criterion_name == "larc05"


# ======================================================================
# LaRC05-specific tests
# ======================================================================

class TestLaRC05Criterion:

    def test_accepts_misalignment_via_context(self, x850_material):
        """Context dict should pass misalignment_angle to evaluate."""
        criterion = LaRC05Criterion()
        stress = np.array([-1000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ctx = {"misalignment_angle": 0.1}
        result = criterion.evaluate(stress, x850_material, ctx)
        assert result.index > 0

    def test_higher_misalignment_higher_fi_compression(self, x850_material):
        """Higher misalignment angle should increase FI under compression."""
        stress = np.array([-1000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        criterion = LaRC05Criterion()
        fi0 = criterion.evaluate(stress, x850_material, {"misalignment_angle": 0.0}).index
        fi1 = criterion.evaluate(stress, x850_material, {"misalignment_angle": 0.15}).index
        assert fi1 > fi0, (
            f"FI with misalignment 0.15 ({fi1}) should be > FI with 0 ({fi0})"
        )

    def test_misalignment_015_greater_than_zero(self, x850_material):
        """FI with misalignment=0.15 > FI with misalignment=0 under compression."""
        stress = np.array([-800.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        criterion = LaRC05Criterion()
        fi0 = criterion.evaluate(stress, x850_material, {"misalignment_angle": 0.0}).index
        fi1 = criterion.evaluate(stress, x850_material, {"misalignment_angle": 0.15}).index
        assert fi1 > fi0

    def test_fiber_tension_mode(self, x850_material):
        criterion = LaRC05Criterion()
        stress = np.array([2000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, x850_material)
        assert result.mode == "fiber_tension"

    def test_fiber_kinking_mode_under_compression(self, x850_material):
        criterion = LaRC05Criterion()
        stress = np.array([-1200.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ctx = {"misalignment_angle": 0.1}
        result = criterion.evaluate(stress, x850_material, ctx)
        assert result.mode in ("fiber_kinking", "matrix_tension", "matrix_compression")

    def test_in_situ_correction_for_thin_ply_no_fracture_toughness(self, x850_material):
        """Thin ply without GIc/GIIc should use simplified in-situ strengths."""
        mat_no_ft = OrthotropicMaterial(GIc=None, GIIc=None)
        c_thin = LaRC05Criterion(ply_thickness=0.183, t_ref=0.183)
        Yt_is, S12_is = c_thin._in_situ_strengths(mat_no_ft)
        assert_allclose(Yt_is, 1.12 * np.sqrt(2) * mat_no_ft.Yt, rtol=1e-10)
        assert_allclose(S12_is, np.sqrt(2) * mat_no_ft.S12, rtol=1e-10)

    def test_in_situ_fracture_toughness_based(self, x850_material):
        """With GIc/GIIc, in-situ strengths should be fracture-toughness-based."""
        c_thin = LaRC05Criterion(ply_thickness=0.183, t_ref=0.183)
        Yt_is, S12_is = c_thin._in_situ_strengths(x850_material)
        assert Yt_is >= x850_material.Yt
        assert S12_is >= x850_material.S12

    def test_no_in_situ_correction_for_thick_ply(self, x850_material):
        """Thick ply (thickness >= 2*t_ref) should use base strengths."""
        c_thick = LaRC05Criterion(ply_thickness=0.5, t_ref=0.183)
        Yt_is, S12_is = c_thick._in_situ_strengths(x850_material)
        assert_allclose(Yt_is, x850_material.Yt, rtol=1e-10)
        assert_allclose(S12_is, x850_material.S12, rtol=1e-10)

    def test_fi_at_Xt(self, x850_material):
        criterion = LaRC05Criterion()
        stress = np.array([x850_material.Xt, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, x850_material)
        assert_allclose(result.index, 1.0, atol=0.01)

    def test_friction_coefficients(self, x850_material):
        """Friction coefficients should be derived from alpha_0."""
        mu_L, mu_T = LaRC05Criterion._friction_coefficients(x850_material)
        assert 0.1 < mu_L < 0.5
        assert 0.1 < mu_T < 0.5

    def test_compression_worse_than_tension(self, x850_material):
        """With misalignment, compression FI should exceed tension FI."""
        criterion = LaRC05Criterion()
        ctx = {"misalignment_angle": 0.15}
        s_comp = np.array([-1610.0, -20.0, -5.0, 1.0, 2.0, -50.0])
        s_tens = np.array([1610.0, 20.0, 5.0, 1.0, 2.0, 50.0])
        fi_comp = criterion.evaluate(s_comp, x850_material, ctx).index
        fi_tens = criterion.evaluate(s_tens, x850_material, ctx).index
        assert fi_comp > fi_tens
