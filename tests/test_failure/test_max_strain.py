"""Tests for the Maximum Strain failure criterion."""

import numpy as np
import pytest

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.failure.base import FailureResult
from wrinklefe.failure.max_strain import MaxStrainCriterion


@pytest.fixture
def material():
    return OrthotropicMaterial()


@pytest.fixture
def criterion():
    return MaxStrainCriterion()


class TestMaxStrainUniaxial:

    def test_zero_stress_gives_fi_zero(self, criterion, material):
        result = criterion.evaluate(np.zeros(6), material)
        assert isinstance(result, FailureResult)
        assert result.index == pytest.approx(0.0, abs=1e-12)

    def test_pure_fibre_tension_at_Xt(self, criterion, material):
        """Under pure sigma_11=Xt the dominant strain ratio is fibre tension."""
        stress = np.array([material.Xt, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, rel=1e-6)
        assert result.mode == "fiber_tension"

    def test_pure_fibre_compression_at_Xc(self, criterion, material):
        stress = np.array([-material.Xc, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, rel=1e-6)
        assert result.mode == "fiber_compression"

    def test_pure_transverse_tension_at_Yt(self, criterion, material):
        stress = np.array([0.0, material.Yt, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, rel=1e-6)
        assert result.mode == "matrix_transverse_tension"

    @pytest.mark.xfail(
        reason="max_strain.py over-predicts FI ~1.379 under pure σ22=-Yc; tracked in #24",
        strict=True,
    )
    def test_pure_transverse_compression_at_Yc(self, criterion, material):
        stress = np.array([0.0, -material.Yc, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, rel=1e-6)
        assert result.mode == "matrix_transverse_compression"

    def test_pure_inplane_shear_at_S12(self, criterion, material):
        """Pure tau_12=S12 produces gamma_12=S12/G12=gamma12_ult, FI=1."""
        stress = np.array([0.0, 0.0, 0.0, 0.0, 0.0, material.S12])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, rel=1e-6)
        assert result.mode == "shear_12"

    def test_reserve_factor_is_inverse_of_index(self, criterion, material):
        stress = np.array([0.5 * material.Xt, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.reserve_factor == pytest.approx(1.0 / result.index, rel=1e-12)
