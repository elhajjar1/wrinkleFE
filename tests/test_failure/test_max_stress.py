"""Tests for the Maximum Stress failure criterion."""

import numpy as np
import pytest

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.failure.base import FailureResult
from wrinklefe.failure.max_stress import MaxStressCriterion


@pytest.fixture
def material():
    """Default orthotropic material (IM7/8552-class)."""
    return OrthotropicMaterial()


@pytest.fixture
def criterion():
    return MaxStressCriterion()


class TestMaxStressUniaxial:

    def test_zero_stress_gives_fi_zero(self, criterion, material):
        result = criterion.evaluate(np.zeros(6), material)
        assert isinstance(result, FailureResult)
        assert result.index == pytest.approx(0.0, abs=1e-12)

    def test_pure_fibre_tension_at_Xt(self, criterion, material):
        stress = np.array([material.Xt, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-12)
        assert result.mode == "fiber_tension"

    def test_pure_fibre_compression_at_Xc(self, criterion, material):
        stress = np.array([-material.Xc, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-12)
        assert result.mode == "fiber_compression"

    def test_pure_transverse_tension_at_Yt(self, criterion, material):
        stress = np.array([0.0, material.Yt, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-12)
        assert result.mode == "matrix_tension"

    def test_pure_transverse_compression_at_Yc(self, criterion, material):
        stress = np.array([0.0, -material.Yc, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-12)
        assert result.mode == "matrix_compression"

    def test_pure_inplane_shear_at_S12(self, criterion, material):
        stress = np.array([0.0, 0.0, 0.0, 0.0, 0.0, material.S12])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-12)
        assert result.mode == "shear_12"

    def test_reserve_factor_is_inverse_of_index(self, criterion, material):
        stress = np.array([0.5 * material.Xt, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.reserve_factor == pytest.approx(1.0 / result.index, rel=1e-12)
