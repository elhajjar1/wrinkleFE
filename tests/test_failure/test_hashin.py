"""Tests for the Hashin 3-D failure criterion."""

import numpy as np
import pytest

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.failure.base import FailureResult
from wrinklefe.failure.hashin import HashinCriterion


@pytest.fixture
def material():
    return OrthotropicMaterial()


@pytest.fixture
def criterion():
    return HashinCriterion()


class TestHashinUniaxial:

    def test_zero_stress_gives_fi_zero(self, criterion, material):
        result = criterion.evaluate(np.zeros(6), material)
        assert isinstance(result, FailureResult)
        assert result.index == pytest.approx(0.0, abs=1e-12)

    def test_pure_fibre_tension_at_Xt(self, criterion, material):
        """sigma_11 = Xt with no shear gives FI_ft = 1."""
        stress = np.array([material.Xt, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-10)
        assert result.mode == "fiber_tension"

    def test_pure_fibre_compression_at_Xc(self, criterion, material):
        """sigma_11 = -Xc gives FI_fc = |sigma_11|/Xc = 1."""
        stress = np.array([-material.Xc, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-10)
        assert result.mode == "fiber_compression"

    def test_pure_transverse_tension_at_Yt(self, criterion, material):
        """sigma_22 = Yt (sigma_33=0, all shears=0) gives FI_mt = 1."""
        stress = np.array([0.0, material.Yt, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-10)
        assert result.mode == "matrix_tension"

    def test_pure_transverse_compression_at_Yc(self, criterion, material):
        """sigma_22 = -Yc gives FI_mc = 1 (the cross-coupling term cancels)."""
        stress = np.array([0.0, -material.Yc, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-10)
        assert result.mode == "matrix_compression"

    def test_pure_inplane_shear_at_S12(self, criterion, material):
        """Pure tau_12=S12 gives FI_ft = sqrt((tau12/S12)^2) = 1
        (no fibre stress, so the only positive contribution is the shear term)."""
        stress = np.array([0.0, 0.0, 0.0, 0.0, 0.0, material.S12])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-10)

    def test_reserve_factor_is_inverse_of_index(self, criterion, material):
        stress = np.array([0.5 * material.Xt, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.reserve_factor == pytest.approx(1.0 / result.index, rel=1e-12)
