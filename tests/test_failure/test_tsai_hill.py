"""Tests for the Tsai-Hill 3-D failure criterion."""

import numpy as np
import pytest

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.failure.base import FailureResult
from wrinklefe.failure.tsai_hill import TsaiHillCriterion


@pytest.fixture
def material():
    return OrthotropicMaterial()


@pytest.fixture
def criterion():
    return TsaiHillCriterion()


class TestTsaiHillUniaxial:

    def test_zero_stress_gives_fi_zero(self, criterion, material):
        result = criterion.evaluate(np.zeros(6), material)
        assert isinstance(result, FailureResult)
        assert result.index == pytest.approx(0.0, abs=1e-12)

    def test_pure_fibre_tension_at_Xt(self, criterion, material):
        """sigma_11 = Xt with all other components zero gives
        FI = (Xt/Xt)^2 = 1 since interaction terms vanish (s2=s3=0)."""
        stress = np.array([material.Xt, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-10)
        assert result.mode == "fiber_tension"

    def test_pure_fibre_compression_at_Xc(self, criterion, material):
        """sigma_11 = -Xc gives FI = (-Xc/Xc)^2 = 1."""
        stress = np.array([-material.Xc, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-10)
        assert result.mode == "fiber_compression"

    def test_pure_transverse_tension_at_Yt(self, criterion, material):
        stress = np.array([0.0, material.Yt, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-10)
        assert result.mode == "matrix_transverse_tension"

    def test_pure_transverse_compression_at_Yc(self, criterion, material):
        stress = np.array([0.0, -material.Yc, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-10)
        assert result.mode == "matrix_transverse_compression"

    def test_pure_inplane_shear_at_S12(self, criterion, material):
        """Pure tau_12 = S12 gives FI = (S12/S12)^2 = 1."""
        stress = np.array([0.0, 0.0, 0.0, 0.0, 0.0, material.S12])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-10)
        assert result.mode == "shear_12"


class TestTsaiHillMixed:

    def test_combined_fibre_and_transverse_tension(self, criterion, material):
        """Mixed-stress hand-computed case: sigma_11=Xt/2, sigma_22=Yt/2.

        FI = (s1/Xt)^2 - s1*s2/Xt^2 + (s2/Yt)^2
        With s1=Xt/2, s2=Yt/2:
            = 0.25 - (Xt*Yt/4)/Xt^2 + 0.25
            = 0.5 - Yt/(4*Xt)
        """
        s1 = 0.5 * material.Xt
        s2 = 0.5 * material.Yt
        expected = (
            (s1 / material.Xt) ** 2
            - s1 * s2 / material.Xt ** 2
            + (s2 / material.Yt) ** 2
        )

        stress = np.array([s1, s2, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(expected, rel=1e-10)

    def test_combined_with_shear(self, criterion, material):
        """Mixed-stress hand-computed case with shear: s1, s2, t12 all present.

        FI = (s1/Xt)^2 - s1*s2/Xt^2 + (s2/Yt)^2 + (t12/S12)^2
        """
        s1 = 0.4 * material.Xt
        s2 = 0.3 * material.Yt
        t12 = 0.2 * material.S12

        expected = (
            (s1 / material.Xt) ** 2
            - s1 * s2 / material.Xt ** 2
            + (s2 / material.Yt) ** 2
            + (t12 / material.S12) ** 2
        )

        stress = np.array([s1, s2, 0.0, 0.0, 0.0, t12])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(expected, rel=1e-10)

    def test_reserve_factor_is_inverse_of_index(self, criterion, material):
        stress = np.array([0.5 * material.Xt, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.reserve_factor == pytest.approx(1.0 / result.index, rel=1e-12)

    def test_criterion_name(self, criterion, material):
        result = criterion.evaluate(np.zeros(6), material)
        assert result.criterion_name == "tsai_hill"
