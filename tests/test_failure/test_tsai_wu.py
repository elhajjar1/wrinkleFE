"""Tests for the Tsai-Wu 3-D failure criterion."""

import numpy as np
import pytest

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.failure.base import FailureResult
from wrinklefe.failure.tsai_wu import TsaiWuCriterion


@pytest.fixture
def material():
    return OrthotropicMaterial()


@pytest.fixture
def criterion():
    return TsaiWuCriterion(f12_star=-0.5)


class TestTsaiWuUniaxial:

    def test_zero_stress_gives_fi_zero(self, criterion, material):
        result = criterion.evaluate(np.zeros(6), material)
        assert isinstance(result, FailureResult)
        assert result.index == pytest.approx(0.0, abs=1e-12)

    def test_pure_fibre_tension_at_Xt(self, criterion, material):
        """sigma_11 = Xt: F1*Xt + F11*Xt^2 = (1-Xt/Xc) + Xt/Xc = 1."""
        stress = np.array([material.Xt, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-10)

    def test_pure_fibre_compression_at_Xc(self, criterion, material):
        """sigma_11 = -Xc: -F1*Xc + F11*Xc^2 = (-1+Xc/Xt) + Xc/Xt
        wait: L = F1*(-Xc) = -Xc/Xt + 1, Q = Xc/Xt, sum = 1."""
        stress = np.array([-material.Xc, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-10)

    def test_pure_transverse_tension_at_Yt(self, criterion, material):
        stress = np.array([0.0, material.Yt, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-10)

    def test_pure_transverse_compression_at_Yc(self, criterion, material):
        stress = np.array([0.0, -material.Yc, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-10)

    def test_pure_inplane_shear_at_S12(self, criterion, material):
        """Pure tau_12 = S12 gives F66*S12^2 = 1."""
        stress = np.array([0.0, 0.0, 0.0, 0.0, 0.0, material.S12])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-10)


class TestTsaiWuMixed:

    def test_combined_fibre_and_transverse_tension(self, criterion, material):
        """Hand-computed mixed case: sigma_11 = Xt/2, sigma_22 = Yt/2.

        Using F1 = 1/Xt - 1/Xc, F2 = 1/Yt - 1/Yc,
              F11 = 1/(Xt*Xc), F22 = 1/(Yt*Yc),
              F12 = -0.5 * sqrt(F11 * F22),
        the polynomial value is computed by hand below and compared.
        """
        Xt, Xc = material.Xt, material.Xc
        Yt, Yc = material.Yt, material.Yc

        s1 = 0.5 * Xt
        s2 = 0.5 * Yt

        F1 = 1.0 / Xt - 1.0 / Xc
        F2 = 1.0 / Yt - 1.0 / Yc
        F11 = 1.0 / (Xt * Xc)
        F22 = 1.0 / (Yt * Yc)
        F12 = -0.5 * np.sqrt(F11 * F22)

        expected = (
            F1 * s1 + F2 * s2
            + F11 * s1**2 + F22 * s2**2
            + 2.0 * F12 * s1 * s2
        )

        stress = np.array([s1, s2, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(expected, rel=1e-10)

    def test_reserve_factor_quadratic_root_for_pure_shear(self, criterion, material):
        """For a load path with L=0 (pure shear), the strength ratio R
        satisfies R^2 * Q = 1, so R = 1/sqrt(Q).  At tau_12 = 0.5 * S12,
        Q = 0.25, hence R = 2.0."""
        stress = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * material.S12])
        result = criterion.evaluate(stress, material)
        # Reserve factor should be the quadratic root, not 1/FI.
        # FI = 0.25 here, but R must equal 2.0.
        assert result.reserve_factor == pytest.approx(2.0, rel=1e-10)

    def test_reserve_factor_satisfies_quadratic(self, criterion, material):
        """For an arbitrary stress state, the reserve factor R must satisfy
        R^2 * Q + R * L = 1 (the closed-form definition of strength ratio)."""
        Xt, Xc = material.Xt, material.Xc
        Yt, Yc = material.Yt, material.Yc

        s1, s2, t12 = 0.4 * Xt, 0.3 * Yt, 0.2 * material.S12
        stress = np.array([s1, s2, 0.0, 0.0, 0.0, t12])

        F1 = 1.0 / Xt - 1.0 / Xc
        F2 = 1.0 / Yt - 1.0 / Yc
        F11 = 1.0 / (Xt * Xc)
        F22 = 1.0 / (Yt * Yc)
        F66 = 1.0 / material.S12 ** 2
        F12 = -0.5 * np.sqrt(F11 * F22)

        L = F1 * s1 + F2 * s2
        Q = F11 * s1**2 + F22 * s2**2 + 2.0 * F12 * s1 * s2 + F66 * t12**2

        result = criterion.evaluate(stress, material)
        R = result.reserve_factor
        assert R**2 * Q + R * L == pytest.approx(1.0, abs=1e-10)

    def test_reserve_factor_asymmetric_material(self, criterion):
        """Asymmetric material (Xt != Xc, Yt != Yc): R must satisfy the
        quadratic R^2 Q + R L = 1, and must NOT equal the naive 1/FI."""
        mat = OrthotropicMaterial(
            Xt=1500.0, Xc=1200.0,
            Yt=50.0, Yc=200.0,
            Zt=50.0, Zc=200.0,
            S12=80.0, S13=80.0, S23=60.0,
        )
        s1, s2, t12 = 0.35 * mat.Xt, 0.25 * mat.Yt, 0.15 * mat.S12
        stress = np.array([s1, s2, 0.0, 0.0, 0.0, t12])

        F1 = 1.0 / mat.Xt - 1.0 / mat.Xc
        F2 = 1.0 / mat.Yt - 1.0 / mat.Yc
        F11 = 1.0 / (mat.Xt * mat.Xc)
        F22 = 1.0 / (mat.Yt * mat.Yc)
        F66 = 1.0 / mat.S12 ** 2
        F12 = -0.5 * np.sqrt(F11 * F22)

        L = F1 * s1 + F2 * s2
        Q = F11 * s1**2 + F22 * s2**2 + 2.0 * F12 * s1 * s2 + F66 * t12**2

        result = criterion.evaluate(stress, mat)
        R = result.reserve_factor

        # The polynomial identity that defines R.
        assert R**2 * Q + R * L == pytest.approx(1.0, abs=1e-10)

        # And L must be meaningfully nonzero so this test would expose a
        # 1/FI bug -- i.e. the correct R must differ from 1/FI.
        assert abs(L) > 1e-6
        naive_rf = 1.0 / result.index
        assert abs(R - naive_rf) > 1e-3 * R

    def test_reserve_factor_symmetric_strengths_equals_one_over_sqrt_fi(self):
        """When Xt == Xc, Yt == Yc, etc., F1 = F2 = F3 = 0, so L = 0 and the
        polynomial reduces to R^2 Q = 1, giving R = 1/sqrt(FI) (NOT 1/FI)."""
        mat = OrthotropicMaterial(
            Xt=1500.0, Xc=1500.0,
            Yt=100.0, Yc=100.0,
            Zt=100.0, Zc=100.0,
            S12=80.0, S13=80.0, S23=60.0,
        )
        criterion = TsaiWuCriterion(f12_star=-0.5)
        stress = np.array([
            0.4 * mat.Xt, 0.3 * mat.Yt, 0.0, 0.0, 0.0, 0.2 * mat.S12,
        ])
        result = criterion.evaluate(stress, mat)
        expected_R = 1.0 / np.sqrt(result.index)
        assert result.reserve_factor == pytest.approx(expected_R, rel=1e-12)
