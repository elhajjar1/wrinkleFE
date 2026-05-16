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

    def test_pure_transverse_compression_at_Yc(self, criterion, material):
        """Regression for #34/#25/#24.

        Pure sigma_22 = -Yc must give FI = 1.0 (failure exactly at the
        transverse compressive allowable).  The previous compliance-based
        implementation returned ~1.379 because Poisson-induced eps_33 was
        compared against the unrelated Zt/E3 allowable.
        """
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


class TestMaxStrainConsistentBasisRegression:
    """Regression suite for #34 / #25 / #24.

    The correctness contract for the engineering max-strain criterion is
    that every uniaxial allowable self-calibrates to FI = 1.0, because the
    working strains and the allowable strains are computed on the *same*
    uncoupled engineering basis (sigma_i / E_i), with no Poisson coupling.
    """

    # (label, stress vector, expected dominant mode)
    UNIAXIAL_CASES = [
        ("sigma11=+Xt", lambda m: [m.Xt, 0, 0, 0, 0, 0], "fiber_tension"),
        ("sigma11=-Xc", lambda m: [-m.Xc, 0, 0, 0, 0, 0], "fiber_compression"),
        ("sigma22=+Yt", lambda m: [0, m.Yt, 0, 0, 0, 0],
         "matrix_transverse_tension"),
        ("sigma22=-Yc", lambda m: [0, -m.Yc, 0, 0, 0, 0],
         "matrix_transverse_compression"),
        ("tau12=+S12", lambda m: [0, 0, 0, 0, 0, m.S12], "shear_12"),
    ]

    @pytest.mark.parametrize(
        "label,build,mode",
        UNIAXIAL_CASES,
        ids=[c[0] for c in UNIAXIAL_CASES],
    )
    def test_every_uniaxial_allowable_maps_to_fi_one(
        self, criterion, material, label, build, mode
    ):
        stress = np.array(build(material), dtype=float)
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, rel=1e-6), label
        assert result.mode == mode, label

    def test_zero_stress_gives_fi_zero(self, criterion, material):
        result = criterion.evaluate(np.zeros(6), material)
        assert result.index == pytest.approx(0.0, abs=1e-12)

    def test_transverse_compression_does_not_fire_thickness_tension(
        self, criterion, material
    ):
        """Core bug: pure sigma_22 = -Yc must NOT spuriously fail in the
        through-thickness tension mode via Poisson-coupled eps_33."""
        stress = np.array([0.0, -material.Yc, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.mode == "matrix_transverse_compression"
        assert result.index == pytest.approx(1.0, rel=1e-6)
        # The old compliance path returned ~1.379 here.
        assert result.index < 1.05

    @pytest.mark.parametrize(
        "build", [c[1] for c in UNIAXIAL_CASES], ids=[c[0] for c in UNIAXIAL_CASES]
    )
    def test_proportional_loading_is_linear(self, criterion, material, build):
        """Max-strain is linear in stress: doubling the stress state
        doubles FI, and the reserve factor scales the load to first
        failure (RF * stress -> FI = 1)."""
        stress = np.array(build(material), dtype=float)
        base = criterion.evaluate(stress, material)
        doubled = criterion.evaluate(2.0 * stress, material)
        assert doubled.index == pytest.approx(2.0 * base.index, rel=1e-9)
        assert base.reserve_factor == pytest.approx(1.0 / base.index, rel=1e-12)
        scaled = criterion.evaluate(base.reserve_factor * stress, material)
        assert scaled.index == pytest.approx(1.0, rel=1e-6)

    def test_working_strain_uses_uncoupled_engineering_basis(
        self, criterion, material
    ):
        """The working strain must be sigma_i/E_i with no Poisson terms.
        Half the transverse compressive allowable -> FI exactly 0.5 and
        no other component (notably eps_33) contributes."""
        stress = np.array([0.0, -0.5 * material.Yc, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(0.5, rel=1e-9)
        assert result.mode == "matrix_transverse_compression"
