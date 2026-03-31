"""Tests for BudianskyFleckKinkBand and InterlaminarDamage models."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.failure.base import FailureResult
from wrinklefe.failure.kinkband import BudianskyFleckKinkBand, InterlaminarDamage


@pytest.fixture
def x850_material():
    return OrthotropicMaterial()


# ======================================================================
# BudianskyFleckKinkBand tests
# ======================================================================

class TestBudianskyFleckKinkBand:

    def test_knockdown_formula(self, x850_material):
        """knockdown = 1 / (1 + theta / gamma_Y)."""
        theta = 0.1
        gamma_Y = x850_material.gamma_Y  # 0.02
        kb = BudianskyFleckKinkBand(theta_eff=theta)
        expected = 1.0 / (1.0 + theta / gamma_Y)
        assert_allclose(kb.knockdown(material=x850_material), expected, rtol=1e-12)

    def test_knockdown_theta_zero_gives_one(self, x850_material):
        """theta=0 should give knockdown=1.0."""
        kb = BudianskyFleckKinkBand(theta_eff=0.0)
        assert_allclose(kb.knockdown(material=x850_material), 1.0, atol=1e-12)

    def test_knockdown_large_theta_approaches_zero(self, x850_material):
        """Large theta should give knockdown approaching 0."""
        kb = BudianskyFleckKinkBand(theta_eff=10.0)
        kd = kb.knockdown(material=x850_material)
        assert kd < 0.01
        assert kd > 0.0

    def test_knockdown_with_explicit_gamma_Y(self):
        """Can pass gamma_Y directly instead of material."""
        kb = BudianskyFleckKinkBand(theta_eff=0.04)
        kd = kb.knockdown(gamma_Y=0.02)
        expected = 1.0 / (1.0 + 0.04 / 0.02)
        assert_allclose(kd, expected, rtol=1e-12)

    def test_knockdown_requires_gamma_Y_or_material(self):
        """Providing neither gamma_Y nor material should raise ValueError."""
        kb = BudianskyFleckKinkBand(theta_eff=0.1)
        with pytest.raises(ValueError):
            kb.knockdown()

    def test_delamination_knockdown_d_zero(self):
        """D=0 should give delamination_knockdown=1.0."""
        kb = BudianskyFleckKinkBand(theta_eff=0.0, damage_index=0.0)
        assert_allclose(kb.delamination_knockdown(), 1.0, atol=1e-12)

    def test_delamination_knockdown_formula(self):
        """delamination_knockdown = (1-D)^1.5."""
        D = 0.3
        kb = BudianskyFleckKinkBand(theta_eff=0.0, damage_index=D)
        expected = (1.0 - D) ** 1.5
        assert_allclose(kb.delamination_knockdown(), expected, rtol=1e-12)

    def test_combined_knockdown_d_zero(self, x850_material):
        """D=0 combined_knockdown should equal knockdown alone."""
        theta = 0.1
        kb = BudianskyFleckKinkBand(theta_eff=theta, damage_index=0.0)
        kd = kb.knockdown(material=x850_material)
        ckd = kb.combined_knockdown(material=x850_material)
        assert_allclose(ckd, kd, rtol=1e-12)

    def test_combined_knockdown_formula(self, x850_material):
        """combined_knockdown = knockdown * (1-D)^1.5."""
        theta = 0.1
        D = 0.2
        kb = BudianskyFleckKinkBand(theta_eff=theta, damage_index=D)
        expected = (1.0 / (1.0 + theta / x850_material.gamma_Y)) * (1.0 - D) ** 1.5
        assert_allclose(kb.combined_knockdown(material=x850_material), expected, rtol=1e-12)

    def test_evaluate_returns_failure_result(self, x850_material):
        kb = BudianskyFleckKinkBand(theta_eff=0.1, damage_index=0.1)
        stress = np.array([-1000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = kb.evaluate(stress, x850_material)
        assert isinstance(result, FailureResult)
        assert result.mode == "kink_band"
        assert result.criterion_name == "budiansky_fleck"

    def test_evaluate_fi_formula(self, x850_material):
        """FI = |sigma_11| / (Xc * combined_knockdown)."""
        theta = 0.1
        D = 0.1
        kb = BudianskyFleckKinkBand(theta_eff=theta, damage_index=D)
        s1 = -800.0
        stress = np.array([s1, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = kb.evaluate(stress, x850_material)
        ckd = kb.combined_knockdown(material=x850_material)
        expected_fi = abs(s1) / (x850_material.Xc * ckd)
        assert_allclose(result.index, expected_fi, rtol=1e-10)

    def test_negative_theta_raises(self):
        with pytest.raises(ValueError):
            BudianskyFleckKinkBand(theta_eff=-0.1)

    def test_damage_out_of_range_raises(self):
        with pytest.raises(ValueError):
            BudianskyFleckKinkBand(theta_eff=0.0, damage_index=1.0)
        with pytest.raises(ValueError):
            BudianskyFleckKinkBand(theta_eff=0.0, damage_index=-0.1)


# ======================================================================
# InterlaminarDamage tests
# ======================================================================

class TestInterlaminarDamage:

    def test_damage_index_computation(self):
        """D = D0 * (A/A1)^1.5 * (1 + beta*max(theta-theta_c, 0)) * M_f."""
        dmg = InterlaminarDamage(D0=0.15, beta_angle=3.0, theta_crit=0.1)
        A = 0.366
        theta = 0.15
        Mf = 1.0
        A1 = 0.183
        expected = 0.15 * (A / A1) ** 1.5 * (1.0 + 3.0 * (theta - 0.1)) * Mf
        D = dmg.damage_index(amplitude=A, theta=theta, morphology_factor=Mf)
        assert_allclose(D, expected, rtol=1e-10)

    def test_damage_zero_when_amplitude_zero(self):
        dmg = InterlaminarDamage()
        D = dmg.damage_index(amplitude=0.0, theta=0.2, morphology_factor=1.0)
        assert_allclose(D, 0.0, atol=1e-12)

    def test_damage_below_theta_crit(self):
        """When theta < theta_crit, angle_excess = 0, so angle term = 1."""
        dmg = InterlaminarDamage(D0=0.15, beta_angle=3.0, theta_crit=0.1)
        A = 0.183
        theta = 0.05  # below theta_crit
        Mf = 1.0
        D = dmg.damage_index(amplitude=A, theta=theta, morphology_factor=Mf)
        expected = 0.15 * 1.0 * 1.0 * 1.0  # (A/A1)^1.5 = 1, angle_term=1
        assert_allclose(D, expected, rtol=1e-10)

    def test_damage_clamped_below_one(self):
        """Damage should be clamped to 0.999."""
        dmg = InterlaminarDamage(D0=0.15)
        # Very large amplitude should produce D > 1, but clamped
        D = dmg.damage_index(amplitude=10.0, theta=1.0, morphology_factor=5.0)
        assert D <= 0.999

    def test_damage_to_strength_d_zero(self):
        dmg = InterlaminarDamage()
        assert_allclose(dmg.damage_to_strength(0.0), 1.0, atol=1e-12)

    def test_damage_to_strength_d_one(self):
        """D=1 should give strength=0."""
        dmg = InterlaminarDamage()
        assert_allclose(dmg.damage_to_strength(1.0), 0.0, atol=1e-12)

    def test_damage_to_strength_formula(self):
        """damage_to_strength = (1-D)^1.5."""
        dmg = InterlaminarDamage()
        D = 0.3
        expected = (1.0 - D) ** 1.5
        assert_allclose(dmg.damage_to_strength(D), expected, rtol=1e-12)

    def test_morphology_factor_effect(self):
        """Higher morphology factor should increase damage."""
        dmg = InterlaminarDamage()
        D_low = dmg.damage_index(amplitude=0.366, theta=0.15, morphology_factor=0.75)
        D_high = dmg.damage_index(amplitude=0.366, theta=0.15, morphology_factor=1.336)
        assert D_high > D_low
