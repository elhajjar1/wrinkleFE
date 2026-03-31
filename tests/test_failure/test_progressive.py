"""Tests for progressive damage models: PlyDiscount and ContinuumDamage."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.failure.base import FailureResult
from wrinklefe.failure.progressive import PlyDiscount, ContinuumDamage


@pytest.fixture
def x850_material():
    return OrthotropicMaterial()


def _make_failure(index, mode, criterion_name="test"):
    """Helper to create a FailureResult."""
    rf = 1.0 / index if index > 0 else float("inf")
    return FailureResult(index=index, mode=mode, reserve_factor=rf,
                         criterion_name=criterion_name)


# ======================================================================
# PlyDiscount tests
# ======================================================================

class TestPlyDiscount:

    def test_no_degradation_below_fi_one(self, x850_material):
        """FI < 1.0 should return original material unchanged."""
        model = PlyDiscount(residual_factor=0.01)
        failure = _make_failure(0.5, "fiber_tension")
        degraded = model.degrade(x850_material, failure)
        assert degraded.E1 == x850_material.E1
        assert degraded.E2 == x850_material.E2

    def test_fiber_failure_degrades_E1(self, x850_material):
        """Fiber failure should degrade E1."""
        model = PlyDiscount(residual_factor=0.01)
        failure = _make_failure(1.5, "fiber_tension")
        degraded = model.degrade(x850_material, failure)
        assert_allclose(degraded.E1, x850_material.E1 * 0.01, rtol=1e-10)
        # E2 should remain unchanged
        assert_allclose(degraded.E2, x850_material.E2, rtol=1e-10)

    def test_fiber_compression_degrades_E1(self, x850_material):
        model = PlyDiscount(residual_factor=0.01)
        failure = _make_failure(1.5, "fiber_compression")
        degraded = model.degrade(x850_material, failure)
        assert_allclose(degraded.E1, x850_material.E1 * 0.01, rtol=1e-10)
        assert_allclose(degraded.nu12, x850_material.nu12 * 0.01, rtol=1e-10)
        assert_allclose(degraded.nu13, x850_material.nu13 * 0.01, rtol=1e-10)

    def test_matrix_failure_degrades_E2(self, x850_material):
        """Matrix failure should degrade E2, G12, G23, nu23."""
        model = PlyDiscount(residual_factor=0.01)
        failure = _make_failure(1.2, "matrix_tension")
        degraded = model.degrade(x850_material, failure)
        assert_allclose(degraded.E2, x850_material.E2 * 0.01, rtol=1e-10)
        assert_allclose(degraded.G12, x850_material.G12 * 0.01, rtol=1e-10)
        assert_allclose(degraded.G23, x850_material.G23 * 0.01, rtol=1e-10)
        assert_allclose(degraded.nu23, x850_material.nu23 * 0.01, rtol=1e-10)
        # E1 should remain unchanged
        assert_allclose(degraded.E1, x850_material.E1, rtol=1e-10)

    def test_matrix_compression_degrades_E2(self, x850_material):
        model = PlyDiscount(residual_factor=0.01)
        failure = _make_failure(1.2, "matrix_compression")
        degraded = model.degrade(x850_material, failure)
        assert_allclose(degraded.E2, x850_material.E2 * 0.01, rtol=1e-10)

    def test_degraded_material_has_lower_stiffness(self, x850_material):
        """After fiber failure, degraded E1 should be much less than original."""
        model = PlyDiscount(residual_factor=0.01)
        failure = _make_failure(2.0, "fiber_tension")
        degraded = model.degrade(x850_material, failure)
        assert degraded.E1 < x850_material.E1 * 0.1

    def test_degraded_material_is_valid(self, x850_material):
        """Degraded material should still be a valid OrthotropicMaterial."""
        model = PlyDiscount(residual_factor=0.01)
        failure = _make_failure(1.5, "fiber_tension")
        degraded = model.degrade(x850_material, failure)
        assert isinstance(degraded, OrthotropicMaterial)
        assert degraded.E1 > 0

    def test_degraded_name_updated(self, x850_material):
        model = PlyDiscount(residual_factor=0.01)
        failure = _make_failure(1.5, "fiber_tension")
        degraded = model.degrade(x850_material, failure)
        assert "degraded" in degraded.name

    def test_shear_12_degrades_G12(self, x850_material):
        model = PlyDiscount(residual_factor=0.01)
        failure = _make_failure(1.5, "shear_12")
        degraded = model.degrade(x850_material, failure)
        assert_allclose(degraded.G12, x850_material.G12 * 0.01, rtol=1e-10)
        # Other properties unchanged
        assert_allclose(degraded.E1, x850_material.E1, rtol=1e-10)
        assert_allclose(degraded.E2, x850_material.E2, rtol=1e-10)

    def test_through_thickness_degrades_E3_G13_G23(self, x850_material):
        model = PlyDiscount(residual_factor=0.01)
        failure = _make_failure(1.5, "through_thickness_tension")
        degraded = model.degrade(x850_material, failure)
        assert_allclose(degraded.E3, x850_material.E3 * 0.01, rtol=1e-10)
        assert_allclose(degraded.G13, x850_material.G13 * 0.01, rtol=1e-10)
        assert_allclose(degraded.G23, x850_material.G23 * 0.01, rtol=1e-10)

    def test_invalid_residual_factor_raises(self):
        with pytest.raises(ValueError):
            PlyDiscount(residual_factor=0.0)
        with pytest.raises(ValueError):
            PlyDiscount(residual_factor=1.0)
        with pytest.raises(ValueError):
            PlyDiscount(residual_factor=-0.1)


# ======================================================================
# ContinuumDamage tests
# ======================================================================

class TestContinuumDamage:

    def test_initial_state_undamaged(self):
        cdm = ContinuumDamage()
        assert_allclose(cdm.d_fiber, 0.0)
        assert_allclose(cdm.d_matrix, 0.0)
        assert_allclose(cdm.d_shear, 0.0)
        assert not cdm.is_damaged

    def test_update_damage_increases_variables(self):
        """Fiber failure should increase d_fiber."""
        cdm = ContinuumDamage()
        failure = _make_failure(2.0, "fiber_tension")
        cdm.update_damage(failure)
        assert cdm.d_fiber > 0.0
        assert cdm.is_damaged

    def test_update_damage_matrix_mode(self):
        cdm = ContinuumDamage()
        failure = _make_failure(1.5, "matrix_tension")
        cdm.update_damage(failure)
        assert cdm.d_matrix > 0.0
        # d_increment = 1 - 1/1.5 = 1/3
        assert_allclose(cdm.d_matrix, 1.0 - 1.0 / 1.5, rtol=1e-10)

    def test_update_damage_shear_mode(self):
        cdm = ContinuumDamage()
        failure = _make_failure(1.5, "shear_12")
        cdm.update_damage(failure)
        assert cdm.d_shear > 0.0

    def test_damage_never_decreases(self):
        """Damage variables should only increase (no healing)."""
        cdm = ContinuumDamage()
        # First apply a large failure
        failure_large = _make_failure(3.0, "matrix_tension")
        cdm.update_damage(failure_large)
        d_after_large = cdm.d_matrix

        # Then apply a smaller failure
        failure_small = _make_failure(1.2, "matrix_tension")
        cdm.update_damage(failure_small)
        d_after_small = cdm.d_matrix

        assert d_after_small >= d_after_large

    def test_damage_no_update_below_fi_one(self):
        """FI < 1 should not change damage variables."""
        cdm = ContinuumDamage()
        failure = _make_failure(0.8, "matrix_tension")
        cdm.update_damage(failure)
        assert_allclose(cdm.d_matrix, 0.0)

    def test_fiber_failure_is_catastrophic(self):
        """Fiber failure should set d_fiber to MAX_DAMAGE (0.99)."""
        cdm = ContinuumDamage()
        failure = _make_failure(1.1, "fiber_compression")
        cdm.update_damage(failure)
        assert_allclose(cdm.d_fiber, 0.99, rtol=1e-10)

    def test_degrade_returns_valid_material(self, x850_material):
        cdm = ContinuumDamage()
        failure = _make_failure(1.5, "matrix_tension")
        degraded = cdm.degrade(x850_material, failure)
        assert isinstance(degraded, OrthotropicMaterial)
        assert degraded.E1 > 0
        assert degraded.E2 > 0

    def test_degrade_e1_for_fiber_damage(self, x850_material):
        """After fiber failure, E1 should be degraded."""
        cdm = ContinuumDamage()
        failure = _make_failure(2.0, "fiber_tension")
        degraded = cdm.degrade(x850_material, failure)
        # d_fiber = 0.99 for catastrophic fiber failure
        expected_E1 = x850_material.E1 * (1.0 - 0.99)
        assert_allclose(degraded.E1, expected_E1, rtol=1e-10)

    def test_degrade_e2_for_matrix_damage(self, x850_material):
        """After matrix failure, E2 should be degraded."""
        cdm = ContinuumDamage()
        failure = _make_failure(2.0, "matrix_compression")
        degraded = cdm.degrade(x850_material, failure)
        d_m = 1.0 - 1.0 / 2.0  # = 0.5
        expected_E2 = x850_material.E2 * (1.0 - d_m)
        assert_allclose(degraded.E2, expected_E2, rtol=1e-10)

    def test_degrade_shear_for_shear_damage(self, x850_material):
        cdm = ContinuumDamage()
        failure = _make_failure(1.5, "shear_12")
        degraded = cdm.degrade(x850_material, failure)
        d_s = 1.0 - 1.0 / 1.5
        expected_G12 = x850_material.G12 * (1.0 - d_s)
        assert_allclose(degraded.G12, expected_G12, rtol=1e-10)

    def test_damage_vector_property(self):
        cdm = ContinuumDamage(d_fiber=0.1, d_matrix=0.2, d_shear=0.3)
        dv = cdm.damage_vector
        assert_allclose(dv, [0.1, 0.2, 0.3], rtol=1e-10)

    def test_reset(self):
        cdm = ContinuumDamage(d_fiber=0.5, d_matrix=0.3, d_shear=0.2)
        cdm.reset()
        assert_allclose(cdm.d_fiber, 0.0)
        assert_allclose(cdm.d_matrix, 0.0)
        assert_allclose(cdm.d_shear, 0.0)

    def test_degraded_name_updated(self, x850_material):
        cdm = ContinuumDamage()
        failure = _make_failure(2.0, "matrix_tension")
        degraded = cdm.degrade(x850_material, failure)
        assert "damaged" in degraded.name

    def test_through_thickness_damages_matrix_and_shear(self):
        cdm = ContinuumDamage()
        failure = _make_failure(2.0, "through_thickness_tension")
        cdm.update_damage(failure)
        assert cdm.d_matrix > 0.0
        assert cdm.d_shear > 0.0

    def test_damage_capped_at_099(self):
        cdm = ContinuumDamage()
        failure = _make_failure(1000.0, "matrix_tension")
        cdm.update_damage(failure)
        assert cdm.d_matrix <= 0.99
