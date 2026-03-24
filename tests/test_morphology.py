"""Tests for wrinklefe.core.morphology module."""

import numpy as np
import numpy.testing as npt
import pytest

from wrinklefe.core.wrinkle import GaussianSinusoidal
from wrinklefe.core.morphology import (
    MORPHOLOGY_PHASES,
    WrinklePlacement,
    WrinkleConfiguration,
)


class TestPhaseFromOffset:
    """Test phase_from_offset static method."""

    def test_quarter_wavelength_gives_pi_over_2(self):
        """delta_x = lambda/4 -> phi = 2*pi*(lambda/4)/lambda = pi/2."""
        phi = WrinkleConfiguration.phase_from_offset(delta_x=4.0, wavelength=16.0)
        npt.assert_allclose(phi, np.pi / 2, atol=1e-14)

    def test_half_wavelength_gives_pi(self):
        phi = WrinkleConfiguration.phase_from_offset(delta_x=8.0, wavelength=16.0)
        npt.assert_allclose(phi, np.pi, atol=1e-14)

    def test_zero_offset_gives_zero(self):
        phi = WrinkleConfiguration.phase_from_offset(delta_x=0.0, wavelength=16.0)
        npt.assert_allclose(phi, 0.0, atol=1e-14)

    def test_negative_offset(self):
        phi = WrinkleConfiguration.phase_from_offset(delta_x=-4.0, wavelength=16.0)
        npt.assert_allclose(phi, -np.pi / 2, atol=1e-14)

    def test_zero_wavelength_raises(self):
        with pytest.raises(ValueError, match="Wavelength must be positive"):
            WrinkleConfiguration.phase_from_offset(4.0, wavelength=0.0)


class TestMorphologyFactorAnalytical:
    """Test morphology_factor_analytical static method."""

    def test_stack_compression_is_one(self):
        """Stack (phi=0): M_f = exp(0) = 1.0."""
        mf = WrinkleConfiguration.morphology_factor_analytical(0.0, "compression")
        npt.assert_allclose(mf, 1.0, atol=1e-10)

    def test_convex_compression_less_than_one(self):
        """Convex (phi=pi/2): M_f < 1 for compression."""
        mf = WrinkleConfiguration.morphology_factor_analytical(np.pi / 2, "compression")
        assert mf < 1.0
        # Expected: exp(-0.288 * sin(pi/2)) = exp(-0.288) ~ 0.750
        npt.assert_allclose(mf, np.exp(-0.288), rtol=1e-6)

    def test_concave_compression_greater_than_one(self):
        """Concave (phi=-pi/2): M_f > 1 for compression."""
        mf = WrinkleConfiguration.morphology_factor_analytical(-np.pi / 2, "compression")
        assert mf > 1.0
        # Expected: exp(-0.288 * sin(-pi/2)) = exp(0.288) ~ 1.334
        npt.assert_allclose(mf, np.exp(0.288), rtol=1e-6)

    def test_concave_is_approximately_1_33(self):
        mf = WrinkleConfiguration.morphology_factor_analytical(-np.pi / 2, "compression")
        npt.assert_allclose(mf, 1.334, atol=0.01)

    def test_convex_is_approximately_0_75(self):
        mf = WrinkleConfiguration.morphology_factor_analytical(np.pi / 2, "compression")
        npt.assert_allclose(mf, 0.750, atol=0.01)

    def test_tension_stack_is_one(self):
        """Stack under tension: M_f = exp(0 - 0) = 1.0."""
        mf = WrinkleConfiguration.morphology_factor_analytical(0.0, "tension")
        npt.assert_allclose(mf, 1.0, atol=1e-10)

    def test_invalid_loading_raises(self):
        with pytest.raises(ValueError, match="Unsupported loading mode"):
            WrinkleConfiguration.morphology_factor_analytical(0.0, "bending")


class TestCurvatureCorrelation:
    """Test curvature_correlation static method."""

    def test_stack_is_one(self):
        c = WrinkleConfiguration.curvature_correlation(0.0)
        npt.assert_allclose(c, 1.0, atol=1e-14)

    def test_convex_is_zero(self):
        c = WrinkleConfiguration.curvature_correlation(np.pi / 2)
        npt.assert_allclose(c, 0.0, atol=1e-14)

    def test_concave_is_zero(self):
        c = WrinkleConfiguration.curvature_correlation(-np.pi / 2)
        npt.assert_allclose(c, 0.0, atol=1e-14)

    def test_anti_stack_is_negative_one(self):
        c = WrinkleConfiguration.curvature_correlation(np.pi)
        npt.assert_allclose(c, -1.0, atol=1e-14)


class TestDualWrinkleConstructor:
    """Test dual_wrinkle class method."""

    def test_creates_two_wrinkles(self, gaussian_wrinkle):
        config = WrinkleConfiguration.dual_wrinkle(
            gaussian_wrinkle, interface1=7, interface2=8, phase=0.0
        )
        assert config.n_wrinkles() == 2

    def test_interfaces_sorted(self, gaussian_wrinkle):
        config = WrinkleConfiguration.dual_wrinkle(
            gaussian_wrinkle, interface1=10, interface2=5, phase=0.0
        )
        interfaces = [w.ply_interface for w in config.wrinkles]
        assert interfaces == sorted(interfaces)

    def test_phase_assigned(self, gaussian_wrinkle):
        config = WrinkleConfiguration.dual_wrinkle(
            gaussian_wrinkle, interface1=7, interface2=8, phase=np.pi / 2
        )
        phases = [w.phase_offset for w in config.wrinkles]
        # First wrinkle has phase 0, second has the specified phase
        assert 0.0 in phases
        npt.assert_allclose(max(phases), np.pi / 2, atol=1e-14)


class TestFromMorphologyName:
    """Test from_morphology_name class method."""

    def test_convex_has_correct_phase(self, gaussian_wrinkle):
        config = WrinkleConfiguration.from_morphology_name(
            "convex", gaussian_wrinkle, interface1=7, interface2=8
        )
        phases = config.pairwise_phases()
        assert len(phases) == 1
        npt.assert_allclose(phases[0], np.pi / 2, atol=1e-14)

    def test_concave_has_correct_phase(self, gaussian_wrinkle):
        config = WrinkleConfiguration.from_morphology_name(
            "concave", gaussian_wrinkle, interface1=7, interface2=8
        )
        phases = config.pairwise_phases()
        assert len(phases) == 1
        npt.assert_allclose(phases[0], -np.pi / 2, atol=1e-14)

    def test_stack_has_zero_phase(self, gaussian_wrinkle):
        config = WrinkleConfiguration.from_morphology_name(
            "stack", gaussian_wrinkle, interface1=7, interface2=8
        )
        phases = config.pairwise_phases()
        npt.assert_allclose(phases[0], 0.0, atol=1e-14)

    def test_case_insensitive(self, gaussian_wrinkle):
        config = WrinkleConfiguration.from_morphology_name(
            "CONVEX", gaussian_wrinkle, interface1=7, interface2=8
        )
        assert config.n_wrinkles() == 2

    def test_unknown_name_raises(self, gaussian_wrinkle):
        with pytest.raises(ValueError, match="Unknown morphology"):
            WrinkleConfiguration.from_morphology_name(
                "zigzag", gaussian_wrinkle, interface1=7, interface2=8
            )


class TestEffectiveAngle:
    """Test effective angle computation."""

    def test_effective_angle_is_max_angle_times_mf(self, gaussian_wrinkle):
        config = WrinkleConfiguration.from_morphology_name(
            "convex", gaussian_wrinkle, interface1=7, interface2=8
        )
        theta_max = config.max_angle()
        mf = config.aggregate_morphology_factor("compression")
        expected = theta_max * mf
        actual = config.effective_angle("compression")
        npt.assert_allclose(actual, expected, rtol=1e-12)

    def test_convex_effective_angle_less_than_max(self, gaussian_wrinkle):
        config = WrinkleConfiguration.from_morphology_name(
            "convex", gaussian_wrinkle, interface1=7, interface2=8
        )
        assert config.effective_angle("compression") < config.max_angle()

    def test_concave_effective_angle_greater_than_max(self, gaussian_wrinkle):
        config = WrinkleConfiguration.from_morphology_name(
            "concave", gaussian_wrinkle, interface1=7, interface2=8
        )
        assert config.effective_angle("compression") > config.max_angle()


class TestAggregateMorphologyFactor:
    """Test aggregate morphology factor computation."""

    def test_single_wrinkle_returns_one(self, gaussian_wrinkle):
        """A single wrinkle has no pairwise interaction -> M_f_agg = 1.0."""
        w = WrinklePlacement(profile=gaussian_wrinkle, ply_interface=7, phase_offset=0.0)
        config = WrinkleConfiguration([w])
        mf = config.aggregate_morphology_factor("compression")
        npt.assert_allclose(mf, 1.0, atol=1e-14)

    def test_dual_wrinkle_stack_returns_one(self, gaussian_wrinkle):
        config = WrinkleConfiguration.dual_wrinkle(
            gaussian_wrinkle, interface1=7, interface2=8, phase=0.0
        )
        mf = config.aggregate_morphology_factor("compression")
        npt.assert_allclose(mf, 1.0, atol=1e-10)

    def test_dual_wrinkle_convex(self, gaussian_wrinkle):
        config = WrinkleConfiguration.from_morphology_name(
            "convex", gaussian_wrinkle, interface1=7, interface2=8
        )
        mf = config.aggregate_morphology_factor("compression")
        assert 0.7 < mf < 0.8  # approximately 0.75


class TestMorphologyPhases:
    """Test predefined morphology phase constants."""

    def test_stack_phase(self):
        npt.assert_allclose(MORPHOLOGY_PHASES["stack"], 0.0)

    def test_convex_phase(self):
        npt.assert_allclose(MORPHOLOGY_PHASES["convex"], np.pi / 2)

    def test_concave_phase(self):
        npt.assert_allclose(MORPHOLOGY_PHASES["concave"], -np.pi / 2)

    def test_three_morphologies(self):
        assert len(MORPHOLOGY_PHASES) == 3


class TestWrinkleConfigurationValidation:
    """Test validation in WrinkleConfiguration."""

    def test_empty_wrinkles_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            WrinkleConfiguration([])

    def test_repr(self, dual_wrinkle_config):
        r = repr(dual_wrinkle_config)
        assert "WrinkleConfiguration" in r
        assert "n_wrinkles=2" in r
