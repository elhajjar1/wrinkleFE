"""Tests for wrinklefe.core.laminate module."""

import numpy as np
import numpy.testing as npt
import pytest

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.core.laminate import Ply, Laminate, LoadState
from wrinklefe.core.transforms import reduced_stiffness_matrix, transform_reduced_stiffness


class TestPly:
    """Test Ply dataclass."""

    def test_creation(self, x850_material):
        ply = Ply(material=x850_material, angle=45.0, thickness=0.183)
        assert ply.angle == 45.0
        assert ply.thickness == 0.183

    def test_angle_rad_conversion(self, x850_material):
        ply = Ply(material=x850_material, angle=90.0, thickness=0.183)
        npt.assert_allclose(ply.angle_rad, np.pi / 2, atol=1e-14)

    def test_angle_rad_zero(self, x850_material):
        ply = Ply(material=x850_material, angle=0.0, thickness=0.183)
        npt.assert_allclose(ply.angle_rad, 0.0, atol=1e-14)

    def test_angle_rad_negative(self, x850_material):
        ply = Ply(material=x850_material, angle=-45.0, thickness=0.183)
        npt.assert_allclose(ply.angle_rad, -np.pi / 4, atol=1e-14)


class TestLaminateFromAngles:
    """Test Laminate.from_angles constructor."""

    def test_basic_creation(self, x850_material):
        lam = Laminate.from_angles([0, 90, 0], material=x850_material, ply_thickness=0.183)
        assert lam.n_plies == 3

    def test_ply_count(self, x850_material):
        angles = [0, 45, -45, 90, 90, -45, 45, 0]
        lam = Laminate.from_angles(angles, material=x850_material, ply_thickness=0.183)
        assert lam.n_plies == 8

    def test_empty_raises(self, x850_material):
        with pytest.raises(ValueError, match="at least one ply"):
            Laminate.from_angles([], material=x850_material, ply_thickness=0.183)


class TestLaminateSymmetric:
    """Test Laminate.symmetric constructor."""

    def test_symmetric_ply_count(self, x850_material):
        """[0, 45, 90] half -> [0, 45, 90, 90, 45, 0] = 6 plies."""
        lam = Laminate.symmetric([0, 45, 90], material=x850_material, ply_thickness=0.183)
        assert lam.n_plies == 6

    def test_symmetric_angles(self, x850_material):
        lam = Laminate.symmetric([0, 45, 90], material=x850_material, ply_thickness=0.183)
        angles = [p.angle for p in lam.plies]
        assert angles == [0, 45, 90, 90, 45, 0]


class TestLaminateGeometry:
    """Test laminate geometric properties."""

    def test_z_coords_shape(self, quasi_iso_laminate):
        zc = quasi_iso_laminate.z_coords()
        assert zc.shape == (quasi_iso_laminate.n_plies + 1,)

    def test_z_coords_range(self, quasi_iso_laminate):
        """Z-coordinates should range from -h/2 to +h/2."""
        zc = quasi_iso_laminate.z_coords()
        h = quasi_iso_laminate.total_thickness
        npt.assert_allclose(zc[0], -h / 2, atol=1e-12)
        npt.assert_allclose(zc[-1], h / 2, atol=1e-12)

    def test_total_thickness(self, quasi_iso_laminate):
        """Total thickness should be n_plies * ply_thickness."""
        expected = 16 * 0.183
        npt.assert_allclose(quasi_iso_laminate.total_thickness, expected, atol=1e-12)

    def test_z_coords_monotonic(self, quasi_iso_laminate):
        """Z-coordinates should be strictly increasing."""
        zc = quasi_iso_laminate.z_coords()
        assert np.all(np.diff(zc) > 0)


class TestABDMatrix:
    """Test ABD stiffness matrix computation."""

    def test_abd_shape_is_6x6(self, quasi_iso_laminate):
        abd = quasi_iso_laminate.abd_matrix()
        assert abd.shape == (6, 6)

    def test_A_symmetric(self, quasi_iso_laminate):
        A = quasi_iso_laminate.A
        npt.assert_allclose(A, A.T, atol=1e-6)

    def test_D_symmetric(self, quasi_iso_laminate):
        D = quasi_iso_laminate.D
        npt.assert_allclose(D, D.T, atol=1e-6)

    def test_B_zero_for_symmetric_laminate(self, quasi_iso_laminate):
        """B matrix should be zero for a symmetric laminate."""
        B = quasi_iso_laminate.B
        npt.assert_allclose(B, np.zeros((3, 3)), atol=1e-6)

    def test_abd_inverse_exists(self, quasi_iso_laminate):
        """ABD matrix should be invertible."""
        abd = quasi_iso_laminate.abd_matrix()
        abd_inv = quasi_iso_laminate.abd_inverse()
        product = abd @ abd_inv
        npt.assert_allclose(product, np.eye(6), atol=1e-8)


class TestABDValidation:
    """Hand-calculation ABD validation for a [0/90]s laminate."""

    @pytest.fixture
    def simple_material(self):
        """Simple material for hand calculation: E1=140000, E2=10000, G12=5000, nu12=0.3."""
        return OrthotropicMaterial(
            E1=140_000.0, E2=10_000.0, E3=10_000.0,
            G12=5_000.0, G13=5_000.0, G23=3_500.0,
            nu12=0.3, nu13=0.3, nu23=0.4,
            name="simple_test",
        )

    @pytest.fixture
    def laminate_0_90_s(self, simple_material):
        """[0/90]s = [0, 90, 90, 0] laminate with t=0.125 mm."""
        return Laminate.from_angles([0, 90, 90, 0], material=simple_material, ply_thickness=0.125)

    def test_A11_hand_calculation(self, laminate_0_90_s, simple_material):
        """Verify A11 against hand calculation.

        For [0/90]s with 4 plies of thickness t:
        Q_0 (0-degree ply): Q11 = E1/(1-nu12*nu21), Q22 = E2/(1-nu12*nu21)
        Q_90 (90-degree ply): Q_bar_11 = Q22, Q_bar_22 = Q11 (from transformation)

        A11 = sum(Q_bar_11_k * t_k) = 2 * Q11 * t + 2 * Q22 * t
        (two 0-deg plies contribute Q11, two 90-deg plies contribute Q22)
        """
        mat = simple_material
        t = 0.125
        nu21 = mat.nu12 * mat.E2 / mat.E1
        denom = 1.0 - mat.nu12 * nu21

        Q11 = mat.E1 / denom
        Q22 = mat.E2 / denom

        # A11 = 2*Q11*t + 2*Q22*t for [0/90]s (two 0-deg, two 90-deg plies)
        expected_A11 = 2.0 * Q11 * t + 2.0 * Q22 * t
        npt.assert_allclose(laminate_0_90_s.A[0, 0], expected_A11, rtol=1e-3)

    def test_A22_hand_calculation(self, laminate_0_90_s, simple_material):
        """A22 should equal A11 for [0/90]s (swap of Q11/Q22 roles)."""
        npt.assert_allclose(
            laminate_0_90_s.A[1, 1],
            laminate_0_90_s.A[0, 0],
            rtol=1e-10,
        )

    def test_D11_hand_calculation(self, laminate_0_90_s, simple_material):
        """Verify D11 against hand calculation for [0/90]s.

        D11 = sum(Q_bar_11_k * (z_top^3 - z_bot^3) / 3)
        For [0/90]s with t=0.125, h=0.5:
        z coords: [-0.25, -0.125, 0.0, 0.125, 0.25]
        Ply 0 (0-deg):  z_bot=-0.25,  z_top=-0.125 -> contributes Q11
        Ply 1 (90-deg): z_bot=-0.125, z_top=0.0    -> contributes Q22
        Ply 2 (90-deg): z_bot=0.0,    z_top=0.125   -> contributes Q22
        Ply 3 (0-deg):  z_bot=0.125,  z_top=0.25    -> contributes Q11
        """
        mat = simple_material
        t = 0.125
        nu21 = mat.nu12 * mat.E2 / mat.E1
        denom = 1.0 - mat.nu12 * nu21

        Q11 = mat.E1 / denom
        Q22 = mat.E2 / denom

        z = np.array([-0.25, -0.125, 0.0, 0.125, 0.25])
        # Ply 0 (0-deg): Q_bar_11 = Q11
        D11_0 = Q11 * (z[1]**3 - z[0]**3) / 3.0
        # Ply 1 (90-deg): Q_bar_11 = Q22
        D11_1 = Q22 * (z[2]**3 - z[1]**3) / 3.0
        # Ply 2 (90-deg): Q_bar_11 = Q22
        D11_2 = Q22 * (z[3]**3 - z[2]**3) / 3.0
        # Ply 3 (0-deg): Q_bar_11 = Q11
        D11_3 = Q11 * (z[4]**3 - z[3]**3) / 3.0

        expected_D11 = D11_0 + D11_1 + D11_2 + D11_3
        npt.assert_allclose(laminate_0_90_s.D[0, 0], expected_D11, rtol=1e-3)

    def test_D12_hand_calculation(self, laminate_0_90_s, simple_material):
        """Verify D12 against hand calculation.

        D12 = sum(Q_bar_12_k * (z_top^3 - z_bot^3) / 3)
        Q_bar_12 for 0-deg and 90-deg plies is the same: Q12.
        """
        mat = simple_material
        nu21 = mat.nu12 * mat.E2 / mat.E1
        denom = 1.0 - mat.nu12 * nu21

        Q12 = mat.nu12 * mat.E2 / denom

        z = np.array([-0.25, -0.125, 0.0, 0.125, 0.25])
        expected_D12 = 0.0
        for k in range(4):
            expected_D12 += Q12 * (z[k+1]**3 - z[k]**3) / 3.0

        npt.assert_allclose(laminate_0_90_s.D[0, 1], expected_D12, rtol=1e-3)


class TestSymmetryAndBalance:
    """Test is_symmetric and is_balanced properties."""

    def test_is_symmetric_true(self, quasi_iso_laminate):
        assert quasi_iso_laminate.is_symmetric is True

    def test_is_symmetric_false(self, x850_material):
        lam = Laminate.from_angles([0, 45, 90], material=x850_material, ply_thickness=0.183)
        assert lam.is_symmetric is False

    def test_is_balanced_quasi_iso(self, quasi_iso_laminate):
        """[0/45/-45/90]2s has matched +45/-45 pairs, so is balanced."""
        assert quasi_iso_laminate.is_balanced is True

    def test_is_balanced_false(self, x850_material):
        """[0/45/90/90/45/0] has no -45 to pair with +45, so not balanced."""
        lam = Laminate.from_angles(
            [0, 45, 90, 90, 45, 0],
            material=x850_material,
            ply_thickness=0.183,
        )
        assert lam.is_balanced is False


class TestStressStrainAnalysis:
    """Test midplane strains and ply stress recovery."""

    def test_midplane_strains_shape(self, quasi_iso_laminate):
        load = LoadState(Nx=-1000.0)
        eps = quasi_iso_laminate.midplane_strains(load)
        assert eps.shape == (6,)

    def test_ply_stresses_local_shape(self, quasi_iso_laminate):
        load = LoadState(Nx=-1000.0)
        sigma_local = quasi_iso_laminate.ply_stresses_local(load, ply_idx=0)
        assert sigma_local.shape == (3,)

    def test_ply_stresses_global_shape(self, quasi_iso_laminate):
        load = LoadState(Nx=-1000.0)
        sigma_global = quasi_iso_laminate.ply_stresses_global(load, ply_idx=0)
        assert sigma_global.shape == (3,)

    def test_zero_load_zero_strain(self, quasi_iso_laminate):
        load = LoadState()  # all zeros
        eps = quasi_iso_laminate.midplane_strains(load)
        npt.assert_allclose(eps, np.zeros(6), atol=1e-14)


class TestEngineeringConstants:
    """Test effective engineering constants."""

    def test_Ex_positive(self, quasi_iso_laminate):
        assert quasi_iso_laminate.Ex > 0

    def test_Ey_positive(self, quasi_iso_laminate):
        assert quasi_iso_laminate.Ey > 0

    def test_Gxy_positive(self, quasi_iso_laminate):
        assert quasi_iso_laminate.Gxy > 0

    def test_Ex_reasonable_range(self, quasi_iso_laminate):
        """For quasi-iso, Ex should be between E2 and E1 of ply material."""
        mat = quasi_iso_laminate.plies[0].material
        assert mat.E2 < quasi_iso_laminate.Ex < mat.E1

    def test_quasi_iso_Ex_equals_Ey(self, quasi_iso_laminate):
        """For a quasi-isotropic laminate, Ex should approximately equal Ey."""
        npt.assert_allclose(
            quasi_iso_laminate.Ex,
            quasi_iso_laminate.Ey,
            rtol=0.05,  # within 5%
        )


class TestLoadState:
    """Test LoadState dataclass."""

    def test_to_vector(self):
        load = LoadState(Nx=100.0, My=50.0)
        vec = load.to_vector()
        assert vec.shape == (6,)
        npt.assert_allclose(vec, [100.0, 0.0, 0.0, 0.0, 50.0, 0.0])

    def test_from_vector_roundtrip(self):
        original = LoadState(Nx=100.0, Ny=200.0, Nxy=50.0, Mx=10.0, My=20.0, Mxy=5.0)
        vec = original.to_vector()
        restored = LoadState.from_vector(vec)
        npt.assert_allclose(restored.to_vector(), vec, atol=1e-14)

    def test_from_vector_wrong_size_raises(self):
        with pytest.raises(ValueError, match="6-component"):
            LoadState.from_vector(np.zeros(5))

    def test_from_vector_with_kwargs(self):
        vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        load = LoadState.from_vector(vec, delta_T=-50.0)
        assert load.Nx == 1.0
        assert load.delta_T == -50.0

    def test_default_zeros(self):
        load = LoadState()
        npt.assert_allclose(load.to_vector(), np.zeros(6))
        assert load.delta_T == 0.0
        assert load.delta_C == 0.0
