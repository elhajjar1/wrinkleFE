"""Tests for wrinklefe.core.transforms module."""

import numpy as np
import numpy.testing as npt
import pytest

from wrinklefe.core.transforms import (
    rotation_matrix_3d,
    stress_transformation_3d,
    strain_transformation_3d,
    rotate_stiffness_3d,
    reduced_stiffness_matrix,
    transform_reduced_stiffness,
)


class TestRotationMatrix3D:
    """Test 3x3 rotation matrices."""

    def test_orthogonal_z(self):
        """R @ R.T should be identity for z-axis rotation."""
        R = rotation_matrix_3d(np.pi / 4, axis='z')
        npt.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)

    def test_orthogonal_y(self):
        """R @ R.T should be identity for y-axis rotation."""
        R = rotation_matrix_3d(np.pi / 3, axis='y')
        npt.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)

    def test_identity_for_zero_angle_z(self):
        R = rotation_matrix_3d(0.0, axis='z')
        npt.assert_allclose(R, np.eye(3), atol=1e-14)

    def test_identity_for_zero_angle_y(self):
        R = rotation_matrix_3d(0.0, axis='y')
        npt.assert_allclose(R, np.eye(3), atol=1e-14)

    def test_determinant_is_one(self):
        for angle in [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, np.pi]:
            for axis in ['z', 'y']:
                R = rotation_matrix_3d(angle, axis=axis)
                npt.assert_allclose(np.linalg.det(R), 1.0, atol=1e-14)

    def test_invalid_axis_raises(self):
        with pytest.raises(ValueError, match="Unsupported axis"):
            rotation_matrix_3d(0.0, axis='x')


class TestStressTransformation3D:
    """Test 6x6 stress transformation matrices."""

    def test_identity_at_zero_angle_z(self):
        T = stress_transformation_3d(0.0, axis='z')
        npt.assert_allclose(T, np.eye(6), atol=1e-14)

    def test_identity_at_zero_angle_y(self):
        T = stress_transformation_3d(0.0, axis='y')
        npt.assert_allclose(T, np.eye(6), atol=1e-14)

    def test_90deg_z_swaps_sigma11_sigma22(self):
        """At 90 degrees about z, sigma_11 and sigma_22 should swap."""
        T = stress_transformation_3d(np.pi / 2, axis='z')
        # Apply to unit stress in 11-direction: [1, 0, 0, 0, 0, 0]
        sigma = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        sigma_rot = T @ sigma
        # After 90-deg rotation: sigma_11' = sin^2(90)*sigma_11 = 0 (oops)
        # Actually: T_11 = cos^2, T_12 = sin^2. At 90: c=0, s=1
        # sigma_11' = c^2*sig11 + s^2*sig22 + 2sc*tau12 = 0 + 0 + 0 = 0
        # sigma_22' = s^2*sig11 + c^2*sig22 - 2sc*tau12 = 1 + 0 + 0 = 1
        expected = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        npt.assert_allclose(sigma_rot, expected, atol=1e-14)

    def test_shape_is_6x6(self):
        T = stress_transformation_3d(np.pi / 6, axis='z')
        assert T.shape == (6, 6)

    def test_invalid_axis_raises(self):
        with pytest.raises(ValueError, match="Unsupported axis"):
            stress_transformation_3d(0.0, axis='x')


class TestStrainTransformation3D:
    """Test strain transformation consistency with stress transformation."""

    def test_identity_at_zero_angle(self):
        T_eps = strain_transformation_3d(0.0, axis='z')
        npt.assert_allclose(T_eps, np.eye(6), atol=1e-14)

    def test_reuter_relationship(self):
        """T_eps = R @ T_sigma @ R_inv where R = diag(1,1,1,2,2,2)."""
        angle = np.pi / 5
        T_sigma = stress_transformation_3d(angle, axis='z')
        T_eps = strain_transformation_3d(angle, axis='z')
        R = np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        R_inv = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
        expected = R @ T_sigma @ R_inv
        npt.assert_allclose(T_eps, expected, atol=1e-14)

    def test_reuter_relationship_y_axis(self):
        angle = np.pi / 7
        T_sigma = stress_transformation_3d(angle, axis='y')
        T_eps = strain_transformation_3d(angle, axis='y')
        R = np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        R_inv = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
        expected = R @ T_sigma @ R_inv
        npt.assert_allclose(T_eps, expected, atol=1e-14)


class TestRotateStiffness3D:
    """Test 6x6 stiffness rotation."""

    def test_zero_angle_returns_same_matrix(self, x850_material):
        C = x850_material.stiffness_matrix
        C_rot = rotate_stiffness_3d(C, 0.0, axis='z')
        npt.assert_allclose(C_rot, C, atol=1e-8)

    def test_zero_angle_y_returns_same_matrix(self, x850_material):
        C = x850_material.stiffness_matrix
        C_rot = rotate_stiffness_3d(C, 0.0, axis='y')
        npt.assert_allclose(C_rot, C, atol=1e-8)

    def test_rotated_matrix_is_symmetric(self, x850_material):
        C = x850_material.stiffness_matrix
        C_rot = rotate_stiffness_3d(C, np.pi / 4, axis='z')
        npt.assert_allclose(C_rot, C_rot.T, atol=1e-8)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="6x6"):
            rotate_stiffness_3d(np.eye(3), 0.0)


class TestReducedStiffnessMatrix:
    """Test 3x3 reduced stiffness computation from individual properties."""

    def test_known_values(self):
        """Test with simple known material properties."""
        E1, E2, nu12, G12 = 140_000.0, 10_000.0, 0.3, 5_000.0
        Q = reduced_stiffness_matrix(E1, E2, nu12, G12)

        nu21 = nu12 * E2 / E1
        denom = 1.0 - nu12 * nu21

        npt.assert_allclose(Q[0, 0], E1 / denom, rtol=1e-12)
        npt.assert_allclose(Q[1, 1], E2 / denom, rtol=1e-12)
        npt.assert_allclose(Q[0, 1], nu12 * E2 / denom, rtol=1e-12)
        npt.assert_allclose(Q[2, 2], G12, rtol=1e-12)

    def test_shape_is_3x3(self):
        Q = reduced_stiffness_matrix(161_000.0, 11_380.0, 0.32, 5_170.0)
        assert Q.shape == (3, 3)

    def test_symmetric(self):
        Q = reduced_stiffness_matrix(161_000.0, 11_380.0, 0.32, 5_170.0)
        npt.assert_allclose(Q, Q.T, atol=1e-12)


class TestTransformReducedStiffness:
    """Test Q-bar transformed reduced stiffness."""

    def test_zero_angle_returns_Q_unchanged(self):
        Q = reduced_stiffness_matrix(161_000.0, 11_380.0, 0.32, 5_170.0)
        Qbar = transform_reduced_stiffness(Q, 0.0)
        npt.assert_allclose(Qbar, Q, atol=1e-8)

    def test_90deg_swaps_Q11_Q22(self):
        """At 90 degrees, Q_bar_11 should become Q22 and Q_bar_22 should become Q11."""
        Q = reduced_stiffness_matrix(161_000.0, 11_380.0, 0.32, 5_170.0)
        Qbar = transform_reduced_stiffness(Q, np.pi / 2)
        npt.assert_allclose(Qbar[0, 0], Q[1, 1], atol=1e-6)
        npt.assert_allclose(Qbar[1, 1], Q[0, 0], atol=1e-6)

    def test_90deg_Q66_unchanged(self):
        """Q66 (shear stiffness) at 90 degrees should equal Q66 at 0 degrees."""
        Q = reduced_stiffness_matrix(161_000.0, 11_380.0, 0.32, 5_170.0)
        Qbar = transform_reduced_stiffness(Q, np.pi / 2)
        npt.assert_allclose(Qbar[2, 2], Q[2, 2], atol=1e-6)

    def test_Qbar_is_symmetric(self):
        Q = reduced_stiffness_matrix(161_000.0, 11_380.0, 0.32, 5_170.0)
        Qbar = transform_reduced_stiffness(Q, np.pi / 4)
        npt.assert_allclose(Qbar, Qbar.T, atol=1e-12)

    def test_coupling_terms_at_45deg(self):
        """At 45 degrees, Q16 and Q26 should be non-zero."""
        Q = reduced_stiffness_matrix(161_000.0, 11_380.0, 0.32, 5_170.0)
        Qbar = transform_reduced_stiffness(Q, np.pi / 4)
        assert abs(Qbar[0, 2]) > 1.0  # Q16 should be significant
        assert abs(Qbar[1, 2]) > 1.0  # Q26 should be significant

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="3x3"):
            transform_reduced_stiffness(np.eye(6), 0.0)
