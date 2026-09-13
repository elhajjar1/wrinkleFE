"""Tests for wrinklefe.core.transforms module."""

import numpy as np
import numpy.testing as npt
import pytest

from wrinklefe.core.transforms import (
    reduced_stiffness_matrix,
    rotate_stiffness_3d,
    rotation_matrix_3d,
    strain_transformation_3d,
    stress_transformation_3d,
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


# ======================================================================
# Orientation-sense and off-axis guards
#
# The assertions above this point are all invariant under the errors they
# would most plausibly be protecting against: orthogonality and det = 1
# hold equally for R and R-transpose; the Reuter check rebuilds
# ``R @ T_sigma @ R_inv``, which is the implementation line for line; and
# the 45-degree coupling check asserts only ``|Q16| > 1``, at the one angle
# where Q16 == Q26 makes a swap of the two invisible.
#
# A mutation audit confirmed it: transposing the 'y' rotation, and swapping
# the Q16/Q26 expressions, each left the entire 2167-test suite green. The
# tests below pin the *sense* and the *distinctness*, against closed forms
# that do not restate the implementation.
# ======================================================================


class TestRotationSense:
    """Pin the handedness, not just membership of SO(3)."""

    def test_y_rotation_sends_x_to_z(self):
        """Documented convention: R_y = [[c,0,-s],[0,1,0],[s,0,c]].

        So ``R_y(pi/2) @ x_hat == z_hat``.  The transpose sends it to
        ``-z_hat``, and every existing test in this module accepts both.
        """
        R = rotation_matrix_3d(np.pi / 2.0, axis="y")
        npt.assert_allclose(R @ np.array([1.0, 0.0, 0.0]),
                            [0.0, 0.0, 1.0], atol=1e-15)

    def test_z_rotation_sends_x_to_minus_y(self):
        """Documented convention: R_z = [[c,s,0],[-s,c,0],[0,0,1]]."""
        R = rotation_matrix_3d(np.pi / 2.0, axis="z")
        npt.assert_allclose(R @ np.array([1.0, 0.0, 0.0]),
                            [0.0, -1.0, 0.0], atol=1e-15)


class TestQBarOffAxis:
    """``Q16``/``Q26`` are distinct, odd in theta, and match a closed form."""

    E1, E2, NU12, G12 = 150_000.0, 10_000.0, 0.30, 5_000.0

    def _Q(self):
        return reduced_stiffness_matrix(self.E1, self.E2, self.NU12, self.G12)

    def test_q16_and_q26_are_not_interchangeable(self):
        """At 30 degrees the two coupling terms differ substantially.

        The existing coupling test uses 45 degrees, where ``Q16 == Q26``
        identically — so a swap of the two expressions is invisible there.
        """
        Qb = transform_reduced_stiffness(self._Q(), np.radians(30.0))
        assert abs(Qb[0, 2] - Qb[1, 2]) > 0.1 * abs(Qb[0, 2])

    def test_coupling_terms_are_odd_in_theta(self):
        """Q16 and Q26 flip sign with the ply angle; Q11/Q22/Q12/Q66 do not."""
        Q = self._Q()
        pos = transform_reduced_stiffness(Q, np.radians(30.0))
        neg = transform_reduced_stiffness(Q, np.radians(-30.0))
        npt.assert_allclose(pos[0, 2], -neg[0, 2], rtol=1e-12)
        npt.assert_allclose(pos[1, 2], -neg[1, 2], rtol=1e-12)
        for i, j in ((0, 0), (1, 1), (0, 1), (2, 2)):
            npt.assert_allclose(pos[i, j], neg[i, j], rtol=1e-12)

    @pytest.mark.parametrize("theta_deg", [0.0, 15.0, 30.0, 45.0, 60.0, 90.0])
    def test_off_axis_modulus_matches_closed_form(self, theta_deg):
        """``1/Ex(theta)`` from the inverted Q-bar must equal the textbook
        transformation equation — an independent statement about the whole
        rotated matrix, not a restatement of how it is built.

        .. math::
            1/E_x = c^4/E_1 + (1/G_{12} - 2\\nu_{12}/E_1) s^2 c^2 + s^4/E_2
        """
        theta = np.radians(theta_deg)
        c, s = np.cos(theta), np.sin(theta)
        expected = (
            c**4 / self.E1
            + (1.0 / self.G12 - 2.0 * self.NU12 / self.E1) * s**2 * c**2
            + s**4 / self.E2
        )
        a = np.linalg.inv(transform_reduced_stiffness(self._Q(), theta))
        npt.assert_allclose(a[0, 0], expected, rtol=1e-12)
