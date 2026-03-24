"""Tests for the 8-node hexahedral element with incompatible modes (C3D8I).

Covers inheritance, incompatible mode functions, stiffness matrix properties,
and differences from the standard Hex8 element.
"""

import warnings

import numpy as np
import pytest

from wrinklefe.elements.hex8 import Hex8Element
from wrinklefe.elements.hex8i import Hex8IElement
from wrinklefe.core.material import OrthotropicMaterial


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def x850_material():
    """Default CYCOM X850/T800 material."""
    return OrthotropicMaterial()


@pytest.fixture
def unit_cube_nodes():
    """Node coordinates for a unit cube [0,1]^3."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=float)


@pytest.fixture
def hex8i_element(unit_cube_nodes, x850_material):
    """Hex8I element on a unit cube with default material."""
    return Hex8IElement(
        node_coords=unit_cube_nodes,
        material=x850_material,
        ply_angle=0.0,
    )


@pytest.fixture
def hex8_element(unit_cube_nodes, x850_material):
    """Standard Hex8 element for comparison."""
    return Hex8Element(
        node_coords=unit_cube_nodes,
        material=x850_material,
        ply_angle=0.0,
    )


# ======================================================================
# Inheritance and construction
# ======================================================================

class TestInheritance:
    """Test that Hex8IElement inherits from Hex8Element."""

    def test_is_subclass(self):
        """Hex8IElement is a subclass of Hex8Element."""
        assert issubclass(Hex8IElement, Hex8Element)

    def test_isinstance(self, hex8i_element):
        """Instance check passes for both Hex8IElement and Hex8Element."""
        assert isinstance(hex8i_element, Hex8IElement)
        assert isinstance(hex8i_element, Hex8Element)

    def test_has_shape_functions(self, hex8i_element):
        """Inherited shape_functions method is available."""
        N = hex8i_element.shape_functions(0.0, 0.0, 0.0)
        assert N.shape == (8,)

    def test_has_volume(self, hex8i_element):
        """Inherited volume property works correctly."""
        assert np.isclose(hex8i_element.volume, 1.0, rtol=1e-12)

    def test_center_jacobian_cached(self, hex8i_element):
        """Center Jacobian is pre-computed at construction."""
        assert hex8i_element._J0 is not None
        assert hex8i_element._detJ0 > 0
        assert hex8i_element._J0_inv is not None


# ======================================================================
# Incompatible modes
# ======================================================================

class TestIncompatibleModes:
    """Tests for incompatible mode functions and their derivatives."""

    def test_modes_at_center(self):
        """At the element center (0,0,0), all modes are 1.

        P1 = 1 - 0^2 = 1, P2 = 1 - 0^2 = 1, P3 = 1 - 0^2 = 1.
        """
        P = Hex8IElement.incompatible_modes(0.0, 0.0, 0.0)
        assert P.shape == (3,)
        np.testing.assert_allclose(P, [1.0, 1.0, 1.0])

    def test_modes_at_corner(self):
        """At corner (-1,-1,-1), all modes are 0.

        P1 = 1 - (-1)^2 = 0, etc.
        """
        P = Hex8IElement.incompatible_modes(-1.0, -1.0, -1.0)
        np.testing.assert_allclose(P, [0.0, 0.0, 0.0])

    def test_modes_at_face_center(self):
        """At the center of the xi=1 face (1, 0, 0):

        P1 = 1 - 1 = 0, P2 = 1 - 0 = 1, P3 = 1 - 0 = 1.
        """
        P = Hex8IElement.incompatible_modes(1.0, 0.0, 0.0)
        np.testing.assert_allclose(P, [0.0, 1.0, 1.0])

    def test_modes_shape(self):
        """Incompatible modes return shape (3,)."""
        P = Hex8IElement.incompatible_modes(0.3, -0.5, 0.7)
        assert P.shape == (3,)

    def test_modes_nonnegative_inside(self):
        """All mode values are between 0 and 1 inside the element."""
        P = Hex8IElement.incompatible_modes(0.3, -0.5, 0.7)
        assert np.all(P >= 0.0)
        assert np.all(P <= 1.0)

    def test_mode_derivatives_shape(self):
        """Incompatible mode derivatives return shape (3, 3)."""
        dP = Hex8IElement.incompatible_mode_derivatives(0.3, -0.5, 0.7)
        assert dP.shape == (3, 3)

    def test_mode_derivatives_at_center(self):
        """At center (0,0,0), derivatives are zero (modes have local extrema)."""
        dP = Hex8IElement.incompatible_mode_derivatives(0.0, 0.0, 0.0)
        np.testing.assert_allclose(dP, np.zeros((3, 3)))

    def test_mode_derivatives_diagonal(self):
        """Derivative matrix is diagonal: dP_k/dxi_i = 0 for k != i."""
        dP = Hex8IElement.incompatible_mode_derivatives(0.3, -0.5, 0.7)
        # Check off-diagonal entries are zero
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert np.isclose(dP[i, j], 0.0)

    def test_mode_derivatives_values(self):
        """Explicit check: dP1/dxi = -2*xi, dP2/deta = -2*eta, dP3/dzeta = -2*zeta."""
        xi, eta, zeta = 0.3, -0.5, 0.7
        dP = Hex8IElement.incompatible_mode_derivatives(xi, eta, zeta)
        assert np.isclose(dP[0, 0], -2.0 * xi)
        assert np.isclose(dP[1, 1], -2.0 * eta)
        assert np.isclose(dP[2, 2], -2.0 * zeta)

    def test_mode_derivatives_numerical(self):
        """Derivatives are consistent with finite differences of the mode functions."""
        xi0, eta0, zeta0 = 0.3, -0.2, 0.5
        h = 1e-7
        dP_analytical = Hex8IElement.incompatible_mode_derivatives(xi0, eta0, zeta0)

        # dP/dxi
        Pp = Hex8IElement.incompatible_modes(xi0 + h, eta0, zeta0)
        Pm = Hex8IElement.incompatible_modes(xi0 - h, eta0, zeta0)
        dP_xi_num = (Pp - Pm) / (2 * h)
        np.testing.assert_allclose(dP_analytical[0, :], dP_xi_num, atol=1e-6)


# ======================================================================
# G-matrix
# ======================================================================

class TestGMatrix:
    """Tests for the enhanced strain-displacement matrix G."""

    def test_g_matrix_shape(self, hex8i_element):
        """G-matrix has shape (6, 9)."""
        G = hex8i_element.G_matrix(0.0, 0.0, 0.0)
        assert G.shape == (6, 9)

    def test_g_matrix_at_center_near_zero(self, hex8i_element):
        """At element center, mode derivatives are zero so G is nearly zero."""
        G = hex8i_element.G_matrix(0.0, 0.0, 0.0)
        # At center, dP/dxi = 0 for all modes, so G should be all zeros
        np.testing.assert_allclose(G, 0.0, atol=1e-14)

    def test_g_matrix_nonzero_away_from_center(self, hex8i_element):
        """Away from center, G has non-zero entries."""
        G = hex8i_element.G_matrix(0.5, 0.5, 0.5)
        assert np.any(np.abs(G) > 1e-10)


# ======================================================================
# Stiffness matrix
# ======================================================================

class TestStiffnessMatrix:
    """Tests for Hex8IElement.stiffness_matrix (with static condensation)."""

    @pytest.fixture(autouse=True)
    def _suppress_runtime_warnings(self):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        yield

    def test_shape(self, hex8i_element):
        """Condensed stiffness matrix has shape (24, 24)."""
        K = hex8i_element.stiffness_matrix()
        assert K.shape == (24, 24)

    def test_symmetric(self, hex8i_element):
        """Condensed stiffness matrix is symmetric."""
        K = hex8i_element.stiffness_matrix()
        np.testing.assert_allclose(K, K.T, atol=1e-8)

    def test_differs_from_hex8(self, hex8i_element, hex8_element):
        """Incompatible mode stiffness differs from standard hex8.

        The static condensation of internal DOFs modifies the stiffness,
        so K_hex8i != K_hex8 in general.
        """
        K_i = hex8i_element.stiffness_matrix()
        K_std = hex8_element.stiffness_matrix()
        # They should have the same shape but different values
        assert K_i.shape == K_std.shape
        assert not np.allclose(K_i, K_std, atol=1e-6)

    def test_caches_condensation_matrices(self, hex8i_element):
        """After stiffness_matrix(), _K_aa_inv and _K_au are populated."""
        assert hex8i_element._K_aa_inv is None
        assert hex8i_element._K_au is None
        hex8i_element.stiffness_matrix()
        assert hex8i_element._K_aa_inv is not None
        assert hex8i_element._K_aa_inv.shape == (9, 9)
        assert hex8i_element._K_au is not None
        assert hex8i_element._K_au.shape == (9, 24)


# ======================================================================
# Internal DOF recovery
# ======================================================================

class TestInternalDOFRecovery:
    """Tests for _recover_internal_dofs method."""

    def test_recovery_requires_stiffness_first(self, hex8i_element):
        """Calling _recover_internal_dofs before stiffness_matrix raises RuntimeError."""
        with pytest.raises(RuntimeError, match="stiffness_matrix"):
            hex8i_element._recover_internal_dofs(np.zeros(24))


# ======================================================================
# Stress and strain recovery
# ======================================================================

class TestStressStrainRecovery:
    """Tests for stress/strain at Gauss points with incompatible modes."""

    @pytest.fixture(autouse=True)
    def _suppress_runtime_warnings(self):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        yield

    def test_zero_displacement_zero_strain(self, hex8i_element):
        """Zero displacement gives zero total strain (compatible + incompatible)."""
        hex8i_element.stiffness_matrix()  # populate condensation matrices
        u = np.zeros(24)
        strains = hex8i_element.strain_at_gauss_points(u)
        assert strains.shape == (8, 6)
        np.testing.assert_allclose(strains, 0.0, atol=1e-14)

    def test_stress_shape(self, hex8i_element):
        """Stress at Gauss points has shape (8, 6)."""
        hex8i_element.stiffness_matrix()
        u = np.zeros(24)
        stresses = hex8i_element.stress_at_gauss_points(u)
        assert stresses.shape == (8, 6)


# ======================================================================
# Class attributes
# ======================================================================

class TestClassAttributes:
    """Test class-level constants."""

    def test_n_modes(self):
        """N_MODES is 3."""
        assert Hex8IElement.N_MODES == 3

    def test_n_internal_dof(self):
        """N_INTERNAL_DOF is 9 (3 modes x 3 displacement components)."""
        assert Hex8IElement.N_INTERNAL_DOF == 9
