"""Tests for the 8-node hexahedral element (wrinklefe.elements.hex8).

Covers shape functions, derivatives, Jacobian, B-matrix, stiffness/mass
matrices, volume computation, stress/strain recovery, and a single-element
patch test.
"""

import numpy as np
import pytest

from wrinklefe.elements.hex8 import Hex8Element
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
    """Node coordinates for a unit cube [0,1]^3 in VTK/Abaqus ordering.

    Bottom face (zeta=-1 -> z=0):  0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0)
    Top face    (zeta=+1 -> z=1):  4:(0,0,1) 5:(1,0,1) 6:(1,1,1) 7:(0,1,1)
    """
    return np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        [0.0, 0.0, 1.0],  # 4
        [1.0, 0.0, 1.0],  # 5
        [1.0, 1.0, 1.0],  # 6
        [0.0, 1.0, 1.0],  # 7
    ], dtype=float)


@pytest.fixture
def unit_cube_element(unit_cube_nodes, x850_material):
    """Hex8 element on a unit cube with default material."""
    return Hex8Element(
        node_coords=unit_cube_nodes,
        material=x850_material,
        ply_angle=0.0,
    )


@pytest.fixture
def isotropic_material():
    """Near-isotropic material for simplified verification.

    E=10000 MPa, nu=0.3, G=E/(2*(1+nu))=3846.15 MPa.
    We set E1=E2=E3 and G12=G13=G23 and nu12=nu13=nu23.
    """
    E = 10000.0
    nu = 0.3
    G = E / (2.0 * (1.0 + nu))
    return OrthotropicMaterial(
        E1=E, E2=E, E3=E,
        G12=G, G13=G, G23=G,
        nu12=nu, nu13=nu, nu23=nu,
        Xt=500, Xc=500, Yt=500, Yc=500, Zt=500, Zc=500,
        S12=300, S13=300, S23=300,
        gamma_Y=0.02,
        name="isotropic_10k",
    )


# ======================================================================
# Shape functions
# ======================================================================

class TestShapeFunctions:
    """Tests for Hex8Element.shape_functions (static method)."""

    def test_sum_to_one_at_origin(self):
        """Shape functions sum to 1 at the element center (0,0,0)."""
        N = Hex8Element.shape_functions(0.0, 0.0, 0.0)
        assert np.isclose(N.sum(), 1.0)

    def test_sum_to_one_at_arbitrary_point(self):
        """Shape functions sum to 1 at an arbitrary interior point."""
        N = Hex8Element.shape_functions(0.3, -0.5, 0.7)
        assert np.isclose(N.sum(), 1.0)

    def test_node_0_kronecker(self):
        """At node 0 (-1,-1,-1): N0=1, all others=0."""
        N = Hex8Element.shape_functions(-1.0, -1.0, -1.0)
        expected = np.zeros(8)
        expected[0] = 1.0
        np.testing.assert_allclose(N, expected, atol=1e-15)

    def test_node_6_kronecker(self):
        """At node 6 (+1,+1,+1): N6=1, all others=0."""
        N = Hex8Element.shape_functions(1.0, 1.0, 1.0)
        expected = np.zeros(8)
        expected[6] = 1.0
        np.testing.assert_allclose(N, expected, atol=1e-15)

    def test_shape_at_center(self):
        """At element center, all shape functions equal 1/8."""
        N = Hex8Element.shape_functions(0.0, 0.0, 0.0)
        np.testing.assert_allclose(N, np.full(8, 0.125), atol=1e-15)

    def test_output_shape(self):
        """Shape functions return array of shape (8,)."""
        N = Hex8Element.shape_functions(0.0, 0.0, 0.0)
        assert N.shape == (8,)

    def test_all_nonnegative_inside(self):
        """All shape functions are non-negative inside the element."""
        # Test at the center
        N = Hex8Element.shape_functions(0.0, 0.0, 0.0)
        assert np.all(N >= 0)

    @pytest.mark.parametrize("node_idx", range(8))
    def test_kronecker_all_nodes(self, node_idx):
        """At each node, only that node's shape function is 1."""
        # Natural coordinates for each node
        node_coords = [
            (-1, -1, -1), (+1, -1, -1), (+1, +1, -1), (-1, +1, -1),
            (-1, -1, +1), (+1, -1, +1), (+1, +1, +1), (-1, +1, +1),
        ]
        xi, eta, zeta = node_coords[node_idx]
        N = Hex8Element.shape_functions(xi, eta, zeta)
        expected = np.zeros(8)
        expected[node_idx] = 1.0
        np.testing.assert_allclose(N, expected, atol=1e-15)


# ======================================================================
# Shape derivatives
# ======================================================================

class TestShapeDerivatives:
    """Tests for Hex8Element.shape_derivatives (static method)."""

    def test_output_shape(self):
        """Shape derivatives return (3, 8) array."""
        dN = Hex8Element.shape_derivatives(0.0, 0.0, 0.0)
        assert dN.shape == (3, 8)

    def test_partition_of_unity_derivative(self):
        """Sum of derivatives in each direction is zero (partition of unity)."""
        dN = Hex8Element.shape_derivatives(0.3, -0.5, 0.7)
        # Sum over the 8 nodes for each natural coordinate direction
        np.testing.assert_allclose(dN.sum(axis=1), np.zeros(3), atol=1e-14)

    def test_numerical_consistency(self):
        """Shape derivatives are consistent with finite differences."""
        xi0, eta0, zeta0 = 0.3, -0.2, 0.5
        h = 1e-7
        dN_analytical = Hex8Element.shape_derivatives(xi0, eta0, zeta0)

        # Numerical dN/dxi
        Np = Hex8Element.shape_functions(xi0 + h, eta0, zeta0)
        Nm = Hex8Element.shape_functions(xi0 - h, eta0, zeta0)
        dN_xi_numerical = (Np - Nm) / (2 * h)
        np.testing.assert_allclose(dN_analytical[0], dN_xi_numerical, atol=1e-6)

        # Numerical dN/deta
        Np = Hex8Element.shape_functions(xi0, eta0 + h, zeta0)
        Nm = Hex8Element.shape_functions(xi0, eta0 - h, zeta0)
        dN_eta_numerical = (Np - Nm) / (2 * h)
        np.testing.assert_allclose(dN_analytical[1], dN_eta_numerical, atol=1e-6)

        # Numerical dN/dzeta
        Np = Hex8Element.shape_functions(xi0, eta0, zeta0 + h)
        Nm = Hex8Element.shape_functions(xi0, eta0, zeta0 - h)
        dN_zeta_numerical = (Np - Nm) / (2 * h)
        np.testing.assert_allclose(dN_analytical[2], dN_zeta_numerical, atol=1e-6)


# ======================================================================
# Jacobian
# ======================================================================

class TestJacobian:
    """Tests for Hex8Element.jacobian."""

    def test_unit_cube_jacobian(self, unit_cube_element):
        """Jacobian of a unit cube [0,1]^3 is 0.5*I everywhere.

        The mapping is x = (xi+1)/2, y = (eta+1)/2, z = (zeta+1)/2,
        so dx/dxi = 0.5 for each axis.
        """
        J = unit_cube_element.jacobian(0.0, 0.0, 0.0)
        expected = 0.5 * np.eye(3)
        np.testing.assert_allclose(J, expected, atol=1e-14)

    def test_jacobian_shape(self, unit_cube_element):
        """Jacobian is a (3,3) matrix."""
        J = unit_cube_element.jacobian(0.3, -0.5, 0.7)
        assert J.shape == (3, 3)

    def test_jacobian_determinant_positive(self, unit_cube_element):
        """Jacobian determinant is positive for a valid element."""
        J = unit_cube_element.jacobian(0.0, 0.0, 0.0)
        assert np.linalg.det(J) > 0

    def test_jacobian_constant_for_cube(self, unit_cube_element):
        """For a unit cube, the Jacobian is the same at all Gauss points."""
        J_center = unit_cube_element.jacobian(0.0, 0.0, 0.0)
        J_corner = unit_cube_element.jacobian(0.5, -0.5, 0.5)
        np.testing.assert_allclose(J_corner, J_center, atol=1e-14)

    def test_scaled_cube_jacobian(self, x850_material):
        """A cube with side length L has Jacobian = (L/2)*I."""
        L = 5.0
        nodes = np.array([
            [0, 0, 0], [L, 0, 0], [L, L, 0], [0, L, 0],
            [0, 0, L], [L, 0, L], [L, L, L], [0, L, L],
        ], dtype=float)
        elem = Hex8Element(nodes, x850_material)
        J = elem.jacobian(0.0, 0.0, 0.0)
        np.testing.assert_allclose(J, (L / 2) * np.eye(3), atol=1e-12)


# ======================================================================
# B-matrix
# ======================================================================

class TestBMatrix:
    """Tests for Hex8Element.B_matrix."""

    def test_shape(self, unit_cube_element):
        """B-matrix has shape (6, 24)."""
        B = unit_cube_element.B_matrix(0.0, 0.0, 0.0)
        assert B.shape == (6, 24)

    def test_rigid_body_translation(self, unit_cube_element):
        """Rigid body translation produces zero strain."""
        B = unit_cube_element.B_matrix(0.0, 0.0, 0.0)
        # Uniform x-displacement: u = [1,0,0] at every node
        u_rigid = np.zeros(24)
        u_rigid[0::3] = 1.0  # ux = 1 at all nodes
        strain = B @ u_rigid
        np.testing.assert_allclose(strain, 0.0, atol=1e-13)

    def test_uniform_x_strain(self, unit_cube_element):
        """Uniform x-stretch gives eps_11 = du/dx only."""
        eps11 = 0.001
        # For unit cube [0,1]^3: ux = eps11 * x, where x maps from node coords
        # Node x-coords: [0,1,1,0,0,1,1,0]
        u = np.zeros(24)
        node_x = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=float)
        for i in range(8):
            u[3 * i] = eps11 * node_x[i]  # ux = eps11 * x

        B = unit_cube_element.B_matrix(0.0, 0.0, 0.0)
        strain = B @ u
        # eps_11 should be eps11, all others ~0
        assert np.isclose(strain[0], eps11, rtol=1e-10)
        np.testing.assert_allclose(strain[1:], 0.0, atol=1e-13)


# ======================================================================
# Stiffness matrix
# ======================================================================

class TestStiffnessMatrix:
    """Tests for Hex8Element.stiffness_matrix."""

    def test_shape(self, unit_cube_element):
        """Stiffness matrix has shape (24, 24)."""
        K = unit_cube_element.stiffness_matrix()
        assert K.shape == (24, 24)

    def test_symmetric(self, unit_cube_element):
        """Stiffness matrix is symmetric."""
        K = unit_cube_element.stiffness_matrix()
        np.testing.assert_allclose(K, K.T, atol=1e-8)

    def test_positive_semi_definite(self, unit_cube_element):
        """Stiffness matrix eigenvalues are non-negative."""
        K = unit_cube_element.stiffness_matrix()
        eigvals = np.linalg.eigvalsh(K)
        # 6 rigid body modes should give 6 near-zero eigenvalues
        assert np.all(eigvals > -1e-6 * np.max(eigvals))

    def test_six_rigid_body_modes(self, unit_cube_element):
        """A free element has exactly 6 zero-eigenvalue rigid body modes."""
        K = unit_cube_element.stiffness_matrix()
        eigvals = np.sort(np.linalg.eigvalsh(K))
        max_eig = eigvals[-1]
        # First 6 eigenvalues should be near zero
        assert np.all(np.abs(eigvals[:6]) < 1e-6 * max_eig)
        # Seventh eigenvalue should be clearly positive
        assert eigvals[6] > 1e-6 * max_eig

    def test_nonzero_entries(self, unit_cube_element):
        """Stiffness matrix has non-zero entries."""
        K = unit_cube_element.stiffness_matrix()
        assert np.any(K != 0.0)


# ======================================================================
# Mass matrix
# ======================================================================

class TestMassMatrix:
    """Tests for Hex8Element.mass_matrix."""

    def test_shape(self, unit_cube_element):
        """Mass matrix has shape (24, 24)."""
        M = unit_cube_element.mass_matrix(density=1.0)
        assert M.shape == (24, 24)

    def test_symmetric(self, unit_cube_element):
        """Mass matrix is symmetric."""
        M = unit_cube_element.mass_matrix(density=1.0)
        np.testing.assert_allclose(M, M.T, atol=1e-14)

    def test_positive_definite(self, unit_cube_element):
        """Mass matrix is positive definite (no zero eigenvalues)."""
        M = unit_cube_element.mass_matrix(density=1.0)
        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > 0)

    def test_total_mass(self, unit_cube_element):
        """Sum of diagonal blocks gives total mass = density * volume.

        For a consistent mass matrix, the trace = 3 * density * volume
        (since there are 3 translational DOFs).
        Actually, the total mass is the sum of the lumped (row-sum) mass
        matrix for each DOF direction. Alternatively, we check that a
        rigid body translation gives U^T M U = m * |v|^2.
        """
        rho = 2.5
        M = unit_cube_element.mass_matrix(density=rho)
        # Rigid body x-translation with unit velocity
        v = np.zeros(24)
        v[0::3] = 1.0  # unit velocity in x for all nodes
        # v^T M v = total_mass * 1^2 = rho * V
        kinetic = v @ M @ v
        expected_mass = rho * unit_cube_element.volume
        assert np.isclose(kinetic, expected_mass, rtol=1e-10)


# ======================================================================
# Volume
# ======================================================================

class TestVolume:
    """Tests for Hex8Element.volume property."""

    def test_unit_cube_volume(self, unit_cube_element):
        """Unit cube has volume = 1.0."""
        assert np.isclose(unit_cube_element.volume, 1.0, rtol=1e-12)

    def test_scaled_cube_volume(self, x850_material):
        """Cube with side length 3.0 has volume = 27.0."""
        L = 3.0
        nodes = np.array([
            [0, 0, 0], [L, 0, 0], [L, L, 0], [0, L, 0],
            [0, 0, L], [L, 0, L], [L, L, L], [0, L, L],
        ], dtype=float)
        elem = Hex8Element(nodes, x850_material)
        assert np.isclose(elem.volume, L**3, rtol=1e-12)

    def test_rectangular_element_volume(self, x850_material):
        """Rectangular element 2x3x0.5 has volume = 3.0."""
        nodes = np.array([
            [0, 0, 0], [2, 0, 0], [2, 3, 0], [0, 3, 0],
            [0, 0, 0.5], [2, 0, 0.5], [2, 3, 0.5], [0, 3, 0.5],
        ], dtype=float)
        elem = Hex8Element(nodes, x850_material)
        assert np.isclose(elem.volume, 2.0 * 3.0 * 0.5, rtol=1e-12)


# ======================================================================
# Stress and strain at Gauss points
# ======================================================================

class TestStressStrain:
    """Tests for stress_at_gauss_points and strain_at_gauss_points."""

    def test_zero_displacement_zero_strain(self, unit_cube_element):
        """Zero displacement gives zero strain at all Gauss points."""
        u = np.zeros(24)
        strains = unit_cube_element.strain_at_gauss_points(u)
        assert strains.shape == (8, 6)
        np.testing.assert_allclose(strains, 0.0, atol=1e-15)

    def test_zero_displacement_zero_stress(self, unit_cube_element):
        """Zero displacement gives zero stress at all Gauss points."""
        u = np.zeros(24)
        stresses = unit_cube_element.stress_at_gauss_points(u)
        assert stresses.shape == (8, 6)
        np.testing.assert_allclose(stresses, 0.0, atol=1e-12)

    def test_strain_shape(self, unit_cube_element):
        """Strain at Gauss points has shape (n_gp, 6)."""
        u = np.random.RandomState(42).randn(24) * 0.001
        strains = unit_cube_element.strain_at_gauss_points(u)
        assert strains.shape[0] == 8
        assert strains.shape[1] == 6

    def test_stress_shape(self, unit_cube_element):
        """Stress at Gauss points has shape (n_gp, 6)."""
        u = np.random.RandomState(42).randn(24) * 0.001
        stresses = unit_cube_element.stress_at_gauss_points(u)
        assert stresses.shape[0] == 8
        assert stresses.shape[1] == 6


# ======================================================================
# Patch test: constant stress under uniform strain
# ======================================================================

class TestPatchTest:
    """Single-element patch test verifying constant stress under uniform strain.

    For a single-element patch test with isotropic material:
    Apply displacement u = [eps11*x, 0, 0] at each node.
    The resulting strain field should be exactly eps_11 = eps11 at all
    Gauss points, and the stress sigma_11 = C11 * eps11 (plus Poisson
    coupling terms).
    """

    def test_constant_strain_field(self, isotropic_material, unit_cube_nodes):
        """Uniform x-strain produces constant strain at all Gauss points."""
        elem = Hex8Element(unit_cube_nodes, isotropic_material, ply_angle=0.0)

        eps11 = 0.001
        # Apply ux = eps11 * x at each node, uy = uz = 0
        u = np.zeros(24)
        for i in range(8):
            x_i = unit_cube_nodes[i, 0]
            u[3 * i] = eps11 * x_i

        strains = elem.strain_at_gauss_points(u)
        # eps_11 should be eps11 at all Gauss points
        np.testing.assert_allclose(strains[:, 0], eps11, rtol=1e-10)
        # Other strains should be zero
        np.testing.assert_allclose(strains[:, 1], 0.0, atol=1e-13)
        np.testing.assert_allclose(strains[:, 2], 0.0, atol=1e-13)
        np.testing.assert_allclose(strains[:, 3:], 0.0, atol=1e-13)

    def test_constant_stress_field(self, isotropic_material, unit_cube_nodes):
        """Uniform x-strain produces constant stress sigma_11 = C11 * eps11."""
        elem = Hex8Element(unit_cube_nodes, isotropic_material, ply_angle=0.0)

        eps11 = 0.001
        u = np.zeros(24)
        for i in range(8):
            x_i = unit_cube_nodes[i, 0]
            u[3 * i] = eps11 * x_i

        stresses = elem.stress_at_gauss_points(u)
        # Get C11 from the stiffness matrix
        C = isotropic_material.stiffness_matrix
        expected_sigma11 = C[0, 0] * eps11  # + C[0,1]*0 + C[0,2]*0 ...

        # sigma_11 should be constant and equal to C11 * eps11
        np.testing.assert_allclose(stresses[:, 0], expected_sigma11, rtol=1e-10)

        # sigma_22, sigma_33 should be C[1,0]*eps11, C[2,0]*eps11 (Poisson effect)
        expected_sigma22 = C[1, 0] * eps11
        expected_sigma33 = C[2, 0] * eps11
        np.testing.assert_allclose(stresses[:, 1], expected_sigma22, rtol=1e-10)
        np.testing.assert_allclose(stresses[:, 2], expected_sigma33, rtol=1e-10)

        # Shear stresses should be zero
        np.testing.assert_allclose(stresses[:, 3:], 0.0, atol=1e-8)

    def test_stress_is_constant_across_gauss_points(self, isotropic_material, unit_cube_nodes):
        """Under uniform strain, stress is identical at all 8 Gauss points."""
        elem = Hex8Element(unit_cube_nodes, isotropic_material, ply_angle=0.0)

        eps11 = 0.001
        u = np.zeros(24)
        for i in range(8):
            u[3 * i] = eps11 * unit_cube_nodes[i, 0]

        stresses = elem.stress_at_gauss_points(u)
        # All rows should be identical
        for gp in range(1, 8):
            np.testing.assert_allclose(stresses[gp], stresses[0], atol=1e-10)


# ======================================================================
# Element construction validation
# ======================================================================

class TestElementConstruction:
    """Tests for element construction and validation."""

    def test_invalid_node_shape(self, x850_material):
        """Non-(8,3) node_coords raises ValueError."""
        with pytest.raises(ValueError, match="shape"):
            Hex8Element(np.zeros((4, 3)), x850_material)

    def test_invalid_wrinkle_angles_shape(self, unit_cube_nodes, x850_material):
        """Non-(8,) wrinkle_angles raises ValueError."""
        with pytest.raises(ValueError, match="shape"):
            Hex8Element(unit_cube_nodes, x850_material, wrinkle_angles=np.zeros(4))

    def test_default_wrinkle_angles_zero(self, unit_cube_nodes, x850_material):
        """When wrinkle_angles is None, defaults to zeros."""
        elem = Hex8Element(unit_cube_nodes, x850_material)
        np.testing.assert_array_equal(elem.wrinkle_angles, np.zeros(8))

    def test_repr(self, unit_cube_element):
        """Repr string contains expected info."""
        r = repr(unit_cube_element)
        assert "Hex8Element" in r
        assert "ply_angle=0.0" in r


# ======================================================================
# Rotated stiffness
# ======================================================================

class TestRotatedStiffness:
    """Tests for Hex8Element.rotated_stiffness."""

    def test_zero_angle_identity(self, unit_cube_element):
        """With ply_angle=0 and no wrinkle, rotated stiffness = material stiffness."""
        C_rot = unit_cube_element.rotated_stiffness(0.0, 0.0, 0.0)
        C_mat = unit_cube_element.material.stiffness_matrix
        np.testing.assert_allclose(C_rot, C_mat, rtol=1e-10)

    def test_shape(self, unit_cube_element):
        """Rotated stiffness returns (6, 6) matrix."""
        C = unit_cube_element.rotated_stiffness(0.0, 0.0, 0.0)
        assert C.shape == (6, 6)

    def test_symmetric(self, unit_cube_element):
        """Rotated stiffness matrix is symmetric."""
        C = unit_cube_element.rotated_stiffness(0.3, -0.5, 0.7)
        np.testing.assert_allclose(C, C.T, atol=1e-8)

    def test_with_wrinkle_angles(self, unit_cube_nodes, x850_material):
        """Non-zero wrinkle angles change the stiffness."""
        wrinkle = np.full(8, 0.1)  # 0.1 rad wrinkle at all nodes
        elem = Hex8Element(unit_cube_nodes, x850_material, wrinkle_angles=wrinkle)
        C_wrinkled = elem.rotated_stiffness(0.0, 0.0, 0.0)
        C_pristine = x850_material.stiffness_matrix
        # Wrinkle rotation should change the stiffness
        assert not np.allclose(C_wrinkled, C_pristine, atol=1e-6)
