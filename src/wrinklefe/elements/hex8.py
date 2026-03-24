"""8-node isoparametric hexahedral element (C3D8) for 3-D composite analysis.

Implements the standard trilinear brick element with:

- Isoparametric shape functions and their derivatives
- Jacobian mapping between natural and physical coordinates
- Strain-displacement (B) matrix in Voigt notation
- Element stiffness matrix via 2x2x2 Gauss quadrature
- Stress and strain recovery at Gauss points
- Consistent mass matrix
- Support for ply-angle rotation and spatially varying wrinkle misalignment

Node ordering follows the VTK / Abaqus convention::

    Bottom face (zeta = -1):        Top face (zeta = +1):
        3 ---- 2                        7 ---- 6
       /|     /|                       /|     /|
      0 ---- 1 |    (CCW from +z)     4 ---- 5 |
      | |    | |                      | |    | |
      | 3 ---| 2   <- hidden          | 7 ---| 6  <- hidden
      |/     |/                       |/     |/
      0 ---- 1                        4 ---- 5

Natural coordinates: xi, eta, zeta in [-1, 1].

References
----------
Zienkiewicz, O.C. & Taylor, R.L. (2000). The Finite Element Method, Vol. 1.
Bathe, K.-J. (2006). Finite Element Procedures.
Cook, R.D. et al. (2002). Concepts and Applications of Finite Element Analysis.
"""

from __future__ import annotations

import numpy as np

from wrinklefe.elements.gauss import gauss_points_hex
from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.core.transforms import rotate_stiffness_3d


# Natural coordinates of the 8 hex nodes in (xi, eta, zeta)
_NODE_COORDS = np.array([
    [-1.0, -1.0, -1.0],  # node 0
    [+1.0, -1.0, -1.0],  # node 1
    [+1.0, +1.0, -1.0],  # node 2
    [-1.0, +1.0, -1.0],  # node 3
    [-1.0, -1.0, +1.0],  # node 4
    [+1.0, -1.0, +1.0],  # node 5
    [+1.0, +1.0, +1.0],  # node 6
    [-1.0, +1.0, +1.0],  # node 7
], dtype=float)


class Hex8Element:
    """8-node isoparametric hexahedral element for 3-D composite analysis.

    Each element stores its physical node coordinates, material properties,
    nominal ply orientation, and per-node wrinkle misalignment angles.  The
    stiffness matrix accounts for both the ply-angle rotation (about the
    z-axis) and the spatially varying wrinkle rotation (about the y-axis).

    Parameters
    ----------
    node_coords : np.ndarray
        Shape ``(8, 3)`` — physical (x, y, z) coordinates of the 8 nodes (mm).
    material : OrthotropicMaterial
        Material properties for this element.
    ply_angle : float, optional
        Nominal ply orientation angle in **degrees**, measured from the
        global x-axis.  Converted to radians internally.  Default is 0.
    wrinkle_angles : np.ndarray or None, optional
        Shape ``(8,)`` — fiber misalignment angle at each node in **radians**.
        Values are interpolated to Gauss points via the shape functions.
        If ``None``, all wrinkle angles are zero (pristine laminate).
    """

    def __init__(
        self,
        node_coords: np.ndarray,
        material: OrthotropicMaterial,
        ply_angle: float = 0.0,
        wrinkle_angles: np.ndarray | None = None,
    ) -> None:
        self.node_coords = np.asarray(node_coords, dtype=float)
        if self.node_coords.shape != (8, 3):
            raise ValueError(
                f"node_coords must have shape (8, 3), got {self.node_coords.shape}."
            )

        self.material = material
        self.ply_angle = ply_angle
        self.wrinkle_angles = (
            np.asarray(wrinkle_angles, dtype=float)
            if wrinkle_angles is not None
            else np.zeros(8)
        )
        if self.wrinkle_angles.shape != (8,):
            raise ValueError(
                f"wrinkle_angles must have shape (8,), got {self.wrinkle_angles.shape}."
            )

        # Pre-compute Gauss quadrature points and weights (2x2x2 for hex8)
        self._gauss_points, self._gauss_weights = gauss_points_hex(order=2)

    # ------------------------------------------------------------------
    # Shape functions
    # ------------------------------------------------------------------

    @staticmethod
    def shape_functions(xi: float, eta: float, zeta: float) -> np.ndarray:
        """Evaluate the 8 trilinear shape functions at natural coordinates.

        .. math::
            N_i = \\frac{1}{8}(1 + \\xi_i \\xi)(1 + \\eta_i \\eta)(1 + \\zeta_i \\zeta)

        where ``(xi_i, eta_i, zeta_i)`` are the natural coordinates of node *i*.

        Parameters
        ----------
        xi, eta, zeta : float
            Natural coordinates in [-1, 1].

        Returns
        -------
        np.ndarray
            Shape ``(8,)`` — values of the 8 shape functions.
        """
        N = np.empty(8)
        for i in range(8):
            N[i] = (
                0.125
                * (1.0 + _NODE_COORDS[i, 0] * xi)
                * (1.0 + _NODE_COORDS[i, 1] * eta)
                * (1.0 + _NODE_COORDS[i, 2] * zeta)
            )
        return N

    @staticmethod
    def shape_derivatives(xi: float, eta: float, zeta: float) -> np.ndarray:
        """Derivatives of shape functions with respect to natural coordinates.

        Returns
        -------
        np.ndarray
            Shape ``(3, 8)`` — ``dN[i, j] = dN_j / d(xi_i)``, where
            ``xi_0 = xi``, ``xi_1 = eta``, ``xi_2 = zeta``.
        """
        dN = np.empty((3, 8))
        for j in range(8):
            xi_j, eta_j, zeta_j = _NODE_COORDS[j]
            dN[0, j] = 0.125 * xi_j * (1.0 + eta_j * eta) * (1.0 + zeta_j * zeta)
            dN[1, j] = 0.125 * (1.0 + xi_j * xi) * eta_j * (1.0 + zeta_j * zeta)
            dN[2, j] = 0.125 * (1.0 + xi_j * xi) * (1.0 + eta_j * eta) * zeta_j
        return dN

    # ------------------------------------------------------------------
    # Jacobian and B-matrix
    # ------------------------------------------------------------------

    def jacobian(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """Jacobian matrix mapping natural to physical coordinates.

        .. math::
            J_{ij} = \\sum_k \\frac{\\partial N_k}{\\partial \\xi_i} \\, x_{k,j}

        Parameters
        ----------
        xi, eta, zeta : float
            Natural coordinates.

        Returns
        -------
        np.ndarray
            Shape ``(3, 3)`` — the Jacobian matrix ``dx/d(xi)``.
        """
        dN = self.shape_derivatives(xi, eta, zeta)  # (3, 8)
        return dN @ self.node_coords  # (3, 8) @ (8, 3) = (3, 3)

    def B_matrix(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """Strain-displacement matrix (6 x 24) in Voigt notation.

        Relates the 6-component engineering strain vector to the 24 nodal
        displacement DOFs:

        .. math::
            \\varepsilon = B \\, u_e

        Strain ordering: ``[eps_11, eps_22, eps_33, gamma_23, gamma_13, gamma_12]``

        DOF ordering: ``[u1x, u1y, u1z, u2x, u2y, u2z, ..., u8x, u8y, u8z]``

        Parameters
        ----------
        xi, eta, zeta : float
            Natural coordinates.

        Returns
        -------
        np.ndarray
            Shape ``(6, 24)`` — the strain-displacement matrix.
        """
        dN_dxi = self.shape_derivatives(xi, eta, zeta)  # (3, 8)
        J = dN_dxi @ self.node_coords  # (3, 3)
        J_inv = np.linalg.inv(J)
        dN_dx = J_inv @ dN_dxi  # (3, 8) — derivatives in physical coords

        B = np.zeros((6, 24))
        for i in range(8):
            col = 3 * i
            dNi_dx = dN_dx[0, i]
            dNi_dy = dN_dx[1, i]
            dNi_dz = dN_dx[2, i]

            # eps_11 = du/dx
            B[0, col] = dNi_dx
            # eps_22 = dv/dy
            B[1, col + 1] = dNi_dy
            # eps_33 = dw/dz
            B[2, col + 2] = dNi_dz
            # gamma_23 = dv/dz + dw/dy
            B[3, col + 1] = dNi_dz
            B[3, col + 2] = dNi_dy
            # gamma_13 = du/dz + dw/dx
            B[4, col] = dNi_dz
            B[4, col + 2] = dNi_dx
            # gamma_12 = du/dy + dv/dx
            B[5, col] = dNi_dy
            B[5, col + 1] = dNi_dx

        return B

    # ------------------------------------------------------------------
    # Material stiffness with ply + wrinkle rotations
    # ------------------------------------------------------------------

    def rotated_stiffness(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """Rotated 6x6 stiffness matrix at a point in the element.

        Two successive rotations are applied to the material stiffness:

        1. **Ply angle** rotation about the z-axis (through-thickness direction).
        2. **Wrinkle misalignment** rotation about the y-axis (in-plane transverse),
           where the wrinkle angle is interpolated from nodal values using the
           shape functions.

        .. math::
            \\bar{C} = T_y^{-1}(\\phi) \\; T_z^{-1}(\\theta) \\; C \\; T_z^\\varepsilon(\\theta) \\; T_y^\\varepsilon(\\phi)

        Parameters
        ----------
        xi, eta, zeta : float
            Natural coordinates at which to evaluate.

        Returns
        -------
        np.ndarray
            Shape ``(6, 6)`` — rotated stiffness matrix (MPa).
        """
        C = self.material.stiffness_matrix  # (6, 6) in material axes

        # 1. Ply angle rotation about z
        ply_rad = np.radians(self.ply_angle)
        if abs(ply_rad) > 1.0e-15:
            C = rotate_stiffness_3d(C, ply_rad, axis='z')

        # 2. Wrinkle misalignment rotation about y (interpolated from nodes)
        N = self.shape_functions(xi, eta, zeta)  # (8,)
        phi = float(N @ self.wrinkle_angles)
        if abs(phi) > 1.0e-15:
            C = rotate_stiffness_3d(C, phi, axis='y')

        return C

    # ------------------------------------------------------------------
    # Element matrices
    # ------------------------------------------------------------------

    def stiffness_matrix(self) -> np.ndarray:
        """Element stiffness matrix (24 x 24).

        Computed via 2x2x2 Gauss quadrature:

        .. math::
            K_e = \\sum_{gp} B^T \\, \\bar{C} \\, B \\, |J| \\, w_{gp}

        Returns
        -------
        np.ndarray
            Shape ``(24, 24)`` — symmetric positive semi-definite stiffness matrix.
        """
        Ke = np.zeros((24, 24))

        for gp_idx in range(len(self._gauss_weights)):
            xi, eta, zeta = self._gauss_points[gp_idx]
            w = self._gauss_weights[gp_idx]

            B = self.B_matrix(xi, eta, zeta)  # (6, 24)
            C_bar = self.rotated_stiffness(xi, eta, zeta)  # (6, 6)
            J = self.jacobian(xi, eta, zeta)  # (3, 3)
            detJ = np.linalg.det(J)

            Ke += (B.T @ C_bar @ B) * detJ * w

        return Ke

    def stress_at_gauss_points(self, u_elem: np.ndarray) -> np.ndarray:
        """Compute stress at all Gauss points from element nodal displacements.

        .. math::
            \\sigma_{gp} = \\bar{C} \\, B \\, u_e

        Parameters
        ----------
        u_elem : np.ndarray
            Shape ``(24,)`` — element nodal displacement vector
            ``[u1x, u1y, u1z, ..., u8x, u8y, u8z]``.

        Returns
        -------
        np.ndarray
            Shape ``(n_gp, 6)`` — stress in Voigt notation at each Gauss point.
            Ordering: ``[sigma_11, sigma_22, sigma_33, tau_23, tau_13, tau_12]``.
        """
        u_elem = np.asarray(u_elem, dtype=float)
        n_gp = len(self._gauss_weights)
        stresses = np.empty((n_gp, 6))

        for gp_idx in range(n_gp):
            xi, eta, zeta = self._gauss_points[gp_idx]
            B = self.B_matrix(xi, eta, zeta)
            C_bar = self.rotated_stiffness(xi, eta, zeta)
            stresses[gp_idx] = C_bar @ (B @ u_elem)

        return stresses

    def strain_at_gauss_points(self, u_elem: np.ndarray) -> np.ndarray:
        """Compute strain at all Gauss points from element nodal displacements.

        .. math::
            \\varepsilon_{gp} = B \\, u_e

        Parameters
        ----------
        u_elem : np.ndarray
            Shape ``(24,)`` — element nodal displacement vector.

        Returns
        -------
        np.ndarray
            Shape ``(n_gp, 6)`` — engineering strain in Voigt notation at each
            Gauss point.  Ordering: ``[eps_11, eps_22, eps_33, gamma_23,
            gamma_13, gamma_12]``.
        """
        u_elem = np.asarray(u_elem, dtype=float)
        n_gp = len(self._gauss_weights)
        strains = np.empty((n_gp, 6))

        for gp_idx in range(n_gp):
            xi, eta, zeta = self._gauss_points[gp_idx]
            B = self.B_matrix(xi, eta, zeta)
            strains[gp_idx] = B @ u_elem

        return strains

    def mass_matrix(self, density: float = 1.0) -> np.ndarray:
        """Consistent mass matrix (24 x 24).

        Computed via 2x2x2 Gauss quadrature:

        .. math::
            M_e = \\rho \\sum_{gp} N^T \\, N \\, |J| \\, w_{gp}

        where *N* is the ``(3, 24)`` shape function matrix formed by placing
        scalar shape functions on the diagonal of each 3x3 block.

        Parameters
        ----------
        density : float, optional
            Mass density (kg/mm^3 or consistent units).  Default is 1.0.

        Returns
        -------
        np.ndarray
            Shape ``(24, 24)`` — consistent mass matrix.
        """
        Me = np.zeros((24, 24))

        for gp_idx in range(len(self._gauss_weights)):
            xi, eta, zeta = self._gauss_points[gp_idx]
            w = self._gauss_weights[gp_idx]

            N_scalar = self.shape_functions(xi, eta, zeta)  # (8,)
            J = self.jacobian(xi, eta, zeta)
            detJ = np.linalg.det(J)

            # Build (3, 24) shape function matrix: N_mat
            N_mat = np.zeros((3, 24))
            for i in range(8):
                N_mat[0, 3 * i] = N_scalar[i]
                N_mat[1, 3 * i + 1] = N_scalar[i]
                N_mat[2, 3 * i + 2] = N_scalar[i]

            Me += density * (N_mat.T @ N_mat) * detJ * w

        return Me

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def volume(self) -> float:
        """Element volume computed via Gauss quadrature.

        .. math::
            V = \\sum_{gp} |J| \\, w_{gp}

        Returns
        -------
        float
            Element volume in cubic length units (mm^3).
        """
        vol = 0.0
        for gp_idx in range(len(self._gauss_weights)):
            xi, eta, zeta = self._gauss_points[gp_idx]
            w = self._gauss_weights[gp_idx]
            J = self.jacobian(xi, eta, zeta)
            vol += np.linalg.det(J) * w
        return float(vol)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Hex8Element(ply_angle={self.ply_angle:.1f}deg, "
            f"material={self.material.name!r}, "
            f"volume={self.volume:.4f}mm3)"
        )
