"""8-node hexahedral element with incompatible modes (C3D8I).

Adds 13 → 9 internal (incompatible) displacement DOFs to the standard 8-node
hex to eliminate shear and volumetric locking in bending-dominated problems
such as composite wrinkle analysis.

The implementation follows Wilson et al. (1973) and Taylor et al. (1976).
Three quadratic bubble functions in the natural coordinates provide 9
internal DOFs (3 modes x 3 displacement components) that are statically
condensed out at the element level, preserving the standard 24 external DOFs.

References
----------
Wilson, E.L., Taylor, R.L., Doherty, W.P., Ghaboussi, J. (1973).
    "Incompatible displacement models." In *Numerical and Computer Methods
    in Structural Mechanics*, Academic Press.

Taylor, R.L., Beresford, P.J., Wilson, E.L. (1976).
    "A non-conforming element for stress analysis." *IJNME*, 10(6), 1211-1219.
"""

from __future__ import annotations

import numpy as np

from wrinklefe.elements.hex8 import Hex8Element
from wrinklefe.elements.gauss import gauss_points_hex
from wrinklefe.core.material import OrthotropicMaterial


class Hex8IElement(Hex8Element):
    """8-node hexahedral element with incompatible modes (C3D8I).

    Adds 9 internal DOFs (incompatible modes) to the standard hex8 to
    eliminate shear and volumetric locking in bending.  The internal DOFs
    are statically condensed out, so the element still exposes 24 external
    DOFs to the global assembler.

    The incompatible modes are based on Wilson et al. (1973) and Taylor
    et al. (1976).

    Incompatible mode functions in natural coordinates
    --------------------------------------------------
    Three quadratic bubble functions, one per natural coordinate axis::

        P1(xi, eta, zeta) = 1 - xi^2
        P2(xi, eta, zeta) = 1 - eta^2
        P3(xi, eta, zeta) = 1 - zeta^2

    Each mode contributes three displacement components (u, v, w), giving
    9 internal DOFs total.  The enhanced displacement is::

        u_enhanced = sum_k  P_k * alpha_k      (k = 1..3, alpha_k in R^3)

    The corresponding enhanced strain field is::

        epsilon_enhanced = G * alpha

    where ``G`` (6 x 9) contains the spatial derivatives of the mode
    functions, and ``alpha`` (9 x 1) collects all internal DOFs.

    Static condensation
    -------------------
    The element stiffness is partitioned as::

        | K_uu   K_ua | | u     |   | f |
        |             | |       | = |   |
        | K_au   K_aa | | alpha |   | 0 |

    Condensing the internal DOFs (whose right-hand side is zero)::

        alpha = -K_aa^{-1} K_au u
        K_condensed = K_uu - K_ua K_aa^{-1} K_au      (24 x 24)

    This ``K_condensed`` is returned by :meth:`stiffness_matrix`.

    Parameters
    ----------
    node_coords : np.ndarray
        Shape ``(8, 3)`` — physical coordinates of the element nodes.
    material : OrthotropicMaterial
        Composite ply material.
    ply_angle : float, optional
        Ply orientation angle about the z-axis in radians.  Default 0.
    wrinkle_angles : np.ndarray or None, optional
        Per-node fiber misalignment angles (rad) in the x-z plane.
        Shape ``(8,)``.  If *None*, no wrinkle rotation is applied.

    Attributes
    ----------
    _K_aa_inv : np.ndarray or None
        Cached inverse of the internal mode stiffness (9 x 9).
        Populated by :meth:`stiffness_matrix`.
    _K_au : np.ndarray or None
        Cached coupling matrix (9 x 24).
        Populated by :meth:`stiffness_matrix`.
    """

    # Number of incompatible modes and corresponding internal DOFs
    N_MODES: int = 3
    N_INTERNAL_DOF: int = 9  # 3 modes x 3 displacement components

    def __init__(
        self,
        node_coords: np.ndarray,
        material: OrthotropicMaterial,
        ply_angle: float = 0.0,
        wrinkle_angles: np.ndarray | None = None,
    ) -> None:
        super().__init__(node_coords, material, ply_angle, wrinkle_angles)

        # Caches for static condensation — populated by stiffness_matrix()
        self._K_aa_inv: np.ndarray | None = None
        self._K_au: np.ndarray | None = None

        # Pre-compute Jacobian at element center (xi=eta=zeta=0) for the
        # incompatible mode derivative mapping.
        self._J0, self._detJ0, self._J0_inv = self._jacobian_at_center()

    # ------------------------------------------------------------------
    # Jacobian at element center
    # ------------------------------------------------------------------

    def _jacobian_at_center(self) -> tuple[np.ndarray, float, np.ndarray]:
        """Evaluate the Jacobian, its determinant, and its inverse at the
        element center (xi = eta = zeta = 0).

        Returns
        -------
        J0 : np.ndarray
            Shape ``(3, 3)`` — Jacobian matrix at the element center.
        detJ0 : float
            Determinant of *J0*.
        J0_inv : np.ndarray
            Shape ``(3, 3)`` — inverse of *J0*.
        """
        J0 = self.jacobian(0.0, 0.0, 0.0)
        detJ0 = np.linalg.det(J0)
        J0_inv = np.linalg.inv(J0)
        return J0, detJ0, J0_inv

    # ------------------------------------------------------------------
    # Incompatible mode functions and derivatives
    # ------------------------------------------------------------------

    @staticmethod
    def incompatible_modes(xi: float, eta: float, zeta: float) -> np.ndarray:
        """Incompatible mode functions at a point in natural coordinates.

        The three Wilson bubble modes are::

            P1(xi, eta, zeta) = 1 - xi^2
            P2(xi, eta, zeta) = 1 - eta^2
            P3(xi, eta, zeta) = 1 - zeta^2

        Parameters
        ----------
        xi, eta, zeta : float
            Natural coordinates in [-1, 1].

        Returns
        -------
        np.ndarray
            Shape ``(3,)`` — mode values ``[P1, P2, P3]``.
        """
        return np.array([
            1.0 - xi * xi,
            1.0 - eta * eta,
            1.0 - zeta * zeta,
        ])

    @staticmethod
    def incompatible_mode_derivatives(
        xi: float, eta: float, zeta: float
    ) -> np.ndarray:
        """Derivatives of incompatible modes w.r.t. natural coordinates.

        The gradient matrix is::

            dP[i, j] = dP_{j+1} / d(xi_i)

        where xi_0 = xi, xi_1 = eta, xi_2 = zeta.  Explicitly::

            dP1/dxi   = -2 xi     dP2/dxi   = 0        dP3/dxi   = 0
            dP1/deta  = 0         dP2/deta  = -2 eta   dP3/deta  = 0
            dP1/dzeta = 0         dP2/dzeta = 0        dP3/dzeta = -2 zeta

        Parameters
        ----------
        xi, eta, zeta : float
            Natural coordinates in [-1, 1].

        Returns
        -------
        np.ndarray
            Shape ``(3, 3)`` — ``dP[i, j] = partial P_{j} / partial xi_{i}``.
        """
        return np.array([
            [-2.0 * xi,  0.0,         0.0],
            [ 0.0,      -2.0 * eta,   0.0],
            [ 0.0,       0.0,        -2.0 * zeta],
        ])

    # ------------------------------------------------------------------
    # Enhanced strain-displacement matrix (G)
    # ------------------------------------------------------------------

    def G_matrix(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """Enhanced strain-displacement matrix from incompatible modes.

        Maps the 9 internal DOFs to 6 engineering strain components using
        derivatives of the incompatible modes mapped to physical space via
        the Jacobian evaluated at the element center (J0).

        The internal DOF vector ``alpha`` is ordered as::

            alpha = [a1_x, a1_y, a1_z,   # mode 1 displacements
                     a2_x, a2_y, a2_z,   # mode 2 displacements
                     a3_x, a3_y, a3_z]   # mode 3 displacements

        and the strain vector (Voigt) is::

            epsilon = [e11, e22, e33, g23, g13, g12]

        Construction follows the same pattern as the standard B matrix but
        uses the mode derivatives instead of shape function derivatives,
        and maps them through the *center* Jacobian inverse J0_inv for
        numerical stability (standard practice for incompatible modes).

        A correction factor ``detJ0 / detJ`` is applied to ensure that the
        incompatible modes satisfy the patch test (Taylor et al. 1976).

        Parameters
        ----------
        xi, eta, zeta : float
            Natural coordinates in [-1, 1].

        Returns
        -------
        np.ndarray
            Shape ``(6, 9)`` — enhanced strain-displacement matrix.
        """
        # Derivatives of mode functions w.r.t. natural coordinates (3 x 3)
        dP_dnat = self.incompatible_mode_derivatives(xi, eta, zeta)

        # Map to physical derivatives using J0 inverse: dP/dx = J0_inv @ dP/dxi
        dP_dphys = self._J0_inv @ dP_dnat  # (3, 3): dP_dphys[i, k] = dP_k/dx_i

        # Patch-test correction: scale by detJ0 / detJ(xi, eta, zeta)
        J_local = self.jacobian(xi, eta, zeta)
        detJ_local = np.linalg.det(J_local)
        scale = self._detJ0 / detJ_local

        dP_dphys = dP_dphys * scale

        # Assemble G matrix (6 x 9)
        # For each mode k (0..2), the sub-block occupies columns [3k : 3k+3]
        # and follows the same strain-displacement pattern as the B matrix:
        #
        #   e11 -> du/dx  :  dP_k/dx  in row 0, col 3k
        #   e22 -> dv/dy  :  dP_k/dy  in row 1, col 3k+1
        #   e33 -> dw/dz  :  dP_k/dz  in row 2, col 3k+2
        #   g23 -> dv/dz + dw/dy  :  dP_k/dz in row 3, col 3k+1; dP_k/dy in row 3, col 3k+2
        #   g13 -> du/dz + dw/dx  :  dP_k/dz in row 4, col 3k;   dP_k/dx in row 4, col 3k+2
        #   g12 -> du/dy + dv/dx  :  dP_k/dy in row 5, col 3k;   dP_k/dx in row 5, col 3k+1

        G = np.zeros((6, 9), dtype=np.float64)

        for k in range(self.N_MODES):
            # Physical derivatives of mode k
            dPk_dx = dP_dphys[0, k]
            dPk_dy = dP_dphys[1, k]
            dPk_dz = dP_dphys[2, k]

            col = 3 * k

            # Normal strains
            G[0, col]     = dPk_dx   # e11
            G[1, col + 1] = dPk_dy   # e22
            G[2, col + 2] = dPk_dz   # e33

            # Engineering shear strains
            G[3, col + 1] = dPk_dz   # g23: dv/dz
            G[3, col + 2] = dPk_dy   # g23: dw/dy

            G[4, col]     = dPk_dz   # g13: du/dz
            G[4, col + 2] = dPk_dx   # g13: dw/dx

            G[5, col]     = dPk_dy   # g12: du/dy
            G[5, col + 1] = dPk_dx   # g12: dv/dx

        return G

    # ------------------------------------------------------------------
    # Stiffness matrix with static condensation
    # ------------------------------------------------------------------

    def stiffness_matrix(self) -> np.ndarray:
        """Condensed element stiffness matrix (24 x 24).

        Assembles the partitioned system using 2x2x2 Gauss quadrature::

            K_uu  (24 x 24) :  standard stiffness from compatible modes
            K_ua  (24 x  9) :  coupling between external and internal DOFs
            K_au  ( 9 x 24) :  transpose of K_ua
            K_aa  ( 9 x  9) :  internal mode stiffness

        Then performs static condensation::

            K = K_uu - K_ua @ K_aa^{-1} @ K_au

        The condensed matrix is symmetric by construction.

        Side effects: stores ``_K_aa_inv`` and ``_K_au`` for use in stress
        and strain recovery.

        Returns
        -------
        np.ndarray
            Shape ``(24, 24)`` — condensed stiffness matrix.
        """
        n_ext = 24  # 8 nodes x 3 DOFs
        n_int = self.N_INTERNAL_DOF  # 9

        K_uu = np.zeros((n_ext, n_ext), dtype=np.float64)
        K_ua = np.zeros((n_ext, n_int), dtype=np.float64)
        K_aa = np.zeros((n_int, n_int), dtype=np.float64)

        # Get quadrature points and weights (2x2x2)
        gp_coords, gp_weights = gauss_points_hex(order=2)

        for i_gp in range(gp_coords.shape[0]):
            xi   = gp_coords[i_gp, 0]
            eta  = gp_coords[i_gp, 1]
            zeta = gp_coords[i_gp, 2]
            w_gp = gp_weights[i_gp]

            # Standard B matrix and Jacobian from parent class
            B = self.B_matrix(xi, eta, zeta)          # (6, 24)
            J = self.jacobian(xi, eta, zeta)           # (3, 3)
            detJ = np.linalg.det(J)

            # Rotated material stiffness at this point
            C_bar = self.rotated_stiffness(xi, eta, zeta)  # (6, 6)

            # Enhanced strain-displacement matrix from incompatible modes
            G = self.G_matrix(xi, eta, zeta)           # (6, 9)

            # Integration weight
            dV = w_gp * detJ

            # Accumulate sub-matrices: K = integral(B^T C B dV), etc.
            CB = C_bar @ B    # (6, 24)
            CG = C_bar @ G    # (6, 9)

            K_uu += (B.T @ CB) * dV
            K_ua += (B.T @ CG) * dV
            K_aa += (G.T @ CG) * dV

        # K_au = K_ua^T (by symmetry of the bilinear form)
        K_au = K_ua.T

        # Static condensation
        K_aa_inv = np.linalg.inv(K_aa)
        K_condensed = K_uu - K_ua @ K_aa_inv @ K_au

        # Enforce exact symmetry (eliminate floating-point asymmetry)
        K_condensed = 0.5 * (K_condensed + K_condensed.T)

        # Cache for stress/strain recovery
        self._K_aa_inv = K_aa_inv
        self._K_au = K_au

        return K_condensed

    # ------------------------------------------------------------------
    # Internal DOF recovery
    # ------------------------------------------------------------------

    def _recover_internal_dofs(self, u_elem: np.ndarray) -> np.ndarray:
        """Recover the statically condensed internal DOFs.

        Given the external displacement vector, the internal DOFs are::

            alpha = -K_aa^{-1} @ K_au @ u

        Parameters
        ----------
        u_elem : np.ndarray
            Shape ``(24,)`` — element external displacement vector.

        Returns
        -------
        np.ndarray
            Shape ``(9,)`` — internal (incompatible) DOFs.

        Raises
        ------
        RuntimeError
            If :meth:`stiffness_matrix` has not been called yet (the
            condensation matrices have not been computed).
        """
        if self._K_aa_inv is None or self._K_au is None:
            raise RuntimeError(
                "Internal DOF recovery requires stiffness_matrix() to be "
                "called first to populate condensation matrices."
            )
        u_elem = np.asarray(u_elem, dtype=np.float64).ravel()
        return -self._K_aa_inv @ (self._K_au @ u_elem)

    # ------------------------------------------------------------------
    # Stress and strain at Gauss points
    # ------------------------------------------------------------------

    def stress_at_gauss_points(self, u_elem: np.ndarray) -> np.ndarray:
        """Compute stress including incompatible mode contributions.

        At each Gauss point the total strain is::

            epsilon = B @ u + G @ alpha

        and the stress is::

            sigma = C_bar @ epsilon

        where ``alpha`` is recovered via static condensation.

        Parameters
        ----------
        u_elem : np.ndarray
            Shape ``(24,)`` — element nodal displacement vector.

        Returns
        -------
        np.ndarray
            Shape ``(n_gp, 6)`` — Voigt stress at each Gauss point.
            Ordering: [sigma_11, sigma_22, sigma_33, tau_23, tau_13, tau_12].
        """
        u_elem = np.asarray(u_elem, dtype=np.float64).ravel()
        alpha = self._recover_internal_dofs(u_elem)

        gp_coords, _ = gauss_points_hex(order=2)
        n_gp = gp_coords.shape[0]
        stresses = np.zeros((n_gp, 6), dtype=np.float64)

        for i_gp in range(n_gp):
            xi   = gp_coords[i_gp, 0]
            eta  = gp_coords[i_gp, 1]
            zeta = gp_coords[i_gp, 2]

            B = self.B_matrix(xi, eta, zeta)            # (6, 24)
            G = self.G_matrix(xi, eta, zeta)            # (6, 9)
            C_bar = self.rotated_stiffness(xi, eta, zeta)  # (6, 6)

            strain = B @ u_elem + G @ alpha              # (6,)
            stresses[i_gp, :] = C_bar @ strain

        return stresses

    def strain_at_gauss_points(self, u_elem: np.ndarray) -> np.ndarray:
        """Compute strain including incompatible mode contributions.

        At each Gauss point the total strain is::

            epsilon = B @ u + G @ alpha

        where ``alpha`` is recovered via static condensation.

        Parameters
        ----------
        u_elem : np.ndarray
            Shape ``(24,)`` — element nodal displacement vector.

        Returns
        -------
        np.ndarray
            Shape ``(n_gp, 6)`` — engineering strains at each Gauss point.
            Ordering: [e11, e22, e33, g23, g13, g12].
        """
        u_elem = np.asarray(u_elem, dtype=np.float64).ravel()
        alpha = self._recover_internal_dofs(u_elem)

        gp_coords, _ = gauss_points_hex(order=2)
        n_gp = gp_coords.shape[0]
        strains = np.zeros((n_gp, 6), dtype=np.float64)

        for i_gp in range(n_gp):
            xi   = gp_coords[i_gp, 0]
            eta  = gp_coords[i_gp, 1]
            zeta = gp_coords[i_gp, 2]

            B = self.B_matrix(xi, eta, zeta)  # (6, 24)
            G = self.G_matrix(xi, eta, zeta)  # (6, 9)

            strains[i_gp, :] = B @ u_elem + G @ alpha

        return strains
