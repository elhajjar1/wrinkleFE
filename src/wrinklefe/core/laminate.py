"""Classical Lamination Theory (CLT) with First-Order Shear Deformation Theory (FSDT).

This module provides a complete CLT implementation for computing the mechanical
response of composite laminates. It includes:

- ABD stiffness matrix computation
- FSDT transverse shear stiffness (H matrix) with shear correction factor
- Ply-level stress and strain recovery in global and local coordinates
- Thermal and moisture resultant forces and moments
- Effective engineering constants for the laminate
- Symmetry and balance checks

References
----------
- Jones, R.M. (1999). Mechanics of Composite Materials, 2nd ed. Taylor & Francis.
- Reddy, J.N. (2004). Mechanics of Laminated Composite Plates and Shells, 2nd ed. CRC Press.
- Herakovich, C.T. (1998). Mechanics of Fibrous Composites. Wiley.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal

import numpy as np

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.core.transforms import reduced_stiffness_matrix, transform_reduced_stiffness


# ---------------------------------------------------------------------------
# LoadState
# ---------------------------------------------------------------------------

@dataclass
class LoadState:
    """Mechanical and environmental load state for a laminate.

    Mechanical resultants are expressed per unit width:
    - N_i : force resultants (N/mm)
    - M_i : moment resultants (N*mm/mm)
    - Q_i : transverse shear resultants (N/mm), used with FSDT

    Environmental loads:
    - delta_T : uniform temperature change from stress-free state (deg C)
    - delta_C : uniform moisture concentration change (%)

    Examples
    --------
    >>> load = LoadState(Nx=-1000.0)  # uniaxial compression
    >>> load.to_vector()
    array([-1000.,     0.,     0.,     0.,     0.,     0.])
    """

    # Mechanical resultants
    Nx: float = 0.0
    Ny: float = 0.0
    Nxy: float = 0.0
    Mx: float = 0.0
    My: float = 0.0
    Mxy: float = 0.0
    Qx: float = 0.0
    Qy: float = 0.0

    # Environmental
    delta_T: float = 0.0
    delta_C: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Return the 6-component CLT load vector [Nx, Ny, Nxy, Mx, My, Mxy].

        Transverse shear resultants (Qx, Qy) and environmental loads are
        not included in the CLT vector; they are handled separately.

        Returns
        -------
        np.ndarray
            Shape (6,) array of force and moment resultants.
        """
        return np.array([self.Nx, self.Ny, self.Nxy,
                         self.Mx, self.My, self.Mxy], dtype=float)

    @classmethod
    def from_vector(cls, vec: np.ndarray, **kwargs) -> LoadState:
        """Create a LoadState from a 6-component CLT vector.

        Parameters
        ----------
        vec : np.ndarray
            Shape (6,) array [Nx, Ny, Nxy, Mx, My, Mxy].
        **kwargs
            Additional keyword arguments passed to the constructor
            (e.g., Qx, Qy, delta_T, delta_C).

        Returns
        -------
        LoadState
        """
        vec = np.asarray(vec, dtype=float).ravel()
        if vec.size != 6:
            raise ValueError(f"Expected 6-component vector, got {vec.size}")
        return cls(
            Nx=float(vec[0]),
            Ny=float(vec[1]),
            Nxy=float(vec[2]),
            Mx=float(vec[3]),
            My=float(vec[4]),
            Mxy=float(vec[5]),
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Ply
# ---------------------------------------------------------------------------

@dataclass
class Ply:
    """A single ply in a composite laminate.

    Parameters
    ----------
    material : OrthotropicMaterial
        Ply material properties.
    angle : float
        Fiber orientation angle in degrees, measured from the laminate
        x-axis (positive counter-clockwise).
    thickness : float
        Ply thickness in mm.

    Examples
    --------
    >>> from wrinklefe.core.material import OrthotropicMaterial
    >>> mat = OrthotropicMaterial(E1=161000, E2=11380, G12=5170,
    ...                          nu12=0.32, G13=5170, G23=3980)
    >>> ply = Ply(mat, angle=45.0, thickness=0.183)
    >>> ply.angle_rad  # doctest: +ELLIPSIS
    0.7853...
    """

    material: OrthotropicMaterial
    angle: float
    thickness: float

    @property
    def angle_rad(self) -> float:
        """Fiber angle in radians."""
        return np.deg2rad(self.angle)

    def Q(self) -> np.ndarray:
        """Reduced stiffness matrix [Q] in material (1-2) coordinates.

        Returns
        -------
        np.ndarray
            3x3 matrix (MPa) relating in-plane stress to strain for a
            unidirectional ply under plane-stress conditions.
        """
        return reduced_stiffness_matrix(
            self.material.E1, self.material.E2,
            self.material.nu12, self.material.G12,
        )

    def Q_bar(self) -> np.ndarray:
        """Transformed reduced stiffness matrix [Q-bar] in laminate (x-y) coordinates.

        Returns
        -------
        np.ndarray
            3x3 matrix (MPa) accounting for ply orientation.
        """
        return transform_reduced_stiffness(self.Q(), self.angle_rad)

    def thermal_strain_global(self) -> np.ndarray:
        """Thermal expansion coefficients transformed to laminate coordinates.

        Returns the vector {alpha_x, alpha_y, alpha_xy} obtained by
        transforming the material CTE values {alpha_1, alpha_2, 0} through
        the ply angle.

        Returns
        -------
        np.ndarray
            Shape (3,) vector [alpha_x, alpha_y, alpha_xy] (1/deg C).
        """
        a1 = getattr(self.material, 'alpha1', 0.0)
        a2 = getattr(self.material, 'alpha2', 0.0)

        c = np.cos(self.angle_rad)
        s = np.sin(self.angle_rad)

        # Strain transformation (Reuter convention: engineering shear strain)
        alpha_x = a1 * c**2 + a2 * s**2
        alpha_y = a1 * s**2 + a2 * c**2
        alpha_xy = 2.0 * (a1 - a2) * s * c  # engineering shear

        return np.array([alpha_x, alpha_y, alpha_xy], dtype=float)


# ---------------------------------------------------------------------------
# Laminate
# ---------------------------------------------------------------------------

class Laminate:
    """Composite laminate analyzed using Classical Lamination Theory (CLT).

    The laminate is defined by an ordered list of :class:`Ply` objects,
    numbered from bottom (ply 0) to top (ply N-1). The reference plane
    (z = 0) is at the geometric midplane.

    Parameters
    ----------
    plies : list[Ply]
        Ordered list of plies from bottom to top.

    Attributes
    ----------
    plies : list[Ply]
        The ply stack.
    A : np.ndarray
        3x3 extensional stiffness matrix (N/mm).
    B : np.ndarray
        3x3 coupling stiffness matrix (N).
    D : np.ndarray
        3x3 bending stiffness matrix (N*mm).

    Examples
    --------
    >>> lam = Laminate.from_angles([0, 45, -45, 90, 90, -45, 45, 0],
    ...                            material=mat, ply_thickness=0.183)
    >>> lam.total_thickness
    1.464
    >>> lam.is_symmetric
    True
    """

    def __init__(self, plies: list[Ply]):
        if not plies:
            raise ValueError("Laminate must contain at least one ply.")
        self.plies = list(plies)
        self._compute_abd()

    # ----- Convenience constructors ----------------------------------------

    @classmethod
    def from_angles(
        cls,
        angles: list[float],
        material: OrthotropicMaterial,
        ply_thickness: float,
    ) -> Laminate:
        """Create a laminate from a list of ply angles.

        Parameters
        ----------
        angles : list[float]
            Ply orientations in degrees, bottom to top.
        material : OrthotropicMaterial
            Common material for all plies.
        ply_thickness : float
            Uniform ply thickness (mm).

        Returns
        -------
        Laminate
        """
        plies = [Ply(material=material, angle=a, thickness=ply_thickness)
                 for a in angles]
        return cls(plies)

    @classmethod
    def symmetric(
        cls,
        half_angles: list[float],
        material: OrthotropicMaterial,
        ply_thickness: float,
    ) -> Laminate:
        """Create a symmetric laminate from the bottom-half ply angles.

        The full stacking sequence is ``half_angles + reversed(half_angles)``.
        For example, ``[0, 45, 90]`` produces ``[0, 45, 90, 90, 45, 0]``.

        Parameters
        ----------
        half_angles : list[float]
            Bottom half of the laminate (excluding the plane of symmetry
            unless you want a repeated midplane ply).
        material : OrthotropicMaterial
            Common material for all plies.
        ply_thickness : float
            Uniform ply thickness (mm).

        Returns
        -------
        Laminate
        """
        full_angles = list(half_angles) + list(reversed(half_angles))
        return cls.from_angles(full_angles, material, ply_thickness)

    # ----- Geometry --------------------------------------------------------

    @property
    def total_thickness(self) -> float:
        """Total laminate thickness (mm)."""
        return sum(p.thickness for p in self.plies)

    @property
    def n_plies(self) -> int:
        """Number of plies."""
        return len(self.plies)

    def z_coords(self) -> np.ndarray:
        """Z-coordinates of ply boundaries from bottom to top.

        The coordinate system places z = 0 at the laminate midplane,
        so boundaries range from -h/2 to +h/2.

        Returns
        -------
        np.ndarray
            Shape (n_plies + 1,) array of z-coordinates (mm).
        """
        h = self.total_thickness
        z = np.empty(self.n_plies + 1, dtype=float)
        z[0] = -h / 2.0
        for k, ply in enumerate(self.plies):
            z[k + 1] = z[k] + ply.thickness
        return z

    def z_mid(self, i: int) -> float:
        """Z-coordinate of the midplane of ply *i*.

        Parameters
        ----------
        i : int
            Ply index (0-based, bottom to top).

        Returns
        -------
        float
            Midplane z-coordinate (mm).
        """
        zc = self.z_coords()
        return 0.5 * (zc[i] + zc[i + 1])

    @property
    def is_symmetric(self) -> bool:
        """Check whether the laminate is symmetric about the midplane.

        A laminate is symmetric if ply *k* from the bottom has the same
        material, angle, and thickness as ply *k* from the top.
        """
        n = self.n_plies
        for k in range(n // 2):
            p_bot = self.plies[k]
            p_top = self.plies[n - 1 - k]
            if (p_bot.material is not p_top.material
                    or not np.isclose(p_bot.angle, p_top.angle)
                    or not np.isclose(p_bot.thickness, p_top.thickness)):
                return False
        return True

    @property
    def is_balanced(self) -> bool:
        """Check whether the laminate is balanced.

        A laminate is balanced when for every +theta ply there exists a
        corresponding -theta ply of the same material and thickness.
        Plies at 0 and 90 degrees are self-balancing.
        """
        # Collect non-zero, non-90 angles and match +/- pairs
        unmatched: list[tuple[float, float, OrthotropicMaterial]] = []
        for p in self.plies:
            angle_mod = p.angle % 180.0
            if np.isclose(angle_mod, 0.0) or np.isclose(angle_mod, 90.0):
                continue
            # Try to find and remove a matching opposite angle
            target = (-p.angle) % 360.0
            found = False
            for j, (a, t, m) in enumerate(unmatched):
                a_norm = a % 360.0
                if (np.isclose(a_norm, target) or np.isclose(a_norm - 360, target - 360)
                        or np.isclose(a + p.angle, 0.0)
                        or np.isclose((a + p.angle) % 360, 0.0)):
                    if m is p.material and np.isclose(t, p.thickness):
                        unmatched.pop(j)
                        found = True
                        break
            if not found:
                unmatched.append((p.angle, p.thickness, p.material))
        return len(unmatched) == 0

    # ----- ABD matrix computation ------------------------------------------

    def _compute_abd(self) -> None:
        """Compute and cache A, B, D stiffness matrices."""
        A = np.zeros((3, 3), dtype=float)
        B = np.zeros((3, 3), dtype=float)
        D = np.zeros((3, 3), dtype=float)

        zc = self.z_coords()
        for k, ply in enumerate(self.plies):
            Qb = ply.Q_bar()
            z_bot = zc[k]
            z_top = zc[k + 1]

            dz1 = z_top - z_bot
            dz2 = z_top**2 - z_bot**2
            dz3 = z_top**3 - z_bot**3

            A += Qb * dz1
            B += Qb * (0.5 * dz2)
            D += Qb * (dz3 / 3.0)

        self.A = A
        self.B = B
        self.D = D

    def abd_matrix(self) -> np.ndarray:
        """Full 6x6 ABD stiffness matrix.

        .. math::

            \\begin{bmatrix} N \\\\ M \\end{bmatrix}
            = \\begin{bmatrix} A & B \\\\ B & D \\end{bmatrix}
            \\begin{bmatrix} \\varepsilon^0 \\\\ \\kappa \\end{bmatrix}

        Returns
        -------
        np.ndarray
            Shape (6, 6) stiffness matrix.
        """
        abd = np.zeros((6, 6), dtype=float)
        abd[0:3, 0:3] = self.A
        abd[0:3, 3:6] = self.B
        abd[3:6, 0:3] = self.B
        abd[3:6, 3:6] = self.D
        return abd

    def abd_inverse(self) -> np.ndarray:
        """Inverse of the 6x6 ABD matrix (compliance).

        Returns
        -------
        np.ndarray
            Shape (6, 6) compliance matrix [abd] = [ABD]^{-1}.
        """
        return np.linalg.inv(self.abd_matrix())

    # ----- FSDT transverse shear stiffness ---------------------------------

    @cached_property
    def H(self) -> np.ndarray:
        """2x2 transverse shear stiffness matrix (FSDT).

        Uses the shear correction factor kappa = 5/6.
        Each ply contributes its transverse shear moduli G13 and G23,
        transformed by the ply orientation angle.

        .. math::

            H_{ij} = \\kappa \\sum_k \\bar{G}_{ij,k} \\, (z_k - z_{k-1})

        Returns
        -------
        np.ndarray
            Shape (2, 2) matrix (N/mm).
        """
        kappa = 5.0 / 6.0
        H = np.zeros((2, 2), dtype=float)
        zc = self.z_coords()

        for k, ply in enumerate(self.plies):
            G13 = getattr(ply.material, 'G13', ply.material.G12)
            G23 = getattr(ply.material, 'G23', ply.material.G12)

            c = np.cos(ply.angle_rad)
            s = np.sin(ply.angle_rad)

            # Transform transverse shear moduli to global coordinates
            # [Gxz, Gyz] plane transformation:
            # G_bar = T_s * [[G13, 0], [0, G23]] * T_s^T
            # where T_s = [[c, s], [-s, c]]
            G_local = np.array([[G13, 0.0],
                                [0.0, G23]], dtype=float)
            T_s = np.array([[c, s],
                            [-s, c]], dtype=float)
            G_bar = T_s @ G_local @ T_s.T

            t_k = zc[k + 1] - zc[k]
            H += G_bar * t_k

        H *= kappa
        return H

    # ----- Stress / strain analysis ----------------------------------------

    def midplane_strains(self, load: LoadState) -> np.ndarray:
        """Compute midplane strains and curvatures from a load state.

        Solves the CLT constitutive equation:

        .. math::

            \\begin{bmatrix} \\varepsilon^0 \\\\ \\kappa \\end{bmatrix}
            = [ABD]^{-1} \\left(
              \\begin{bmatrix} N \\\\ M \\end{bmatrix}
              - \\begin{bmatrix} N^T \\\\ M^T \\end{bmatrix}
            \\right)

        Parameters
        ----------
        load : LoadState
            Applied loads and environmental conditions.

        Returns
        -------
        np.ndarray
            Shape (6,) vector [eps0_x, eps0_y, gamma0_xy,
            kappa_x, kappa_y, kappa_xy].
        """
        NM = load.to_vector()

        # Subtract thermal resultants if temperature change is present
        if not np.isclose(load.delta_T, 0.0):
            NT, MT = self.thermal_resultants(load.delta_T)
            NM[0:3] -= NT
            NM[3:6] -= MT

        return self.abd_inverse() @ NM

    def _ply_z(self, ply_idx: int, position: Literal['top', 'mid', 'bottom'] = 'mid') -> float:
        """Return the z-coordinate for a given position within a ply.

        Parameters
        ----------
        ply_idx : int
            Ply index (0-based).
        position : {'top', 'mid', 'bottom'}
            Location within the ply.

        Returns
        -------
        float
            Z-coordinate (mm).
        """
        zc = self.z_coords()
        z_bot = zc[ply_idx]
        z_top = zc[ply_idx + 1]
        if position == 'bottom':
            return z_bot
        elif position == 'top':
            return z_top
        else:
            return 0.5 * (z_bot + z_top)

    def ply_strains(
        self,
        load: LoadState,
        ply_idx: int,
        position: Literal['top', 'mid', 'bottom'] = 'mid',
    ) -> np.ndarray:
        """In-plane strains at a point in a ply (global coordinates).

        .. math::

            \\varepsilon(z) = \\varepsilon^0 + z \\, \\kappa

        Parameters
        ----------
        load : LoadState
            Applied load state.
        ply_idx : int
            Ply index (0-based, bottom to top).
        position : {'top', 'mid', 'bottom'}
            Through-thickness position within the ply.

        Returns
        -------
        np.ndarray
            Shape (3,) strain vector [eps_x, eps_y, gamma_xy].
        """
        mid = self.midplane_strains(load)
        eps0 = mid[0:3]
        kappa = mid[3:6]
        z = self._ply_z(ply_idx, position)
        return eps0 + z * kappa

    def ply_stresses_global(
        self,
        load: LoadState,
        ply_idx: int,
        position: Literal['top', 'mid', 'bottom'] = 'mid',
    ) -> np.ndarray:
        """In-plane stresses at a point in a ply (global/laminate coordinates).

        .. math::

            \\sigma = \\bar{Q} \\, \\varepsilon(z)

        Parameters
        ----------
        load : LoadState
            Applied load state.
        ply_idx : int
            Ply index.
        position : {'top', 'mid', 'bottom'}
            Through-thickness position.

        Returns
        -------
        np.ndarray
            Shape (3,) stress vector [sigma_x, sigma_y, tau_xy] (MPa).
        """
        eps = self.ply_strains(load, ply_idx, position)
        Qb = self.plies[ply_idx].Q_bar()
        return Qb @ eps

    def ply_stresses_local(
        self,
        load: LoadState,
        ply_idx: int,
        position: Literal['top', 'mid', 'bottom'] = 'mid',
    ) -> np.ndarray:
        """In-plane stresses at a point in a ply (local/material coordinates).

        Transforms global stresses into the ply material coordinate system
        using the stress transformation matrix.

        Parameters
        ----------
        load : LoadState
            Applied load state.
        ply_idx : int
            Ply index.
        position : {'top', 'mid', 'bottom'}
            Through-thickness position.

        Returns
        -------
        np.ndarray
            Shape (3,) stress vector [sigma_1, sigma_2, tau_12] (MPa).
        """
        sig_global = self.ply_stresses_global(load, ply_idx, position)
        theta = self.plies[ply_idx].angle_rad
        T = self._stress_transformation_matrix(theta)
        return T @ sig_global

    @staticmethod
    def _stress_transformation_matrix(theta: float) -> np.ndarray:
        """2D stress transformation matrix from global to local coordinates.

        Parameters
        ----------
        theta : float
            Ply angle in radians.

        Returns
        -------
        np.ndarray
            Shape (3, 3) transformation matrix T such that
            {sigma_1, sigma_2, tau_12} = T @ {sigma_x, sigma_y, tau_xy}.
        """
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([
            [c**2,      s**2,       2*s*c     ],
            [s**2,      c**2,      -2*s*c     ],
            [-s*c,      s*c,        c**2 - s**2],
        ], dtype=float)

    # ----- Thermal / moisture resultants -----------------------------------

    def thermal_resultants(self, delta_T: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute thermal force and moment resultants.

        .. math::

            N^T_i = \\Delta T \\sum_k \\bar{Q}_{ij,k} \\, \\alpha_{j,k} \\, t_k

            M^T_i = \\Delta T \\sum_k \\bar{Q}_{ij,k} \\, \\alpha_{j,k} \\, \\bar{z}_k \\, t_k

        Parameters
        ----------
        delta_T : float
            Temperature change from stress-free state (deg C).

        Returns
        -------
        NT : np.ndarray
            Shape (3,) thermal force resultant [N_Tx, N_Ty, N_Txy] (N/mm).
        MT : np.ndarray
            Shape (3,) thermal moment resultant [M_Tx, M_Ty, M_Txy] (N*mm/mm).
        """
        NT = np.zeros(3, dtype=float)
        MT = np.zeros(3, dtype=float)
        zc = self.z_coords()

        for k, ply in enumerate(self.plies):
            Qb = ply.Q_bar()
            alpha = ply.thermal_strain_global()
            t_k = zc[k + 1] - zc[k]
            z_mid_k = 0.5 * (zc[k] + zc[k + 1])

            Qa = Qb @ alpha
            NT += Qa * delta_T * t_k
            MT += Qa * delta_T * z_mid_k * t_k

        return NT, MT

    # ----- Engineering constants -------------------------------------------

    @cached_property
    def _a_matrix(self) -> np.ndarray:
        """Extensional compliance [a] = [A]^{-1} (mm/N)."""
        return np.linalg.inv(self.A)

    @cached_property
    def Ex(self) -> float:
        """Effective laminate Young's modulus in x-direction (MPa).

        Computed from the extensional compliance: E_x = 1 / (a_11 * h).
        """
        h = self.total_thickness
        return 1.0 / (self._a_matrix[0, 0] * h)

    @cached_property
    def Ey(self) -> float:
        """Effective laminate Young's modulus in y-direction (MPa)."""
        h = self.total_thickness
        return 1.0 / (self._a_matrix[1, 1] * h)

    @cached_property
    def Gxy(self) -> float:
        """Effective laminate shear modulus (MPa)."""
        h = self.total_thickness
        return 1.0 / (self._a_matrix[2, 2] * h)

    @cached_property
    def nu_xy(self) -> float:
        """Effective laminate Poisson's ratio nu_xy.

        Defined as nu_xy = -a_12 / a_11.
        """
        return -self._a_matrix[0, 1] / self._a_matrix[0, 0]

    # ----- String representation -------------------------------------------

    def __repr__(self) -> str:
        angles = [p.angle for p in self.plies]
        return (f"Laminate(n_plies={self.n_plies}, "
                f"h={self.total_thickness:.3f} mm, "
                f"angles={angles})")
