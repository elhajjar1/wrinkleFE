"""
Coordinate transformation utilities for 3D composite mechanics.

Provides rotation matrices and stress/strain transformations for:
- Ply orientation: rotation about the z-axis (through-thickness direction)
- Fiber waviness (wrinkle misalignment): rotation about the y-axis (in the x-z plane)

All angles are specified in radians. Stress and strain use Voigt notation:
    stress: [sigma_11, sigma_22, sigma_33, tau_23, tau_13, tau_12]
    strain: [eps_11, eps_22, eps_33, gamma_23, gamma_13, gamma_12]
"""

import numpy as np


def rotation_matrix_3d(angle_rad: float, axis: str = 'z') -> np.ndarray:
    """
    Construct a 3x3 rotation matrix for rotation about a principal axis.

    For axis='z' (ply orientation), rotation by angle theta:

        R_z = [[ cos(theta), sin(theta), 0],
               [-sin(theta), cos(theta), 0],
               [     0,          0,      1]]

    For axis='y' (wrinkle misalignment in the x-z plane), rotation by angle phi:

        R_y = [[cos(phi), 0, -sin(phi)],
               [   0,     1,     0    ],
               [sin(phi), 0,  cos(phi)]]

    Parameters
    ----------
    angle_rad : float
        Rotation angle in radians.
    axis : str, optional
        Axis of rotation: 'z' for ply orientation, 'y' for wrinkle
        misalignment. Default is 'z'.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.

    Raises
    ------
    ValueError
        If axis is not 'y' or 'z'.
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    if axis == 'z':
        return np.array([
            [ c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ])
    elif axis == 'y':
        return np.array([
            [c, 0.0, -s],
            [0.0, 1.0, 0.0],
            [s, 0.0,  c],
        ])
    else:
        raise ValueError(f"Unsupported axis '{axis}'. Use 'z' or 'y'.")


def stress_transformation_3d(angle_rad: float, axis: str = 'z') -> np.ndarray:
    """
    Construct the 6x6 stress transformation matrix in Voigt notation.

    Transforms stress components from global to rotated coordinates:

        {sigma'} = [T_sigma] {sigma}

    where stress vectors use the ordering:
        [sigma_11, sigma_22, sigma_33, tau_23, tau_13, tau_12]

    For rotation about z by angle theta (c = cos(theta), s = sin(theta)):

        T_sigma = [[ c^2,  s^2, 0,  0,  0,  2sc ],
                   [ s^2,  c^2, 0,  0,  0, -2sc ],
                   [  0,    0,  1,  0,  0,   0  ],
                   [  0,    0,  0,  c, -s,   0  ],
                   [  0,    0,  0,  s,  c,   0  ],
                   [ -sc,  sc,  0,  0,  0, c^2-s^2]]

    For rotation about y by angle phi (c = cos(phi), s = sin(phi)):

        T_sigma = [[ c^2, 0,  s^2, 0, -2sc, 0],
                   [  0,  1,   0,  0,   0,  0],
                   [ s^2, 0,  c^2, 0,  2sc, 0],
                   [  0,  0,   0,  c,   0,  s],
                   [ sc,  0, -sc,  0, c^2-s^2, 0],
                   [  0,  0,   0, -s,   0,  c]]

    Parameters
    ----------
    angle_rad : float
        Rotation angle in radians.
    axis : str, optional
        Axis of rotation: 'z' for ply orientation, 'y' for wrinkle
        misalignment. Default is 'z'.

    Returns
    -------
    np.ndarray
        6x6 stress transformation matrix.

    Raises
    ------
    ValueError
        If axis is not 'y' or 'z'.
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    c2 = c * c
    s2 = s * s
    sc = s * c

    if axis == 'z':
        # Voigt: [11, 22, 33, 23, 13, 12]
        # Rotation about z (3-axis): 1-2 plane rotates, 3 unchanged
        return np.array([
            [ c2,   s2,  0.0,  0.0,  0.0,  2.0 * sc],
            [ s2,   c2,  0.0,  0.0,  0.0, -2.0 * sc],
            [0.0,  0.0,  1.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,    c,   -s,  0.0],
            [0.0,  0.0,  0.0,    s,    c,  0.0],
            [-sc,   sc,  0.0,  0.0,  0.0,  c2 - s2],
        ])
    elif axis == 'y':
        # Rotation about y (2-axis): 1-3 plane rotates, 2 unchanged
        return np.array([
            [ c2,  0.0,  s2,  0.0, -2.0 * sc, 0.0],
            [0.0,  1.0, 0.0,  0.0,  0.0,      0.0],
            [ s2,  0.0,  c2,  0.0,  2.0 * sc,  0.0],
            [0.0,  0.0, 0.0,    c,  0.0,         s],
            [ sc,  0.0, -sc,  0.0,  c2 - s2,   0.0],
            [0.0,  0.0, 0.0,   -s,  0.0,         c],
        ])
    else:
        raise ValueError(f"Unsupported axis '{axis}'. Use 'z' or 'y'.")


def strain_transformation_3d(angle_rad: float, axis: str = 'z') -> np.ndarray:
    """
    Construct the 6x6 engineering strain transformation matrix.

    Transforms engineering strain components from global to rotated coordinates:

        {epsilon'} = [T_epsilon] {epsilon}

    where strain vectors use the ordering:
        [eps_11, eps_22, eps_33, gamma_23, gamma_13, gamma_12]

    The strain transformation is related to the stress transformation
    through the Reuter matrix:

        T_epsilon = R * T_sigma * R^{-1}

    where R = diag(1, 1, 1, 2, 2, 2) converts engineering shear strains
    to tensor shear strains, and R^{-1} = diag(1, 1, 1, 0.5, 0.5, 0.5).

    This ensures consistency between stress and engineering strain
    transformations: sigma' = T_sigma * sigma and C' = T_sigma^{-1} * C * T_epsilon.

    Parameters
    ----------
    angle_rad : float
        Rotation angle in radians.
    axis : str, optional
        Axis of rotation: 'z' for ply orientation, 'y' for wrinkle
        misalignment. Default is 'z'.

    Returns
    -------
    np.ndarray
        6x6 engineering strain transformation matrix.

    Raises
    ------
    ValueError
        If axis is not 'y' or 'z'.
    """
    T_sigma = stress_transformation_3d(angle_rad, axis=axis)

    # Reuter matrix: converts engineering strain to tensor strain
    R = np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    R_inv = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])

    return R @ T_sigma @ R_inv


def rotate_stiffness_3d(
    C: np.ndarray, angle_rad: float, axis: str = 'z'
) -> np.ndarray:
    """
    Rotate a 6x6 stiffness matrix to a new coordinate system.

    The rotated stiffness is computed as:

        C_bar = T_sigma^{-1} * C * T_epsilon

    where T_sigma is the stress transformation matrix and T_epsilon is the
    engineering strain transformation matrix. This relationship ensures that
    the constitutive relation sigma = C * epsilon holds in both the original
    and rotated frames.

    Parameters
    ----------
    C : np.ndarray
        6x6 stiffness matrix in the original (material) coordinate system.
    angle_rad : float
        Rotation angle in radians.
    axis : str, optional
        Axis of rotation: 'z' for ply orientation, 'y' for wrinkle
        misalignment. Default is 'z'.

    Returns
    -------
    np.ndarray
        6x6 rotated stiffness matrix.

    Raises
    ------
    ValueError
        If C is not a 6x6 array or if axis is not 'y' or 'z'.
    """
    C = np.asarray(C, dtype=float)
    if C.shape != (6, 6):
        raise ValueError(f"Stiffness matrix must be 6x6, got {C.shape}.")

    T_sigma = stress_transformation_3d(angle_rad, axis=axis)
    T_epsilon = strain_transformation_3d(angle_rad, axis=axis)
    T_sigma_inv = np.linalg.inv(T_sigma)

    return T_sigma_inv @ C @ T_epsilon


def reduced_stiffness_matrix(
    E1: float, E2: float, nu12: float, G12: float
) -> np.ndarray:
    """
    Compute the 3x3 reduced stiffness matrix [Q] for plane stress.

    For a unidirectional ply under plane stress (sigma_33 = tau_23 = tau_13 = 0),
    the in-plane constitutive relation is:

        {sigma_11}     [Q11  Q12   0 ] {eps_11}
        {sigma_22}  =  [Q12  Q22   0 ] {eps_22}
        {tau_12  }     [ 0    0   Q66] {gamma_12}

    where:
        nu21 = nu12 * E2 / E1
        denom = 1 - nu12 * nu21
        Q11 = E1 / denom
        Q12 = nu12 * E2 / denom
        Q22 = E2 / denom
        Q66 = G12

    Parameters
    ----------
    E1 : float
        Longitudinal Young's modulus (fiber direction).
    E2 : float
        Transverse Young's modulus.
    nu12 : float
        Major Poisson's ratio.
    G12 : float
        In-plane shear modulus.

    Returns
    -------
    np.ndarray
        3x3 reduced stiffness matrix [Q].
    """
    nu21 = nu12 * E2 / E1
    denom = 1.0 - nu12 * nu21

    Q11 = E1 / denom
    Q12 = nu12 * E2 / denom
    Q22 = E2 / denom
    Q66 = G12

    return np.array([
        [Q11, Q12, 0.0],
        [Q12, Q22, 0.0],
        [0.0, 0.0, Q66],
    ])


def transform_reduced_stiffness(
    Q: np.ndarray, angle_rad: float
) -> np.ndarray:
    """
    Compute the transformed reduced stiffness matrix [Q-bar] for a ply at angle theta.

    Used in Classical Lamination Theory (CLT) for computing the ABD matrices.
    Given the on-axis reduced stiffness [Q] and a ply orientation angle theta,
    the transformed components are (c = cos(theta), s = sin(theta)):

        Q_bar_11 = Q11*c^4 + 2*(Q12 + 2*Q66)*s^2*c^2 + Q22*s^4
        Q_bar_12 = (Q11 + Q22 - 4*Q66)*s^2*c^2 + Q12*(s^4 + c^4)
        Q_bar_22 = Q11*s^4 + 2*(Q12 + 2*Q66)*s^2*c^2 + Q22*c^4
        Q_bar_16 = (Q11 - Q12 - 2*Q66)*s*c^3 + (Q12 - Q22 + 2*Q66)*s^3*c
        Q_bar_26 = (Q11 - Q12 - 2*Q66)*s^3*c + (Q12 - Q22 + 2*Q66)*s*c^3
        Q_bar_66 = (Q11 + Q22 - 2*Q12 - 2*Q66)*s^2*c^2 + Q66*(s^4 + c^4)

    Parameters
    ----------
    Q : np.ndarray
        3x3 on-axis reduced stiffness matrix from ``reduced_stiffness_matrix``.
    angle_rad : float
        Ply orientation angle in radians (measured from the reference axis).

    Returns
    -------
    np.ndarray
        3x3 transformed reduced stiffness matrix [Q-bar].

    Raises
    ------
    ValueError
        If Q is not a 3x3 array.
    """
    Q = np.asarray(Q, dtype=float)
    if Q.shape != (3, 3):
        raise ValueError(f"Reduced stiffness matrix must be 3x3, got {Q.shape}.")

    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    c2 = c * c
    s2 = s * s
    c4 = c2 * c2
    s4 = s2 * s2
    s2c2 = s2 * c2
    sc3 = s * c * c2
    s3c = s * s2 * c

    Q11 = Q[0, 0]
    Q12 = Q[0, 1]
    Q22 = Q[1, 1]
    Q66 = Q[2, 2]

    Qb11 = Q11 * c4 + 2.0 * (Q12 + 2.0 * Q66) * s2c2 + Q22 * s4
    Qb12 = (Q11 + Q22 - 4.0 * Q66) * s2c2 + Q12 * (s4 + c4)
    Qb22 = Q11 * s4 + 2.0 * (Q12 + 2.0 * Q66) * s2c2 + Q22 * c4
    Qb16 = (Q11 - Q12 - 2.0 * Q66) * sc3 + (Q12 - Q22 + 2.0 * Q66) * s3c
    Qb26 = (Q11 - Q12 - 2.0 * Q66) * s3c + (Q12 - Q22 + 2.0 * Q66) * sc3
    Qb66 = (Q11 + Q22 - 2.0 * Q12 - 2.0 * Q66) * s2c2 + Q66 * (s4 + c4)

    return np.array([
        [Qb11, Qb12, Qb16],
        [Qb12, Qb22, Qb26],
        [Qb16, Qb26, Qb66],
    ])
