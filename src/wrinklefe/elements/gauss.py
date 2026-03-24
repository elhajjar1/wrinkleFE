"""Gauss-Legendre quadrature rules for hexahedral finite elements.

Provides exact quadrature points and weights for 1-D intervals and their
tensor-product extensions to 3-D hexahedral domains.  All points and weights
are hard-coded from analytical expressions (no dependency on scipy).

Supported orders:
    - n=1: 1-point rule (exact for polynomials up to degree 1)
    - n=2: 2-point rule (exact for polynomials up to degree 3) — standard for hex8
    - n=3: 3-point rule (exact for polynomials up to degree 5) — for hex20/hex27

References
----------
Abramowitz, M. & Stegun, I.A. (1964). Handbook of Mathematical Functions,
    Table 25.4.
"""

from __future__ import annotations

import numpy as np


def gauss_points_1d(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Legendre quadrature points and weights on [-1, 1].

    Uses exact analytical values (not numerical root-finding).

    Parameters
    ----------
    n : int
        Number of quadrature points.  Must be 1, 2, or 3.

    Returns
    -------
    points : np.ndarray
        Shape ``(n,)`` — quadrature point coordinates in [-1, 1].
    weights : np.ndarray
        Shape ``(n,)`` — corresponding quadrature weights.

    Raises
    ------
    ValueError
        If *n* is not 1, 2, or 3.

    Examples
    --------
    >>> pts, wts = gauss_points_1d(2)
    >>> pts  # array([-1/sqrt(3), 1/sqrt(3)])
    >>> wts  # array([1.0, 1.0])
    """
    if n == 1:
        points = np.array([0.0])
        weights = np.array([2.0])
    elif n == 2:
        g = 1.0 / np.sqrt(3.0)
        points = np.array([-g, g])
        weights = np.array([1.0, 1.0])
    elif n == 3:
        g = np.sqrt(3.0 / 5.0)
        points = np.array([-g, 0.0, g])
        weights = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    else:
        raise ValueError(
            f"Only n=1, 2, 3 are supported for Gauss-Legendre quadrature, got n={n}."
        )
    return points, weights


def gauss_points_hex(order: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return 3-D Gauss-Legendre quadrature points and weights for a hexahedron.

    The hexahedral domain is the reference cube [-1, 1]^3.  Points and weights
    are formed as the tensor product of 1-D Gauss-Legendre rules:

        (xi_i, eta_j, zeta_k),   w_ijk = w_i * w_j * w_k

    Parameters
    ----------
    order : int, optional
        Number of quadrature points along each axis.  Default is 2 (giving
        2x2x2 = 8 points, standard for linear hex8 elements).  Use 3 for
        3x3x3 = 27 points (quadratic hex20/hex27 elements).

    Returns
    -------
    points : np.ndarray
        Shape ``(n_points, 3)`` — (xi, eta, zeta) natural coordinates.
    weights : np.ndarray
        Shape ``(n_points,)`` — quadrature weights.

    Raises
    ------
    ValueError
        If *order* is not 1, 2, or 3.

    Examples
    --------
    >>> pts, wts = gauss_points_hex(order=2)
    >>> pts.shape
    (8, 3)
    >>> wts.shape
    (8,)
    >>> np.isclose(wts.sum(), 8.0)  # volume of reference cube
    True
    """
    pts_1d, wts_1d = gauss_points_1d(order)

    # Tensor product via meshgrid
    xi, eta, zeta = np.meshgrid(pts_1d, pts_1d, pts_1d, indexing="ij")
    wi, wj, wk = np.meshgrid(wts_1d, wts_1d, wts_1d, indexing="ij")

    points = np.column_stack([xi.ravel(), eta.ravel(), zeta.ravel()])
    weights = (wi * wj * wk).ravel()

    return points, weights
