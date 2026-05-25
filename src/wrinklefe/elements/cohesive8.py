"""8-node zero-thickness cohesive interface element with bilinear traction-separation.

Geometry
--------
8 nodes arranged as 4 coincident node-pairs at a (possibly curved) interface
surface.  Bottom face (nodes 0-3) and top face (nodes 4-7) share planar
(x, y) positions in the reference configuration; the displacement jump
``delta = u_top - u_bottom`` is the kinematic quantity the element resists.

::

    Bottom face (z = 0, reference):  0 -- 1
                                     |    |
                                     3 -- 2

    Top face    (z = 0, reference):  4 -- 5    (initially coincident
                                     |    |    with the bottom face)
                                     7 -- 6

Bottom node ``i`` is paired with top node ``i + 4``.

Constitutive law
----------------
Intrinsic bilinear traction-separation:

- Penalty stiffness ``K`` (applied component-wise in the (n, s, t) frame).
- Effective opening
  ``delta_eff = sqrt(max(0, d_n)^2 + beta^2 * (d_s^2 + d_t^2))``
- Damage initiation ``delta_0 = sigma_max / K``.
- Final separation ``delta_f = 2 G_c / sigma_max``.
- Damage variable
  ``d = clamp(delta_f * (delta_eff - delta_0) / (delta_eff * (delta_f - delta_0)), 0, 1)``,
  monotonically non-decreasing.
- Traction in tension:   ``T_i = (1 - d) * K * delta_i``.
- Traction in compression (``delta_n < 0``): ``T_n = K * delta_n`` (penalty
  contact), no damage accumulation from compression.  Mode-II damage is
  also suppressed while the interface is in normal contact (Abaqus default).

Mode-mixity via Benzeggagh-Kenane:
``G_c(psi) = G_Ic + (G_IIc - G_Ic) * (G_II / (G_I + G_II))^eta``.
The mode ratio ``G_II / (G_I + G_II)`` is frozen at damage initiation so
that mixed-mode ``delta_0``, ``delta_f`` and ``Gc_mixed`` are path-
independent (Camanho & Davila, NASA/TM-2002-211737).

References
----------
Camanho, P.P. & Davila, C.G. (2002). Mixed-mode decohesion finite elements
    for the simulation of delamination in composite materials.
    NASA/TM-2002-211737.
Benzeggagh, M.L. & Kenane, M. (1996). Measurement of mixed-mode
    delamination fracture toughness of unidirectional glass/epoxy composites
    with mixed-mode bending apparatus.  Composites Science and Technology
    56(4), 439-449.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wrinklefe.elements.gauss import gauss_points_1d

# ----------------------------------------------------------------------
# Material parameter container
# ----------------------------------------------------------------------

@dataclass
class CohesiveProperties:
    """Parameters of the intrinsic bilinear traction-separation law.

    Parameters
    ----------
    K : float
        Initial penalty stiffness (applied to all three components, v1).
    sigma_max : float
        Mode-I peak normal traction.
    tau_max : float
        Mode-II peak shear traction.
    GIc : float
        Mode-I fracture toughness.
    GIIc : float
        Mode-II fracture toughness.
    eta_BK : float
        Benzeggagh-Kenane exponent (default 1.45 for epoxy composites).
    beta : float
        Shear weighting in the effective opening (default 1).
    """

    K: float = 1.0e5
    sigma_max: float = 50.0
    tau_max: float = 80.0
    GIc: float = 0.5
    GIIc: float = 1.5
    eta_BK: float = 1.45
    beta: float = 1.0


# ----------------------------------------------------------------------
# Per Gauss-point state
# ----------------------------------------------------------------------

@dataclass
class CohesiveState:
    """History variables tracked per Gauss point per element.

    ``mode_ratio_init`` is the BK mode ratio G_II/(G_I + G_II) captured at
    the instant damage first initiated at this Gauss point.  The sentinel
    value ``-1.0`` means "not yet initiated"; once set, it is held fixed
    for the rest of the analysis so the mixed-mode envelope is path-
    independent (Camanho & Davila convention).
    """

    d: float = 0.0
    mode_ratio_init: float = -1.0


def make_initial_state(n_gp: int = 4) -> list[CohesiveState]:
    return [CohesiveState() for _ in range(n_gp)]


# ----------------------------------------------------------------------
# 2x2 Gauss-quadrature on the [-1, 1]^2 reference quad
# ----------------------------------------------------------------------

def _gauss_points_quad(order: int = 2) -> tuple[np.ndarray, np.ndarray]:
    pts_1d, wts_1d = gauss_points_1d(order)
    xi, eta = np.meshgrid(pts_1d, pts_1d, indexing="ij")
    wi, wj = np.meshgrid(wts_1d, wts_1d, indexing="ij")
    points = np.column_stack([xi.ravel(), eta.ravel()])
    weights = (wi * wj).ravel()
    return points, weights


# ----------------------------------------------------------------------
# Bilinear quad shape functions on the 4 bottom nodes (0..3)
# ----------------------------------------------------------------------

# Reference quad nodal coordinates matching the bottom face ordering
# (0 at (-1,-1), 1 at (+1,-1), 2 at (+1,+1), 3 at (-1,+1)).
_QUAD_NODE_COORDS = np.array([
    [-1.0, -1.0],
    [+1.0, -1.0],
    [+1.0, +1.0],
    [-1.0, +1.0],
], dtype=float)


def _quad_shape_functions(xi: float, eta: float) -> np.ndarray:
    N = np.empty(4)
    for i in range(4):
        xi_i, eta_i = _QUAD_NODE_COORDS[i]
        N[i] = 0.25 * (1.0 + xi_i * xi) * (1.0 + eta_i * eta)
    return N


def _quad_shape_derivatives(xi: float, eta: float) -> np.ndarray:
    dN = np.empty((2, 4))
    for i in range(4):
        xi_i, eta_i = _QUAD_NODE_COORDS[i]
        dN[0, i] = 0.25 * xi_i * (1.0 + eta_i * eta)
        dN[1, i] = 0.25 * (1.0 + xi_i * xi) * eta_i
    return dN


# ----------------------------------------------------------------------
# Cohesive element
# ----------------------------------------------------------------------

class Cohesive8Element:
    """8-node zero-thickness cohesive interface element.

    Parameters
    ----------
    node_coords : np.ndarray
        Shape ``(8, 3)`` — reference-config coordinates.  Rows 0-3 are
        the bottom face nodes; rows 4-7 are the top face nodes, paired
        index-by-index with the bottom (node 4 sits on node 0, etc.).
    properties : CohesiveProperties
        Material parameters for the bilinear law.
    node_ids : np.ndarray or None, optional
        Shape ``(8,)`` global node indices in the same order as
        ``node_coords``.  Required by :class:`GlobalAssembler` to build
        the element DOF map; may be omitted for unit-tests of the law
        in isolation.
    elem_id : int or None, optional
        Optional identifier used in error messages.
    """

    def __init__(
        self,
        node_coords: np.ndarray,
        properties: CohesiveProperties,
        node_ids: np.ndarray | None = None,
        elem_id: int | None = None,
    ) -> None:
        self.node_coords = np.asarray(node_coords, dtype=float)
        if self.node_coords.shape != (8, 3):
            raise ValueError(
                "node_coords must have shape (8, 3), got "
                f"{self.node_coords.shape}."
            )
        self.properties = properties
        if node_ids is not None:
            node_ids = np.asarray(node_ids, dtype=np.intp).reshape(-1)
            if node_ids.shape != (8,):
                raise ValueError(
                    f"node_ids must be length 8, got {node_ids.shape}."
                )
            if np.any(node_ids < 0):
                raise ValueError(
                    f"node_ids must be non-negative integers; got values "
                    f"< 0 at positions "
                    f"{np.where(node_ids < 0)[0].tolist()}"
                )
        self.node_ids = node_ids
        self.elem_id = elem_id

        self._gp_points, self._gp_weights = _gauss_points_quad(order=2)
        self._n_gp = self._gp_points.shape[0]

        # Pre-compute local frame and 2D detJ at each Gauss point from the
        # bottom-face reference geometry — these are constant for the
        # life of the element (small-strain assumption).
        self._R_gp, self._detJ_gp = self._precompute_local_frames()

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @property
    def _bottom_coords(self) -> np.ndarray:
        return self.node_coords[:4]

    def _precompute_local_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """Local (n, s, t) frame and 2D Jacobian determinant at each GP.

        Built from the **bottom face** reference-config nodes.  The normal
        ``n`` points from bottom into top (right-hand rule on the s, t
        in-plane edges).  ``s`` is aligned with d/dxi at the mid-surface,
        ``t`` is chosen orthogonal to ``s`` and ``n`` to make (n, s, t)
        right-handed.
        """
        bottom = self._bottom_coords  # (4, 3)
        R_gp = np.empty((self._n_gp, 3, 3))
        detJ_gp = np.empty(self._n_gp)

        for g in range(self._n_gp):
            xi, eta = self._gp_points[g]
            dN = _quad_shape_derivatives(xi, eta)  # (2, 4)
            # In-plane tangents in physical coords.
            dx_dxi = dN[0] @ bottom   # (3,)
            dx_deta = dN[1] @ bottom  # (3,)

            # Normal direction (bottom -> top), unit-length.
            n_vec = np.cross(dx_dxi, dx_deta)
            n_norm = np.linalg.norm(n_vec)
            if n_norm < 1.0e-14:
                raise ValueError(
                    f"Cohesive element {self.elem_id}: degenerate face "
                    f"at gauss point {g}; cross product norm = {n_norm:.3e}."
                )
            n_hat = n_vec / n_norm

            # Tangent s: align with dxi direction projected into the
            # tangent plane.
            s_vec = dx_dxi - np.dot(dx_dxi, n_hat) * n_hat
            s_hat = s_vec / np.linalg.norm(s_vec)

            # t completes the right-handed frame.
            t_hat = np.cross(n_hat, s_hat)

            R = np.vstack([n_hat, s_hat, t_hat])  # rows are basis vectors
            R_gp[g] = R

            # 2D Jacobian on the mid-surface: ||dx_dxi x dx_deta||.
            detJ_gp[g] = n_norm

        return R_gp, detJ_gp

    # ------------------------------------------------------------------
    # Displacement jump B-matrix
    # ------------------------------------------------------------------

    def _B_jump(self, gp: int) -> np.ndarray:
        """(3, 24) matrix mapping element DOFs to the global-frame jump
        ``delta_global = B_jump @ u_elem`` at Gauss point ``gp``."""
        xi, eta = self._gp_points[gp]
        N = _quad_shape_functions(xi, eta)  # (4,)
        B = np.zeros((3, 24))
        for i in range(4):
            # Bottom node i (DOF cols 3i .. 3i+2): contribution -N_i * I.
            B[0, 3 * i + 0] = -N[i]
            B[1, 3 * i + 1] = -N[i]
            B[2, 3 * i + 2] = -N[i]
            # Top node i+4 (DOF cols 3(i+4) .. 3(i+4)+2): +N_i * I.
            j = i + 4
            B[0, 3 * j + 0] = N[i]
            B[1, 3 * j + 1] = N[i]
            B[2, 3 * j + 2] = N[i]
        return B

    # ------------------------------------------------------------------
    # Constitutive law (in local n, s, t frame)
    # ------------------------------------------------------------------

    def _law_local(
        self,
        delta_local: np.ndarray,
        state_prev: CohesiveState,
    ) -> tuple[np.ndarray, np.ndarray, CohesiveState]:
        """Bilinear intrinsic CZM with BK mixed-mode in the local frame.

        Returns
        -------
        T_local : np.ndarray, shape (3,)
        D_local : np.ndarray, shape (3, 3) — material tangent ``dT/ddelta``.
        state_new : CohesiveState
        """
        p = self.properties
        K = p.K
        beta = p.beta

        d_n = float(delta_local[0])
        d_s = float(delta_local[1])
        d_t = float(delta_local[2])

        # Compression branch: no damage growth, contact penalty in the
        # normal direction, secant in shear using previously committed
        # damage.  Handle this as a structural early return so the
        # mode-ratio math below is only ever evaluated when there is
        # actual opening (d_n >= 0); this avoids a fragile dependence
        # on a downstream branch to suppress an ill-defined mode_ratio
        # under closed contact (matches the docstring and Abaqus default).
        if d_n < 0.0:
            d = state_prev.d
            T_local = np.array(
                [K * d_n, (1.0 - d) * K * d_s, (1.0 - d) * K * d_t]
            )
            D_local = np.diag([K, (1.0 - d) * K, (1.0 - d) * K])
            state_new = CohesiveState(
                d=d, mode_ratio_init=state_prev.mode_ratio_init
            )
            return T_local, D_local, state_new

        # Opening branch (d_n >= 0): the mode-ratio math is well-defined.
        d_n_pos = d_n
        shear_sq = d_s * d_s + d_t * d_t
        delta_eff = float(np.sqrt(d_n_pos * d_n_pos + (beta * beta) * shear_sq))

        # Helper: evaluate (Gc_mixed, sigma_mixed, delta_0, delta_f) at a
        # given BK mode ratio.  Used to test initiation against the current
        # ratio and again with the frozen ratio after initiation.
        def _bk_at(mode_ratio: float) -> tuple[float, float, float, float]:
            Gc_m = p.GIc + (p.GIIc - p.GIc) * (mode_ratio ** p.eta_BK)
            sigma_m_sq = (
                (p.sigma_max ** 2)
                + ((p.tau_max ** 2) - (p.sigma_max ** 2))
                * (mode_ratio ** p.eta_BK)
            )
            sigma_m = float(np.sqrt(max(sigma_m_sq, 1.0e-30)))
            return Gc_m, sigma_m, sigma_m / K, 2.0 * Gc_m / sigma_m

        # Current mode ratio from the opening (BK ratio of normal energy
        # to total).  Only valid when there is some opening to take a ratio
        # of; otherwise default to pure mode I.
        G_I = 0.5 * K * d_n_pos * d_n_pos
        G_II = 0.5 * K * shear_sq * (beta * beta)
        G_total = G_I + G_II
        mode_ratio_current = (G_II / G_total) if G_total > 0.0 else 0.0

        # Pick the mode ratio used to evaluate the bilinear envelope.
        # Frozen at initiation per Camanho & Davila so that non-radial
        # paths cannot drift along the envelope.
        mode_ratio_init_new = state_prev.mode_ratio_init
        if state_prev.mode_ratio_init >= 0.0:
            mode_ratio_use = state_prev.mode_ratio_init
        else:
            mode_ratio_use = mode_ratio_current

        _Gc_mixed, _sigma_mixed, delta_0, delta_f = _bk_at(mode_ratio_use)

        if delta_eff <= delta_0:
            d_new = state_prev.d
        elif delta_eff >= delta_f:
            d_new = 1.0
            if state_prev.mode_ratio_init < 0.0:
                mode_ratio_init_new = mode_ratio_current
        else:
            d_trial = (
                (delta_f * (delta_eff - delta_0))
                / (delta_eff * (delta_f - delta_0))
            )
            d_new = max(state_prev.d, min(d_trial, 1.0))
            if state_prev.mode_ratio_init < 0.0 and d_new > 0.0:
                # Damage just initiated this call; freeze the current mode
                # ratio.  delta_0/delta_f from the helper above were already
                # evaluated at this same mode_ratio (since state_prev had
                # no frozen ratio), so no re-evaluation is needed.
                mode_ratio_init_new = mode_ratio_current

        d = d_new

        # Traction (opening branch).
        T_local = np.array(
            [(1.0 - d) * K * d_n, (1.0 - d) * K * d_s, (1.0 - d) * K * d_t]
        )

        # Consistent tangent — secant for the damaged components.
        D_local = np.diag([(1.0 - d) * K, (1.0 - d) * K, (1.0 - d) * K])

        state_new = CohesiveState(
            d=d,
            mode_ratio_init=mode_ratio_init_new,
        )
        return T_local, D_local, state_new

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tangent_and_force(
        self,
        u_element: np.ndarray,
        state_prev: list[CohesiveState],
    ) -> tuple[np.ndarray, np.ndarray, list[CohesiveState]]:
        """Compute the (24, 24) tangent and (24,) internal force vector.

        Parameters
        ----------
        u_element : np.ndarray
            Shape ``(24,)`` element nodal displacement.
        state_prev : list[CohesiveState]
            Length-``n_gp`` history at the start of the increment.

        Returns
        -------
        K_t : np.ndarray, shape (24, 24)
        F_int : np.ndarray, shape (24,)
        state_new : list[CohesiveState]
        """
        u_element = np.asarray(u_element, dtype=float).reshape(24)

        K_t = np.zeros((24, 24))
        F_int = np.zeros(24)
        state_new: list[CohesiveState] = []

        for g in range(self._n_gp):
            B = self._B_jump(g)            # (3, 24)
            R = self._R_gp[g]              # (3, 3) — local frame rows = (n,s,t)
            detJ = self._detJ_gp[g]
            w = self._gp_weights[g]

            delta_global = B @ u_element     # (3,)
            delta_local = R @ delta_global   # (3,)

            T_local, D_local, st_new = self._law_local(delta_local, state_prev[g])
            T_global = R.T @ T_local             # (3,)
            D_global = R.T @ D_local @ R        # (3, 3)

            F_int += B.T @ T_global * detJ * w
            K_t += B.T @ D_global @ B * detJ * w

            state_new.append(st_new)

        return K_t, F_int, state_new

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def area(self) -> float:
        """Mid-surface area via Gauss quadrature on the bottom face."""
        return float(np.sum(self._detJ_gp * self._gp_weights))

    @property
    def n_gp(self) -> int:
        """Number of Gauss points (4 for the 2x2 rule on the mid-surface)."""
        return self._n_gp
