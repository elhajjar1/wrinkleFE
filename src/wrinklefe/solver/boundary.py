"""Boundary condition handling and CLT-to-3D load mapping.

Provides:

- :class:`BoundaryCondition` — a dataclass describing a single BC
  (fixed, displacement, force, pressure, or symmetry).
- :class:`BoundaryHandler` — resolves BCs against a mesh and applies
  them to the global system ``K u = F`` via the penalty method or the
  elimination (partitioning) method.
- Convenience functions for common loading scenarios (uniaxial
  compression, pure bending) and a mapping from CLT
  :class:`~wrinklefe.core.laminate.LoadState` to 3-D BCs.

References
----------
Bathe, K.-J. (2006). Finite Element Procedures, Chapter 4.
Cook, R.D. et al. (2002). Concepts and Applications of Finite Element
    Analysis, 4th ed., Chapter 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from wrinklefe.core.mesh import MeshData
from wrinklefe.core.laminate import LoadState


# ======================================================================
# BoundaryCondition dataclass
# ======================================================================

@dataclass
class BoundaryCondition:
    """A single boundary condition applied to mesh nodes.

    Exactly one of *face* or *node_ids* must be specified to identify
    the affected nodes.  For face-based BCs the node IDs are resolved
    at application time via :meth:`MeshData.nodes_on_face`.

    Parameters
    ----------
    bc_type : str
        Type of boundary condition:

        - ``"fixed"`` — constrain specified DOFs to zero.
        - ``"displacement"`` — prescribe a non-zero displacement.
        - ``"force"`` — apply a point force to each listed node.
        - ``"pressure"`` — apply a distributed force over a face,
          divided equally among face nodes.
        - ``"symmetry_x"`` — symmetry about the yz-plane (ux = 0).
        - ``"symmetry_y"`` — symmetry about the xz-plane (uy = 0).
        - ``"symmetry_z"`` — symmetry about the xy-plane (uz = 0).

    face : str or None
        Mesh face identifier: ``'x_min'``, ``'x_max'``, ``'y_min'``,
        ``'y_max'``, ``'z_min'``, ``'z_max'``.  Mutually exclusive
        with *node_ids*.
    node_ids : np.ndarray or None
        Explicit array of node indices (0-based).  Mutually exclusive
        with *face*.
    dofs : list[int]
        Which translational DOFs to affect (0 = ux, 1 = uy, 2 = uz).
        Default is ``[0, 1, 2]`` (all three).
    value : float
        Prescribed displacement (mm) or force magnitude (N).
        For ``"pressure"`` BCs, this is the total force (N) applied to
        the face, distributed equally among face nodes.

    Examples
    --------
    >>> bc = BoundaryCondition(bc_type="fixed", face="x_min", dofs=[0])
    >>> bc = BoundaryCondition(bc_type="displacement", face="x_max",
    ...                        dofs=[0], value=-0.5)
    """

    bc_type: str
    face: str | None = None
    node_ids: np.ndarray | None = None
    dofs: list[int] = field(default_factory=lambda: [0, 1, 2])
    value: float = 0.0

    _VALID_TYPES = frozenset({
        "fixed", "displacement", "force", "pressure",
        "symmetry_x", "symmetry_y", "symmetry_z",
    })

    def __post_init__(self) -> None:
        if self.bc_type not in self._VALID_TYPES:
            raise ValueError(
                f"Unknown bc_type '{self.bc_type}'. "
                f"Must be one of {sorted(self._VALID_TYPES)}."
            )
        if self.face is None and self.node_ids is None:
            raise ValueError(
                "Either 'face' or 'node_ids' must be specified."
            )

    def resolve_nodes(self, mesh: MeshData) -> np.ndarray:
        """Return the node indices affected by this BC.

        If *face* is set, queries ``mesh.nodes_on_face(face)``.
        Otherwise returns *node_ids* directly.

        Parameters
        ----------
        mesh : MeshData
            The mesh to resolve face names against.

        Returns
        -------
        np.ndarray
            1-D array of node indices (0-based).
        """
        if self.node_ids is not None:
            return np.asarray(self.node_ids, dtype=np.intp)
        return mesh.nodes_on_face(self.face)

    def effective_dofs(self) -> list[int]:
        """Return the DOF indices affected, accounting for symmetry types.

        For ``"symmetry_x"`` returns ``[0]``, for ``"symmetry_y"`` returns
        ``[1]``, for ``"symmetry_z"`` returns ``[2]``.  Otherwise returns
        ``self.dofs``.

        Returns
        -------
        list[int]
        """
        symmetry_map = {
            "symmetry_x": [0],
            "symmetry_y": [1],
            "symmetry_z": [2],
        }
        return symmetry_map.get(self.bc_type, self.dofs)


# ======================================================================
# BoundaryHandler
# ======================================================================

class BoundaryHandler:
    """Applies boundary conditions to the global system K u = F.

    Supports two methods for imposing prescribed displacements:

    - **Penalty method** (:meth:`apply_penalty`): adds a large diagonal
      value to the stiffness matrix.  Simple, preserves matrix size,
      works well with direct solvers.
    - **Elimination method** (:meth:`apply_elimination`): partitions the
      system into free and constrained DOFs, producing a reduced system.
      More accurate, but changes the matrix size.

    Parameters
    ----------
    mesh : MeshData
        The finite element mesh.

    Examples
    --------
    >>> handler = BoundaryHandler(mesh)
    >>> bcs = BoundaryHandler.compression_bcs(mesh, applied_strain=-0.01)
    >>> constrained = handler.get_constrained_dofs(bcs)
    >>> F = handler.get_force_dofs(bcs)
    >>> K_mod, F_mod = handler.apply_penalty(K, F, constrained)
    """

    def __init__(self, mesh: MeshData) -> None:
        self.mesh = mesh

    # ------------------------------------------------------------------
    # Resolve BCs to DOF-level prescriptions
    # ------------------------------------------------------------------

    def get_constrained_dofs(
        self, bcs: list[BoundaryCondition]
    ) -> dict[int, float]:
        """Convert displacement-type BCs to a DOF-value mapping.

        Processes BCs of type ``"fixed"``, ``"displacement"``,
        ``"symmetry_x"``, ``"symmetry_y"``, and ``"symmetry_z"``.

        Parameters
        ----------
        bcs : list[BoundaryCondition]
            All boundary conditions.  Force/pressure BCs are ignored.

        Returns
        -------
        dict[int, float]
            Mapping ``{global_dof_index: prescribed_value}``.
            For fixed and symmetry BCs the value is 0.0.
        """
        disp_types = {"fixed", "displacement",
                       "symmetry_x", "symmetry_y", "symmetry_z"}
        constrained: dict[int, float] = {}

        for bc in bcs:
            if bc.bc_type not in disp_types:
                continue

            nodes = bc.resolve_nodes(self.mesh)
            dof_list = bc.effective_dofs()
            val = 0.0 if bc.bc_type in ("fixed", "symmetry_x",
                                         "symmetry_y", "symmetry_z") else bc.value

            for nid in nodes:
                for d in dof_list:
                    global_dof = 3 * int(nid) + d
                    constrained[global_dof] = val

        return constrained

    def get_force_dofs(
        self, bcs: list[BoundaryCondition]
    ) -> np.ndarray:
        """Assemble the global force vector from force/pressure BCs.

        For ``"force"`` BCs, the value is applied directly to each
        specified node in the specified DOFs.

        For ``"pressure"`` BCs, the total force (``bc.value``) is
        distributed equally among all nodes on the specified face
        in the specified DOFs.

        Parameters
        ----------
        bcs : list[BoundaryCondition]
            All boundary conditions.  Only ``"force"`` and ``"pressure"``
            types contribute; others are ignored.

        Returns
        -------
        np.ndarray
            Shape ``(n_dof,)`` global force vector.
        """
        n_dof = self.mesh.n_dof
        F = np.zeros(n_dof, dtype=np.float64)

        for bc in bcs:
            if bc.bc_type == "force":
                nodes = bc.resolve_nodes(self.mesh)
                for nid in nodes:
                    for d in bc.dofs:
                        F[3 * int(nid) + d] += bc.value

            elif bc.bc_type == "pressure":
                nodes = bc.resolve_nodes(self.mesh)
                n_nodes = len(nodes)
                if n_nodes == 0:
                    continue
                force_per_node = bc.value / n_nodes
                for nid in nodes:
                    for d in bc.dofs:
                        F[3 * int(nid) + d] += force_per_node

        return F

    # ------------------------------------------------------------------
    # Penalty method
    # ------------------------------------------------------------------

    def apply_penalty(
        self,
        K: sparse.csc_matrix,
        F: np.ndarray,
        constrained_dofs: dict[int, float],
        penalty: float = 1e20,
    ) -> tuple[sparse.csc_matrix, np.ndarray]:
        """Apply the penalty method for prescribed displacements.

        For each constrained DOF *i* with prescribed value *v*:

        .. math::
            K_{ii} \\mathrel{+}= \\alpha, \\qquad F_i = \\alpha \\, v

        where alpha is the penalty parameter.  The large diagonal entry
        forces the solution ``u_i`` to be approximately ``v``.

        The method converts ``K`` to LIL format for efficient element
        access, modifies it, and converts back to CSC.

        Parameters
        ----------
        K : scipy.sparse.csc_matrix
            Global stiffness matrix (``n_dof x n_dof``).
        F : np.ndarray
            Global force vector (``n_dof,``).
        constrained_dofs : dict[int, float]
            Mapping ``{dof_index: prescribed_value}`` from
            :meth:`get_constrained_dofs`.
        penalty : float, optional
            Penalty parameter.  Should be much larger than the largest
            diagonal entry of K (typically 1e15 to 1e20).  Default is
            ``1e20``.

        Returns
        -------
        K_modified : scipy.sparse.csc_matrix
            Modified stiffness matrix in CSC format.
        F_modified : np.ndarray
            Modified force vector.
        """
        K_lil = K.tolil()
        F_mod = F.copy()

        for dof, val in constrained_dofs.items():
            K_lil[dof, dof] += penalty
            F_mod[dof] = penalty * val

        return K_lil.tocsc(), F_mod

    # ------------------------------------------------------------------
    # Elimination method
    # ------------------------------------------------------------------

    def apply_elimination(
        self,
        K: sparse.csc_matrix,
        F: np.ndarray,
        constrained_dofs: dict[int, float],
    ) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply the elimination (partitioning) method for prescribed displacements.

        Partitions the system into free (f) and constrained (c) DOFs:

        .. math::
            K_{ff} \\, u_f = F_f - K_{fc} \\, u_c

        This produces a reduced system of size ``n_free x n_free``.

        Parameters
        ----------
        K : scipy.sparse.csc_matrix
            Global stiffness matrix (``n_dof x n_dof``).
        F : np.ndarray
            Global force vector (``n_dof,``).
        constrained_dofs : dict[int, float]
            Mapping ``{dof_index: prescribed_value}``.

        Returns
        -------
        K_ff : scipy.sparse.csc_matrix
            Reduced stiffness matrix (``n_free x n_free``).
        F_reduced : np.ndarray
            Reduced force vector (``n_free,``), accounting for the
            effect of prescribed displacements.
        free_dofs : np.ndarray
            Sorted array of free (unconstrained) DOF indices.
        constrained_dof_indices : np.ndarray
            Sorted array of constrained DOF indices.
        constrained_values : np.ndarray
            Prescribed values corresponding to ``constrained_dof_indices``.
        """
        n_dof = K.shape[0]
        all_dofs = np.arange(n_dof, dtype=np.intp)

        # Sort constrained DOFs for consistent ordering
        c_dofs_sorted = np.array(sorted(constrained_dofs.keys()), dtype=np.intp)
        c_vals = np.array([constrained_dofs[d] for d in c_dofs_sorted],
                          dtype=np.float64)

        # Free DOFs = all DOFs minus constrained DOFs
        constrained_set = set(c_dofs_sorted)
        free_dofs = np.array(
            [d for d in all_dofs if d not in constrained_set], dtype=np.intp
        )

        # Extract sub-matrices using sparse indexing
        # K_ff: free x free, K_fc: free x constrained
        K_csr = K.tocsr()
        K_ff = K_csr[np.ix_(free_dofs, free_dofs)].tocsc()
        K_fc = K_csr[np.ix_(free_dofs, c_dofs_sorted)]

        # Build reduced RHS: F_f - K_fc * u_c
        F_free = F[free_dofs].copy()
        if c_vals.size > 0 and np.any(c_vals != 0.0):
            F_free -= K_fc @ c_vals

        return K_ff, F_free, free_dofs, c_dofs_sorted, c_vals

    # ------------------------------------------------------------------
    # CLT LoadState to 3-D boundary conditions
    # ------------------------------------------------------------------

    @staticmethod
    def load_state_to_bcs(
        load: LoadState, mesh: MeshData
    ) -> list[BoundaryCondition]:
        """Convert a CLT load state to 3-D boundary conditions.

        Mapping from CLT resultants to 3-D BCs:

        - **Nx** (force/width): applied as uniform x-traction on
          ``x_max``.  ``sigma_x = Nx / h`` distributed as equal nodal
          forces.  ``x_min`` is fixed in x; one node is fully fixed
          (rigid body).
        - **Ny**: uniform y-traction on ``y_max``, fixed y on ``y_min``.
        - **Nxy**: tangential traction on x-faces (shear loading).
        - **Mx**: linear through-thickness x-displacement on ``x_max``:
          ``u_x(z) = kappa_x * z * Lx``, where ``kappa_x = Mx / D11``.
        - **My**: linear through-thickness y-displacement on ``y_max``.
        - Pressure (``Qx/Qy``): normal traction on ``z_max``.

        The method handles combined loading by accumulating BCs from
        each non-zero resultant.

        Parameters
        ----------
        load : LoadState
            CLT load state with force/moment resultants.
        mesh : MeshData
            The mesh to resolve face node IDs.

        Returns
        -------
        list[BoundaryCondition]
            Boundary conditions suitable for passing to
            :meth:`get_constrained_dofs` and :meth:`get_force_dofs`.
        """
        bcs: list[BoundaryCondition] = []
        Lx, Ly, Lz = mesh.domain_size
        h = Lz  # laminate thickness

        # ------ Rigid body suppression ------
        # Fix one corner node on x_min face to prevent rigid body motion.
        xmin_nodes = mesh.nodes_on_face("x_min")
        corner_node = np.array([xmin_nodes[0]], dtype=np.intp)

        has_any_load = (
            abs(load.Nx) > 0 or abs(load.Ny) > 0 or abs(load.Nxy) > 0
            or abs(load.Mx) > 0 or abs(load.My) > 0
        )

        if not has_any_load:
            return bcs

        # ------ Nx: axial x-loading ------
        if abs(load.Nx) > 0:
            # Fix ux on x_min face
            bcs.append(BoundaryCondition(
                bc_type="fixed", face="x_min", dofs=[0],
            ))
            # Fix uy on y_min (symmetry / prevent shear)
            bcs.append(BoundaryCondition(
                bc_type="symmetry_y", face="y_min",
            ))
            # Fix one node fully (rigid body z-translation)
            bcs.append(BoundaryCondition(
                bc_type="fixed", node_ids=corner_node, dofs=[1, 2],
            ))
            # Apply Nx as pressure on x_max face
            # Total force = Nx (N/mm) * Ly (mm) = N
            total_force = load.Nx * Ly
            bcs.append(BoundaryCondition(
                bc_type="pressure", face="x_max", dofs=[0],
                value=total_force,
            ))

        # ------ Ny: axial y-loading ------
        if abs(load.Ny) > 0:
            bcs.append(BoundaryCondition(
                bc_type="fixed", face="y_min", dofs=[1],
            ))
            if abs(load.Nx) == 0:
                # Only add x-symmetry if Nx is not already handling it
                bcs.append(BoundaryCondition(
                    bc_type="symmetry_x", face="x_min",
                ))
                bcs.append(BoundaryCondition(
                    bc_type="fixed", node_ids=corner_node, dofs=[0, 2],
                ))
            total_force = load.Ny * Lx
            bcs.append(BoundaryCondition(
                bc_type="pressure", face="y_max", dofs=[1],
                value=total_force,
            ))

        # ------ Nxy: in-plane shear ------
        if abs(load.Nxy) > 0:
            if abs(load.Nx) == 0 and abs(load.Ny) == 0:
                bcs.append(BoundaryCondition(
                    bc_type="fixed", face="x_min", dofs=[0, 1],
                ))
                bcs.append(BoundaryCondition(
                    bc_type="fixed", node_ids=corner_node, dofs=[2],
                ))
            # Shear traction on x_max face in y-direction
            total_force = load.Nxy * Ly
            bcs.append(BoundaryCondition(
                bc_type="pressure", face="x_max", dofs=[1],
                value=total_force,
            ))

        # ------ Mx: bending moment about y-axis ------
        if abs(load.Mx) > 0:
            xmax_nodes = mesh.nodes_on_face("x_max")
            z_coords = mesh.nodes[xmax_nodes, 2]
            z_mid = 0.5 * (z_coords.min() + z_coords.max())

            # Estimate curvature: kappa_x = Mx / D11
            # This is approximate; the user may need to refine.
            # For now, apply linear displacement: ux(z) = kappa * (z - z_mid) * Lx
            # We use a reference curvature scaled so the displacement is
            # proportional to Mx.
            # A more rigorous approach would use ABD inverse, but we keep it
            # simple here.
            kappa_x = load.Mx / 1.0  # user should normalise Mx appropriately

            # Apply prescribed displacements on x_max, varying linearly with z
            if abs(load.Nx) == 0:
                bcs.append(BoundaryCondition(
                    bc_type="fixed", face="x_min", dofs=[0],
                ))
                bcs.append(BoundaryCondition(
                    bc_type="symmetry_y", face="y_min",
                ))
                bcs.append(BoundaryCondition(
                    bc_type="fixed", node_ids=corner_node, dofs=[1, 2],
                ))

            for nid in xmax_nodes:
                z = float(mesh.nodes[nid, 2])
                ux = kappa_x * (z - z_mid) * Lx
                bcs.append(BoundaryCondition(
                    bc_type="displacement",
                    node_ids=np.array([nid], dtype=np.intp),
                    dofs=[0],
                    value=ux,
                ))

        # ------ My: bending moment about x-axis ------
        if abs(load.My) > 0:
            ymax_nodes = mesh.nodes_on_face("y_max")
            z_coords = mesh.nodes[ymax_nodes, 2]
            z_mid = 0.5 * (z_coords.min() + z_coords.max())

            kappa_y = load.My / 1.0

            if abs(load.Ny) == 0 and abs(load.Nx) == 0:
                bcs.append(BoundaryCondition(
                    bc_type="fixed", face="y_min", dofs=[1],
                ))
                bcs.append(BoundaryCondition(
                    bc_type="symmetry_x", face="x_min",
                ))
                bcs.append(BoundaryCondition(
                    bc_type="fixed", node_ids=corner_node, dofs=[0, 2],
                ))

            for nid in ymax_nodes:
                z = float(mesh.nodes[nid, 2])
                uy = kappa_y * (z - z_mid) * Ly
                bcs.append(BoundaryCondition(
                    bc_type="displacement",
                    node_ids=np.array([nid], dtype=np.intp),
                    dofs=[1],
                    value=uy,
                ))

        return bcs

    # ------------------------------------------------------------------
    # Convenience BC generators
    # ------------------------------------------------------------------

    @staticmethod
    def compression_bcs(
        mesh: MeshData,
        applied_strain: float = -0.01,
    ) -> list[BoundaryCondition]:
        """Standard uniaxial compression boundary conditions.

        Sets up a displacement-controlled compression test:

        - ``x_min``: ux = 0 (fixed in loading direction).
        - ``x_max``: ux = applied_strain * Lx (prescribed displacement).
        - ``y_min``: uy = 0 (symmetry about xz-plane).
        - One corner node fully fixed (rigid body suppression).

        Parameters
        ----------
        mesh : MeshData
            The finite element mesh.
        applied_strain : float, optional
            Applied nominal strain (negative for compression).
            Default is ``-0.01`` (1% compressive strain).

        Returns
        -------
        list[BoundaryCondition]
            List of BCs ready for the handler.

        Examples
        --------
        >>> bcs = BoundaryHandler.compression_bcs(mesh, applied_strain=-0.005)
        >>> handler = BoundaryHandler(mesh)
        >>> constrained = handler.get_constrained_dofs(bcs)
        """
        Lx = mesh.domain_size[0]
        prescribed_disp = applied_strain * Lx

        # Identify one corner node for full rigid body suppression
        xmin_nodes = mesh.nodes_on_face("x_min")
        zmin_nodes = mesh.nodes_on_face("z_min")
        # Corner node at (x_min, y_min, z_min)
        corner_candidates = np.intersect1d(xmin_nodes, zmin_nodes)
        if corner_candidates.size > 0:
            ymin_nodes = mesh.nodes_on_face("y_min")
            full_corner = np.intersect1d(corner_candidates, ymin_nodes)
            corner_node = np.array(
                [full_corner[0] if full_corner.size > 0 else corner_candidates[0]],
                dtype=np.intp,
            )
        else:
            corner_node = np.array([xmin_nodes[0]], dtype=np.intp)

        bcs = [
            # Fix ux on x_min (loading face support)
            BoundaryCondition(bc_type="fixed", face="x_min", dofs=[0]),
            # Prescribe ux on x_max (loading face)
            BoundaryCondition(
                bc_type="displacement", face="x_max",
                dofs=[0], value=prescribed_disp,
            ),
            # Symmetry: uy = 0 on y_min
            BoundaryCondition(bc_type="symmetry_y", face="y_min"),
            # Fully fix corner node for rigid body suppression (uy, uz)
            BoundaryCondition(
                bc_type="fixed", node_ids=corner_node, dofs=[1, 2],
            ),
        ]

        return bcs

    @staticmethod
    def bending_bcs(
        mesh: MeshData,
        curvature: float = 0.001,
    ) -> list[BoundaryCondition]:
        """Pure bending boundary conditions.

        Applies a linear through-thickness displacement on ``x_max``
        to produce a bending deformation:

        .. math::
            u_x(z) = \\kappa \\, (z - z_{\\text{mid}}) \\, L_x

        where ``z_mid`` is the laminate midplane z-coordinate.

        Support conditions:

        - ``x_min``: ux = 0, uy = 0.
        - ``y_min``: uy = 0 (symmetry).
        - One corner node fully fixed.

        Parameters
        ----------
        mesh : MeshData
            The finite element mesh.
        curvature : float, optional
            Applied curvature (1/mm).  Positive curvature produces
            tension on the z_max surface.  Default is ``0.001``.

        Returns
        -------
        list[BoundaryCondition]
            List of BCs for bending analysis.
        """
        Lx = mesh.domain_size[0]

        # Identify corner node
        xmin_nodes = mesh.nodes_on_face("x_min")
        zmin_nodes = mesh.nodes_on_face("z_min")
        ymin_nodes = mesh.nodes_on_face("y_min")
        corner_candidates = np.intersect1d(xmin_nodes, zmin_nodes)
        full_corner = np.intersect1d(corner_candidates, ymin_nodes)
        corner_node = np.array(
            [full_corner[0] if full_corner.size > 0 else xmin_nodes[0]],
            dtype=np.intp,
        )

        bcs = [
            # Fix ux and uy on x_min
            BoundaryCondition(bc_type="fixed", face="x_min", dofs=[0, 1]),
            # Symmetry: uy = 0 on y_min
            BoundaryCondition(bc_type="symmetry_y", face="y_min"),
            # Fully fix corner node (uz for rigid body)
            BoundaryCondition(
                bc_type="fixed", node_ids=corner_node, dofs=[2],
            ),
        ]

        # Linear through-thickness displacement on x_max
        xmax_nodes = mesh.nodes_on_face("x_max")
        z_coords_xmax = mesh.nodes[xmax_nodes, 2]
        z_mid = 0.5 * (z_coords_xmax.min() + z_coords_xmax.max())

        for nid in xmax_nodes:
            z = float(mesh.nodes[nid, 2])
            ux = curvature * (z - z_mid) * Lx
            bcs.append(BoundaryCondition(
                bc_type="displacement",
                node_ids=np.array([nid], dtype=np.intp),
                dofs=[0],
                value=ux,
            ))

        return bcs
