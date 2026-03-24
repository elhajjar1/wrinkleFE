"""Structured hexahedral mesh generation for wrinkled composite laminates.

Generates 3-D hex8 meshes where the z-coordinates align with ply boundaries
from a :class:`~wrinklefe.core.laminate.Laminate` and can be deformed by a
:class:`~wrinklefe.core.morphology.WrinkleConfiguration` to represent fiber
waviness defects.

The mesh uses standard finite-element node ordering:

* i (x-direction) varies fastest
* j (y-direction) varies next
* k (z-direction) varies slowest

So node index for grid point (i, j, k) is::

    k * (ny + 1) * (nx + 1)  +  j * (nx + 1)  +  i

Hex8 node numbering follows the VTK/Abaqus convention (bottom face CCW,
then top face CCW).

References
----------
Jin, L. et al. (2026). Thin-Walled Structures 219, 114237.
Elhajjar, R. (2025). Scientific Reports 15, 25977.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from wrinklefe.core.laminate import Laminate

if TYPE_CHECKING:
    from wrinklefe.core.morphology import WrinkleConfiguration


# ---------------------------------------------------------------------------
# MeshData — immutable container for a generated mesh
# ---------------------------------------------------------------------------

@dataclass
class MeshData:
    """Container for finite element mesh data.

    All arrays are read-only after construction.  Node and element numbering
    is 0-based throughout; the Abaqus exporter converts to 1-based on write.

    Parameters
    ----------
    nodes : np.ndarray
        Shape ``(n_nodes, 3)`` — x, y, z coordinates (mm).
    elements : np.ndarray
        Shape ``(n_elements, 8)`` — node connectivity for hex8 elements.
    ply_ids : np.ndarray
        Shape ``(n_elements,)`` — ply index (0-based) for each element.
    fiber_angles : np.ndarray
        Shape ``(n_nodes,)`` — local fiber misalignment angle from wrinkle
        geometry (radians).
    ply_angles : np.ndarray
        Shape ``(n_elements,)`` — nominal ply orientation angle (degrees).
    nx : int
        Number of elements in x.
    ny : int
        Number of elements in y.
    nz : int
        Number of elements in z (= ``n_plies * nz_per_ply``).
    """

    nodes: np.ndarray
    elements: np.ndarray
    ply_ids: np.ndarray
    fiber_angles: np.ndarray
    ply_angles: np.ndarray
    nx: int
    ny: int
    nz: int

    # ---- derived quantities ------------------------------------------------

    @property
    def n_nodes(self) -> int:
        """Total number of nodes."""
        return self.nodes.shape[0]

    @property
    def n_elements(self) -> int:
        """Total number of elements."""
        return self.elements.shape[0]

    @property
    def n_dof(self) -> int:
        """Total degrees of freedom (3 per node)."""
        return self.n_nodes * 3

    @property
    def domain_size(self) -> tuple[float, float, float]:
        """Domain extents ``(Lx, Ly, Lz)`` from node coordinate ranges (mm)."""
        mins = self.nodes.min(axis=0)
        maxs = self.nodes.max(axis=0)
        extent = maxs - mins
        return (float(extent[0]), float(extent[1]), float(extent[2]))

    # ---- element queries ---------------------------------------------------

    def element_nodes(self, elem_idx: int) -> np.ndarray:
        """Return the ``(8, 3)`` array of node coordinates for an element.

        Parameters
        ----------
        elem_idx : int
            Element index (0-based).

        Returns
        -------
        np.ndarray
            Shape ``(8, 3)`` coordinate array.
        """
        return self.nodes[self.elements[elem_idx]]

    def element_center(self, elem_idx: int) -> np.ndarray:
        """Return the ``(3,)`` centroid of an element.

        Parameters
        ----------
        elem_idx : int
            Element index (0-based).

        Returns
        -------
        np.ndarray
            Shape ``(3,)`` centroid coordinates (mm).
        """
        return self.element_nodes(elem_idx).mean(axis=0)

    def element_fiber_angle(self, elem_idx: int) -> float:
        """Return the average fiber misalignment angle for an element (radians).

        Averages the nodal fiber angles over the 8 nodes of the hex element.

        Parameters
        ----------
        elem_idx : int
            Element index (0-based).

        Returns
        -------
        float
            Mean fiber misalignment angle (radians).
        """
        node_ids = self.elements[elem_idx]
        return float(self.fiber_angles[node_ids].mean())

    def element_fiber_angles_array(self) -> np.ndarray:
        """Return per-element fiber misalignment angles for all elements.

        Returns
        -------
        np.ndarray
            Shape ``(n_elements,)`` array of mean fiber angles (radians).
        """
        return self.fiber_angles[self.elements].mean(axis=1)

    # ---- boundary / set queries --------------------------------------------

    def nodes_on_face(self, face: str) -> np.ndarray:
        """Return node indices on a boundary face of the domain.

        Parameters
        ----------
        face : str
            One of ``'x_min'``, ``'x_max'``, ``'y_min'``, ``'y_max'``,
            ``'z_min'``, ``'z_max'``.

        Returns
        -------
        np.ndarray
            1-D array of node indices (0-based).

        Raises
        ------
        ValueError
            If *face* is not a recognised name.
        """
        axis_map = {
            "x_min": (0, "min"),
            "x_max": (0, "max"),
            "y_min": (1, "min"),
            "y_max": (1, "max"),
            "z_min": (2, "min"),
            "z_max": (2, "max"),
        }
        if face not in axis_map:
            raise ValueError(
                f"Unknown face '{face}'. Must be one of {list(axis_map)}"
            )
        axis, side = axis_map[face]
        coords = self.nodes[:, axis]
        tol = 1.0e-10 * (coords.max() - coords.min() + 1.0e-30)
        if side == "min":
            target = coords.min()
        else:
            target = coords.max()
        return np.flatnonzero(np.abs(coords - target) < tol)

    def elements_in_ply(self, ply_idx: int) -> np.ndarray:
        """Return element indices belonging to a given ply.

        Parameters
        ----------
        ply_idx : int
            Ply index (0-based, bottom to top).

        Returns
        -------
        np.ndarray
            1-D array of element indices.
        """
        return np.flatnonzero(self.ply_ids == ply_idx)

    def midplane_elements(self) -> np.ndarray:
        """Return element indices at or nearest the laminate midplane.

        The midplane is at z = 0.  This method selects the layer of elements
        whose centroids have the smallest absolute z-coordinate.

        Returns
        -------
        np.ndarray
            1-D array of element indices.
        """
        # Compute z-centroid for every element using vectorised indexing
        # elements shape: (n_elem, 8)
        node_z = self.nodes[:, 2]  # (n_nodes,)
        elem_z = node_z[self.elements].mean(axis=1)  # (n_elem,)
        abs_z = np.abs(elem_z)
        min_abs = abs_z.min()
        tol = 1.0e-10 * (node_z.max() - node_z.min() + 1.0e-30)
        return np.flatnonzero(abs_z - min_abs < tol)

    def interface_nodes(self, ply_above: int, ply_below: int) -> np.ndarray:
        """Return node indices on the interface between two adjacent plies.

        The interface is the shared z-plane between the top surface of
        *ply_below* and the bottom surface of *ply_above*.  The method finds
        all nodes whose z-coordinate matches that boundary.

        Parameters
        ----------
        ply_above : int
            Index of the ply above the interface.
        ply_below : int
            Index of the ply below the interface.

        Returns
        -------
        np.ndarray
            1-D array of node indices.

        Raises
        ------
        ValueError
            If the plies are not adjacent (``ply_above != ply_below + 1``).
        """
        if ply_above != ply_below + 1:
            raise ValueError(
                f"Plies must be adjacent: ply_above ({ply_above}) "
                f"must equal ply_below ({ply_below}) + 1."
            )
        # The interface z is at the top of ply_below.  In an undeformed mesh
        # this is a constant, but for a wrinkled mesh the z-coordinates vary.
        # We identify interface nodes as those shared by elements in both plies.
        elems_above = set(self.elements[self.elements_in_ply(ply_above)].ravel())
        elems_below = set(self.elements[self.elements_in_ply(ply_below)].ravel())
        shared = np.array(sorted(elems_above & elems_below), dtype=np.intp)
        return shared


# ---------------------------------------------------------------------------
# WrinkleMesh — mesh generator
# ---------------------------------------------------------------------------

class WrinkleMesh:
    """Generate a structured hexahedral mesh for a wrinkled composite laminate.

    The mesh covers a rectangular domain ``[0, Lx] x [0, Ly]`` with
    through-thickness coordinates determined by the laminate ply boundaries.
    An optional :class:`WrinkleConfiguration` deforms the nodal z-coordinates
    to represent fiber waviness.

    Parameters
    ----------
    laminate : Laminate
        Laminate definition providing ply thicknesses and orientations.
    wrinkle_config : WrinkleConfiguration or None
        Wrinkle geometry applied to the mesh.  If ``None``, the mesh is flat.
    Lx : float
        Domain length in x (mm).
    Ly : float
        Domain width in y (mm).
    nx : int
        Number of elements in x.
    ny : int
        Number of elements in y.
    nz_per_ply : int
        Number of elements per ply through the thickness.
    """

    def __init__(
        self,
        laminate: Laminate,
        wrinkle_config: "WrinkleConfiguration | None" = None,
        Lx: float = 48.0,
        Ly: float = 20.0,
        nx: int = 50,
        ny: int = 20,
        nz_per_ply: int = 1,
    ) -> None:
        if Lx <= 0 or Ly <= 0:
            raise ValueError("Domain dimensions Lx, Ly must be positive.")
        if nx < 1 or ny < 1 or nz_per_ply < 1:
            raise ValueError("Mesh resolution (nx, ny, nz_per_ply) must be >= 1.")

        self.laminate = laminate
        self.wrinkle_config = wrinkle_config
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.nz_per_ply = nz_per_ply
        self.nz = laminate.n_plies * nz_per_ply

    # =======================================================================
    # Public API
    # =======================================================================

    def generate(self) -> MeshData:
        """Generate the full mesh.

        Steps
        -----
        1. Create regular grid of nodes ``(nx+1) x (ny+1) x (nz+1)``.
        2. Set z-coordinates based on ply boundaries from the laminate.
        3. If a wrinkle configuration exists, deform nodes via
           ``wrinkle_config.apply_to_nodes()``.
        4. Create hex8 element connectivity.
        5. Compute fiber angles at nodes from wrinkle geometry.
        6. Assign ply IDs and ply angles to elements.

        Returns
        -------
        MeshData
            The generated mesh.
        """
        nodes = self._create_regular_grid()
        node_ply_ids = self._node_to_element_ply()

        # Wrinkle deformation
        if self.wrinkle_config is not None:
            nodes = self.wrinkle_config.apply_to_nodes(
                nodes, node_ply_ids, self.laminate.n_plies
            )

        elements = self._create_hex_connectivity()
        ply_ids, ply_angles = self._assign_ply_ids()
        fiber_angles = self._compute_fiber_angles(nodes, node_ply_ids)

        return MeshData(
            nodes=nodes,
            elements=elements,
            ply_ids=ply_ids,
            fiber_angles=fiber_angles,
            ply_angles=ply_angles,
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
        )

    # =======================================================================
    # Grid and connectivity helpers
    # =======================================================================

    def _create_regular_grid(self) -> np.ndarray:
        """Create ``(nx+1)*(ny+1)*(nz+1)`` nodes on a regular grid.

        * x: uniform from 0 to ``Lx``
        * y: uniform from 0 to ``Ly``
        * z: matches ply boundaries from ``laminate.z_coords()``.
          Within each ply, subdivide uniformly if ``nz_per_ply > 1``.

        Node ordering: k (z, slowest) -> j (y) -> i (x, fastest).
        Node index ``(i, j, k) = k*(ny+1)*(nx+1) + j*(nx+1) + i``.

        Returns
        -------
        np.ndarray
            Shape ``(n_nodes, 3)`` coordinate array (mm).
        """
        nxp = self.nx + 1
        nyp = self.ny + 1
        nzp = self.nz + 1

        # Build 1-D coordinate arrays
        x1d = np.linspace(0.0, self.Lx, nxp)
        y1d = np.linspace(0.0, self.Ly, nyp)
        z1d = self._build_z_coordinates()  # length nzp

        # Create full grid using broadcasting (vectorised, no Python loops)
        # Indices: i -> x, j -> y, k -> z
        # Output shape of each meshgrid component: (nzp, nyp, nxp)
        xi, yj, zk = np.meshgrid(x1d, y1d, z1d, indexing="ij")
        # meshgrid with indexing='ij' gives shape (nxp, nyp, nzp)
        # We want flat order: k slowest, j middle, i fastest
        # Transpose to (nzp, nyp, nxp) then ravel
        xi = xi.transpose(2, 1, 0).ravel()
        yj = yj.transpose(2, 1, 0).ravel()
        zk = zk.transpose(2, 1, 0).ravel()

        nodes = np.column_stack([xi, yj, zk])
        return nodes

    def _build_z_coordinates(self) -> np.ndarray:
        """Build the 1-D array of z-coordinates matching ply boundaries.

        Returns an array of length ``nz + 1``.  Ply interfaces are located at
        the positions given by ``laminate.z_coords()``, and each ply is
        subdivided into ``nz_per_ply`` equal sub-layers.

        Returns
        -------
        np.ndarray
            Shape ``(nz + 1,)`` z-coordinates (mm).
        """
        ply_z = self.laminate.z_coords()  # (n_plies + 1,)
        n_plies = self.laminate.n_plies

        z_all = np.empty(self.nz + 1, dtype=float)
        for p in range(n_plies):
            z_bot = ply_z[p]
            z_top = ply_z[p + 1]
            k_start = p * self.nz_per_ply
            sub_z = np.linspace(z_bot, z_top, self.nz_per_ply + 1)
            # Write sub-layer boundaries (avoid duplicating shared interfaces)
            z_all[k_start: k_start + self.nz_per_ply + 1] = sub_z

        return z_all

    def _create_hex_connectivity(self) -> np.ndarray:
        """Create hex8 element connectivity.

        Element ``(i, j, k)`` has 8 nodes following the standard convention:

        * Bottom face (k):   n0(i,j,k), n1(i+1,j,k), n2(i+1,j+1,k), n3(i,j+1,k)
        * Top face (k+1):    n4(i,j,k+1), n5(i+1,j,k+1), n6(i+1,j+1,k+1), n7(i,j+1,k+1)

        Returns
        -------
        np.ndarray
            Shape ``(n_elements, 8)`` array of node indices (0-based).
        """
        nxp = self.nx + 1
        nyp = self.ny + 1

        # Element grid indices
        ei = np.arange(self.nx)
        ej = np.arange(self.ny)
        ek = np.arange(self.nz)

        # 3-D index arrays — shape (nz, ny, nx) with k slowest, j middle, i fastest
        ki, ji, ii = np.meshgrid(ek, ej, ei, indexing="ij")
        ki = ki.ravel()
        ji = ji.ravel()
        ii = ii.ravel()

        def node_id(i: np.ndarray, j: np.ndarray, k: np.ndarray) -> np.ndarray:
            """Convert (i, j, k) grid indices to flat node index."""
            return k * (nyp * nxp) + j * nxp + i

        n0 = node_id(ii, ji, ki)
        n1 = node_id(ii + 1, ji, ki)
        n2 = node_id(ii + 1, ji + 1, ki)
        n3 = node_id(ii, ji + 1, ki)
        n4 = node_id(ii, ji, ki + 1)
        n5 = node_id(ii + 1, ji, ki + 1)
        n6 = node_id(ii + 1, ji + 1, ki + 1)
        n7 = node_id(ii, ji + 1, ki + 1)

        elements = np.column_stack([n0, n1, n2, n3, n4, n5, n6, n7])
        return elements

    def _assign_ply_ids(self) -> tuple[np.ndarray, np.ndarray]:
        """Assign ply index and ply orientation angle to each element.

        Element ``(i, j, k)`` belongs to ply ``k // nz_per_ply``.

        Returns
        -------
        ply_ids : np.ndarray
            Shape ``(n_elements,)`` — ply index (0-based).
        ply_angles : np.ndarray
            Shape ``(n_elements,)`` — nominal ply orientation (degrees).
        """
        n_elem = self.nx * self.ny * self.nz

        # Element k-indices in the same ravel order used by _create_hex_connectivity
        ek = np.arange(self.nz)
        # Repeat for all (j, i) in each k-layer
        k_per_elem = np.repeat(ek, self.nx * self.ny)

        ply_ids = k_per_elem // self.nz_per_ply

        # Map ply IDs to ply angles
        angles_lookup = np.array(
            [self.laminate.plies[p].angle for p in range(self.laminate.n_plies)],
            dtype=float,
        )
        ply_angles = angles_lookup[ply_ids]

        return ply_ids, ply_angles

    def _node_to_element_ply(self) -> np.ndarray:
        """Map each node to a ply ID based on its k-index.

        For nodes on ply boundaries, the node is assigned to the ply *below*
        (i.e., the lower ply owns the shared interface).  The top-surface
        nodes (k = nz) are assigned to the topmost ply.

        Returns
        -------
        np.ndarray
            Shape ``(n_nodes,)`` array of ply indices (0-based).
        """
        nxp = self.nx + 1
        nyp = self.ny + 1
        nzp = self.nz + 1
        n_nodes = nxp * nyp * nzp

        # k-index for every node
        k_idx = np.arange(n_nodes) // (nxp * nyp)

        # Map k-index to ply: k // nz_per_ply, clamped to valid range
        ply_ids = np.minimum(k_idx // self.nz_per_ply, self.laminate.n_plies - 1)
        return ply_ids

    def _compute_fiber_angles(
        self, nodes: np.ndarray, ply_ids_per_node: np.ndarray
    ) -> np.ndarray:
        """Compute fiber misalignment angle at each node from wrinkle geometry.

        If no ``wrinkle_config`` is set, all angles are zero.  Otherwise, the
        angles are obtained from ``wrinkle_config.fiber_angles_at_nodes()``.

        Parameters
        ----------
        nodes : np.ndarray
            Shape ``(n_nodes, 3)`` node coordinates.
        ply_ids_per_node : np.ndarray
            Shape ``(n_nodes,)`` ply index for each node.

        Returns
        -------
        np.ndarray
            Shape ``(n_nodes,)`` fiber misalignment angle (radians).
        """
        if self.wrinkle_config is None:
            return np.zeros(nodes.shape[0], dtype=float)
        return self.wrinkle_config.fiber_angles_at_nodes(nodes, ply_ids_per_node)

    # =======================================================================
    # Export methods
    # =======================================================================

    def to_abaqus_inp(self, mesh: MeshData, filename: str) -> None:
        """Write the mesh in Abaqus ``.inp`` format.

        Nodes and elements use **1-based** indexing as required by Abaqus.
        Elements are grouped into element sets ``PLY_<k>`` for easy section
        assignment.

        Parameters
        ----------
        mesh : MeshData
            Generated mesh data.
        filename : str
            Output file path (should end with ``.inp``).
        """
        with open(filename, "w") as f:
            f.write("*HEADING\n")
            f.write("Wrinkled composite laminate mesh\n")
            f.write(f"** {mesh.n_nodes} nodes, {mesh.n_elements} C3D8 elements, "
                    f"{self.laminate.n_plies} plies\n")

            # Nodes (1-based)
            f.write("*NODE\n")
            for nid in range(mesh.n_nodes):
                x, y, z = mesh.nodes[nid]
                f.write(f"{nid + 1:8d}, {x:14.6e}, {y:14.6e}, {z:14.6e}\n")

            # Elements grouped by ply
            for ply_idx in range(self.laminate.n_plies):
                elem_ids = mesh.elements_in_ply(ply_idx)
                if elem_ids.size == 0:
                    continue
                f.write(f"*ELEMENT, TYPE=C3D8, ELSET=PLY_{ply_idx}\n")
                for eid in elem_ids:
                    conn = mesh.elements[eid] + 1  # 1-based
                    conn_str = ", ".join(str(int(n)) for n in conn)
                    f.write(f"{eid + 1:8d}, {conn_str}\n")

            # Node sets for boundary faces
            for face in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]:
                face_nodes = mesh.nodes_on_face(face) + 1  # 1-based
                f.write(f"*NSET, NSET={face.upper()}\n")
                for line_start in range(0, len(face_nodes), 16):
                    chunk = face_nodes[line_start: line_start + 16]
                    f.write(", ".join(str(int(n)) for n in chunk) + "\n")

    def to_vtk(self, mesh: MeshData, filename: str) -> None:
        """Write the mesh in VTK legacy format for ParaView visualisation.

        Writes an unstructured grid (``.vtk``) file with:

        * **Point data**: ``fiber_angle`` (radians)
        * **Cell data**: ``ply_id``, ``ply_angle`` (degrees)

        The VTK cell type for hexahedra is 12 (``VTK_HEXAHEDRON``).

        Parameters
        ----------
        mesh : MeshData
            Generated mesh data.
        filename : str
            Output file path (should end with ``.vtk``).
        """
        n_nodes = mesh.n_nodes
        n_elem = mesh.n_elements

        with open(filename, "w") as f:
            # Header
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Wrinkled composite laminate mesh\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")

            # Points
            f.write(f"POINTS {n_nodes} double\n")
            for nid in range(n_nodes):
                x, y, z = mesh.nodes[nid]
                f.write(f"{x:.8e} {y:.8e} {z:.8e}\n")

            # Cells — each hex8 line: 8 node_id0 node_id1 ... node_id7
            total_ints = n_elem * 9  # 8 nodes + 1 count per element
            f.write(f"CELLS {n_elem} {total_ints}\n")
            for eid in range(n_elem):
                conn = mesh.elements[eid]
                conn_str = " ".join(str(int(n)) for n in conn)
                f.write(f"8 {conn_str}\n")

            # Cell types — 12 = VTK_HEXAHEDRON
            f.write(f"CELL_TYPES {n_elem}\n")
            for _ in range(n_elem):
                f.write("12\n")

            # Point data
            f.write(f"POINT_DATA {n_nodes}\n")
            f.write("SCALARS fiber_angle double 1\n")
            f.write("LOOKUP_TABLE default\n")
            for nid in range(n_nodes):
                f.write(f"{mesh.fiber_angles[nid]:.8e}\n")

            # Cell data
            f.write(f"CELL_DATA {n_elem}\n")
            f.write("SCALARS ply_id int 1\n")
            f.write("LOOKUP_TABLE default\n")
            for eid in range(n_elem):
                f.write(f"{int(mesh.ply_ids[eid])}\n")
            f.write("SCALARS ply_angle double 1\n")
            f.write("LOOKUP_TABLE default\n")
            for eid in range(n_elem):
                f.write(f"{mesh.ply_angles[eid]:.4f}\n")
