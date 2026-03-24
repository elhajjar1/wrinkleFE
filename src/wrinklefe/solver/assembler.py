"""Global stiffness matrix assembly using sparse COO to CSC format.

Assembles the global stiffness matrix and force vector from element-level
contributions computed by :class:`~wrinklefe.elements.hex8.Hex8Element`.

The assembly uses COO (coordinate) format for efficient incremental
construction, then converts to CSC (compressed sparse column) for
compatibility with sparse direct solvers (e.g. ``scipy.sparse.linalg.spsolve``).

Algorithm
---------
1. Pre-allocate flat COO arrays (rows, cols, values) sized for the
   maximum number of non-zero entries: ``n_elements * 24 * 24``.
2. Loop over elements, computing the 24x24 element stiffness and
   mapping local DOF indices to global DOF indices.
3. Construct ``scipy.sparse.coo_matrix`` and convert to CSC.
   Duplicate entries at the same (row, col) are summed automatically.

References
----------
Zienkiewicz, O.C. & Taylor, R.L. (2000). The Finite Element Method, Vol. 1.
Bathe, K.-J. (2006). Finite Element Procedures.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from wrinklefe.core.mesh import MeshData
from wrinklefe.core.laminate import Laminate
from wrinklefe.elements.hex8 import Hex8Element


class GlobalAssembler:
    """Assembles global stiffness matrix from element contributions.

    Uses COO (coordinate) format for assembly, then converts to CSC
    (compressed sparse column) for efficient solving.

    Parameters
    ----------
    mesh : MeshData
        Mesh containing nodes, elements, fiber_angles, ply_ids, and ply_angles.
    laminate : Laminate
        Laminate definition providing material properties per ply.
    element_type : str, optional
        Element formulation to use.  Currently only ``'hex8'`` is supported.
        Default is ``'hex8'``.

    Attributes
    ----------
    mesh : MeshData
        The finite element mesh.
    laminate : Laminate
        The laminate definition.
    element_type : str
        Element formulation identifier.

    Examples
    --------
    >>> assembler = GlobalAssembler(mesh, laminate)
    >>> K = assembler.assemble_stiffness(verbose=True)
    >>> dofs = assembler.element_dof_indices(0)
    >>> dofs.shape
    (24,)
    """

    def __init__(
        self,
        mesh: MeshData,
        laminate: Laminate,
        element_type: str = "hex8",
    ) -> None:
        if element_type not in ("hex8",):
            raise ValueError(
                f"Unsupported element type '{element_type}'. "
                f"Currently only 'hex8' is supported."
            )
        self.mesh = mesh
        self.laminate = laminate
        self.element_type = element_type

    # ------------------------------------------------------------------
    # Element construction
    # ------------------------------------------------------------------

    def create_element(self, elem_idx: int) -> Hex8Element:
        """Create an element object for the given element index.

        Extracts node coordinates, material, ply angle, and per-node
        wrinkle misalignment angles from the mesh data.

        Parameters
        ----------
        elem_idx : int
            Element index (0-based).

        Returns
        -------
        Hex8Element
            Fully initialised element ready for stiffness computation.
        """
        # Node connectivity for this element: shape (8,)
        node_ids = self.mesh.elements[elem_idx]

        # Physical coordinates of the 8 nodes: shape (8, 3)
        node_coords = self.mesh.nodes[node_ids]

        # Ply index and orientation for this element
        ply_idx = int(self.mesh.ply_ids[elem_idx])
        ply_angle = float(self.mesh.ply_angles[elem_idx])

        # Material from the laminate's ply stack
        material = self.laminate.plies[ply_idx].material

        # Wrinkle misalignment angles at the 8 element nodes (radians)
        wrinkle_angles = self.mesh.fiber_angles[node_ids]

        return Hex8Element(
            node_coords=node_coords,
            material=material,
            ply_angle=ply_angle,
            wrinkle_angles=wrinkle_angles,
        )

    # ------------------------------------------------------------------
    # DOF mapping
    # ------------------------------------------------------------------

    def element_dof_indices(self, elem_idx: int) -> np.ndarray:
        """Global DOF indices for an element's nodes.

        Each of the 8 nodes has 3 translational DOFs (ux, uy, uz).
        For node indices ``[n0, n1, ..., n7]``, the 24 DOFs are::

            [3*n0, 3*n0+1, 3*n0+2,
             3*n1, 3*n1+1, 3*n1+2,
             ...
             3*n7, 3*n7+1, 3*n7+2]

        Parameters
        ----------
        elem_idx : int
            Element index (0-based).

        Returns
        -------
        np.ndarray
            Shape ``(24,)`` array of global DOF indices (int).
        """
        node_ids = self.mesh.elements[elem_idx]  # (8,)
        # For each node, generate 3 consecutive DOF indices
        dofs = np.empty(24, dtype=np.intp)
        for i, nid in enumerate(node_ids):
            base = 3 * nid
            dofs[3 * i] = base
            dofs[3 * i + 1] = base + 1
            dofs[3 * i + 2] = base + 2
        return dofs

    # ------------------------------------------------------------------
    # Global stiffness assembly
    # ------------------------------------------------------------------

    def assemble_stiffness(self, verbose: bool = False) -> sparse.csc_matrix:
        """Assemble the global stiffness matrix K.

        Algorithm
        ---------
        1. Pre-allocate COO arrays (rows, cols, values).
           Each hex8 element contributes 24 x 24 = 576 entries.
           Total pre-allocated non-zeros = ``n_elements * 576``.
        2. Loop over elements:
           a. Create element from mesh data.
           b. Compute element stiffness Ke (24 x 24).
           c. Get global DOF indices for element nodes.
           d. Fill COO arrays with Ke entries at global DOF positions.
        3. Create COO sparse matrix and convert to CSC.  Duplicate entries
           at the same (i, j) position are summed automatically by
           ``scipy.sparse.coo_matrix``.

        Parameters
        ----------
        verbose : bool, optional
            If ``True``, print progress every 1000 elements.
            Default is ``False``.

        Returns
        -------
        scipy.sparse.csc_matrix
            Shape ``(n_dof, n_dof)`` global stiffness matrix in CSC format.
        """
        n_elem = self.mesh.n_elements
        n_dof = self.mesh.n_dof
        entries_per_elem = 24 * 24  # 576

        # Pre-allocate COO arrays
        total_entries = n_elem * entries_per_elem
        coo_rows = np.empty(total_entries, dtype=np.intp)
        coo_cols = np.empty(total_entries, dtype=np.intp)
        coo_vals = np.empty(total_entries, dtype=np.float64)

        # Local row/col index pairs for a 24x24 matrix (reused every element)
        local_ii, local_jj = np.meshgrid(
            np.arange(24), np.arange(24), indexing="ij"
        )
        local_ii = local_ii.ravel()  # (576,)
        local_jj = local_jj.ravel()  # (576,)

        for e in range(n_elem):
            if verbose and e % 1000 == 0:
                print(f"  Assembling element {e}/{n_elem} "
                      f"({100.0 * e / n_elem:.1f}%)")

            # Create element and compute stiffness
            elem = self.create_element(e)
            Ke = elem.stiffness_matrix()  # (24, 24)

            # Global DOF indices
            dofs = self.element_dof_indices(e)  # (24,)

            # Map local indices to global indices
            offset = e * entries_per_elem
            coo_rows[offset:offset + entries_per_elem] = dofs[local_ii]
            coo_cols[offset:offset + entries_per_elem] = dofs[local_jj]
            coo_vals[offset:offset + entries_per_elem] = Ke.ravel()

        if verbose:
            print(f"  Assembling element {n_elem}/{n_elem} (100.0%) — done.")
            print(f"  Building sparse matrix: {n_dof} DOFs, "
                  f"{total_entries} COO entries")

        # Build COO matrix and convert to CSC
        K_coo = sparse.coo_matrix(
            (coo_vals, (coo_rows, coo_cols)),
            shape=(n_dof, n_dof),
        )
        K_csc = K_coo.tocsc()

        if verbose:
            print(f"  CSC matrix: {K_csc.nnz} stored entries "
                  f"({K_csc.nnz / n_dof:.1f} per DOF)")

        return K_csc

    # ------------------------------------------------------------------
    # Force vector assembly
    # ------------------------------------------------------------------

    def assemble_force_vector(
        self, boundary_conditions: list
    ) -> np.ndarray:
        """Assemble global force vector F from boundary conditions.

        Delegates to the boundary condition objects to build the
        force vector.  This is a convenience method; typically the
        :class:`~wrinklefe.solver.boundary.BoundaryHandler` is used
        directly for more control.

        Parameters
        ----------
        boundary_conditions : list[BoundaryCondition]
            List of boundary conditions, some of which may specify
            applied forces or pressures.

        Returns
        -------
        np.ndarray
            Shape ``(n_dof,)`` global force vector.

        See Also
        --------
        wrinklefe.solver.boundary.BoundaryHandler.get_force_dofs
        """
        from wrinklefe.solver.boundary import BoundaryHandler

        handler = BoundaryHandler(self.mesh)
        return handler.get_force_dofs(boundary_conditions)
