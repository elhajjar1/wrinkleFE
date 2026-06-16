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

import copy

import numpy as np
from scipy import sparse

from wrinklefe.core.mesh import MeshData
from wrinklefe.core.laminate import Laminate
from wrinklefe.elements.cohesive8 import (
    Cohesive8Element,
    CohesiveState,
    make_initial_state,
)
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
        cohesive_elements: (
            list[tuple[int, Cohesive8Element]] | None
        ) = None,
    ) -> None:
        if element_type not in ("hex8",):
            raise ValueError(
                f"Unsupported element type '{element_type}'. "
                f"Currently only 'hex8' is supported."
            )
        self.mesh = mesh
        self.laminate = laminate
        self.element_type = element_type

        # Pre-build hex8 elements once so per-Newton-iteration assembly
        # doesn't pay the construction cost (finding #15).  Element state
        # is captured at construction time; the mesh is treated as
        # immutable for the life of the assembler.
        n_elem = self.mesh.n_elements
        self._hex8_elements: list[Hex8Element] = [
            self.create_element(e) for e in range(n_elem)
        ]
        self._hex8_dofs: list[np.ndarray] = [
            self.element_dof_indices(e) for e in range(n_elem)
        ]

        # Cohesive element bookkeeping.  Each entry is keyed by the
        # caller-supplied element index (a stable identifier independent
        # of the hex8 element numbering), and carries pre-built per-GP
        # history state for both the committed and trial (in-Newton-loop)
        # views.
        self.cohesive_elements: list[tuple[int, Cohesive8Element]] = list(
            cohesive_elements or []
        )
        self.cohesive_state: dict[int, list[CohesiveState]] = {}
        self.cohesive_state_trial: dict[int, list[CohesiveState]] = {}
        n_nodes = self.mesh.nodes.shape[0]
        for e_idx, c_elem in self.cohesive_elements:
            if c_elem.node_ids is None:
                raise ValueError(
                    f"Cohesive element at index {e_idx} has no node_ids; "
                    "pass node_ids to the Cohesive8Element constructor."
                )
            if np.any(c_elem.node_ids >= n_nodes):
                raise ValueError(
                    f"Cohesive element {e_idx}: node_ids out of range; "
                    f"mesh has {n_nodes} nodes (max valid index "
                    f"{n_nodes - 1}), got max(node_ids) = "
                    f"{int(np.max(c_elem.node_ids))}."
                )
            # Element-wise exact check: catches mismatches between
            # mesh.nodes[node_ids] and the node_coords the user passed to
            # the element constructor.
            # NOTE: this CANNOT detect node_id permutations among
            # coincident nodes (common in cohesive split-mesh setups
            # where bottom and top faces share xyz in the reference
            # configuration).  A swap within the bottom face (e.g.,
            # [4,5,6,7,8,9,10,11] -> [5,4,6,7,8,9,10,11]) is invisible
            # to this check.  Callers must ensure node ordering follows
            # the documented face convention (rows 0-3 = bottom, CCW
            # from -,-; rows 4-7 = top).
            if not np.array_equal(
                self.mesh.nodes[c_elem.node_ids], c_elem.node_coords
            ):
                raise ValueError(
                    f"Cohesive element {e_idx}: mesh.nodes[node_ids] does "
                    "not match the node_coords passed to the element "
                    "constructor — likely permutation error or stale "
                    "coordinates."
                )
            self.cohesive_state[e_idx] = make_initial_state(c_elem.n_gp)
            self.cohesive_state_trial[e_idx] = make_initial_state(c_elem.n_gp)

        # Cache the linear hex8 stiffness matrices once.  These depend
        # only on geometry/material — both fixed at construction — so
        # there is no point recomputing them per Newton iteration.
        self._hex8_Ke: list[np.ndarray] = [
            elem.stiffness_matrix() for elem in self._hex8_elements
        ]

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

        # Material from the laminate's ply stack — overridden by the
        # resin-pocket material for elements inside the resin lens.
        ply_material = self.laminate.plies[ply_idx].material
        material = self.mesh.element_material(elem_idx, ply_material)

        # Wrinkle misalignment angles at the 8 element nodes (radians),
        # scaled by the resin-pocket retention factor: a fibre-free resin
        # centre carries no misalignment (scale 0), the lens boundary the
        # full angle (scale 1), so the wrinkle defect is counted once.
        angle_scale = self.mesh.resin_angle_scale(elem_idx)
        wrinkle_angles = angle_scale * self.mesh.fiber_angles[node_ids]

        return Hex8Element(
            node_coords=node_coords,
            material=material,
            ply_angle=ply_angle,
            wrinkle_angles=wrinkle_angles,
        )

    def update_element(self, elem_idx: int) -> None:
        """Rebuild one element's cached object and stiffness after a material
        change.

        The progressive-damage solver mutates
        ``mesh.element_material_override`` as elements fail; calling this
        for each changed element refreshes the cached ``Hex8Element`` and
        its 24x24 stiffness so the next :meth:`assemble_stiffness` picks up
        the degraded material without rebuilding the whole assembler.

        Parameters
        ----------
        elem_idx : int
            Element index (0-based) to refresh.
        """
        elem = self.create_element(elem_idx)
        self._hex8_elements[elem_idx] = elem
        self._hex8_Ke[elem_idx] = elem.stiffness_matrix()

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
        """Assemble the linear global stiffness matrix K (hex8 only).

        This is the LINEAR stiffness path and does not include cohesive
        contributions.  When cohesive elements are registered with this
        assembler, callers must use :meth:`assemble_tangent` (or
        :meth:`assemble_residual_and_tangent`) instead — see the runtime
        check below.

        Algorithm
        ---------
        1. Pre-allocate COO arrays (rows, cols, values).
           Each hex8 element contributes 24 x 24 = 576 entries.
           Total pre-allocated non-zeros = ``n_elements * 576``.
        2. Loop over elements:
           a. Look up the cached element stiffness ``Ke`` (24 x 24).
           b. Get global DOF indices for element nodes.
           c. Fill COO arrays with Ke entries at global DOF positions.
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
            Shape ``(n_dof, n_dof)`` linear global stiffness matrix in
            CSC format.

        Raises
        ------
        RuntimeError
            If cohesive elements are registered.  Use
            :meth:`assemble_tangent` (consistent tangent including the
            cohesive contribution) for nonlinear CZM problems.
        """
        if self.cohesive_elements:
            raise RuntimeError(
                "GlobalAssembler.assemble_stiffness() is the linear "
                "stiffness path and does not include cohesive "
                "contributions. When cohesive elements are registered, "
                "use NewtonRaphsonSolver (which calls assemble_tangent) "
                "instead of StaticSolver."
            )
        return self._assemble_hex8_stiffness(verbose=verbose)

    def assemble_geometric_stiffness(
        self, displacement: np.ndarray
    ) -> sparse.csc_matrix:
        """Assemble the global geometric (initial-stress) stiffness K_geo.

        Evaluates each hex8 element's geometric stiffness from the pre-
        stress state implied by *displacement* (the linear-static
        solution) and scatters into a global sparse matrix.  Used by the
        linearized-buckling microbuckling knockdown (item D.4): the
        buckling load factors solve ``K phi = -lambda K_geo phi``.

        Parameters
        ----------
        displacement : np.ndarray
            Global nodal displacement vector ``(n_dof,)`` defining the
            pre-stress.

        Returns
        -------
        scipy.sparse.csc_matrix
            ``(n_dof, n_dof)`` geometric stiffness matrix.
        """
        u = np.asarray(displacement, dtype=float).ravel()
        n_elem = self.mesh.n_elements
        n_dof = self.mesh.n_dof
        entries = 24 * 24
        coo_rows = np.empty(n_elem * entries, dtype=np.intp)
        coo_cols = np.empty(n_elem * entries, dtype=np.intp)
        coo_vals = np.empty(n_elem * entries, dtype=np.float64)
        local_ii, local_jj = np.meshgrid(
            np.arange(24), np.arange(24), indexing="ij"
        )
        local_ii = local_ii.ravel()
        local_jj = local_jj.ravel()

        for e in range(n_elem):
            dofs = self._hex8_dofs[e]
            u_elem = u[dofs]
            Kg = self._hex8_elements[e].geometric_stiffness_matrix(u_elem)
            off = e * entries
            coo_rows[off:off + entries] = dofs[local_ii]
            coo_cols[off:off + entries] = dofs[local_jj]
            coo_vals[off:off + entries] = Kg.ravel()

        return sparse.coo_matrix(
            (coo_vals, (coo_rows, coo_cols)), shape=(n_dof, n_dof)
        ).tocsc()

    def _assemble_hex8_stiffness(
        self, verbose: bool = False
    ) -> sparse.csc_matrix:
        """Hex8-only linear stiffness, no cohesive contribution.

        Shared by :meth:`assemble_stiffness` and
        :meth:`assemble_tangent` / :meth:`assemble_residual_and_tangent`.
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

            Ke = self._hex8_Ke[e]  # cached at construction time
            dofs = self._hex8_dofs[e]  # (24,)

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
    # Consistent tangent (Newton-Raphson)
    # ------------------------------------------------------------------

    def assemble_tangent(self, u: np.ndarray) -> sparse.csc_matrix:
        """Assemble the consistent global tangent at ``u``.

        Sibling to :meth:`assemble_stiffness` for the nonlinear solver:
        identical to the linear hex8 path, plus per-Gauss-point cohesive-
        element contributions linearised at the current trial state.
        Updates ``self.cohesive_state_trial`` as a side effect.

        Parameters
        ----------
        u : np.ndarray
            Shape ``(n_dof,)`` global displacement vector.

        Returns
        -------
        scipy.sparse.csc_matrix
            Shape ``(n_dof, n_dof)`` global tangent in CSC format.
        """
        u = np.asarray(u, dtype=float).reshape(-1)
        K = self._assemble_hex8_stiffness()

        if not self.cohesive_elements:
            return K

        n_dof = self.mesh.n_dof
        coh_rows: list[np.ndarray] = []
        coh_cols: list[np.ndarray] = []
        coh_vals: list[np.ndarray] = []
        for e_idx, c_elem in self.cohesive_elements:
            dofs = self._cohesive_dof_indices(c_elem)
            u_e = u[dofs]
            K_e, _F_e, state_new = c_elem.tangent_and_force(
                u_e, state_prev=self.cohesive_state[e_idx]
            )
            self.cohesive_state_trial[e_idx] = state_new

            ii, jj = np.meshgrid(dofs, dofs, indexing="ij")
            coh_rows.append(ii.ravel())
            coh_cols.append(jj.ravel())
            coh_vals.append(K_e.ravel())

        K_coh = sparse.coo_matrix(
            (
                np.concatenate(coh_vals),
                (np.concatenate(coh_rows), np.concatenate(coh_cols)),
            ),
            shape=(n_dof, n_dof),
        ).tocsc()
        return (K + K_coh).tocsc()

    def assemble_residual_and_tangent(
        self, u: np.ndarray
    ) -> tuple[sparse.csc_matrix, np.ndarray]:
        """Combined consistent tangent + internal force in one pass.

        Calling :meth:`assemble_internal_force` and :meth:`assemble_tangent`
        back-to-back would invoke each cohesive element's
        :meth:`tangent_and_force` twice with identical arguments.  This
        method runs a single pass per cohesive element and returns both
        outputs, halving the cohesive-law cost per Newton iteration.

        Parameters
        ----------
        u : np.ndarray
            Shape ``(n_dof,)`` global displacement vector.

        Returns
        -------
        K_t : scipy.sparse.csc_matrix
            Shape ``(n_dof, n_dof)`` global tangent.
        F_int : np.ndarray
            Shape ``(n_dof,)`` global internal force vector.
        """
        u = np.asarray(u, dtype=float).reshape(-1)
        n_dof = self.mesh.n_dof

        # Linear hex8 stiffness + F_int = K_e @ u_e per element.
        K_linear = self._assemble_hex8_stiffness()
        F_int = np.zeros(n_dof)
        for e, _elem in enumerate(self._hex8_elements):
            dofs = self._hex8_dofs[e]
            Ke = self._hex8_Ke[e]
            F_int[dofs] += Ke @ u[dofs]

        if not self.cohesive_elements:
            return K_linear, F_int

        coh_rows: list[np.ndarray] = []
        coh_cols: list[np.ndarray] = []
        coh_vals: list[np.ndarray] = []
        for e_idx, c_elem in self.cohesive_elements:
            dofs = self._cohesive_dof_indices(c_elem)
            u_e = u[dofs]
            K_e, F_e, state_new = c_elem.tangent_and_force(
                u_e, state_prev=self.cohesive_state[e_idx]
            )
            self.cohesive_state_trial[e_idx] = state_new
            F_int[dofs] += F_e

            ii, jj = np.meshgrid(dofs, dofs, indexing="ij")
            coh_rows.append(ii.ravel())
            coh_cols.append(jj.ravel())
            coh_vals.append(K_e.ravel())

        K_coh = sparse.coo_matrix(
            (
                np.concatenate(coh_vals),
                (np.concatenate(coh_rows), np.concatenate(coh_cols)),
            ),
            shape=(n_dof, n_dof),
        ).tocsc()
        return (K_linear + K_coh).tocsc(), F_int

    # ------------------------------------------------------------------
    # Cohesive history-state management
    # ------------------------------------------------------------------

    def commit_state(self) -> None:
        """Promote trial cohesive state to committed state.

        Called by the nonlinear solver at the end of a converged
        increment.  Deep-copies each ``CohesiveState`` so the committed
        view cannot be mutated by later trial updates — using
        :func:`copy.deepcopy` so the contract remains correct if
        ``CohesiveState`` ever grows nested fields.
        """
        self.cohesive_state = {
            k: [copy.deepcopy(s) for s in v]
            for k, v in self.cohesive_state_trial.items()
        }

    def revert_state(self) -> None:
        """Reset trial cohesive state to the last committed state.

        Called by the nonlinear solver on a failed increment so the next
        attempt starts from clean history.  Uses :func:`copy.deepcopy`
        for the same forward-compatibility reason as :meth:`commit_state`.
        """
        self.cohesive_state_trial = {
            k: [copy.deepcopy(s) for s in v]
            for k, v in self.cohesive_state.items()
        }

    # ------------------------------------------------------------------
    # Cohesive DOF mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _cohesive_dof_indices(c_elem: Cohesive8Element) -> np.ndarray:
        """Build the (24,) global DOF index list from a cohesive element.

        The cohesive element carries ``node_ids`` — the global node
        indices in the same order as ``node_coords`` — which are set in
        the :class:`Cohesive8Element` constructor.
        """
        if c_elem.node_ids is None:
            raise ValueError(
                "Cohesive8Element instance is missing `node_ids` "
                "(global node indices); pass node_ids to the "
                "Cohesive8Element constructor."
            )
        node_ids = c_elem.node_ids
        dofs = np.empty(24, dtype=np.intp)
        for i, nid in enumerate(node_ids):
            base = 3 * int(nid)
            dofs[3 * i] = base
            dofs[3 * i + 1] = base + 1
            dofs[3 * i + 2] = base + 2
        return dofs

    # ------------------------------------------------------------------
    # Force vector assembly
    # ------------------------------------------------------------------

    def assemble_internal_force(self, u: np.ndarray) -> np.ndarray:
        """Assemble the global internal force vector ``F_int(u)``.

        For the linear hex8 path this is ``K_e @ u_e`` per element.
        Cohesive elements use the path-dependent
        :meth:`Cohesive8Element.tangent_and_force`; their trial history
        state is stored in ``self.cohesive_state_trial`` for the next
        ``commit_state`` call.

        Parameters
        ----------
        u : np.ndarray
            Shape ``(n_dof,)`` global displacement vector.

        Returns
        -------
        np.ndarray
            Shape ``(n_dof,)`` internal force vector.
        """
        u = np.asarray(u, dtype=float).reshape(-1)
        n_elem = self.mesh.n_elements
        n_dof = self.mesh.n_dof
        F = np.zeros(n_dof)

        for e in range(n_elem):
            Ke = self._hex8_Ke[e]
            dofs = self._hex8_dofs[e]
            ue = u[dofs]
            F[dofs] += Ke @ ue

        for e_idx, c_elem in self.cohesive_elements:
            dofs = self._cohesive_dof_indices(c_elem)
            u_e = u[dofs]
            _K_e, F_e, state_new = c_elem.tangent_and_force(
                u_e, state_prev=self.cohesive_state[e_idx]
            )
            self.cohesive_state_trial[e_idx] = state_new
            F[dofs] += F_e

        return F

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
