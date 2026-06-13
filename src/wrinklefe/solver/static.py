"""Linear static finite element solver for composite laminates.

Solves the equilibrium equation :math:`K u = F` for the displacement field,
then recovers stresses and strains at Gauss points in both global and local
(material) coordinate systems.

The solver supports:

- **Direct** solution via ``scipy.sparse.linalg.spsolve`` (robust for any size).
- **Iterative** solution via conjugate gradient with ILU preconditioner
  (memory-efficient for large problems with >100 k DOFs).
- Automatic conversion from a CLT :class:`~wrinklefe.core.laminate.LoadState`
  to 3-D boundary conditions.

Workflow
--------
1. Create ``StaticSolver(mesh, laminate)``
2. Define boundary conditions (list of ``BoundaryCondition``)
3. Call ``solver.solve(bcs)`` to get :class:`~wrinklefe.solver.results.FieldResults`

References
----------
Bathe, K.-J. (2006). Finite Element Procedures.
Zienkiewicz, O.C. & Taylor, R.L. (2000). The Finite Element Method, Vol. 1.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


@contextmanager
def _suppress_assembly_warnings():
    """Suppress divide-by-zero and invalid-value warnings during assembly."""
    with np.errstate(divide='ignore', invalid='ignore'):
        yield

from wrinklefe.core.mesh import MeshData
from wrinklefe.core.laminate import Laminate, LoadState
from wrinklefe.core.transforms import stress_transformation_3d
from wrinklefe.solver.assembler import GlobalAssembler
from wrinklefe.solver.results import FieldResults

if TYPE_CHECKING:
    from wrinklefe.solver.boundary import BoundaryHandler, BoundaryCondition


class StaticSolver:
    """Linear static finite element solver.

    Solves :math:`K \\cdot u = F` for the displacement field, then
    recovers stresses and strains at Gauss points.

    Parameters
    ----------
    mesh : MeshData
        Finite element mesh.
    laminate : Laminate
        Laminate definition with ply materials and orientations.
    element_type : str, optional
        Element formulation: ``'hex8'`` (standard 2x2x2 integration).
        Default is ``'hex8'``.
    """

    def __init__(
        self,
        mesh: MeshData,
        laminate: Laminate,
        element_type: str = "hex8",
    ) -> None:
        self.mesh = mesh
        self.laminate = laminate
        self.element_type = element_type
        self.assembler = GlobalAssembler(mesh, laminate, element_type)

        # Populated after solve
        self._displacement: np.ndarray | None = None
        self._K: sparse.csc_matrix | None = None
        self._constrained_dofs: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Main solve interface
    # ------------------------------------------------------------------

    def solve(
        self,
        boundary_conditions: list[BoundaryCondition],
        solver: str = "direct",
        verbose: bool = False,
        keep_stiffness: bool = False,
    ) -> FieldResults:
        """Solve the static problem.

        Steps
        -----
        1. Assemble global stiffness matrix K.
        2. Assemble force vector F from boundary conditions.
        3. Apply displacement BCs via the penalty method.
        4. Solve K u = F.
        5. Post-process: recover stresses and strains.

        Parameters
        ----------
        boundary_conditions : list[BoundaryCondition]
            List of boundary conditions (displacement and force BCs).
        solver : str, optional
            ``'direct'`` uses ``spsolve``; ``'iterative'`` uses CG with
            ILU preconditioner. Default is ``'direct'``.
        verbose : bool, optional
            Print progress information. Default is ``False``.
        keep_stiffness : bool, optional
            If True, retain a copy of the unmodified (pre-penalty) global
            stiffness matrix on ``self._K`` after solve. Default is False,
            which avoids the memory cost of holding a full sparse matrix
            copy on every solver instance (relevant for parametric sweeps
            and analyses that keep multiple solvers alive). Enable only
            when callers need post-solve access to K (e.g., to compute
            reaction forces without re-assembly).

        Returns
        -------
        FieldResults
            Complete solution with displacement, stress, and strain fields.

        Raises
        ------
        RuntimeError
            If the iterative solver fails to converge.
        """
        from wrinklefe.solver.boundary import BoundaryHandler

        t0 = time.perf_counter()

        # 1. Assemble global stiffness
        if verbose:
            print("Assembling global stiffness matrix...")
        with _suppress_assembly_warnings():
            K = self.assembler.assemble_stiffness(verbose=verbose)
        # Opt-in retention of the unmodified K (see ``keep_stiffness``).
        # Default: do not store a copy to avoid doubling peak FE memory.
        self._K = K.copy() if keep_stiffness else None

        if verbose:
            t1 = time.perf_counter()
            print(f"  Assembly time: {t1 - t0:.2f} s")

        # 2. Assemble force vector
        bc_handler = BoundaryHandler(self.mesh)
        F = bc_handler.get_force_dofs(boundary_conditions)

        # 3. Apply displacement BCs via penalty method
        self._constrained_dofs = bc_handler.get_constrained_dofs(
            boundary_conditions
        )
        K, F = self._apply_penalty_bcs(K, F, self._constrained_dofs, verbose)

        # 4. Solve
        if verbose:
            print(f"Solving system ({self.mesh.n_dof} DOFs, "
                  f"solver={solver})...")

        if solver == "direct":
            u = self._solve_direct(K, F, verbose=verbose)
        elif solver == "iterative":
            u = self._solve_iterative(K, F, verbose=verbose)
        else:
            raise ValueError(
                f"Unknown solver '{solver}'. Use 'direct' or 'iterative'."
            )

        if verbose:
            t2 = time.perf_counter()
            print(f"  Solve time: {t2 - t1:.2f} s")
            t1 = t2

        # 5. Post-process
        if verbose:
            print("Recovering element stresses and strains...")

        stress_g, stress_l, strain_g, strain_l = self.recover_element_results(
            u, verbose=verbose
        )

        # Reshape displacement to (n_nodes, 3)
        displacement = u.reshape(-1, 3)
        self._displacement = displacement

        if verbose:
            t3 = time.perf_counter()
            print(f"  Post-processing time: {t3 - t1:.2f} s")
            print(f"Total solve time: {t3 - t0:.2f} s")

        return FieldResults(
            displacement=displacement,
            stress_global=stress_g,
            stress_local=stress_l,
            strain_global=strain_g,
            strain_local=strain_l,
            mesh=self.mesh,
            laminate=self.laminate,
        )

    def solve_load_state(
        self,
        load: LoadState,
        solver: str = "direct",
        verbose: bool = False,
    ) -> FieldResults:
        """Convenience method: solve from a CLT LoadState.

        Converts the CLT force and moment resultants into 3-D boundary
        conditions on the mesh faces:

        - **x_min** face: fully clamped (ux = uy = uz = 0).
        - **x_max** face: uniform traction derived from Nx, Ny, Nxy
          distributed over the face area.
        - **y_min** / **y_max** faces: free (natural BC) unless Ny or Nxy
          are non-zero (handled through x_max traction).
        - **z_min** / **z_max** faces: free.

        Parameters
        ----------
        load : LoadState
            CLT load state (Nx, Ny, Nxy, Mx, My, Mxy).
        solver : str, optional
            ``'direct'`` or ``'iterative'``. Default is ``'direct'``.
        verbose : bool, optional
            Print progress. Default is ``False``.

        Returns
        -------
        FieldResults
            Complete solution data.
        """
        from wrinklefe.solver.boundary import BoundaryCondition

        bcs = self._load_state_to_bcs(load)
        return self.solve(bcs, solver=solver, verbose=verbose)

    # ------------------------------------------------------------------
    # Linear algebra solvers
    # ------------------------------------------------------------------

    def _solve_direct(
        self,
        K: sparse.csc_matrix,
        F: np.ndarray,
        verbose: bool = False,
    ) -> np.ndarray:
        """Direct sparse solver using ``scipy.sparse.linalg.spsolve``.

        Parameters
        ----------
        K : scipy.sparse.csc_matrix
            Global stiffness matrix with BCs applied.
        F : np.ndarray
            Shape ``(n_dof,)`` global force vector.
        verbose : bool, optional
            Print solver info.

        Returns
        -------
        np.ndarray
            Shape ``(n_dof,)`` displacement vector.
        """
        u = spla.spsolve(K, F)
        if verbose:
            residual = np.linalg.norm(K @ u - F)
            print(f"  Direct solver residual: {residual:.4e}")
        return u

    def _solve_iterative(
        self,
        K: sparse.csc_matrix,
        F: np.ndarray,
        tol: float = 1e-10,
        maxiter: int = 10000,
        verbose: bool = False,
    ) -> np.ndarray:
        """Iterative CG solver with ILU preconditioner.

        Uses ``scipy.sparse.linalg.cg`` with an incomplete LU
        factorisation as preconditioner, wrapped in a
        ``LinearOperator``.  Suitable for large problems (>100 k DOFs).

        Parameters
        ----------
        K : scipy.sparse.csc_matrix
            Global stiffness matrix with BCs applied.
        F : np.ndarray
            Shape ``(n_dof,)`` global force vector.
        tol : float, optional
            Convergence tolerance for the relative residual. Default 1e-10.
        maxiter : int, optional
            Maximum number of iterations. Default 10000.
        verbose : bool, optional
            Print solver info and iteration count.

        Returns
        -------
        np.ndarray
            Shape ``(n_dof,)`` displacement vector.

        Raises
        ------
        RuntimeError
            If the CG solver fails to converge.
        """
        n = K.shape[0]

        # Build ILU preconditioner
        if verbose:
            print("  Building ILU preconditioner...")
        try:
            ilu = spla.spilu(K, drop_tol=1e-4)
            M_op = spla.LinearOperator(
                shape=(n, n),
                matvec=ilu.solve,
                dtype=K.dtype,
            )
        except Exception:
            # Fall back to diagonal preconditioner if ILU fails
            if verbose:
                print("  ILU failed, falling back to diagonal preconditioner.")
            diag = K.diagonal()
            diag[diag == 0] = 1.0
            M_op = spla.LinearOperator(
                shape=(n, n),
                matvec=lambda x: x / diag,
                dtype=K.dtype,
            )

        # Iteration counter for verbose output
        iter_count = [0]

        def _callback(xk: np.ndarray) -> None:
            iter_count[0] += 1

        callback = _callback if verbose else None

        # SciPy >=1.12 deprecated ``tol=`` in favour of ``rtol=``.
        u, info = spla.cg(K, F, rtol=tol, maxiter=maxiter, M=M_op,
                          callback=callback)

        if info != 0:
            raise RuntimeError(
                f"CG solver did not converge: info={info} "
                f"(iterations={iter_count[0]}, tol={tol})"
            )

        if verbose:
            residual = np.linalg.norm(K @ u - F)
            print(f"  CG converged in {iter_count[0]} iterations, "
                  f"residual: {residual:.4e}")

        return u

    # ------------------------------------------------------------------
    # Boundary condition helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_penalty_bcs(
        K: sparse.csc_matrix,
        F: np.ndarray,
        constrained_dofs: dict[int, float],
        verbose: bool = False,
    ) -> tuple[sparse.csc_matrix, np.ndarray]:
        """Apply displacement boundary conditions via the penalty method.

        Thin wrapper around
        :func:`wrinklefe.solver.boundary.apply_penalty_bcs` that uses
        ``in_place=True`` (this solver owns ``K`` and ``F`` exclusively
        within :meth:`solve`).  See the helper docstring for the math
        and the choice of penalty scaling.
        """
        from wrinklefe.solver.boundary import apply_penalty_bcs, _PENALTY_SCALE

        if not constrained_dofs:
            return K, F

        K_out, F_out = apply_penalty_bcs(
            K, F, constrained_dofs, in_place=True,
        )

        if verbose:
            diag_max = float(np.abs(K.diagonal()).max())
            alpha = _PENALTY_SCALE * max(diag_max, 1.0)
            print(f"  Applied {len(constrained_dofs)} displacement BCs "
                  f"(penalty alpha={alpha:.2e})")

        return K_out, F_out

    def _load_state_to_bcs(self, load: LoadState) -> list:
        """Convert a CLT LoadState to 3-D boundary conditions.

        Parameters
        ----------
        load : LoadState
            CLT-level load state.

        Returns
        -------
        list[BoundaryCondition]
            List of BCs suitable for ``self.solve()``.
        """
        from wrinklefe.solver.boundary import BoundaryCondition

        bcs: list[BoundaryCondition] = []

        # Clamp x_min face: ux = uy = uz = 0
        x_min_nodes = self.mesh.nodes_on_face("x_min")
        bcs.append(
            BoundaryCondition(
                bc_type="fixed",
                node_ids=x_min_nodes,
                dofs=[0, 1, 2],
                value=0.0,
            )
        )

        # Apply traction on x_max face from Nx via a pressure BC so the
        # total face force is distributed by consistent face integration
        # (see boundary.BoundaryHandler.get_force_dofs and issue #50).
        # CLT Nx has units of force per unit width (N/mm), so the total
        # nodal force across the face is Nx * Ly.
        _, Ly, Lz = self.mesh.domain_size
        x_max_has_elements = self.mesh.nx > 0 and self.mesh.ny > 0 and self.mesh.nz > 0

        if x_max_has_elements and not np.isclose(load.Nx, 0.0):
            total_force_x = load.Nx * Ly
            bcs.append(
                BoundaryCondition(
                    bc_type="pressure",
                    face="x_max",
                    dofs=[0],
                    value=total_force_x,
                )
            )

        if x_max_has_elements and not np.isclose(load.Nxy, 0.0):
            total_force_y = load.Nxy * Ly
            bcs.append(
                BoundaryCondition(
                    bc_type="pressure",
                    face="x_max",
                    dofs=[1],
                    value=total_force_y,
                )
            )

        return bcs

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    # Cached (gp_coords, N_gp, dN_dxi_gp) for the standard hex8 / 2x2x2
    # quadrature.  Populated lazily by :meth:`_hex8_gauss_shape_functions`
    # and reused across every solve because shape functions in natural
    # coordinates are identical for every hex8 element (issue #187).
    _hex8_gp_shape_cache: (
        tuple[np.ndarray, np.ndarray, np.ndarray] | None
    ) = None

    @classmethod
    def _hex8_gauss_shape_functions(
        cls,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return cached 2x2x2 Gauss-point coords, shape functions, and derivatives.

        Returns
        -------
        gp_coords : np.ndarray
            Shape ``(n_gp, 3)`` natural coordinates of the Gauss points
            (lexicographic order, matching ``Hex8Element._gauss_points``).
        N_gp : np.ndarray
            Shape ``(n_gp, 8)`` — row ``i`` holds the 8 trilinear shape
            functions evaluated at Gauss point ``i``.  For a per-element
            nodal field ``f`` (shape ``(8,)``), ``N_gp @ f`` returns the
            interpolated value at each Gauss point in one matmul.
        dN_dxi_gp : np.ndarray
            Shape ``(n_gp, 3, 8)`` — natural-coordinate derivatives of the
            8 shape functions at each Gauss point.  Constant across all
            hex8 elements; used to build per-element Jacobians via a
            single batched matmul.
        """
        if cls._hex8_gp_shape_cache is not None:
            return cls._hex8_gp_shape_cache

        from wrinklefe.elements.hex8 import Hex8Element
        from wrinklefe.elements.gauss import gauss_points_hex

        gp_coords, _ = gauss_points_hex(order=2)
        n_gp = gp_coords.shape[0]
        N_gp = np.empty((n_gp, 8))
        dN_dxi_gp = np.empty((n_gp, 3, 8))
        for i, (xi, eta, zeta) in enumerate(gp_coords):
            N_gp[i] = Hex8Element.shape_functions(xi, eta, zeta)
            dN_dxi_gp[i] = Hex8Element.shape_derivatives(xi, eta, zeta)
        cls._hex8_gp_shape_cache = (gp_coords, N_gp, dN_dxi_gp)
        return cls._hex8_gp_shape_cache

    def recover_element_results(
        self,
        displacement: np.ndarray,
        verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Post-process element-level stresses and strains.

        For each element:

        1. Extract element displacements from the global vector.
        2. Compute global-frame stress and strain at each Gauss point.
        3. Transform stress and strain to local material coordinates
           using the ply angle and wrinkle misalignment.

        Quantities that are constant per element (``T_ply``, the element's
        node ids and per-node wrinkle angles) are computed once outside
        the Gauss loop, and all wrinkle angles at the 8 Gauss points are
        obtained via a single ``N_gp @ fiber_angles_local`` matmul.  See
        issue #187 for the performance motivation.

        Parameters
        ----------
        displacement : np.ndarray
            Shape ``(n_dof,)`` global displacement vector.
        verbose : bool, optional
            Print progress. Default is ``False``.

        Returns
        -------
        stress_global : np.ndarray
            Shape ``(n_elements, n_gauss, 6)`` stress in global coordinates.
        stress_local : np.ndarray
            Shape ``(n_elements, n_gauss, 6)`` stress in local material coordinates.
        strain_global : np.ndarray
            Shape ``(n_elements, n_gauss, 6)`` strain in global coordinates.
        strain_local : np.ndarray
            Shape ``(n_elements, n_gauss, 6)`` strain in local material coordinates.
        """
        from wrinklefe.core.transforms import rotate_stiffness_3d

        n_elem = self.mesh.n_elements
        # Shape functions and natural-coord derivatives at the 8 Gauss
        # points are constant across all hex8 elements; build once and
        # reuse (cached on the class).
        _gp_coords, N_gp, dN_dxi_gp = self._hex8_gauss_shape_functions()
        n_gp = N_gp.shape[0]

        stress_global = np.empty((n_elem, n_gp, 6))
        stress_local = np.empty((n_elem, n_gp, 6))
        strain_global = np.empty((n_elem, n_gp, 6))
        strain_local = np.empty((n_elem, n_gp, 6))

        # Pre-fetch mesh arrays so attribute lookups don't happen in the
        # hot loop.
        mesh_elements = self.mesh.elements
        mesh_nodes = self.mesh.nodes
        mesh_fiber_angles = self.mesh.fiber_angles
        mesh_ply_angles = self.mesh.ply_angles
        mesh_ply_ids = self.mesh.ply_ids

        # Material stiffness in the material frame — constant per ply.
        plies = self.laminate.plies
        material_C_by_ply: dict[int, np.ndarray] = {}
        # Resin-pocket zone (Li et al. 2024/2025): elements inside the
        # lens use the isotropic resin stiffness and a zero wrinkle angle.
        mesh_resin_mask = self.mesh.resin_mask
        resin_material = self.mesh.resin_material
        resin_C = (
            resin_material.stiffness_matrix
            if resin_material is not None
            else None
        )
        # Per-element material override (progressive damage): degraded
        # materials keyed by element index, taking precedence over the
        # resin pocket and host ply.  Stiffness cached by material id.
        override = self.mesh.element_material_override
        override_C_cache: dict[int, np.ndarray] = {}

        # Reusable scratch B-matrix template (zeros are stable between GPs
        # at the slots we never touch; we overwrite the populated slots
        # for every GP via vector assignment).
        B_scratch = np.zeros((n_gp, 6, 24))

        for e in range(n_elem):
            if verbose and e % 1000 == 0:
                print(f"  Post-processing element {e}/{n_elem} "
                      f"({100.0 * e / n_elem:.1f}%)")

            node_ids = mesh_elements[e]
            node_coords = mesh_nodes[node_ids]  # (8, 3)

            # Batched Jacobians across all 8 GPs in one matmul.
            J_all = dN_dxi_gp @ node_coords  # (n_gp, 3, 3)
            J_inv_all = np.linalg.inv(J_all)  # (n_gp, 3, 3)
            dN_dx_all = J_inv_all @ dN_dxi_gp  # (n_gp, 3, 8) — physical-coord derivs

            # Build the 6x24 strain-displacement matrix at every GP via
            # vectorised slot assignment (matches Hex8Element.B_matrix).
            B = B_scratch
            # eps_11 = du/dx -> rows 0, cols 0,3,6,...,21
            B[:, 0, 0::3] = dN_dx_all[:, 0, :]
            # eps_22 = dv/dy -> rows 1, cols 1,4,...,22
            B[:, 1, 1::3] = dN_dx_all[:, 1, :]
            # eps_33 = dw/dz -> rows 2, cols 2,5,...,23
            B[:, 2, 2::3] = dN_dx_all[:, 2, :]
            # gamma_23 = dv/dz + dw/dy -> rows 3, cols (1,2,4,5,...)
            B[:, 3, 1::3] = dN_dx_all[:, 2, :]
            B[:, 3, 2::3] = dN_dx_all[:, 1, :]
            # gamma_13 = du/dz + dw/dx -> rows 4
            B[:, 4, 0::3] = dN_dx_all[:, 2, :]
            B[:, 4, 2::3] = dN_dx_all[:, 0, :]
            # gamma_12 = du/dy + dv/dx -> rows 5
            B[:, 5, 0::3] = dN_dx_all[:, 1, :]
            B[:, 5, 1::3] = dN_dx_all[:, 0, :]

            # Element nodal displacements (24,) via flat node-id expansion.
            u_elem = displacement[
                (3 * node_ids[:, None] + np.arange(3)).ravel()
            ]

            # Strain at each GP — one batched matmul.
            eps_g = B @ u_elem  # (n_gp, 6)

            # Rotated stiffness per GP: ply rotation (about z) once per
            # element, then wrinkle rotation (about y) per GP.
            override_mat = (
                override.get(e) if override is not None else None
            )
            is_resin_e = (
                resin_C is not None
                and mesh_resin_mask is not None
                and bool(mesh_resin_mask[e])
            )
            if override_mat is not None:
                # Progressive-damage override wins over resin / ply.  Keep
                # the wrinkle angle unless the override is the (fibre-free)
                # resin material at a lens element.
                oid = id(override_mat)
                if oid not in override_C_cache:
                    override_C_cache[oid] = override_mat.stiffness_matrix
                C_material = override_C_cache[oid]
            elif is_resin_e:
                # Isotropic resin: stiffness is rotation-invariant, so the
                # ply/wrinkle rotations are identities and the wrinkle angle
                # is zeroed (no fibres to kink).
                C_material = resin_C
            else:
                ply_idx = int(mesh_ply_ids[e])
                if ply_idx not in material_C_by_ply:
                    material_C_by_ply[ply_idx] = (
                        plies[ply_idx].material.stiffness_matrix
                    )
                C_material = material_C_by_ply[ply_idx]

            ply_angle_rad = np.radians(float(mesh_ply_angles[e]))
            if abs(ply_angle_rad) > 1.0e-15:
                C_ply = rotate_stiffness_3d(C_material, ply_angle_rad, axis='z')
            else:
                C_ply = C_material

            if is_resin_e and override_mat is None:
                wrinkle_angles_gp = np.zeros(n_gp)
            else:
                fiber_angles_local = mesh_fiber_angles[node_ids]  # (8,)
                # One matmul gives the interpolated wrinkle angle per GP.
                wrinkle_angles_gp = N_gp @ fiber_angles_local  # (n_gp,)

            T_ply = stress_transformation_3d(ply_angle_rad, axis='z')
            sig_g = np.empty((n_gp, 6))
            for g in range(n_gp):
                phi = float(wrinkle_angles_gp[g])
                if abs(phi) > 1.0e-15:
                    C_bar = rotate_stiffness_3d(C_ply, phi, axis='y')
                else:
                    C_bar = C_ply
                sig_g[g] = C_bar @ eps_g[g]

                T_wrinkle = stress_transformation_3d(phi, axis='y')
                T_total = T_wrinkle @ T_ply
                stress_local[e, g] = T_total @ sig_g[g]
                strain_local[e, g] = T_total @ eps_g[g]

            stress_global[e] = sig_g
            strain_global[e] = eps_g

        if verbose:
            print(f"  Post-processing element {n_elem}/{n_elem} (100.0%) -- done.")

        return stress_global, stress_local, strain_global, strain_local

    # Cached (extrapolation_matrix, node_to_gp) for hex8 / 2x2x2.
    # Populated lazily by :meth:`_extrapolate_to_nodes`.
    _hex8_extrap_cache: tuple[np.ndarray, np.ndarray] | None = None

    @classmethod
    def _build_hex8_extrapolation(cls) -> tuple[np.ndarray, np.ndarray]:
        """Build (and cache) the hex8 / 2x2x2 Gauss-to-node extrapolation matrix.

        Two ordering conventions are at play and **must not be conflated**:

        - ``gauss_points_hex(order=2)`` orders the 8 Gauss points
          **lexicographically** in ``(xi, eta, zeta)`` via
          ``np.meshgrid(..., indexing="ij")`` — zeta varies fastest.
        - :data:`wrinklefe.elements.hex8._NODE_COORDS` orders the 8 nodes by
          the **VTK / Abaqus** convention (bottom face CCW, then top face CCW).

        The i-th Gauss point is **not** at the natural coordinate of the
        i-th node.  Pairing them by index alone (a tempting but wrong
        simplification) scrambles the extrapolated nodal field — this is
        the trap reported in issue #51.

        We build the relationship explicitly: for each VTK node *j*, find
        the lexicographic Gauss-point index ``node_to_gp[j]`` whose natural
        coordinates have the same sign pattern as node *j*.  That gives
        the 1-to-1 nearest-neighbour mapping the caller usually wants for
        diagnostics, and is used by the regression test in
        ``tests/test_solver/test_extrapolate.py``.

        The extrapolation itself uses the full inverse shape-function
        matrix (not just the nearest-neighbour pairing), which is exact
        for a tri-linear field:

        .. math::

            \\mathbf{N}_{gp}[i, j] = N_j(\\xi_{gp,i})
            \\quad\\Rightarrow\\quad
            \\mathbf{f}_{nodes} = \\mathbf{N}_{gp}^{-1} \\, \\mathbf{f}_{gp}

        where row *i* uses **lex** Gauss-point ordering and column *j* uses
        **VTK** node ordering — so the output is naturally indexed by VTK
        node order, matching ``mesh.elements`` connectivity.

        Returns
        -------
        N_inv : np.ndarray
            Shape ``(8, 8)`` — extrapolation matrix; ``f_nodes = N_inv @ f_gp``
            where ``f_gp`` is in **lex** order and ``f_nodes`` is in **VTK**
            order.
        node_to_gp : np.ndarray
            Shape ``(8,)`` integer array; ``node_to_gp[j]`` is the lex GP
            index nearest to VTK node *j*.  For the default conventions
            this is ``[0, 4, 6, 2, 1, 5, 7, 3]``.
        """
        if cls._hex8_extrap_cache is not None:
            return cls._hex8_extrap_cache

        from wrinklefe.elements.hex8 import Hex8Element, _NODE_COORDS
        from wrinklefe.elements.gauss import gauss_points_hex

        gp_coords, _ = gauss_points_hex(order=2)  # lex order, shape (8, 3)

        # Build node->GP nearest mapping by matching sign patterns.  Each
        # GP sits at the centroid of one of the 8 sub-cubes; for a hex8
        # the unambiguous pairing is by sign of (xi, eta, zeta).
        node_to_gp = np.empty(8, dtype=int)
        for j in range(8):
            target = _NODE_COORDS[j]  # (+/-1, +/-1, +/-1)
            # Use sign-match: argmin over distance is equivalent here and
            # is robust to any future change in the GP magnitudes.
            sign_match = np.all(
                np.sign(gp_coords) == np.sign(target)[None, :], axis=1
            )
            matches = np.flatnonzero(sign_match)
            if matches.size != 1:
                raise RuntimeError(
                    "Failed to pair hex8 VTK node {} with a unique 2x2x2 "
                    "Gauss point (matches={}). Did the GP or node ordering "
                    "convention change?".format(j, matches.tolist())
                )
            node_to_gp[j] = int(matches[0])

        # Sanity check: mapping must be a permutation of 0..7.
        if sorted(node_to_gp.tolist()) != list(range(8)):
            raise RuntimeError(
                "hex8 node->GP mapping is not a permutation: "
                f"{node_to_gp.tolist()}"
            )

        # Build shape-function matrix: row i uses lex GP i, col j is VTK node j.
        N_gp = np.empty((8, 8))
        for i in range(8):
            xi, eta, zeta = gp_coords[i]
            N_gp[i] = Hex8Element.shape_functions(xi, eta, zeta)
        N_inv = np.linalg.inv(N_gp)

        cls._hex8_extrap_cache = (N_inv, node_to_gp)
        return cls._hex8_extrap_cache

    def _extrapolate_to_nodes(self, gauss_values: np.ndarray) -> np.ndarray:
        """Extrapolate values from 2x2x2 Gauss points to 8 hex nodes.

        Uses the inverse of the shape function matrix evaluated at the
        Gauss points.  For a hex8 element with 2x2x2 Gauss quadrature
        this is exact for any tri-linear field.

        **Ordering contract** (see :meth:`_build_hex8_extrapolation` for
        the full discussion of issue #51):

        - ``gauss_values`` is indexed by **lexicographic** Gauss-point
          order — exactly what ``Hex8Element._gauss_points`` and
          ``stress_at_gauss_points`` produce.
        - The returned nodal array is indexed by **VTK / Abaqus** node
          order — exactly what ``mesh.elements`` connectivity expects.

        These two orders are *not* the same; a naive "pair GP i with node
        i" simplification would scramble the result spatially.

        Parameters
        ----------
        gauss_values : np.ndarray
            Shape ``(8,)`` or ``(8, n_components)`` values at the 8 Gauss
            points, in lexicographic order.

        Returns
        -------
        np.ndarray
            Same shape as ``gauss_values`` — extrapolated nodal values in
            VTK node order.

        Notes
        -----
        The Gauss points for the 2-point rule are at
        ``xi = +/- 1/sqrt(3) ~ +/- 0.57735``.  The extrapolation matrix
        is constant across all hex8 elements and is cached on the class.
        """
        gauss_values = np.asarray(gauss_values, dtype=float)
        squeeze = False
        if gauss_values.ndim == 1:
            gauss_values = gauss_values[:, np.newaxis]
            squeeze = True
        if gauss_values.shape[0] != 8:
            raise ValueError(
                "gauss_values must have 8 rows (one per 2x2x2 Gauss point), "
                f"got shape {gauss_values.shape}."
            )

        N_inv, _node_to_gp = self._build_hex8_extrapolation()
        nodal = N_inv @ gauss_values
        return nodal[:, 0] if squeeze else nodal
