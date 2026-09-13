"""Post-processing and field results container for FE analysis.

Provides :class:`FieldResults`, a structured container for displacement, stress,
and strain fields from a static (or other) finite element solution.  Derived
quantities such as von Mises stress, principal stresses, interlaminar stresses,
and CLT-equivalent resultants are computed lazily on first access.

Stress and strain use Voigt notation throughout::

    [sigma_11, sigma_22, sigma_33, tau_23, tau_13, tau_12]

References
----------
Zienkiewicz, O.C. & Taylor, R.L. (2000). The Finite Element Method, Vol. 1.
Jones, R.M. (1999). Mechanics of Composite Materials, 2nd ed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from wrinklefe.core.laminate import Laminate
from wrinklefe.core.mesh import MeshData


@dataclass
class FieldResults:
    """Complete solution field data from an FE analysis.

    Contains displacement, stress, and strain at nodes and/or Gauss points,
    plus derived quantities computed lazily.

    Parameters
    ----------
    displacement : np.ndarray
        Shape ``(n_nodes, 3)`` nodal displacements (ux, uy, uz) in mm.
    stress_global : np.ndarray
        Shape ``(n_elements, n_gauss, 6)`` stress in global coordinates (MPa).
        Voigt ordering: [sigma_11, sigma_22, sigma_33, tau_23, tau_13, tau_12].
    stress_local : np.ndarray
        Shape ``(n_elements, n_gauss, 6)`` stress in local material coordinates (MPa).
    strain_global : np.ndarray
        Shape ``(n_elements, n_gauss, 6)`` engineering strain in global coordinates.
    strain_local : np.ndarray
        Shape ``(n_elements, n_gauss, 6)`` engineering strain in local material coordinates.
    mesh : MeshData
        Reference to the finite element mesh.
    laminate : Laminate
        Reference to the laminate definition.

    Notes
    -----
    Derived quantities (von Mises stress, principal stresses) are computed
    lazily on first access via ``@property`` methods.  This keeps the
    object lightweight when only displacements are needed.
    """

    # Primary solution
    displacement: np.ndarray           # (n_nodes, 3)

    # Element-level results at Gauss points
    stress_global: np.ndarray          # (n_elements, n_gauss, 6)
    stress_local: np.ndarray           # (n_elements, n_gauss, 6)
    strain_global: np.ndarray          # (n_elements, n_gauss, 6)
    strain_local: np.ndarray           # (n_elements, n_gauss, 6)

    # Mesh reference
    mesh: MeshData = field(repr=False)
    laminate: Laminate = field(repr=False)

    # Derived quantities (computed lazily)
    _von_mises: np.ndarray | None = field(default=None, repr=False)
    _max_principal: np.ndarray | None = field(default=None, repr=False)
    _elem_centers: np.ndarray | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Derived stress quantities
    # ------------------------------------------------------------------

    @property
    def von_mises(self) -> np.ndarray:
        """Von Mises equivalent stress at each Gauss point.

        .. math::

            \\sigma_{vm} = \\sqrt{
                \\sigma_{11}^2 + \\sigma_{22}^2 + \\sigma_{33}^2
                - \\sigma_{11}\\sigma_{22} - \\sigma_{22}\\sigma_{33}
                - \\sigma_{11}\\sigma_{33}
                + 3(\\tau_{12}^2 + \\tau_{23}^2 + \\tau_{13}^2)
            }

        Returns
        -------
        np.ndarray
            Shape ``(n_elements, n_gauss)`` von Mises stress (MPa).
        """
        if self._von_mises is not None:
            return self._von_mises

        if self.stress_global.size == 0:
            self._von_mises = np.empty((0, 0))
            return self._von_mises

        s = self.stress_global  # (n_elem, n_gp, 6)
        s11 = s[:, :, 0]
        s22 = s[:, :, 1]
        s33 = s[:, :, 2]
        t23 = s[:, :, 3]
        t13 = s[:, :, 4]
        t12 = s[:, :, 5]

        vm_sq = (
            s11**2 + s22**2 + s33**2
            - s11 * s22 - s22 * s33 - s11 * s33
            + 3.0 * (t12**2 + t23**2 + t13**2)
        )
        # Guard against small negative values from floating-point arithmetic
        vm_sq = np.maximum(vm_sq, 0.0)
        self._von_mises = np.sqrt(vm_sq)
        return self._von_mises

    @property
    def max_principal_stress(self) -> np.ndarray:
        """Maximum principal stress at each Gauss point.

        Computed from eigenvalues of the 3x3 symmetric stress tensor
        reconstructed from the 6-component Voigt vector.

        Returns
        -------
        np.ndarray
            Shape ``(n_elements, n_gauss)`` maximum principal stress (MPa).
        """
        if self._max_principal is not None:
            return self._max_principal

        if self.stress_global.size == 0:
            self._max_principal = np.empty((0, 0))
            return self._max_principal

        # Reconstruct all symmetric 3x3 tensors from the Voigt vectors by
        # slice assignment and solve them in one batched eigvalsh call
        # (issue #295) — replaces a Python double loop that called LAPACK
        # once per Gauss point.
        # Voigt: [s11, s22, s33, t23, t13, t12]
        s = self.stress_global  # (n_elem, n_gp, 6)
        n_elem, n_gp, _ = s.shape
        tensors = np.empty((n_elem, n_gp, 3, 3), dtype=np.float64)
        tensors[..., 0, 0] = s[..., 0]
        tensors[..., 1, 1] = s[..., 1]
        tensors[..., 2, 2] = s[..., 2]
        tensors[..., 0, 1] = tensors[..., 1, 0] = s[..., 5]
        tensors[..., 0, 2] = tensors[..., 2, 0] = s[..., 4]
        tensors[..., 1, 2] = tensors[..., 2, 1] = s[..., 3]

        eigvals = np.linalg.eigvalsh(tensors)  # (n_elem, n_gp, 3), ascending
        self._max_principal = np.ascontiguousarray(eigvals[..., -1])
        return self._max_principal

    @property
    def element_centers(self) -> np.ndarray:
        """Centroids of all elements, computed once and cached (#295).

        Returns
        -------
        np.ndarray
            Shape ``(n_elements, 3)`` centroid coordinates (mm).
            Equivalent to stacking ``mesh.element_center(e)`` for every
            element, but built with one vectorised gather-and-mean so
            repeated through-thickness / column queries do not rebuild
            the array.
        """
        if self._elem_centers is not None:
            return self._elem_centers
        if self.mesh.n_elements == 0:
            self._elem_centers = np.empty((0, 3))
            return self._elem_centers
        self._elem_centers = self.mesh.nodes[self.mesh.elements].mean(axis=1)
        return self._elem_centers

    # ------------------------------------------------------------------
    # Through-thickness queries
    # ------------------------------------------------------------------

    def stress_through_thickness(
        self, x: float, y: float, component: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract stress component vs z at a specific (x, y) location.

        Finds the column of elements nearest to ``(x, y)`` and extracts
        the average stress at the centroid z-level of each element.

        Parameters
        ----------
        x, y : float
            Physical coordinates (mm).
        component : int, optional
            Stress component index in Voigt notation:
            0 = sigma_11, 1 = sigma_22, 2 = sigma_33,
            3 = tau_23, 4 = tau_13, 5 = tau_12.
            Default is 0.

        Returns
        -------
        z_values : np.ndarray
            Shape ``(n_through,)`` z-coordinate at element centroids (mm).
        stress_values : np.ndarray
            Shape ``(n_through,)`` stress component values (MPa).
        """
        if self.stress_global.size == 0:
            return np.empty(0), np.empty(0)

        # Element centroids — computed once per FieldResults and cached
        # (issue #295); repeated column queries reuse the array.
        elem_centers = self.element_centers

        # Find elements closest to (x, y) in the x-y plane
        xy_dist = np.sqrt(
            (elem_centers[:, 0] - x) ** 2 + (elem_centers[:, 1] - y) ** 2
        )
        min_dist = xy_dist.min()
        tol = min_dist + 1.0e-6 * (xy_dist.max() - min_dist + 1.0e-30)
        column_mask = xy_dist <= tol

        column_indices = np.flatnonzero(column_mask)
        if column_indices.size == 0:
            return np.empty(0), np.empty(0)

        # Extract z and average stress at each element in the column
        z_values = elem_centers[column_indices, 2]
        # Average over Gauss points within each element
        stress_values = self.stress_global[column_indices, :, component].mean(axis=1)

        # Sort by z
        order = np.argsort(z_values)
        return z_values[order], stress_values[order]

    def interlaminar_stresses(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Peak interlaminar stresses at each ply interface.

        For every interface, the Gauss points **on the interface side** of
        the two adjoining element layers are collected — the upper half of
        the elements below, the lower half of those above — and each
        component is reduced to the signed value of largest magnitude
        across the interface plane.

        Returns
        -------
        sigma_33 : np.ndarray
            Shape ``(n_interfaces,)`` peak through-thickness normal stress
            (MPa, signed) at each ply interface.
        tau_13 : np.ndarray
            Shape ``(n_interfaces,)`` peak transverse shear ``tau_13`` (MPa).
        tau_23 : np.ndarray
            Shape ``(n_interfaces,)`` peak transverse shear ``tau_23`` (MPa).

        Notes
        -----
        Interface *k* lies between ply *k* (below) and ply *k+1* (above).

        **This used to return a plane average, and that number was
        meaningless.**  The previous implementation took the mean over
        *all* Gauss points of *all* elements in the two adjoining plies —
        a domain average of a field whose mean is ~0 by equilibrium.  On a
        flat laminate it reported exactly 0.0000 MPa where the free-edge
        peak is 24.8 MPa; on a wrinkled one, 0.92 MPa against a peak of
        24.5 MPa, a 26x under-report.  Since delamination is driven by the
        peak and not the average, and since a wrinkled mesh — this
        package's whole subject — never has a uniform interlaminar field,
        the reduction is now an extremum.

        Each component is reduced independently, so the three values may
        come from different points on the interface.
        """
        n_plies = self.laminate.n_plies
        n_interfaces = n_plies - 1

        if n_interfaces <= 0 or self.stress_global.size == 0:
            empty = np.empty(0)
            return empty, empty.copy(), empty.copy()

        n_gp = self.stress_global.shape[1]
        # 2x2x2 Gauss points alternate zeta = -/+ 1/sqrt(3), so the odd
        # indices are the element's upper half and the even ones its lower
        # half.  Anything else (a different rule) falls back to all points.
        if n_gp == 8:
            upper_half = np.arange(1, 8, 2)   # zeta > 0
            lower_half = np.arange(0, 8, 2)   # zeta < 0
        else:  # pragma: no cover - defensive, only 2x2x2 is produced today
            upper_half = lower_half = np.arange(n_gp)

        sigma_33 = np.zeros(n_interfaces)
        tau_13 = np.zeros(n_interfaces)
        tau_23 = np.zeros(n_interfaces)

        for k in range(n_interfaces):
            samples = []
            below = self.mesh.elements_in_ply(k)
            if below.size > 0:
                samples.append(
                    self.stress_global[np.ix_(below, upper_half)].reshape(-1, 6)
                )
            above = self.mesh.elements_in_ply(k + 1)
            if above.size > 0:
                samples.append(
                    self.stress_global[np.ix_(above, lower_half)].reshape(-1, 6)
                )
            if not samples:
                continue
            at_interface = np.concatenate(samples, axis=0)  # (n_pts, 6)

            for out, comp in ((sigma_33, 2), (tau_23, 3), (tau_13, 4)):
                col = at_interface[:, comp]
                finite = np.isfinite(col)
                if finite.any():
                    col = col[finite]
                    out[k] = float(col[np.argmax(np.abs(col))])

        return sigma_33, tau_13, tau_23

    # ------------------------------------------------------------------
    # CLT-equivalent resultants
    # ------------------------------------------------------------------

    def equivalent_resultants(self) -> tuple[np.ndarray, np.ndarray]:
        """Integrate stresses through thickness to get CLT-equivalent resultants.

        Performs numerical integration of the stress field through the
        laminate thickness at the element centroid column nearest to the
        domain centre.

        .. math::

            N_i = \\int_{-h/2}^{h/2} \\sigma_i \\, dz \\quad (i = 1, 2, 6)

            M_i = \\int_{-h/2}^{h/2} \\sigma_i \\cdot z \\, dz

        Returns
        -------
        N : np.ndarray
            Shape ``(3,)`` — force resultants [Nx, Ny, Nxy] (N/mm).
        M : np.ndarray
            Shape ``(3,)`` — moment resultants [Mx, My, Mxy] (N*mm/mm).

        Notes
        -----
        Useful for comparing 3-D FE results with CLT predictions.

        The integration is a **midpoint rule over each element's own
        through-thickness extent**, which is exact for the
        piecewise-constant stress the recovery produces.

        **This replaces a trapezoid over element centroids, which violated
        equilibrium.**  Sampling at centroids spans only ``h - t`` rather
        than ``h``, so the outer half-element at each surface was omitted
        entirely, and a trapezoid additionally smooths across the stress
        jump at every ply boundary.  For ``[0, 90, 90, 0]`` the exact sum
        ``s1+s2+s3+s4`` degenerated to ``s1/2+s2+s3+s4/2`` — dropping half
        of each stiff surface ply.  Recovered ``Nx`` against an applied
        ``-100``:

        * ``nz_per_ply=1``: was ``-52.5`` (0.52x), now ``-100`` to
          solver tolerance
        * ``nz_per_ply=2``: was ``-76.2`` (0.76x), now ``-100``
        * ``nz_per_ply=4``: was ``-92.4`` (0.92x), now ``-100``

        The old form converged only as O(1/nz); the new one is exact at
        any mesh density.
        """
        if self.stress_global.size == 0:
            return np.zeros(3), np.zeros(3)

        # Use domain centre as the evaluation point
        Lx, Ly, _ = self.mesh.domain_size
        x_mid = self.mesh.nodes[:, 0].min() + Lx / 2.0
        y_mid = self.mesh.nodes[:, 1].min() + Ly / 2.0

        centres = self.element_centers
        xy_dist = (centres[:, 0] - x_mid) ** 2 + (centres[:, 1] - y_mid) ** 2
        near = xy_dist.min()
        tol = near + 1.0e-6 * (xy_dist.max() - near + 1.0e-30)
        column = np.flatnonzero(xy_dist <= tol)
        if column.size == 0:
            return np.zeros(3), np.zeros(3)
        column = column[np.argsort(centres[column, 2])]

        # Each element's OWN through-thickness extent, from its nodes.
        z_mid = np.empty(column.size)
        dz = np.empty(column.size)
        for i, e in enumerate(column):
            z_nodes = self.mesh.element_nodes(int(e))[:, 2]
            lo, hi = float(z_nodes.min()), float(z_nodes.max())
            z_mid[i] = 0.5 * (lo + hi)
            dz[i] = hi - lo
        if not np.all(dz > 0.0):
            return np.zeros(3), np.zeros(3)

        # sigma_11, sigma_22, tau_12, averaged over each element's Gauss points
        sigma = self.stress_global[column][:, :, [0, 1, 5]].mean(axis=1)

        # The equidistance test can tie: when the domain centre falls on a
        # node or an edge, two or four stacks are all "nearest".  Collapse
        # them by z level, otherwise the sum below counts the thickness
        # once per tied stack (a clean 2x or 4x on the resultants).
        levels, inverse = np.unique(
            np.round(z_mid, 9), return_inverse=True,
        )
        n_levels = levels.size
        counts = np.bincount(inverse, minlength=n_levels).astype(float)
        sigma_lvl = np.zeros((n_levels, 3))
        for j in range(3):
            sigma_lvl[:, j] = (
                np.bincount(inverse, weights=sigma[:, j], minlength=n_levels)
                / counts
            )
        dz_lvl = np.bincount(inverse, weights=dz, minlength=n_levels) / counts

        # Midpoint rule over each element's own slab: EXACT for the
        # piecewise-constant field the recovery actually produces.
        N = (sigma_lvl * dz_lvl[:, None]).sum(axis=0)
        M = (sigma_lvl * (levels * dz_lvl)[:, None]).sum(axis=0)
        return N, M

    # ------------------------------------------------------------------
    # Scalar queries
    # ------------------------------------------------------------------

    def max_displacement(self) -> tuple[float, int]:
        """Maximum displacement magnitude and its node index.

        Returns
        -------
        mag : float
            Maximum displacement magnitude (mm).
        node_idx : int
            Node index where maximum occurs.
        """
        magnitudes = np.linalg.norm(self.displacement, axis=1)
        node_idx = int(np.argmax(magnitudes))
        return float(magnitudes[node_idx]), node_idx

    def max_stress(
        self, component: int = 0, coord: str = 'local'
    ) -> tuple[float, int, int]:
        """Maximum stress value, element index, and Gauss point index.

        Parameters
        ----------
        component : int, optional
            Stress component in Voigt notation (0-5). Default is 0 (sigma_11).
        coord : str, optional
            ``'global'`` or ``'local'`` coordinate system. Default is ``'local'``.

        Returns
        -------
        value : float
            Maximum stress value (MPa).
        elem_idx : int
            Element index where maximum occurs.
        gp_idx : int
            Gauss point index within the element.
        """
        if coord == 'local':
            arr = self.stress_local
        elif coord == 'global':
            arr = self.stress_global
        else:
            raise ValueError(f"coord must be 'global' or 'local', got {coord!r}")

        if arr.size == 0:
            return 0.0, 0, 0

        comp_data = arr[:, :, component]  # (n_elem, n_gp)
        flat_idx = int(np.argmax(np.abs(comp_data)))
        n_gp = comp_data.shape[1]
        elem_idx = flat_idx // n_gp
        gp_idx = flat_idx % n_gp
        return float(comp_data[elem_idx, gp_idx]), elem_idx, gp_idx

    def max_displacement_location(self) -> tuple[float, np.ndarray]:
        """Maximum displacement magnitude and its physical location.

        Coordinate-aware companion to :meth:`max_displacement`: resolves the
        node index to its ``(x, y, z)`` coordinates via the mesh, so
        reporting / visualisation need no manual index -> coordinate lookup.

        Returns
        -------
        mag : float
            Maximum displacement magnitude (mm).
        location : np.ndarray
            Shape ``(3,)`` node coordinates ``(x, y, z)`` (mm) where the
            maximum occurs (undeformed reference coordinates).
        """
        mag, node_idx = self.max_displacement()
        return mag, np.asarray(self.mesh.nodes[node_idx], dtype=float)

    def max_stress_location(
        self, component: int = 0, coord: str = 'local'
    ) -> tuple[float, np.ndarray]:
        """Maximum stress value and the physical location of its element.

        Coordinate-aware companion to :meth:`max_stress`: resolves the
        element index to its centroid ``(x, y, z)`` via the mesh.

        Parameters
        ----------
        component : int, optional
            Stress component in Voigt notation (0-5). Default is 0 (sigma_11).
        coord : str, optional
            ``'global'`` or ``'local'`` coordinate system. Default ``'local'``.

        Returns
        -------
        value : float
            Maximum stress value (MPa).
        location : np.ndarray
            Shape ``(3,)`` centroid coordinates ``(x, y, z)`` (mm) of the
            element containing the maximum.
        """
        value, elem_idx, _gp_idx = self.max_stress(
            component=component, coord=coord
        )
        if self.mesh.n_elements == 0:
            return value, np.zeros(3, dtype=float)
        return value, np.asarray(
            self.mesh.element_center(elem_idx), dtype=float
        )

    def reaction_forces(
        self,
        K_global: sparse.csc_matrix,
        constrained_dofs: dict[int, float],
    ) -> np.ndarray:
        """Compute reaction forces at nodes with constrained DOFs.

        The reaction force vector is:

        .. math::

            R = K \\cdot u - F_{ext}

        Since the external force at constrained DOFs is typically zero
        (displacement BCs), the reaction at those DOFs is simply ``K @ u``
        restricted to the constrained rows.

        Results are grouped by node (using the convention ``dof = 3 * node_id + d``
        with ``d in {0, 1, 2}`` for ``Rx, Ry, Rz``).  For a node where only some
        of the three DOFs are constrained, the unconstrained components are
        reported as ``0.0``.

        Parameters
        ----------
        K_global : scipy.sparse.csc_matrix
            Global stiffness matrix (before BC modification).
        constrained_dofs : dict[int, float]
            Mapping from global DOF index to prescribed displacement value.

        Returns
        -------
        np.ndarray
            Shape ``(n_nodes_with_reactions, 4)`` array where each row is
            ``[node_id, Rx, Ry, Rz]``.  Rows are sorted by ``node_id``.
        """
        u_flat = self.displacement.ravel()
        R_full = np.asarray(K_global @ u_flat).ravel()  # full residual

        # Group constrained DOFs by node id (dof = 3 * node + d).
        node_reactions: dict[int, np.ndarray] = {}
        for dof in constrained_dofs:
            node_id = int(dof) // 3
            d = int(dof) % 3
            if node_id not in node_reactions:
                node_reactions[node_id] = np.zeros(3, dtype=float)
            node_reactions[node_id][d] = float(R_full[int(dof)])

        node_ids = sorted(node_reactions.keys())
        result = np.empty((len(node_ids), 4), dtype=float)
        for i, node_id in enumerate(node_ids):
            result[i, 0] = node_id
            result[i, 1:4] = node_reactions[node_id]

        return result

    def strain_energy(self) -> float:
        """Total strain energy of the model.

        .. math::

            U = \\frac{1}{2} \\sum_{\\text{elem}} \\sum_{\\text{gp}}
                \\boldsymbol{\\sigma}^T \\boldsymbol{\\varepsilon} \\, |J| \\, w

        For computational efficiency, this approximates the integration
        weight by dividing the element volume equally among Gauss points.

        Returns
        -------
        float
            Total strain energy (N*mm = mJ).
        """
        if self.stress_global.size == 0 or self.strain_global.size == 0:
            return 0.0

        n_elem, n_gp, _ = self.stress_global.shape

        # sigma^T * epsilon at each Gauss point
        # (n_elem, n_gp) via element-wise dot product over the 6 components
        sigma_eps = np.sum(
            self.stress_global * self.strain_global, axis=2
        )  # (n_elem, n_gp)

        # Approximate: distribute element volume equally among Gauss points
        # For more precise results, the caller should pass Gauss-weighted volumes.
        # Here we use element volumes computed from mesh geometry.
        total_energy = 0.0
        for e in range(n_elem):
            # Compute element volume from node coordinates using the Hex8Element
            # volume property would require creating elements.  Instead, estimate
            # from the 8-node bounding box as a simple approximation.
            node_coords = self.mesh.element_nodes(e)  # (8, 3)
            # Simple volume estimate: product of extents (exact for rectangular elements)
            extents = node_coords.max(axis=0) - node_coords.min(axis=0)
            vol_approx = float(extents[0] * extents[1] * extents[2])
            w_gp = vol_approx / n_gp
            total_energy += float(np.sum(sigma_eps[e]) * w_gp)

        return 0.5 * total_energy

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Generate a text summary of key results.

        Returns
        -------
        str
            Multi-line summary including max displacement, max stresses
            (each component), max strains, and total strain energy.
        """
        lines = ["=" * 60, "  FE Analysis Results Summary", "=" * 60]

        # Displacement
        max_disp, max_disp_node = self.max_displacement()
        lines.append(
            f"  Max displacement: {max_disp:.6e} mm "
            f"(node {max_disp_node})"
        )
        lines.append(
            f"    Components at max node: "
            f"ux={self.displacement[max_disp_node, 0]:.6e}, "
            f"uy={self.displacement[max_disp_node, 1]:.6e}, "
            f"uz={self.displacement[max_disp_node, 2]:.6e}"
        )

        # Stress (local coordinates)
        labels = ["sigma_11", "sigma_22", "sigma_33", "tau_23", "tau_13", "tau_12"]
        lines.append("")
        lines.append("  Max stress (local coords, absolute value):")
        for comp, label in enumerate(labels):
            if self.stress_local.size > 0:
                val, e_idx, gp_idx = self.max_stress(comp, coord='local')
                lines.append(
                    f"    {label:>10s}: {val:12.4f} MPa  "
                    f"(elem {e_idx}, gp {gp_idx})"
                )
            else:
                lines.append(f"    {label:>10s}:  (no data)")

        # Strain
        lines.append("")
        lines.append("  Max strain (global coords, absolute value):")
        strain_labels = ["eps_11", "eps_22", "eps_33", "gamma_23", "gamma_13", "gamma_12"]
        for comp, label in enumerate(strain_labels):
            if self.strain_global.size > 0:
                comp_data = self.strain_global[:, :, comp]
                flat_idx = int(np.argmax(np.abs(comp_data)))
                n_gp = comp_data.shape[1]
                e_idx = flat_idx // n_gp
                gp_idx = flat_idx % n_gp
                val = float(comp_data[e_idx, gp_idx])
                lines.append(
                    f"    {label:>10s}: {val:12.6e}  "
                    f"(elem {e_idx}, gp {gp_idx})"
                )
            else:
                lines.append(f"    {label:>10s}:  (no data)")

        # Strain energy
        U = self.strain_energy()
        lines.append("")
        lines.append(f"  Total strain energy: {U:.6e} N*mm")
        lines.append("=" * 60)

        return "\n".join(lines)
