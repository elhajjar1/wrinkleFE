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

from wrinklefe.core.mesh import MeshData
from wrinklefe.core.laminate import Laminate
from wrinklefe.core.transforms import stress_transformation_3d


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

        n_elem, n_gp, _ = self.stress_global.shape
        result = np.empty((n_elem, n_gp))

        for e in range(n_elem):
            for g in range(n_gp):
                sv = self.stress_global[e, g]
                # Reconstruct symmetric 3x3 tensor from Voigt vector
                # Voigt: [s11, s22, s33, t23, t13, t12]
                tensor = np.array([
                    [sv[0], sv[5], sv[4]],
                    [sv[5], sv[1], sv[3]],
                    [sv[4], sv[3], sv[2]],
                ])
                eigvals = np.linalg.eigvalsh(tensor)
                result[e, g] = eigvals[-1]  # largest eigenvalue

        self._max_principal = result
        return self._max_principal

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

        # Compute element centroids
        elem_centers = np.empty((self.mesh.n_elements, 3))
        for e in range(self.mesh.n_elements):
            elem_centers[e] = self.mesh.element_center(e)

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
        """Extract interlaminar stress components at ply interfaces.

        Computes sigma_33 (through-thickness normal), tau_13, and tau_23
        at each ply interface by averaging stresses from elements
        immediately above and below the interface.

        Returns
        -------
        sigma_33 : np.ndarray
            Shape ``(n_interfaces,)`` through-thickness normal stress (MPa)
            at each ply interface (averaged over elements sharing the interface).
        tau_13 : np.ndarray
            Shape ``(n_interfaces,)`` transverse shear stress tau_13 (MPa).
        tau_23 : np.ndarray
            Shape ``(n_interfaces,)`` transverse shear stress tau_23 (MPa).

        Notes
        -----
        Interlaminar stresses are critical for delamination prediction.
        Interface *k* lies between ply *k* (below) and ply *k+1* (above).
        """
        n_plies = self.laminate.n_plies
        n_interfaces = n_plies - 1

        if n_interfaces <= 0 or self.stress_global.size == 0:
            empty = np.empty(0)
            return empty, empty.copy(), empty.copy()

        sigma_33 = np.empty(n_interfaces)
        tau_13 = np.empty(n_interfaces)
        tau_23 = np.empty(n_interfaces)

        for k in range(n_interfaces):
            # Elements in ply below and above the interface
            elems_below = self.mesh.elements_in_ply(k)
            elems_above = self.mesh.elements_in_ply(k + 1)

            # Average stress over all Gauss points in these elements
            stresses = []
            if elems_below.size > 0:
                # Take the mean over Gauss points for each element, then mean over elements
                stresses.append(self.stress_global[elems_below].mean(axis=(0, 1)))
            if elems_above.size > 0:
                stresses.append(self.stress_global[elems_above].mean(axis=(0, 1)))

            if stresses:
                avg_stress = np.mean(stresses, axis=0)  # (6,)
            else:
                avg_stress = np.zeros(6)

            sigma_33[k] = avg_stress[2]  # sigma_33
            tau_23[k] = avg_stress[3]    # tau_23
            tau_13[k] = avg_stress[4]    # tau_13

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
        This uses midplane-column stresses and trapezoidal integration.
        Useful for comparing 3D FE results with CLT predictions.
        """
        if self.stress_global.size == 0:
            return np.zeros(3), np.zeros(3)

        # Use domain centre as the evaluation point
        Lx, Ly, _ = self.mesh.domain_size
        x_mid = self.mesh.nodes[:, 0].min() + Lx / 2.0
        y_mid = self.mesh.nodes[:, 1].min() + Ly / 2.0

        # Get through-thickness stress profiles for components 0 (s11), 1 (s22), 5 (t12)
        components = [0, 1, 5]  # sigma_11, sigma_22, tau_12
        N = np.zeros(3)
        M = np.zeros(3)

        for idx, comp in enumerate(components):
            z_vals, s_vals = self.stress_through_thickness(x_mid, y_mid, comp)
            if z_vals.size < 2:
                continue
            # Trapezoidal integration
            N[idx] = np.trapz(s_vals, z_vals)
            M[idx] = np.trapz(s_vals * z_vals, z_vals)

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

    def reaction_forces(
        self,
        K_global: sparse.csc_matrix,
        constrained_dofs: dict[int, float],
    ) -> np.ndarray:
        """Compute reaction forces at constrained DOFs.

        The reaction force vector is:

        .. math::

            R = K \\cdot u - F_{ext}

        Since the external force at constrained DOFs is typically zero
        (displacement BCs), the reaction at those DOFs is simply ``K @ u``
        restricted to the constrained rows.

        Parameters
        ----------
        K_global : scipy.sparse.csc_matrix
            Global stiffness matrix (before BC modification).
        constrained_dofs : dict[int, float]
            Mapping from global DOF index to prescribed displacement value.

        Returns
        -------
        np.ndarray
            Shape ``(n_constrained, 4)`` array where each row is
            ``[dof_index, Rx, Ry, Rz]``.  For a single-DOF constraint,
            only the constrained component is non-zero, but the full
            node reaction is returned when all 3 DOFs of a node are constrained.
        """
        u_flat = self.displacement.ravel()
        R_full = K_global @ u_flat  # full residual

        dof_indices = sorted(constrained_dofs.keys())
        result = np.empty((len(dof_indices), 2))
        for i, dof in enumerate(dof_indices):
            result[i, 0] = dof
            result[i, 1] = R_full[dof]

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
