"""Linearized microbuckling solver (geometric nonlinearity, item D.4).

Compressive failure of a wrinkled unidirectional laminate is, at root, a
geometric instability: under axial compression the misaligned fibres
rotate further until the matrix can no longer restrain them (fibre
kinking / microbuckling).  A small-strain static solve at a fixed
misalignment cannot capture this — it freezes the geometry — which is why
the progressive-damage FE under-predicts the knockdown of severe wrinkles
(Li 2025 S-M-3).

This module supplies the linearized form of that instability.  Given the
pre-stress from a reference linear solve, the critical load factor solves
the generalized eigenproblem

    K phi = -lambda K_geo phi,

where ``K`` is the material stiffness and ``K_geo`` the geometric
(initial-stress) stiffness assembled from the pre-stress.  The smallest
positive ``lambda`` scales the reference load to the buckling load.  A
wrinkle imperfection lowers ``lambda`` relative to the pristine laminate,
so the **microbuckling knockdown** is

    KD = lambda_crit(wrinkled) / lambda_crit(pristine).

Unlike the progressive-damage path this is a single eigenvalue solve (no
load stepping, no arc-length), and it captures the rotation-amplification
the linear stress analysis misses.

.. note::

   **Negative finding (item D.4): this is not the production UD wrinkle
   predictor.** Calibrated against the Li (2025) grids, the linear
   eigenvalue *over-predicts* the knockdown badly (e.g. F
   0.30 / 0.06 / 0.38 vs measured 0.89 / 0.63 / 0.47) for two physical
   reasons: (1) the wrinkled structure is imperfection-sensitive (Koiter
   — the bifurcation load sits far below the limit load), and (2)
   buckling of the homogenised ply-mesh is local *structural* buckling of
   the soft wrinkle region, not the sub-ply *fibre kinking* that actually
   governs. ``K_geo`` is correct, tested infrastructure for genuine
   structural-buckling analyses and is retained as such, but the
   unidirectional wrinkle knockdown should be taken from the
   :mod:`~wrinklefe.core.penetration_gate` instead. See the wrinkle
   modelling findings (item D.4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import eigsh

from wrinklefe.core.laminate import Laminate
from wrinklefe.core.mesh import MeshData
from wrinklefe.solver.boundary import BoundaryHandler
from wrinklefe.solver.static import StaticSolver


@dataclass
class BucklingResult:
    """Outcome of a linearized buckling solve.

    Attributes
    ----------
    critical_load_factor : float
        Smallest positive eigenvalue ``lambda`` scaling the reference load
        to the buckling load (``inf`` if no positive mode was found).
    load_factors : np.ndarray
        The computed eigenvalues (load factors), ascending.
    """

    critical_load_factor: float
    load_factors: np.ndarray


class LinearBucklingSolver:
    """Linearized (eigenvalue) microbuckling solver.

    Parameters
    ----------
    mesh : MeshData
        The (wrinkled or flat) hex8 mesh.
    laminate : Laminate
        Laminate definition.
    applied_strain : float
        Reference compressive strain for the pre-stress (negative).
        Default ``-0.01``.  The returned load factor scales this state.
    n_modes : int
        Number of eigenvalues to extract.  Default 6.
    solver : str
        Linear backend for the reference static solve.  Default
        ``"direct"``.
    """

    def __init__(
        self,
        mesh: MeshData,
        laminate: Laminate,
        *,
        applied_strain: float = -0.01,
        n_modes: int = 6,
        solver: str = "direct",
    ) -> None:
        self.mesh = mesh
        self.laminate = laminate
        self.applied_strain = float(applied_strain)
        self.n_modes = int(n_modes)
        self.solver = solver

    def solve(self) -> BucklingResult:
        """Compute the critical microbuckling load factor."""
        mesh = self.mesh
        static = StaticSolver(mesh, self.laminate)
        bcs = BoundaryHandler.compression_bcs(
            mesh, applied_strain=self.applied_strain
        )
        field = static.solve(
            bcs, solver=self.solver, keep_stiffness=True, verbose=False
        )
        K = static._K  # material stiffness, pre-BC
        assert K is not None
        u = field.displacement.ravel()

        Kg = static.assembler.assemble_geometric_stiffness(u)

        # Homogeneous kinematic constraints (the buckling perturbation is
        # zero where the static solve prescribes a displacement).
        bc_handler = BoundaryHandler(mesh)
        constrained = bc_handler.get_constrained_dofs(bcs)
        n_dof = K.shape[0]
        free = np.setdiff1d(
            np.arange(n_dof, dtype=np.intp),
            np.fromiter(constrained.keys(), dtype=np.intp,
                        count=len(constrained)),
        )
        K_ff = K[free][:, free].tocsc()
        Kg_ff = Kg[free][:, free].tocsc()

        # Buckling eigenproblem K phi = -lambda K_geo phi.  Under
        # compression K_geo is negative-definite, so -K_geo is SPD and the
        # eigenvalues are positive.  Shift-invert near 0 returns the
        # smallest-magnitude (first) modes.
        M = (-Kg_ff).tocsc()
        k = min(self.n_modes, K_ff.shape[0] - 2)
        try:
            vals = eigsh(
                K_ff, k=k, M=M, sigma=0.0, which="LM",
                return_eigenvectors=False,
            )
        except Exception:
            # Fall back to a regular (non-shift-invert) smallest-algebraic
            # search if the factorization is singular.
            vals = eigsh(
                K_ff, k=k, M=M, which="SM", return_eigenvectors=False,
            )

        vals = np.sort(np.real(vals))
        positive = vals[vals > 1e-9]
        crit = float(positive[0]) if positive.size else float("inf")
        return BucklingResult(critical_load_factor=crit, load_factors=vals)


def microbuckling_knockdown(
    wrinkled_mesh: MeshData,
    pristine_mesh: MeshData,
    laminate: Laminate,
    *,
    applied_strain: float = -0.01,
    n_modes: int = 6,
) -> float:
    """Microbuckling knockdown ``lambda_wrinkled / lambda_pristine``.

    Both meshes must share the laminate and boundary setup; only the
    wrinkle geometry differs.  Returns the ratio of critical load factors,
    capped at 1.0 (a wrinkle cannot raise the buckling load).
    """
    lam_w = LinearBucklingSolver(
        wrinkled_mesh, laminate, applied_strain=applied_strain,
        n_modes=n_modes,
    ).solve().critical_load_factor
    lam_0 = LinearBucklingSolver(
        pristine_mesh, laminate, applied_strain=applied_strain,
        n_modes=n_modes,
    ).solve().critical_load_factor
    if not np.isfinite(lam_0) or lam_0 <= 0:
        return 1.0
    return min(1.0, lam_w / lam_0)
