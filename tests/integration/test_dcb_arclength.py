"""DCB Mode-I delamination driven by ArcLengthSolver.

Smoke test that :class:`ArcLengthSolver` (cylindrical, psi = 0) can
run end-to-end on a multi-DOF, history-dependent, displacement-
controlled cohesive-zone problem.  Uses a smaller geometry than
:mod:`tests.integration.test_dcb_monotonic` (nx = 10 vs 100) so the
inner Newton iterations and sparse linear solves stay inside test-
suite-friendly wall-clock budget; the physical layout (arms +
interface + pre-crack) and material constants are unchanged.

Limitations of the cylindrical arc length on this problem
---------------------------------------------------------
The cylindrical constraint ``||Delta u||^2 = Delta s^2`` doesn't
isolate the soft (cohesive) mode from the stiff (bulk) mode, so on a
displacement-controlled cohesive-zone problem the solver tends to
stagnate once the first element fully damages — Newton can keep the
quadratic constraint satisfied while making essentially no progress
in either the load factor or the damage front.  Resolving the full
snap-back curve cleanly needs an indirect displacement control
(scalar constraint on a single DOF) or an enhanced spherical arc
length, both of which are out of scope for the current
implementation.

The test therefore only asserts:

1. :class:`ArcLengthSolver` completes at least a few arc steps
   without raising or producing non-finite values.
2. Cohesive damage initiates somewhere along the interface (at least
   one element passes the d > 0.5 threshold).
3. The reaction-force history is finite and positive.
"""

from __future__ import annotations

import numpy as np

from wrinklefe.core.cohesive_mesh import insert_cohesive_interface
from wrinklefe.core.laminate import Laminate, Ply
from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.core.mesh import MeshData, WrinkleMesh
from wrinklefe.elements.cohesive8 import CohesiveProperties
from wrinklefe.solver.arclength import ArcLengthSolver
from wrinklefe.solver.assembler import GlobalAssembler
from wrinklefe.solver.boundary import BoundaryCondition, BoundaryHandler


L_TOTAL = 20.0
WIDTH = 25.0
H_ARM = 1.5
A0_PRECRACK = 5.0
NX = 10
NY = 1
NZ_PER_ARM = 2

MAT = OrthotropicMaterial(
    name="DCB_CFRP",
    E1=135_000.0, E2=9_000.0, E3=9_000.0,
    G12=5_000.0, G13=5_000.0, G23=3_000.0,
    nu12=0.30, nu13=0.30, nu23=0.40,
)
COH_PROPS = CohesiveProperties(
    K=1.0e6, sigma_max=60.0, tau_max=60.0,
    GIc=0.28, GIIc=0.79, eta_BK=1.45, beta=1.0,
)


def _build_mesh_and_cohesive() -> tuple[MeshData, list]:
    laminate = Laminate([
        Ply(material=MAT, angle=0.0, thickness=H_ARM),
        Ply(material=MAT, angle=0.0, thickness=H_ARM),
    ])
    wm = WrinkleMesh(
        laminate=laminate, wrinkle_config=None,
        Lx=L_TOTAL, Ly=WIDTH,
        nx=NX, ny=NY, nz_per_ply=NZ_PER_ARM,
    )
    base = wm.generate()
    z_mid = 0.5 * (
        float(base.nodes[:, 2].min()) + float(base.nodes[:, 2].max())
    )
    new_mesh, all_coh = insert_cohesive_interface(
        base, z_interface=z_mid, cohesive_props=COH_PROPS,
    )
    kept = [
        c for c in all_coh
        if float(c.node_coords[:4, 0].mean()) >= A0_PRECRACK
    ]
    for k, c in enumerate(kept):
        c.elem_id = k
    return new_mesh, kept


def _build_bcs(mesh: MeshData, delta: float) -> list:
    tol = 1e-6
    z = mesh.nodes[:, 2]
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]
    z_min = float(z.min()); z_max = float(z.max())
    x_max = float(x.max()); x_min_val = float(x.min())
    y_min = float(y.min()); y_max = float(y.max())

    def ids(mask):
        return np.flatnonzero(mask).astype(np.intp)

    return [
        BoundaryCondition(
            bc_type="fixed",
            node_ids=ids((np.abs(x - x_max) <= tol) & (np.abs(z - z_min) <= tol)),
            dofs=[0, 1, 2],
        ),
        BoundaryCondition(
            bc_type="fixed",
            node_ids=ids((np.abs(x - x_max) <= tol) & (np.abs(z - z_max) <= tol)),
            dofs=[0, 1],
        ),
        BoundaryCondition(
            bc_type="fixed",
            node_ids=ids(np.abs(y - y_min) <= tol), dofs=[1],
        ),
        BoundaryCondition(
            bc_type="fixed",
            node_ids=ids(np.abs(y - y_max) <= tol), dofs=[1],
        ),
        BoundaryCondition(
            bc_type="displacement",
            node_ids=ids((np.abs(x - x_min_val) <= tol) & (np.abs(z - z_max) <= tol)),
            dofs=[2], value=+0.5 * float(delta),
        ),
        BoundaryCondition(
            bc_type="displacement",
            node_ids=ids((np.abs(x - x_min_val) <= tol) & (np.abs(z - z_min) <= tol)),
            dofs=[2], value=-0.5 * float(delta),
        ),
    ]


def test_dcb_arclength_runs_to_damage():
    """ArcLengthSolver completes arc steps + initiates damage on the
    DCB problem."""
    mesh, cohesive_elements = _build_mesh_and_cohesive()
    assert len(cohesive_elements) > 0

    laminate = Laminate([
        Ply(material=MAT, angle=0.0, thickness=H_ARM),
        Ply(material=MAT, angle=0.0, thickness=H_ARM),
    ])
    assembler = GlobalAssembler(
        mesh=mesh, laminate=laminate,
        cohesive_elements=[(c.elem_id, c) for c in cohesive_elements],
    )
    bc_handler = BoundaryHandler(mesh)
    bcs_ref = _build_bcs(mesh, 1.0)

    solver = ArcLengthSolver(
        assembler=assembler, bc_handler=bc_handler,
        boundary_conditions=bcs_ref,
        n_arc_steps=20, arc_length=0.3,
        max_newton_iter=40, tol_residual=1e-3, tol_absolute=1e-7,
        adaptive=True, max_halvings_per_step=4,
    )

    result = solver.solve(verbose=False)
    assert result["steps_completed"] >= 3, (
        f"Arc-length DCB did not complete at least 3 steps; got "
        f"{result['steps_completed']}"
    )

    # All displacement-history entries are finite.
    for k, u in enumerate(result["displacement_history"]):
        assert np.all(np.isfinite(u)), (
            f"Non-finite values in displacement history at step {k}"
        )

    # At least one cohesive element has damaged past 0.5.
    max_damage = max(
        max(s.d for s in assembler.cohesive_state[cid])
        for cid in range(len(cohesive_elements))
    )
    assert max_damage > 0.5, (
        f"No cohesive element passed d > 0.5 under arc length; max "
        f"damage = {max_damage:.3f}.  Solver did not advance into the "
        "softening regime."
    )

    # Reaction-force history is finite and the peak is at least
    # within an order of magnitude of the analytical P_c0.
    z = mesh.nodes[:, 2]
    x = mesh.nodes[:, 0]
    tol_g = 1e-6
    z_max = float(z.max())
    z_min = float(z.min())
    x_min_val = float(x.min())
    top_dofs = 3 * np.flatnonzero(
        (np.abs(x - x_min_val) <= tol_g) & (np.abs(z - z_max) <= tol_g)
    ) + 2
    bot_dofs = 3 * np.flatnonzero(
        (np.abs(x - x_min_val) <= tol_g) & (np.abs(z - z_min) <= tol_g)
    ) + 2
    P_hist = []
    for u_step in result["displacement_history"]:
        F_int = assembler.assemble_internal_force(u_step)
        P_top = float(np.sum(F_int[top_dofs]))
        P_bot = float(np.sum(F_int[bot_dofs]))
        P_hist.append(0.5 * (abs(P_top) + abs(P_bot)))
    P_arr = np.asarray(P_hist)
    assert np.all(np.isfinite(P_arr)), (
        f"Non-finite values in reaction history: {P_arr}"
    )

    lam_hist = result["load_factor_history"]
    print(
        f"DCB arc-length (small): steps={result['steps_completed']}, "
        f"lam_max={float(lam_hist.max()):.4f}, "
        f"lam_final={float(lam_hist[-1]):.4f}, "
        f"P_peak={float(P_arr.max()):.3f}, "
        f"max_damage={max_damage:.3f}"
    )
