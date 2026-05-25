"""End-to-end integration tests for Cohesive8 + Newton + GlobalAssembler.

These tests exercise the FULL production stack with no mocking of the
assembler or BC handler:

    NewtonRaphsonSolver -> GlobalAssembler -> Hex8Element/Cohesive8Element
                                            -> BoundaryHandler (penalty)

Geometry: two stacked 1 mm cubes with a zero-thickness Cohesive8 element
sandwiched between them.  The middle interface uses duplicate coincident
nodes (4 in the lower hex's top face, 4 in the upper hex's bottom face)
so the cohesive law sees the full displacement jump.

Node layout (12 nodes total):

    k = 0 (z = 0):    nodes 0..3   -> bottom of lower hex
    k = 1 (z = 1):    nodes 4..7   -> top of lower hex / cohesive bottom
    k = 2 (z = 1):    nodes 8..11  -> cohesive top / bottom of upper hex
    k = 3 (z = 2):    nodes 12..15 -> top of upper hex

In each k-layer the 4 nodes are CCW from (-, -) -> (+, -) -> (+, +) ->
(-, +) at x in {0, 1}, y in {0, 1}.

The hex8 elements are:

    lower hex: nodes [0,1,2,3, 4,5,6,7]
    upper hex: nodes [8,9,10,11, 12,13,14,15]

The cohesive element is:

    nodes [4,5,6,7, 8,9,10,11]
"""

from __future__ import annotations

import numpy as np
import pytest

from wrinklefe.core.laminate import Laminate, Ply
from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.core.mesh import MeshData
from wrinklefe.elements.cohesive8 import Cohesive8Element, CohesiveProperties
from wrinklefe.solver.assembler import GlobalAssembler
from wrinklefe.solver.boundary import BoundaryCondition, BoundaryHandler
from wrinklefe.solver.nonlinear import NewtonRaphsonSolver


# ======================================================================
# Mesh + element construction
# ======================================================================

def _two_cube_mesh_with_cohesive() -> tuple[MeshData, list[int], list[int]]:
    """Build a 16-node, 2-hex, 1-cohesive sandwich.

    Returns
    -------
    mesh : MeshData
    bottom_node_ids : list[int]
        4 node indices on the very bottom face (z = 0), to be fully fixed.
    top_node_ids : list[int]
        4 node indices on the very top face (z = 2), to receive the
        prescribed z-displacement.
    """
    quad_xy = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=float)

    layers = [
        (0.0,),   # k=0: nodes 0..3   (z=0)
        (1.0,),   # k=1: nodes 4..7   (z=1, lower hex top)
        (1.0,),   # k=2: nodes 8..11  (z=1, upper hex bottom, duplicates)
        (2.0,),   # k=3: nodes 12..15 (z=2)
    ]
    node_blocks = []
    for (z,) in layers:
        block = np.column_stack(
            [quad_xy[:, 0], quad_xy[:, 1], np.full(4, z)]
        )
        node_blocks.append(block)
    nodes = np.vstack(node_blocks)  # (16, 3)

    elements = np.array([
        [0, 1, 2, 3,  4, 5, 6, 7],     # lower hex
        [8, 9, 10, 11, 12, 13, 14, 15],  # upper hex
    ], dtype=np.intp)

    ply_ids = np.array([0, 0], dtype=np.intp)
    ply_angles = np.array([0.0, 0.0], dtype=float)
    fiber_angles = np.zeros(16, dtype=float)

    mesh = MeshData(
        nodes=nodes,
        elements=elements,
        ply_ids=ply_ids,
        fiber_angles=fiber_angles,
        ply_angles=ply_angles,
        nx=1, ny=1, nz=2,  # NB: nx/ny/nz reflect element counts in a
                            # *structured* grid; we don't rely on the
                            # nodes_on_face topology here (we use explicit
                            # node_ids in the BCs instead).
    )
    return mesh, [0, 1, 2, 3], [12, 13, 14, 15]


def _make_assembler(mesh: MeshData, cohesive_props: CohesiveProperties) -> tuple[
    GlobalAssembler, Cohesive8Element
]:
    # Stiff hex8 blocks so the cohesive interface opening is essentially
    # equal to the applied top-face displacement (E_hex >> K_coh * L).
    mat = OrthotropicMaterial(
        E1=1.0e8, E2=1.0e8, E3=1.0e8,
        G12=4.0e7, G13=4.0e7, G23=4.0e7,
        nu12=0.25, nu13=0.25, nu23=0.25,
        name="iso-test",
    )
    laminate = Laminate([Ply(material=mat, angle=0.0, thickness=1.0)])

    # Cohesive element on the middle interface (z = 1).
    coh_node_ids = np.array([4, 5, 6, 7, 8, 9, 10, 11], dtype=np.intp)
    coh_coords = mesh.nodes[coh_node_ids]
    coh_elem = Cohesive8Element(
        node_coords=coh_coords,
        properties=cohesive_props,
        elem_id=0,
    )
    coh_elem.node_ids = coh_node_ids  # tag for global DOF lookup

    assembler = GlobalAssembler(
        mesh=mesh,
        laminate=laminate,
        cohesive_elements=[(0, coh_elem)],
    )
    return assembler, coh_elem


def _ramp_bcs(
    bottom_ids: list[int],
    top_ids: list[int],
    uz_target: float,
) -> list[BoundaryCondition]:
    """Build BCs: bottom face fully fixed, top face uz prescribed."""
    bcs: list[BoundaryCondition] = []
    bottom_arr = np.asarray(bottom_ids, dtype=np.intp)
    top_arr = np.asarray(top_ids, dtype=np.intp)

    bcs.append(BoundaryCondition(
        bc_type="fixed", node_ids=bottom_arr, dofs=[0, 1, 2],
    ))
    # Pin x, y on the top face too, so only uz is the loading DOF.
    bcs.append(BoundaryCondition(
        bc_type="fixed", node_ids=top_arr, dofs=[0, 1],
    ))
    bcs.append(BoundaryCondition(
        bc_type="displacement", node_ids=top_arr,
        dofs=[2], value=float(uz_target),
    ))
    return bcs


def _cohesive_props() -> CohesiveProperties:
    return CohesiveProperties(
        K=1.0e5, sigma_max=50.0, tau_max=80.0,
        GIc=0.5, GIIc=1.5, eta_BK=1.45, beta=1.0,
    )


# ======================================================================
# Test 1: Mode-I full damage, energy ~= GIc * area
# ======================================================================

def test_single_cohesive_element_mode_I_through_newton():
    mesh, bottom_ids, top_ids = _two_cube_mesh_with_cohesive()
    props = _cohesive_props()
    assembler, coh_elem = _make_assembler(mesh, props)

    delta_0 = props.sigma_max / props.K
    delta_f = 2.0 * props.GIc / props.sigma_max  # critical opening
    # Ramp to 1.5 * delta_f (full damage) — the post-failure tail at
    # zero traction contributes no work, so there's no need to spend
    # increments on it.  400 increments resolves the corner at delta_0
    # (peak traction) well enough for 5% trapezoidal-rule accuracy on
    # the dissipated energy.
    n_inc = 400
    u_top_target = 1.5 * delta_f

    bcs = _ramp_bcs(bottom_ids, top_ids, u_top_target)
    bc_handler = BoundaryHandler(mesh)

    solver = NewtonRaphsonSolver(
        assembler=assembler,
        bc_handler=bc_handler,
        boundary_conditions=bcs,
        n_increments=n_inc,
        max_newton_iter=40,
        tol_residual=1e-6,
        tol_displacement=1e-9,
        line_search=True,
    )

    # Track top-face z reaction force at each converged increment.
    reactions: list[float] = []
    top_arr = np.asarray(top_ids, dtype=np.intp)
    top_z_dofs = 3 * top_arr + 2

    # We instrument the solver by stepping through increments manually.
    # Reuse solver state machinery: do this by running solve() and then
    # recovering F_int at each load-displacement sample point.  However
    # solve() does not return per-increment displacements; so we run
    # increments one at a time via a thin loop here that mirrors the
    # solver's bookkeeping.
    u = np.zeros(mesh.n_dof)
    constrained_full = bc_handler.get_constrained_dofs(bcs)
    F_ext_full = bc_handler.get_force_dofs(bcs)
    increments = list(range(1, n_inc + 1))
    u_top_history = [0.0]
    work_increments = [0.0]
    prev_F_top = 0.0

    for inc in increments:
        lam = inc / n_inc
        constrained_inc = {
            d: lam * v for d, v in constrained_full.items()
        }
        u_new, _it, ok = solver._newton_step(
            u, lam * F_ext_full, constrained_inc, verbose=False, inc=inc,
        )
        assert ok, f"Newton failed at increment {inc}"
        u = u_new
        solver._commit_state()

        # Reaction at the top face: cohesive resistance == sum of top-face
        # internal force in z (equals applied force on those DOFs).
        F_int = assembler.assemble_internal_force(u)
        F_top = float(np.sum(F_int[top_z_dofs]))

        u_top = float(u[top_z_dofs[0]])  # all top nodes prescribed same uz
        du_top = u_top - u_top_history[-1]
        # Trapezoidal work increment using the reaction (= F_int on top).
        work_increments.append(
            work_increments[-1] + 0.5 * (prev_F_top + F_top) * du_top
        )
        u_top_history.append(u_top)
        reactions.append(F_top)
        prev_F_top = F_top

    # Convergence achieved for every increment above (asserted in loop).
    # Final damage at all 4 GPs should be ~ 1.0.
    final_state = assembler.cohesive_state[0]
    damages = np.array([s.d for s in final_state])
    assert np.all(damages > 0.99), (
        f"Expected full damage at all GPs, got d = {damages}"
    )

    # Energy dissipated ~ GIc * area.
    total_work = work_increments[-1]
    expected = props.GIc * coh_elem.area
    rel_err = abs(total_work - expected) / expected
    assert rel_err < 0.05, (
        f"Mode-I energy mismatch through Newton: work={total_work:.4f}, "
        f"GIc*A={expected:.4f}, rel={rel_err:.3%}"
    )

    # Sanity: peak reaction should be roughly sigma_max * area (before
    # damage knocks it down).  Just confirm the peak is in the right
    # order of magnitude.
    peak = max(reactions)
    assert peak > 0.5 * props.sigma_max * coh_elem.area, (
        f"Peak reaction {peak} suspiciously low vs sigma*A "
        f"{props.sigma_max * coh_elem.area}"
    )

    # Tiny sanity: at full damage the residual reaction should be ~ 0
    # (interface fully open, top hex moved as rigid body under the
    # remaining penalty stiffness from BCs).
    assert reactions[-1] < 0.05 * peak, (
        f"Final reaction {reactions[-1]} still high vs peak {peak}: "
        "interface did not fully fail."
    )

    # Suppress unused-variable warning.
    _ = delta_0


# ======================================================================
# Test 2: Unload + reload — d monotonic, reload follows secant
# ======================================================================

def test_unload_reload_through_newton():
    mesh, bottom_ids, top_ids = _two_cube_mesh_with_cohesive()
    props = _cohesive_props()
    assembler, _coh = _make_assembler(mesh, props)
    bc_handler = BoundaryHandler(mesh)

    delta_0 = props.sigma_max / props.K
    u_peak = 1.5 * delta_0  # partial-damage opening
    top_arr = np.asarray(top_ids, dtype=np.intp)
    top_z_dofs = 3 * top_arr + 2

    def _ramp(u_start: np.ndarray, u_top_start: float,
              u_top_end: float, n_inc: int) -> tuple[np.ndarray, float]:
        """Manually walk Newton increments along a linear ramp from
        u_top_start to u_top_end on the top face's z-DOFs.  Returns the
        final ``u`` and the top-face z-reaction at the final step.
        """
        u_local = u_start.copy()

        # Fixed BCs that do not change with the ramp.
        fixed_bcs = [
            BoundaryCondition(
                bc_type="fixed",
                node_ids=np.asarray(bottom_ids, dtype=np.intp),
                dofs=[0, 1, 2],
            ),
            BoundaryCondition(
                bc_type="fixed", node_ids=top_arr, dofs=[0, 1],
            ),
        ]
        fixed_constrained = bc_handler.get_constrained_dofs(fixed_bcs)
        F_ext = np.zeros(mesh.n_dof)

        solver = NewtonRaphsonSolver(
            assembler=assembler,
            bc_handler=bc_handler,
            boundary_conditions=fixed_bcs + [BoundaryCondition(
                bc_type="displacement", node_ids=top_arr,
                dofs=[2], value=u_top_end,
            )],
            n_increments=n_inc,
            max_newton_iter=40,
            tol_residual=1e-6,
            tol_displacement=1e-9,
            line_search=True,
        )
        for inc in range(1, n_inc + 1):
            lam = inc / n_inc
            target_now = (1.0 - lam) * u_top_start + lam * u_top_end
            constrained = dict(fixed_constrained)
            for d in top_z_dofs:
                constrained[int(d)] = float(target_now)
            u_new, _it, ok = solver._newton_step(
                u_local, F_ext, constrained, verbose=False, inc=inc,
            )
            assert ok, (
                f"Newton failed mid-ramp: target={u_top_end}, inc={inc}"
            )
            u_local = u_new
            solver._commit_state()
        F_int = assembler.assemble_internal_force(u_local)
        return u_local, float(np.sum(F_int[top_z_dofs]))

    u = np.zeros(mesh.n_dof)

    # Phase 1: load to 1.5 * delta_0 in many fine increments so the
    # damage state at u_peak is accurate.
    u, F_load = _ramp(u, 0.0, u_peak, n_inc=200)
    d_after_load = float(assembler.cohesive_state[0][0].d)
    assert 0.0 < d_after_load < 1.0, (
        f"Expected partial damage after loading to 1.5*delta_0, "
        f"got d={d_after_load}"
    )

    # Phase 2: unload to 0.
    u, F_unload = _ramp(u, u_peak, 0.0, n_inc=40)
    d_after_unload = float(assembler.cohesive_state[0][0].d)
    assert d_after_unload >= d_after_load - 1e-12, (
        f"Damage decreased on unload: {d_after_load} -> {d_after_unload}"
    )
    assert abs(F_unload) < 1e-2, (
        f"Residual reaction at zero opening should be ~0, "
        f"got {F_unload}"
    )

    # Phase 3: reload to u_peak.  No new damage should accumulate (the
    # opening never exceeds the historical max), and at u_peak the force
    # should match the load-leg force exactly (secant unloading).
    u, F_reload = _ramp(u, 0.0, u_peak, n_inc=40)
    d_after_reload = float(assembler.cohesive_state[0][0].d)
    # Non-decreasing damage (within penalty-BC overshoot, ~1e-4 relative).
    assert d_after_reload >= d_after_unload - 1e-12, (
        f"Damage decreased on reload: {d_after_unload} -> {d_after_reload}"
    )
    assert d_after_reload <= d_after_unload * (1.0 + 1e-3), (
        f"Damage grew too much on reload past historical max: "
        f"{d_after_unload} -> {d_after_reload}"
    )
    rel = abs(F_reload - F_load) / max(abs(F_load), 1e-9)
    assert rel < 0.05, (
        f"Reload force at peak does not match load-leg force: "
        f"load={F_load:.4f}, reload={F_reload:.4f}, rel={rel:.3%}"
    )


# ======================================================================
# Test 3: Compression then tension — no damage from compression
# ======================================================================

def test_compression_followed_by_tension():
    mesh, bottom_ids, top_ids = _two_cube_mesh_with_cohesive()
    props = _cohesive_props()
    assembler, coh_elem = _make_assembler(mesh, props)
    bc_handler = BoundaryHandler(mesh)

    delta_0 = props.sigma_max / props.K

    def _ramp_to(u_start: np.ndarray, u_top_target: float, n_inc: int = 6):
        bcs = _ramp_bcs(bottom_ids, top_ids, u_top_target)
        solver = NewtonRaphsonSolver(
            assembler=assembler,
            bc_handler=bc_handler,
            boundary_conditions=bcs,
            n_increments=n_inc,
            max_newton_iter=40,
            tol_residual=1e-6,
            tol_displacement=1e-9,
            line_search=True,
        )
        constrained_full = bc_handler.get_constrained_dofs(bcs)
        F_ext_full = bc_handler.get_force_dofs(bcs)
        top_arr = np.asarray(top_ids, dtype=np.intp)
        top_z_dofs = 3 * top_arr + 2

        u_start_top = float(u_start[top_z_dofs[0]])
        u_local = u_start.copy()
        results: list[tuple[float, float, float]] = []
        for inc in range(1, n_inc + 1):
            lam = inc / n_inc
            target_now = (1 - lam) * u_start_top + lam * u_top_target
            constrained_inc = {
                d: lam * v for d, v in constrained_full.items()
            }
            for d in top_z_dofs:
                constrained_inc[int(d)] = float(target_now)

            u_new, _it, ok = solver._newton_step(
                u_local, lam * F_ext_full, constrained_inc,
                verbose=False, inc=inc,
            )
            assert ok, f"Newton failed: target {u_top_target}, inc {inc}"
            u_local = u_new
            solver._commit_state()
            F_int = assembler.assemble_internal_force(u_local)
            results.append((
                float(u_local[top_z_dofs[0]]),
                float(np.sum(F_int[top_z_dofs])),
                float(assembler.cohesive_state[0][0].d),
            ))
        return u_local, results

    u = np.zeros(mesh.n_dof)

    # Phase 1: compress to delta_n = -0.5 mm.
    u, comp_results = _ramp_to(u, -0.5)
    d_after_compression = comp_results[-1][2]
    assert d_after_compression == 0.0, (
        f"Damage accumulated under compression: d={d_after_compression}"
    )

    # Phase 2: return to 0.
    u, zero_results = _ramp_to(u, 0.0)
    d_at_zero = zero_results[-1][2]
    assert d_at_zero == 0.0, f"Damage from compression cycle: d={d_at_zero}"

    # Phase 3: tension into elastic regime (delta_n = delta_0).
    u, tens_results = _ramp_to(u, 0.99 * delta_0)
    d_after_tens = tens_results[-1][2]
    assert d_after_tens == 0.0, (
        f"Damage accumulated within elastic regime: d={d_after_tens}"
    )

    # Linear elasticity check at the end of each phase.  At small openings
    # the cohesive resistance is K * delta * A.  The two stacked hex8
    # blocks are also linear elastic, so the top-face reaction is a
    # series-spring combination; we don't need to compute the exact
    # value, but the ratio of (reaction in compression at delta=-0.5)
    # to (reaction in tension at delta=delta_0) should equal the ratio
    # of openings (linear response).
    f_comp_peak = comp_results[-1][1]
    f_tens_peak = tens_results[-1][1]
    # f_comp_peak is the reaction at u_top=-0.5 (compressive => negative).
    # f_tens_peak is the reaction at u_top=0.99*delta_0 (tensile =>
    # positive).  Same sign convention; compute ratio of magnitudes
    # vs ratio of openings.
    ratio_force = abs(f_comp_peak) / max(abs(f_tens_peak), 1e-12)
    ratio_disp = 0.5 / (0.99 * delta_0)
    assert np.isclose(ratio_force, ratio_disp, rtol=0.05), (
        f"Linear elasticity broken: f_comp/f_tens={ratio_force:.3f}, "
        f"u_comp/u_tens={ratio_disp:.3f}; "
        f"reactions: comp={f_comp_peak}, tens={f_tens_peak}"
    )

    _ = coh_elem
