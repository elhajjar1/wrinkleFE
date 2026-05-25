"""Tests for :func:`wrinklefe.core.cohesive_mesh.insert_cohesive_interface`.

The first three tests exercise topology (node duplication, error
handling); the fourth re-runs the through-Newton Mode-I integration test
from :mod:`tests.integration.test_cohesive_newton_integration` but
constructs the mesh via the new utility, so any regression in the
utility's wiring shows up as a physics failure.
"""

from __future__ import annotations

import numpy as np
import pytest

from wrinklefe.core.cohesive_mesh import insert_cohesive_interface
from wrinklefe.core.laminate import Laminate, Ply
from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.core.mesh import MeshData
from wrinklefe.elements.cohesive8 import CohesiveProperties
from wrinklefe.solver.assembler import GlobalAssembler
from wrinklefe.solver.boundary import BoundaryCondition, BoundaryHandler
from wrinklefe.solver.nonlinear import NewtonRaphsonSolver


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _two_cube_mesh_sharing_interface() -> MeshData:
    """Build a 12-node, 2-hex mesh with a shared interface at z = 1.

    Layers (4 nodes each, CCW from -,-):

        k = 0 (z = 0): nodes 0..3
        k = 1 (z = 1): nodes 4..7  (shared between both hexes)
        k = 2 (z = 2): nodes 8..11
    """
    quad_xy = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=float)

    nodes = np.vstack([
        np.column_stack([quad_xy[:, 0], quad_xy[:, 1], np.full(4, 0.0)]),
        np.column_stack([quad_xy[:, 0], quad_xy[:, 1], np.full(4, 1.0)]),
        np.column_stack([quad_xy[:, 0], quad_xy[:, 1], np.full(4, 2.0)]),
    ])
    elements = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8, 9, 10, 11],
    ], dtype=np.intp)

    return MeshData(
        nodes=nodes,
        elements=elements,
        ply_ids=np.array([0, 0], dtype=np.intp),
        fiber_angles=np.zeros(12, dtype=float),
        ply_angles=np.array([0.0, 0.0], dtype=float),
        nx=1,
        ny=1,
        nz=2,
    )


def _cohesive_props() -> CohesiveProperties:
    return CohesiveProperties(
        K=1.0e5, sigma_max=50.0, tau_max=80.0,
        GIc=0.5, GIIc=1.5, eta_BK=1.45, beta=1.0,
    )


# ======================================================================
# 1. Insert into a 2-cube sandwich
# ======================================================================


def test_insert_into_two_cube_sandwich():
    mesh = _two_cube_mesh_sharing_interface()
    props = _cohesive_props()

    new_mesh, cohesive_elements = insert_cohesive_interface(
        mesh, z_interface=1.0, cohesive_props=props,
    )

    # ---- shape contracts ----
    assert new_mesh.nodes.shape == (16, 3), (
        f"expected 16 nodes after insertion, got {new_mesh.nodes.shape[0]}"
    )
    assert new_mesh.elements.shape == (2, 8), (
        f"hex8 connectivity shape changed: {new_mesh.elements.shape}"
    )
    assert len(cohesive_elements) == 1

    # ---- node duplicates carry identical xyz ----
    interface_orig_ids = [4, 5, 6, 7]
    interface_dup_ids = [12, 13, 14, 15]
    np.testing.assert_allclose(
        new_mesh.nodes[interface_orig_ids],
        new_mesh.nodes[interface_dup_ids],
    )

    # ---- the BELOW hex (centroid z = 0.5) keeps original IDs ----
    # ---- the ABOVE hex (centroid z = 1.5) is rewired to duplicates ----
    # We don't depend on per-element ordering — check both hex elements
    # individually instead.
    elem_centroids_z = mesh.nodes[mesh.elements][:, :, 2].mean(axis=1)
    below_orig_eid = int(np.flatnonzero(elem_centroids_z < 1.0)[0])
    above_orig_eid = int(np.flatnonzero(elem_centroids_z > 1.0)[0])

    below_conn = new_mesh.elements[below_orig_eid]
    above_conn = new_mesh.elements[above_orig_eid]
    # Below: top-face local indices 4..7 should still be on {4, 5, 6, 7}
    assert set(below_conn[4:8].tolist()) == set(interface_orig_ids), (
        f"below-element top face was modified: {below_conn[4:8]}"
    )
    # Above: bottom-face local indices 0..3 should now be the duplicates
    assert set(above_conn[0:4].tolist()) == set(interface_dup_ids), (
        f"above-element bottom face still uses originals: {above_conn[0:4]}"
    )
    # Untouched nodes (k=0 and k=2 layers) preserved.
    assert set(below_conn[0:4].tolist()) == {0, 1, 2, 3}
    assert set(above_conn[4:8].tolist()) == {8, 9, 10, 11}

    # ---- cohesive element node_ids: bottom = originals, top = dups ----
    coh = cohesive_elements[0]
    assert coh.node_ids is not None
    coh_ids = coh.node_ids.tolist()
    assert set(coh_ids[0:4]) == set(interface_orig_ids), (
        f"cohesive bottom face != original interface nodes: {coh_ids[0:4]}"
    )
    assert set(coh_ids[4:8]) == set(interface_dup_ids), (
        f"cohesive top face != duplicate node ids: {coh_ids[4:8]}"
    )
    # Pairing: top node i+4 must sit on bottom node i (same xyz).
    for i in range(4):
        bot, top = coh_ids[i], coh_ids[i + 4]
        np.testing.assert_allclose(
            new_mesh.nodes[bot], new_mesh.nodes[top],
            err_msg=f"cohesive node pair {i} not coincident",
        )

    # ---- original mesh untouched ----
    assert mesh.nodes.shape == (12, 3)
    assert mesh.elements.shape == (2, 8)


# ======================================================================
# 2. Missing-plane error
# ======================================================================


def test_insert_at_missing_plane_raises():
    mesh = _two_cube_mesh_sharing_interface()
    with pytest.raises(ValueError, match="no mesh nodes found"):
        insert_cohesive_interface(
            mesh, z_interface=0.5, cohesive_props=_cohesive_props(),
        )


# ======================================================================
# 3. Invalid topology (element with 2 nodes on the interface)
# ======================================================================


def test_insert_at_invalid_topology_raises():
    """Build a mesh where one element has exactly 2 nodes on the plane.

    Geometry: a 4-node 'staircase' element straddling z = 1 such that
    its top 4 nodes sit at z = 1 on TWO of the 4 corners and at z = 2
    on the other two.  An interface request at z = 1 then sees a 2-node
    intersection (illegal).
    """
    quad_xy = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=float)

    nodes = np.vstack([
        # Bottom (z = 0): nodes 0..3
        np.column_stack([quad_xy[:, 0], quad_xy[:, 1], np.full(4, 0.0)]),
        # Top: staggered — 2 nodes at z = 1, 2 at z = 2.
        # nodes 4..7
        np.array([
            [0.0, 0.0, 1.0],   # 4 — on the plane
            [1.0, 0.0, 2.0],   # 5 — above
            [1.0, 1.0, 2.0],   # 6 — above
            [0.0, 1.0, 1.0],   # 7 — on the plane
        ], dtype=float),
    ])
    elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.intp)
    mesh = MeshData(
        nodes=nodes,
        elements=elements,
        ply_ids=np.array([0], dtype=np.intp),
        fiber_angles=np.zeros(8, dtype=float),
        ply_angles=np.array([0.0], dtype=float),
        nx=1, ny=1, nz=1,
    )

    with pytest.raises(ValueError, match="not structured-hex compatible"):
        insert_cohesive_interface(
            mesh, z_interface=1.0, cohesive_props=_cohesive_props(),
        )


# ======================================================================
# 4. End-to-end: drive the inserted mesh through Newton (Mode-I)
# ======================================================================


def _make_assembler_with_inserted_cohesive(
    mesh: MeshData,
    cohesive_props: CohesiveProperties,
) -> tuple[GlobalAssembler, MeshData, list]:
    """Mirror :func:`_make_assembler` from the Newton integration test,
    but build the cohesive layer via the new mesh utility instead of
    hand-rolled node duplication."""
    mat = OrthotropicMaterial(
        E1=1.0e8, E2=1.0e8, E3=1.0e8,
        G12=4.0e7, G13=4.0e7, G23=4.0e7,
        nu12=0.25, nu13=0.25, nu23=0.25,
        name="iso-test",
    )
    laminate = Laminate([Ply(material=mat, angle=0.0, thickness=1.0)])

    new_mesh, cohesive_elements = insert_cohesive_interface(
        mesh, z_interface=1.0, cohesive_props=cohesive_props,
    )
    assembler = GlobalAssembler(
        mesh=new_mesh,
        laminate=laminate,
        cohesive_elements=[(e.elem_id, e) for e in cohesive_elements],
    )
    return assembler, new_mesh, cohesive_elements


def test_solver_drives_inserted_mesh():
    """Same physics check as ``test_single_cohesive_element_mode_I``
    in the Newton integration suite, but the cohesive layer is wired up
    by :func:`insert_cohesive_interface` rather than hand-built."""
    mesh = _two_cube_mesh_sharing_interface()
    props = _cohesive_props()
    assembler, new_mesh, cohesive_elements = (
        _make_assembler_with_inserted_cohesive(mesh, props)
    )

    delta_f = 2.0 * props.GIc / props.sigma_max
    n_inc = 400
    u_top_target = 1.5 * delta_f

    # Bottom face (z = 0) fully fixed; top face (z = 2) prescribed uz.
    bottom_ids = np.flatnonzero(np.isclose(new_mesh.nodes[:, 2], 0.0))
    top_ids = np.flatnonzero(np.isclose(new_mesh.nodes[:, 2], 2.0))
    assert bottom_ids.size == 4 and top_ids.size == 4

    bcs = [
        BoundaryCondition(
            bc_type="fixed",
            node_ids=bottom_ids.astype(np.intp),
            dofs=[0, 1, 2],
        ),
        BoundaryCondition(
            bc_type="fixed",
            node_ids=top_ids.astype(np.intp),
            dofs=[0, 1],
        ),
        BoundaryCondition(
            bc_type="displacement",
            node_ids=top_ids.astype(np.intp),
            dofs=[2],
            value=float(u_top_target),
        ),
    ]

    bc_handler = BoundaryHandler(new_mesh)
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

    result = solver.solve()
    assert result["converged"], (
        f"Newton failed to converge: {result['iteration_counts']}"
    )

    # Final damage at every GP should be ~ 1.0.
    coh_id = cohesive_elements[0].elem_id
    final_state = assembler.cohesive_state[coh_id]
    damages = np.array([s.d for s in final_state])
    assert np.all(damages > 0.99), (
        f"Expected full damage at all GPs after Mode-I ramp, got "
        f"d = {damages}"
    )
