"""Coordinate-aware max-query methods on FieldResults (issue #297)."""

import numpy as np

from wrinklefe.core.mesh import MeshData
from wrinklefe.solver.results import FieldResults


def _two_element_mesh() -> MeshData:
    """Two axis-aligned unit hexes side-by-side along x (12 nodes).

    Element 0 spans x in [0, 1] (centroid (0.5, 0.5, 0.5)); element 1 spans
    x in [1, 2] (centroid (1.5, 0.5, 0.5)).
    """
    nodes = np.array(
        [
            [0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0],
            [0, 0, 1], [1, 0, 1], [2, 0, 1], [0, 1, 1], [1, 1, 1], [2, 1, 1],
        ],
        dtype=float,
    )
    elements = np.array(
        [[0, 1, 4, 3, 6, 7, 10, 9], [1, 2, 5, 4, 7, 8, 11, 10]], dtype=int
    )
    return MeshData(
        nodes=nodes, elements=elements,
        ply_ids=np.zeros(2, dtype=int), fiber_angles=np.zeros(12),
        ply_angles=np.zeros(2), nx=2, ny=1, nz=1,
    )


def _field(mesh, displacement, stress_local):
    n_elem, n_gp = stress_local.shape[:2]
    z = np.zeros((n_elem, n_gp, 6))
    return FieldResults(
        displacement=displacement,
        stress_global=z.copy(), stress_local=stress_local,
        strain_global=z.copy(), strain_local=z.copy(),
        mesh=mesh, laminate=None,
    )


def test_max_displacement_location_resolves_node_coords():
    mesh = _two_element_mesh()
    disp = np.zeros((12, 3))
    disp[5] = [0.0, 0.0, 3.0]  # largest magnitude at node 5 = (2, 1, 0)
    res = _field(mesh, disp, np.zeros((2, 8, 6)))

    mag, loc = res.max_displacement_location()
    assert mag == 3.0
    np.testing.assert_allclose(loc, [2.0, 1.0, 0.0])
    # Consistent with the index-returning method + the mesh lookup.
    _, node_idx = res.max_displacement()
    np.testing.assert_allclose(loc, mesh.nodes[node_idx])


def test_max_stress_location_resolves_element_centroid():
    mesh = _two_element_mesh()
    stress = np.zeros((2, 8, 6))
    stress[0, :, 0] = 10.0
    stress[1, :, 0] = 50.0  # sigma_11 max in element 1
    res = _field(mesh, np.zeros((12, 3)), stress)

    value, loc = res.max_stress_location(component=0)
    assert value == 50.0
    np.testing.assert_allclose(loc, mesh.element_center(1))
    np.testing.assert_allclose(loc, [1.5, 0.5, 0.5])


def test_location_methods_do_not_change_index_methods():
    """Additive (non-breaking): the index-returning methods are unchanged."""
    mesh = _two_element_mesh()
    disp = np.zeros((12, 3))
    disp[5] = [0.0, 0.0, 3.0]
    stress = np.zeros((2, 8, 6))
    stress[1, :, 0] = 50.0
    res = _field(mesh, disp, stress)

    assert res.max_displacement() == (3.0, 5)
    value, elem_idx, gp_idx = res.max_stress(component=0)
    assert (value, elem_idx) == (50.0, 1)
    assert 0 <= gp_idx < 8
