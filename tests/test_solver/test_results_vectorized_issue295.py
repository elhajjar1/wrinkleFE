"""Vectorised max_principal_stress and cached centroids (issue #295).

``FieldResults.max_principal_stress`` used a Python double loop over
(element, Gauss point), reconstructing a 3x3 tensor and calling
``np.linalg.eigvalsh`` once per point; it now builds the whole
``(n_elem, n_gp, 3, 3)`` tensor stack by slice assignment and solves it
in one batched call. ``stress_through_thickness`` rebuilt every element
centroid with a Python loop on every call; centroids are now computed
once per ``FieldResults`` (lazy ``element_centers``) and reused.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt

from wrinklefe.core.mesh import MeshData
from wrinklefe.solver.results import FieldResults


def _grid_mesh(nx: int = 3, ny: int = 2, nz: int = 2) -> MeshData:
    """Axis-aligned unit-cube grid mesh with nx*ny*nz hex elements."""
    xs = np.arange(nx + 1, dtype=float)
    ys = np.arange(ny + 1, dtype=float)
    zs = np.arange(nz + 1, dtype=float)

    def nid(i, j, k):
        return (k * (ny + 1) + j) * (nx + 1) + i

    nodes = np.array(
        [[x, y, z] for z in zs for y in ys for x in xs], dtype=float
    )
    elements = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                elements.append([
                    nid(i, j, k), nid(i + 1, j, k),
                    nid(i + 1, j + 1, k), nid(i, j + 1, k),
                    nid(i, j, k + 1), nid(i + 1, j, k + 1),
                    nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1),
                ])
    elements = np.asarray(elements, dtype=int)
    n_elem = elements.shape[0]
    return MeshData(
        nodes=nodes, elements=elements,
        ply_ids=np.zeros(n_elem, dtype=int),
        fiber_angles=np.zeros(len(nodes)),
        ply_angles=np.zeros(n_elem), nx=nx, ny=ny, nz=nz,
    )


def _field(mesh, stress_global) -> FieldResults:
    n_elem, n_gp = stress_global.shape[:2]
    z = np.zeros((n_elem, n_gp, 6))
    return FieldResults(
        displacement=np.zeros((mesh.nodes.shape[0], 3)),
        stress_global=stress_global, stress_local=z.copy(),
        strain_global=z.copy(), strain_local=z.copy(),
        mesh=mesh, laminate=None,
    )


def _loop_reference(stress_global: np.ndarray) -> np.ndarray:
    """The pre-#295 per-point implementation, kept as the oracle."""
    n_elem, n_gp, _ = stress_global.shape
    result = np.empty((n_elem, n_gp))
    for e in range(n_elem):
        for g in range(n_gp):
            sv = stress_global[e, g]
            tensor = np.array([
                [sv[0], sv[5], sv[4]],
                [sv[5], sv[1], sv[3]],
                [sv[4], sv[3], sv[2]],
            ])
            result[e, g] = np.linalg.eigvalsh(tensor)[-1]
    return result


def test_max_principal_matches_loop_reference():
    """Batched eigvalsh reproduces the per-point loop to 1e-10 on a
    randomized full-tensor field (all six Voigt components active)."""
    mesh = _grid_mesh()
    rng = np.random.default_rng(20260704)
    stress = rng.uniform(-500.0, 500.0, (mesh.n_elements, 8, 6))
    res = _field(mesh, stress)
    npt.assert_allclose(
        res.max_principal_stress, _loop_reference(stress),
        rtol=1e-12, atol=1e-10,
    )


def test_max_principal_known_values():
    """Analytic anchors: pure uniaxial, hydrostatic, and pure-shear
    states have closed-form largest principal stresses."""
    mesh = _grid_mesh(nx=1, ny=1, nz=1)
    stress = np.zeros((1, 8, 6))
    stress[0, 0] = [100.0, 0, 0, 0, 0, 0]        # uniaxial -> 100
    stress[0, 1] = [-50.0, -50.0, -50.0, 0, 0, 0]  # hydrostatic -> -50
    stress[0, 2] = [0, 0, 0, 0, 0, 60.0]         # pure shear t12 -> +60
    stress[0, 3] = [30.0, -30.0, 0, 0, 0, 0]     # biaxial -> +30
    res = _field(mesh, stress)
    mp = res.max_principal_stress
    npt.assert_allclose(mp[0, 0], 100.0, atol=1e-10)
    npt.assert_allclose(mp[0, 1], -50.0, atol=1e-10)
    npt.assert_allclose(mp[0, 2], 60.0, atol=1e-10)
    npt.assert_allclose(mp[0, 3], 30.0, atol=1e-10)
    # Untouched Gauss points are zero-stress -> 0.
    npt.assert_allclose(mp[0, 4:], 0.0, atol=1e-12)


def test_max_principal_cached_and_empty_guard():
    mesh = _grid_mesh(nx=1, ny=1, nz=1)
    res = _field(mesh, np.random.default_rng(0).normal(size=(1, 8, 6)))
    first = res.max_principal_stress
    assert res.max_principal_stress is first  # cached, not recomputed

    empty = FieldResults(
        displacement=np.zeros((0, 3)),
        stress_global=np.empty((0, 0, 6)), stress_local=np.empty((0, 0, 6)),
        strain_global=np.empty((0, 0, 6)), strain_local=np.empty((0, 0, 6)),
        mesh=_grid_mesh(1, 1, 1), laminate=None,
    )
    assert empty.max_principal_stress.shape == (0, 0)


def test_element_centers_match_mesh_and_are_cached(monkeypatch):
    """element_centers equals per-element mesh.element_center and is
    computed at most once across repeated through-thickness queries."""
    mesh = _grid_mesh()
    rng = np.random.default_rng(1)
    res = _field(mesh, rng.normal(size=(mesh.n_elements, 8, 6)))

    centers = res.element_centers
    expected = np.vstack([
        mesh.element_center(e) for e in range(mesh.n_elements)
    ])
    npt.assert_allclose(centers, expected, rtol=0, atol=1e-14)
    assert res.element_centers is centers  # lazy cache

    # Repeated column queries must not rebuild the centroid array.
    calls = {"n": 0}
    real = MeshData.element_center

    def _counting(self, e):
        calls["n"] += 1
        return real(self, e)

    monkeypatch.setattr(MeshData, "element_center", _counting)
    z1, s1 = res.stress_through_thickness(0.5, 0.5, component=0)
    z2, s2 = res.stress_through_thickness(1.5, 0.5, component=0)
    assert calls["n"] == 0  # served entirely from the cached array
    assert z1.size > 0 and z2.size > 0


def test_stress_through_thickness_unchanged():
    """Column extraction is unchanged by the centroid cache: z-sorted
    centroid heights and per-element Gauss means at the queried column."""
    mesh = _grid_mesh(nx=2, ny=1, nz=3)
    rng = np.random.default_rng(2)
    stress = rng.normal(size=(mesh.n_elements, 8, 6))
    res = _field(mesh, stress)

    z, s = res.stress_through_thickness(0.5, 0.5, component=0)
    npt.assert_allclose(z, [0.5, 1.5, 2.5])
    # Column at x=0.5 is elements 0, 2, 4 (nx=2 stride).
    expected = [stress[e, :, 0].mean() for e in (0, 2, 4)]
    npt.assert_allclose(s, expected, rtol=1e-12)
