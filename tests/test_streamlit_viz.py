"""Tests for the vectorized boundary-face extraction and Plotly payload
trimming in :mod:`streamlit_viz`.

These tests cover the perf-critical hot path used by the Streamlit 3D
views (stress contour, deformed mesh, failure-index surface) on
medium-sized hex8 meshes.  See issue #78.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# streamlit_viz lives at the repo root, not under src/, so make sure the
# top-level directory is importable regardless of pytest's cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pytest.importorskip("plotly")

import streamlit_viz as sv  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _structured_hex_mesh(nx: int, ny: int, nz: int):
    """Build a unit-spaced structured hex8 mesh of shape (nx, ny, nz).

    Returns ``(nodes, elements)`` with the same node numbering convention
    used in :mod:`wrinklefe.core.mesh`:

        node id (i, j, k) = k * (ny+1)*(nx+1) + j * (nx+1) + i.
    """
    nxp, nyp, nzp = nx + 1, ny + 1, nz + 1
    nodes = np.array(
        [
            [i, j, k]
            for k in range(nzp)
            for j in range(nyp)
            for i in range(nxp)
        ],
        dtype=float,
    )

    def nid(i: int, j: int, k: int) -> int:
        return k * nyp * nxp + j * nxp + i

    elems = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                elems.append(
                    [
                        nid(i, j, k),
                        nid(i + 1, j, k),
                        nid(i + 1, j + 1, k),
                        nid(i, j + 1, k),
                        nid(i, j, k + 1),
                        nid(i + 1, j, k + 1),
                        nid(i + 1, j + 1, k + 1),
                        nid(i, j + 1, k + 1),
                    ]
                )
    return nodes, np.asarray(elems, dtype=np.int64)


def _boundary_faces_reference(elements: np.ndarray) -> np.ndarray:
    """Pure-Python reference: a face is on the boundary iff its sorted
    node-id tuple is unique across the mesh."""
    HEX_FACES = sv.HEX_FACES
    face_owners: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for ei in range(elements.shape[0]):
        conn = elements[ei]
        for fi, face in enumerate(HEX_FACES):
            key = tuple(sorted(int(v) for v in conn[face]))
            face_owners.setdefault(key, []).append((ei, fi))
    out: list[list[int]] = []
    for owners in face_owners.values():
        if len(owners) == 1:
            ei, fi = owners[0]
            face = HEX_FACES[fi]
            out.append([int(elements[ei][n]) for n in face] + [ei])
    return np.asarray(out, dtype=np.int64)


# ---------------------------------------------------------------------------
# boundary_faces — correctness
# ---------------------------------------------------------------------------


def test_boundary_face_count_3x3x3():
    """Structured 3x3x3 brick: surface area = 6 * 9 = 54 quad faces."""
    _, elems = _structured_hex_mesh(3, 3, 3)
    bf = sv.boundary_faces(elems)
    assert bf.shape == (54, 5)
    # Expected by the analytic formula 2*(nx*ny + nx*nz + ny*nz).
    assert bf.shape[0] == 2 * (3 * 3 + 3 * 3 + 3 * 3)


@pytest.mark.parametrize("shape", [(2, 2, 2), (4, 3, 2), (5, 5, 5), (3, 1, 4)])
def test_boundary_face_count_matches_formula(shape):
    nx, ny, nz = shape
    _, elems = _structured_hex_mesh(nx, ny, nz)
    bf = sv.boundary_faces(elems)
    expected = 2 * (nx * ny + nx * nz + ny * nz)
    assert bf.shape[0] == expected, (
        f"{shape}: expected {expected} boundary faces, got {bf.shape[0]}"
    )


def test_vectorized_matches_reference_full_structured():
    """The vectorized algorithm must return the same set of boundary
    quads as the Python reference (face-counting) algorithm."""
    _, elems = _structured_hex_mesh(4, 3, 5)
    bf_new = sv.boundary_faces(elems)
    bf_ref = _boundary_faces_reference(elems)
    assert bf_new.shape == bf_ref.shape
    # Compare as sets of sorted-node tuples (ignore winding direction
    # and ordering of rows).
    set_new = {tuple(sorted(r[:4])) for r in bf_new}
    set_ref = {tuple(sorted(r[:4])) for r in bf_ref}
    assert set_new == set_ref


def test_vectorized_matches_reference_cutaway():
    """A general (unstructured) subset: remove a corner element and
    check that the algorithm correctly exposes the new interior faces."""
    _, elems = _structured_hex_mesh(3, 3, 3)
    # Drop element at (i=0, j=0, k=0) -> flat index 0.
    sub_elems = elems[1:]
    bf_new = sv.boundary_faces(sub_elems)
    bf_ref = _boundary_faces_reference(sub_elems)
    # Removing a corner element exposes 3 new boundary faces (and
    # removes 3 of the original outer ones) -> net same total.
    assert bf_new.shape[0] == bf_ref.shape[0]
    assert {tuple(sorted(r[:4])) for r in bf_new} == {
        tuple(sorted(r[:4])) for r in bf_ref
    }


def test_boundary_faces_empty_input():
    bf = sv.boundary_faces(np.empty((0, 8), dtype=np.int64))
    assert bf.shape == (0, 5)


def test_parent_element_id_is_in_range():
    _, elems = _structured_hex_mesh(3, 3, 3)
    bf = sv.boundary_faces(elems)
    assert bf[:, 4].min() >= 0
    assert bf[:, 4].max() < elems.shape[0]


# ---------------------------------------------------------------------------
# quads_to_triangles
# ---------------------------------------------------------------------------


def test_quads_to_triangles_doubles_rows_and_preserves_owner():
    _, elems = _structured_hex_mesh(2, 2, 2)
    bf = sv.boundary_faces(elems)
    tri = sv.quads_to_triangles(bf)
    assert tri.shape == (2 * bf.shape[0], 4)
    # Both triangles of each quad must reference the same parent elem.
    assert np.array_equal(tri[0::2, 3], bf[:, 4])
    assert np.array_equal(tri[1::2, 3], bf[:, 4])


# ---------------------------------------------------------------------------
# mesh3d_figure — vertex trimming + smoke
# ---------------------------------------------------------------------------


def test_mesh3d_figure_trims_vertex_payload():
    """Plotly should receive only the nodes referenced by boundary
    triangles, not the full mesh.  On a structured brick the interior
    nodes are dropped, so the saving is substantial."""
    nodes, elems = _structured_hex_mesh(4, 4, 4)
    fig = sv.mesh3d_figure(
        nodes, elems, cell_scalar=np.arange(elems.shape[0], dtype=float)
    )
    sent_x = np.asarray(fig.data[0].x)
    # Compute the expected unique-boundary-node count for cross-check.
    bf = sv.boundary_faces(elems)
    tri = sv.quads_to_triangles(bf)
    unique_boundary_nodes = np.unique(tri[:, :3].ravel())
    assert sent_x.size == unique_boundary_nodes.size
    # And it must be strictly smaller than the full node array
    # (otherwise the trimming did nothing).
    assert sent_x.size < nodes.shape[0]


def test_mesh3d_figure_i_indices_in_range():
    """The re-indexed (i, j, k) triangle indices must fit into the
    trimmed vertex array; an out-of-range index here would crash the
    browser's WebGL renderer with no Python error."""
    nodes, elems = _structured_hex_mesh(3, 3, 3)
    fig = sv.mesh3d_figure(nodes, elems)
    mesh = fig.data[0]
    n_verts = len(mesh.x)
    for arr_name in ("i", "j", "k"):
        idx = np.asarray(getattr(mesh, arr_name))
        assert idx.min() >= 0
        assert idx.max() < n_verts


def test_mesh3d_figure_cell_scalar_intensity_matches_triangles():
    nodes, elems = _structured_hex_mesh(3, 3, 3)
    cell_scalar = np.arange(elems.shape[0], dtype=float)
    fig = sv.mesh3d_figure(nodes, elems, cell_scalar=cell_scalar)
    intensity = np.asarray(fig.data[0].intensity)
    bf = sv.boundary_faces(elems)
    tri = sv.quads_to_triangles(bf)
    # intensity is per-triangle (cell mode), broadcast from parent elem.
    assert intensity.shape == (tri.shape[0],)
    assert np.array_equal(intensity, cell_scalar[tri[:, 3]])


def test_mesh3d_figure_vertex_scalar_sliced_to_kept_nodes():
    nodes, elems = _structured_hex_mesh(3, 3, 3)
    vertex_scalar = np.arange(nodes.shape[0], dtype=float)
    fig = sv.mesh3d_figure(nodes, elems, vertex_scalar=vertex_scalar)
    intensity = np.asarray(fig.data[0].intensity)
    sent_x = np.asarray(fig.data[0].x)
    assert intensity.shape == (sent_x.size,)


def test_deformed_mesh_figure_smoke():
    nodes, elems = _structured_hex_mesh(3, 3, 3)
    disp = np.zeros_like(nodes)
    disp[:, 2] = 0.01 * nodes[:, 0]
    fig = sv.deformed_mesh_figure(nodes, elems, disp, scale=10.0)
    assert fig is not None
    assert len(fig.data) == 1


def test_stress_contour_figure_smoke():
    nodes, elems = _structured_hex_mesh(3, 3, 3)
    stress = np.zeros((elems.shape[0], 6))
    stress[:, 2] = np.linspace(-100.0, 100.0, elems.shape[0])
    fig = sv.stress_contour_figure(nodes, elems, stress, component_index=2)
    assert fig is not None
    # cmin/cmax should be symmetric for signed stress.
    mesh = fig.data[0]
    assert mesh.cmin == pytest.approx(-mesh.cmax)


def test_fi_3d_figure_smoke():
    nodes, elems = _structured_hex_mesh(3, 3, 3)
    # fi_per_gauss has shape (n_elem, n_gauss).
    fi = np.random.RandomState(0).rand(elems.shape[0], 8)
    fig = sv.fi_3d_figure(nodes, elems, fi, criterion="MaxStress")
    assert fig is not None


# ---------------------------------------------------------------------------
# precomputed_geometry cache (issue #198)
# ---------------------------------------------------------------------------


def test_compute_mesh3d_geometry_shape_invariants():
    """The connectivity-derived geometry has the shapes the figure
    helpers expect: ``tri`` is (n_tri, 4), ``tri_ijk`` is (n_tri, 3),
    and ``kept_nodes`` indexes into the full node array."""
    nodes, elems = _structured_hex_mesh(3, 3, 3)
    geom = sv.compute_mesh3d_geometry(elems)
    assert geom["tri"].shape[1] == 4
    assert geom["tri_ijk"].shape == (geom["tri"].shape[0], 3)
    # tri_ijk indices must address the kept-nodes axis.
    assert geom["tri_ijk"].max() < geom["kept_nodes"].size
    # kept_nodes references real global node ids.
    assert geom["kept_nodes"].max() < nodes.shape[0]


def test_mesh3d_figure_precomputed_matches_uncached():
    """Passing ``precomputed_geometry=`` must produce a figure
    bit-identical to the uncached call."""
    nodes, elems = _structured_hex_mesh(4, 3, 3)
    cell_scalar = np.arange(elems.shape[0], dtype=float)

    fig_a = sv.mesh3d_figure(nodes, elems, cell_scalar=cell_scalar)
    geom = sv.compute_mesh3d_geometry(elems)
    fig_b = sv.mesh3d_figure(
        nodes, elems, cell_scalar=cell_scalar, precomputed_geometry=geom
    )

    for attr in ("x", "y", "z", "i", "j", "k", "intensity"):
        a = np.asarray(getattr(fig_a.data[0], attr))
        b = np.asarray(getattr(fig_b.data[0], attr))
        assert a.shape == b.shape, f"{attr} shape mismatch"
        assert np.array_equal(a, b), f"{attr} values differ"


def test_mesh3d_figure_precomputed_skips_boundary_faces(monkeypatch):
    """When ``precomputed_geometry`` is supplied, the expensive
    ``boundary_faces`` + ``quads_to_triangles`` calls must NOT run —
    the whole point of the cache."""
    nodes, elems = _structured_hex_mesh(4, 3, 3)
    geom = sv.compute_mesh3d_geometry(elems)

    bf_calls = {"n": 0}
    qt_calls = {"n": 0}

    real_bf = sv.boundary_faces
    real_qt = sv.quads_to_triangles

    def _counted_bf(e):
        bf_calls["n"] += 1
        return real_bf(e)

    def _counted_qt(q):
        qt_calls["n"] += 1
        return real_qt(q)

    monkeypatch.setattr(sv, "boundary_faces", _counted_bf)
    monkeypatch.setattr(sv, "quads_to_triangles", _counted_qt)

    sv.mesh3d_figure(
        nodes, elems,
        cell_scalar=np.arange(elems.shape[0], dtype=float),
        precomputed_geometry=geom,
    )
    assert bf_calls["n"] == 0, "boundary_faces ran despite cache hit"
    assert qt_calls["n"] == 0, "quads_to_triangles ran despite cache hit"

    # Sanity check: without the cache, both helpers DO run.
    sv.mesh3d_figure(nodes, elems)
    assert bf_calls["n"] == 1
    assert qt_calls["n"] == 1


def test_precomputed_geometry_threaded_through_wrappers():
    """The 3 high-level figure helpers must forward
    ``precomputed_geometry`` down to ``mesh3d_figure``; otherwise the
    app.py call sites silently pay the boundary-cull cost on every
    slider move."""
    nodes, elems = _structured_hex_mesh(3, 3, 3)
    geom = sv.compute_mesh3d_geometry(elems)
    stress = np.zeros((elems.shape[0], 6))
    stress[:, 2] = np.linspace(-100.0, 100.0, elems.shape[0])
    disp = np.zeros_like(nodes)
    fi = np.random.RandomState(0).rand(elems.shape[0], 8)

    # Each wrapper accepts the kwarg (it would raise TypeError if not).
    f1 = sv.stress_contour_figure(
        nodes, elems, stress, component_index=2, precomputed_geometry=geom,
    )
    f2 = sv.deformed_mesh_figure(
        nodes, elems, disp, scale=10.0, precomputed_geometry=geom,
    )
    f3 = sv.fi_3d_figure(
        nodes, elems, fi, criterion="MaxStress", precomputed_geometry=geom,
    )
    for fig in (f1, f2, f3):
        assert fig is not None
        assert len(fig.data) == 1
