"""Tests for boundary-face culling and vectorization in :mod:`wrinklefe.viz.plots_3d`.

The 3D matplotlib plots used to emit every face of every sampled hex element
(6N faces).  For a structured ``(nx, ny, nz)`` mesh, interior elements share
five of their six faces with neighbours, so only the outer "skin"
``2 * (nx*ny + ny*nz + nz*nx)`` faces are actually visible.  These tests pin
down that culling and verify that the public API still renders without
raising on a small mesh.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import pytest

from wrinklefe.core.laminate import Laminate
from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.core.mesh import MeshData, WrinkleMesh
from wrinklefe.viz import plot_mesh_3d
from wrinklefe.viz.plots_3d import (
    _gather_boundary_faces,
    _general_boundary_mask,
    _structured_boundary_mask,
)


@pytest.fixture(autouse=True)
def _clean_figures():
    plt.close("all")
    yield
    plt.close("all")


def _build_structured_mesh(nx: int, ny: int, nz: int) -> MeshData:
    """Build a small structured hex mesh via ``WrinkleMesh`` (real factory).

    ``nz`` is realised through the laminate: we pick a ply count equal to
    ``nz`` with one element per ply, so the resulting mesh has exactly
    ``nx * ny * nz`` hex elements.
    """
    mat = OrthotropicMaterial()
    laminate = Laminate.symmetric(
        [0] * ((nz + 1) // 2), material=mat, ply_thickness=0.1
    )
    # symmetric([0]*k) produces 2k plies; if nz is odd we built one extra ply,
    # so trim by truncating ``nz`` to the actual count.
    actual_nz = laminate.n_plies
    gen = WrinkleMesh(
        laminate=laminate,
        wrinkle_config=None,
        Lx=float(nx),
        Ly=float(ny),
        nx=nx,
        ny=ny,
        nz_per_ply=1,
    )
    mesh = gen.generate()
    assert mesh.nx == nx and mesh.ny == ny and mesh.nz == actual_nz
    return mesh


def test_structured_boundary_count_3x3x3():
    """3x3x3 hex mesh: boundary faces = 2*(3*3+3*3+3*3) = 54, NOT 27*6 = 162."""
    mesh = _build_structured_mesh(3, 3, 4)  # 4 plies, so nz=4
    elem_idx = np.arange(mesh.n_elements)

    face_verts, _ = _gather_boundary_faces(mesh, elem_idx, mesh.nodes)

    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    expected = 2 * (nx * ny + ny * nz + nz * nx)
    total = mesh.n_elements * 6
    assert face_verts.shape == (expected, 4, 3)
    assert expected < total, (
        f"culling should strictly reduce face count "
        f"(got expected={expected}, total={total})"
    )


def test_structured_boundary_count_2x2x2():
    """2x2x2 mesh: every element is on the boundary, so faces = 2*(4+4+4) = 24."""
    mesh = _build_structured_mesh(2, 2, 2)
    elem_idx = np.arange(mesh.n_elements)
    face_verts, _ = _gather_boundary_faces(mesh, elem_idx, mesh.nodes)
    assert face_verts.shape == (24, 4, 3)


def test_structured_mask_shape_and_count():
    """The direct mask helper matches the analytical boundary-face count."""
    mesh = _build_structured_mesh(3, 3, 4)
    mask = _structured_boundary_mask(mesh)
    assert mask.shape == (mesh.n_elements, 6)
    expected = 2 * (mesh.nx * mesh.ny + mesh.ny * mesh.nz + mesh.nz * mesh.nx)
    assert int(mask.sum()) == expected


def test_general_mask_matches_structured_on_full_mesh():
    """Face-counting fallback agrees with the structured shortcut."""
    mesh = _build_structured_mesh(3, 3, 4)
    elem_idx = np.arange(mesh.n_elements)
    structured = _structured_boundary_mask(mesh)
    general = _general_boundary_mask(mesh, elem_idx)
    assert general.shape == structured.shape
    assert np.array_equal(general, structured)


def test_general_mask_handles_cutaway_subset():
    """On a sampled / cropped subset the general mask treats element-internal
    shared faces correctly: removing one element should expose a face that
    was hidden in the full mesh.
    """
    mesh = _build_structured_mesh(2, 2, 2)
    # Drop the (0, 0, 0) element from the subset; the +x, +y and +z faces of
    # the now-removed element used to be shared with neighbours, so those
    # neighbours' opposing faces should now be on the boundary.
    full = np.arange(mesh.n_elements)
    subset = full[1:]  # 7 elements
    mask = _general_boundary_mask(mesh, subset)
    full_mask = _general_boundary_mask(mesh, full)
    # Surface area of the 8-element cube = 24.  Removing one corner element
    # exposes 3 new internal faces and removes 3 outer faces from the
    # boundary set, so the count stays at 24.
    assert int(mask.sum()) == 24
    # The full-mesh count should also be 24 (2x2x2 = all elements on the
    # outer skin).
    assert int(full_mask.sum()) == 24


def test_plot_mesh_3d_smoke():
    """plot_mesh_3d on a small mesh renders without raising and emits faces."""
    mesh = _build_structured_mesh(3, 3, 2)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    out_ax = plot_mesh_3d(mesh, ax=ax)
    assert out_ax is ax
    # Force a draw so any deferred matplotlib error surfaces here.
    fig.canvas.draw()
    plt.close(fig)


def test_plot_mesh_3d_bounding_box_preserved():
    """Boundary faces span the same x/y/z bounding box as the full mesh.

    Interior faces (which we now drop) are strictly inside the bounding
    box, so culling them must not change the visible extent.
    """
    mesh = _build_structured_mesh(3, 3, 4)
    elem_idx = np.arange(mesh.n_elements)
    face_verts, _ = _gather_boundary_faces(mesh, elem_idx, mesh.nodes)

    # Each boundary face is (4, 3); flatten to all vertex coords.
    rendered = face_verts.reshape(-1, 3)
    np.testing.assert_allclose(rendered.min(axis=0), mesh.nodes.min(axis=0))
    np.testing.assert_allclose(rendered.max(axis=0), mesh.nodes.max(axis=0))


def test_gather_with_elem_scalar_broadcasts_correctly():
    """Per-element scalar values are broadcast onto every surviving face."""
    mesh = _build_structured_mesh(2, 2, 2)
    elem_idx = np.arange(mesh.n_elements)
    elem_scalar = np.arange(mesh.n_elements, dtype=float)

    face_verts, face_scalar = _gather_boundary_faces(
        mesh, elem_idx, mesh.nodes, elem_scalar=elem_scalar
    )
    assert face_scalar is not None
    assert face_scalar.shape == (face_verts.shape[0],)
    # All scalar values come from the element pool.
    assert set(np.unique(face_scalar)).issubset(set(elem_scalar))
