"""Regression tests for issue #194: 3D matplotlib plots distort aspect ratio.

The four 3D plot helpers in :mod:`wrinklefe.viz.plots_3d` previously set
``xlim``/``ylim``/``zlim`` from the node-bounding box but never called
``ax.set_box_aspect(...)``. Matplotlib's default behaviour stretches data
into a roughly cubical viewport, so a thin laminate (~24 mm x ~12 mm x
~1 mm) appeared with its through-thickness axis blown up ~10x relative
to in-plane.

These tests render a tall-thin mesh (20 x 10 x 1 mm) and assert the
returned axes' ``get_box_aspect()`` is proportional to the physical
extents.
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
from wrinklefe.viz.style import set_axes_equal_aspect


@pytest.fixture(autouse=True)
def _clean_figures():
    plt.close("all")
    yield
    plt.close("all")


def _build_tall_thin_mesh() -> MeshData:
    """Build a 20 mm x 10 mm x ~1 mm structured hex mesh.

    Uses the real ``WrinkleMesh`` factory so the returned object is a
    valid :class:`MeshData` with sane node coordinates.
    """
    mat = OrthotropicMaterial()
    # 10 plies of 0.1 mm each => total z extent ~1.0 mm.
    laminate = Laminate.symmetric([0] * 5, material=mat, ply_thickness=0.1)
    gen = WrinkleMesh(
        laminate=laminate,
        wrinkle_config=None,
        Lx=20.0,
        Ly=10.0,
        nx=4,
        ny=3,
        nz_per_ply=1,
    )
    return gen.generate()


def _assert_proportional(observed, expected, *, rtol=1e-6):
    """Assert ``observed`` is a positive scalar multiple of ``expected``."""
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert observed.shape == expected.shape
    # Both must be strictly positive so the ratio is well-defined.
    assert np.all(observed > 0)
    assert np.all(expected > 0)
    ratios = observed / expected
    np.testing.assert_allclose(ratios, ratios[0] * np.ones_like(ratios), rtol=rtol)


def test_set_axes_equal_aspect_matches_extents():
    """The helper sets ``get_box_aspect`` proportional to ``maxs - mins``."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    mins = np.array([0.0, 0.0, 0.0])
    maxs = np.array([20.0, 10.0, 1.0])
    set_axes_equal_aspect(ax, mins, maxs)
    box = np.asarray(ax.get_box_aspect(), dtype=float)
    _assert_proportional(box, maxs - mins)
    plt.close(fig)


def test_set_axes_equal_aspect_handles_zero_extent():
    """A zero-extent axis must not crash and must yield a positive aspect."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    mins = np.array([0.0, 0.0, 0.0])
    maxs = np.array([20.0, 10.0, 0.0])  # degenerate z
    # Must not raise.
    set_axes_equal_aspect(ax, mins, maxs)
    box = np.asarray(ax.get_box_aspect(), dtype=float)
    # All entries strictly positive (zero extent substituted by 1.0).
    assert np.all(box > 0)
    # In-plane (x, y) must still preserve their ratio.
    _assert_proportional(box[:2], np.array([20.0, 10.0]))
    plt.close(fig)


def test_plot_mesh_3d_preserves_aspect_on_tall_thin_mesh():
    """plot_mesh_3d's axes box aspect is proportional to physical extents."""
    mesh = _build_tall_thin_mesh()
    extents = mesh.nodes.max(axis=0) - mesh.nodes.min(axis=0)

    # Sanity: this mesh really is tall-thin (~20 x 10 x 1).
    assert extents[0] > 5 * extents[2]
    assert extents[1] > 5 * extents[2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    out_ax = plot_mesh_3d(mesh, ax=ax)
    assert out_ax is ax

    box = np.asarray(ax.get_box_aspect(), dtype=float)
    _assert_proportional(box, extents)
    plt.close(fig)
