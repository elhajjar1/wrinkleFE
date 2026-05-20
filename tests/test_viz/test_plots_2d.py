"""Tests for :mod:`wrinklefe.viz.plots_2d`.

Regression test for issue #12: ``plot_stress_through_thickness`` called
``ensure_axes(ax, figsize=FIGSIZE_SINGLE_TALL)`` but never imported the
constant, raising ``NameError`` on the default-axes path.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless; no display required

import matplotlib.pyplot as plt
import numpy as np
import pytest

from wrinklefe.core.mesh import MeshData
from wrinklefe.solver.results import FieldResults
from wrinklefe.viz.plots_2d import plot_stress_through_thickness


@pytest.fixture(autouse=True)
def _clean_figures():
    """Start and end each test with no open figures."""
    plt.close("all")
    yield
    plt.close("all")


def _make_minimal_field_results() -> FieldResults:
    """Build a tiny single-element FieldResults usable as a smoke fixture."""
    n_nodes = 8
    n_elem = 1
    n_gauss = 8

    # Unit cube hex8 — element_center will return (0.5, 0.5, 0.5).
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    mesh = MeshData(
        nodes=nodes,
        elements=np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=int),
        ply_ids=np.zeros(n_elem, dtype=int),
        fiber_angles=np.zeros(n_nodes),
        ply_angles=np.zeros(n_elem),
        nx=1,
        ny=1,
        nz=1,
    )

    stress_global = np.full((n_elem, n_gauss, 6), 10.0)
    zeros = np.zeros((n_elem, n_gauss, 6))

    return FieldResults(
        displacement=np.zeros((n_nodes, 3)),
        stress_global=stress_global,
        stress_local=zeros.copy(),
        strain_global=zeros.copy(),
        strain_local=zeros.copy(),
        mesh=mesh,
        laminate=None,
    )


def test_plot_stress_through_thickness_default_axes():
    """Smoke test: ax=None must not raise NameError (issue #12)."""
    field = _make_minimal_field_results()
    ax = plot_stress_through_thickness(field, x=0.5, y=0.5, component=0)
    assert ax is not None
    assert ax.figure is not None
    plt.close(ax.figure)


def test_plot_stress_through_thickness_with_explicit_axes():
    """The explicit-axes path also works and reuses the caller's figure."""
    field = _make_minimal_field_results()
    fig, ax_in = plt.subplots()
    ax_out = plot_stress_through_thickness(
        field, x=0.5, y=0.5, component=0, ax=ax_in
    )
    assert ax_out is ax_in
    plt.close(fig)
