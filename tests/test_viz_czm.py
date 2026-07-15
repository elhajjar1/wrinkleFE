"""Smoke tests for the Phase 5 CZM visualization functions.

These tests verify that each new plot helper in :mod:`wrinklefe.viz.plots_2d`
/ :mod:`wrinklefe.viz.plots_3d` runs without error on representative
synthetic inputs and returns the documented figure-like object.

Anti-goal: this is **not** image-regression.  Matplotlib's per-version
font / layout drift makes pixel comparisons too brittle to be useful for
plotting helpers like these.  We only check construction succeeds and
the return type / shape is sane.

The one end-to-end test (``test_czm_overview_figure_from_real_run``)
exercises :func:`czm_overview_figure` against an actual
``WrinkleAnalysis(enable_czm=True).run()`` to confirm the wrapper picks
up the new ``czm_element_centroids`` field correctly.
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")  # headless; no display required

import matplotlib.pyplot as plt
import numpy as np
import pytest

from wrinklefe.analysis import AnalysisConfig, AnalysisResults, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary
from wrinklefe.viz import (
    czm_overview_figure,
    plot_crack_front_3d,
    plot_damage_histogram,
    plot_energy_per_interface,
    plot_interface_damage_3d,
    plot_interface_damage_field,
    plot_load_displacement,
    plot_traction_separation,
)

# Force PyVista off-screen at import time so any plotter created during the
# test session does not attempt to open a window on a headless CI runner.
try:  # pragma: no cover - exercised in environments with PyVista
    import pyvista  # noqa: F401

    pyvista.OFF_SCREEN = True
except ImportError:  # pragma: no cover
    pass


@pytest.fixture(autouse=True)
def _clean_figures():
    """Start and end each test with no open matplotlib figures."""
    plt.close("all")
    yield
    plt.close("all")


# ----------------------------------------------------------------------
# Synthetic-input smoke tests for the 2D CZM plots
# ----------------------------------------------------------------------


def test_plot_traction_separation_runs():
    rng = np.random.default_rng(seed=0)
    n_inc = 12
    # Monotone normal opening + correlated traction (mock bilinear ramp).
    sep = np.zeros((n_inc, 3))
    sep[:, 0] = np.linspace(0.0, 0.05, n_inc)
    sep[:, 1:] = 0.5 * rng.standard_normal((n_inc, 2)) * 1e-3
    trc = np.zeros((n_inc, 3))
    trc[:, 0] = np.linspace(0.0, 40.0, n_inc) * np.exp(-np.linspace(0, 1.5, n_inc))
    trc[:, 1:] = rng.standard_normal((n_inc, 2)) * 0.5

    ax = plot_traction_separation(sep, trc, label="GP0")
    assert ax is not None
    # Single curve + legend = at least 1 line drawn.
    assert len(ax.get_lines()) >= 1


def test_plot_load_displacement_runs():
    n_inc = 20
    lam = np.linspace(0.0, 1.0, n_inc)
    u_norm = lam * 0.3 + 0.05 * lam**2
    ld = np.column_stack([lam, u_norm])

    ax = plot_load_displacement(ld, label="run-A")
    assert ax is not None
    assert ax.get_xlabel() != ""


def test_plot_damage_histogram_runs():
    rng = np.random.default_rng(seed=1)
    # Mixture: most points intact, a handful in the soft zone, a few failed.
    n_elem = 50
    n_gauss = 4
    d = np.clip(
        np.concatenate([
            rng.uniform(0.0, 0.05, n_elem * n_gauss - 12),
            rng.uniform(0.4, 0.7, 8),
            rng.uniform(0.95, 1.0, 4),
        ]).reshape(n_elem, n_gauss),
        0.0, 1.0,
    )

    ax = plot_damage_histogram(d, bins=15)
    assert ax is not None
    # Three reference lines (0, 0.5, 1) should be present.
    assert sum(1 for ln in ax.get_lines() if ln.get_linestyle() in (":", "--")) >= 3


def test_plot_interface_damage_field_runs():
    rng = np.random.default_rng(seed=2)
    n_elem = 40
    centroids = rng.uniform(low=[0.0, 0.0], high=[24.0, 12.0], size=(n_elem, 2))
    damage = rng.uniform(0.0, 1.0, n_elem)

    ax = plot_interface_damage_field(damage, centroids)
    assert ax is not None
    # Scatter creates exactly one PathCollection.
    assert len(ax.collections) >= 1


def test_plot_energy_per_interface_runs():
    # Multi-interface dict
    energy_multi = {3: 1.2e-2, 4: 0.4e-2, 5: 0.0}
    ax = plot_energy_per_interface(energy_multi)
    assert ax is not None
    # 3 bars
    assert len(ax.patches) == 3

    # Single-interface dict (graceful handling)
    plt.close("all")
    ax_single = plot_energy_per_interface({7: 5.0e-3})
    assert ax_single is not None
    assert len(ax_single.patches) == 1


# ----------------------------------------------------------------------
# czm_overview_figure: synthetic + real-run paths
# ----------------------------------------------------------------------


def _make_synthetic_results_with_czm() -> AnalysisResults:
    """Construct a barebones ``AnalysisResults`` with populated CZM fields.

    Used by tests that just want to drive ``czm_overview_figure`` and
    don't need a real solve.  Mesh / laminate are left ``None`` because
    the overview figure only reads ``czm_*`` fields.
    """
    mat = MaterialLibrary().get("IM7_8552")
    cfg = AnalysisConfig(
        amplitude=0.2, wavelength=16.0, width=12.0,
        morphology="concave", loading="tension",
        material=mat,
    )

    n_elem = 24
    n_gauss = 4
    rng = np.random.default_rng(seed=3)
    damage = np.clip(rng.uniform(0.0, 0.6, size=(n_elem, n_gauss)), 0.0, 1.0)
    centroids = rng.uniform(low=[0.0, 0.0], high=[24.0, 12.0], size=(n_elem, 2))
    load_disp = np.column_stack([
        np.linspace(0.0, 1.0, 10),
        np.linspace(0.0, 0.25, 10),
    ])

    results = AnalysisResults(
        config=cfg,
        czm_damage=damage,
        czm_separation=np.zeros((n_elem, n_gauss, 3)),
        czm_traction=np.zeros((n_elem, n_gauss, 3)),
        czm_energy_dissipated=1.0e-2,
        czm_energy_per_interface={3: 0.4e-2, 5: 0.6e-2},
        czm_crack_length_per_interface={3: 0.5, 5: 1.2},
        czm_load_displacement=load_disp,
        czm_converged=True,
        czm_interfaces_used=[3, 5],
        czm_element_centroids=centroids,
    )
    return results


def test_czm_overview_figure_from_synthetic():
    results = _make_synthetic_results_with_czm()
    fig = czm_overview_figure(results)
    assert fig is not None
    # 2x2 grid -> 4 main axes; matplotlib may add one extra axes per
    # colorbar (the interface-damage-field panel adds one), so count
    # only subplots whose title is one of the expected panel headers.
    titled = [ax for ax in fig.axes if ax.get_title()]
    assert len(titled) == 4


def test_czm_overview_raises_when_czm_disabled():
    mat = MaterialLibrary().get("IM7_8552")
    cfg = AnalysisConfig(
        amplitude=0.2, wavelength=16.0, width=12.0,
        morphology="stack", loading="compression",
        material=mat,
    )
    results = AnalysisResults(config=cfg)  # czm_damage left None.
    with pytest.raises(ValueError, match="enable_czm"):
        czm_overview_figure(results)


_LAYUP_0_90_4S = ([0, 90] * 4) + ([90, 0] * 4)


def test_czm_overview_figure_from_real_run():
    """End-to-end: run a tiny ``WrinkleAnalysis(enable_czm=True)`` and feed
    the result into :func:`czm_overview_figure`.

    The mesh is intentionally minimal (nx=6, ny=3, nz_per_ply=1) so the
    Newton-Raphson loop completes in a few seconds; we still get a
    nonzero damage field from the concave-tension geometry, which is all
    the overview figure needs to render every panel non-trivially.

    The applied strain (0.025) matches the recalibration in
    ``tests/test_analysis_czm.py`` for the corrected dual-wrinkle
    amplitude contract (issue #305): the concave mesh now composes to
    ~0.70*A, so the previous 0.015 no longer opened the crest past the
    cohesive initiation threshold.
    """
    mat = MaterialLibrary().get("IM7_8552")
    cfg = AnalysisConfig(
        amplitude=0.366, wavelength=16.0, width=12.0,
        morphology="concave", loading="tension",
        material=mat,
        angles=list(_LAYUP_0_90_4S),
        ply_thickness=0.183,
        nx=6, ny=3, nz_per_ply=1,
        applied_strain=0.025,
        enable_czm=True,
        czm_n_load_increments=8,
        verbose=False,
    )
    results = WrinkleAnalysis(cfg).run()

    assert results.czm_damage is not None
    assert results.czm_element_centroids is not None
    assert results.czm_element_centroids.shape[1] == 2
    assert results.czm_element_centroids.shape[0] == results.czm_damage.shape[0]

    fig = czm_overview_figure(results)
    assert fig is not None
    titled = [ax for ax in fig.axes if ax.get_title()]
    assert len(titled) == 4


# ----------------------------------------------------------------------
# PyVista 3D plots — gated on PyVista availability
# ----------------------------------------------------------------------


def _make_synthetic_interface_quads() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a small (2x2) quad grid with a synthetic damage field."""
    nx_q, ny_q = 2, 2  # 2x2 quads = 9 nodes
    xs = np.linspace(0.0, 1.0, nx_q + 1)
    ys = np.linspace(0.0, 1.0, ny_q + 1)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    nodes = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])

    def nid(i: int, j: int) -> int:
        return j * (nx_q + 1) + i

    quads = []
    for j in range(ny_q):
        for i in range(nx_q):
            quads.append([
                nid(i, j),
                nid(i + 1, j),
                nid(i + 1, j + 1),
                nid(i, j + 1),
            ])
    conn = np.asarray(quads, dtype=np.int64)
    damage = np.linspace(0.0, 1.0, conn.shape[0])
    return nodes, conn, damage


def test_plot_interface_damage_3d_runs():
    pytest.importorskip("pyvista")
    nodes, conn, damage = _make_synthetic_interface_quads()
    plotter = plot_interface_damage_3d(nodes, conn, damage)
    assert plotter is not None
    plotter.close()


def test_plot_crack_front_3d_runs():
    pytest.importorskip("pyvista")
    nodes, conn, damage = _make_synthetic_interface_quads()
    plotter = plot_crack_front_3d(nodes, conn, damage, threshold=0.5)
    assert plotter is not None
    plotter.close()


# ----------------------------------------------------------------------
# PyVista absent — the 3D plots must raise an actionable ImportError that
# names the `wrinklefe[vtk]` extra.  Runs on every CI job (with or without
# PyVista installed) because it forces the import to fail deterministically
# by planting a ``None`` sentinel in ``sys.modules`` (issue #302).
# ----------------------------------------------------------------------


def test_require_pyvista_raises_actionable_error_without_pyvista(monkeypatch):
    from wrinklefe.viz.plots_3d import _require_pyvista

    # A ``None`` entry makes ``import pyvista`` raise ImportError, regardless
    # of whether PyVista is actually installed in this environment.
    monkeypatch.setitem(sys.modules, "pyvista", None)
    with pytest.raises(ImportError, match=r"wrinklefe\[vtk\]"):
        _require_pyvista()


def test_plot_interface_damage_3d_raises_without_pyvista(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyvista", None)
    nodes, conn, damage = _make_synthetic_interface_quads()
    with pytest.raises(ImportError, match=r"wrinklefe\[vtk\]"):
        plot_interface_damage_3d(nodes, conn, damage)
