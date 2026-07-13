"""Benchmark: vectorised wrinkle morphology on a large node cloud (#185).

``apply_to_nodes`` (mesh deformation) and ``fiber_angles_at_nodes``
(per-node misalignment) were vectorised over nodes in #185/#252. This
benchmark drives both on ~50k nodes to guard that vectorisation.
"""
from __future__ import annotations

import numpy as np
import pytest

from wrinklefe.core.morphology import WrinkleConfiguration
from wrinklefe.core.wrinkle import GaussianSinusoidal

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]

N_PLIES = 8
NX = 80
NY = 80  # 8 * 80 * 80 = 51_200 nodes


def _node_cloud():
    x = np.linspace(-8.0, 8.0, NX)
    y = np.linspace(-6.0, 6.0, NY)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    n_per_ply = xy.shape[0]
    ply_thickness = 0.183
    nodes = np.empty((N_PLIES * n_per_ply, 3), dtype=float)
    ply_ids = np.empty(N_PLIES * n_per_ply, dtype=int)
    for p in range(N_PLIES):
        sl = slice(p * n_per_ply, (p + 1) * n_per_ply)
        nodes[sl, 0] = xy[:, 0]
        nodes[sl, 1] = xy[:, 1]
        nodes[sl, 2] = (p + 0.5) * ply_thickness
        ply_ids[sl] = p
    return nodes, ply_ids


def test_bench_apply_and_angles(benchmark):
    nodes, ply_ids = _node_cloud()
    wrinkle = GaussianSinusoidal(
        amplitude=0.3, wavelength=8.0, width=6.0, center=0.0
    )
    cfg = WrinkleConfiguration.dual_wrinkle(
        profile=wrinkle, interface1=3, interface2=4, phase=0.0
    )

    def _morph():
        deformed = cfg.apply_to_nodes(nodes, ply_ids, n_plies=N_PLIES)
        angles = cfg.fiber_angles_at_nodes(nodes, ply_ids, n_plies=N_PLIES)
        return deformed, angles

    deformed, angles = benchmark(_morph)

    # Correctness invariant: shapes preserved, all finite, the wrinkle
    # actually displaces nodes and induces non-zero fibre tilt.
    assert deformed.shape == nodes.shape
    assert np.all(np.isfinite(deformed))
    assert np.abs(deformed[:, 2] - nodes[:, 2]).max() > 0.0
    assert angles.shape == (nodes.shape[0],)
    assert np.all(np.isfinite(angles))
    assert np.all(angles >= 0.0)
    assert angles.max() > 0.0
