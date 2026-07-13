"""Benchmark: linear static direct solve on a coarse wrinkled mesh."""
from __future__ import annotations

import numpy as np
import pytest

from wrinklefe.solver.boundary import BoundaryHandler
from wrinklefe.solver.static import StaticSolver

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]


def test_bench_static_solve(benchmark, coarse_wrinkled_mesh):
    mesh, laminate = coarse_wrinkled_mesh
    bcs = BoundaryHandler.compression_bcs(mesh, applied_strain=-0.01)

    def _solve():
        return StaticSolver(mesh, laminate).solve(bcs)

    results = benchmark(_solve)

    # Correctness invariant: a full displacement field, all finite, and a
    # non-trivial response to the applied compressive strain.
    assert results.displacement.shape == (mesh.n_nodes, 3)
    assert np.all(np.isfinite(results.displacement))
    assert np.abs(results.displacement).max() > 0.0
