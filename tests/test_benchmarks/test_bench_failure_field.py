"""Benchmark: vectorised LaRC05 field evaluation (guards #299).

The per-Gauss-point failure evaluation is one of the hottest kernels in a
full analysis; #299 replaced the Python per-point loop with a broadcast
``evaluate_field``. This benchmark exercises it on a ~20k-point stress
array so a regression back toward the per-point path is caught.
"""
from __future__ import annotations

import numpy as np
import pytest

from wrinklefe.core.material import MaterialLibrary
from wrinklefe.failure.larc05 import LaRC05Criterion

from .conftest import stress_sample

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]

N_POINTS = 20_000


def test_bench_larc05_evaluate_field(benchmark):
    material = MaterialLibrary().get("IM7_8552")
    criterion = LaRC05Criterion()
    stress = stress_sample(N_POINTS)

    def _evaluate():
        return criterion.evaluate_field(stress, material)

    indices, modes, reserve_factors = benchmark(_evaluate)

    # Correctness invariant: one result per point, all finite and
    # physically ordered (failure index >= 0, reserve factor > 0).
    assert indices.shape == (N_POINTS,)
    assert modes.shape == (N_POINTS,)
    assert reserve_factors.shape == (N_POINTS,)
    assert np.all(np.isfinite(indices))
    assert np.all(indices >= 0.0)
    assert np.all(reserve_factors > 0.0)
