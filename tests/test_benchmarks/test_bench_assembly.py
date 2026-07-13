"""Benchmark: global stiffness assembly on a coarse wrinkled mesh."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from wrinklefe.solver.assembler import GlobalAssembler

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]


def test_bench_assemble_stiffness(benchmark, coarse_wrinkled_mesh):
    mesh, laminate = coarse_wrinkled_mesh

    def _assemble():
        return GlobalAssembler(mesh, laminate).assemble_stiffness()

    K = benchmark(_assemble)

    # Correctness invariant: a well-formed global stiffness matrix.
    n = mesh.n_dof
    assert K.shape == (n, n)
    assert sparse.issparse(K)
    assert K.nnz > 0
    assert np.all(np.isfinite(K.data))
    # Symmetric to numerical tolerance.
    assert abs(K - K.T).max() <= 1e-6 * abs(K).max()
