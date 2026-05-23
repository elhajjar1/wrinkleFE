"""Vectorised ``evaluate_field`` regression tests.

For each criterion that provides a vectorised ``evaluate_field`` override,
this test suite generates a random ``(N, 6)`` stress field and asserts that
the vectorised path produces failure index, mode, and reserve-factor values
identical (within float tolerance) to the scalar ``evaluate`` path.

It also exercises the :class:`FailureEvaluator.evaluate_field` plumbing to
confirm that material grouping + per-criterion vectorised dispatch returns
the same field arrays as the original triple-loop implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.failure.evaluator import FailureEvaluator
from wrinklefe.failure.hashin import HashinCriterion
from wrinklefe.failure.max_strain import MaxStrainCriterion
from wrinklefe.failure.max_stress import MaxStressCriterion
from wrinklefe.failure.tsai_hill import TsaiHillCriterion
from wrinklefe.failure.tsai_wu import TsaiWuCriterion


@pytest.fixture
def material() -> OrthotropicMaterial:
    """Default IM7/8552 material."""
    return OrthotropicMaterial()


@pytest.fixture
def stress_field() -> np.ndarray:
    """Random ``(N, 6)`` stress field used by every vectorised check."""
    rng = np.random.default_rng(20260521)
    # Realistic-magnitude stresses for IM7/8552: scale to a few hundred MPa
    # so all branches (tension/compression, matrix/fibre) are exercised.
    return rng.normal(0.0, 200.0, size=(256, 6))


# ----------------------------------------------------------------------
# Per-criterion vectorised-vs-scalar equivalence
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "criterion_cls",
    [MaxStressCriterion, MaxStrainCriterion, HashinCriterion,
     TsaiWuCriterion, TsaiHillCriterion],
)
def test_vectorised_matches_scalar(criterion_cls, material, stress_field):
    """Vectorised ``evaluate_field`` matches the scalar loop elementwise."""
    crit = criterion_cls()

    # Scalar reference path
    scalar_idx = np.empty(stress_field.shape[0], dtype=np.float64)
    scalar_modes = np.empty(stress_field.shape[0], dtype=object)
    scalar_rf = np.empty(stress_field.shape[0], dtype=np.float64)
    for i in range(stress_field.shape[0]):
        r = crit.evaluate(stress_field[i], material)
        scalar_idx[i] = r.index
        scalar_modes[i] = r.mode
        scalar_rf[i] = r.reserve_factor

    # Vectorised path
    vec_idx, vec_modes, vec_rf = crit.evaluate_field(stress_field, material)

    assert vec_idx.shape == (stress_field.shape[0],)
    assert vec_modes.shape == (stress_field.shape[0],)
    assert vec_rf.shape == (stress_field.shape[0],)

    assert_allclose(vec_idx, scalar_idx, rtol=1e-10, atol=1e-12)
    # Modes must match exactly (string equality).
    assert_array_equal(vec_modes.astype(str), scalar_modes.astype(str))
    # Reserve factors may include +inf for zero-stress branches; allow that.
    finite = np.isfinite(scalar_rf) & np.isfinite(vec_rf)
    assert_allclose(vec_rf[finite], scalar_rf[finite], rtol=1e-10, atol=1e-12)
    # Where one is inf, the other must be too.
    assert np.array_equal(np.isfinite(scalar_rf), np.isfinite(vec_rf))


def test_max_stress_pure_fibre_tension(material):
    """Spot check: pure fibre tension at Xt should yield FI = 1.0."""
    crit = MaxStressCriterion()
    s = np.zeros((1, 6))
    s[0, 0] = material.Xt
    fi, modes, rf = crit.evaluate_field(s, material)
    assert_allclose(fi, [1.0])
    assert modes[0] == "fiber_tension"
    assert_allclose(rf, [1.0])


def test_hashin_pure_fibre_compression(material):
    """Spot check: pure fibre compression at half Xc yields FI = 0.5."""
    crit = HashinCriterion()
    s = np.zeros((1, 6))
    s[0, 0] = -0.5 * material.Xc
    fi, modes, rf = crit.evaluate_field(s, material)
    assert_allclose(fi, [0.5])
    assert modes[0] == "fiber_compression"
    assert_allclose(rf, [2.0])


# ----------------------------------------------------------------------
# FailureEvaluator: vectorised dispatch produces identical field arrays
# ----------------------------------------------------------------------

def test_evaluator_field_matches_legacy_triple_loop(material):
    """``FailureEvaluator.evaluate_field`` matches a hand-rolled triple loop."""
    n_elem, n_gauss = 64, 4
    rng = np.random.default_rng(0xC0FFEE)
    stress = rng.normal(0.0, 150.0, size=(n_elem, n_gauss, 6))
    materials = [material]
    ply_ids = np.zeros(n_elem, dtype=int)

    ev = FailureEvaluator([
        MaxStressCriterion(),
        MaxStrainCriterion(),
        HashinCriterion(),
        TsaiWuCriterion(),
        TsaiHillCriterion(),
    ])
    fi_fields, mode_fields = ev.evaluate_field(stress, materials, ply_ids)

    # Reference: scalar triple loop, criterion-by-criterion
    for crit in ev.criteria:
        ref_fi = np.zeros((n_elem, n_gauss), dtype=np.float64)
        ref_modes = np.empty((n_elem, n_gauss), dtype="U32")
        for e in range(n_elem):
            for g in range(n_gauss):
                r = crit.evaluate(stress[e, g], material)
                ref_fi[e, g] = r.index
                ref_modes[e, g] = r.mode

        assert_allclose(fi_fields[crit.name], ref_fi, rtol=1e-10, atol=1e-12)
        assert_array_equal(mode_fields[crit.name], ref_modes)


def test_evaluator_field_multi_material(material):
    """Material grouping: distinct ply ids select distinct materials."""
    # Two materials with different Xt so that the same stress produces
    # different failure indices in each group.
    mat_a = material  # default
    mat_b = OrthotropicMaterial(name="weak", Xt=500.0)

    n_elem, n_gauss = 12, 2
    rng = np.random.default_rng(7)
    stress = rng.normal(0.0, 100.0, size=(n_elem, n_gauss, 6))
    # Half the elements use mat_a, half mat_b.
    ply_ids = np.array([0] * (n_elem // 2) + [1] * (n_elem - n_elem // 2),
                       dtype=int)

    ev = FailureEvaluator([MaxStressCriterion()])
    fi_fields, _ = ev.evaluate_field(stress, [mat_a, mat_b], ply_ids)

    # Reference triple loop using the correct material per element.
    crit = MaxStressCriterion()
    ref_fi = np.zeros((n_elem, n_gauss), dtype=np.float64)
    for e in range(n_elem):
        mat = mat_a if ply_ids[e] == 0 else mat_b
        for g in range(n_gauss):
            r = crit.evaluate(stress[e, g], mat)
            ref_fi[e, g] = r.index

    assert_allclose(fi_fields["max_stress"], ref_fi, rtol=1e-10, atol=1e-12)
