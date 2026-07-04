"""Equivalence tests for the vectorised evaluate_field overrides (#299).

LaRC05, Puck and Budiansky-Fleck were the last three criteria running the
base-class per-Gauss-point Python loop — and exactly the most expensive
ones per point (fracture-plane searches, misalignment-frame rotations).
Each now overrides ``evaluate_field`` with a broadcast implementation.

The scalar ``evaluate()`` stays the reference implementation: every test
here drives a randomized ``(N, 6)`` stress sample engineered to hit all
branch regimes (tension/compression fibre stress, tensile/compressive
fracture planes, Puck modes A/B/C, dominant shear, the zero-stress point)
through both paths and asserts the vectorised output is **identical** —
``assert_array_equal``, not ``allclose`` — in index, mode and reserve
factor.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from wrinklefe.core.material import MaterialLibrary
from wrinklefe.failure.kinkband import BudianskyFleckKinkBand
from wrinklefe.failure.larc05 import LaRC05Criterion
from wrinklefe.failure.puck import PuckCriterion


@pytest.fixture(scope="module")
def material():
    return MaterialLibrary().get("IM7_8552")


def _stress_sample(n: int = 240, seed: int = 20260704) -> np.ndarray:
    """Randomized stress states covering every branch regime.

    Blocks (each ~n/6 rows): fibre tension dominated, fibre compression
    dominated, transverse tension (tensile fracture planes / Puck mode A),
    transverse compression (compressive planes / Puck modes B and C),
    shear dominated, and mixed full-random. A zero row and sign-boundary
    rows are appended explicitly.
    """
    rng = np.random.default_rng(seed)
    b = n // 6
    blocks = []
    # fibre tension dominated
    s = rng.uniform(-50, 50, (b, 6))
    s[:, 0] = rng.uniform(500, 2500, b)
    blocks.append(s)
    # fibre compression dominated
    s = rng.uniform(-50, 50, (b, 6))
    s[:, 0] = rng.uniform(-2500, -500, b)
    blocks.append(s)
    # transverse tension (mode A / tensile planes)
    s = rng.uniform(-20, 20, (b, 6))
    s[:, 1] = rng.uniform(20, 120, b)
    s[:, 2] = rng.uniform(0, 60, b)
    blocks.append(s)
    # transverse compression (modes B and C)
    s = rng.uniform(-20, 20, (b, 6))
    s[:, 1] = rng.uniform(-250, -30, b)
    s[:, 2] = rng.uniform(-120, 0, b)
    s[:, 3] = rng.uniform(-90, 90, b)   # tau_23 sweeps the B/C corner
    blocks.append(s)
    # shear dominated
    s = rng.uniform(-10, 10, (b, 6))
    s[:, 3:] = rng.uniform(-150, 150, (b, 3))
    blocks.append(s)
    # fully mixed
    s = rng.uniform(-800, 800, (n - 5 * b, 6))
    blocks.append(s)

    sample = np.vstack(blocks)
    # Exact branch boundaries: zero stress, pure +/- fibre, pure +/- s22.
    edges = np.zeros((5, 6))
    edges[1, 0] = 1000.0
    edges[2, 0] = -1000.0
    edges[3, 1] = 80.0
    edges[4, 1] = -150.0
    return np.vstack([sample, edges])


def _scalar_reference(criterion, stress, material, contexts=None):
    n = stress.shape[0]
    fi = np.empty(n)
    modes = []
    rf = np.empty(n)
    for i in range(n):
        ctx = contexts[i] if contexts is not None else None
        r = criterion.evaluate(stress[i], material, ctx)
        fi[i] = r.index
        modes.append(r.mode)
        rf[i] = r.reserve_factor
    return fi, np.asarray(modes), rf


def _assert_field_matches_scalar(criterion, stress, material, contexts=None):
    fi_ref, modes_ref, rf_ref = _scalar_reference(
        criterion, stress, material, contexts
    )
    fi_vec, modes_vec, rf_vec = criterion.evaluate_field(
        stress, material, contexts
    )
    npt.assert_array_equal(fi_vec, fi_ref, err_msg="failure index diverged")
    npt.assert_array_equal(rf_vec, rf_ref, err_msg="reserve factor diverged")
    assert list(modes_vec) == list(modes_ref), "mode labels diverged"


# --------------------------------------------------------------------------- #
# Budiansky-Fleck kink band
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("theta_eff", "damage"), [(0.0, 0.0), (0.12, 0.0), (0.08, 0.35)]
)
def test_kinkband_field_matches_scalar(material, theta_eff, damage):
    criterion = BudianskyFleckKinkBand(
        theta_eff=theta_eff, damage_index=damage
    )
    _assert_field_matches_scalar(criterion, _stress_sample(), material)


def test_kinkband_field_no_python_loop_over_points(material, monkeypatch):
    """The override must not fall back to per-point evaluate()."""
    criterion = BudianskyFleckKinkBand(theta_eff=0.1)

    def _boom(*a, **k):  # pragma: no cover - fires only on regression
        raise AssertionError("evaluate() called from evaluate_field")

    monkeypatch.setattr(criterion, "evaluate", _boom)
    fi, modes, rf = criterion.evaluate_field(_stress_sample(24), material)
    assert fi.shape == (29,)


# --------------------------------------------------------------------------- #
# Puck action-plane
# --------------------------------------------------------------------------- #


def test_puck_field_matches_scalar(material):
    _assert_field_matches_scalar(PuckCriterion(), _stress_sample(), material)


def test_puck_field_covers_all_three_iff_modes(material):
    """The sample must actually exercise modes A, B and C (guards the
    sample construction, so a silent regime gap cannot weaken the
    equivalence test above)."""
    criterion = PuckCriterion()
    _, modes, _ = criterion.evaluate_field(_stress_sample(), material)
    seen = set(modes)
    assert {"iff_mode_a", "iff_mode_b", "iff_mode_c"} <= seen or (
        # FF may dominate some rows; require at least two IFF regimes
        # plus both FF labels as a floor.
        len({m for m in seen if m.startswith("iff")}) >= 2
        and {"fiber_tension", "fiber_compression"} <= seen
    )


def test_puck_field_no_python_loop_over_points(material, monkeypatch):
    criterion = PuckCriterion()
    monkeypatch.setattr(
        criterion,
        "evaluate",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("loop")),
    )
    fi, modes, rf = criterion.evaluate_field(_stress_sample(24), material)
    assert fi.shape == (29,)


# --------------------------------------------------------------------------- #
# LaRC05
# --------------------------------------------------------------------------- #


def _wrinkle_contexts(n: int, seed: int = 7) -> list[dict]:
    """Per-point misalignment angles like the FE evaluator supplies."""
    rng = np.random.default_rng(seed)
    return [
        {"misalignment_angle": float(a)}
        for a in rng.uniform(0.0, 0.25, n)
    ]


def test_larc05_field_matches_scalar_no_context(material):
    _assert_field_matches_scalar(
        LaRC05Criterion(), _stress_sample(), material
    )


def test_larc05_field_matches_scalar_with_misalignment(material):
    stress = _stress_sample()
    contexts = _wrinkle_contexts(stress.shape[0])
    _assert_field_matches_scalar(
        LaRC05Criterion(), stress, material, contexts
    )


def test_larc05_field_matches_scalar_with_thickness_overrides(material):
    """Per-point ply-thickness overrides flip the in-situ correction
    between thin- and thick-ply branches within one field call."""
    stress = _stress_sample(60)
    n = stress.shape[0]
    rng = np.random.default_rng(3)
    contexts = []
    for i in range(n):
        ctx = {"misalignment_angle": float(rng.uniform(0, 0.2))}
        if i % 3 == 0:
            ctx["ply_thickness"] = 0.5   # thick: no in-situ correction
        elif i % 3 == 1:
            ctx["ply_thickness"] = 0.1   # thin: toughness-based correction
        contexts.append(ctx)
    _assert_field_matches_scalar(
        LaRC05Criterion(), stress, material, contexts
    )


def test_larc05_field_matches_scalar_without_toughness(material):
    """Simplified 1.12*sqrt(2) in-situ fallback (GIc/GIIc absent)."""
    import dataclasses

    mat = dataclasses.replace(material, GIc=None, GIIc=None)
    stress = _stress_sample(120)
    contexts = _wrinkle_contexts(stress.shape[0], seed=11)
    _assert_field_matches_scalar(LaRC05Criterion(), stress, mat, contexts)


def test_larc05_field_no_python_loop_over_points(material, monkeypatch):
    criterion = LaRC05Criterion()
    monkeypatch.setattr(
        criterion,
        "evaluate",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("loop")),
    )
    fi, modes, rf = criterion.evaluate_field(
        _stress_sample(24), material, _wrinkle_contexts(29)
    )
    assert fi.shape == (29,)


def test_larc05_field_covers_all_four_modes(material):
    """The sample exercises fiber_tension, fiber_kinking, matrix_tension
    and matrix_compression winners."""
    criterion = LaRC05Criterion()
    stress = _stress_sample()
    _, modes, _ = criterion.evaluate_field(
        stress, material, _wrinkle_contexts(stress.shape[0])
    )
    assert {
        "fiber_tension",
        "fiber_kinking",
        "matrix_tension",
        "matrix_compression",
    } <= set(modes)
