"""Iterative-solver controls + loud ILU->diagonal fallback (issue #265).

Covers:

* The five ``AnalysisConfig`` iterative knobs plumb through to
  :class:`StaticSolver` and are honoured by ``_solve_iterative``.
* An iterative solve with the default knobs is bit-identical to an
  iterative solve built with the (formerly hardcoded) values passed
  explicitly — the zero-drift guard for the config plumbing.
* When ``spilu`` fails with one of the errors it uses to signal a real
  factorisation failure, a WARNING is logged *unconditionally* and the
  solve completes via the diagonal (Jacobi) fallback.  An *unexpected*
  exception type from ``spilu`` propagates instead of being swallowed.
* The CG non-convergence ``RuntimeError`` names the active
  preconditioner and the final relative residual.
* ``preconditioner='jacobi'`` never calls ``spilu``; ``'none'`` runs
  unpreconditioned.
* Round-trip of the new fields through ``to_dict`` / ``from_dict``.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import linalg as spla

from wrinklefe.analysis import AnalysisConfig
from wrinklefe.core.laminate import Laminate
from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.core.mesh import WrinkleMesh
from wrinklefe.solver import static as static_mod
from wrinklefe.solver.boundary import BoundaryHandler
from wrinklefe.solver.static import StaticSolver

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def single_ply_laminate():
    return Laminate.from_angles(
        [0.0], material=OrthotropicMaterial(), ply_thickness=0.183
    )


@pytest.fixture
def small_mesh(single_ply_laminate):
    gen = WrinkleMesh(
        laminate=single_ply_laminate,
        wrinkle_config=None,
        Lx=3.0, Ly=2.0,
        nx=3, ny=2, nz_per_ply=1,
    )
    return gen.generate()


@pytest.fixture
def solver(small_mesh, single_ply_laminate):
    """A StaticSolver on a tiny mesh (used to drive ``_solve_iterative``)."""
    return StaticSolver(small_mesh, single_ply_laminate)


def _spd_system(n: int = 40, cond_ill: bool = False):
    """Return a small symmetric-positive-definite (K, F) test system.

    ``cond_ill=True`` scales the diagonal so the matrix is poorly
    conditioned, useful to force CG non-convergence under a tight
    iteration cap.
    """
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n))
    K = A @ A.T + n * np.eye(n)  # SPD
    if cond_ill:
        scale = np.geomspace(1.0, 1e8, n)
        K = K * scale[:, None] * scale[None, :]
    F = rng.standard_normal(n)
    return sparse.csc_matrix(K), F


# ----------------------------------------------------------------------
# Config plumbing
# ----------------------------------------------------------------------

def test_config_defaults_match_previous_hardcoded_values():
    cfg = AnalysisConfig()
    assert cfg.iterative_rtol == 1e-10
    assert cfg.iterative_maxiter == 10000
    assert cfg.ilu_drop_tol == 1e-4
    assert cfg.ilu_fill_factor is None
    assert cfg.preconditioner == "ilu"


def test_static_solver_defaults_match_config_defaults(solver):
    """StaticSolver's own kwarg defaults equal the AnalysisConfig defaults."""
    assert solver.iterative_rtol == 1e-10
    assert solver.iterative_maxiter == 10000
    assert solver.ilu_drop_tol == 1e-4
    assert solver.ilu_fill_factor is None
    assert solver.preconditioner == "ilu"


def test_config_roundtrip_new_fields():
    cfg = AnalysisConfig(
        iterative_rtol=1e-8,
        iterative_maxiter=500,
        ilu_drop_tol=1e-3,
        ilu_fill_factor=12.0,
        preconditioner="jacobi",
    )
    restored = AnalysisConfig.from_dict(cfg.to_dict())
    assert restored == cfg
    for name in (
        "iterative_rtol", "iterative_maxiter", "ilu_drop_tol",
        "ilu_fill_factor", "preconditioner",
    ):
        assert getattr(restored, name) == getattr(cfg, name)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iterative_rtol": 0.0},
        {"iterative_rtol": -1e-10},
        {"iterative_maxiter": 0},
        {"iterative_maxiter": 1.5},
        {"ilu_drop_tol": -1e-4},
        {"ilu_fill_factor": 0.5},
        {"preconditioner": "bogus"},
    ],
)
def test_config_rejects_invalid_iterative_knobs(kwargs):
    with pytest.raises(ValueError):
        AnalysisConfig(**kwargs)


# ----------------------------------------------------------------------
# Zero-drift: default iterative solve bit-identical to explicit defaults
# ----------------------------------------------------------------------

@pytest.mark.slow
def test_default_iterative_solve_bit_identical(small_mesh, single_ply_laminate):
    """Default-knob iterative solve == explicit-(old-hardcoded)-knob solve."""
    bcs = BoundaryHandler.compression_bcs(small_mesh, applied_strain=-0.001)

    s_default = StaticSolver(small_mesh, single_ply_laminate)
    u_default = s_default.solve(bcs, solver="iterative").displacement

    s_explicit = StaticSolver(
        small_mesh, single_ply_laminate,
        iterative_rtol=1e-10, iterative_maxiter=10000,
        ilu_drop_tol=1e-4, ilu_fill_factor=None, preconditioner="ilu",
    )
    u_explicit = s_explicit.solve(bcs, solver="iterative").displacement

    # Bit-for-bit: identical construction of spilu/cg arguments.
    assert np.array_equal(u_default, u_explicit)


# ----------------------------------------------------------------------
# Preconditioner selection
# ----------------------------------------------------------------------

def test_build_preconditioner_none(solver):
    K, _ = _spd_system()
    solver.preconditioner = "none"
    M_op, active = solver._build_preconditioner(K)
    assert M_op is None
    assert active == "none"


def test_build_preconditioner_jacobi_skips_spilu(solver, monkeypatch):
    K, _ = _spd_system()
    solver.preconditioner = "jacobi"

    def _boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("spilu must not be called for preconditioner='jacobi'")

    monkeypatch.setattr(static_mod.spla, "spilu", _boom)
    M_op, active = solver._build_preconditioner(K)
    assert active == "jacobi"
    assert isinstance(M_op, spla.LinearOperator)


def test_build_preconditioner_ilu_forwards_drop_tol(solver, monkeypatch):
    K, _ = _spd_system()
    solver.preconditioner = "ilu"
    solver.ilu_drop_tol = 5e-3
    solver.ilu_fill_factor = None
    captured: dict = {}

    real_spilu = spla.spilu

    def _spy(mat, **kwargs):
        captured.update(kwargs)
        return real_spilu(mat, **kwargs)

    monkeypatch.setattr(static_mod.spla, "spilu", _spy)
    _M_op, active = solver._build_preconditioner(K)
    assert active == "ilu"
    assert captured["drop_tol"] == 5e-3
    # fill_factor is only forwarded when set.
    assert "fill_factor" not in captured


def test_build_preconditioner_ilu_forwards_fill_factor(solver, monkeypatch):
    K, _ = _spd_system()
    solver.preconditioner = "ilu"
    solver.ilu_fill_factor = 15.0
    captured: dict = {}

    real_spilu = spla.spilu

    def _spy(mat, **kwargs):
        captured.update(kwargs)
        return real_spilu(mat, **kwargs)

    monkeypatch.setattr(static_mod.spla, "spilu", _spy)
    solver._build_preconditioner(K)
    assert captured["fill_factor"] == 15.0


# ----------------------------------------------------------------------
# Loud, narrow ILU fallback
# ----------------------------------------------------------------------

def test_ilu_failure_warns_and_falls_back(solver, monkeypatch, caplog):
    """A spilu RuntimeError -> WARNING logged + solve completes via Jacobi."""
    K, F = _spd_system()

    def _raise_runtime(*a, **k):
        raise RuntimeError("simulated singular factor")

    monkeypatch.setattr(static_mod.spla, "spilu", _raise_runtime)

    with caplog.at_level(logging.WARNING, logger="wrinklefe.solver.static"):
        u = solver._solve_iterative(K, F)

    # The warning fired unconditionally and quotes the original error.
    assert any(
        "ILU preconditioner failed" in r.message
        and "RuntimeError" in r.message
        and "simulated singular factor" in r.message
        for r in caplog.records
    ), caplog.text
    # And the solve still produced a correct answer via the fallback.
    assert np.linalg.norm(K @ u - F) / np.linalg.norm(F) < 1e-6


def test_ilu_fallback_active_preconditioner_is_jacobi(solver, monkeypatch):
    K, _ = _spd_system()

    def _raise_memory(*a, **k):
        raise MemoryError("simulated OOM")

    monkeypatch.setattr(static_mod.spla, "spilu", _raise_memory)
    M_op, active = solver._build_preconditioner(K)
    assert active == "jacobi"
    assert isinstance(M_op, spla.LinearOperator)


def test_unexpected_spilu_exception_propagates(solver, monkeypatch):
    """An unexpected exception type is NOT swallowed by the fallback."""

    class WeirdError(Exception):
        pass

    def _raise_weird(*a, **k):
        raise WeirdError("not a factorisation failure")

    monkeypatch.setattr(static_mod.spla, "spilu", _raise_weird)
    K, F = _spd_system()
    with pytest.raises(WeirdError, match="not a factorisation failure"):
        solver._solve_iterative(K, F)


# ----------------------------------------------------------------------
# CG non-convergence diagnostics
# ----------------------------------------------------------------------

def test_cg_failure_message_names_preconditioner_and_residual(solver):
    """A capped, ill-conditioned solve raises naming precond + residual."""
    K, F = _spd_system(n=60, cond_ill=True)
    solver.preconditioner = "none"
    solver.iterative_maxiter = 1
    with pytest.raises(RuntimeError) as exc:
        solver._solve_iterative(K, F, maxiter=1)
    msg = str(exc.value)
    assert "preconditioner='none'" in msg
    assert "relative residual" in msg
    assert "maxiter=1" in msg
