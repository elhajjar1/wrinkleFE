"""Guards for reusing the mechanical solve in the reaction-modulus step.

``_reaction_modulus`` used to solve its own compression problem. When
``delta_T == 0`` and no ``load_state`` is set, that problem is the one
the run has already solved — bit for bit — so the solve is skipped and
the stiffness and field are handed over instead. Four FE solves per
analysis become two.

Two things have to hold, and both are tested here:

1. **The answer must not move.** Every test that checks a number
   recomputes it with a genuinely independent solve and demands an
   exact match, not a tolerance.
2. **The fast path must actually be taken.** A regression that always
   fell back would still be bit-identical and would silently give the
   speedup back, so the solve count is pinned directly.

The gate is narrow on purpose: with a cure ``delta_T`` the reaction
solve is deliberately thermal-free and its displacement genuinely
differs, and with a ``load_state`` the mechanical BCs are a traction set
rather than the uniaxial compression the reaction solve uses.
"""

from __future__ import annotations

import warnings

import pytest

from wrinklefe.analysis import (
    AnalysisConfig,
    WrinkleAnalysis,
    _reaction_solve_duplicates_mechanical,
)
from wrinklefe.core.laminate import LoadState
from wrinklefe.core.material import MaterialLibrary
from wrinklefe.solver.static import StaticSolver


def _config(**over) -> AnalysisConfig:
    kw = dict(
        amplitude=0.3, wavelength=15.0, width=10.0,
        morphology="stack", loading="compression",
        material=MaterialLibrary().get("IM7_8552"),
        angles=[0, 45, -45, 90, 90, -45, 45, 0],
        ply_thickness=0.183,
        nx=6, ny=3, nz_per_ply=1,
        domain_length=20.0, domain_width=8.0,
        applied_strain=-0.005, analytical_only=False, verbose=False,
    )
    kw.update(over)
    return AnalysisConfig(**kw)


def _run(cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WrinkleAnalysis(cfg).run()


def _count_solves(cfg) -> int:
    """How many full FE solves one analysis performs."""
    calls = []
    original = StaticSolver.solve

    def counting(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    StaticSolver.solve = counting
    try:
        _run(cfg)
    finally:
        StaticSolver.solve = original
    return len(calls)


# ----------------------------------------------------------------------
# The gate
# ----------------------------------------------------------------------

class TestPrecondition:
    """The predicate is the single source of truth for when reuse is safe."""

    def test_plain_mechanical_run_is_reusable(self):
        assert _reaction_solve_duplicates_mechanical(_config()) is True

    def test_thermal_run_is_not_reusable(self):
        """``_reaction_modulus`` is thermal-free by design, so a cure
        ``delta_T`` makes the two displacements genuinely different."""
        assert _reaction_solve_duplicates_mechanical(
            _config(delta_T=-155.0)
        ) is False

    def test_load_state_run_is_not_reusable(self):
        """A load state replaces the uniaxial BCs with a traction set."""
        assert _reaction_solve_duplicates_mechanical(
            _config(load_state=LoadState(Nx=-1200.0))
        ) is False

    def test_both_together_are_not_reusable(self):
        assert _reaction_solve_duplicates_mechanical(
            _config(delta_T=-155.0, load_state=LoadState(Nx=-1200.0))
        ) is False

    def test_a_tiny_delta_t_still_blocks_reuse(self):
        """The test is exact equality with zero, not a tolerance: any
        thermal load at all changes the displacement."""
        assert _reaction_solve_duplicates_mechanical(
            _config(delta_T=1.0e-12)
        ) is False


# ----------------------------------------------------------------------
# The fast path is real
# ----------------------------------------------------------------------

class TestSolveCount:
    """Without this, an always-fall-back regression would pass every
    equivalence test in this file while quietly undoing the speedup."""

    def test_reusable_run_performs_two_solves_not_four(self):
        assert _count_solves(_config()) == 2

    def test_thermal_run_still_performs_four(self):
        assert _count_solves(_config(delta_T=-155.0)) == 4

    def test_load_state_run_still_performs_four(self):
        assert _count_solves(_config(load_state=LoadState(Nx=-1200.0))) == 4


# ----------------------------------------------------------------------
# The answer does not move
# ----------------------------------------------------------------------

class TestEquivalence:
    """Recompute independently and demand exactness."""

    def _independent_global_modulus(self, results, cfg):
        """``modulus_retention_global`` from two fresh, unaided solves."""
        analysis = WrinkleAnalysis(cfg)
        laminate = results.laminate
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e_w = analysis._reaction_modulus(
                results.mesh, laminate, cfg.applied_strain,
            )
            flat_mesh = analysis._build_flat_mesh(laminate)
            e_p = analysis._reaction_modulus(
                flat_mesh, laminate, cfg.applied_strain,
            )
        return abs(e_w) / abs(e_p)

    def test_reused_modulus_equals_an_independent_solve(self):
        cfg = _config()
        results = _run(cfg)
        assert results.modulus_retention_global == (
            self._independent_global_modulus(results, cfg)
        )

    @pytest.mark.parametrize(
        "over",
        [
            {"delta_T": -155.0},
            {"load_state": LoadState(Nx=-1200.0)},
            {"delta_T": -155.0, "load_state": LoadState(Nx=-1200.0)},
        ],
        ids=["thermal", "load_state", "both"],
    )
    def test_fallback_paths_are_unaffected(self, over):
        """These never take the fast path; they must be untouched."""
        cfg = _config(**over)
        results = _run(cfg)
        assert results.modulus_retention_global == (
            self._independent_global_modulus(results, cfg)
        )

    def test_reuse_and_fallback_agree_with_each_other(self):
        """The strongest statement: the same physical problem, answered
        both ways, gives the same bits."""
        cfg = _config()
        reused = _run(cfg).modulus_retention_global
        independent = self._independent_global_modulus(_run(cfg), cfg)
        assert reused == independent


# ----------------------------------------------------------------------
# Hand-off hygiene
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def solved():
    """One plain analysis, reused by the hand-off tests."""
    cfg = _config()
    return cfg, _run(cfg)


class TestHandoffSafety:
    """A partial hand-off must fall back, never mix."""

    def test_stiffness_without_a_field_falls_back(self, solved):
        cfg, results = solved
        analysis = WrinkleAnalysis(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = analysis._reaction_modulus(
                results.mesh, results.laminate, cfg.applied_strain,
                K=object(),      # a stiffness that must never be used
                field=None,
            )
            expected = analysis._reaction_modulus(
                results.mesh, results.laminate, cfg.applied_strain,
            )
        assert value == expected

    def test_field_without_a_stiffness_falls_back(self, solved):
        cfg, results = solved
        analysis = WrinkleAnalysis(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = analysis._reaction_modulus(
                results.mesh, results.laminate, cfg.applied_strain,
                K=None, field=results.field_results,
            )
            expected = analysis._reaction_modulus(
                results.mesh, results.laminate, cfg.applied_strain,
            )
        assert value == expected

    def test_zero_applied_strain_short_circuits_before_any_solve(self, solved):
        """Guarded before the hand-off, so a bogus K can't be reached."""
        cfg, results = solved
        analysis = WrinkleAnalysis(cfg)
        assert analysis._reaction_modulus(
            results.mesh, results.laminate, 0.0, K=object(), field=object(),
        ) is None


class TestCzmPathUnaffected:
    def test_czm_returns_before_retention_factors(self):
        """The CZM displacement comes from Newton, not from the linear
        solver, so it must never be handed to the reaction step. It is
        not, because ``run`` returns on that branch first — pinned here
        so a future reordering has to notice."""
        import inspect

        src = inspect.getsource(WrinkleAnalysis.run)
        czm = src.index("if cfg.enable_czm:")
        ret = src.index("_compute_retention_factors")
        branch_return = src.index("return results", czm)
        assert branch_return < ret, (
            "the CZM branch no longer returns before retention factors are "
            "computed; the reaction-solve reuse gate assumes it does"
        )
