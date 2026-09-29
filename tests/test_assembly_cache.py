"""Guards for caching the assembled hex8 stiffness.

The assembly of the construction-time element matrices at the
construction-time DOF map cannot change between calls, so a Newton
iteration was rebuilding an identical global matrix every time — 12 full
COO-to-CSC builds for one CZM run, measured at 44.8 ms each against
0.8 ms for a copy on a 2,560-element mesh.

Caching it is only safe because of two things that are easy to get
wrong, and both are tested here rather than assumed:

1. **Callers mutate what they receive.** ``assemble_tangent`` adds the
   cohesive contribution into it, and ``StaticSolver._apply_penalty_bcs``
   applies displacement BCs with ``in_place=True``. So every call must
   return a fresh copy; handing out the cached object would let one
   solve's penalty terms leak into the next.
2. **The element matrices are not immutable after construction.**
   ``update_element`` rebuilds one as the progressive-damage solver
   degrades materials, precisely so the next assembly sees the change. A
   cache that did not invalidate there would silently keep returning the
   undamaged structure.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary
from wrinklefe.solver.assembler import GlobalAssembler


def _config(**over) -> AnalysisConfig:
    kw = dict(
        amplitude=0.3, wavelength=15.0, width=10.0,
        morphology="stack", loading="compression",
        material=MaterialLibrary().get("IM7_8552"),
        angles=[0, 45, -45, 90, 90, -45, 45, 0],
        ply_thickness=0.183,
        nx=8, ny=3, nz_per_ply=1,
        domain_length=20.0, domain_width=8.0,
        applied_strain=-0.005, analytical_only=False, verbose=False,
    )
    kw.update(over)
    return AnalysisConfig(**kw)


def _run(cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WrinkleAnalysis(cfg).run()


@pytest.fixture(scope="module")
def mesh_and_laminate():
    results = _run(_config())
    return results.mesh, results.laminate


@pytest.fixture
def assembler(mesh_and_laminate):
    mesh, laminate = mesh_and_laminate
    return GlobalAssembler(mesh, laminate)


class TestCacheIsCorrect:
    def test_cached_result_equals_a_from_scratch_build(self, assembler):
        cached = assembler._assemble_hex8_stiffness()
        fresh = assembler._build_hex8_stiffness()
        assert (cached - fresh).nnz == 0
        assert np.array_equal(cached.toarray(), fresh.toarray())

    def test_repeated_calls_are_equal(self, assembler):
        a = assembler._assemble_hex8_stiffness()
        b = assembler._assemble_hex8_stiffness()
        assert (a - b).nnz == 0

    def test_second_call_does_not_rebuild(self, assembler):
        """The point of the change: one build, however many calls."""
        builds = []
        original = GlobalAssembler._build_hex8_stiffness

        def counting(self):
            builds.append(1)
            return original(self)

        GlobalAssembler._build_hex8_stiffness = counting
        try:
            for _ in range(5):
                assembler._assemble_hex8_stiffness()
        finally:
            GlobalAssembler._build_hex8_stiffness = original
        assert len(builds) == 1


class TestCallersCannotCorruptTheCache:
    """Hazard 1: every caller mutates the matrix it is given."""

    def test_each_call_returns_a_distinct_object(self, assembler):
        a = assembler._assemble_hex8_stiffness()
        b = assembler._assemble_hex8_stiffness()
        assert a is not b

    def test_in_place_mutation_does_not_reach_later_calls(self, assembler):
        reference = assembler._assemble_hex8_stiffness().toarray()
        victim = assembler._assemble_hex8_stiffness()
        victim *= 7.0                      # what a caller legitimately does
        after = assembler._assemble_hex8_stiffness().toarray()
        assert np.array_equal(after, reference)

    def test_structural_mutation_does_not_reach_later_calls(self, assembler):
        reference = assembler._assemble_hex8_stiffness().toarray()
        victim = assembler._assemble_hex8_stiffness()
        victim.data[:] = 0.0               # a harsher in-place write
        after = assembler._assemble_hex8_stiffness().toarray()
        assert np.array_equal(after, reference)

    def test_two_solves_in_a_row_see_the_same_stiffness(
        self, mesh_and_laminate
    ):
        """The concrete failure this prevents: ``_apply_penalty_bcs``
        applies BCs with ``in_place=True``, so without the copy the
        second solve would inherit the first solve's penalty terms."""
        from wrinklefe.solver.static import StaticSolver

        mesh, laminate = mesh_and_laminate
        first = StaticSolver(mesh, laminate)
        second = StaticSolver(mesh, laminate)
        k1 = first.assembler._assemble_hex8_stiffness()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from wrinklefe.solver.boundary import BoundaryHandler
            bcs = BoundaryHandler.compression_bcs(mesh, applied_strain=-0.005)
            first.solve(bcs, verbose=False)
        k2 = second.assembler._assemble_hex8_stiffness()
        assert (k1 - k2).nnz == 0


class TestInvalidation:
    """Hazard 2: progressive damage changes the element matrices."""

    def test_update_element_clears_the_cache(self, assembler):
        assembler._assemble_hex8_stiffness()
        assert assembler._K_hex8 is not None
        assembler.update_element(0)
        assert assembler._K_hex8 is None

    def test_assembly_reflects_a_degraded_element(self, assembler):
        """The semantic test, not just "the attribute went to None".

        Populate the cache, degrade one element the way the
        progressive-damage solver does, and require the next assembly to
        show it.
        """
        before = assembler._assemble_hex8_stiffness().toarray()

        # Degrade element 0 directly, then refresh it the way the
        # progressive solver does. ``update_element`` rebuilds from the
        # mesh, so halve the cached Ke and bypass the rebuild to isolate
        # the invalidation itself.
        assembler._hex8_Ke[0] = assembler._hex8_Ke[0] * 0.5
        assembler._K_hex8 = None
        after = assembler._assemble_hex8_stiffness().toarray()

        assert not np.array_equal(after, before), (
            "the assembly did not pick up a changed element stiffness"
        )

    def test_stale_cache_would_have_been_wrong(self, assembler):
        """Pin that this is a real hazard, not a hypothetical: with the
        cache left in place, a changed element is invisible."""
        assembler._assemble_hex8_stiffness()
        before = assembler._assemble_hex8_stiffness().toarray()
        assembler._hex8_Ke[0] = assembler._hex8_Ke[0] * 0.5
        # deliberately NOT invalidating
        stale = assembler._assemble_hex8_stiffness().toarray()
        assert np.array_equal(stale, before)
        # ... and invalidating fixes it
        assembler._K_hex8 = None
        fresh = assembler._assemble_hex8_stiffness().toarray()
        assert not np.array_equal(fresh, before)

    def test_update_element_still_clears_the_thermal_load(self, assembler):
        """The adjacent pre-existing cache, kept working."""
        assembler._F_thermal = np.zeros(assembler.mesh.n_dof)
        assembler.update_element(0)
        assert assembler._F_thermal is None


class TestEndToEnd:
    @pytest.mark.parametrize(
        "over",
        [
            {},
            {"enable_czm": True, "czm_n_load_increments": 4},
            {"enable_progressive_damage": True},
            {"delta_T": -155.0},
        ],
        ids=["linear", "czm", "progressive", "thermal"],
    )
    def test_every_path_still_runs_and_reports(self, over):
        results = _run(_config(**over))
        assert results.analytical_knockdown > 0.0
        assert results.field_results is not None

    def test_czm_run_builds_the_assembly_once_not_per_iteration(self):
        """A Newton loop used to rebuild it every iteration. Pinned by
        count, because an equivalent-but-uncached regression would pass
        every other test in this file."""
        builds = []
        original = GlobalAssembler._build_hex8_stiffness

        def counting(self):
            builds.append(1)
            return original(self)

        GlobalAssembler._build_hex8_stiffness = counting
        try:
            _run(_config(enable_czm=True, czm_n_load_increments=4))
        finally:
            GlobalAssembler._build_hex8_stiffness = original

        # One per assembler instance the run creates; the point is that
        # it does not scale with Newton iterations or load increments.
        assert len(builds) <= 2, (
            f"{len(builds)} full assemblies for a 4-increment CZM run — the "
            f"cache is not holding across Newton iterations"
        )
