"""The CZM path's per-increment damage record, end to end (issue #453).

``AnalysisResults.czm_damage_history`` is filled on every CZM run. These
tests pin, through the public :class:`~wrinklefe.analysis.WrinkleAnalysis`
API:

* one record per converged increment, consistent with the final-state
  ``czm_damage`` the path already reported;
* the initiation load is exact on the real path: an elastic one-increment
  run and a ramp through initiation give the same onset;
* the record survives the lossless result archive;
* the CZM path now carries the material zones the linear path tagged
  (crest resin lens, tool-flat surface pockets) onto its rebuilt cohesive
  mesh, so ``tool_flat`` is combinable with ``enable_czm`` (needed to
  score Dataset M's delamination onset on its own FE recipe).
"""

from __future__ import annotations

import numpy as np
import pytest

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary

_LAYUP = ([0, 90] * 4) + ([90, 0] * 4)


def _cfg(**overrides) -> AnalysisConfig:
    """The concave tension case of ``test_analysis_czm`` on a coarser mesh.

    Initiation sits near 3.4 % strain on this mesh (2.75 % on the finer
    one there), so a ramp to 4 % crosses it.
    """
    kw = dict(
        amplitude=0.366, wavelength=16.0, width=12.0, morphology="concave",
        loading="tension", material=MaterialLibrary().get("IM7_8552"),
        angles=list(_LAYUP), ply_thickness=0.183, nx=10, ny=2,
        nz_per_ply=1, applied_strain=0.04, enable_czm=True,
        czm_n_load_increments=12, verbose=False,
    )
    kw.update(overrides)
    return AnalysisConfig(**kw)


@pytest.fixture(scope="module")
def ramp():
    return WrinkleAnalysis(_cfg()).run()


class TestRecord:
    def test_one_row_per_converged_increment(self, ramp):
        h = ramp.czm_damage_history
        assert h is not None and ramp.czm_converged
        assert h.n_recorded == 12 == h.n_increments and h.completed
        np.testing.assert_allclose(h.load_fraction, np.arange(1, 13) / 12)
        np.testing.assert_allclose(h.applied_strain, 0.04 * h.load_fraction)

    def test_last_row_matches_the_final_state(self, ramp):
        h = ramp.czm_damage_history
        assert h.max_damage[-1] == pytest.approx(float(ramp.czm_damage.max()))
        # The (wrinkled) interface is a little larger than its footprint.
        footprint = ramp.mesh.domain_size[0] * ramp.mesh.domain_size[1]
        assert footprint <= h.total_area <= 1.01 * footprint

    def test_the_ramp_crosses_initiation(self, ramp):
        h = ramp.czm_damage_history
        on = h.initiation()
        assert on is not None and on.exact and not on.extrapolated
        assert h.max_damage[0] == 0.0 and h.max_damage[-1] > 0.0
        assert h.damaged_area[-1] > 0.0
        # The onset lies inside the bracketing increments.
        k = on.increment - 1
        assert h.load_fraction[k - 1] <= on.load_fraction <= h.load_fraction[k]
        assert h.initiation_index[k - 1] < 1.0 <= h.initiation_index[k]

    def test_carried_stress_is_the_reaction_over_the_section(self, ramp):
        """Before any damage the coupon is linear: stress / strain is the
        same at every increment."""
        h = ramp.czm_damage_history
        undamaged = h.max_damage == 0.0
        ratio = h.nominal_stress[undamaged] / h.applied_strain[undamaged]
        assert ratio.size >= 2
        np.testing.assert_allclose(ratio, ratio[0], rtol=1e-6)
        # ~ the laminate's membrane modulus (the wrinkle is mild).
        assert 0.8 < ratio[0] / ramp.laminate.Ex < 1.05

    def test_initiation_is_exact_from_one_elastic_increment(self, ramp):
        """The onset does not depend on how the ramp reached it."""
        elastic = WrinkleAnalysis(
            _cfg(applied_strain=0.001, czm_n_load_increments=1)
        ).run().czm_damage_history
        on_e = elastic.initiation()
        on_r = ramp.czm_damage_history.initiation()
        assert on_e is not None and on_e.extrapolated and on_e.exact
        assert on_e.nominal_stress == pytest.approx(on_r.nominal_stress,
                                                    rel=1e-6)
        assert on_e.applied_strain == pytest.approx(on_r.applied_strain,
                                                    rel=1e-6)

    def test_the_summary_reports_the_onset(self, ramp):
        assert "Damage onset:" in ramp.summary()

    def test_no_record_without_czm(self):
        res = WrinkleAnalysis(_cfg(enable_czm=False)).run()
        assert res.czm_damage_history is None


def test_the_record_survives_the_archive(ramp, tmp_path):
    from wrinklefe.io.archive import load_results, save_results

    path = tmp_path / "czm.wfr"
    save_results(ramp, path)
    back = load_results(path).czm_damage_history
    h = ramp.czm_damage_history
    assert back is not None and back is not h
    for name in ("load_fraction", "nominal_stress", "max_damage",
                 "damaged_area", "separated_area", "initiation_index"):
        np.testing.assert_array_equal(getattr(back, name), getattr(h, name))
    assert back.initiation() == h.initiation()


# ----------------------------------------------------------------------
# Material zones on the rebuilt cohesive mesh
# ----------------------------------------------------------------------

def _toolflat(**overrides) -> AnalysisConfig:
    kw = dict(
        amplitude=0.3, wavelength=6.0, width=3.0, morphology="tool_flat",
        loading="tension", material=MaterialLibrary().get("T700GC_2510_AGATE"),
        angles=[0] * 8, ply_thickness=0.152, nx=12, ny=1, nz_per_ply=1,
        surface_pocket_side="bottom", surface_transition_plies=7,
        enable_surface_resin_pockets=True, applied_strain=0.002,
        czm_interfaces="all", czm_n_load_increments=1, verbose=False,
    )
    kw.update(overrides)
    return AnalysisConfig(**kw)


def test_tool_flat_runs_with_czm_and_keeps_its_surface_pockets():
    """Before issue #453 this combination was rejected; the CZM path
    rebuilt the mesh without the pockets. Now the cohesive mesh carries
    exactly the pocket tags the linear path computed."""
    linear = WrinkleAnalysis(_toolflat(enable_czm=False)).run()
    czm = WrinkleAnalysis(_toolflat(enable_czm=True)).run()
    assert czm.czm_converged
    assert linear.mesh.resin_blend is not None
    assert np.any(linear.mesh.resin_blend > 0.0)
    np.testing.assert_array_equal(czm.mesh.resin_blend,
                                  linear.mesh.resin_blend)
    assert czm.mesh.resin_material == linear.mesh.resin_material
    assert (czm.mesh.resin_blend_materials.keys()
            == linear.mesh.resin_blend_materials.keys())
    # The same hex geometry: the cohesive mesh only appends duplicated
    # interface nodes after the originals.
    n = linear.mesh.n_nodes
    np.testing.assert_allclose(czm.mesh.nodes[:n], linear.mesh.nodes,
                               atol=1e-12)
    assert czm.czm_damage_history.initiation() is not None


def test_the_crest_resin_lens_reaches_the_czm_mesh():
    cfg = dict(enable_resin_pocket=True, czm_n_load_increments=1,
               applied_strain=0.001)
    linear = WrinkleAnalysis(_cfg(enable_czm=False, **cfg)).run()
    czm = WrinkleAnalysis(_cfg(**cfg)).run()
    tagged = (linear.mesh.resin_blend if linear.mesh.resin_blend is not None
              else linear.mesh.resin_mask)
    assert tagged is not None and np.any(tagged)
    for name in ("resin_blend", "resin_mask"):
        a, b = getattr(linear.mesh, name), getattr(czm.mesh, name)
        assert (a is None) == (b is None)
        if a is not None:
            np.testing.assert_array_equal(a, b)
