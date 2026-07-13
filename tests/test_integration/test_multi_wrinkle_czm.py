"""CZM delamination in multi-wrinkle FE configurations (issue #283).

Until #283 the combination of the two headline capabilities —
multi-wrinkle FE (issue #252) and cohesive-zone delamination — was
guarded off with ``NotImplementedError``. The guard encoded an
implementation assumption, not physics: cohesive interface placement
derived from "the single wrinkle crest". Placement is now resolved
per configuration: ``czm_interfaces="near_crest"`` nominates the
interface nearest *each* wrinkle (deduplicated), and every nominated
interface receives a cohesive layer along its **full length** — so a
delamination initiating at one crest can propagate along the shared
interface toward a neighbouring wrinkle (crest-to-crest link-up, the
recognised multi-wrinkle failure pattern in the Li 2025 specimen
family).

Regression anchors (from the issue):

* single-entry ``wrinkles`` CZM ≡ scalar-config CZM (bit-tight);
* two far-separated wrinkles ≈ two independent single-wrinkle solves
  (interaction → 0 with distance);
* two close wrinkles on a shared interface develop bridging damage in
  the region *between* the crests (qualitative link-up signal), which
  the far-separated pair does not.

The CZM Newton solves here use deliberately small meshes (8 plies,
ny=2, nz_per_ply=1, 8 increments) so each solve stays in the
tens-of-seconds range.
"""

from __future__ import annotations

import numpy as np
import pytest

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis, WrinkleSpec

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_ANGLES_8 = [0.0, 45.0, -45.0, 90.0, 90.0, -45.0, 45.0, 0.0]

# Shared geometry: A=0.3, lambda=8, w=2 at ply interface 3 under 3%
# tension initiates cohesive damage (max d ~ 0.15) and converges
# through the Newton solve on this mesh.
_SPEC = dict(amplitude=0.3, wavelength=8.0, width=2.0, ply_interface=3)
_COMMON = dict(
    amplitude=0.3, wavelength=8.0, width=2.0,
    morphology="graded", loading="tension",
    angles=list(_ANGLES_8),
    nx=32, ny=2, nz_per_ply=1,
    domain_length=28.0,
    applied_strain=0.03,
    enable_czm=True, czm_n_load_increments=8, verbose=False,
)

# phase = 2*pi*dx/lambda -> +/-2*pi shifts a crest by +/-lambda = 8 mm.
_FAR_PAIR = [
    WrinkleSpec(**_SPEC, phase_offset=-2.0 * np.pi),
    WrinkleSpec(**_SPEC, phase_offset=+2.0 * np.pi),
]


def _crest_damage(result, x_center: float, half_window: float = 4.0) -> float:
    """Max cohesive damage among interface elements near ``x_center``."""
    x = result.czm_element_centroids[:, 0]
    d = result.czm_damage.max(axis=1)
    mask = np.abs(x - x_center) < half_window
    assert mask.any()
    return float(d[mask].max())


class TestMultiWrinkleCzmEndToEnd:

    def test_two_wrinkles_czm_runs_and_links_one_interface(self):
        """Two non-overlapping wrinkles sharing an interface run
        end-to-end with ``enable_czm=True``; both nominate the same
        interface, which resolves to one continuous cohesive surface."""
        cfg = AnalysisConfig(**_COMMON, wrinkles=list(_FAR_PAIR))
        r = WrinkleAnalysis(cfg).run()

        assert r.czm_converged is True
        assert r.czm_interfaces_used == [3]
        assert r.czm_damage is not None and r.czm_damage.size > 0
        # The damage-prone geometry initiates at both crests.
        assert float(np.max(r.czm_damage)) > 0.05
        # CZM mesh carries the duplicated interface nodes.
        assert r.mesh is not None
        assert r.field_results is not None

    def test_two_interfaces_attributed_separately(self):
        """Wrinkles at different ply interfaces nominate one cohesive
        surface each; per-interface reporting attributes results to the
        correct interface index."""
        cfg = AnalysisConfig(
            **{**_COMMON, "applied_strain": 0.01,
               "czm_n_load_increments": 4},
            wrinkles=[
                WrinkleSpec(amplitude=0.3, wavelength=8.0, width=2.0,
                            ply_interface=2, phase_offset=-2.0 * np.pi),
                WrinkleSpec(amplitude=0.3, wavelength=8.0, width=2.0,
                            ply_interface=5, phase_offset=+2.0 * np.pi),
            ],
        )
        r = WrinkleAnalysis(cfg).run()

        assert r.czm_interfaces_used == [2, 5]
        assert sorted(r.czm_energy_per_interface) == [2, 5]
        assert sorted(r.czm_crack_length_per_interface) == [2, 5]
        report = r.czm_delamination_report
        assert report is not None


class TestSingleEntryEquivalence:

    def test_single_spec_czm_matches_scalar_graded_czm(self):
        """A one-entry ``wrinkles`` list produces the same CZM solution
        as the scalar graded config it denotes — same interface
        resolution, same mesh, same damage/energy/load-displacement
        (the multi-wrinkle CZM analogue of PR #276's linear-path
        equivalence)."""
        scalar_cfg = AnalysisConfig(**_COMMON, interface_1=3)
        multi_cfg = AnalysisConfig(
            **_COMMON,
            wrinkles=[WrinkleSpec(**_SPEC, phase_offset=0.0)],
        )
        r_scalar = WrinkleAnalysis(scalar_cfg).run()
        r_multi = WrinkleAnalysis(multi_cfg).run()

        assert r_scalar.czm_converged and r_multi.czm_converged
        assert r_multi.czm_interfaces_used == r_scalar.czm_interfaces_used
        # The equivalence is meaningful only if damage actually
        # initiated (otherwise everything is trivially zero).
        assert float(np.max(r_scalar.czm_damage)) > 0.05

        np.testing.assert_allclose(
            r_multi.mesh.nodes, r_scalar.mesh.nodes, rtol=0, atol=1e-12,
        )
        np.testing.assert_allclose(
            r_multi.czm_damage, r_scalar.czm_damage,
            rtol=1e-9, atol=1e-12,
        )
        np.testing.assert_allclose(
            r_multi.czm_load_displacement, r_scalar.czm_load_displacement,
            rtol=1e-9,
        )
        assert r_multi.czm_energy_dissipated == pytest.approx(
            r_scalar.czm_energy_dissipated, rel=1e-9,
        )


class TestFarSeparationIndependence:

    def test_far_separated_pair_matches_solo_runs(self):
        """Two far-separated wrinkles behave like two independent
        single-wrinkle solves: each crest's damage matches the solo run
        with the *same* off-centre placement (isolating interaction
        from boundary-proximity effects), and no damage develops in the
        gap between them."""
        L = _COMMON["domain_length"]
        pair_cfg = AnalysisConfig(**_COMMON, wrinkles=list(_FAR_PAIR))
        solo_left = AnalysisConfig(
            **_COMMON,
            wrinkles=[WrinkleSpec(**_SPEC, phase_offset=-2.0 * np.pi)],
        )
        solo_right = AnalysisConfig(
            **_COMMON,
            wrinkles=[WrinkleSpec(**_SPEC, phase_offset=+2.0 * np.pi)],
        )
        r_pair = WrinkleAnalysis(pair_cfg).run()
        r_left = WrinkleAnalysis(solo_left).run()
        r_right = WrinkleAnalysis(solo_right).run()
        assert r_pair.czm_converged
        assert r_left.czm_converged and r_right.czm_converged

        d_pair_left = _crest_damage(r_pair, L / 2.0 - 8.0)
        d_pair_right = _crest_damage(r_pair, L / 2.0 + 8.0)
        d_solo_left = _crest_damage(r_left, L / 2.0 - 8.0)
        d_solo_right = _crest_damage(r_right, L / 2.0 + 8.0)

        # Damage genuinely initiated at every crest.
        assert min(d_solo_left, d_solo_right) > 0.05
        # Interaction -> 0 with distance: measured deviation is ~2%
        # (left) / <1% (right) on this mesh; 10% leaves platform slack.
        assert d_pair_left == pytest.approx(d_solo_left, rel=0.10)
        assert d_pair_right == pytest.approx(d_solo_right, rel=0.10)

        # No damage bridges the 16 mm gap between the far crests.
        x = r_pair.czm_element_centroids[:, 0]
        d = r_pair.czm_damage.max(axis=1)
        mid_gap = np.abs(x - L / 2.0) < 1.5
        assert mid_gap.any()
        assert float(d[mid_gap].max()) == pytest.approx(0.0, abs=1e-12)


class TestCrestToCrestLinkUp:

    def test_close_wrinkles_bridge_damage_between_crests(self):
        """Two close wrinkles (crests 6 mm apart, overlapping supports)
        on a shared interface develop cohesive damage in the region
        *between* the crests — the crest-to-crest link-up mechanism —
        at a level comparable to the crest damage itself. The
        far-separated pair (previous test) shows exactly zero damage in
        its gap, so the bridging here is the wrinkle-interaction
        signal, not a far-field artefact."""
        L = _COMMON["domain_length"]
        # phase = 2*pi*dx/lambda; dx = +/-3 mm -> +/-0.75*pi.
        cfg = AnalysisConfig(
            **_COMMON,
            wrinkles=[
                WrinkleSpec(**_SPEC, phase_offset=-0.75 * np.pi),
                WrinkleSpec(**_SPEC, phase_offset=+0.75 * np.pi),
            ],
        )
        r = WrinkleAnalysis(cfg).run()
        assert r.czm_converged
        assert r.czm_interfaces_used == [3]

        x = r.czm_element_centroids[:, 0]
        d = r.czm_damage.max(axis=1)

        bridge = np.abs(x - L / 2.0) < 1.5
        far_field = np.abs(x - L / 2.0) > 10.0
        assert bridge.any() and far_field.any()

        d_bridge = float(d[bridge].max())
        d_crests = max(
            _crest_damage(r, L / 2.0 - 3.0, half_window=2.0),
            _crest_damage(r, L / 2.0 + 3.0, half_window=2.0),
        )
        # Bridging damage is present (measured ~0.11 vs crest ~0.13)...
        assert d_bridge > 0.5 * d_crests
        assert d_bridge > 0.05
        # ...while the far field stays undamaged.
        assert float(d[far_field].max()) == pytest.approx(0.0, abs=1e-12)
