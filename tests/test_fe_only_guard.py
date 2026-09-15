"""FE-only features are refused on the analytical path — at BOTH entry points.

Several features exist only in the FE mesh or the FE solve: a transverse
surface, a general load state, cohesive interfaces, the progressive-damage
ramp, and the three per-element material zones.  Running any of them
through the analytical/CLT path cannot apply them, so the config is
refused rather than quietly producing a knockdown with the requested
physics switched off.

This used to be four separate ``if self.analytical_only`` blocks inside
``_validate``, and an audit found two gaps in that arrangement:

1. **A run-time override bypassed all of them.**  ``_validate`` only ever
   sees ``cfg.analytical_only``, so ``WrinkleAnalysis(cfg).run(
   analytical_only=True)`` on an FE-legal config ran straight through and
   dropped the feature.  That is the *common* way in, because
   ``parametric_sweep`` and ``probabilistic_analysis`` both default to the
   analytical path.
2. **Three features had no guard at either point** — ``enable_czm``,
   ``enable_progressive_damage`` and ``enable_resin_pocket`` were accepted
   at construction with ``analytical_only=True`` and ran silently.

Both entry points now consult one table, ``_FE_ONLY_FEATURES``.  The
parametrisation below is deliberately driven from that table, so a feature
added to it without a guard cannot slip through, and a feature guarded
without being listed cannot drift out of the tests.
"""

from __future__ import annotations

import pytest

from wrinklefe.analysis import (
    _FE_ONLY_FEATURES,
    AnalysisConfig,
    WrinkleAnalysis,
)
from wrinklefe.core.laminate import LoadState
from wrinklefe.core.material import MaterialLibrary

# One switch per FE-only feature, chosen so the config is otherwise valid.
FEATURE_KWARGS: dict[str, dict] = {
    "transverse_mode": dict(transverse_mode="gaussian_decay",
                            transverse_span=6.0, transverse_width=3.0),
    "load_state": dict(load_state=LoadState(Nx=-800.0)),
    "enable_czm": dict(enable_czm=True),
    "enable_progressive_damage": dict(enable_progressive_damage=True),
    "enable_resin_pocket": dict(enable_resin_pocket=True),
    "enable_surface_resin_pockets": dict(enable_surface_resin_pockets=True),
    "enable_vf_gradient": dict(enable_vf_gradient=True),
}

# The two material-zone features that only make sense on a tool-flat coupon.
TOOL_FLAT_ONLY = {"enable_surface_resin_pockets", "enable_vf_gradient"}


def _base(feature: str) -> dict:
    material = MaterialLibrary().get("IM7_8552")
    if feature in TOOL_FLAT_ONLY:
        return dict(
            amplitude=0.25, wavelength=16.0, width=12.0,
            morphology="tool_flat", surface_pocket_side="both",
            surface_transition_plies=2, material=material,
            angles=[0.0] * 24, ply_thickness=0.183,
            domain_length=40.0, domain_width=4.0,
            nx=40, ny=2, nz_per_ply=1,
        )
    return dict(
        amplitude=0.15, wavelength=12.0, width=8.0, morphology="graded",
        material=material,
        angles=[0.0, 45.0, -45.0, 90.0, 90.0, -45.0, 45.0, 0.0],
        ply_thickness=0.125, domain_length=16.0, domain_width=8.0,
        nx=8, ny=3, nz_per_ply=1,
    )


FEATURE_NAMES = [name for name, _pred, _why in _FE_ONLY_FEATURES]


def test_every_listed_feature_has_a_test_switch():
    """The table and this module must not drift apart."""
    assert set(FEATURE_NAMES) == set(FEATURE_KWARGS), (
        "_FE_ONLY_FEATURES and FEATURE_KWARGS disagree; a feature was added "
        "to one without the other"
    )


@pytest.mark.parametrize("feature", FEATURE_NAMES)
class TestRefusedOnBothPaths:

    def test_refused_at_construction(self, feature):
        with pytest.raises(ValueError, match=feature):
            AnalysisConfig(
                **_base(feature), analytical_only=True,
                **FEATURE_KWARGS[feature],
            )

    def test_refused_at_run_time(self, feature):
        """The gap an audit found: ``run(analytical_only=True)`` bypassed
        every construction guard."""
        cfg = AnalysisConfig(
            **_base(feature), analytical_only=False,
            **FEATURE_KWARGS[feature],
        )
        with pytest.raises(ValueError, match=feature):
            WrinkleAnalysis(cfg).run(analytical_only=True)

    def test_the_message_says_what_to_do(self, feature):
        with pytest.raises(ValueError) as exc:
            AnalysisConfig(
                **_base(feature), analytical_only=True,
                **FEATURE_KWARGS[feature],
            )
        msg = str(exc.value)
        assert feature in msg
        assert "analytical_only=False" in msg


class TestMessageQuality:

    def test_every_offender_is_named_not_just_the_first(self):
        with pytest.raises(ValueError) as exc:
            AnalysisConfig(
                **_base("load_state"), analytical_only=True,
                load_state=LoadState(Nx=-800.0),
                enable_czm=True,
                enable_progressive_damage=True,
            )
        msg = str(exc.value)
        for name in ("load_state", "enable_czm", "enable_progressive_damage"):
            assert name in msg

    def test_message_distinguishes_the_two_entry_points(self):
        kw = dict(**_base("enable_czm"), enable_czm=True)
        with pytest.raises(ValueError) as at_construction:
            AnalysisConfig(**kw, analytical_only=True)
        assert "AnalysisConfig(analytical_only=True)" in str(
            at_construction.value)

        cfg = AnalysisConfig(**kw, analytical_only=False)
        with pytest.raises(ValueError) as at_run:
            WrinkleAnalysis(cfg).run(analytical_only=True)
        assert "run(analytical_only=True)" in str(at_run.value)


class TestCleanConfigsAreUnaffected:
    """The guard must not fire for a config that asks for nothing FE-only."""

    def test_analytical_only_config_still_runs(self):
        cfg = AnalysisConfig(**_base("load_state"), analytical_only=True)
        result = WrinkleAnalysis(cfg).run(analytical_only=True)
        assert result.analytical_knockdown > 0.0

    def test_fe_config_still_runs_on_the_fe_path(self):
        cfg = AnalysisConfig(
            **_base("load_state"), analytical_only=False,
            load_state=LoadState(Nx=-800.0),
        )
        result = WrinkleAnalysis(cfg).run(analytical_only=False)
        assert result.field_results is not None


class TestLibraryEntryPointsThatDefaultToAnalytical:
    """``parametric_sweep`` and ``probabilistic_analysis`` default to the
    analytical path, so they were the common route into the bypass."""

    def test_parametric_sweep_refuses_rather_than_dropping(self):
        cfg = AnalysisConfig(
            **_base("load_state"), analytical_only=False,
            load_state=LoadState(Nx=-800.0),
        )
        with pytest.raises(ValueError, match="load_state"):
            WrinkleAnalysis.parametric_sweep(
                cfg, "amplitude", [0.1, 0.2], analytical_only=True,
            )

    def test_probabilistic_analysis_refuses_rather_than_dropping(self):
        from wrinklefe.stochastic import probabilistic_analysis

        cfg = AnalysisConfig(
            **_base("load_state"), analytical_only=False,
            load_state=LoadState(Nx=-800.0),
        )
        with pytest.raises(ValueError, match="load_state"):
            probabilistic_analysis(
                cfg, {"amplitude": ("normal", 0.15, 0.02)},
                n_samples=4, seed=1,
            )
