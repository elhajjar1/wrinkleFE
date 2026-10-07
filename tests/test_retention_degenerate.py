"""Guards for the undefined-retention case.

``retention_factors[c] = max_FI_pristine(c) / max_FI_wrinkled(c)``. When a
criterion cannot fail the *pristine* coupon at all, that ratio is an
undefined quantity reported as ~0, which reads as "no strength retained"
when the truth is "this comparison does not apply here". The number is
still reported, but it is flagged, warned about, exported, and refused by a
mesh-convergence study (where the artefact is identical at every
refinement and so looks perfectly converged).

LaRC05 used to be that criterion on unidirectional layups: its kinking mode
had no intrinsic misalignment, so a flat UD coupon could never kink
(pristine max FI ~1e-10 on the Li 2025 S-M-2 recipe). That is fixed — the
kinking model now carries the Xc-calibrated misalignment, so a pristine
ply kinks at its compressive strength (see ``TestLaRC05NoLongerTriggersIt``).

The guard is general, so it is now exercised with a criterion that
genuinely cannot fail a flat coupon: ``FI = |tau_13| / S13``, which is
exactly zero without a wrinkle and positive with one. It runs through the
real pipeline (both FE solves, retention, flags, exports), injected as the
default evaluator.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pytest

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.convergence import _qoi_strength_retention
from wrinklefe.core.material import MaterialLibrary
from wrinklefe.failure.base import FailureCriterion, FailureResult
from wrinklefe.failure.evaluator import FailureEvaluator
from wrinklefe.failure.larc05 import LaRC05Criterion


class _OutOfPlaneShearOnly(FailureCriterion):
    """``FI = |tau_13| / S13``: zero on a flat coupon, so its pristine
    baseline cannot fail — the condition the guard exists for."""

    name = "tau13"

    def evaluate(self, stress_local, material, context=None):
        fi = abs(float(stress_local[4])) / material.S13
        return FailureResult(
            index=fi, mode="shear_13",
            reserve_factor=1.0 / fi if fi > 0 else float("inf"),
            criterion_name=self.name,
        )


def _ud_config(**over) -> AnalysisConfig:
    """An all-0 deg UD laminate (the case that used to trigger the guard)."""
    kw = dict(
        amplitude=0.75, wavelength=12.9, width=12.9,
        morphology="graded", loading="compression",
        material=MaterialLibrary().get("AC318_S6C10"),
        angles=[0.0] * 14, ply_thickness=0.44,
        nx=8, ny=3, nz_per_ply=1,
        applied_strain=-0.01, analytical_only=False, verbose=False,
    )
    kw.update(over)
    return AnalysisConfig(**kw)


def _multidirectional_config(**over) -> AnalysisConfig:
    """A layup with off-axis plies — the baseline *can* fail."""
    kw = dict(
        amplitude=0.25, wavelength=16.0, width=12.0,
        morphology="stack", loading="compression",
        material=MaterialLibrary().get("IM7_8552"),
        angles=[0, 45, -45, 90, 90, -45, 45, 0],
        ply_thickness=0.183,
        nx=8, ny=3, nz_per_ply=1,
        applied_strain=-0.005, analytical_only=False, verbose=False,
    )
    kw.update(over)
    return AnalysisConfig(**kw)


def _run(cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WrinkleAnalysis(cfg).run()


def _run_with(criteria, cfg):
    """Run with ``criteria`` as the default evaluator (restored after)."""
    original = FailureEvaluator.default_criteria
    FailureEvaluator.default_criteria = classmethod(  # type: ignore[method-assign]
        lambda cls: cls(list(criteria))
    )
    try:
        return _run(cfg)
    finally:
        FailureEvaluator.default_criteria = original  # type: ignore[method-assign]


@pytest.fixture(scope="module")
def ud_result():
    """Default criteria (LaRC05) on the UD case: now sound."""
    return _run(_ud_config())


@pytest.fixture(scope="module")
def degenerate_result():
    """Every criterion degenerate: only the tau_13 criterion."""
    return _run_with([_OutOfPlaneShearOnly()], _ud_config())


@pytest.fixture(scope="module")
def partly_degenerate_result():
    """One sound criterion (LaRC05) alongside the degenerate one."""
    return _run_with(
        [LaRC05Criterion(), _OutOfPlaneShearOnly()], _ud_config()
    )


@pytest.fixture(scope="module")
def md_result():
    return _run(_multidirectional_config())


class TestLaRC05NoLongerTriggersIt:
    """Pin the fix: a UD pristine coupon now fails under LaRC05."""

    def test_the_pristine_ud_baseline_can_fail(self, ud_result):
        # 1.66e-10 before the kinking fix; a pristine ply now kinks at Xc.
        assert ud_result.baseline_fi["larc05"] > 0.1

    def test_it_is_not_flagged(self, ud_result):
        assert ud_result.retention_degenerate == {"larc05": False}

    def test_its_retention_is_a_real_strength_ratio(self, ud_result):
        assert 0.1 < ud_result.retention_factors["larc05"] <= 1.0


class TestTheProblemIsReal:
    """The condition the guard handles, reproduced through the pipeline."""

    def test_the_pristine_baseline_cannot_fail(self, degenerate_result):
        assert degenerate_result.baseline_fi["tau13"] < 1e-6

    def test_the_wrinkled_case_does_fail_normally(self, degenerate_result):
        """So the ~0 is the *baseline*, not a dead run."""
        fi = np.asarray(
            degenerate_result.failure_indices["tau13"]
        ).mean(axis=-1)
        assert float(fi[np.isfinite(fi)].max()) > 0.01

    def test_the_resulting_retention_looks_like_total_loss(
        self, degenerate_result
    ):
        """The trap: a plausible-looking number meaning nothing."""
        assert degenerate_result.retention_factors["tau13"] < 1e-6


class TestFlagging:
    def test_the_degenerate_criterion_is_flagged(self, degenerate_result):
        assert degenerate_result.retention_degenerate == {"tau13": True}

    def test_only_the_degenerate_criterion_is_flagged(
        self, partly_degenerate_result
    ):
        assert partly_degenerate_result.retention_degenerate == {
            "larc05": False, "tau13": True,
        }

    def test_multidirectional_run_is_not_flagged(self, md_result):
        assert md_result.retention_degenerate is not None
        assert not any(md_result.retention_degenerate.values())

    def test_multidirectional_retention_is_a_real_number(self, md_result):
        """The control: an off-axis layup gives a usable retention."""
        values = list(md_result.retention_factors.values())
        assert values and all(v > 1e-3 for v in values)

    def test_a_warning_is_logged_naming_the_cause(self, caplog):
        with caplog.at_level(logging.WARNING, logger="wrinklefe.analysis"):
            _run_with([_OutOfPlaneShearOnly()], _ud_config())
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "tau13" in text
        assert "retention_degenerate" in text

    def test_a_sound_run_logs_no_such_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="wrinklefe.analysis"):
            _run(_ud_config())
        assert not any(
            "retention_degenerate" in r.getMessage() for r in caplog.records
        )

    def test_flag_keys_match_the_retention_keys(
        self, partly_degenerate_result
    ):
        r = partly_degenerate_result
        assert set(r.retention_degenerate) == set(r.retention_factors)


class TestConvergenceRefusesIt:
    """The one place acting on the artefact silently is worst."""

    def test_strength_retention_qoi_raises_on_a_degenerate_result(
        self, degenerate_result
    ):
        with pytest.raises(ValueError, match="undefined for this"):
            _qoi_strength_retention(degenerate_result)

    def test_the_message_names_an_alternative(self, degenerate_result):
        with pytest.raises(ValueError) as exc:
            _qoi_strength_retention(degenerate_result)
        assert "max_fi" in str(exc.value)

    def test_a_real_partly_degenerate_run_returns_the_sound_minimum(
        self, partly_degenerate_result
    ):
        r = partly_degenerate_result
        assert _qoi_strength_retention(r) == pytest.approx(
            r.retention_factors["larc05"]
        )

    def test_a_healthy_result_still_returns_its_retention(self, md_result):
        value = _qoi_strength_retention(md_result)
        assert value == min(float(v) for v in md_result.retention_factors.values())

    def test_a_partially_degenerate_result_is_still_usable(self, md_result):
        """Only refuse when *every* criterion is degenerate — otherwise
        the minimum over the sound ones is still meaningful."""
        import copy

        patched = copy.copy(md_result)
        keys = list(patched.retention_factors)
        patched.retention_degenerate = {
            k: (i == 0) for i, k in enumerate(keys)
        }
        if len(keys) > 1:
            assert _qoi_strength_retention(patched) is not None


class TestExports:
    def test_structured_export_carries_the_flag(
        self, partly_degenerate_result, tmp_path
    ):
        import json

        from wrinklefe.io.results import export_results_json

        out = tmp_path / "r.json"
        export_results_json(partly_degenerate_result, out)
        knockdowns = json.loads(out.read_text())["knockdown_factors"]
        assert knockdowns["fe_retention_degenerate"] == ["tau13"]

    def test_structured_export_omits_it_when_sound(self, md_result, tmp_path):
        """A normal run's document is unchanged."""
        import json

        from wrinklefe.io.results import export_results_json

        out = tmp_path / "r.json"
        export_results_json(md_result, out)
        knockdowns = json.loads(out.read_text())["knockdown_factors"]
        assert "fe_retention_degenerate" not in knockdowns

    def test_summary_block_carries_the_flag(self, ud_result):
        # The summary takes the flag as given; any result supplies the
        # retention dict.
        from wrinklefe.io.export import build_analysis_summary

        summary = build_analysis_summary(
            defect={"amplitude_mm": 0.75},
            engineering={
                "analytical_knockdown": 0.5,
                "fe": {
                    "retention_factors": dict(ud_result.retention_factors),
                    "retention_degenerate": ["larc05"],
                },
            },
        )
        assert summary["engineering_analysis"]["finite_element"][
            "retention_degenerate"
        ] == ["larc05"]

    def test_summary_block_defaults_to_empty_when_sound(self):
        """A run with no degenerate criterion carries an empty list, not
        a missing key, so a consumer can check it unconditionally."""
        from wrinklefe.io.export import build_analysis_summary

        summary = build_analysis_summary(
            defect={"amplitude_mm": 0.25},
            engineering={
                "analytical_knockdown": 0.8,
                "fe": {"retention_factors": {"larc05": 0.81}},
            },
        )
        assert summary["engineering_analysis"]["finite_element"][
            "retention_degenerate"
        ] == []
