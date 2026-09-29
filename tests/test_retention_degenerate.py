"""Guards for the undefined-retention case on unidirectional layups.

``retention_factors[c] = max_FI_pristine(c) / max_FI_wrinkled(c)``. For an
all-0 deg laminate scored by LaRC05 the *pristine* term is ~1e-10: fibre
kinking needs a nonzero initial misalignment and a flat UD coupon has
none. The ratio is then an undefined quantity reported as ~0, which reads
as "no strength retained" when the truth is "this comparison does not
apply here".

Measured on the Li 2025 S-M-2 recipe: pristine max FI 1.66e-10 against a
wrinkled 0.682, giving a retention of 2.4e-10.

The number is still reported — nothing changes shape for existing
consumers — but it is now flagged, warned about, exported, and refused by
the one consumer where acting on it silently is worst: a mesh-convergence
study, where the artefact is identical at every refinement and so looks
perfectly converged.
"""

from __future__ import annotations

import logging
import warnings

import pytest

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.convergence import _qoi_strength_retention
from wrinklefe.core.material import MaterialLibrary


def _ud_config(**over) -> AnalysisConfig:
    """An all-0 deg UD laminate — the configuration that triggers it."""
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


@pytest.fixture(scope="module")
def ud_result():
    return _run(_ud_config())


@pytest.fixture(scope="module")
def md_result():
    return _run(_multidirectional_config())


class TestTheProblemIsReal:
    """Pin the underlying numbers, so this is documented behaviour and
    not a guard someone later removes as speculative."""

    def test_pristine_baseline_cannot_fail_on_a_ud_layup(self, ud_result):
        assert ud_result.baseline_fi is not None
        assert ud_result.baseline_fi["larc05"] < 1e-6

    def test_the_wrinkled_case_does_fail_normally(self, ud_result):
        """So the ~0 is the *baseline*, not a dead run."""
        assert ud_result.failure_indices is not None
        import numpy as np

        fi = np.asarray(ud_result.failure_indices["larc05"]).mean(axis=-1)
        assert float(fi[np.isfinite(fi)].max()) > 0.1

    def test_the_resulting_retention_looks_like_total_loss(self, ud_result):
        """The trap: a plausible-looking number meaning nothing."""
        assert ud_result.retention_factors["larc05"] < 1e-6


class TestFlagging:
    def test_ud_run_is_flagged(self, ud_result):
        assert ud_result.retention_degenerate == {"larc05": True}

    def test_multidirectional_run_is_not_flagged(self, md_result):
        assert md_result.retention_degenerate is not None
        assert not any(md_result.retention_degenerate.values())

    def test_multidirectional_retention_is_a_real_number(self, md_result):
        """The control: an off-axis layup gives a usable retention."""
        values = list(md_result.retention_factors.values())
        assert values and all(v > 1e-3 for v in values)

    def test_a_warning_is_logged_naming_the_cause(self, caplog):
        with caplog.at_level(logging.WARNING, logger="wrinklefe.analysis"):
            _run(_ud_config())
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "larc05" in text
        assert "unidirectional" in text
        assert "retention_degenerate" in text

    def test_flag_keys_match_the_retention_keys(self, ud_result):
        assert set(ud_result.retention_degenerate) == set(
            ud_result.retention_factors
        )


class TestConvergenceRefusesIt:
    """The one place acting on the artefact silently is worst."""

    def test_strength_retention_qoi_raises_on_a_degenerate_result(
        self, ud_result
    ):
        with pytest.raises(ValueError, match="undefined for this"):
            _qoi_strength_retention(ud_result)

    def test_the_message_names_an_alternative(self, ud_result):
        with pytest.raises(ValueError) as exc:
            _qoi_strength_retention(ud_result)
        assert "max_fi" in str(exc.value)

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
    def test_structured_export_carries_the_flag(self, ud_result, tmp_path):
        import json

        from wrinklefe.io.results import export_results_json

        out = tmp_path / "r.json"
        export_results_json(ud_result, out)
        knockdowns = json.loads(out.read_text())["knockdown_factors"]
        assert knockdowns["fe_retention_degenerate"] == ["larc05"]

    def test_structured_export_omits_it_when_sound(self, md_result, tmp_path):
        """A normal run's document is unchanged."""
        import json

        from wrinklefe.io.results import export_results_json

        out = tmp_path / "r.json"
        export_results_json(md_result, out)
        knockdowns = json.loads(out.read_text())["knockdown_factors"]
        assert "fe_retention_degenerate" not in knockdowns

    def test_summary_block_carries_the_flag(self, ud_result):
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
