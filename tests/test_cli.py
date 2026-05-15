"""Tests for the ``wrinklefe`` command-line interface.

These tests focus on how CLI flags map into :class:`AnalysisConfig` and how
they propagate into :meth:`WrinkleAnalysis.run` and the comparison/sweep
helpers. To keep them cheap, we patch the engine entry points and capture
the configs/kwargs the CLI hands them.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from wrinklefe.cli import main as cli_main


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _stub_analysis_run():
    """Build a context manager that stubs ``WrinkleAnalysis.run``.

    Returns a tuple ``(patcher, captured)`` where ``captured`` is a dict
    that, after the CLI call, holds ``"config"`` (the AnalysisConfig the
    handler built) and ``"analytical_only"`` (the kwarg passed to
    :meth:`WrinkleAnalysis.run`).
    """
    captured: dict = {}

    real_init = None  # filled in below

    def fake_run(self, analytical_only=None):
        captured["analytical_only"] = analytical_only
        captured["config"] = self.config
        result = MagicMock()
        result.summary.return_value = "<stubbed summary>"
        return result

    return captured, patch(
        "wrinklefe.analysis.WrinkleAnalysis.run", new=fake_run
    )


# --------------------------------------------------------------------------- #
# analyze: flag -> config plumbing
# --------------------------------------------------------------------------- #


def test_analyze_default_runs_full_fe_solve():
    """No flags should run a full FE solve, not analytical-only."""
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main(["analyze"])

    cfg = captured["config"]
    assert cfg.run_buckling is False
    assert cfg.run_montecarlo is False
    # The CLI used to hard-code analytical_only=True. Default is now
    # full FE.
    assert captured["analytical_only"] is False
    assert cfg.analytical_only is False


def test_analyze_buckling_flag_propagates():
    """--buckling must set run_buckling and force a full FE solve."""
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main(["analyze", "--buckling"])

    cfg = captured["config"]
    assert cfg.run_buckling is True
    assert cfg.run_montecarlo is False
    assert captured["analytical_only"] is False
    assert cfg.analytical_only is False


def test_analyze_montecarlo_flag_propagates():
    """--montecarlo must set run_montecarlo + mc_samples and run full FE."""
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main(["analyze", "--montecarlo", "--mc-samples", "37"])

    cfg = captured["config"]
    assert cfg.run_buckling is False
    assert cfg.run_montecarlo is True
    assert cfg.mc_samples == 37
    assert captured["analytical_only"] is False


def test_analyze_buckling_and_montecarlo_together():
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main(["analyze", "--buckling", "--montecarlo"])

    cfg = captured["config"]
    assert cfg.run_buckling is True
    assert cfg.run_montecarlo is True
    assert captured["analytical_only"] is False


def test_analyze_analytical_only_flag_skips_fe():
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main(["analyze", "--analytical-only"])

    cfg = captured["config"]
    assert cfg.run_buckling is False
    assert cfg.run_montecarlo is False
    assert captured["analytical_only"] is True
    assert cfg.analytical_only is True


def test_analyze_no_fe_flag_skips_fe():
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main(["analyze", "--no-fe"])

    assert captured["analytical_only"] is True


def test_analyze_fe_flag_runs_full_fe():
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main(["analyze", "--fe"])

    assert captured["analytical_only"] is False


def test_analyze_analytical_only_with_buckling_errors(capsys):
    """--analytical-only conflicts with --buckling and must exit 2."""
    captured, patcher = _stub_analysis_run()
    with patcher, pytest.raises(SystemExit) as exc_info:
        cli_main(["analyze", "--analytical-only", "--buckling"])

    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "analytical-only" in err.lower()


def test_analyze_passes_solver_and_mesh_options():
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main([
            "analyze",
            "--nx", "8",
            "--ny", "4",
            "--strain", "-0.005",
            "--solver", "iterative",
        ])

    cfg = captured["config"]
    assert cfg.nx == 8
    assert cfg.ny == 4
    assert cfg.applied_strain == pytest.approx(-0.005)
    assert cfg.solver == "iterative"


# --------------------------------------------------------------------------- #
# compare: --analytical-only / --no-analytical-only
# --------------------------------------------------------------------------- #


def _make_fake_result():
    r = MagicMock()
    r.morphology_factor = 1.0
    r.max_angle_rad = 0.0
    r.effective_angle_rad = 0.0
    r.damage_index = 0.0
    r.analytical_knockdown = 1.0
    r.analytical_strength_MPa = 100.0
    return r


def test_compare_default_is_analytical_only():
    captured: dict = {}

    def fake_compare(base_config, morphologies, analytical_only):
        captured["analytical_only"] = analytical_only
        return {m: _make_fake_result() for m in morphologies}

    with patch(
        "wrinklefe.analysis.WrinkleAnalysis.compare_morphologies",
        new=staticmethod(fake_compare),
    ):
        cli_main(["compare"])

    assert captured["analytical_only"] is True


def test_compare_no_analytical_only_runs_full_fe():
    captured: dict = {}

    def fake_compare(base_config, morphologies, analytical_only):
        captured["analytical_only"] = analytical_only
        return {m: _make_fake_result() for m in morphologies}

    with patch(
        "wrinklefe.analysis.WrinkleAnalysis.compare_morphologies",
        new=staticmethod(fake_compare),
    ):
        cli_main(["compare", "--no-analytical-only"])

    assert captured["analytical_only"] is False


# --------------------------------------------------------------------------- #
# sweep: --analytical-only / --no-analytical-only
# --------------------------------------------------------------------------- #


def test_sweep_default_is_analytical_only():
    captured: dict = {}

    def fake_sweep(base_config, parameter, values, analytical_only):
        captured["analytical_only"] = analytical_only
        captured["parameter"] = parameter
        captured["values"] = list(values)
        return [_make_fake_result() for _ in values]

    with patch(
        "wrinklefe.analysis.WrinkleAnalysis.parametric_sweep",
        new=staticmethod(fake_sweep),
    ):
        cli_main([
            "sweep",
            "--parameter", "amplitude",
            "--min", "0.1",
            "--max", "0.3",
            "--steps", "3",
        ])

    assert captured["analytical_only"] is True
    assert captured["parameter"] == "amplitude"
    assert len(captured["values"]) == 3


def test_sweep_no_analytical_only_runs_full_fe():
    captured: dict = {}

    def fake_sweep(base_config, parameter, values, analytical_only):
        captured["analytical_only"] = analytical_only
        return [_make_fake_result() for _ in values]

    with patch(
        "wrinklefe.analysis.WrinkleAnalysis.parametric_sweep",
        new=staticmethod(fake_sweep),
    ):
        cli_main([
            "sweep",
            "--parameter", "amplitude",
            "--min", "0.1",
            "--max", "0.3",
            "--steps", "2",
            "--no-analytical-only",
        ])

    assert captured["analytical_only"] is False


# --------------------------------------------------------------------------- #
# argparse exit codes for invalid input
# --------------------------------------------------------------------------- #


def test_unknown_morphology_exits_with_code_2():
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["analyze", "--morphology", "bogus"])
    assert exc_info.value.code == 2


def test_invalid_numeric_value_exits_with_code_2():
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["analyze", "--amplitude", "not-a-number"])
    assert exc_info.value.code == 2
