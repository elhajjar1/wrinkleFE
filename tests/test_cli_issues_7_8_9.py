"""Regression smoke tests for issues #7, #8, and #9.

These three critical bugs all sat in the same CLI -> analysis path and
caused ``wrinklefe analyze``/``compare``/``sweep`` to crash with cascading
``TypeError``/``AttributeError`` errors.  The fixes were:

* #7: ``AnalysisConfig`` accepts ``run_buckling``, ``run_montecarlo``,
  and ``mc_samples`` (the docstring already documented them).
* #8: ``WrinkleAnalysis.run`` accepts ``analytical_only=True`` and
  short-circuits past the FE assembly / static solve / failure /
  retention steps.
* #9: ``compare_morphologies`` and ``parametric_sweep`` accept
  ``analytical_only`` and propagate it to the per-config
  :meth:`WrinkleAnalysis.run` call.

The tests below are intentionally narrow construction/smoke checks that
would have caught each crash; broader plumbing behaviour is covered in
``tests/test_cli.py`` and ``tests/test_analysis_sweep.py``.
"""

from __future__ import annotations

import numpy as np

from wrinklefe.analysis import (
    AnalysisConfig,
    AnalysisResults,
    WrinkleAnalysis,
)


# --------------------------------------------------------------------------- #
# Issue #7: AnalysisConfig accepts run_buckling / run_montecarlo / mc_samples
# --------------------------------------------------------------------------- #


def test_issue_7_analysis_config_accepts_buckling_and_mc_kwargs():
    """Issue #7: constructing AnalysisConfig with the kwargs the CLI
    forwards must not raise ``TypeError``."""
    cfg = AnalysisConfig(
        run_buckling=True,
        run_montecarlo=True,
        mc_samples=50,
    )
    assert cfg.run_buckling is True
    assert cfg.run_montecarlo is True
    assert cfg.mc_samples == 50


# --------------------------------------------------------------------------- #
# Issue #8: WrinkleAnalysis.run(analytical_only=True) skips the FE solve
# --------------------------------------------------------------------------- #


def test_issue_8_run_analytical_only_returns_result_without_fe():
    """Issue #8: ``run(analytical_only=True)`` must return an
    :class:`AnalysisResults` without touching the mesh / FE solver."""
    cfg = AnalysisConfig()
    result = WrinkleAnalysis(cfg).run(analytical_only=True)

    assert isinstance(result, AnalysisResults)
    # The FE path is skipped, so the mesh and field_results attributes
    # must not be populated.
    assert result.mesh is None
    assert result.field_results is None
    # But the analytical predictions must still be present.
    assert result.analytical_knockdown is not None
    assert result.analytical_strength_MPa is not None


# --------------------------------------------------------------------------- #
# Issue #9: compare_morphologies + parametric_sweep accept analytical_only
# --------------------------------------------------------------------------- #


def test_issue_9_compare_morphologies_accepts_analytical_only():
    """Issue #9: ``compare_morphologies(..., analytical_only=True)`` must
    not raise ``TypeError`` and must propagate the flag (skip FE)."""
    cfg = AnalysisConfig()
    all_results = WrinkleAnalysis.compare_morphologies(
        cfg,
        morphologies=("stack", "convex", "concave"),
        analytical_only=True,
    )
    assert set(all_results) == {"stack", "convex", "concave"}
    for r in all_results.values():
        assert isinstance(r, AnalysisResults)
        # FE was skipped.
        assert r.mesh is None
        assert r.field_results is None


def test_issue_9_parametric_sweep_accepts_analytical_only():
    """Issue #9: ``parametric_sweep(..., analytical_only=True)`` must not
    raise ``TypeError`` and must propagate the flag (skip FE)."""
    cfg = AnalysisConfig()
    values = np.linspace(0.1, 0.4, 3)
    results = WrinkleAnalysis.parametric_sweep(
        cfg,
        parameter="amplitude",
        values=values,
        analytical_only=True,
    )
    assert len(results) == 3
    for r in results:
        assert isinstance(r, AnalysisResults)
        # FE was skipped.
        assert r.mesh is None
        assert r.field_results is None


# --------------------------------------------------------------------------- #
# End-to-end: the CLI entry point itself does not crash
# --------------------------------------------------------------------------- #


def test_issue_7_8_9_cli_analyze_end_to_end(capsys):
    """End-to-end: ``wrinklefe analyze --analytical-only`` must exit
    cleanly (return None, no SystemExit) and print a summary."""
    from wrinklefe.cli import main as cli_main

    # ``--analytical-only`` keeps the test cheap (no FE solve).
    cli_main(["analyze", "--analytical-only"])
    out = capsys.readouterr().out
    assert out  # something was printed (the summary)


def test_issue_9_cli_compare_end_to_end(capsys):
    """End-to-end: ``wrinklefe compare`` (default analytical_only=True)
    must exit cleanly."""
    from wrinklefe.cli import main as cli_main

    cli_main(["compare"])
    out = capsys.readouterr().out
    assert "Morphology Comparison" in out


def test_issue_9_cli_sweep_end_to_end(capsys):
    """End-to-end: ``wrinklefe sweep`` (default analytical_only=True)
    must exit cleanly."""
    from wrinklefe.cli import main as cli_main

    cli_main([
        "sweep",
        "--parameter", "amplitude",
        "--min", "0.1",
        "--max", "0.4",
        "--steps", "3",
    ])
    out = capsys.readouterr().out
    assert "Parametric Sweep" in out
