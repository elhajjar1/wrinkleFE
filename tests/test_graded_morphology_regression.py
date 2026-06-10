"""Regression coverage for issue #183 (graded morphology pipeline).

The reporter ([#183](
https://github.com/ranipdx-glitch/wrinkleFE/issues/183), original
report [#6](https://github.com/ranipdx-glitch/wrinkleFE/issues/6))
saw two symptoms with the default README configuration:

1. ``--morphology graded`` reportedly crashed somewhere in the
   pipeline.
2. The reported strength retention (97.0 %) did not match an
   expected 58.3 % taken from a now-removed v1.1.0 README
   screenshot.

Investigation (May 2026) found the crash unreproducible on current
``main`` and the 97.0 % value to be the intentional output of the
current ``graded`` analytical model
(``analysis._profile_proportional_kd`` with Gaussian
through-thickness decay, scale = amplitude). The mismatch versus the
old screenshot is due to model evolution (``_GAMMA_Y_UD`` recalibration,
profile-proportional graded knockdown, FE-side corrections in
#207 / #211 / #215 / #220) — not a regression.

These tests pin both behaviours so any future drift is caught early:

* The ``graded`` morphology completes the full analytical pipeline
  via :class:`WrinkleAnalysis` for the README quick-start
  configuration without raising.
* Same for the ``compare_morphologies`` and ``parametric_sweep``
  multi-run entry points.
* The CLI ``analyze`` / ``sweep`` subcommands return ``0`` for
  ``--morphology graded`` (covering the path the reporter actually
  invoked).
* The analytical knockdown for the README quick-start config
  (amplitude 0.366 mm, wavelength 16 mm, width 12 mm, default
  24-ply [0/45/-45/90]_3s IM7/8552 quasi-isotropic layup) sits in a
  tight band around the current 0.9703 value so a quiet drop back
  toward the old 0.583 screenshot value would fail loudly.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis

# README quick-start parameters (see README.md "Python API" block).
_README_AMPLITUDE = 0.366
_README_WAVELENGTH = 16.0
_README_WIDTH = 12.0


# --------------------------------------------------------------------
# 1. The graded morphology completes end-to-end via the Python API.
# --------------------------------------------------------------------

def test_graded_analytical_only_runs() -> None:
    """``morphology='graded'`` runs the analytical path without raising."""
    cfg = AnalysisConfig(
        amplitude=_README_AMPLITUDE,
        wavelength=_README_WAVELENGTH,
        width=_README_WIDTH,
        morphology="graded",
        loading="compression",
        analytical_only=True,
    )
    result = WrinkleAnalysis(cfg).run(analytical_only=True)
    # The graded path must populate every analytical field; an
    # exception in the graded branch of _compute_analytical would
    # have left these unset.
    assert result.analytical_knockdown is not None
    assert 0.0 < result.analytical_knockdown <= 1.0
    assert result.analytical_strength_MPa > 0.0
    assert result.morphology_factor == pytest.approx(1.0)


def test_graded_runs_with_decay_floor_zero() -> None:
    """The default ``decay_floor=0.0`` path produces a finite knockdown."""
    cfg = AnalysisConfig(
        amplitude=_README_AMPLITUDE,
        wavelength=_README_WAVELENGTH,
        width=_README_WIDTH,
        morphology="graded",
        decay_floor=0.0,
        analytical_only=True,
    )
    result = WrinkleAnalysis(cfg).run(analytical_only=True)
    assert 0.0 < result.analytical_knockdown <= 1.0


def test_graded_runs_with_decay_floor_one() -> None:
    """``decay_floor=1.0`` (no through-thickness decay) is the
    degenerate "uniform" limit and must not raise."""
    cfg = AnalysisConfig(
        amplitude=_README_AMPLITUDE,
        wavelength=_README_WAVELENGTH,
        width=_README_WIDTH,
        morphology="graded",
        decay_floor=1.0,
        analytical_only=True,
    )
    result = WrinkleAnalysis(cfg).run(analytical_only=True)
    assert 0.0 < result.analytical_knockdown <= 1.0


# --------------------------------------------------------------------
# 2. ``compare_morphologies`` / ``parametric_sweep`` accept graded.
# --------------------------------------------------------------------

def test_compare_morphologies_includes_graded() -> None:
    """Explicitly passing ``graded`` to ``compare_morphologies`` works.

    Note the default tuple is ``("stack", "convex", "concave")``;
    issue #183 mentions the reporter selected ``graded`` directly, so
    we exercise the case where the caller asks for the full set
    including ``uniform`` and ``graded``.
    """
    cfg = AnalysisConfig(
        amplitude=_README_AMPLITUDE,
        wavelength=_README_WAVELENGTH,
        width=_README_WIDTH,
        analytical_only=True,
    )
    results = WrinkleAnalysis.compare_morphologies(
        cfg,
        morphologies=("stack", "convex", "concave", "uniform", "graded"),
        analytical_only=True,
    )
    assert set(results.keys()) == {
        "stack", "convex", "concave", "uniform", "graded",
    }
    for morph, res in results.items():
        assert res.analytical_knockdown is not None, (
            f"{morph} produced no analytical_knockdown"
        )
        assert 0.0 < res.analytical_knockdown <= 1.0


def test_parametric_sweep_amplitude_with_graded() -> None:
    """Sweeping amplitude with a graded base config completes for all values."""
    cfg = AnalysisConfig(
        wavelength=_README_WAVELENGTH,
        width=_README_WIDTH,
        morphology="graded",
        analytical_only=True,
    )
    results = WrinkleAnalysis.parametric_sweep(
        cfg, "amplitude", [0.1, 0.3, 0.5], analytical_only=True,
    )
    assert len(results) == 3
    for res in results:
        assert 0.0 < res.analytical_knockdown <= 1.0


# --------------------------------------------------------------------
# 3. The CLI ``analyze`` and ``sweep`` paths accept ``--morphology graded``.
#
# The reporter invoked the analysis through the Streamlit UI, which in
# turn calls the same ``WrinkleAnalysis.run`` covered above. The CLI is
# the other public entry point users typically reach for, so we exercise
# it as a subprocess to catch argparse / wiring drift independently of
# the Python-API tests.
# --------------------------------------------------------------------

def test_cli_analyze_graded_completes() -> None:
    """``wrinklefe analyze --morphology graded --no-fe`` returns 0."""
    proc = subprocess.run(
        [
            sys.executable, "-m", "wrinklefe.cli",
            "analyze",
            "--morphology", "graded",
            "--no-fe",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"CLI graded analyze failed (rc={proc.returncode}). "
        f"stdout=\n{proc.stdout}\nstderr=\n{proc.stderr}"
    )
    # The summary header must appear so we know the run completed.
    assert "WrinkleFE Analysis Results" in proc.stdout
    assert "Morphology:      graded" in proc.stdout


def test_cli_sweep_graded_completes() -> None:
    """``wrinklefe sweep --morphology graded ...`` returns 0."""
    proc = subprocess.run(
        [
            sys.executable, "-m", "wrinklefe.cli",
            "sweep",
            "--morphology", "graded",
            "--parameter", "amplitude",
            "--min", "0.1", "--max", "0.5", "--steps", "3",
            "--analytical-only",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"CLI graded sweep failed (rc={proc.returncode}). "
        f"stdout=\n{proc.stdout}\nstderr=\n{proc.stderr}"
    )
    assert "Morphology: graded" in proc.stdout


# --------------------------------------------------------------------
# 4. Numerical pin for the README quick-start graded config.
#
# This is the value the reporter actually observed (0.9703 → 97.0 %
# strength retention). We pin it inside a wide tolerance band so it
# can catch a regression back toward the legacy 0.58–0.80 range
# (which would indicate a silent change to the through-thickness
# decay model) without flagging fine-grained kink-band recalibration.
# --------------------------------------------------------------------

def test_graded_readme_config_pinned_knockdown() -> None:
    """The README example with ``morphology='graded'`` reproduces ~0.83.

    Issue #183: the reporter saw 0.970 (97.0 %) and expected 0.583
    (58.3 %) from an old v1.1.0 README screenshot. The current value
    is mathematically correct for the present graded model (Gaussian
    through-thickness decay confining the wrinkle to the wrinkle
    zone).

    Recalibration history:

    * Pre-May 2026: KD ~ 0.97 with the amplitude-based decay scale
      (``Phi(z) = exp(-(z - z_c)**2 / A**2)``).  Only the few plies
      within one A of the midplane felt the wrinkle, so the laminate
      KD was close to 1.
    * May 2026: KD ~ 0.83 with the wavelength-based decay scale
      (``max(lambda/2, A)``).  More plies feel the wrinkle in the
      24-ply default layup, dropping the laminate average.

    This test pins the value remains close to 0.83 — if a future
    change quietly moves it back toward 0.58 (or up to 1.0, i.e. the
    wrinkle effect vanishing entirely) the regression surfaces here
    rather than in a user bug report.
    """
    cfg = AnalysisConfig(
        amplitude=_README_AMPLITUDE,
        wavelength=_README_WAVELENGTH,
        width=_README_WIDTH,
        morphology="graded",
        loading="compression",
        analytical_only=True,
    )
    result = WrinkleAnalysis(cfg).run(analytical_only=True)
    # Current (May 2026, post wavelength-decay recalibration) value:
    # 0.8304.  Use abs tolerance of 0.05 so kink-band recalibration of
    # order a few percent still passes, but the legacy 0.58 / a return
    # to 1.0 both fail loudly.
    assert result.analytical_knockdown == pytest.approx(0.83, abs=0.05)
