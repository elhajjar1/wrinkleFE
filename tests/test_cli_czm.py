"""End-to-end tests for the ``wrinklefe analyze`` CZM CLI flags.

These tests invoke the installed ``wrinklefe`` console-script entry
point via :mod:`subprocess` so that:

1. argparse argument parsing, flag mapping into :class:`AnalysisConfig`,
   the analyze handler's printout, and the optional CZM-figure export
   are all exercised together as the user sees them.
2. Regression coverage protects the **absence** of a CZM section in the
   baseline (no ``--enable-czm``) output — the contract is that adding
   the new flags must not perturb existing CLI behaviour.

A tiny ``[0/90]_4s`` mesh (``--nx 4 --ny 2 --czm-load-increments 5``) is
used everywhere CZM runs so each test completes in single-digit seconds.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


_WRINKLEFE_CLI = shutil.which("wrinklefe")


def _run_cli(*args: str, timeout: float = 180.0) -> subprocess.CompletedProcess:
    """Invoke ``wrinklefe <args>`` and return the completed process.

    Prefers the installed console script (``wrinklefe``); falls back to
    ``python -m wrinklefe.cli`` so the suite still passes in a venv that
    has the package importable but not pip-installed with entry points
    registered.
    """
    if _WRINKLEFE_CLI is not None:
        cmd = [_WRINKLEFE_CLI, *args]
    else:
        cmd = [sys.executable, "-m", "wrinklefe.cli", *args]
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
    )


# ----------------------------------------------------------------------
# Baseline: --enable-czm absent must reproduce the existing analyze output
# ----------------------------------------------------------------------


def test_cli_without_czm_unchanged():
    """No ``--enable-czm`` flag: exit code 0 and *no* CZM section in stdout.

    Guards the anti-goal: 'do not change the existing UI flow for users
    who leave CZM off; default behaviour must be bit-identical.'
    """
    proc = _run_cli(
        "analyze",
        "--analytical-only",
        "--amplitude", "0.3",
        "--wavelength", "16",
    )
    assert proc.returncode == 0, (
        f"baseline CLI run failed: stderr={proc.stderr!r}"
    )
    assert "WrinkleFE Analysis Results" in proc.stdout
    # CZM-only headings/keys must not appear when the user did not opt
    # in. ``czm_overview_figure`` is also part of the CZM extras path.
    assert "Cohesive Zone Modeling" not in proc.stdout
    assert "Max damage" not in proc.stdout
    assert "CZM overview figure saved" not in proc.stdout


# ----------------------------------------------------------------------
# Happy path: --enable-czm with a tiny mesh runs end-to-end
# ----------------------------------------------------------------------


def test_cli_with_czm_runs():
    """``--enable-czm`` with a tiny mesh produces a CZM section in stdout."""
    proc = _run_cli(
        "analyze",
        "--enable-czm",
        "--amplitude", "0.3",
        "--wavelength", "16",
        "--nx", "4",
        "--ny", "2",
        "--strain", "0.01",
        "--layup", "[0/90]_4s",
        "--czm-load-increments", "5",
    )
    assert proc.returncode == 0, (
        f"CZM CLI run failed: stderr={proc.stderr!r}, stdout={proc.stdout!r}"
    )
    # ``AnalysisResults.summary()`` already prints the core CZM block;
    # the ``analyze`` handler also appends the *extras* line.
    assert "Cohesive Zone Modeling" in proc.stdout
    assert "Max damage" in proc.stdout
    assert "Elements with damage > 0.5" in proc.stdout


def test_cli_czm_save_figure(tmp_path):
    """``--save-czm-figure`` writes a non-empty image file to disk."""
    out_png = tmp_path / "czm_overview.png"
    proc = _run_cli(
        "analyze",
        "--enable-czm",
        "--amplitude", "0.3",
        "--wavelength", "16",
        "--nx", "4",
        "--ny", "2",
        "--strain", "0.01",
        "--layup", "[0/90]_4s",
        "--czm-load-increments", "5",
        "--save-czm-figure", str(out_png),
    )
    assert proc.returncode == 0, (
        f"--save-czm-figure run failed: stderr={proc.stderr!r}"
    )
    assert out_png.exists(), "CZM overview figure was not created"
    assert out_png.stat().st_size > 1000, (
        f"CZM overview figure suspiciously small: "
        f"{out_png.stat().st_size} bytes"
    )
    assert "CZM overview figure saved" in proc.stdout


# ----------------------------------------------------------------------
# Defensive: the new interface-list parser must reject garbage cleanly
# ----------------------------------------------------------------------


def test_cli_czm_interfaces_rejects_garbage():
    """``--czm-interfaces foo`` is not a sentinel or int list -> exit 2."""
    proc = _run_cli(
        "analyze",
        "--enable-czm",
        "--amplitude", "0.3",
        "--wavelength", "16",
        "--nx", "4",
        "--ny", "2",
        "--strain", "0.01",
        "--layup", "[0/90]_4s",
        "--czm-load-increments", "5",
        "--czm-interfaces", "foo",
    )
    assert proc.returncode != 0, (
        "garbage --czm-interfaces should exit non-zero, got "
        f"stdout={proc.stdout!r}"
    )
    # Either the CLI's own parser error or AnalysisConfig._validate
    # rejects it; both are acceptable, both are clearly flagged.
    assert (
        "czm-interfaces" in proc.stderr.lower()
        or "interfaces" in proc.stderr.lower()
    )
