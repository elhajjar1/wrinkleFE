"""Smoke tests for ``wrinklefe analyze --amplitude-profile ...`` (issue #182).

PR #178 added the spatially varying amplitude profile to
:class:`~wrinklefe.core.morphology.WrinkleConfiguration`. Issue #182
surfaces the same three knobs on the CLI:

- ``--amplitude-profile {constant,gaussian,linear}``
- ``--amplitude-profile-decay-length FLOAT`` (mm)
- ``--amplitude-profile-axis {x,y}``

These tests drive the real CLI through ``subprocess`` (so argparse,
:class:`AnalysisConfig` construction, and the analytical pipeline are all
exercised end-to-end) and assert that:

1. Every profile name returns exit code 0.
2. The chosen profile is echoed back in the summary so a user can
   confirm the flag took effect (acceptance criterion in #182).
"""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize("profile", ["constant", "gaussian", "linear"])
def test_cli_amplitude_profile_smoke(profile):
    """``wrinklefe analyze --amplitude-profile {profile}`` runs cleanly."""
    cmd = [
        sys.executable,
        "-m",
        "wrinklefe.cli",
        "analyze",
        "--amplitude-profile",
        profile,
        "--amplitude-profile-decay-length",
        "8.0",
        "--no-fe",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    assert proc.returncode == 0, (
        f"CLI exited {proc.returncode}; stderr=\n{proc.stderr}"
    )
    # The summary echoes the chosen profile so the user can verify the
    # flag landed (and so this smoke test pins the plumbing).
    assert profile in proc.stdout, (
        f"Profile {profile!r} not echoed in summary; "
        f"stdout=\n{proc.stdout}"
    )


def test_cli_amplitude_profile_axis_y():
    """The ``--amplitude-profile-axis`` flag also threads through cleanly."""
    cmd = [
        sys.executable,
        "-m",
        "wrinklefe.cli",
        "analyze",
        "--amplitude-profile",
        "gaussian",
        "--amplitude-profile-decay-length",
        "8.0",
        "--amplitude-profile-axis",
        "y",
        "--no-fe",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    assert proc.returncode == 0, (
        f"CLI exited {proc.returncode}; stderr=\n{proc.stderr}"
    )
    assert "axis=y" in proc.stdout, (
        f"Profile axis not echoed in summary; stdout=\n{proc.stdout}"
    )


def test_cli_default_is_backwards_compatible():
    """Omitting the new flags must keep the legacy ``constant`` behaviour."""
    cmd = [
        sys.executable,
        "-m",
        "wrinklefe.cli",
        "analyze",
        "--no-fe",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, (
        f"CLI exited {proc.returncode}; stderr=\n{proc.stderr}"
    )
    assert "constant" in proc.stdout
