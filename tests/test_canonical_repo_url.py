"""Every link to this repository uses one canonical URL (issue #284).

The repository moved from ``ranipdx-glitch`` to ``elhajjar1``, and its name
has been written both ``wrinkleFE`` and ``wrinklefe``. Old links still
resolve through GitHub's redirect, so nothing breaks visibly when a stale
one is copied forward. That is what makes the drift worth a test: citations
split across two addresses, a deployment guide can point at a repository
that no longer exists under that name, and the next copy-paste propagates
the stale form. PR #395 unified seven sites by hand; three more survived in
places it didn't search. This test checks every tracked text file, so the
next one fails CI instead of waiting for someone to notice.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

CANONICAL = "github.com/elhajjar1/wrinkleFE"

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Any owner, any casing of the repository name: the two ways this has
# drifted. ``wrinkleFE.git`` (a clone URL) is the same repository.
_REPO_URL = re.compile(r"github\.com/[A-Za-z0-9_.-]+/wrinklefe", re.IGNORECASE)

# Binary and generated files are not where a person copies a link from.
_SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".pdf", ".npz",
    ".wfr", ".vtk", ".inp", ".pkl", ".zip", ".gz", ".whl",
}


def _tracked_text_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=_REPO_ROOT, capture_output=True,
            text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout (e.g. an sdist); nothing to scan")
    return [
        _REPO_ROOT / name for name in out.splitlines()
        if Path(name).suffix.lower() not in _SKIP_SUFFIXES
    ]


def test_every_repository_link_is_canonical():
    offenders = []
    for path in _tracked_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for match in _REPO_URL.finditer(line):
                if match.group(0) != CANONICAL:
                    rel = path.relative_to(_REPO_ROOT)
                    offenders.append(f"{rel}:{lineno}: {match.group(0)}")
    assert not offenders, (
        f"links to this repository must use {CANONICAL!r}:\n  "
        + "\n  ".join(offenders)
    )


def test_the_check_catches_both_kinds_of_drift():
    """Guard the guard: the pattern must see an old owner and a re-cased
    name, or the test above passes on nothing."""
    for stale in (
        "https://github.com/ranipdx-glitch/wrinkleFE/issues/6",
        "https://github.com/elhajjar1/wrinklefe",
        "https://github.com/elhajjar1/WrinkleFE",
    ):
        match = _REPO_URL.search(stale)
        assert match and match.group(0) != CANONICAL, stale
    assert _REPO_URL.search(f"https://{CANONICAL}").group(0) == CANONICAL
