"""Verify every declared version agrees with pyproject.toml.

``[project].version`` in pyproject.toml is the single source of truth.
Two other files restate it and can drift out of sync silently:

* ``wrinklefe.__version__`` — the runtime value users report in bug
  reports (issue #21 shipped 0.1.0 while pyproject said 1.0.0);
* ``CITATION.cff`` ``version:`` — what the GitHub "Cite this repository"
  button and Zenodo hand to anyone citing the software (issue #284).

Both are locked here so a version bump that forgets one fails CI rather
than reaching a release. The release workflow adds the third leg: it
checks the git tag against the built wheel's metadata.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

import wrinklefe

_REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = _REPO_ROOT / "pyproject.toml"
CITATION = _REPO_ROOT / "CITATION.cff"


def _read_pyproject_version() -> str:
    """Return the [project].version string declared in pyproject.toml.

    Uses the stdlib ``tomllib`` on Python 3.11+ and falls back to the
    third-party ``tomli`` package on older interpreters.
    """
    if sys.version_info >= (3, 11):
        import tomllib  # type: ignore[import-not-found]
    else:  # pragma: no cover - exercised only on <3.11 runners
        try:
            import tomli as tomllib  # type: ignore[import-not-found]
        except ImportError:
            pytest.skip("tomllib/tomli unavailable on this interpreter")

    with PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    return data["project"]["version"]


def test_version_matches_pyproject() -> None:
    """``wrinklefe.__version__`` must match the version in pyproject.toml."""
    expected = _read_pyproject_version()
    assert wrinklefe.__version__ == expected, (
        f"wrinklefe.__version__={wrinklefe.__version__!r} but "
        f"pyproject.toml declares {expected!r}"
    )


def _read_citation_version() -> str:
    """Return the top-level ``version:`` declared in CITATION.cff.

    Parsed with a regex rather than a YAML library on purpose: PyYAML is
    not a dependency of this project (nor of its ``dev`` extra), and a
    one-field lookup does not justify adding one to the test environment.
    CITATION.cff's grammar makes this safe — ``version`` is a top-level
    scalar key, so anchoring the pattern at column 0 cannot match a
    nested ``version`` under ``references:``.
    """
    text = CITATION.read_text(encoding="utf-8")
    match = re.search(r"^version:\s*[\"']?([^\"'\s]+)[\"']?\s*$", text, re.MULTILINE)
    assert match is not None, "CITATION.cff has no top-level 'version:' key"
    return match.group(1)


def test_citation_cff_version_matches_pyproject() -> None:
    """``CITATION.cff`` must declare the same version as pyproject.toml.

    A stale CITATION.cff means the GitHub citation widget — and, once
    Zenodo archiving is on, the archived record — advertises a version
    that was never released (issue #284). Bump both together; the
    release procedure in CONTRIBUTING.md lists them side by side.
    """
    expected = _read_pyproject_version()
    found = _read_citation_version()
    assert found == expected, (
        f"CITATION.cff declares version {found!r} but pyproject.toml "
        f"declares {expected!r} — bump both together."
    )


def test_version_is_canonical_one_zero_zero() -> None:
    """Sanity check the canonical published version is 1.0.0."""
    assert _read_pyproject_version() == "1.0.0"
