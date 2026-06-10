"""Verify the package __version__ stays in sync with pyproject.toml."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import wrinklefe

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


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


def test_version_is_canonical_one_zero_zero() -> None:
    """Sanity check the canonical published version is 1.0.0."""
    assert _read_pyproject_version() == "1.0.0"
