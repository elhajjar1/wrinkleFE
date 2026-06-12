"""Structural guard for CHANGELOG.md (issue #282).

Verifies the Keep-a-Changelog skeleton stays intact. It deliberately
does NOT assert that ``[Unreleased]`` is non-empty — that check belongs
to the release flow (issue #264), where an empty Unreleased at tag time
is the error; an empty Unreleased right after a release is normal.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_CHANGELOG = Path(__file__).resolve().parents[1] / "CHANGELOG.md"


@pytest.fixture(scope="module")
def text() -> str:
    return _CHANGELOG.read_text(encoding="utf-8")


def test_changelog_exists():
    assert _CHANGELOG.is_file()


def test_has_unreleased_section(text):
    assert re.search(r"^## \[Unreleased\]", text, re.MULTILINE)


def test_has_a_released_version_section(text):
    # At least one semver version heading, e.g. "## [1.0.0]".
    assert re.search(r"^## \[\d+\.\d+\.\d+\]", text, re.MULTILINE)


def test_link_references_resolve(text):
    # Every "[x.y.z]" / "[Unreleased]" heading has a matching link ref
    # definition at the bottom (Keep-a-Changelog convention).
    headings = set(re.findall(r"^## \[([^\]]+)\]", text, re.MULTILINE))
    link_defs = set(re.findall(r"^\[([^\]]+)\]:\s+http", text, re.MULTILINE))
    missing = headings - link_defs
    assert not missing, f"changelog headings without link refs: {missing}"


def test_numerical_results_category_documented(text):
    # The project-specific category must be explained in the preamble.
    assert "Numerical results" in text
