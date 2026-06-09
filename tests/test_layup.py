"""Tests for the shared layup parser :mod:`wrinklefe.core.layup`.

This parser is the single source of truth for both the Streamlit app
(``app.py``) and the CLI (``wrinklefe.cli``); see issue #83. These tests
cover the parser directly and assert that ``app.py`` imports it rather
than defining its own copy.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from wrinklefe.core.layup import parse_layup, to_contracted_layup


# --------------------------------------------------------------------------- #
# Contracted notation
# --------------------------------------------------------------------------- #


def test_contracted_quasi_iso_3s_expands_to_24():
    quarter = [0.0, 45.0, -45.0, 90.0]
    expected = quarter * 3
    expected = expected + expected[::-1]
    assert parse_layup("[0/45/-45/90]_3s") == expected
    assert len(parse_layup("[0/45/-45/90]_3s")) == 24


def test_contracted_symmetry_only():
    assert parse_layup("[0/90]s") == [0.0, 90.0, 90.0, 0.0]


def test_contracted_plus_minus_token():
    assert parse_layup("[0/±45/90]s") == [
        0.0, 45.0, -45.0, 90.0, 90.0, -45.0, 45.0, 0.0
    ]


def test_contracted_ply_level_repeat():
    assert parse_layup("[0_2/90]") == [0.0, 0.0, 90.0]


def test_contracted_repeat_without_symmetry():
    assert parse_layup("[0/90]_2") == [0.0, 90.0, 0.0, 90.0]


# --------------------------------------------------------------------------- #
# Explicit list notation
# --------------------------------------------------------------------------- #


def test_explicit_comma_list():
    assert parse_layup("0, 45, -45, 90") == [0.0, 45.0, -45.0, 90.0]


def test_explicit_semicolon_and_newline_separators():
    assert parse_layup("0; 45\n-45;90") == [0.0, 45.0, -45.0, 90.0]


def test_explicit_plus_minus_token():
    assert parse_layup("0, ±45, 90") == [0.0, 45.0, -45.0, 90.0]


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", ["", "   ", "[0/45/", "[]", "[abc]"])
def test_malformed_layup_raises_value_error(bad):
    with pytest.raises(ValueError):
        parse_layup(bad)


# --------------------------------------------------------------------------- #
# Contracted rendering (to_contracted_layup)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("angles", "expected"),
    [
        ([0.0, 90.0] * 4 + [90.0, 0.0] * 4, "[0/90]_4s"),
        (
            ([0.0, 45.0, -45.0, 90.0] * 3)
            + ([0.0, 45.0, -45.0, 90.0] * 3)[::-1],
            "[0/±45/90]_3s",
        ),
        (
            [45.0, -45.0, 45.0, -45.0, -45.0, 45.0, -45.0, 45.0],
            "[±45]_2s",
        ),
        ([0.0] * 8, "[0]_8"),
        ([0.0, 45.0, 90.0], "[0/45/90]"),
        ([0.0, 90.0, 90.0, 0.0], "[0/90]s"),
        ([22.5, -22.5], "[±22.5]"),
    ],
)
def test_contracted_rendering(angles, expected):
    assert to_contracted_layup(angles) == expected


@pytest.mark.parametrize(
    "angles",
    [
        [0.0, 90.0] * 4 + [90.0, 0.0] * 4,
        ([0.0, 45.0, -45.0, 90.0] * 3) + ([0.0, 45.0, -45.0, 90.0] * 3)[::-1],
        [45.0, -45.0, 45.0, -45.0, -45.0, 45.0, -45.0, 45.0],
        [0.0] * 8,
        [0.0, 45.0, 90.0],
        [0.0],
        [-45.0, 45.0],
        [30.0, -30.0, 60.0],
        [0.0, 12.345, -12.345, 90.0],
    ],
)
def test_contracted_round_trip(angles):
    """parse_layup(to_contracted_layup(a)) == a for any angle list."""
    assert parse_layup(to_contracted_layup(angles)) == angles


def test_contracted_asymmetric_falls_back_to_expanded():
    """No misleading shorthand: asymmetric stacks render in full."""
    angles = [0.0, 45.0, 90.0, -45.0, 0.0]
    out = to_contracted_layup(angles)
    assert out == "[0/45/90/-45/0]"
    assert parse_layup(out) == angles


def test_contracted_empty_raises():
    with pytest.raises(ValueError):
        to_contracted_layup([])


# --------------------------------------------------------------------------- #
# app.py must consume the shared parser (no duplicate definition)
# --------------------------------------------------------------------------- #


def _app_py_path() -> Path:
    return Path(__file__).resolve().parents[1] / "app.py"


def test_app_py_imports_shared_parser():
    """app.py imports parse_layup from the package, not a local copy."""
    src = _app_py_path().read_text(encoding="utf-8")
    tree = ast.parse(src)

    imports_shared = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "wrinklefe.core.layup"
        and any(alias.name == "parse_layup" for alias in node.names)
        for node in ast.walk(tree)
    )
    assert imports_shared, "app.py must import parse_layup from wrinklefe.core.layup"

    # And it must NOT redefine the parser locally anymore.
    local_defs = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    assert "parse_layup" not in local_defs
    assert "_parse_contracted_layup" not in local_defs
    assert "_expand_ply_token" not in local_defs
