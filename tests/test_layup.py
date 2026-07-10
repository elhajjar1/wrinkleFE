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

from wrinklefe.core.layup import (
    parse_layup,
    to_contracted_layup,
    validate_ply_angle,
)

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
# Angle-range validation (issue #308)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", ["[900]", "[452]", "[0/91/90]", "900", "-452"])
def test_out_of_range_angle_raises(bad):
    with pytest.raises(ValueError, match="canonical"):
        parse_layup(bad)


def test_canonical_bounds_are_accepted():
    # ±90 are the same fibre direction and must stay valid.
    assert parse_layup("[90/-90/0]") == [90.0, -90.0, 0.0]
    assert parse_layup("0, 90, -90, 45.5, -45.5") == [
        0.0, 90.0, -90.0, 45.5, -45.5
    ]


def test_subscript_typo_suggests_underscore_syntax():
    with pytest.raises(ValueError, match=r"90_2"):
        parse_layup("[902]")
    with pytest.raises(ValueError, match=r"45_2"):
        parse_layup("452")


# --------------------------------------------------------------------------- #
# Leading-zero token rejection (issue #308)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", ["[02/902]s", "02", "045", "[0/030/90]"])
def test_leading_zero_token_raises(bad):
    with pytest.raises(ValueError, match="leading zero"):
        parse_layup(bad)


def test_leading_zero_suggests_repeat_syntax():
    # The headline #308 case: [02/902]s must fail loudly toward [0_2/90_2]s.
    with pytest.raises(ValueError, match=r"0_2"):
        parse_layup("[02/902]s")


# --------------------------------------------------------------------------- #
# ASCII sign aliases +-, -+ for the Unicode ± / ∓ (issue #308)
# --------------------------------------------------------------------------- #


def test_ascii_plus_minus_matches_unicode():
    assert parse_layup("[0/+-45/90]s") == parse_layup("[0/±45/90]s")
    assert parse_layup("0, +-45, 90") == [0.0, 45.0, -45.0, 90.0]


def test_ascii_minus_plus_is_reversed_order():
    assert parse_layup("[0/-+45/90]s") == [
        0.0, -45.0, 45.0, 90.0, 90.0, 45.0, -45.0, 0.0
    ]
    # -+ is the ASCII spelling of the reversed pair, opposite of +-.
    assert parse_layup("-+30") == [-30.0, 30.0]
    assert parse_layup("+-30") == [30.0, -30.0]


def test_ascii_sign_alias_respects_repeat():
    assert parse_layup("+-45_2") == [45.0, -45.0, 45.0, -45.0]


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


@pytest.mark.parametrize("bad", [100.0, -95.0, 900.0])
def test_contracted_non_canonical_raises(bad):
    """Non-canonical angles raise a renderer error, not a parser error."""
    with pytest.raises(ValueError) as exc:
        to_contracted_layup([0.0, bad, 45.0])
    msg = str(exc.value)
    assert f"{bad:g}" in msg
    assert "[-90, 90]" in msg
    # Must read as a renderer error, not leak contracted-string syntax.
    assert "_n" not in msg
    assert "Did you mean" not in msg


def test_contracted_boundary_angles_round_trip():
    """±90 and 0 are canonical and still render/round-trip."""
    angles = [90.0, -90.0, 0.0]
    out = to_contracted_layup(angles)
    assert parse_layup(out) == angles


def test_validate_ply_angle_accepts_canonical():
    for a in (0.0, 90.0, -90.0, 45.5, -12.345):
        assert validate_ply_angle(a) == a


@pytest.mark.parametrize("bad", [90.1, -90.1, 900.0, -450.0])
def test_validate_ply_angle_rejects_non_canonical(bad):
    with pytest.raises(ValueError) as exc:
        validate_ply_angle(bad)
    assert "[-90, 90]" in str(exc.value)


def test_validate_ply_angle_context_prefix():
    with pytest.raises(ValueError) as exc:
        validate_ply_angle(900.0, context="AnalysisConfig.angles[0] = ")
    assert str(exc.value).startswith("AnalysisConfig.angles[0] = ")


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
