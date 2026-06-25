"""Layup notation parsing shared across the WrinkleFE UIs and CLI.

This module is the single source of truth for turning a human-written
layup string into a flat list of ply angles (degrees). Both the
Streamlit app (``app.py``) and the command-line interface
(``wrinklefe.cli``) import :func:`parse_layup` from here so the two
front-ends can never drift apart.

Two notations are accepted:

* **Contracted** -- ``[0/45/-45/90]_3s`` (sublaminate in brackets,
  optional repeat count, trailing ``s`` for symmetry). ``±`` (ASCII
  ``+-``; ``-+`` gives the reversed ``∓`` order) and ``_n`` ply-level
  modifiers are also supported, e.g. ``[0/±45/90_2]s``.
* **Explicit list** -- comma-, semicolon-, or newline-separated angles,
  e.g. ``0, 45, -45, 90, ...``.

Ply angles must be canonical (``|angle| <= 90``). Common ASCII typos for
the ``_n`` repeat syntax -- a leading-zero token like ``02`` or an
out-of-range token like ``902`` -- are rejected with a hint toward the
intended ``0_2`` / ``90_2`` spelling rather than silently parsed.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

__all__ = ["parse_layup", "to_contracted_layup"]


_MAX_PLY_ANGLE = 90.0


_PLY_TOKEN_RE = re.compile(
    r"""\s*
        (?P<sign>±|\+-|-\+)?\s*
        (?P<angle>[+-]?\d+(?:\.\d+)?)\s*
        (?:_(?P<rep>\d+))?\s*
    """,
    re.VERBOSE,
)


def _subscript_hint(angle_str: str) -> str | None:
    """Spelling a bare-digit-subscript typo most likely meant.

    For an out-of-range *integer* angle that reads as a canonical angle
    followed by a bare repeat count (``902`` -> ``90_2``, ``452`` ->
    ``45_2``), return the ``_n`` spelling; otherwise ``None``. A trailing
    ``_1`` is a no-op repeat, so a split is only offered for counts >= 2.
    """
    sign = ""
    mag = angle_str
    if mag[:1] in "+-":
        sign, mag = mag[0], mag[1:]
    if not mag.isdigit():  # decimals are never subscript typos
        return None
    for k in range(len(mag) - 1, 0, -1):
        head, tail = mag[:k], mag[k:]
        if 0 <= int(head) <= _MAX_PLY_ANGLE and int(tail) >= 2:
            return f"{sign}{head}_{tail}"
    return None


def _expand_ply_token(token: str) -> list[float]:
    """Expand a single ply entry like ``0``, ``45_2``, ``±45``, ``+-45``,
    ``-+30``, or ``±30_2``.

    ``±45`` and the ASCII aliases ``+-45`` expand to ``[45, -45]``;
    ``-+45`` (the reversed order, the ASCII spelling of ``∓``) expands to
    ``[-45, 45]``. Angles must be canonical (``|angle| <= 90``); leading
    zeros (``02``) and out-of-range angles (``902``) are rejected with a
    hint toward the ``_n`` repeat syntax they were likely meant to be.
    """
    token = token.strip()
    if not token:
        return []
    m = _PLY_TOKEN_RE.fullmatch(token)
    if not m:
        raise ValueError(f"Could not parse ply token: {token!r}")
    angle_str = m.group("angle")
    rep = int(m.group("rep")) if m.group("rep") else 1

    # Leading-zero integer tokens (02, 045) are never an angle — almost
    # always a bare-digit repeat the user meant to write as `0_n`.
    has_sign = angle_str[:1] in "+-"
    mag = angle_str[1:] if has_sign else angle_str
    if "." not in mag and len(mag) > 1 and mag[0] == "0":
        sign_str = angle_str[0] if has_sign else ""
        rest = mag.lstrip("0") or "0"
        lead_hint = (
            f" Did you mean {sign_str}0_{rest} (use _n for a repeated ply)?"
            if int(rest) >= 2
            else ""
        )
        raise ValueError(
            f"Invalid ply angle {angle_str!r}: leading zeros are not "
            f"allowed.{lead_hint}"
        )

    angle = float(angle_str)
    if abs(angle) > _MAX_PLY_ANGLE:
        hint = _subscript_hint(angle_str)
        suggestion = (
            f" Did you mean {hint} (use _n for a repeated ply)?"
            if hint
            else ""
        )
        raise ValueError(
            f"Ply angle {angle:g}° is outside the canonical fibre-angle "
            f"range [-90, 90].{suggestion}"
        )

    sign = m.group("sign")
    if sign in ("±", "+-"):
        return [angle, -angle] * rep
    if sign == "-+":
        return [-angle, angle] * rep
    return [angle] * rep


def _parse_contracted_layup(s: str) -> list[float]:
    """Parse contracted notation like ``[0/45/-45/90]_3s`` or ``[0/±45/90]s``."""
    m = re.fullmatch(
        r"\s*\[\s*(?P<inner>[^\[\]]*)\]\s*_?\s*(?P<rep>\d+)?\s*(?P<sym>[sS])?\s*",
        s,
    )
    if not m:
        raise ValueError(
            f"Could not parse contracted layup {s!r}. "
            "Expected a form like '[0/45/-45/90]_3s'."
        )
    plies: list[float] = []
    for tok in m.group("inner").split("/"):
        plies.extend(_expand_ply_token(tok))
    if not plies:
        raise ValueError("Contracted layup contains no plies.")
    repeat = int(m.group("rep")) if m.group("rep") else 1
    plies = plies * repeat
    if m.group("sym"):
        plies = plies + plies[::-1]
    return plies


def parse_layup(s: str) -> list[float]:
    """Parse a layup string into a flat list of ply angles (degrees).

    Two notations are accepted:

    * **Contracted** -- ``[0/45/-45/90]_3s`` (sublaminate in brackets,
      optional repeat count, trailing ``s`` for symmetry). ``±`` and
      ``_n`` ply-level modifiers are also supported, e.g.
      ``[0/±45/90_2]s``.
    * **Explicit list** -- comma-, semicolon-, or newline-separated
      angles, e.g. ``0, 45, -45, 90, ...``.
    """
    s = s.strip()
    if not s:
        raise ValueError("Layup is empty.")
    if "[" in s or "]" in s:
        return _parse_contracted_layup(s)
    out: list[float] = []
    for tok in s.replace(";", ",").replace("\n", ",").split(","):
        out.extend(_expand_ply_token(tok))
    if not out:
        raise ValueError("Layup is empty.")
    return out


def _fmt_angle(angle: float) -> str:
    """Format an angle for contracted notation (45.0 -> '45')."""
    return f"{angle:g}"


def _render_bracket(base: Sequence[float]) -> str:
    """Render a sublaminate as ``[a/b/...]``, collapsing ``a, -a`` to ``±a``."""
    tokens: list[str] = []
    i = 0
    while i < len(base):
        a = base[i]
        if a > 0 and i + 1 < len(base) and base[i + 1] == -a:
            tokens.append(f"±{_fmt_angle(a)}")
            i += 2
        else:
            tokens.append(_fmt_angle(a))
            i += 1
    return "[" + "/".join(tokens) + "]"


def _smallest_repeat(seq: list[float]) -> tuple[list[float], int]:
    """Return the shortest base ``b`` and count ``k`` with ``seq == b * k``."""
    n = len(seq)
    for size in range(1, n + 1):
        if n % size == 0 and seq == seq[:size] * (n // size):
            return seq[:size], n // size
    return seq, 1


def to_contracted_layup(angles: Sequence[float]) -> str:
    """Render a flat ply-angle list in contracted layup notation.

    The inverse of :func:`parse_layup` for the notations it accepts:
    symmetry (``s`` suffix), sublaminate repeats (``_k``), and ``±``
    pair collapse. The most compact representation found is returned;
    when no contraction applies the full bracketed angle list is
    returned, so the output is never ambiguous.

    Round-trip property: ``parse_layup(to_contracted_layup(a)) == a``
    for any non-empty list of angles.

    Examples
    --------
    >>> to_contracted_layup([0, 45, -45, 90, 90, -45, 45, 0])
    '[0/±45/90]s'
    >>> to_contracted_layup([0] * 8)
    '[0]_8'
    >>> to_contracted_layup([0, 45, 90])
    '[0/45/90]'
    """
    plies = [float(a) for a in angles]
    if not plies:
        raise ValueError("Layup is empty.")

    candidates: list[str] = []
    n = len(plies)
    if n % 2 == 0 and plies == plies[::-1]:
        half_base, half_rep = _smallest_repeat(plies[: n // 2])
        candidates.append(
            _render_bracket(half_base)
            + (f"_{half_rep}" if half_rep > 1 else "")
            + "s"
        )
    base, rep = _smallest_repeat(plies)
    if rep > 1:
        candidates.append(_render_bracket(base) + f"_{rep}")
    full = _render_bracket(plies)
    candidates.append(full)

    for cand in sorted(candidates, key=len):
        if parse_layup(cand) == plies:
            return cand
    return full
