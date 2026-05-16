"""Layup notation parsing shared across the WrinkleFE UIs and CLI.

This module is the single source of truth for turning a human-written
layup string into a flat list of ply angles (degrees). Both the
Streamlit app (``app.py``) and the command-line interface
(``wrinklefe.cli``) import :func:`parse_layup` from here so the two
front-ends can never drift apart.

Two notations are accepted:

* **Contracted** -- ``[0/45/-45/90]_3s`` (sublaminate in brackets,
  optional repeat count, trailing ``s`` for symmetry). ``+/-`` (written
  ``±``) and ``_n`` ply-level modifiers are also supported, e.g.
  ``[0/±45/90_2]s``.
* **Explicit list** -- comma-, semicolon-, or newline-separated angles,
  e.g. ``0, 45, -45, 90, ...``.
"""

from __future__ import annotations

import re
from typing import List

__all__ = ["parse_layup"]


_PLY_TOKEN_RE = re.compile(
    r"""\s*
        (?P<sign>±)?\s*
        (?P<angle>[+-]?\d+(?:\.\d+)?)\s*
        (?:_(?P<rep>\d+))?\s*
    """,
    re.VERBOSE,
)


def _expand_ply_token(token: str) -> List[float]:
    """Expand a single ply entry like ``0``, ``45_2``, ``±45``, or ``±30_2``."""
    token = token.strip()
    if not token:
        return []
    m = _PLY_TOKEN_RE.fullmatch(token)
    if not m:
        raise ValueError(f"Could not parse ply token: {token!r}")
    angle = float(m.group("angle"))
    rep = int(m.group("rep")) if m.group("rep") else 1
    if m.group("sign") == "±":
        return [angle, -angle] * rep
    return [angle] * rep


def _parse_contracted_layup(s: str) -> List[float]:
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
    plies: List[float] = []
    for tok in m.group("inner").split("/"):
        plies.extend(_expand_ply_token(tok))
    if not plies:
        raise ValueError("Contracted layup contains no plies.")
    repeat = int(m.group("rep")) if m.group("rep") else 1
    plies = plies * repeat
    if m.group("sym"):
        plies = plies + plies[::-1]
    return plies


def parse_layup(s: str) -> List[float]:
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
    out: List[float] = []
    for tok in s.replace(";", ",").replace("\n", ",").split(","):
        out.extend(_expand_ply_token(tok))
    if not out:
        raise ValueError("Layup is empty.")
    return out
