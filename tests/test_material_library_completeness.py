"""Regression tests pinning the built-in MaterialLibrary roster (refs #88).

Issue #88 requested the addition of common aerospace prepregs (M21, T800/M21,
IM10, S-2 glass, Kevlar) to the built-in material library. By the time this
test was authored, all five had already been added (T800S_M21, IM10_8552,
S2_GLASS_EPOXY, KEVLAR49_EPOXY — M21 is covered by T800S_M21). See
``src/wrinklefe/core/material.py::MaterialLibrary._load_builtins`` for the
data sources.

The tests here pin:

* the exact set of 9 built-in names (so a silent deletion regresses);
* a handful of headline allowables for each #88 material (so a silent edit
  that zeros out or wildly perturbs a property also regresses).

Tolerances are loose (±5%) because the precise values are sourced from
public datasheets and may be re-tuned within that band as references are
refreshed; what we want to catch is *placeholder* / *zero* / *wrong-order-
of-magnitude* regressions, not legitimate datasheet updates.
"""

from __future__ import annotations

import pytest

from wrinklefe.core.material import MaterialLibrary

# Frozen built-in roster. Update only if the library intentionally adds
# or removes a material. The nine fibre/epoxy laminate prepregs are the
# issue #88 roster; ``EPOXY_S6C10`` is the isotropic neat-epoxy card
# added for the resin-pocket zone (Li 2024/2025 UD glass datasets).
EXPECTED_BUILTIN_NAMES = frozenset({
    "AC318_S6C10",
    "AS4_3501_6",
    "IM10_8552",
    "IM7_8552",
    "KEVLAR49_EPOXY",
    "S2_GLASS_EPOXY",
    "T300_914",
    "T700_2510",
    "T800S_M21",
    "IM6G_3501_6",
    "EPOXY_S6C10",
    "AC318_S6C10_vacbag",
})


# Headline allowables for the four #88 prepregs. Values mirror the
# datasheet-cited numbers in MaterialLibrary._load_builtins; a 5% tolerance
# permits legitimate datasheet refreshes while still catching zero /
# wrong-order regressions.
#
# Schema: name -> dict(attr -> expected MPa value)
ISSUE_88_HEADLINE_ALLOWABLES = {
    "T800S_M21": {
        "E1": 157_000.0, "E2": 8_500.0, "G12": 4_200.0,
        "Xt": 2_950.0, "Xc": 1_680.0,
        "Yt": 70.0, "Yc": 290.0, "S12": 98.0,
    },
    "IM10_8552": {
        "E1": 185_000.0, "E2": 9_400.0, "G12": 5_500.0,
        "Xt": 3_310.0, "Xc": 1_690.0,
        "Yt": 63.0, "Yc": 240.0, "S12": 95.0,
    },
    "S2_GLASS_EPOXY": {
        "E1": 52_000.0, "E2": 19_000.0, "G12": 6_700.0,
        "Xt": 1_700.0, "Xc": 970.0,
        "Yt": 49.0, "Yc": 158.0, "S12": 83.0,
    },
    "KEVLAR49_EPOXY": {
        "E1": 76_000.0, "E2": 5_500.0, "G12": 2_300.0,
        "Xt": 1_400.0, "Xc": 235.0,
        "Yt": 12.0, "Yc": 53.0, "S12": 34.0,
    },
}


REL_TOL = 0.05  # 5% — permits datasheet refresh, catches zero/order-of-mag drift


def test_builtin_count_pinned():
    """The library ships exactly the pinned built-in roster (9 prepregs
    plus the EPOXY_S6C10 resin-pocket card)."""
    lib = MaterialLibrary()
    assert len(lib) == len(EXPECTED_BUILTIN_NAMES), (
        f"Built-in count drifted: got {len(lib)} "
        f"({sorted(lib.list_names())}), expected "
        f"{len(EXPECTED_BUILTIN_NAMES)} ({sorted(EXPECTED_BUILTIN_NAMES)})."
    )


def test_builtin_names_pinned():
    """Exact set of built-in names is frozen — guards against silent drops."""
    lib = MaterialLibrary()
    assert set(lib.list_names()) == EXPECTED_BUILTIN_NAMES


@pytest.mark.parametrize("name", sorted(ISSUE_88_HEADLINE_ALLOWABLES))
def test_issue_88_material_allowables(name):
    """Headline elastic and strength allowables match cited datasheet values.

    Loose 5% tolerance — flags zeroed-out placeholders or order-of-magnitude
    typos without forbidding legitimate datasheet refreshes.
    """
    lib = MaterialLibrary()
    mat = lib.get(name)
    expected = ISSUE_88_HEADLINE_ALLOWABLES[name]
    for attr, want in expected.items():
        got = getattr(mat, attr)
        assert got == pytest.approx(want, rel=REL_TOL), (
            f"{name}.{attr}: got {got}, expected {want} (±{REL_TOL:.0%})"
        )
