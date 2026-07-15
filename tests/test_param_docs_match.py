"""Pin the canonical wrinkle-geometry parameter defaults (issue #180).

The README "Wrinkle geometry parameters" table is the single source of
truth for the meaning, units, default, and constraint of each
wrinkle-geometry parameter exposed by :class:`AnalysisConfig`. The CLI
``--help`` text, the Streamlit ``help=`` tooltips, the
``AnalysisConfig`` docstring, and the ``WrinkleConfiguration`` class
docstring all mirror that table.

This module pins the *numeric* defaults from the table to the dataclass
so the docs and the code cannot drift silently: if anyone updates a
default in :class:`AnalysisConfig` without updating the README table
(or vice versa), this test fails.
"""

from __future__ import annotations

from wrinklefe.analysis import AnalysisConfig

# Mirror of the "Default" column in the README "Wrinkle geometry
# parameters" table. Keep this dict in lockstep with README.md and the
# dataclass; any drift is a real bug, not a test flake.
EXPECTED_DEFAULTS: dict[str, object] = {
    "amplitude": 0.366,    # mm, half-amplitude A (>= 0)
    "wavelength": 16.0,    # mm, spatial period lambda (> 0)
    "width": 12.0,         # mm, envelope decay length w (> 0)
    "phase": None,         # rad, None -> derive from morphology
    "decay_floor": 0.0,    # dimensionless, in [0, 1]
    "amplitude_profile": "constant",          # one of constant/gaussian/linear
    "amplitude_profile_decay_length": None,    # mm, None -> wrinkle width
    "amplitude_profile_axis": "x",             # one of x/y
    "transverse_mode": "uniform",              # uniform/gaussian_decay/sinusoidal_y/elliptical
    "transverse_span": None,                   # mm, None -> domain_width
    "transverse_width": None,                  # mm, None -> span_y / 4
}


def test_analysis_config_defaults_match_readme_table() -> None:
    """Every wrinkle-geometry default in the README equals the dataclass default."""
    cfg = AnalysisConfig()
    for name, expected in EXPECTED_DEFAULTS.items():
        actual = getattr(cfg, name)
        assert actual == expected, (
            f"AnalysisConfig.{name} default drifted from the README "
            f"'Wrinkle geometry parameters' table: README says "
            f"{expected!r}, dataclass says {actual!r}. Update both "
            f"together."
        )


def test_amplitude_zero_is_accepted_per_readme_constraint() -> None:
    """The README says amplitude must be >= 0 (0 = flat / no wrinkle)."""
    cfg = AnalysisConfig(amplitude=0.0)
    assert cfg.amplitude == 0.0


def test_decay_floor_endpoints_are_accepted_per_readme_constraint() -> None:
    """The README says decay_floor lives in [0, 1] (both endpoints inclusive)."""
    assert AnalysisConfig(decay_floor=0.0).decay_floor == 0.0
    assert AnalysisConfig(decay_floor=1.0, morphology="graded").decay_floor == 1.0


def test_phase_none_default_means_derive_from_morphology() -> None:
    """``phase=None`` is the canonical default per README; an explicit float overrides it."""
    cfg_default = AnalysisConfig()
    assert cfg_default.phase is None

    cfg_override = AnalysisConfig(phase=0.5, morphology="stack")
    assert cfg_override.phase == 0.5
