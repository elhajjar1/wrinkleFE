"""Regression tests for AnalysisConfig.__post_init__ input validation.

Verifies the fix for issues #39 / #29: ``AnalysisConfig.__post_init__``
previously performed zero validation, so physically invalid inputs
(negative amplitude, non-positive wavelength/width, unknown morphology,
etc.) were silently accepted and surfaced only as obscure tracebacks
deep in the solver/mesh path.  Construction must now fail fast with a
clear :class:`ValueError` naming the field and the offending value,
while every valid configuration (including boundary values) still
constructs unchanged.
"""

from __future__ import annotations

import pytest

from wrinklefe.analysis import AnalysisConfig
from wrinklefe.core.material import MaterialLibrary

# ----------------------------------------------------------------------
# Valid configurations must still construct unchanged
# ----------------------------------------------------------------------

def test_default_config_constructs():
    """A default AnalysisConfig must construct without error."""
    cfg = AnalysisConfig()
    # __post_init__ side effects preserved (valid-input behaviour unchanged).
    assert cfg.domain_length == pytest.approx(3.0 * cfg.wavelength)
    assert cfg.material is not None
    assert cfg.angles is not None and len(cfg.angles) == 24


def test_fully_specified_valid_config_constructs():
    """A realistic fully-specified valid config must construct."""
    cfg = AnalysisConfig(
        amplitude=0.366,
        wavelength=16.0,
        width=12.0,
        morphology="concave",
        loading="tension",
        material=MaterialLibrary().get("IM7_8552"),
        angles=[0, 90, 0, 90, 0, 90, 90, 0, 90, 0, 90, 0],
        interface_1=5,
        interface_2=6,
        nx=4,
        ny=2,
        nz_per_ply=1,
        domain_width=10.0,
        applied_strain=-0.005,
    )
    assert cfg.morphology == "concave"


def test_nonpositive_domain_length_is_auto_derived_not_rejected():
    """Existing behaviour preserved: domain_length <= 0 is a sentinel
    meaning "auto = 3 * wavelength", not an invalid value."""
    cfg = AnalysisConfig(domain_length=-5.0, wavelength=10.0)
    assert cfg.domain_length == pytest.approx(30.0)
    cfg2 = AnalysisConfig(domain_length=0.0, wavelength=8.0)
    assert cfg2.domain_length == pytest.approx(24.0)


@pytest.mark.parametrize("morphology", ["stack", "convex", "concave",
                                        "uniform", "graded"])
def test_all_named_morphologies_accepted(morphology):
    """Every canonical morphology name must be accepted."""
    assert AnalysisConfig(morphology=morphology).morphology == morphology


@pytest.mark.parametrize("morphology", ["STACK", " Convex ", "Concave"])
def test_morphology_case_and_whitespace_tolerated(morphology):
    """run() lower/strip's the name, so validation must too."""
    assert AnalysisConfig(morphology=morphology) is not None


@pytest.mark.parametrize("loading", ["compression", "tension"])
def test_valid_loadings_accepted(loading):
    assert AnalysisConfig(loading=loading).loading == loading


def test_boundary_valid_values_accepted():
    """Smallest/edge values that are still valid must be accepted."""
    # amplitude == 0 is the legitimate "no wrinkle" (flat) case.
    AnalysisConfig(amplitude=0.0)
    # Strictly-positive fields at a small positive value.
    AnalysisConfig(wavelength=1e-6, width=1e-6, domain_width=1e-6,
                   ply_thickness=1e-6)
    # decay_floor inclusive bounds.
    AnalysisConfig(decay_floor=0.0)
    AnalysisConfig(decay_floor=1.0)
    # Smallest valid structural integers / counts.
    AnalysisConfig(nx=1, ny=1, nz_per_ply=1)
    # interface indices at the inclusive lower / exclusive upper bound.
    AnalysisConfig(angles=[0, 90, 0, 90], interface_1=0, interface_2=3)
    # explicit numeric phase must not require a dual-wrinkle name and
    # must coexist with a single-wrinkle morphology.
    AnalysisConfig(phase=0.7, morphology="graded")


# ----------------------------------------------------------------------
# Invalid configurations must raise ValueError naming field + value
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"amplitude": -0.1}, r"amplitude must be >= 0.*got -0\.1"),
        ({"wavelength": 0.0}, r"wavelength must be > 0.*got 0\.0"),
        ({"wavelength": -16.0}, r"wavelength must be > 0.*got -16\.0"),
        ({"width": 0.0}, r"width must be > 0.*got 0\.0"),
        ({"width": -1.0}, r"width must be > 0.*got -1\.0"),
        ({"domain_width": 0.0}, r"domain_width must be > 0.*got 0\.0"),
        ({"ply_thickness": 0.0}, r"ply_thickness must be > 0.*got 0\.0"),
        ({"ply_thickness": -0.183}, r"ply_thickness must be > 0"),
        ({"morphology": "spiral"},
         r"morphology must be one of.*got 'spiral'"),
        ({"morphology": "unknown"}, r"morphology must be one of"),
        ({"morphology": 5}, r"morphology must be one of"),
        ({"loading": "shear"}, r"loading must be one of.*got 'shear'"),
        ({"loading": None}, r"loading must be one of"),
        ({"decay_floor": -0.1}, r"decay_floor must be in \[0, 1\]"),
        ({"decay_floor": 1.5}, r"decay_floor must be in \[0, 1\]"),
        ({"applied_strain": float("nan")},
         r"applied_strain must be finite"),
        ({"applied_strain": float("inf")},
         r"applied_strain must be finite"),
        ({"phase": float("inf")}, r"phase must be finite"),
        ({"angles": [0, 90, 0, 90], "interface_1": 4},
         r"interface_1 must be in \[0, 4\).*got 4"),
        ({"angles": [0, 90, 0, 90], "interface_1": -1},
         r"interface_1 must be in \[0, 4\)"),
        ({"angles": [0, 90, 0, 90], "interface_1": 1, "interface_2": 10},
         r"interface_2 must be in \[0, 4\)"),
        ({"nx": 0}, r"nx must be >= 1.*got 0"),
        ({"ny": 0}, r"ny must be >= 1.*got 0"),
        ({"nz_per_ply": -2}, r"nz_per_ply must be >= 1.*got -2"),
        ({"angles": [900.0]},
         r"AnalysisConfig\.angles\[0\] = .*900.*\[-90, 90\]"),
        ({"angles": [0.0, 452.0, 0.0]},
         r"AnalysisConfig\.angles\[1\] = .*452.*\[-90, 90\]"),
        ({"angles": [0.0, 90.0, -95.0, 0.0]},
         r"AnalysisConfig\.angles\[2\] = .*-95.*\[-90, 90\]"),
    ],
)
def test_invalid_config_raises_value_error(kwargs, match):
    with pytest.raises(ValueError, match=match):
        AnalysisConfig(**kwargs)


def test_canonical_angles_accepted():
    """Boundary and decimal canonical angles construct fine."""
    cfg = AnalysisConfig(angles=[90.0, -90.0, 0.0, 45.5], analytical_only=True)
    assert cfg.angles == [90.0, -90.0, 0.0, 45.5]


def test_explicit_phase_does_not_require_dual_wrinkle_name():
    """A numeric ``phase`` must not force ``morphology`` to be a known
    dual-wrinkle name, but the name must still be a valid morphology
    (run() always consumes it via from_morphology_name)."""
    AnalysisConfig(phase=1.0, morphology="stack")
    AnalysisConfig(phase=1.0, morphology="uniform")
    with pytest.raises(ValueError, match=r"morphology must be one of"):
        AnalysisConfig(phase=1.0, morphology="bogus")


# ----------------------------------------------------------------------
# Auto-derived interface indices (issues #154 / #156)
# ----------------------------------------------------------------------

def test_default_interfaces_24_ply_unchanged():
    """Default 24-ply layup must still resolve to (11, 12).

    Backwards-compat guard: the auto-derivation formula was chosen so
    that the previous hard-coded ``interface_1=11, interface_2=12``
    defaults are preserved exactly for the canonical 24-ply layup.
    """
    cfg = AnalysisConfig(angles=[0] * 24)
    assert cfg.interface_1 == 11
    assert cfg.interface_2 == 12


def test_default_interfaces_default_layup_unchanged():
    """Default (None) layup also resolves to (11, 12) — same 24-ply
    canonical quasi-isotropic stack as before."""
    cfg = AnalysisConfig()
    assert cfg.interface_1 == 11
    assert cfg.interface_2 == 12


@pytest.mark.parametrize(
    "n_plies, expected_i1, expected_i2",
    [
        (2, 0, 1),
        (4, 1, 2),
        (6, 2, 3),
        (8, 3, 4),
        (24, 11, 12),
    ],
)
def test_default_interfaces_small_layups(n_plies, expected_i1, expected_i2):
    """Small layups (< 13 plies) used to crash on the old hard-coded
    defaults (11, 12); auto-derivation must now produce in-range
    interior interfaces for every realistic layup size."""
    cfg = AnalysisConfig(angles=[0] * n_plies)
    assert cfg.interface_1 == expected_i1
    assert cfg.interface_2 == expected_i2
    # And they must satisfy the same validator that used to reject them.
    assert 0 <= cfg.interface_1 < n_plies
    assert 0 <= cfg.interface_2 < n_plies


def test_explicit_interfaces_still_validated():
    """Passing an explicit out-of-range interface still raises
    ValueError — the validator's behaviour for explicit values is
    unchanged by the auto-derivation."""
    with pytest.raises(ValueError, match=r"interface_1 must be in \[0, 8\)"):
        AnalysisConfig(angles=[0] * 8, interface_1=20)
    with pytest.raises(ValueError, match=r"interface_2 must be in \[0, 8\)"):
        AnalysisConfig(angles=[0] * 8, interface_2=8)


def test_explicit_interfaces_preserved_when_in_range():
    """An explicit, in-range interface is preserved verbatim."""
    cfg = AnalysisConfig(angles=[0] * 8, interface_1=2, interface_2=5)
    assert cfg.interface_1 == 2
    assert cfg.interface_2 == 5


# ----------------------------------------------------------------------
# Amplitude profile (issue #182 — surfacing #178 in the CLI/Streamlit)
# ----------------------------------------------------------------------


def test_amplitude_profile_defaults_match_wrinkle_configuration():
    """``AnalysisConfig`` defaults for the spatially varying amplitude
    profile must match the underlying
    :class:`~wrinklefe.core.morphology.WrinkleConfiguration` defaults so
    the CLI / Streamlit surfaces (#182) cannot drift from the engine."""
    import inspect

    from wrinklefe.core.morphology import WrinkleConfiguration

    cfg = AnalysisConfig()
    sig = inspect.signature(WrinkleConfiguration.__init__)
    for name in (
        "amplitude_profile",
        "amplitude_profile_decay_length",
        "amplitude_profile_axis",
    ):
        assert getattr(cfg, name) == sig.parameters[name].default, (
            f"AnalysisConfig.{name} default drifted from "
            f"WrinkleConfiguration.{name}: dataclass says "
            f"{getattr(cfg, name)!r}, WrinkleConfiguration says "
            f"{sig.parameters[name].default!r}"
        )


@pytest.mark.parametrize("profile", ["constant", "gaussian", "linear"])
def test_amplitude_profile_accepts_every_valid_name(profile):
    assert (
        AnalysisConfig(amplitude_profile=profile).amplitude_profile == profile
    )


@pytest.mark.parametrize("axis", ["x", "y"])
def test_amplitude_profile_axis_accepts_both_axes(axis):
    assert (
        AnalysisConfig(amplitude_profile_axis=axis).amplitude_profile_axis
        == axis
    )


def test_amplitude_profile_decay_length_accepts_positive_and_none():
    AnalysisConfig(amplitude_profile_decay_length=None)
    AnalysisConfig(amplitude_profile_decay_length=4.0)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"amplitude_profile": "bogus"},
         r"amplitude_profile must be one of"),
        ({"amplitude_profile": 5},
         r"amplitude_profile must be one of"),
        ({"amplitude_profile_axis": "z"},
         r"amplitude_profile_axis must be one of"),
        ({"amplitude_profile_decay_length": 0.0},
         r"amplitude_profile_decay_length must be a finite positive float"),
        ({"amplitude_profile_decay_length": -1.0},
         r"amplitude_profile_decay_length must be a finite positive float"),
        ({"amplitude_profile_decay_length": float("inf")},
         r"amplitude_profile_decay_length must be a finite positive float"),
    ],
)
def test_invalid_amplitude_profile_raises_value_error(kwargs, match):
    with pytest.raises(ValueError, match=match):
        AnalysisConfig(**kwargs)


def test_amplitude_profile_threaded_into_wrinkle_configuration():
    """The fields must reach the underlying ``WrinkleConfiguration`` so the
    CLI/Streamlit values actually affect the analysis."""
    from wrinklefe.analysis import WrinkleAnalysis

    cfg = AnalysisConfig(
        amplitude_profile="gaussian",
        amplitude_profile_decay_length=5.0,
        amplitude_profile_axis="y",
        analytical_only=True,
    )
    result = WrinkleAnalysis(cfg).run(analytical_only=True)
    wc = result.wrinkle_config
    assert wc.amplitude_profile == "gaussian"
    assert wc.amplitude_profile_decay_length == 5.0
    assert wc.amplitude_profile_axis == "y"
