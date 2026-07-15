"""Through-width (transverse) wrinkle surfaces via ``AnalysisConfig`` (#300).

The ``WrinkleSurface3D`` transverse modes are wired to three new config
fields (``transverse_mode`` / ``transverse_span`` / ``transverse_width``).
These tests pin the contract:

* the default (``"uniform"``) path is bit-identical — it builds the bare,
  x-only ``GaussianSinusoidal`` and never a ``WrinkleSurface3D``;
* a localized mode (``"gaussian_decay"``) gives a *milder* FE knockdown
  than uniform at the same crest amplitude (the headline behaviour);
* invalid modes and the out-of-scope combinations (analytical-only,
  multi-wrinkle, CZM) fail fast at construction with actionable messages;
* the new fields round-trip through ``to_dict`` / ``from_dict`` (#259).

FE-solve tests carry the ``slow`` marker (#267).
"""

from __future__ import annotations

import pytest

from wrinklefe.analysis import (
    AnalysisConfig,
    WrinkleAnalysis,
    WrinkleSpec,
)
from wrinklefe.core.wrinkle import GaussianSinusoidal, WrinkleSurface3D

# --------------------------------------------------------------------------- #
# Defaults / regression
# --------------------------------------------------------------------------- #

def test_default_transverse_fields() -> None:
    """The new fields default to the uniform, auto-resolved contract."""
    cfg = AnalysisConfig()
    assert cfg.transverse_mode == "uniform"
    assert cfg.transverse_span is None
    assert cfg.transverse_width is None


def test_uniform_builds_bare_profile_not_wrapped() -> None:
    """Default uniform mode must build a bare GaussianSinusoidal (#300).

    Wrapping even in "uniform" mode would risk numerical drift; the
    contract is that the default path is left completely untouched.
    """
    cfg = AnalysisConfig()
    result = WrinkleAnalysis(cfg).run(analytical_only=True)
    profile = result.wrinkle_config.wrinkles[0].profile
    assert isinstance(profile, GaussianSinusoidal)
    assert not isinstance(profile, WrinkleSurface3D)


def test_non_uniform_wraps_in_surface3d() -> None:
    """A non-uniform mode wraps the profile in a WrinkleSurface3D."""
    cfg = AnalysisConfig(transverse_mode="gaussian_decay")
    result = WrinkleAnalysis(cfg).run(analytical_only=False)
    profile = result.wrinkle_config.wrinkles[0].profile
    assert isinstance(profile, WrinkleSurface3D)
    assert profile.transverse_mode == "gaussian_decay"
    # span_y tracks the mesh width; width_y defaults to span_y / 4.
    assert profile.span_y == pytest.approx(cfg.domain_width)
    assert profile.width_y == pytest.approx(cfg.domain_width / 4.0)


def test_explicit_span_and_width_forwarded() -> None:
    """Explicit transverse_span / transverse_width reach the surface."""
    cfg = AnalysisConfig(
        transverse_mode="elliptical",
        transverse_span=18.0,
        transverse_width=4.0,
    )
    result = WrinkleAnalysis(cfg).run(analytical_only=False)
    profile = result.wrinkle_config.wrinkles[0].profile
    assert isinstance(profile, WrinkleSurface3D)
    assert profile.span_y == pytest.approx(18.0)
    assert profile.width_y == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# Directional acceptance: localized => milder knockdown
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_gaussian_decay_is_milder_than_uniform() -> None:
    """Same crest amplitude, localized surface => milder FE knockdown (#300).

    The crest fibre angle at the mid-width centreline is identical (both
    surfaces are full-amplitude there), so ``mesh_max_angle_rad`` matches;
    the width-averaged FE knockdown is milder for the localized wrinkle,
    i.e. it retains more stiffness and strength.
    """
    def solve(mode: str):
        cfg = AnalysisConfig(
            amplitude=0.366,
            wavelength=16.0,
            width=12.0,
            morphology="stack",
            loading="compression",
            transverse_mode=mode,
        )
        return WrinkleAnalysis(cfg).run(analytical_only=False)

    uniform = solve("uniform")
    localized = solve("gaussian_decay")

    # Crest angle unchanged (the centreline geometry is identical).
    assert localized.mesh_max_angle_rad == pytest.approx(
        uniform.mesh_max_angle_rad, rel=1e-9
    )

    # Milder stiffness knockdown: higher retention.
    assert localized.modulus_retention_global > uniform.modulus_retention_global

    # Milder strength knockdown: higher weakest-criterion retention.
    u_str = min(uniform.retention_factors.values())
    l_str = min(localized.retention_factors.values())
    assert l_str > u_str


# --------------------------------------------------------------------------- #
# Validation / scope
# --------------------------------------------------------------------------- #

def test_invalid_mode_names_valid_set() -> None:
    with pytest.raises(ValueError) as exc:
        AnalysisConfig(transverse_mode="banana")
    msg = str(exc.value)
    assert "transverse_mode" in msg
    # The valid set is reused from WrinkleSurface3D so the two never drift.
    for mode in WrinkleSurface3D._VALID_MODES:
        assert mode in msg


def test_analytical_only_rejects_transverse_mode() -> None:
    with pytest.raises(ValueError) as exc:
        AnalysisConfig(transverse_mode="gaussian_decay", analytical_only=True)
    msg = str(exc.value)
    assert "analytical_only" in msg
    assert "FE" in msg


def test_run_time_analytical_override_rejected() -> None:
    """A run(analytical_only=True) override is rejected too (not silent)."""
    cfg = AnalysisConfig(transverse_mode="gaussian_decay")  # cfg default FE
    with pytest.raises(ValueError):
        WrinkleAnalysis(cfg).run(analytical_only=True)


def test_multi_wrinkle_combination_rejected() -> None:
    with pytest.raises(NotImplementedError) as exc:
        AnalysisConfig(
            transverse_mode="elliptical",
            wrinkles=[
                WrinkleSpec(
                    amplitude=0.3, wavelength=16.0, width=12.0, ply_interface=11
                )
            ],
        )
    assert "wrinkles" in str(exc.value)


def test_czm_combination_rejected() -> None:
    with pytest.raises(NotImplementedError) as exc:
        AnalysisConfig(transverse_mode="gaussian_decay", enable_czm=True)
    assert "enable_czm" in str(exc.value)


@pytest.mark.parametrize("field", ["transverse_span", "transverse_width"])
@pytest.mark.parametrize("bad", [-1.0, 0.0, float("nan"), float("inf")])
def test_non_positive_span_width_rejected(field: str, bad: float) -> None:
    with pytest.raises(ValueError) as exc:
        AnalysisConfig(transverse_mode="gaussian_decay", **{field: bad})
    assert field in str(exc.value)


def test_uniform_ignores_span_width_overrides() -> None:
    """Uniform mode is unaffected by span/width and never wraps."""
    cfg = AnalysisConfig(
        transverse_mode="uniform", transverse_span=8.0, transverse_width=2.0
    )
    result = WrinkleAnalysis(cfg).run(analytical_only=True)
    assert not isinstance(
        result.wrinkle_config.wrinkles[0].profile, WrinkleSurface3D
    )


# --------------------------------------------------------------------------- #
# Serialisation round-trip (#259)
# --------------------------------------------------------------------------- #

def test_transverse_fields_round_trip() -> None:
    cfg = AnalysisConfig(
        transverse_mode="gaussian_decay",
        transverse_span=18.0,
        transverse_width=4.0,
    )
    payload = cfg.to_dict()
    assert payload["transverse_mode"] == "gaussian_decay"
    assert payload["transverse_span"] == 18.0
    assert payload["transverse_width"] == 4.0

    restored = AnalysisConfig.from_dict(payload)
    assert restored.transverse_mode == "gaussian_decay"
    assert restored.transverse_span == pytest.approx(18.0)
    assert restored.transverse_width == pytest.approx(4.0)


def test_default_transverse_fields_round_trip() -> None:
    cfg = AnalysisConfig()
    restored = AnalysisConfig.from_dict(cfg.to_dict())
    assert restored.transverse_mode == "uniform"
    assert restored.transverse_span is None
    assert restored.transverse_width is None
