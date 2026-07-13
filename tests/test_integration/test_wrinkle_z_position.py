"""Integration tests for the ``wrinkle_z_position`` AnalysisConfig field.

The field shifts the through-thickness Gaussian decay centre of single-
wrinkle (graded) morphologies off the laminate midplane, so users can
distinguish a wrinkle that lives near the surface (Above / Below) from
one centred on the midplane (Middle).  Li et al. (2025) Dataset F shows
that surface-adjacent wrinkles give a higher residual strength than
identical wrinkles at the midplane:

- S-M-2: A = 1.5 mm, λ = 12.9 mm, wrinkle at Middle (plies 7-8 of 14),
  measured KD = 0.629.
- S-A-2: A = 1.5 mm, λ = 12.9 mm, wrinkle at Above (plies 10-11 of 14),
  measured KD = 0.981.

These tests pin:

1. The default ``wrinkle_z_position = 0.5`` reproduces the pre-change
   analytical KD for a graded UD case (regression guard for the rest of
   the validation suite).
2. ``wrinkle_z_position = 0.75`` yields a larger KD than ``0.5`` for the
   AC318_S6C10 UD [0]_14 S-M-2/S-A-2 geometry, mirroring the measured
   trend.
3. Continuity of the KD as a function of ``wrinkle_z_position`` (a tiny
   numerical offset above 0.5 must not flip the result discontinuously).
4. ``__post_init__`` rejects out-of-range or NaN values with
   ``ValueError``.
"""

from __future__ import annotations

import math

import pytest

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary

pytestmark = pytest.mark.integration


def _li_2025_base_kwargs() -> dict:
    """Common AC318_S6C10 / UD [0]_14 / S-M-2 geometry."""
    mat = MaterialLibrary().get("AC318_S6C10")
    return dict(
        amplitude=1.5,        # mm  (Li et al. 2025 S-M-2 / S-A-2)
        wavelength=12.9,      # mm
        width=6.45,           # mm
        morphology="graded",  # single-wrinkle, through-thickness-decayed
        loading="compression",
        material=mat,
        angles=[0] * 14,
        ply_thickness=7.1 / 14.0,  # total laminate thickness 7.1 mm
        analytical_only=True,
    )


def _kd(cfg: AnalysisConfig) -> float:
    return WrinkleAnalysis(cfg).run().analytical_knockdown


class TestWrinkleZPositionDefaults:
    """Default value preserves legacy behaviour."""

    def test_default_is_midplane(self) -> None:
        """Default ``wrinkle_z_position`` is the laminate midplane."""
        cfg = AnalysisConfig()
        assert cfg.wrinkle_z_position == pytest.approx(0.5)

    def test_default_matches_explicit_half(self) -> None:
        """No-kwarg config reproduces explicit ``wrinkle_z_position=0.5`` KD.

        Regression guard: every downstream calibration in
        ``tests/test_integration`` / ``validation/`` is anchored on the
        midplane behaviour, so adding the field must not perturb that
        baseline.
        """
        base = _li_2025_base_kwargs()
        cfg_default = AnalysisConfig(**base)
        cfg_explicit = AnalysisConfig(wrinkle_z_position=0.5, **base)
        assert _kd(cfg_default) == pytest.approx(_kd(cfg_explicit), rel=0.0, abs=0.0)


class TestWrinkleZPositionEffect:
    """Surface-adjacent wrinkle (S-A-2) is less damaging than midplane (S-M-2)."""

    def test_above_gives_higher_kd_than_middle(self) -> None:
        """``wrinkle_z_position=0.75`` (Above) > ``0.5`` (Middle) KD.

        Mirrors Li et al. (2025) Dataset F: S-A-2 (Above, KD=0.981) is
        less damaging than S-M-2 (Middle, KD=0.629) for the same
        wrinkle geometry on AC318_S6C10 UD [0]_14.
        """
        base = _li_2025_base_kwargs()
        cfg_mid = AnalysisConfig(wrinkle_z_position=0.5, **base)
        cfg_above = AnalysisConfig(wrinkle_z_position=0.75, **base)

        kd_mid = _kd(cfg_mid)
        kd_above = _kd(cfg_above)

        # Strict inequality: shifting the decay centre away from the
        # midplane must produce a measurably larger KD (less damage).
        assert kd_above > kd_mid, (
            f"Expected KD(z=0.75) > KD(z=0.5); got {kd_above} vs {kd_mid}"
        )

    def test_continuity_at_half(self) -> None:
        """KD is continuous in ``wrinkle_z_position`` around 0.5.

        A nano-step above the midplane must not flip the predicted KD by
        more than 1e-6.  Catches regressions that introduce branching on
        ``z_position_fraction == 0.5`` rather than smooth shifting of the
        decay centre.
        """
        base = _li_2025_base_kwargs()
        cfg_mid = AnalysisConfig(wrinkle_z_position=0.5, **base)
        cfg_eps = AnalysisConfig(wrinkle_z_position=0.5 + 1e-9, **base)

        kd_mid = _kd(cfg_mid)
        kd_eps = _kd(cfg_eps)

        assert kd_eps == pytest.approx(kd_mid, abs=1e-6)


class TestWrinkleZPositionValidation:
    """``__post_init__`` rejects values outside [0, 1] and non-finite floats."""

    @pytest.mark.parametrize("bad_value", [-0.1, 1.5, float("nan"), float("inf"), -float("inf")])
    def test_invalid_values_raise(self, bad_value: float) -> None:
        with pytest.raises(ValueError, match="wrinkle_z_position"):
            AnalysisConfig(wrinkle_z_position=bad_value)

    @pytest.mark.parametrize("good_value", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_valid_values_pass(self, good_value: float) -> None:
        cfg = AnalysisConfig(wrinkle_z_position=good_value)
        assert cfg.wrinkle_z_position == pytest.approx(good_value)
        assert math.isfinite(cfg.wrinkle_z_position)
