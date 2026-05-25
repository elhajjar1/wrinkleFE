"""Integration tests for the through-thickness Gaussian decay scale.

The compression graded morphology's profile-proportional knockdown
applies a Gaussian taper across the laminate thickness::

    Phi(z_p) = exp(-(z_p - z_c)**2 / (2 * decay_scale**2))

where ``z_c = wrinkle_z_position * T`` is the decay centre and
``decay_scale`` (mm) is the standard deviation.  This module pins:

1. The default ``through_thickness_decay_scale=None`` resolves to the
   auto formula ``max(wavelength / 2, amplitude)``.
2. Setting ``through_thickness_decay_scale=A`` (in the new convention)
   approximately reproduces the pre-change predicted KD for the
   Mukhopadhyay M-C1 case.
3. For a thick UD coupon, increasing ``decay_scale`` from 0.5 mm to
   3.7 mm (= L/2 for L = 7.4 mm) drops the predicted KD by at least
   0.05 — i.e. more plies feel the wrinkle when the through-thickness
   reach is wider.
4. The Argon-Fleck quadratic coefficient ``kink_band_quadratic_coeff``
   strictly lowers the KD for the same theta when set to 0.5 versus 0.
5. Negative ``kink_band_quadratic_coeff`` or non-positive
   ``through_thickness_decay_scale`` raises ``ValueError``.
"""

from __future__ import annotations

import math

import pytest

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.layup import parse_layup
from wrinklefe.core.material import MaterialLibrary


def _kd(cfg: AnalysisConfig) -> float:
    return float(WrinkleAnalysis(cfg).run(analytical_only=True).analytical_knockdown)


# ----------------------------------------------------------------------
# 1.  Default decay scale uses the auto formula
# ----------------------------------------------------------------------

class TestDecayScaleAutoDefault:
    """``through_thickness_decay_scale=None`` resolves to ``max(L/2, A)``."""

    def test_default_field_is_none(self) -> None:
        """The dataclass default is ``None`` (auto formula)."""
        cfg = AnalysisConfig()
        assert cfg.through_thickness_decay_scale is None

    def test_default_matches_explicit_auto_value(self) -> None:
        """Leaving the field unset reproduces ``max(L/2, A)`` exactly.

        For the Mukhopadhyay M-C1 case (A=0.168 mm, L=10 mm,
        48-ply blocked layup) ``max(L/2, A) = 5.0`` mm.  The
        auto-resolved KD must equal the KD with that explicit value to
        full precision (no spurious branching on the sentinel).
        """
        layup = parse_layup("[45_2/90_2/-45_2/0_2]_3s")
        mat = MaterialLibrary().get("IM7_8552")
        base = dict(
            amplitude=0.168,
            wavelength=10.0,
            width=7.5,
            morphology="graded",
            loading="compression",
            material=mat,
            angles=layup,
            ply_thickness=0.125,
            analytical_only=True,
        )
        cfg_auto = AnalysisConfig(**base)
        cfg_explicit = AnalysisConfig(
            through_thickness_decay_scale=max(10.0 / 2.0, 0.168), **base
        )
        assert _kd(cfg_auto) == pytest.approx(_kd(cfg_explicit), rel=0.0, abs=0.0)


# ----------------------------------------------------------------------
# 2.  decay_scale = A reproduces pre-change behaviour (M-C1)
# ----------------------------------------------------------------------

class TestDecayScaleLegacyBehaviour:
    """``decay_scale=A`` reproduces the pre-change (amplitude-only) KD."""

    def test_mc1_legacy_value_recovered(self) -> None:
        """M-C1 KD with ``decay_scale=A`` matches the pre-change value.

        The pre-change form ``Phi = exp(-(z-z_c)**2 / A**2)`` is a
        Gaussian with standard deviation ``A / sqrt(2)``; the new
        form ``Phi = exp(-(z-z_c)**2 / (2 * sigma**2))`` with
        ``sigma = A`` is wider by ``sqrt(2)``.  For the M-C1 geometry
        the predicted KDs nonetheless agree to within 0.01 because
        only the central plies see appreciable wrinkle activation in
        either case.

        Pre-change KD for M-C1 (legacy ``Phi = exp(-d**2 / A**2)``):
        ``KD ~= 0.982``.  Setting ``decay_scale = A`` in the new
        convention gives ``KD ~= 0.975`` — the spec-allowed ±0.01
        tolerance comfortably covers this offset.
        """
        layup = parse_layup("[45_2/90_2/-45_2/0_2]_3s")
        mat = MaterialLibrary().get("IM7_8552")
        A = 0.168
        cfg_old = AnalysisConfig(
            amplitude=A,
            wavelength=10.0,
            width=7.5,
            morphology="graded",
            loading="compression",
            material=mat,
            angles=layup,
            ply_thickness=0.125,
            through_thickness_decay_scale=A,
            analytical_only=True,
        )
        kd_old = _kd(cfg_old)
        # Pre-change KD for M-C1 (legacy decay) was ~0.982.  Allow
        # ±0.01 absolute tolerance per the spec.
        assert kd_old == pytest.approx(0.982, abs=0.01)


# ----------------------------------------------------------------------
# 3.  Wider decay reaches more plies in a thick UD laminate
# ----------------------------------------------------------------------

class TestDecayScaleReachesMorePlies:
    """Increasing ``decay_scale`` drops KD for a thick UD wrinkled coupon."""

    def test_thick_ud_kd_drops_with_wider_decay(self) -> None:
        """A=1.0, L=7.4, ply=0.2, [0]_32: KD(ds=3.7) <= KD(ds=0.5) − 0.05.

        The thick UD coupon mirrors the Li 2024 / Dataset E geometry.
        With a narrow decay (``ds = 0.5 mm``) only the few plies
        nearest the midplane feel the wrinkle and the laminate-
        averaged BF knockdown stays high.  Widening the decay to
        ``ds = L / 2 = 3.7 mm`` brings most plies into the wrinkle
        zone, dropping the KD by at least 0.05.
        """
        mat = MaterialLibrary().get("AC318_S6C10")
        base = dict(
            amplitude=1.0,
            wavelength=7.4,
            width=3.7,
            morphology="graded",
            loading="compression",
            material=mat,
            angles=[0.0] * 32,
            ply_thickness=0.2,
            analytical_only=True,
        )
        cfg_narrow = AnalysisConfig(through_thickness_decay_scale=0.5, **base)
        cfg_wide = AnalysisConfig(through_thickness_decay_scale=3.7, **base)

        kd_narrow = _kd(cfg_narrow)
        kd_wide = _kd(cfg_wide)

        assert kd_narrow - kd_wide >= 0.05, (
            f"Expected KD(ds=0.5) - KD(ds=3.7) >= 0.05, got "
            f"{kd_narrow:.4f} - {kd_wide:.4f} = {kd_narrow - kd_wide:.4f}"
        )


# ----------------------------------------------------------------------
# 4.  Argon-Fleck quadratic term strictly lowers KD at high theta
# ----------------------------------------------------------------------

class TestArgonFleckEffect:
    """``kink_band_quadratic_coeff > 0`` strictly lowers KD at theta = 30°."""

    def test_quadratic_lowers_kd_at_30deg(self) -> None:
        """KD with ``c_AF = 0.5`` < KD with ``c_AF = 0`` at theta_max = 30°.

        Build A, L so that ``arctan(2*pi*A/L) = 30°`` exactly
        (``A/L = tan(30°)/(2*pi)``) and pin a thick UD coupon so the
        BF response dominates.  The quadratic term ``c_AF * r**2``
        adds an additional positive contribution to the denominator,
        so the KD strictly decreases.
        """
        mat = MaterialLibrary().get("AC318_S6C10")
        A = 0.5
        L = A / (math.tan(math.radians(30.0)) / (2.0 * math.pi))
        assert math.degrees(math.atan(2.0 * math.pi * A / L)) == pytest.approx(30.0, abs=1e-9)

        base = dict(
            amplitude=A,
            wavelength=L,
            width=L / 2.0,
            morphology="graded",
            loading="compression",
            material=mat,
            angles=[0.0] * 32,
            ply_thickness=0.2,
            analytical_only=True,
        )
        cfg_linear = AnalysisConfig(kink_band_quadratic_coeff=0.0, **base)
        cfg_quadratic = AnalysisConfig(kink_band_quadratic_coeff=0.5, **base)

        kd_linear = _kd(cfg_linear)
        kd_quadratic = _kd(cfg_quadratic)

        assert kd_quadratic < kd_linear, (
            f"Expected KD(c_AF=0.5) < KD(c_AF=0); got {kd_quadratic:.4f} "
            f"vs {kd_linear:.4f}"
        )


# ----------------------------------------------------------------------
# 5.  Validation of new fields
# ----------------------------------------------------------------------

class TestValidation:
    """``__post_init__`` rejects invalid values for the new fields."""

    @pytest.mark.parametrize(
        "bad_value", [-1.0, -0.1, float("nan"), float("-inf")]
    )
    def test_negative_or_nan_quadratic_coeff_raises(
        self, bad_value: float
    ) -> None:
        with pytest.raises(ValueError, match="kink_band_quadratic_coeff"):
            AnalysisConfig(kink_band_quadratic_coeff=bad_value)

    @pytest.mark.parametrize(
        "bad_value",
        [-1.0, -0.1, 0.0, float("nan"), float("inf"), float("-inf")],
    )
    def test_negative_or_zero_decay_scale_raises(
        self, bad_value: float
    ) -> None:
        with pytest.raises(ValueError, match="through_thickness_decay_scale"):
            AnalysisConfig(through_thickness_decay_scale=bad_value)

    @pytest.mark.parametrize("good_value", [0.0, 0.1, 0.5, 1.0, 5.0])
    def test_valid_quadratic_coeff_pass(self, good_value: float) -> None:
        cfg = AnalysisConfig(kink_band_quadratic_coeff=good_value)
        assert cfg.kink_band_quadratic_coeff == pytest.approx(good_value)

    @pytest.mark.parametrize("good_value", [0.01, 0.1, 1.0, 5.0])
    def test_valid_decay_scale_pass(self, good_value: float) -> None:
        cfg = AnalysisConfig(through_thickness_decay_scale=good_value)
        assert cfg.through_thickness_decay_scale == pytest.approx(good_value)

    def test_decay_scale_default_is_none(self) -> None:
        """``None`` is the sentinel for the auto formula and must be accepted."""
        cfg = AnalysisConfig(through_thickness_decay_scale=None)
        assert cfg.through_thickness_decay_scale is None

    def test_quadratic_coeff_default_is_zero(self) -> None:
        cfg = AnalysisConfig()
        assert cfg.kink_band_quadratic_coeff == 0.0
