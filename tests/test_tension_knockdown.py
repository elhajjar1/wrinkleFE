"""The analytical tension knockdown on steep wrinkles (issue #454).

Dataset M (Elhajjar & Shams 2016) showed the three-mechanism tension
model collapsing on steep wrinkles (pre-floor KD 0.01-0.44 against a
measured 0.40-0.78) with the compression floor setting every
prediction, unsafely. These tests pin the three corrections that stop
the collapse at its source and the removal of the floor:

1. a graded ply's curvature decays with the same factor as its angle;
2. each interlaminar traction is bounded by its interface strength;
3. the in-situ shear strength is never below the lamina S12;
4. the tension KD no longer depends on the compression strain budget.
"""

from __future__ import annotations

import math

import pytest

import wrinklefe.analysis as analysis
from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary

_ML = MaterialLibrary()
_MD = [0, 45, 90, -45, 0, 45, -45, 0]
_MD = _MD + _MD[::-1]


def _cfg(**overrides) -> AnalysisConfig:
    """Dataset M-MD-1-like coupon: steep (30 deg) graded wrinkle."""
    kw = dict(
        amplitude=0.8, wavelength=8.7062, width=4.6, morphology="graded",
        loading="tension", material=_ML.get("T700GC_2510_AGATE"),
        angles=_MD, ply_thickness=0.152, analytical_only=True,
    )
    kw.update(overrides)
    return AnalysisConfig(**kw)


def _kd0(theta: float, cfg: AnalysisConfig, scale: float = 1.0) -> float:
    kd, _ = WrinkleAnalysis._tension_knockdown_analytical(
        theta, cfg, _return_kd0_only=True, amplitude_scale=scale)
    return kd


def test_a_flat_graded_ply_carries_no_crest_curvature():
    """A ply the grading leaves flat (B_p = 0: zero angle AND zero
    amplitude) keeps its full strength. Before #454 it was charged the
    crest curvature and its KD_oop (0.175 here)."""
    cfg = _cfg()
    assert _kd0(0.0, cfg, scale=0.0) == 1.0
    # The peak ply is unchanged by the scale argument's default.
    theta = math.atan(2 * math.pi * cfg.amplitude / cfg.wavelength)
    assert _kd0(theta, cfg) == _kd0(theta, cfg, scale=1.0)


def test_ply_curvature_scales_the_interlaminar_stress():
    """At a fixed angle, halving the ply's amplitude can only raise (or
    keep) its knockdown: less curvature, less interlaminar stress."""
    cfg = _cfg(amplitude=0.1, wavelength=16.0, width=12.0)
    theta = math.radians(1.0)
    assert _kd0(theta, cfg, scale=0.5) >= _kd0(theta, cfg, scale=1.0)


@pytest.mark.parametrize("amplitude", [0.05, 0.3, 0.8, 2.0])
def test_interlaminar_traction_is_bounded_by_the_interface_strength(
        amplitude):
    """KD_oop >= 1/sqrt(2) however steep the wrinkle: past the interface
    strength the interface delaminates and sheds the traction."""
    cfg = _cfg(amplitude=amplitude, morphology="stack")
    theta = math.atan(2 * math.pi * amplitude / cfg.wavelength)
    _, mech = WrinkleAnalysis._tension_knockdown_analytical(theta, cfg)
    assert mech["kd_oop"] >= 1.0 / math.sqrt(2.0) - 1e-12


def test_below_the_interface_strength_kd_oop_is_unchanged():
    """Dataset B, A = 0.1215 mm: sigma33 is below Yt, KD_oop governs and
    keeps its pre-#454 value (laminate KD 0.8065 against 0.77 measured)."""
    angles = [0, 45, 90, -45, 0, 45, -45, 0]
    cfg = AnalysisConfig(
        amplitude=0.1215, wavelength=8.2, width=0.75 * 8.2,
        morphology="uniform", loading="tension",
        material=_ML.get("T700_2510"), angles=angles + angles[::-1],
        ply_thickness=0.152, analytical_only=True,
    )
    res = WrinkleAnalysis(cfg).run(analytical_only=True)
    assert res.tension_mechanisms["kd_oop"] == pytest.approx(0.726, abs=5e-4)
    assert res.tension_mechanisms["mode"].startswith("OOP")
    assert res.analytical_knockdown == pytest.approx(0.8065, abs=5e-4)


def test_in_situ_shear_is_never_below_the_lamina_strength():
    """A [0]16 block is far past the thin-ply range: the thin-ply
    formula alone gives S12_is = 66.5 MPa, below the lamina's own
    154.4 MPa. The matrix KD now uses S12, the lower bound."""
    mat = _ML.get("T700GC_2510_AGATE")
    cfg = _cfg(angles=[0] * 16, morphology="stack")
    theta = math.radians(25.0)
    _, mech = WrinkleAnalysis._tension_knockdown_analytical(theta, cfg)
    s, c = math.sin(theta), math.cos(theta)
    yt_is = 1.12 * math.sqrt(2.0) * mat.Yt
    expected = 1.0 / math.sqrt((s * s / yt_is) ** 2
                               + (s * c / mat.S12) ** 2) / mat.Xt
    assert mech["kd_matrix"] == pytest.approx(expected, rel=1e-12)


def test_tension_does_not_depend_on_the_compression_strain_budget(
        monkeypatch):
    """The compression floor is gone, so recalibrating the kink-band
    gamma_Y (e.g. issue #451) cannot move a tension prediction."""
    cfgs = [_cfg(), _cfg(angles=[0] * 16), _cfg(morphology="stack")]
    before = [WrinkleAnalysis(c).run(analytical_only=True).analytical_knockdown
              for c in cfgs]
    monkeypatch.setattr(analysis, "_effective_gamma_Y", lambda angles: 0.5)
    after = [WrinkleAnalysis(c).run(analytical_only=True).analytical_knockdown
             for c in cfgs]
    assert after == before


def test_steep_graded_wrinkle_no_longer_collapses():
    """The pre-floor tension KD of Dataset M-UD-10 was 0.010 (the floor
    then held it at 0.539, measured 0.544). It is now the tension
    model's own value, 0.42."""
    cfg = _cfg(amplitude=0.28, wavelength=3.4, width=1.7, angles=[0] * 16)
    kd = WrinkleAnalysis(cfg).run(analytical_only=True).analytical_knockdown
    assert kd == pytest.approx(0.4205, abs=5e-4)
