"""Analytical stiffness (axial-modulus) knockdown on the analytical path.

Covers ``AnalysisResults.analytical_modulus_knockdown`` and the underlying
``_profile_modulus_knockdown`` / ``_is_unidirectional`` helpers: the
closed-form CLT series-average estimate populated for unidirectional
laminates (the shipped analytical path otherwise reports no stiffness
knockdown).
"""
from __future__ import annotations

import pytest

from wrinklefe.analysis import (
    AnalysisConfig,
    WrinkleAnalysis,
    _is_unidirectional,
    _profile_modulus_knockdown,
)
from wrinklefe.core.material import MaterialLibrary

ML = MaterialLibrary()


def _ud_config(amplitude=0.75, wavelength=12.9, loading="compression",
               angles=None, morphology="graded"):
    return AnalysisConfig(
        amplitude=amplitude, wavelength=wavelength, width=wavelength / 2.0,
        morphology=morphology, loading=loading,
        material=ML.get("AC318_S6C10_vacbag"),
        angles=angles if angles is not None else [0.0] * 14,
        ply_thickness=0.44,
        domain_length=max(3.0 * wavelength, 10.0), domain_width=10.0,
    )


def _kd(cfg):
    return WrinkleAnalysis(cfg).run(analytical_only=True).analytical_modulus_knockdown


# --------------------------------------------------------------------------
# Helper unit tests
# --------------------------------------------------------------------------

@pytest.mark.parametrize("angles, expected", [
    ([0.0] * 8, True),
    ([0.0, 180.0, -180.0], True),
    ([0.0, 2.0, -3.0], True),       # within 5-degree tolerance
    ([0.0, 45.0, -45.0, 90.0], False),
    ([90.0] * 4, False),
    ([], False),
])
def test_is_unidirectional(angles, expected):
    assert _is_unidirectional(angles) is expected


def test_profile_modulus_unity_for_flat():
    """Zero amplitude (no wrinkle) -> no stiffness loss."""
    kd = _profile_modulus_knockdown(
        amplitude=0.0, wavelength=12.9, width=6.45, domain_length=40.0,
        ply_thickness=0.44, n_plies=14,
        E1=50_800.0, E2=12_000.0, G12=5_500.0, nu12=0.28,
    )
    assert kd == pytest.approx(1.0, abs=1e-9)


def test_profile_modulus_in_unit_interval():
    kd = _profile_modulus_knockdown(
        amplitude=0.75, wavelength=12.9, width=6.45, domain_length=40.0,
        ply_thickness=0.44, n_plies=14,
        E1=50_800.0, E2=12_000.0, G12=5_500.0, nu12=0.28,
    )
    assert 0.0 < kd < 1.0


# --------------------------------------------------------------------------
# Pipeline integration
# --------------------------------------------------------------------------

def test_ud_wrinkle_populates_modulus_knockdown():
    """A UD graded wrinkle yields a sub-unity analytical modulus knockdown
    (the shipped analytical path used to leave this at 1.0)."""
    kd = _kd(_ud_config())
    assert 0.80 < kd < 0.98


def test_zero_amplitude_is_unity():
    assert _kd(_ud_config(amplitude=0.0)) == pytest.approx(1.0, abs=1e-9)


def test_multidirectional_is_populated():
    """Issue #327: a multidirectional layup now gets a sub-unity analytical
    modulus knockdown (it used to be left at 1.0)."""
    kd = _kd(_ud_config(angles=[0.0, 45.0, -45.0, 90.0] * 2))
    assert 0.0 < kd < 1.0


def test_multidirectional_knocks_down_less_than_ud():
    """Off-axis plies carry little axial load and are insensitive to axial
    fibre misalignment, so the same wrinkle knocks a quasi-isotropic /
    cross-ply laminate down LESS than a pure-UD one (issue #327)."""
    kd_ud = _kd(_ud_config(angles=[0.0] * 14))
    kd_qi = _kd(_ud_config(angles=[0.0, 45.0, -45.0, 90.0] * 2))
    kd_cross = _kd(_ud_config(angles=[0.0, 90.0] * 4))
    assert kd_ud < kd_qi < 1.0
    assert kd_ud < kd_cross < 1.0


def test_multidirectional_zero_amplitude_is_unity():
    """A degenerate (zero-amplitude) wrinkle leaves a multidirectional
    laminate exactly unaffected — the CLT reference cancels."""
    kd = _kd(_ud_config(angles=[0.0, 45.0, -45.0, 90.0] * 2, amplitude=0.0))
    assert kd == pytest.approx(1.0, abs=1e-12)


def test_multidirectional_reduces_to_ud_for_all_zero():
    """The generalized path is only taken for non-UD layups; an explicit UD
    layup still routes through the scalar fast path and matches it exactly
    (the [0]_n reduction guarantee, issue #327)."""
    # A 4-ply [0/45/-45/90] knocks down; the same geometry with every ply
    # forced to 0 deg must equal the scalar UD result bit-for-bit.
    kd_ud_pipeline = _kd(_ud_config(angles=[0.0] * 4))
    kd_scalar = _profile_modulus_knockdown(
        amplitude=0.75, wavelength=12.9, width=6.45,
        domain_length=max(3.0 * 12.9, 10.0),
        ply_thickness=0.44, n_plies=4,
        E1=ML.get("AC318_S6C10_vacbag").E1,
        E2=ML.get("AC318_S6C10_vacbag").E2,
        G12=ML.get("AC318_S6C10_vacbag").G12,
        nu12=ML.get("AC318_S6C10_vacbag").nu12,
        through_thickness_decay=True,
        decay_scale=max(12.9 / 2.0, 0.75),
    )
    assert kd_ud_pipeline == pytest.approx(kd_scalar, rel=1e-12)


def test_loading_independent():
    """Linear-elastic stiffness knockdown is the same in tension and
    compression."""
    kd_c = _kd(_ud_config(loading="compression"))
    kd_t = _kd(_ud_config(loading="tension"))
    assert kd_c == pytest.approx(kd_t, rel=1e-12)


def test_decreases_with_amplitude():
    kds = [_kd(_ud_config(amplitude=a)) for a in (0.25, 0.5, 0.75, 1.0)]
    assert all(b < a for a, b in zip(kds, kds[1:])), kds
    assert kds[0] < 1.0


def test_exported_in_json_payloads():
    from wrinklefe.io.export import analysis_results_to_dict
    from wrinklefe.io.results import results_to_dict

    result = WrinkleAnalysis(_ud_config()).run(analytical_only=True)
    rd = results_to_dict(result)
    assert "analytical_modulus_knockdown" in rd["analytical"]
    assert rd["analytical"]["analytical_modulus_knockdown"] < 1.0
    assert (rd["knockdown_factors"]["analytical_modulus"]
            == rd["analytical"]["analytical_modulus_knockdown"])

    ed = analysis_results_to_dict(result)
    assert ed["analytical_predictions"]["analytical_modulus_knockdown"] < 1.0


def test_summary_reports_modulus_knockdown():
    result = WrinkleAnalysis(_ud_config()).run(analytical_only=True)
    assert "Modulus knockdown" in result.summary()
