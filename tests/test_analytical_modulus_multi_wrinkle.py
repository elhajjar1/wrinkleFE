"""Multi-wrinkle analytical axial-modulus knockdown (issue #329).

Covers ``AnalysisResults.analytical_modulus_knockdown`` when the config
carries a ``wrinkles=[WrinkleSpec, ...]`` list. The local angle field is
composed from every wrinkle spec ("compose then differentiate", mirroring
the FE :meth:`WrinkleConfiguration.fiber_angles_at_nodes`) and the laminate
membrane stiffness is series-averaged along the wrinkle. Single-wrinkle
results are unchanged; coincident half-amplitude wrinkles reproduce the
full-amplitude wrinkle exactly.
"""
from __future__ import annotations

import numpy as np
import pytest

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis, WrinkleSpec
from wrinklefe.core.material import MaterialLibrary

ML = MaterialLibrary()
_ANGLES_8 = [0.0, 45.0, -45.0, 90.0, 90.0, -45.0, 45.0, 0.0]


def _cfg(wrinkles=None, angles=None, amplitude=0.15, wavelength=16.0,
         width=8.0, morphology="graded", loading="compression", **kw):
    return AnalysisConfig(
        amplitude=amplitude, wavelength=wavelength, width=width,
        morphology=morphology, loading=loading,
        material=ML.get("AC318_S6C10_vacbag"),
        angles=list(angles) if angles is not None else list(_ANGLES_8),
        ply_thickness=0.44, domain_length=48.0, domain_width=10.0,
        wrinkles=wrinkles, **kw,
    )


def _kd(cfg):
    return WrinkleAnalysis(cfg).run(analytical_only=True).analytical_modulus_knockdown


# --------------------------------------------------------------------------
# Population / range
# --------------------------------------------------------------------------

def test_multi_wrinkle_populates_modulus_knockdown():
    """A multi-wrinkle layout yields a sub-unity analytical modulus
    knockdown (the analytical path used to leave this at 1.0)."""
    cfg = _cfg(wrinkles=[
        WrinkleSpec(0.15, 8.0, 2.0, ply_interface=3, phase_offset=-2.0 * np.pi),
        WrinkleSpec(0.15, 8.0, 2.0, ply_interface=3, phase_offset=+2.0 * np.pi),
    ])
    kd = _kd(cfg)
    assert 0.0 < kd < 1.0


def test_multi_wrinkle_ud_layup_also_populated():
    """The multi-wrinkle generalization works for a UD layup too (it does
    not depend on the layup being multidirectional)."""
    cfg = _cfg(angles=[0.0] * 8, wrinkles=[
        WrinkleSpec(0.15, 8.0, 2.0, ply_interface=3, phase_offset=-2.0 * np.pi),
        WrinkleSpec(0.15, 8.0, 2.0, ply_interface=3, phase_offset=+2.0 * np.pi),
    ])
    kd = _kd(cfg)
    assert 0.0 < kd < 1.0


# --------------------------------------------------------------------------
# Reduction guarantees
# --------------------------------------------------------------------------

def test_coincident_halves_equal_full_amplitude():
    """Issue #329 / #252 "compose then differentiate": two coincident
    half-amplitude wrinkles reproduce the single full-amplitude wrinkle
    exactly, because the slope fields add before the angle is taken."""
    spec_kw = dict(wavelength=16.0, width=8.0, ply_interface=3,
                   phase_offset=0.0)
    full = _cfg(wrinkles=[WrinkleSpec(amplitude=0.15, **spec_kw)])
    halves = _cfg(wrinkles=[
        WrinkleSpec(amplitude=0.075, **spec_kw),
        WrinkleSpec(amplitude=0.075, **spec_kw),
    ])
    kd_full = _kd(full)
    kd_halves = _kd(halves)
    assert kd_halves == pytest.approx(kd_full, abs=1e-12)


def test_single_spec_matches_scalar_path():
    """A one-entry wrinkles list reproduces the scalar single-wrinkle
    config it denotes (same placement, geometry, layup)."""
    common = dict(amplitude=0.15, wavelength=16.0, width=8.0, angles=_ANGLES_8)
    scalar = _cfg(**common, interface_1=3)
    spec = _cfg(**common, wrinkles=[
        WrinkleSpec(0.15, 16.0, 8.0, ply_interface=3, phase_offset=0.0)
    ])
    assert _kd(spec) == pytest.approx(_kd(scalar), rel=1e-9)


def test_single_wrinkle_unchanged_by_multi_path():
    """Sanity: the scalar single-wrinkle multidirectional knockdown is
    independent of whether it is expressed as a scalar config or a
    one-entry wrinkles list — the single-wrinkle answer is unchanged."""
    scalar = _cfg(amplitude=0.15, interface_1=3)
    spec = _cfg(amplitude=0.15, wrinkles=[
        WrinkleSpec(0.15, 16.0, 8.0, ply_interface=3, phase_offset=0.0)
    ])
    assert _kd(spec) == pytest.approx(_kd(scalar), rel=1e-9)


# --------------------------------------------------------------------------
# Monotonicity / sanity
# --------------------------------------------------------------------------

def test_two_disjoint_wrinkles_knock_down_more_than_one():
    """Adding a second (disjoint) wrinkle of the same geometry can only
    lower the effective axial modulus."""
    one = _cfg(wrinkles=[
        WrinkleSpec(0.15, 8.0, 2.0, ply_interface=3, phase_offset=-2.0 * np.pi),
    ])
    two = _cfg(wrinkles=[
        WrinkleSpec(0.15, 8.0, 2.0, ply_interface=3, phase_offset=-2.0 * np.pi),
        WrinkleSpec(0.15, 8.0, 2.0, ply_interface=3, phase_offset=+2.0 * np.pi),
    ])
    assert _kd(two) < _kd(one) < 1.0


def test_loading_independent():
    """Linear-elastic stiffness knockdown is loading-independent."""
    wr = [WrinkleSpec(0.15, 8.0, 2.0, ply_interface=3, phase_offset=0.0)]
    kd_c = _kd(_cfg(wrinkles=wr, loading="compression"))
    kd_t = _kd(_cfg(wrinkles=wr, loading="tension"))
    assert kd_c == pytest.approx(kd_t, rel=1e-12)


# --------------------------------------------------------------------------
# FE cross-check (trend, not accuracy — no validation dataset exists)
# --------------------------------------------------------------------------

def test_trend_against_fe_modulus_retention():
    """The analytical modulus knockdown should track the FE
    ``modulus_retention`` for a multi-wrinkle layout: both populated, both
    in (0, 1], and both within a sane band of one another. This is a trend
    sanity check, not an accuracy claim (no multidirectional modulus
    validation dataset exists)."""
    cfg = _cfg(
        wrinkles=[
            WrinkleSpec(0.15, 8.0, 2.0, ply_interface=3,
                        phase_offset=-2.0 * np.pi),
            WrinkleSpec(0.15, 8.0, 2.0, ply_interface=3,
                        phase_offset=+2.0 * np.pi),
        ],
        nx=32, ny=2, nz_per_ply=1,
    )
    result = WrinkleAnalysis(cfg).run()
    kd_analytical = result.analytical_modulus_knockdown
    kd_fe = result.modulus_retention
    assert 0.0 < kd_analytical <= 1.0
    assert 0.0 < kd_fe <= 1.0
    # Both describe the same physical stiffness loss; for these shallow
    # wrinkles they sit near unity and agree to within 5 percentage points.
    assert abs(kd_analytical - kd_fe) < 0.05
