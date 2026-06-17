"""Tests for the two-parameter (theta, D/T) penetration-gate model (D.3)."""

from __future__ import annotations

import numpy as np
import pytest

from wrinklefe.core.penetration_gate import (
    GATE_LI2024_MOULDED,
    GATE_LI2025_VACBAG,
    GateParameters,
    angle_floor,
    calibrate_gate,
    penetration_gate_kd,
    predict_from_geometry,
)

# VALIDATION_DATA section 2.7 single-wrinkle grids (theta_deg, D/T, KD).
F_GRID = [
    (10.3, 0.122, 0.891), (20.1, 0.122, 0.629), (30.2, 0.122, 0.472),
    (20.1, 0.081, 0.943), (20.1, 0.041, 1.000),
]
E_GRID = [
    (4.9, 0.025, 0.907), (10.6, 0.026, 0.823), (16.0, 0.026, 0.758),
    (16.7, 0.056, 0.612), (15.8, 0.079, 0.523), (16.5, 0.083, 0.545),
    (14.2, 0.105, 0.506), (16.6, 0.042, 0.657), (15.9, 0.059, 0.558),
]


class TestGateParameters:
    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            GateParameters(gamma_Y=0.0, dt0=0.1, p=4.0)
        with pytest.raises(ValueError):
            GateParameters(gamma_Y=0.3, dt0=-1.0, p=4.0)


class TestGateModel:
    def test_shallow_limit_no_knockdown(self):
        # D/T -> 0 => gate S -> 0 => KD -> 1 regardless of angle.
        kd = penetration_gate_kd(30.0, 1e-6, GATE_LI2025_VACBAG)
        assert kd == pytest.approx(1.0, abs=1e-3)

    def test_deep_limit_is_angle_floor(self):
        # Large D/T => S = 1 => KD = KD_angle(theta).
        p = GATE_LI2025_VACBAG
        kd = penetration_gate_kd(20.0, 5.0, p)
        assert kd == pytest.approx(angle_floor(20.0, p.gamma_Y), rel=1e-6)

    def test_monotonic_in_penetration(self):
        # At fixed angle, deeper penetration cannot raise the knockdown.
        dts = np.linspace(0.0, 0.3, 20)
        kd = penetration_gate_kd(np.full_like(dts, 20.0), dts,
                                 GATE_LI2025_VACBAG)
        assert np.all(np.diff(kd) <= 1e-9)

    def test_monotonic_in_angle(self):
        # At fixed penetration, a steeper wrinkle knocks down at least as
        # much.
        ths = np.linspace(1.0, 40.0, 20)
        kd = penetration_gate_kd(ths, np.full_like(ths, 0.122),
                                 GATE_LI2025_VACBAG)
        assert np.all(np.diff(kd) <= 1e-9)

    def test_scalar_returns_float(self):
        assert isinstance(
            penetration_gate_kd(20.0, 0.1, GATE_LI2025_VACBAG), float
        )


def _mae_pass(grid, params):
    errs = [abs(penetration_gate_kd(th, dt, params) - kd) / kd
            for th, dt, kd in grid]
    return float(np.mean(errs)), sum(1 for e in errs if e <= 0.20)


class TestCalibration:
    def test_presets_reproduce_li_fits(self):
        # The shipped presets must hit the documented accuracy.
        mae_e, npass_e = _mae_pass(E_GRID, GATE_LI2024_MOULDED)
        mae_f, npass_f = _mae_pass(F_GRID, GATE_LI2025_VACBAG)
        assert mae_e < 0.05 and npass_e == len(E_GRID)
        assert mae_f < 0.10 and npass_f == len(F_GRID)

    def test_calibrate_recovers_good_fit(self):
        pytest.importorskip("scipy")
        th, dt, kd = zip(*F_GRID)
        params = calibrate_gate(th, dt, kd, name="F_refit")
        mae, npass = _mae_pass(F_GRID, params)
        assert mae < 0.10 and npass == len(F_GRID)

    def test_gate_beats_angle_only_on_F(self):
        # The two-parameter gate must beat an angle-only fit on F (the
        # whole point of the (theta, D/T) model).
        mae_gate, _ = _mae_pass(F_GRID, GATE_LI2025_VACBAG)
        # Angle-only: dt0 tiny so S=1 everywhere (pure angle floor).
        angle_only = GateParameters(gamma_Y=GATE_LI2025_VACBAG.gamma_Y,
                                    dt0=1e-6, p=1.0)
        mae_angle, _ = _mae_pass(F_GRID, angle_only)
        assert mae_gate < mae_angle


class TestGeometryAndPipeline:
    def test_predict_from_geometry_matches_F_S_M_2(self):
        # A=0.75, L=12.9, 14 plies x 0.44 -> theta~20deg, D/T~0.122.
        kd = predict_from_geometry(0.75, 12.9, 14, 0.44, GATE_LI2025_VACBAG)
        assert kd == pytest.approx(0.629, abs=0.05)

    def test_pipeline_uses_gate(self):
        from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
        from wrinklefe.core.material import MaterialLibrary
        mat = MaterialLibrary().get("AC318_S6C10_vacbag")
        common = dict(amplitude=0.75, wavelength=12.9, width=6.45,
                      morphology="graded", loading="compression",
                      material=mat, angles=[0.0] * 14, ply_thickness=0.44)
        gated = WrinkleAnalysis(
            AnalysisConfig(**common, penetration_gate=GATE_LI2025_VACBAG)
        ).run(analytical_only=True)
        baseline = WrinkleAnalysis(AnalysisConfig(**common)).run(
            analytical_only=True)
        # The gate result matches the standalone predictor and differs
        # from the Budiansky-Fleck baseline.
        assert gated.analytical_knockdown == pytest.approx(
            predict_from_geometry(0.75, 12.9, 14, 0.44, GATE_LI2025_VACBAG),
            abs=1e-6)
        assert gated.analytical_knockdown != pytest.approx(
            baseline.analytical_knockdown, abs=1e-3)
        assert gated.analytical_strength_MPa == pytest.approx(
            mat.Xc * gated.analytical_knockdown, rel=1e-6)

    def test_pipeline_rejects_bad_gate(self):
        from wrinklefe.analysis import AnalysisConfig
        with pytest.raises(ValueError):
            AnalysisConfig(penetration_gate="not a gate")
