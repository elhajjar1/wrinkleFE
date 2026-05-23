"""Smoke tests for the multi-parameter batch sweep module (issue #1).

``WrinkleAnalysis.parametric_sweep`` already has thorough coverage in
``tests/test_analysis_sweep.py`` (issues #13, #44, #49).  The higher-level
``wrinklefe.sweep.run_sweep`` helper — which adds multi-parameter
cross-product sweeps, JSON serialisation, and the
``wrinklefe.sweep.parametric_sweep`` script entry point — had no
dedicated tests.  These cover the public surface relied on by issue #1
(batch parametric sweep capability).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from wrinklefe.analysis import AnalysisResults
from wrinklefe.sweep import run_sweep, save_sweep_results


def _stub_run(self, analytical_only=None):
    """Return a deterministic stub AnalysisResults capturing the config.

    ``run_sweep`` only reads a handful of scalar fields off the result via
    ``_result_to_metrics``; populating those with the swept amplitude makes
    the test assertions self-checking without paying for a real solve.
    """
    res = AnalysisResults(config=self.config)
    a = float(self.config.amplitude)
    # Drive the extracted metrics off amplitude so we can verify the grid.
    res.analytical_knockdown = max(0.0, 1.0 - a)
    res.analytical_strength_MPa = 1500.0 * res.analytical_knockdown
    res.max_angle_rad = float(np.arctan(2.0 * np.pi * a / self.config.wavelength))
    res.effective_angle_rad = res.max_angle_rad
    res.morphology_factor = 1.0
    return res


class TestRunSweepSingleParameter:
    """Single-parameter sweep via the higher-level batch helper."""

    def test_amplitude_sweep_produces_one_entry_per_value(self):
        amps = np.array([0.2, 0.4, 0.6])
        with patch(
            "wrinklefe.sweep.parametric_sweep.WrinkleAnalysis.run", _stub_run
        ):
            out = run_sweep({"amplitude": amps})

        assert out["swept_params"] == ["amplitude"]
        assert out["param_values"]["amplitude"] == pytest.approx(amps.tolist())
        # One entry per swept value, keyed by float.
        assert sorted(out["results"].keys()) == pytest.approx(amps.tolist())
        # Each entry has the three named morphologies populated.
        for v in amps:
            morphs = out["results"][float(v)]
            assert set(morphs) == {"stack", "convex", "concave"}
            assert morphs["stack"]["knockdown_factor"] == pytest.approx(1.0 - v)

    def test_phase_sweep_uses_single_custom_morphology(self):
        """Phase sweeps run a single morphology per point (issue #49)."""
        phases = np.array([0.0, 0.5, 1.0])
        with patch(
            "wrinklefe.sweep.parametric_sweep.WrinkleAnalysis.run", _stub_run
        ):
            out = run_sweep({"phase": phases})

        assert out["swept_params"] == ["phase"]
        # Phase sweeps key the per-morphology dict as 'custom' (one entry,
        # not the three named morphologies), and stamp phase_rad/phase_deg
        # so downstream plotting can find them.
        for v in phases:
            point = out["results"][float(v)]
            assert set(point) == {"custom"}
            assert point["custom"]["phase_rad"] == pytest.approx(v)


class TestRunSweepCrossProduct:
    """Multi-parameter cross-product sweep — the headline batch feature."""

    def test_two_param_grid_yields_cartesian_product(self):
        amps = np.array([0.2, 0.4])
        wls = np.array([12.0, 20.0])

        with patch(
            "wrinklefe.sweep.parametric_sweep.WrinkleAnalysis.run", _stub_run
        ):
            out = run_sweep({"amplitude": amps, "wavelength": wls})

        # swept_params is sorted alphabetically inside run_sweep.
        assert out["swept_params"] == ["amplitude", "wavelength"]
        # 2 x 2 = 4 grid points, keyed by stringified tuple.
        assert len(out["results"]) == 4
        expected_keys = {
            str((float(a), float(w))) for a in amps for w in wls
        }
        assert set(out["results"]) == expected_keys

        # Each point has all three morphologies and the swept amplitude
        # propagates into the knockdown via the stub.
        for key, morphs in out["results"].items():
            a, _ = eval(key)  # safe: only floats produced by run_sweep
            assert morphs["stack"]["knockdown_factor"] == pytest.approx(1.0 - a)


class TestSaveSweepResults:
    """JSON serialisation round-trip — the persistence surface for batch
    sweeps that callers wire into downstream analysis / plotting."""

    def test_save_sweep_results_writes_loadable_json(self, tmp_path: Path):
        amps = np.array([0.2, 0.4])
        with patch(
            "wrinklefe.sweep.parametric_sweep.WrinkleAnalysis.run", _stub_run
        ):
            out = run_sweep({"amplitude": amps})

        filepath = save_sweep_results(out, str(tmp_path))
        assert Path(filepath).exists()
        data = json.loads(Path(filepath).read_text())
        assert data["swept_params"] == ["amplitude"]
        # JSON keys are strings, but the underlying floats round-trip.
        assert set(data["results"].keys()) == {"0.2", "0.4"}
        # Sanity: the recorded knockdowns match our stub.
        assert data["results"]["0.2"]["stack"]["knockdown_factor"] == pytest.approx(
            0.8
        )
