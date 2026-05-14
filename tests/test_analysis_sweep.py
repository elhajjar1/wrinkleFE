"""Regression tests for AnalysisConfig field preservation in sweeps.

Verifies the fix for issue #13: ``compare_morphologies`` and
``parametric_sweep`` previously rebuilt a fresh :class:`AnalysisConfig`
field-by-field from ``base_config``, silently dropping fields like
``decay_floor``.  Both helpers must now preserve every existing field
and only override the swept parameters.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from wrinklefe.analysis import AnalysisConfig, AnalysisResults, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary


@pytest.fixture
def base_config_with_decay_floor():
    """Tiny config with a non-default decay_floor (issue #13)."""
    return AnalysisConfig(
        amplitude=0.366,
        wavelength=16.0,
        width=12.0,
        morphology="stack",
        decay_floor=0.3,  # non-default
        loading="compression",
        material=MaterialLibrary().get("IM7_8552"),
        angles=[0, 90, 0, 90, 0, 90, 90, 0, 90, 0, 90, 0],
        interface_1=5,
        interface_2=6,
        nx=4,
        ny=2,
        nz_per_ply=1,
        domain_width=10.0,
        applied_strain=-0.005,
        verbose=False,
    )


def _make_stub_result(config):
    """Build a bare AnalysisResults capturing the config it was given."""
    return AnalysisResults(config=config)


class TestDecayFloorPreservedInSweep:
    """Issue #13: swept configs must retain user-set decay_floor."""

    def test_parametric_sweep_preserves_decay_floor(
        self, base_config_with_decay_floor
    ):
        """Sweeping amplitude must not reset decay_floor to 0.0."""
        captured_configs = []

        def fake_run(self, analytical_only=None):
            captured_configs.append(self.config)
            return _make_stub_result(self.config)

        amps = [0.183, 0.366, 0.549]
        with patch.object(WrinkleAnalysis, "run", fake_run):
            results = WrinkleAnalysis.parametric_sweep(
                base_config_with_decay_floor, "amplitude", amps
            )

        assert len(results) == 3
        assert len(captured_configs) == 3
        # All swept configs preserve the user-set decay_floor.
        for cfg, expected_amp in zip(captured_configs, amps):
            assert cfg.decay_floor == pytest.approx(0.3), (
                f"decay_floor was silently dropped during sweep "
                f"(got {cfg.decay_floor}, expected 0.3)"
            )
            assert cfg.amplitude == pytest.approx(expected_amp)

        # The same invariant holds when inspecting the returned results.
        for res in results:
            assert res.config.decay_floor == pytest.approx(0.3)

    def test_compare_morphologies_preserves_decay_floor(
        self, base_config_with_decay_floor
    ):
        """compare_morphologies must not reset decay_floor across morphologies."""
        captured_configs = []

        def fake_run(self, analytical_only=None):
            captured_configs.append(self.config)
            return _make_stub_result(self.config)

        morphologies = ("stack", "convex", "concave")
        with patch.object(WrinkleAnalysis, "run", fake_run):
            results = WrinkleAnalysis.compare_morphologies(
                base_config_with_decay_floor, morphologies=morphologies
            )

        assert set(results.keys()) == set(morphologies)
        assert len(captured_configs) == len(morphologies)
        for cfg in captured_configs:
            assert cfg.decay_floor == pytest.approx(0.3), (
                f"decay_floor was silently dropped during compare "
                f"(got {cfg.decay_floor}, expected 0.3)"
            )
        # Each morphology slot keeps its own decay_floor.
        for morph in morphologies:
            assert results[morph].config.morphology == morph
            assert results[morph].config.decay_floor == pytest.approx(0.3)

    def test_parametric_sweep_preserves_other_nondefault_fields(
        self, base_config_with_decay_floor
    ):
        """Other user-set fields must also survive the sweep."""

        def fake_run(self, analytical_only=None):
            return _make_stub_result(self.config)

        with patch.object(WrinkleAnalysis, "run", fake_run):
            results = WrinkleAnalysis.parametric_sweep(
                base_config_with_decay_floor, "amplitude", [0.2, 0.4]
            )

        for res in results:
            cfg = res.config
            assert cfg.morphology == "stack"
            assert cfg.interface_1 == 5
            assert cfg.interface_2 == 6
            assert cfg.applied_strain == pytest.approx(-0.005)
            assert cfg.nx == 4
            assert cfg.ny == 2
            assert cfg.domain_width == pytest.approx(10.0)
            assert cfg.decay_floor == pytest.approx(0.3)
