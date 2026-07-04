"""Process-parallel parametric sweeps (issue #260).

Every sweep point is an independent analysis (own mesh, own solve, no
shared state), so ``WrinkleAnalysis.parametric_sweep`` and
``wrinklefe.sweep.run_sweep`` gain ``n_workers`` and fan the per-point
solves out over a ``ProcessPoolExecutor``.

Coverage:

* results and ordering identical to the sequential path (real
  cross-process analytical runs — no patching, works on every start
  method);
* ``n_workers=1`` remains the untouched sequential path and
  ``n_workers`` validation rejects nonsense;
* config/results picklability (the process-boundary precondition);
* wall-clock speedup and worker-failure propagation via a stubbed
  solve — these patch ``WrinkleAnalysis.run`` in the parent, which only
  reaches fork-started workers, so they are skipped where the default
  start method is not fork (e.g. macOS/Windows).
"""

from __future__ import annotations

import multiprocessing
import pickle
import time

import numpy as np
import pytest

from wrinklefe.analysis import AnalysisConfig, AnalysisResults, WrinkleAnalysis
from wrinklefe.sweep import run_sweep

_FORK_ONLY = pytest.mark.skipif(
    multiprocessing.get_start_method() != "fork",
    reason="stubbed-solve tests patch the parent process; the patch only "
    "reaches fork-started workers",
)


def _analytical_cfg(**overrides) -> AnalysisConfig:
    base = dict(
        amplitude=0.2, wavelength=16.0, width=12.0,
        analytical_only=True,
    )
    base.update(overrides)
    return AnalysisConfig(**base)


# --------------------------------------------------------------------------- #
# WrinkleAnalysis.parametric_sweep
# --------------------------------------------------------------------------- #


class TestParametricSweepParallel:

    def test_parallel_matches_sequential_and_order(self):
        """Real cross-process run: identical results, identical order."""
        cfg = _analytical_cfg()
        vals = np.linspace(0.1, 0.5, 6).tolist()
        seq = WrinkleAnalysis.parametric_sweep(
            cfg, "amplitude", vals, analytical_only=True, n_workers=1
        )
        par = WrinkleAnalysis.parametric_sweep(
            cfg, "amplitude", vals, analytical_only=True, n_workers=3
        )
        assert [r.config.amplitude for r in par] == vals
        assert [r.analytical_knockdown for r in par] == [
            r.analytical_knockdown for r in seq
        ]
        assert [r.analytical_strength_MPa for r in par] == [
            r.analytical_strength_MPa for r in seq
        ]

    def test_wavelength_domain_rederivation_survives_parallel(self):
        """The auto-derived domain_length reset (sweeping wavelength)
        behaves identically through the parallel path."""
        cfg = _analytical_cfg()
        vals = [8.0, 16.0, 24.0]
        seq = WrinkleAnalysis.parametric_sweep(
            cfg, "wavelength", vals, analytical_only=True, n_workers=1
        )
        par = WrinkleAnalysis.parametric_sweep(
            cfg, "wavelength", vals, analytical_only=True, n_workers=2
        )
        assert [r.config.domain_length for r in par] == [
            r.config.domain_length for r in seq
        ]
        assert [r.analytical_knockdown for r in par] == [
            r.analytical_knockdown for r in seq
        ]

    @pytest.mark.parametrize("bad", [-1, 2.5, True])
    def test_invalid_n_workers_rejected(self, bad):
        with pytest.raises(ValueError, match="n_workers"):
            WrinkleAnalysis.parametric_sweep(
                _analytical_cfg(), "amplitude", [0.1, 0.2],
                analytical_only=True, n_workers=bad,
            )

    def test_unknown_parameter_still_raises_before_any_work(self):
        with pytest.raises(AttributeError, match="no field"):
            WrinkleAnalysis.parametric_sweep(
                _analytical_cfg(), "not_a_field", [0.1],
                analytical_only=True, n_workers=4,
            )

    @_FORK_ONLY
    def test_worker_failure_propagates_and_pool_shuts_down(self):
        def _boom(self, analytical_only=None):
            raise RuntimeError("worker exploded")

        from unittest.mock import patch

        with patch.object(WrinkleAnalysis, "run", _boom):
            with pytest.raises(RuntimeError, match="worker exploded"):
                WrinkleAnalysis.parametric_sweep(
                    _analytical_cfg(), "amplitude", [0.1, 0.2, 0.3],
                    analytical_only=True, n_workers=2,
                )


# --------------------------------------------------------------------------- #
# run_sweep
# --------------------------------------------------------------------------- #


def _stub_run(self, analytical_only=None):
    """Deterministic, amplitude-driven stub (mirrors the existing
    run_sweep tests)."""
    res = AnalysisResults(config=self.config)
    a = float(self.config.amplitude)
    res.analytical_knockdown = max(0.0, 1.0 - a)
    res.analytical_strength_MPa = 1500.0 * res.analytical_knockdown
    res.max_angle_rad = float(
        np.arctan(2.0 * np.pi * a / self.config.wavelength)
    )
    res.effective_angle_rad = res.max_angle_rad
    res.morphology_factor = 1.0
    return res


def _sleepy_run(self, analytical_only=None):
    time.sleep(0.25)
    return _stub_run(self, analytical_only)


class TestRunSweepParallel:

    @_FORK_ONLY
    def test_parallel_matches_sequential_grid(self):
        from unittest.mock import patch

        amps = np.array([0.2, 0.4, 0.6])
        with patch(
            "wrinklefe.sweep.parametric_sweep.WrinkleAnalysis.run", _stub_run
        ):
            seq = run_sweep({"amplitude": amps}, n_workers=1)
            par = run_sweep({"amplitude": amps}, n_workers=3)

        assert list(par["results"].keys()) == list(seq["results"].keys())
        for key in seq["results"]:
            assert par["results"][key] == seq["results"][key]

    @_FORK_ONLY
    def test_parallel_is_faster_than_sequential(self):
        """4 workers on 8 quarter-second points must beat sequential by
        a wide margin (generous threshold to stay CI-robust)."""
        from unittest.mock import patch

        amps = np.linspace(0.1, 0.8, 8)
        with patch(
            "wrinklefe.sweep.parametric_sweep.WrinkleAnalysis.run",
            _sleepy_run,
        ):
            t0 = time.perf_counter()
            run_sweep({"amplitude": amps}, n_workers=1)
            t_seq = time.perf_counter() - t0
            t0 = time.perf_counter()
            run_sweep({"amplitude": amps}, n_workers=4)
            t_par = time.perf_counter() - t0
        # 8 points x 3 morphologies x 0.25 s = 6 s sequential; 4 workers
        # should land near 1.5 s. Require >= 2x.
        assert t_par < t_seq / 2.0, (t_seq, t_par)

    @pytest.mark.parametrize("bad", [-2, 1.5])
    def test_invalid_n_workers_rejected(self, bad):
        with pytest.raises(ValueError, match="n_workers"):
            run_sweep({"amplitude": np.array([0.1, 0.2])}, n_workers=bad)


# --------------------------------------------------------------------------- #
# Process-boundary preconditions
# --------------------------------------------------------------------------- #


class TestPicklability:

    def test_config_and_analytical_results_roundtrip(self):
        cfg = _analytical_cfg()
        assert pickle.loads(pickle.dumps(cfg)).amplitude == cfg.amplitude
        res = WrinkleAnalysis(cfg).run(analytical_only=True)
        back = pickle.loads(pickle.dumps(res))
        assert back.analytical_knockdown == res.analytical_knockdown

    def test_fe_results_roundtrip(self):
        """Full FE results (mesh + field arrays + failure reports) must
        cross the process boundary intact."""
        cfg = AnalysisConfig(
            amplitude=0.2, wavelength=16.0, width=12.0,
            angles=[0.0, 90.0] * 4, nx=6, ny=2, nz_per_ply=1,
        )
        res = WrinkleAnalysis(cfg).run()
        back = pickle.loads(pickle.dumps(res))
        assert back.analytical_knockdown == res.analytical_knockdown
        assert back.modulus_retention == res.modulus_retention
        np.testing.assert_array_equal(
            back.field_results.displacement, res.field_results.displacement
        )
