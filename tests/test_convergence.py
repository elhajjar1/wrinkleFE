"""Tests for the mesh-convergence study helper (issue #257)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import wrinklefe.convergence as conv
from wrinklefe.analysis import AnalysisConfig
from wrinklefe.convergence import mesh_convergence_study


# --------------------------------------------------------------------------- #
# Logic tests against a stubbed analysis (fast, deterministic)
# --------------------------------------------------------------------------- #


@dataclass
class _FakeMesh:
    n_dof: int


class _FakeResults:
    """Smooth synthetic QoI: q = q_inf + C / nx**2 (2nd-order behaviour)."""

    def __init__(self, cfg):
        self.config = cfg
        self.mesh = _FakeMesh(n_dof=cfg.nx * cfg.ny * cfg.nz_per_ply * 24)
        self.qoi_value = 1.0 + 50.0 / cfg.nx**2


class _FakeAnalysis:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, analytical_only=False):
        assert analytical_only is False
        return _FakeResults(self.cfg)


def _synthetic_qoi(results) -> float:
    return results.qoi_value


@pytest.fixture()
def stubbed_analysis(monkeypatch):
    monkeypatch.setattr(conv, "WrinkleAnalysis", _FakeAnalysis)


def _base_config(**overrides):
    defaults = dict(
        amplitude=0.366, wavelength=16.0, width=12.0,
        morphology="stack", loading="compression",
        nx=8, ny=2, nz_per_ply=1,
    )
    defaults.update(overrides)
    return AnalysisConfig(**defaults)


class TestConvergenceLogic:

    def test_monotone_delta_decrease_and_table(self, stubbed_analysis):
        study = mesh_convergence_study(
            _base_config(), levels=4, refine=("nx",),
            qoi=_synthetic_qoi, tolerance=0.01,
        )
        assert len(study.levels) == 4
        deltas = [lv.delta_pct for lv in study.levels]
        assert deltas[0] is None
        rest = deltas[1:]
        assert all(d is not None for d in rest)
        # Successive relative change decreases monotonically for the
        # smooth 2nd-order synthetic QoI.
        assert all(a > b for a, b in zip(rest, rest[1:]))
        # DOFs strictly increase level over level.
        dofs = [lv.n_dof for lv in study.levels]
        assert all(a < b for a, b in zip(dofs, dofs[1:]))

    def test_recommendation_respects_tolerance(self, stubbed_analysis):
        # With base nx=8 and factors (1, 1.5, 2, 3), the synthetic QoI's
        # relative distances to the finest level are ~64%, ~24%, ~10% —
        # a 12% tolerance admits exactly the third level.
        tol = 0.12
        study = mesh_convergence_study(
            _base_config(), levels=4, refine=("nx",),
            qoi=_synthetic_qoi, tolerance=tol,
        )
        assert study.recommended_level is not None
        k = study.recommended_level
        q_ref = study.levels[-1].qoi
        assert abs(study.levels[k].qoi - q_ref) <= tol * abs(q_ref)
        # The recommendation is the *coarsest* such level: every coarser
        # level violates the tolerance.
        for j in range(k):
            assert abs(study.levels[j].qoi - q_ref) > tol * abs(q_ref)
        # recommended_config carries the recommended mesh densities.
        assert study.recommended_config.nx == study.levels[k].nx

    def test_no_recommendation_when_tolerance_unreachable(
        self, stubbed_analysis
    ):
        study = mesh_convergence_study(
            _base_config(), levels=3, refine=("nx",),
            qoi=_synthetic_qoi, tolerance=1e-9,
        )
        assert study.recommended_level is None
        assert study.recommended_config is None
        assert "refine further" in study.summary()

    def test_observed_rate_with_constant_ratio(self, stubbed_analysis):
        # Constant refinement ratio 2 -> the 2nd-order synthetic QoI
        # yields an observed rate near 2.
        study = mesh_convergence_study(
            _base_config(), levels=4, refine=("nx",),
            qoi=_synthetic_qoi, factors=(1, 2, 4, 8),
        )
        assert study.observed_rate == pytest.approx(2.0, rel=0.05)

    def test_summary_contains_rows(self, stubbed_analysis):
        study = mesh_convergence_study(
            _base_config(), levels=2, refine=("nx",), qoi=_synthetic_qoi
        )
        text = study.summary()
        assert "Mesh-convergence study" in text
        for lv in study.levels:
            assert str(lv.n_dof) in text

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            (dict(levels=1), "levels"),
            (dict(refine=("bogus",)), "refine"),
            (dict(refine=()), "refine"),
            (dict(qoi="bogus"), "qoi"),
            (dict(levels=4, factors=(1, 2)), "factors"),
        ],
    )
    def test_argument_validation(self, stubbed_analysis, kwargs, match):
        with pytest.raises(ValueError, match=match):
            mesh_convergence_study(_base_config(), **kwargs)


# --------------------------------------------------------------------------- #
# Real-FE smoke tests (coarse and fast)
# --------------------------------------------------------------------------- #


class TestConvergenceRealFE:

    def test_pure_fe_max_fi(self):
        # 8 plies and a modest amplitude keep the coarse meshes valid
        # (a large amplitude on a thin laminate inverts hex elements).
        cfg = _base_config(
            amplitude=0.15, angles=[0, 90, 90, 0, 0, 90, 90, 0]
        )
        study = mesh_convergence_study(
            cfg, levels=2, refine=("nx",), qoi="max_fi",
            factors=(1.0, 1.5),
        )
        assert len(study.levels) == 2
        assert study.levels[0].n_dof < study.levels[1].n_dof
        assert all(np.isfinite(lv.qoi) and lv.qoi > 0 for lv in study.levels)
        assert all(lv.runtime_s > 0 for lv in study.levels)

    def test_czm_mode_max_damage(self):
        from wrinklefe.core.material import MaterialLibrary

        cfg = _base_config(
            morphology="concave", loading="tension",
            material=MaterialLibrary().get("IM7_8552"),
            angles=([0, 90] * 2) + ([90, 0] * 2),
            ply_thickness=0.183,
            nx=8, ny=2, nz_per_ply=1,
            applied_strain=0.015,
            enable_czm=True,
            czm_n_load_increments=5,
        )
        study = mesh_convergence_study(
            cfg, levels=2, refine=("nx",), qoi="max_damage",
            factors=(1.0, 1.5),
        )
        assert len(study.levels) == 2
        assert all(np.isfinite(lv.qoi) for lv in study.levels)
