"""Tests for FailureEvaluator and LaminateFailureReport with LaRC05."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.core.laminate import Laminate, LoadState
from wrinklefe.failure.base import FailureResult
from wrinklefe.failure.larc05 import LaRC05Criterion
from wrinklefe.failure.evaluator import FailureEvaluator, LaminateFailureReport


@pytest.fixture
def x850_material():
    return OrthotropicMaterial()


@pytest.fixture
def evaluator():
    """Evaluator with LaRC05 criterion."""
    return FailureEvaluator([LaRC05Criterion()])


@pytest.fixture
def quasi_iso_laminate(x850_material):
    """[0/45/-45/90]2s laminate."""
    half_angles = [0, 45, -45, 90, 0, 45, -45, 90]
    return Laminate.symmetric(half_angles, material=x850_material, ply_thickness=0.183)


# ======================================================================
# FailureEvaluator creation tests
# ======================================================================

class TestFailureEvaluatorCreation:

    def test_creation_with_larc05(self):
        ev = FailureEvaluator([LaRC05Criterion()])
        assert len(ev.criteria) == 1

    def test_empty_criteria_raises(self):
        with pytest.raises(ValueError):
            FailureEvaluator([])

    def test_default_criteria_creates_larc05(self):
        ev = FailureEvaluator.default_criteria()
        assert len(ev.criteria) == 1
        assert ev.criteria[0].name == "larc05"


# ======================================================================
# evaluate_point tests
# ======================================================================

class TestEvaluatePoint:

    def test_returns_dict_with_larc05(self, evaluator, x850_material):
        stress = np.array([500.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        results = evaluator.evaluate_point(stress, x850_material)
        assert isinstance(results, dict)
        assert "larc05" in results

    def test_result_is_failure_result(self, evaluator, x850_material):
        stress = np.array([500.0, 30.0, 0.0, 0.0, 0.0, 50.0])
        results = evaluator.evaluate_point(stress, x850_material)
        assert isinstance(results["larc05"], FailureResult)

    def test_invalid_stress_shape_raises(self, evaluator, x850_material):
        stress = np.array([500.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            evaluator.evaluate_point(stress, x850_material)

    def test_zero_stress_fi_zero(self, evaluator, x850_material):
        stress = np.zeros(6)
        results = evaluator.evaluate_point(stress, x850_material)
        assert results["larc05"].index < 1e-10


# ======================================================================
# evaluate_laminate tests
# ======================================================================

class TestEvaluateLaminate:

    def test_returns_laminate_failure_report(self, evaluator, quasi_iso_laminate):
        load = LoadState(Nx=-500.0)
        report = evaluator.evaluate_laminate(quasi_iso_laminate, load)
        assert isinstance(report, LaminateFailureReport)

    def test_report_has_fpf_for_larc05(self, evaluator, quasi_iso_laminate):
        load = LoadState(Nx=-500.0)
        report = evaluator.evaluate_laminate(quasi_iso_laminate, load)
        assert "larc05" in report.fpf

    def test_report_has_ply_failure_indices(self, evaluator, quasi_iso_laminate):
        load = LoadState(Nx=-500.0)
        report = evaluator.evaluate_laminate(quasi_iso_laminate, load)
        fi_arr = report.ply_failure_indices["larc05"]
        assert fi_arr.shape == (quasi_iso_laminate.n_plies,)

    def test_critical_ply_is_valid_index(self, evaluator, quasi_iso_laminate):
        load = LoadState(Nx=-1000.0)
        report = evaluator.evaluate_laminate(quasi_iso_laminate, load)
        assert 0 <= report.critical_ply < quasi_iso_laminate.n_plies

    def test_critical_criterion_is_larc05(self, evaluator, quasi_iso_laminate):
        load = LoadState(Nx=-1000.0)
        report = evaluator.evaluate_laminate(quasi_iso_laminate, load)
        assert report.critical_criterion == "larc05"

    def test_summary_returns_string(self, evaluator, quasi_iso_laminate):
        load = LoadState(Nx=-500.0)
        report = evaluator.evaluate_laminate(quasi_iso_laminate, load)
        summary = report.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0


# ======================================================================
# evaluate_field tests
# ======================================================================

class TestEvaluateField:

    def test_evaluate_field_with_synthetic_data(self, x850_material):
        evaluator = FailureEvaluator([LaRC05Criterion()])
        n_elements = 10
        n_gauss = 4
        rng = np.random.default_rng(42)
        stress_field = rng.uniform(-100, 100, (n_elements, n_gauss, 6))
        materials = [x850_material]
        ply_ids = np.zeros(n_elements, dtype=int)

        fi_fields, mode_fields = evaluator.evaluate_field(stress_field, materials, ply_ids)

        assert isinstance(fi_fields, dict)
        assert "larc05" in fi_fields
        assert fi_fields["larc05"].shape == (n_elements, n_gauss)

    def test_evaluate_field_invalid_shape_raises(self, x850_material):
        evaluator = FailureEvaluator([LaRC05Criterion()])
        stress_field = np.zeros((5, 4, 3))  # wrong last dim
        materials = [x850_material]
        ply_ids = np.zeros(5, dtype=int)
        with pytest.raises(ValueError):
            evaluator.evaluate_field(stress_field, materials, ply_ids)

    def test_evaluate_field_mismatched_ply_ids_raises(self, x850_material):
        evaluator = FailureEvaluator([LaRC05Criterion()])
        stress_field = np.zeros((5, 4, 6))
        materials = [x850_material]
        ply_ids = np.zeros(3, dtype=int)  # wrong length
        with pytest.raises(ValueError):
            evaluator.evaluate_field(stress_field, materials, ply_ids)

    def test_evaluate_field_zero_stress(self, x850_material):
        evaluator = FailureEvaluator([LaRC05Criterion()])
        stress_field = np.zeros((3, 2, 6))
        materials = [x850_material]
        ply_ids = np.zeros(3, dtype=int)
        fi_fields, mode_fields = evaluator.evaluate_field(stress_field, materials, ply_ids)
        assert_allclose(fi_fields["larc05"], 0.0, atol=1e-12)

    def test_evaluate_field_with_fiber_angles(self, x850_material):
        evaluator = FailureEvaluator([LaRC05Criterion()])
        n_elements = 5
        n_gauss = 2
        stress_field = np.zeros((n_elements, n_gauss, 6))
        stress_field[:, :, 0] = -800.0  # compression
        materials = [x850_material]
        ply_ids = np.zeros(n_elements, dtype=int)
        fiber_angles = np.array([0.0, 0.05, 0.10, 0.15, 0.20])

        fi_fields, mode_fields = evaluator.evaluate_field(
            stress_field, materials, ply_ids, fiber_angles=fiber_angles
        )

        # Higher misalignment should give higher FI
        fi_means = fi_fields["larc05"].mean(axis=1)
        assert fi_means[-1] > fi_means[0], "Higher misalignment should increase FI"

    def test_evaluate_field_returns_modes(self, x850_material):
        evaluator = FailureEvaluator([LaRC05Criterion()])
        stress_field = np.zeros((3, 2, 6))
        stress_field[:, :, 0] = -800.0  # compression
        materials = [x850_material]
        ply_ids = np.zeros(3, dtype=int)
        fiber_angles = np.array([0.1, 0.1, 0.1])

        fi_fields, mode_fields = evaluator.evaluate_field(
            stress_field, materials, ply_ids, fiber_angles=fiber_angles
        )

        assert "larc05" in mode_fields
        assert mode_fields["larc05"].shape == (3, 2)
        assert mode_fields["larc05"][0, 0] in (
            "fiber_kinking", "matrix_tension", "matrix_compression"
        )
