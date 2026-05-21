"""Regression tests for ply-level context forwarding in FailureEvaluator.

These tests pin the fix for issue #193: ``FailureEvaluator.evaluate_laminate``
must forward a per-ply ``context`` dict to each criterion.  Without it,
LaRC05 silently evaluates with ``phi_0 = 0`` and the fibre-kinking model
collapses to the no-wrinkle case.
"""

from __future__ import annotations

import numpy as np
import pytest

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.core.laminate import Laminate, LoadState
from wrinklefe.failure.evaluator import FailureEvaluator
from wrinklefe.failure.larc05 import LaRC05Criterion


@pytest.fixture
def im7_8552():
    """Default IM7/8552 material."""
    return OrthotropicMaterial()


@pytest.fixture
def unidirectional_laminate(im7_8552):
    """Simple [0]_4 unidirectional laminate."""
    return Laminate.from_angles(
        angles=[0.0, 0.0, 0.0, 0.0],
        material=im7_8552,
        ply_thickness=0.183,
    )


@pytest.fixture
def evaluator():
    return FailureEvaluator([LaRC05Criterion()])


class TestPlyContextForwarding:
    """Misalignment must propagate through the laminate path to LaRC05."""

    def test_misalignment_reduces_load_factor_list(
        self, evaluator, unidirectional_laminate
    ):
        """A compressive [0]_4 laminate with phi_0 = 0.05 rad should
        reach FI = 1 at a smaller load factor than the same laminate
        with phi_0 = 0 (i.e. wrinkles weaken the laminate).
        """
        # Uniaxial compression in the fibre direction
        load = LoadState(Nx=-200.0)

        # Baseline: no misalignment (no per-ply context)
        report_no_wrinkle = evaluator.evaluate_laminate(
            unidirectional_laminate, load
        )
        lf_no_wrinkle = report_no_wrinkle.fpf["larc05"]["load_factor"]

        # Wrinkled: phi_0 = 0.05 rad on every ply, supplied as a list
        n_plies = unidirectional_laminate.n_plies
        ply_contexts_list = [
            {"misalignment_angle": 0.05} for _ in range(n_plies)
        ]
        report_wrinkle = evaluator.evaluate_laminate(
            unidirectional_laminate, load, ply_contexts=ply_contexts_list
        )
        lf_wrinkle = report_wrinkle.fpf["larc05"]["load_factor"]

        assert np.isfinite(lf_no_wrinkle)
        assert np.isfinite(lf_wrinkle)
        # The wrinkled laminate must fail at a lower load factor.
        assert lf_wrinkle < lf_no_wrinkle, (
            f"misalignment_angle did not lower the load factor: "
            f"no_wrinkle={lf_no_wrinkle}, wrinkle={lf_wrinkle}"
        )

    def test_misalignment_reduces_load_factor_dict(
        self, evaluator, unidirectional_laminate
    ):
        """Same regression check as above, but passing the context as a
        ``dict`` keyed by ply index (the alternative supported form)."""
        load = LoadState(Nx=-200.0)

        report_no_wrinkle = evaluator.evaluate_laminate(
            unidirectional_laminate, load
        )
        lf_no_wrinkle = report_no_wrinkle.fpf["larc05"]["load_factor"]

        ply_contexts_dict = {
            k: {"misalignment_angle": 0.05}
            for k in range(unidirectional_laminate.n_plies)
        }
        report_wrinkle = evaluator.evaluate_laminate(
            unidirectional_laminate, load, ply_contexts=ply_contexts_dict
        )
        lf_wrinkle = report_wrinkle.fpf["larc05"]["load_factor"]

        assert lf_wrinkle < lf_no_wrinkle

    def test_default_no_context_matches_explicit_zero(
        self, evaluator, unidirectional_laminate
    ):
        """Omitting ply_contexts must be equivalent to phi_0 = 0 on every ply."""
        load = LoadState(Nx=-200.0)

        report_default = evaluator.evaluate_laminate(
            unidirectional_laminate, load
        )
        ply_contexts = [
            {"misalignment_angle": 0.0}
            for _ in range(unidirectional_laminate.n_plies)
        ]
        report_zero = evaluator.evaluate_laminate(
            unidirectional_laminate, load, ply_contexts=ply_contexts
        )
        assert report_default.fpf["larc05"]["load_factor"] == pytest.approx(
            report_zero.fpf["larc05"]["load_factor"], rel=1e-12
        )

    def test_partial_dict_context_defaults_missing_to_none(
        self, evaluator, unidirectional_laminate
    ):
        """A dict that omits some ply indices must not raise; the missing
        plies should be treated as having no context (phi_0 = 0)."""
        load = LoadState(Nx=-200.0)

        # Only specify context for ply 0; the rest fall back to None.
        ply_contexts = {0: {"misalignment_angle": 0.05}}
        report = evaluator.evaluate_laminate(
            unidirectional_laminate, load, ply_contexts=ply_contexts
        )
        assert "larc05" in report.fpf


class TestStrengthRatioEnvelopeContext:
    """``strength_ratio_envelope`` must accept and forward ``ply_contexts``."""

    def test_envelope_with_misalignment_shrinks(
        self, evaluator, unidirectional_laminate
    ):
        """For a compressive sector of the Nx-Ny envelope, adding fibre
        misalignment should shrink the failure envelope (smaller |Nx|
        at failure)."""
        env_no_wrinkle = evaluator.strength_ratio_envelope(
            unidirectional_laminate, load_type="Nx-Ny", n_points=8
        )["larc05"]

        ply_contexts = [
            {"misalignment_angle": 0.05}
            for _ in range(unidirectional_laminate.n_plies)
        ]
        env_wrinkle = evaluator.strength_ratio_envelope(
            unidirectional_laminate,
            load_type="Nx-Ny",
            n_points=8,
            ply_contexts=ply_contexts,
        )["larc05"]

        # theta = pi corresponds to pure -Nx (compression).  With n_points=8
        # the angle index for theta = pi is 4 (linspace endpoint=False).
        nx_no_wrinkle = abs(env_no_wrinkle[4, 0])
        nx_wrinkle = abs(env_wrinkle[4, 0])
        assert nx_wrinkle < nx_no_wrinkle, (
            f"misalignment did not shrink the compressive envelope: "
            f"|Nx| no_wrinkle={nx_no_wrinkle}, wrinkle={nx_wrinkle}"
        )

    def test_envelope_accepts_dict_form(self, evaluator, unidirectional_laminate):
        """Smoke test: the dict form is also accepted by
        ``strength_ratio_envelope``."""
        ply_contexts = {
            k: {"misalignment_angle": 0.02}
            for k in range(unidirectional_laminate.n_plies)
        }
        env = evaluator.strength_ratio_envelope(
            unidirectional_laminate,
            load_type="Nx-Ny",
            n_points=4,
            ply_contexts=ply_contexts,
        )
        assert "larc05" in env
        assert env["larc05"].shape == (4, 2)


class TestEvaluatePointContext:
    """``evaluate_point`` must also forward an optional context dict."""

    def test_point_misalignment_increases_fi(self, evaluator, im7_8552):
        """A compressive stress in the fibre direction should yield a higher
        LaRC05 failure index when a non-zero misalignment is supplied."""
        stress = np.array([-1000.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        res_no_ctx = evaluator.evaluate_point(stress, im7_8552)
        res_with_ctx = evaluator.evaluate_point(
            stress, im7_8552, context={"misalignment_angle": 0.05}
        )

        assert res_with_ctx["larc05"].index > res_no_ctx["larc05"].index
