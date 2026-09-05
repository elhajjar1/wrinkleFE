"""Thermal / cure-residual loading through the CLT pipeline (issue #273).

Stage 1 of #273 made ``AnalysisConfig.delta_T`` reachable: the config
value is threaded into the :class:`~wrinklefe.core.laminate.LoadState`
the analysis pipeline evaluates, so the CLT thermal resultants enter the
ABD solve and the recovered ply stresses carry the cure residual stress.

Three things are pinned here, in rising order of how badly a regression
would hurt:

1. **Closed-form CLT.** ``config -> LoadState -> midplane_strains``
   reproduces an *independently* computed reference (a small textbook CLT
   implementation local to this module — it does not call the functions
   under test to build its own expectation). Symmetric layups show zero
   thermal curvature; unsymmetric ones do not.
2. **Sign.** ΔT is the temperature change *from the stress-free (cure)
   state*, so a cool-down is negative, and a negative ΔT must put the
   matrix direction of a cross-ply in **tension** — the matrix-cracking
   driver. Getting this backwards is exactly the failure #133 fixed once
   already at the resultant level, and it reappeared at the ply-stress
   level (stress was recovered from the total rather than the mechanical
   strain) until this issue.
3. **Reachability.** A non-zero ΔT must actually change the analytical
   result. Stage 2 also opened the FE path (element thermal
   initial-strain load vector), so what used to be a refusal here is now
   an acceptance check; the FE physics itself lives in
   ``tests/test_thermal_fe.py``.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.laminate import Laminate, LoadState
from wrinklefe.core.material import MaterialLibrary, OrthotropicMaterial

# A cure cool-down for a 177 deg C-cure epoxy prepreg taken to 22 deg C
# room temperature. Negative by the from-stress-free convention.
CURE_COOLDOWN_DT = -155.0

PLY_T = 0.183


# --------------------------------------------------------------------------- #
# Independent CLT reference
#
# Deliberately written from the textbook definitions (Jones, *Mechanics of
# Composite Materials*, ch. 4) rather than by calling ``Laminate``: it is
# the yardstick the pipeline is measured against, so it must not share an
# implementation with it.
# --------------------------------------------------------------------------- #


def _q_local(mat: OrthotropicMaterial) -> np.ndarray:
    """Reduced stiffness [Q] in the ply material frame (MPa)."""
    nu21 = mat.nu12 * mat.E2 / mat.E1
    den = 1.0 - mat.nu12 * nu21
    return np.array(
        [
            [mat.E1 / den, mat.nu12 * mat.E2 / den, 0.0],
            [mat.nu12 * mat.E2 / den, mat.E2 / den, 0.0],
            [0.0, 0.0, mat.G12],
        ]
    )


def _strain_transform(theta: float) -> np.ndarray:
    """Strain transformation [T_eps] (global -> local, engineering shear)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c * c, s * s, c * s],
            [s * s, c * c, -c * s],
            [-2.0 * c * s, 2.0 * c * s, c * c - s * s],
        ]
    )


def _q_bar(Q: np.ndarray, theta: float) -> np.ndarray:
    """[Q-bar] = [T_eps]^T [Q] [T_eps] in the laminate frame."""
    T = _strain_transform(theta)
    return T.T @ Q @ T


def _alpha_global(mat: OrthotropicMaterial, theta: float) -> np.ndarray:
    """CTE vector rotated into the laminate frame (engineering shear)."""
    c, s = np.cos(theta), np.sin(theta)
    a1, a2 = mat.alpha1, mat.alpha2
    return np.array(
        [
            a1 * c * c + a2 * s * s,
            a1 * s * s + a2 * c * c,
            2.0 * (a1 - a2) * s * c,
        ]
    )


def _reference_thermal_response(
    angles_deg: list[float],
    mat: OrthotropicMaterial,
    ply_thickness: float,
    delta_T: float,
) -> dict[str, np.ndarray]:
    """Independent CLT solution for a laminate under pure ΔT.

    Returns the midplane strain/curvature vector, the ABD sub-matrices and
    the thermal resultants, all assembled from the textbook definitions.
    """
    n = len(angles_deg)
    h = n * ply_thickness
    z = np.array([-h / 2.0 + i * ply_thickness for i in range(n + 1)])
    Q = _q_local(mat)

    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    NT = np.zeros(3)
    MT = np.zeros(3)

    for k, angle in enumerate(angles_deg):
        theta = np.radians(angle)
        Qb = _q_bar(Q, theta)
        A += Qb * (z[k + 1] - z[k])
        B += Qb * (z[k + 1] ** 2 - z[k] ** 2) / 2.0
        D += Qb * (z[k + 1] ** 3 - z[k] ** 3) / 3.0
        Qa = Qb @ _alpha_global(mat, theta)
        NT += Qa * delta_T * (z[k + 1] - z[k])
        MT += Qa * delta_T * (z[k + 1] ** 2 - z[k] ** 2) / 2.0

    ABD = np.block([[A, B], [B, D]])
    response = np.linalg.solve(ABD, np.concatenate([NT, MT]))
    return {"response": response, "A": A, "B": B, "NT": NT, "MT": MT}


def _pipeline_load_state(delta_T: float, **cfg_kwargs) -> LoadState:
    """Build the CLT load state the pipeline evaluates, from a config.

    This is the seam the issue is about: ``AnalysisConfig.delta_T`` ->
    ``WrinkleAnalysis._clt_load_state()`` -> ``LoadState.delta_T``.
    """
    cfg = AnalysisConfig(
        delta_T=delta_T, analytical_only=True, **cfg_kwargs
    )
    return WrinkleAnalysis(cfg)._clt_load_state()


# --------------------------------------------------------------------------- #
# 1. Closed-form CLT through the pipeline
# --------------------------------------------------------------------------- #


class TestClosedFormThermalCLT:
    """``config -> LoadState -> midplane_strains`` vs an independent CLT."""

    def test_symmetric_cross_ply_matches_independent_reference(
        self, x850_material
    ):
        """``[0/90]s`` under pure ΔT reproduces the hand-assembled CLT
        solution to machine precision, and shows zero curvature."""
        angles = [0.0, 90.0, 90.0, 0.0]
        ref = _reference_thermal_response(
            angles, x850_material, PLY_T, CURE_COOLDOWN_DT
        )

        lam = Laminate.from_angles(
            angles, material=x850_material, ply_thickness=PLY_T
        )
        load = _pipeline_load_state(
            CURE_COOLDOWN_DT,
            angles=angles,
            material=x850_material,
            ply_thickness=PLY_T,
            applied_strain=0.0,
        )
        assert load.delta_T == CURE_COOLDOWN_DT
        # Pure thermal: no mechanical resultant on this check.
        thermal_only = LoadState(delta_T=load.delta_T)
        observed = lam.midplane_strains(thermal_only)

        npt.assert_allclose(
            observed, ref["response"], rtol=1e-10, atol=1e-14
        )
        # Symmetric layup: B = 0 and M^T = 0, hence no thermal curvature.
        npt.assert_allclose(ref["B"], np.zeros((3, 3)), atol=1e-9)
        npt.assert_allclose(ref["MT"], np.zeros(3), atol=1e-10)
        npt.assert_allclose(observed[3:6], np.zeros(3), atol=1e-12)

        # And the in-plane part is the decoupled A eps0 = N^T solution.
        npt.assert_allclose(
            observed[0:3],
            np.linalg.solve(ref["A"], ref["NT"]),
            rtol=1e-10,
            atol=1e-14,
        )

    def test_symmetric_cross_ply_strain_is_equibiaxial_and_contracting(
        self, x850_material
    ):
        """A balanced ``[0/90]s`` cooling down contracts equally in x and y.

        Both directions see the same mix of one fibre-dominated and one
        matrix-dominated ply, so ``eps0_x == eps0_y``; and cooling a
        laminate whose CTEs are positive on balance shrinks it.
        """
        angles = [0.0, 90.0, 90.0, 0.0]
        lam = Laminate.from_angles(
            angles, material=x850_material, ply_thickness=PLY_T
        )
        eps0 = lam.midplane_strains(LoadState(delta_T=CURE_COOLDOWN_DT))[0:3]

        npt.assert_allclose(eps0[0], eps0[1], rtol=1e-12)
        assert eps0[0] < 0.0, "cool-down must contract the laminate"
        npt.assert_allclose(eps0[2], 0.0, atol=1e-14)

    def test_unsymmetric_cross_ply_develops_curvature(self, x850_material):
        """``[0/90]`` (unsymmetric) under pure ΔT bends: kappa != 0.

        The classic unsymmetric-cross-ply thermal warpage, and the check
        that the ``B`` / ``M^T`` coupling is really being carried through
        the pipeline rather than dropped.
        """
        angles = [0.0, 90.0]
        ref = _reference_thermal_response(
            angles, x850_material, PLY_T, CURE_COOLDOWN_DT
        )

        lam = Laminate.from_angles(
            angles, material=x850_material, ply_thickness=PLY_T
        )
        observed = lam.midplane_strains(LoadState(delta_T=CURE_COOLDOWN_DT))

        npt.assert_allclose(
            observed, ref["response"], rtol=1e-10, atol=1e-14
        )
        assert np.linalg.norm(observed[3:6]) > 1e-6, (
            "an unsymmetric cross-ply must warp under a pure thermal load"
        )
        # Saddle: the two bending curvatures are equal and opposite.
        npt.assert_allclose(observed[3], -observed[4], rtol=1e-10)


# --------------------------------------------------------------------------- #
# 2. Sign / direction — the engineer's sanity check
# --------------------------------------------------------------------------- #


class TestThermalSignConvention:
    """Cool-down puts the matrix direction of a cross-ply in tension."""

    def test_cooldown_puts_cross_ply_transverse_in_tension(
        self, x850_material
    ):
        """Every ply of ``[0/90]s`` sees sigma_2 > 0 after a cool-down.

        In a cross-ply each ply's transverse (matrix) direction is the
        neighbouring ply's stiff fibre direction, which restrains its much
        larger transverse contraction — so cooling leaves the matrix in
        **tension** and the fibres in compression. This is the residual
        stress that drives cure-induced transverse microcracking; reporting
        its sign backwards would make a cracking-prone laminate look safe.
        """
        angles = [0.0, 90.0, 90.0, 0.0]
        lam = Laminate.from_angles(
            angles, material=x850_material, ply_thickness=PLY_T
        )
        load = LoadState(delta_T=CURE_COOLDOWN_DT)

        for k in range(len(angles)):
            sigma_1, sigma_2, _tau = lam.ply_stresses_local(load, k)
            assert sigma_2 > 0.0, (
                f"ply {k} ({angles[k]} deg): cool-down must put the "
                f"transverse direction in tension, got sigma_2={sigma_2}"
            )
            assert sigma_1 < 0.0, (
                f"ply {k} ({angles[k]} deg): cool-down must put the fibre "
                f"direction in compression, got sigma_1={sigma_1}"
            )

    def test_heat_up_reverses_every_sign(self, x850_material):
        """A positive ΔT of the same magnitude mirrors the stress state."""
        angles = [0.0, 90.0, 90.0, 0.0]
        lam = Laminate.from_angles(
            angles, material=x850_material, ply_thickness=PLY_T
        )
        cold = lam.ply_stresses_local(LoadState(delta_T=-100.0), 0)
        hot = lam.ply_stresses_local(LoadState(delta_T=+100.0), 0)
        npt.assert_allclose(hot, -cold, rtol=1e-10, atol=1e-12)

    def test_residual_stress_self_equilibrates(self, x850_material):
        """With no mechanical load the ply stresses must sum to zero force.

        ``sum_k sigma_k t_k == 0`` in the laminate frame: a purely thermal
        state is self-equilibrated. This is what catches a stress recovery
        that forgets to subtract the free thermal strain — that version
        reports a net compressive resultant out of nothing.
        """
        angles = [0.0, 45.0, -45.0, 90.0, 90.0, -45.0, 45.0, 0.0]
        lam = Laminate.from_angles(
            angles, material=x850_material, ply_thickness=PLY_T
        )
        load = LoadState(delta_T=CURE_COOLDOWN_DT)
        resultant = sum(
            lam.ply_stresses_global(load, k) * PLY_T
            for k in range(len(angles))
        )
        npt.assert_allclose(resultant, np.zeros(3), atol=1e-8)

    def test_unrestrained_ply_free_expansion_is_stress_free(
        self, x850_material
    ):
        """A single unrestrained ply under pure ΔT carries no stress."""
        lam = Laminate.from_angles(
            [0.0], material=x850_material, ply_thickness=PLY_T
        )
        for dT in (CURE_COOLDOWN_DT, +120.0):
            sigma = lam.ply_stresses_local(LoadState(delta_T=dT), 0)
            npt.assert_allclose(sigma, np.zeros(3), atol=1e-9)

    def test_delta_T_zero_leaves_mechanical_stress_untouched(
        self, x850_material
    ):
        """The ΔT = 0 path is byte-identical to the pre-#273 behaviour."""
        angles = [0.0, 90.0, 90.0, 0.0]
        lam = Laminate.from_angles(
            angles, material=x850_material, ply_thickness=PLY_T
        )
        load = LoadState(Nx=-120.0, My=3.0, delta_T=0.0)
        for k in range(len(angles)):
            eps = lam.ply_strains(load, k)
            expected = lam.plies[k].Q_bar() @ eps
            npt.assert_allclose(
                lam.ply_stresses_global(load, k), expected, rtol=0, atol=0
            )


# --------------------------------------------------------------------------- #
# 3. Config plumbing, the FE guard, and the measured knockdown effect
# --------------------------------------------------------------------------- #


class TestConfigPlumbing:
    """``AnalysisConfig.delta_T`` reaches the CLT load state and no further."""

    def test_default_is_zero_and_load_state_is_thermally_neutral(self):
        cfg = AnalysisConfig()
        assert cfg.delta_T == 0.0
        assert WrinkleAnalysis(cfg)._clt_load_state().delta_T == 0.0

    def test_delta_T_reaches_the_load_state(self):
        load = _pipeline_load_state(CURE_COOLDOWN_DT)
        assert load.delta_T == CURE_COOLDOWN_DT

    def test_mechanical_part_of_the_load_state_is_unchanged(self):
        """Adding ΔT must not disturb the mechanical resultant."""
        cfg_kwargs = {"applied_strain": -0.012}
        cold = _pipeline_load_state(CURE_COOLDOWN_DT, **cfg_kwargs)
        neutral = _pipeline_load_state(0.0, **cfg_kwargs)
        npt.assert_allclose(cold.to_vector(), neutral.to_vector(), atol=0)

    def test_round_trips_through_to_dict_from_dict(self):
        cfg = AnalysisConfig(delta_T=CURE_COOLDOWN_DT, analytical_only=True)
        data = cfg.to_dict()
        assert data["delta_T"] == CURE_COOLDOWN_DT
        assert AnalysisConfig.from_dict(data) == cfg

    @pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
    def test_non_finite_rejected(self, bad):
        with pytest.raises(ValueError, match="delta_T must be finite"):
            AnalysisConfig(delta_T=float(bad), analytical_only=True)

    @pytest.mark.parametrize("bad", [1000.1, -5000.0])
    def test_absurd_magnitude_rejected_with_the_convention_restated(
        self, bad
    ):
        """The out-of-range message must explain the from-cure convention.

        The overwhelmingly likely cause of a huge value is an *absolute*
        temperature typed into a *change* field, so the error says so.
        """
        with pytest.raises(ValueError) as exc:
            AnalysisConfig(delta_T=bad, analytical_only=True)
        msg = str(exc.value)
        assert "stress-free" in msg
        assert "-155" in msg


class TestFEPathAccepted:
    """Stage 2 opened the FE path to ΔT — construction must not refuse it.

    Stage 1 rejected ``delta_T != 0`` with ``analytical_only=False``
    because the element formulation carried no initial-strain term.  That
    term now exists (``GlobalAssembler.assemble_thermal_force``), so the
    refusal is gone; the physics it guarded is covered in
    ``tests/test_thermal_fe.py``.
    """

    def test_construction_accepts_delta_T_on_the_fe_path(self):
        cfg = AnalysisConfig(
            delta_T=CURE_COOLDOWN_DT, analytical_only=False
        )
        assert cfg.delta_T == CURE_COOLDOWN_DT
        assert cfg.analytical_only is False

    def test_runtime_analytical_only_override_is_allowed(self):
        """``run(analytical_only=False)`` no longer needs its own guard."""
        cfg = AnalysisConfig(
            delta_T=CURE_COOLDOWN_DT, analytical_only=True,
            angles=[0.0, 90.0, 90.0, 0.0],
            nx=4, ny=2, nz_per_ply=1,
            domain_length=12.0, domain_width=8.0,
        )
        results = WrinkleAnalysis(cfg).run(analytical_only=False)
        # The FE path actually ran (fields populated), carrying ΔT.
        assert results.field_results is not None

    def test_zero_delta_T_leaves_the_fe_path_open(self):
        """The default is unchanged."""
        cfg = AnalysisConfig(delta_T=0.0, analytical_only=False)
        assert cfg.delta_T == 0.0


class TestKnockdownEffect:
    """The measured effect of a cure cool-down on a wrinkled coupon.

    Numbers below are *measured* from this code, not assumed: for the
    IM7/8552 quasi-isotropic coupon used here a ΔT = -155 cool-down adds
    about +34 MPa of transverse (matrix) tension to every ply — roughly
    55 % of ``Yt`` — and drives the CLT first-ply-failure load factor down
    by ~98 %, flipping the critical mode to matrix tension. The assertions
    are pinned loosely (direction and order of magnitude) so the physics is
    guarded without freezing the calibration.
    """

    LAYUP = [0.0, 45.0, -45.0, 90.0, 90.0, -45.0, 45.0, 0.0]

    def _config(self, delta_T: float) -> AnalysisConfig:
        return AnalysisConfig(
            amplitude=0.5,
            wavelength=16.0,
            width=12.0,
            morphology="graded",
            loading="compression",
            applied_strain=-0.01,
            material=MaterialLibrary().get("IM7_8552"),
            angles=list(self.LAYUP),
            ply_thickness=PLY_T,
            analytical_only=True,
            delta_T=delta_T,
        )

    def test_cooldown_adds_matrix_tension_of_the_expected_size(self):
        """Residual sigma_2 is a large fraction of the transverse strength."""
        analysis = WrinkleAnalysis(self._config(CURE_COOLDOWN_DT))
        lam = analysis._build_laminate()
        mat = MaterialLibrary().get("IM7_8552")

        cold = analysis._clt_load_state()
        neutral = LoadState(Nx=cold.Nx)

        deltas = [
            lam.ply_stresses_local(cold, k)[1]
            - lam.ply_stresses_local(neutral, k)[1]
            for k in range(len(self.LAYUP))
        ]
        assert min(deltas) > 0.0, (
            "cool-down must add transverse tension in every ply"
        )
        # Measured 34.3 MPa (55 % of Yt = 62.3). Pinned loosely.
        assert 10.0 < max(deltas) < 60.0
        assert max(deltas) / mat.Yt > 0.25, (
            "the cure residual stress is a first-order fraction of Yt and "
            "must not quietly shrink to a rounding error"
        )

    def test_cooldown_lowers_the_clt_first_ply_failure_margin(self):
        """FPF load factor drops and the critical mode becomes matrix tension.

        Measured on this case: 173.3 -> 3.62 (a 97.9 % reduction), with the
        critical mode moving from ``matrix_compression`` to
        ``matrix_tension``. The reference mechanical resultant the pipeline
        uses is deliberately small, so the thermal term dominates — which is
        precisely why leaving it out of a run that asks for it would be a
        wrong number rather than a small one.
        """
        from wrinklefe.failure.evaluator import FailureEvaluator

        cold_analysis = WrinkleAnalysis(self._config(CURE_COOLDOWN_DT))
        warm_analysis = WrinkleAnalysis(self._config(0.0))
        lam = warm_analysis._build_laminate()
        evaluator = FailureEvaluator.default_criteria()

        warm = evaluator.evaluate_laminate(
            lam, warm_analysis._clt_load_state()
        )
        cold = evaluator.evaluate_laminate(
            lam, cold_analysis._clt_load_state()
        )

        criterion = cold.critical_criterion or next(iter(cold.fpf))
        lf_warm = warm.fpf[criterion]["load_factor"]
        lf_cold = cold.fpf[criterion]["load_factor"]

        assert lf_cold < lf_warm, (
            "a cure cool-down superposes matrix tension and must reduce, "
            f"not raise, the first-ply-failure margin ({lf_warm} -> {lf_cold})"
        )
        # Direction pinned tightly, magnitude loosely.
        assert lf_cold < 0.5 * lf_warm
        assert "tension" in (cold.fpf[criterion]["mode"] or "")

    def test_analytical_run_populates_a_thermal_failure_report(self):
        """A thermal analytical run produces an output that depends on ΔT.

        Without this the whole feature would be a no-op on the only path it
        is currently valid for — the closed-form knockdown carries no
        temperature term.
        """
        cold = WrinkleAnalysis(self._config(CURE_COOLDOWN_DT)).run()
        assert cold.failure_report is not None
        assert cold.failure_report.critical_criterion

    def test_default_analytical_run_is_unchanged(self):
        """ΔT = 0 must leave the analytical path exactly as it was.

        In particular ``failure_report`` stays ``None`` on an
        analytical-only run, which the JSON/CSV exporters rely on for their
        load-factor fallback.
        """
        warm = WrinkleAnalysis(self._config(0.0)).run()
        assert warm.failure_report is None

    def test_summary_states_the_sign_convention(self):
        """The run summary must not report ΔT without saying what it means."""
        cold = WrinkleAnalysis(self._config(CURE_COOLDOWN_DT)).run()
        text = cold.summary()
        assert "-155.0 deg C" in text
        assert "cool-down from the stress-free/cure state" in text

        warm = WrinkleAnalysis(self._config(0.0)).run()
        assert "delta_T" not in warm.summary()


# --------------------------------------------------------------------------- #
# 4. CLI surface
# --------------------------------------------------------------------------- #


class TestCLIDeltaT:
    """``wrinklefe analyze --delta-T`` reaches the config, or refuses loudly."""

    @staticmethod
    def _run_capturing_config(argv: list[str]) -> AnalysisConfig:
        from unittest.mock import MagicMock, patch

        from wrinklefe.cli import main as cli_main

        captured: dict = {}

        def fake_run(self, analytical_only=None):
            captured["config"] = self.config
            result = MagicMock()
            result.summary.return_value = "<stub>"
            result.czm_damage = None
            return result

        with patch("wrinklefe.analysis.WrinkleAnalysis.run", new=fake_run):
            cli_main(argv)
        return captured["config"]

    def test_flag_reaches_the_config(self):
        cfg = self._run_capturing_config(
            ["analyze", "--delta-T", "-155", "--analytical-only",
             "--angles", "[0/90]s"]
        )
        assert cfg.delta_T == CURE_COOLDOWN_DT

    def test_flag_omitted_leaves_the_default(self):
        cfg = self._run_capturing_config(
            ["analyze", "--analytical-only", "--angles", "[0/90]s"]
        )
        assert cfg.delta_T == 0.0

    def test_fe_path_accepts_delta_T(self):
        """``--delta-T`` without ``--analytical-only`` reaches the config.

        Stage 1 exited 2 here.  Stage 2 assembles the FE thermal load, so
        the flag is accepted on both paths.
        """
        cfg = self._run_capturing_config(
            ["analyze", "--delta-T", "-155", "--angles", "[0/90]s"]
        )
        assert cfg.delta_T == -155.0
        assert cfg.analytical_only is False

    def test_help_states_the_sign_convention(self, capsys):
        from wrinklefe.cli import main as cli_main

        with pytest.raises(SystemExit):
            cli_main(["analyze", "--help"])
        out = capsys.readouterr().out
        # Collapse argparse's wrapping before matching.
        flat = " ".join(out.split())
        assert "FROM the stress-free (cure) state" in flat
        assert "cool-down is NEGATIVE" in flat
        assert "--delta-T -155" in flat
