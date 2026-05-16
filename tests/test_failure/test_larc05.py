"""Dedicated unit tests for the LaRC04/05 failure criterion.

These tests target the LaRC05-specific physics that ``test_criteria.py``
and ``test_evaluator.py`` do not exercise directly:

- fibre tension with the quadratic shear-interaction term,
- fibre kinking under compression via the misalignment-frame rotation
  (the criterion only engages kinking when an initial misalignment
  ``phi_0`` is supplied, by design of ``_compute_phi_c``),
- matrix tension governed by the *in-situ* transverse strength,
- matrix compression resolved on the fracture plane at +/- alpha_0,
- monotonic / proportional-loading behaviour and reserve factor.

Material: default ``OrthotropicMaterial()`` == IM7/8552 (matching the
convention used by the other ``tests/test_failure/`` modules; this is
the same data as ``MaterialLibrary().get("IM7_8552")``).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.failure.base import FailureResult
from wrinklefe.failure.larc05 import LaRC05Criterion


@pytest.fixture
def material():
    """Default IM7/8552 orthotropic material (same as MaterialLibrary)."""
    return OrthotropicMaterial()


@pytest.fixture
def criterion():
    return LaRC05Criterion()


# ======================================================================
# Zero stress
# ======================================================================

class TestLaRC05ZeroStress:

    def test_zero_stress_gives_fi_zero(self, criterion, material):
        result = criterion.evaluate(np.zeros(6), material)
        assert isinstance(result, FailureResult)
        assert result.index == pytest.approx(0.0, abs=1e-12)
        assert result.criterion_name == "larc05"

    def test_zero_stress_reserve_factor_infinite(self, criterion, material):
        result = criterion.evaluate(np.zeros(6), material)
        assert np.isinf(result.reserve_factor)


# ======================================================================
# Fibre tension (with shear interaction)
# ======================================================================

class TestLaRC05FibreTension:

    def test_pure_longitudinal_tension_at_Xt(self, criterion, material):
        """sigma_11 = Xt -> fibre-tension FI = 1.0, mode fiber_tension."""
        stress = np.array([material.Xt, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-10)
        assert result.mode == "fiber_tension"
        # Fibre sub-criterion governs; no matrix damage under pure sigma_11.
        assert result.detail["fi_fiber"] == pytest.approx(1.0, abs=1e-10)
        assert result.detail["fi_matrix"] == pytest.approx(0.0, abs=1e-10)

    def test_below_Xt_fi_less_than_one(self, criterion, material):
        stress = np.array([0.5 * material.Xt, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(0.5, abs=1e-10)
        assert result.index < 1.0

    def test_above_Xt_fi_greater_than_one(self, criterion, material):
        stress = np.array([1.5 * material.Xt, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.5, abs=1e-10)
        assert result.index > 1.0

    def test_shear_interaction_amplifies_fibre_tension(
        self, criterion, material
    ):
        """Adding tau_12 to a longitudinal-tension state must raise the
        fibre-tension FI (the LaRC05 quadratic shear-interaction term)."""
        base = np.array([0.5 * material.Xt, 0.0, 0.0, 0.0, 0.0, 0.0])
        with_shear = base.copy()
        with_shear[5] = 0.5 * material.S12
        fi_base = criterion.evaluate(base, material).index
        fi_shear = criterion.evaluate(with_shear, material).index
        assert fi_shear > fi_base


# ======================================================================
# Fibre kinking (compression)
# ======================================================================

class TestLaRC05FibreKinking:

    def test_compression_no_misalignment_takes_kinking_branch(
        self, criterion, material
    ):
        """Under sigma_11 < 0 the fibre branch is kinking. With phi_0 = 0
        the load-induced rotation is zero, so the rotated kink-band shear
        vanishes and the kinking FI is 0 (documented behaviour of
        ``_compute_phi_c``)."""
        stress = np.array([-material.Xc, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.detail["mode_fiber"] == "fiber_kinking"
        assert result.detail["fi_fiber"] == pytest.approx(0.0, abs=1e-12)

    def test_kinking_engages_with_misalignment(self, criterion, material):
        """With a finite misalignment angle the kink-band frame sees a
        non-zero shear and the kinking FI grows with |sigma_11|."""
        ctx = {"misalignment_angle": 0.1}
        fi_lo = criterion.evaluate(
            np.array([-800.0, 0.0, 0.0, 0.0, 0.0, 0.0]), material, ctx
        ).index
        fi_hi = criterion.evaluate(
            np.array([-1400.0, 0.0, 0.0, 0.0, 0.0, 0.0]), material, ctx
        ).index
        assert 0.0 < fi_lo < fi_hi

    def test_kinking_fi_unity_at_kinking_allowable(
        self, criterion, material
    ):
        """For phi_0 = 0.1 the kinking allowable is sigma_11 ~= -1164.7 MPa
        (located numerically). At that load FI ~= 1.0, the *kinking* path
        governs and the matrix path does not contribute."""
        ctx = {"misalignment_angle": 0.1}
        sigma_kink = -1164.6909607060759
        stress = np.array([sigma_kink, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material, ctx)
        assert result.index == pytest.approx(1.0, abs=1e-3)
        assert result.mode == "fiber_kinking"
        # Kinking, not matrix, must be the governing sub-criterion.
        assert result.detail["fi_fiber"] >= result.detail["fi_matrix"]
        assert result.detail["fi_matrix"] == pytest.approx(0.0, abs=1e-10)

    def test_higher_misalignment_increases_kinking_fi(
        self, criterion, material
    ):
        stress = np.array([-1000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        fi0 = criterion.evaluate(
            stress, material, {"misalignment_angle": 0.05}
        ).index
        fi1 = criterion.evaluate(
            stress, material, {"misalignment_angle": 0.15}
        ).index
        assert fi1 > fi0


# ======================================================================
# Matrix tension (Mode A) -- in-situ strength governs
# ======================================================================

class TestLaRC05MatrixTension:

    def test_transverse_tension_unity_at_in_situ_Yt(
        self, criterion, material
    ):
        """LaRC05 uses the *in-situ* transverse strength. For a thin ply
        without GIc/GIIc this is 1.12*sqrt(2)*Yt, so FI reaches 1.0 at
        sigma_22 = Yt_is (not at the raw Yt)."""
        Yt_is, _ = criterion._in_situ_strengths(material)
        assert Yt_is > material.Yt  # in-situ enhancement is real
        stress = np.array([0.0, Yt_is, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=0.01)
        assert result.mode == "matrix_tension"

    def test_transverse_tension_at_raw_Yt_below_unity(
        self, criterion, material
    ):
        """At the *raw* Yt the in-situ correction keeps FI below 1.0."""
        stress = np.array([0.0, material.Yt, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index < 1.0
        assert result.mode == "matrix_tension"


# ======================================================================
# Matrix compression (fracture-plane search) -- Mode B/C
# ======================================================================

class TestLaRC05MatrixCompression:

    def test_transverse_compression_unity_at_Yc(self, criterion, material):
        """sigma_22 = -Yc -> matrix-compression FI = 1.0 (the fracture
        plane angle calibration recovers the uniaxial allowable)."""
        stress = np.array([0.0, -material.Yc, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-6)
        assert result.mode == "matrix_compression"

    def test_transverse_compression_governing_fracture_angle(
        self, criterion, material
    ):
        """The fracture-plane search must select the angle |alpha| = alpha_0
        (~53 deg for CFRP) for pure transverse compression."""
        stress = np.array([0.0, -material.Yc, 0.0, 0.0, 0.0, 0.0])
        Yt_is, S12_is = criterion._in_situ_strengths(material)
        mu_L, mu_T = LaRC05Criterion._friction_coefficients(material)

        alpha_0_rad = np.radians(material.alpha_0)
        tan_2a = np.tan(2.0 * alpha_0_rad)
        S_T = material.Yc * np.cos(alpha_0_rad) * (
            np.sin(alpha_0_rad) + np.cos(alpha_0_rad) / tan_2a
        )
        S_L = S12_is

        thetas = np.linspace(-np.pi / 2, np.pi / 2, criterion.n_theta)
        ct, st = np.cos(thetas), np.sin(thetas)
        s2 = -material.Yc
        sigma_n = s2 * ct ** 2
        tau_nt = -s2 * st * ct
        fi = np.zeros(len(thetas))
        for i in range(len(thetas)):
            sn, tnt = sigma_n[i], tau_nt[i]
            if sn >= 0:
                fi[i] = (tnt / S_T) ** 2 + (sn / Yt_is) ** 2
            else:
                denom_t = S_T + mu_T * abs(sn)
                fi[i] = (tnt / denom_t) ** 2
        alpha_governing = abs(np.degrees(thetas[int(np.argmax(fi))]))
        assert alpha_governing == pytest.approx(material.alpha_0, abs=1.0)
        assert alpha_governing == pytest.approx(53.0, abs=1.0)

    def test_transverse_compression_below_Yc_under_unity(
        self, criterion, material
    ):
        stress = np.array([0.0, -0.75 * material.Yc, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index < 1.0
        assert result.mode == "matrix_compression"


# ======================================================================
# In-plane shear
# ======================================================================

class TestLaRC05InPlaneShear:

    def test_pure_inplane_shear_at_S12(self, criterion, material):
        """Pure tau_12 = S12. With sigma_11 = 0 the fibre-tension branch is
        taken and its quadratic shear-interaction term gives
        FI = tau_12 / S12 = 1.0."""
        stress = np.array([0.0, 0.0, 0.0, 0.0, 0.0, material.S12])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-10)

    def test_inplane_shear_scales_linearly(self, criterion, material):
        half = criterion.evaluate(
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * material.S12]),
            material,
        ).index
        assert half == pytest.approx(0.5, abs=1e-10)


# ======================================================================
# Monotonicity / proportional loading / reserve factor
# ======================================================================

class TestLaRC05MonotonicityAndReserve:

    def test_doubling_stress_increases_fi(self, criterion, material):
        stress = np.array([300.0, 20.0, 0.0, 0.0, 0.0, 30.0])
        fi1 = criterion.evaluate(stress, material).index
        fi2 = criterion.evaluate(2.0 * stress, material).index
        assert fi2 > fi1

    def test_proportional_scaling_is_linear_in_load(
        self, criterion, material
    ):
        """Every LaRC05 sub-FI is linear-in-load (issue #79), so a
        proportional load doubling exactly doubles FI."""
        stress = np.array([300.0, 20.0, 0.0, 0.0, 0.0, 30.0])
        fi1 = criterion.evaluate(stress, material).index
        fi2 = criterion.evaluate(2.0 * stress, material).index
        assert fi2 == pytest.approx(2.0 * fi1, rel=1e-9)

    def test_reserve_factor_scales_proportional_load_to_failure(
        self, criterion, material
    ):
        """Scaling a proportional load by its reserve factor must drive
        FI to exactly 1.0 (first failure)."""
        stress = np.array([300.0, 20.0, 0.0, 0.0, 0.0, 30.0])
        result = criterion.evaluate(stress, material)
        rf = result.reserve_factor
        at_failure = criterion.evaluate(rf * stress, material)
        assert at_failure.index == pytest.approx(1.0, rel=1e-9)

    def test_reserve_factor_is_inverse_of_index(self, criterion, material):
        for stress, ctx in [
            (np.array([0.5 * material.Xt, 0.0, 0.0, 0.0, 0.0, 0.0]), None),
            (np.array([0.0, 0.5 * material.Yt, 0.0, 0.0, 0.0, 0.0]), None),
            (np.array([0.0, -0.5 * material.Yc, 0.0, 0.0, 0.0, 0.0]), None),
            (
                np.array([-1000.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                {"misalignment_angle": 0.10},
            ),
        ]:
            result = criterion.evaluate(stress, material, ctx)
            if result.index > 0:
                assert result.reserve_factor == pytest.approx(
                    1.0 / result.index, rel=1e-12
                )
