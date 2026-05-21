"""Regression tests for Hashin matrix-compression reserve factor (issue #195).

The matrix-compression branch of Hashin builds an FI^2 polynomial that
mixes a *linear* stress term ``((Yc/(2*S23))**2 - 1) * (s22+s33) / Yc``
with quadratic terms.  Under proportional load scaling R, the polynomial
``FI_mc^2(R) = A*R^2 + B*R`` is *not* a pure quadratic, so the naive
``RF = 1/FI`` is incorrect for that mode.  The fix solves
``A*R_f^2 + B*R_f = 1`` for the positive root and uses that as the
reserve factor when matrix_compression is the dominant mode.

For the other three modes (fiber_tension, fiber_compression,
matrix_tension), FI is a pure quadratic in stress and ``RF = 1/FI``
remains correct.
"""

import numpy as np
import pytest

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.failure.hashin import HashinCriterion
from wrinklefe.failure.max_stress import MaxStressCriterion


@pytest.fixture
def material():
    return OrthotropicMaterial()


@pytest.fixture
def criterion():
    return HashinCriterion()


class TestHashinMatrixCompressionRF:
    """Acceptance tests for the closed-form RF in the matrix-compression branch."""

    def test_pure_transverse_compression_at_Yc_index_and_rf_unity(
        self, criterion, material
    ):
        """At s22 = -Yc, FI = 1 *and* RF = 1.

        The two coefficients should satisfy A + B = 1 at the failure
        surface, so the quadratic root must also be R_f = 1.
        """
        stress = np.array([0.0, -material.Yc, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.mode == "matrix_compression"
        assert result.index == pytest.approx(1.0, abs=1e-10)
        assert result.reserve_factor == pytest.approx(1.0, abs=1e-10)

    def test_half_transverse_compression_rf_is_two(self, criterion, material):
        """At s22 = -Yc/2, scaling the load by 2 gives s22 = -Yc => failure.

        Therefore RF must be exactly 2.0 (the true strength ratio), even
        though 1/FI != 2 due to the linear-term mixing.
        """
        stress = np.array([0.0, -material.Yc / 2.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.mode == "matrix_compression"
        # The strength ratio is exactly 2 by construction
        assert result.reserve_factor == pytest.approx(2.0, rel=1e-10)
        # FI must remain < 1 (still in the safe regime).  Crucially, the
        # naive ``1/FI`` would give ~4.2, not 2.0; this is the bug.
        assert result.index < 1.0
        assert result.reserve_factor != pytest.approx(1.0 / result.index, rel=1e-3)

    def test_rf_matches_closed_form_quadratic_root(self, criterion, material):
        """For an arbitrary mixed compression state, RF matches the
        positive root of ``A*R^2 + B*R = 1`` computed independently.
        """
        # A state where matrix_compression dominates.  Use uniaxial s22
        # with a small tau_23 contribution that does not flip the cross
        # coupling term ``-s22*s33`` sign (s33 = 0 here).  The Hashin
        # matrix-compression FI^2 = A+B can dip below zero when |s22|/Yc
        # is small (since the linear term has negative slope vs sig_t<0
        # before the quadratic dominates), so we pick a state where the
        # mode is unambiguously dominant.
        s22 = -0.7 * material.Yc
        s33 = 0.0
        tau_23 = 0.1 * material.S23
        tau_12 = 0.0
        stress = np.array([0.0, s22, s33, tau_23, 0.0, tau_12])

        result = criterion.evaluate(stress, material)
        assert result.mode == "matrix_compression"

        # Independently compute A, B (mirrors the implementation but kept
        # explicit so the test pins the math, not the code path).
        sig_t = s22 + s33
        A = (
            sig_t ** 2 / (4 * material.S23 ** 2)
            + (tau_23 ** 2 - s22 * s33) / material.S23 ** 2
            + tau_12 ** 2 / material.S12 ** 2
        )
        B = ((material.Yc / (2 * material.S23)) ** 2 - 1) * sig_t / material.Yc
        # Positive root of A*R^2 + B*R - 1 = 0
        r_expected = (-B + np.sqrt(B * B + 4 * A)) / (2 * A)

        assert result.reserve_factor == pytest.approx(r_expected, rel=1e-10)
        # FI at the scaled state should be 1 to high precision
        scaled = stress * result.reserve_factor
        scaled_result = criterion.evaluate(scaled, material)
        assert scaled_result.index == pytest.approx(1.0, rel=1e-8)


class TestHashinPureQuadraticModesUnchanged:
    """The other three modes are pure quadratics; RF = 1/sqrt(FI^2) = 1/FI."""

    def test_matrix_tension_rf_is_inverse_of_index(self, criterion, material):
        """s22 = Yt/2 => RF = 2 = 1/index (matrix_tension is a pure quadratic)."""
        stress = np.array([0.0, 0.5 * material.Yt, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.mode == "matrix_tension"
        assert result.reserve_factor == pytest.approx(2.0, rel=1e-10)
        assert result.reserve_factor == pytest.approx(1.0 / result.index, rel=1e-12)

    def test_fiber_tension_rf_is_inverse_of_index(self, criterion, material):
        """s11 = Xt/2 => RF = 2 = 1/index (fiber_tension is a pure quadratic)."""
        stress = np.array([0.5 * material.Xt, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.mode == "fiber_tension"
        assert result.reserve_factor == pytest.approx(2.0, rel=1e-10)
        assert result.reserve_factor == pytest.approx(1.0 / result.index, rel=1e-12)

    def test_fiber_compression_rf_is_inverse_of_index(self, criterion, material):
        """s11 = -Xc/2 => RF = 2 = 1/index (fiber_compression is linear in |s11|)."""
        stress = np.array([-0.5 * material.Xc, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.mode == "fiber_compression"
        assert result.reserve_factor == pytest.approx(2.0, rel=1e-10)
        assert result.reserve_factor == pytest.approx(1.0 / result.index, rel=1e-12)


class TestHashinMaxStressFPFAgreement:
    """For purely uniaxial transverse compression sweeps, Hashin's
    matrix_compression RF and Max-Stress's matrix_compression RF must
    agree on the first-ply-failure load (both predict R_f = Yc / |s22|).
    """

    def test_uniaxial_s22_compression_sweep(self, criterion, material):
        max_stress = MaxStressCriterion()
        # Sweep through fractions of Yc.  For low |s22|/Yc the Hashin
        # FI_mc^2 polynomial dips below zero (the linear term outweighs
        # the quadratic at small load), so matrix_compression is not the
        # nominal dominant mode at the *applied* load.  We restrict the
        # sweep to states where matrix_compression dominates, which is
        # the regime where the FPF prediction is meaningful.
        for k in (0.5, 0.75, 0.9, 1.0, 1.25):
            s22 = -k * material.Yc
            stress = np.array([0.0, s22, 0.0, 0.0, 0.0, 0.0])

            r_hashin = criterion.evaluate(stress, material)
            r_maxstr = max_stress.evaluate(stress, material)

            assert r_hashin.mode == "matrix_compression"
            assert r_maxstr.mode == "matrix_compression"
            # Both must agree on the strength ratio (the true FPF load)
            assert r_hashin.reserve_factor == pytest.approx(
                r_maxstr.reserve_factor, rel=1e-10
            ), f"Disagreement at k={k}"
            # And that strength ratio should be 1/k by construction
            assert r_hashin.reserve_factor == pytest.approx(1.0 / k, rel=1e-10)
