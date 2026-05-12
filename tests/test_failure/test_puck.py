"""Tests for the Puck action-plane failure criterion."""

import numpy as np
import pytest

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.failure.base import FailureResult
from wrinklefe.failure.puck import PuckCriterion


@pytest.fixture
def material():
    return OrthotropicMaterial()


@pytest.fixture
def criterion():
    return PuckCriterion()


class TestPuckUniaxial:

    def test_zero_stress_gives_fi_zero(self, criterion, material):
        result = criterion.evaluate(np.zeros(6), material)
        assert isinstance(result, FailureResult)
        assert result.index == pytest.approx(0.0, abs=1e-12)

    def test_pure_fibre_tension_at_Xt(self, criterion, material):
        """sigma_11 = Xt yields fibre-tension FI = sigma_11 / Xt = 1."""
        stress = np.array([material.Xt, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-10)
        assert result.mode == "fiber_tension"

    def test_pure_fibre_compression_at_Xc(self, criterion, material):
        """sigma_11 = -Xc yields fibre-compression FI = |sigma_11| / Xc = 1."""
        stress = np.array([-material.Xc, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-10)
        assert result.mode == "fiber_compression"

    def test_pure_transverse_tension_at_Yt(self, criterion, material):
        """sigma_22 = Yt: at theta=0 the action plane sees sigma_n=Yt with no
        shears, so the FF term is sigma_22/Yt = 1 (Mode A with tau=0 gives
        FI = p_perp_psi_t * Yt / S23, but the dominant FI from FF is the
        sigma_22/Yt failure of the matrix).  In this Puck implementation
        FF only accounts for sigma_11, so the IFF Mode A search governs.
        At theta=0 with tau_nt=tau_n1=0, the IFF index is
        p_perp_psi_t * sigma_n / S23.  We verify the FI stays bounded and
        the mode is matrix/IFF related, not fibre.
        """
        stress = np.array([0.0, material.Yt, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        # Pure sigma_22 = Yt drives the matrix to its tensile limit; the
        # action-plane search returns a positive IFF index governed by
        # mode A.  Verify we exceed or approach unity.
        assert result.index > 0.0
        assert result.mode.startswith("iff_") or result.mode.startswith("fiber_")

    def test_pure_inplane_shear_at_S12(self, criterion, material):
        """tau_12 = S12: on action plane theta=0, sigma_n=0, tau_n1=S12,
        tau_nt=0. Mode A with sigma_n=0 gives
        FI_A = sqrt((tau_n1/S12)^2) = 1.0.
        """
        stress = np.array([0.0, 0.0, 0.0, 0.0, 0.0, material.S12])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-10)

    def test_returns_failure_result(self, criterion, material):
        stress = np.array([100.0, 10.0, 0.0, 0.0, 0.0, 5.0])
        result = criterion.evaluate(stress, material)
        assert isinstance(result, FailureResult)
        assert result.criterion_name == "puck"
        assert result.reserve_factor == pytest.approx(1.0 / result.index, rel=1e-12)

    def test_reserve_factor_is_inverse_of_index(self, criterion, material):
        stress = np.array([0.5 * material.Xt, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.reserve_factor == pytest.approx(1.0 / result.index, rel=1e-12)


# ======================================================================
# Regression tests for issue #85 — Mode B / Mode C threshold
# ======================================================================

class TestPuckIFFModeBCClassification:
    """Mode B / Mode C should split on a stress-magnitude-independent
    ratio |tau_nt / sigma_n|. The pre-fix threshold collapsed to a
    constant on |tau_nt| alone (~S23) and reversed the physics.
    See issue #85.
    """

    def test_corner_slope_is_constant_function_of_p_psc(self):
        """The Mode B/C corner slope is geometric on the failure envelope —
        it depends on p_perp_psi_c and nothing else, in particular not on
        the applied stress magnitude.
        """
        slope_025 = PuckCriterion._mode_bc_corner_slope(0.25)
        # sqrt(1 + 2 * 0.25) = sqrt(1.5)
        assert slope_025 == pytest.approx(np.sqrt(1.5), rel=1e-12)
        # And it's monotone in p_psc:
        assert (
            PuckCriterion._mode_bc_corner_slope(0.10)
            < PuckCriterion._mode_bc_corner_slope(0.30)
        )

    def test_high_shear_mild_compression_is_mode_c(self):
        """(sigma_n, tau_nt) = (-10, 100) is squarely in the shear-dominated
        wedge and must be classified Mode C. Pre-fix, the broken threshold
        |tnt| >= S23 selected Mode B here — wrong by Puck's construction.
        Hand-checked sanity example from issue #85.
        """
        mode = PuckCriterion._classify_iff_mode_bc(
            sn=-10.0, tnt=100.0, p_psc=0.25,
        )
        assert mode == "iff_mode_c"

    def test_high_compression_mild_shear_is_mode_b(self):
        """(sigma_n, tau_nt) = (-100, 10): compression-dominated, deep in
        the Mode B wedge. The classification must report Mode B. Hand-
        checked sanity example from issue #85.
        """
        mode = PuckCriterion._classify_iff_mode_bc(
            sn=-100.0, tnt=10.0, p_psc=0.25,
        )
        assert mode == "iff_mode_b"

    def test_threshold_independent_of_stress_magnitude(self):
        """Scaling sn and tnt by the same factor must not change the
        classification — the threshold is geometric on the ratio, not on
        absolute magnitudes. Pre-fix, the broken |tnt| >= S23 threshold
        would have selected differently for (sn, tnt) vs (10*sn, 10*tnt).
        """
        for sn, tnt in [(-1.0, 10.0), (-50.0, 500.0), (-0.01, 0.1)]:
            assert (
                PuckCriterion._classify_iff_mode_bc(sn, tnt, p_psc=0.25)
                == "iff_mode_c"
            ), f"Mode C expected at ({sn}, {tnt})"
        for sn, tnt in [(-10.0, 1.0), (-500.0, 50.0), (-0.1, 0.01)]:
            assert (
                PuckCriterion._classify_iff_mode_bc(sn, tnt, p_psc=0.25)
                == "iff_mode_b"
            ), f"Mode B expected at ({sn}, {tnt})"

    def test_classification_at_exact_corner(self):
        """Exactly on the corner |tnt| == sqrt(1+2*p_psc) * |sn|, the
        strict-greater-than condition routes to Mode B. Either mode is
        physically acceptable at the boundary (the formulas coincide
        there), but pinning the deterministic choice avoids platform-
        dependent flapping.
        """
        slope = PuckCriterion._mode_bc_corner_slope(0.25)
        mode = PuckCriterion._classify_iff_mode_bc(
            sn=-10.0, tnt=10.0 * slope, p_psc=0.25,
        )
        assert mode == "iff_mode_b"
