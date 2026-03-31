"""Tests for the confinement model: _confined_fraction() and _effective_gamma_Y().

Validates that the layup-dependent matrix yield strain computation
correctly handles UD, dispersed, quasi-isotropic, all-offaxis, and
single-ply stacking sequences.

References
----------
- Elhajjar (2025) Scientific Reports 15:25977
- Mukhopadhyay et al. (2015) blocked layup calibration
"""

from __future__ import annotations

import pytest

from wrinklefe.analysis import _confined_fraction, _effective_gamma_Y

# Constants from analysis.py
_GAMMA_Y_UD = 0.032
_ALPHA_CONF = 0.050


class TestConfinedFraction:
    """Tests for _confined_fraction()."""

    def test_ud_layup(self):
        """[0]_24 -> all 0-deg plies with 0-deg neighbors -> f_confined ~ 0.0.

        Interior plies have both neighbors at 0-deg, so score = 0.0.
        Only the two surface plies get partial credit (one free surface).
        """
        angles = [0] * 24
        fc = _confined_fraction(angles)
        # Surface plies score 0.5 each (one free surface = off-axis-like),
        # interior plies score 0.0.  fc = (0.5 + 0.5) / 24 ~ 0.042
        assert fc < 0.1
        assert fc >= 0.0

    def test_dispersed_layup(self):
        """[0/90]_12 alternating -> each 0-deg ply has 90-deg neighbors -> f ~ 1.0."""
        angles = [0, 90] * 12  # 24 plies
        fc = _confined_fraction(angles)
        assert fc == pytest.approx(1.0, abs=0.05)

    def test_quasi_isotropic(self):
        """[0/45/-45/90]_3s -> mixed confinement, between UD and fully dispersed."""
        base = [0, 45, -45, 90]
        angles = (base * 3) + list(reversed(base * 3))
        fc = _confined_fraction(angles)
        # Each 0-deg ply is surrounded by off-axis plies -> high confinement
        # but exact value depends on position; should be well above 0 and at most 1
        assert 0.3 < fc <= 1.0

    def test_all_offaxis(self):
        """[45/-45/90]_4s -> no 0-deg plies -> f_confined = 0.0."""
        base = [45, -45, 90]
        angles = (base * 4) + list(reversed(base * 4))
        fc = _confined_fraction(angles)
        assert fc == 0.0

    def test_single_ply_zero(self):
        """[0] -> single ply with two free surfaces -> f_confined = 1.0.

        A lone 0-deg ply has no 0-deg neighbors (both boundaries treated
        as off-axis), so it gets full confinement score.
        """
        angles = [0]
        fc = _confined_fraction(angles)
        assert fc == pytest.approx(1.0)

    def test_single_ply_offaxis(self):
        """[45] -> single off-axis ply -> f_confined = 0.0."""
        angles = [45]
        fc = _confined_fraction(angles)
        assert fc == 0.0

    def test_blocked_zeros(self):
        """[0_4/90_4]_s -> blocked 0-deg plies get reduced confinement.

        Interior 0-deg plies have 0-deg neighbors on both sides -> score 0.
        Only edge plies of the block get partial or full credit.
        """
        angles = [0, 0, 0, 0, 90, 90, 90, 90, 90, 90, 90, 90, 0, 0, 0, 0]
        fc = _confined_fraction(angles)
        # 8 zero-deg plies, most are interior -> low confinement
        assert fc < 0.5

    def test_symmetric_result(self):
        """Confinement should be the same regardless of layup reversal."""
        angles = [0, 45, 0, 90, 0, -45]
        fc_fwd = _confined_fraction(angles)
        fc_rev = _confined_fraction(list(reversed(angles)))
        assert fc_fwd == pytest.approx(fc_rev)


class TestEffectiveGammaY:
    """Tests for _effective_gamma_Y()."""

    def test_ud_gamma_y(self):
        """UD layup -> gamma_Y close to base value (low confinement)."""
        angles = [0] * 24
        gamma = _effective_gamma_Y(angles)
        # UD has very low confinement; gamma should be near _GAMMA_Y_UD
        assert gamma >= _GAMMA_Y_UD
        assert gamma < _GAMMA_Y_UD + 0.01  # small boost from surface plies only

    def test_dispersed_gamma_y(self):
        """Fully dispersed layup -> gamma_Y at upper end."""
        angles = [0, 90] * 12
        gamma = _effective_gamma_Y(angles)
        expected_max = _GAMMA_Y_UD + _ALPHA_CONF * 1.0
        assert gamma == pytest.approx(expected_max, abs=0.005)

    def test_quasi_isotropic_gamma_y(self):
        """Quasi-iso layup -> gamma_Y between UD and fully dispersed."""
        base = [0, 45, -45, 90]
        angles = (base * 3) + list(reversed(base * 3))
        gamma = _effective_gamma_Y(angles)
        assert gamma > _GAMMA_Y_UD + 0.01
        assert gamma <= _GAMMA_Y_UD + _ALPHA_CONF

    def test_all_offaxis_gamma_y(self):
        """No 0-deg plies -> f_confined = 0 -> gamma_Y = base value."""
        angles = [45, -45, 90] * 8
        gamma = _effective_gamma_Y(angles)
        assert gamma == pytest.approx(_GAMMA_Y_UD)

    def test_monotonic_with_confinement(self):
        """More confinement -> higher gamma_Y (monotonically increasing)."""
        # UD (low confinement)
        gamma_ud = _effective_gamma_Y([0] * 24)
        # Mixed
        gamma_mixed = _effective_gamma_Y([0, 0, 90, 90] * 6)
        # Dispersed (high confinement)
        gamma_disp = _effective_gamma_Y([0, 90] * 12)

        assert gamma_ud <= gamma_mixed <= gamma_disp
