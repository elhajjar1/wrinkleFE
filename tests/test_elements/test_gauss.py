"""Tests for Gauss-Legendre quadrature module (wrinklefe.elements.gauss).

Covers 1-D rules (n=1,2,3), 3-D hexahedral tensor-product rules,
weight-sum invariants, and invalid-order error handling.
"""

import numpy as np
import pytest

from wrinklefe.elements.gauss import gauss_points_1d, gauss_points_hex


# ======================================================================
# gauss_points_1d
# ======================================================================

class TestGaussPoints1D:
    """Tests for the 1-D Gauss-Legendre quadrature rule."""

    def test_order_1_point(self):
        """n=1: single point at xi=0."""
        pts, wts = gauss_points_1d(1)
        assert pts.shape == (1,)
        assert np.isclose(pts[0], 0.0)

    def test_order_1_weight(self):
        """n=1: weight = 2 (length of [-1,1])."""
        pts, wts = gauss_points_1d(1)
        assert wts.shape == (1,)
        assert np.isclose(wts[0], 2.0)

    def test_order_2_points(self):
        """n=2: points at +/-1/sqrt(3)."""
        pts, wts = gauss_points_1d(2)
        assert pts.shape == (2,)
        g = 1.0 / np.sqrt(3.0)
        assert np.isclose(pts[0], -g)
        assert np.isclose(pts[1], g)

    def test_order_2_weights(self):
        """n=2: both weights equal to 1."""
        pts, wts = gauss_points_1d(2)
        assert wts.shape == (2,)
        np.testing.assert_allclose(wts, [1.0, 1.0])

    def test_order_3_points_shape(self):
        """n=3: three quadrature points."""
        pts, wts = gauss_points_1d(3)
        assert pts.shape == (3,)
        assert wts.shape == (3,)

    def test_order_3_weights_sum(self):
        """n=3: weights sum to 2 (length of [-1,1])."""
        pts, wts = gauss_points_1d(3)
        assert np.isclose(wts.sum(), 2.0)

    def test_order_3_points_sorted(self):
        """n=3: points are sorted ascending."""
        pts, wts = gauss_points_1d(3)
        assert pts[0] < pts[1] < pts[2]

    def test_order_3_symmetry(self):
        """n=3: points are symmetric about 0, middle point at 0."""
        pts, wts = gauss_points_1d(3)
        assert np.isclose(pts[1], 0.0)
        assert np.isclose(pts[0], -pts[2])

    def test_order_3_weights_symmetric(self):
        """n=3: end weights equal, center weight larger."""
        pts, wts = gauss_points_1d(3)
        assert np.isclose(wts[0], wts[2])
        assert wts[1] > wts[0]

    def test_invalid_order_zero(self):
        """n=0 raises ValueError."""
        with pytest.raises(ValueError, match="n=0"):
            gauss_points_1d(0)

    def test_invalid_order_four(self):
        """n=4 raises ValueError."""
        with pytest.raises(ValueError, match="n=4"):
            gauss_points_1d(4)

    def test_invalid_order_negative(self):
        """Negative order raises ValueError."""
        with pytest.raises(ValueError):
            gauss_points_1d(-1)

    def test_exact_integration_constant(self):
        """1-point rule integrates f(x)=1 exactly over [-1,1]."""
        pts, wts = gauss_points_1d(1)
        result = np.sum(wts * 1.0)
        assert np.isclose(result, 2.0)

    def test_exact_integration_linear(self):
        """1-point rule integrates f(x)=x exactly (integral = 0)."""
        pts, wts = gauss_points_1d(1)
        result = np.sum(wts * pts)
        assert np.isclose(result, 0.0)

    def test_exact_integration_quadratic(self):
        """2-point rule integrates f(x)=x^2 exactly (integral = 2/3)."""
        pts, wts = gauss_points_1d(2)
        result = np.sum(wts * pts**2)
        assert np.isclose(result, 2.0 / 3.0)

    def test_exact_integration_cubic(self):
        """2-point rule integrates f(x)=x^3 exactly (integral = 0)."""
        pts, wts = gauss_points_1d(2)
        result = np.sum(wts * pts**3)
        assert np.isclose(result, 0.0, atol=1e-15)

    def test_exact_integration_quartic(self):
        """3-point rule integrates f(x)=x^4 exactly (integral = 2/5)."""
        pts, wts = gauss_points_1d(3)
        result = np.sum(wts * pts**4)
        assert np.isclose(result, 2.0 / 5.0)


# ======================================================================
# gauss_points_hex
# ======================================================================

class TestGaussPointsHex:
    """Tests for the 3-D hexahedral tensor-product quadrature rule."""

    def test_order_2_shapes(self):
        """2x2x2 rule: 8 points with shape (8,3) and (8,)."""
        pts, wts = gauss_points_hex(order=2)
        assert pts.shape == (8, 3)
        assert wts.shape == (8,)

    def test_order_3_shapes(self):
        """3x3x3 rule: 27 points."""
        pts, wts = gauss_points_hex(order=3)
        assert pts.shape == (27, 3)
        assert wts.shape == (27,)

    def test_order_1_shapes(self):
        """1x1x1 rule: 1 point."""
        pts, wts = gauss_points_hex(order=1)
        assert pts.shape == (1, 3)
        assert wts.shape == (1,)

    def test_order_2_weights_sum(self):
        """2x2x2 weights sum to 8 (volume of reference cube [-1,1]^3)."""
        pts, wts = gauss_points_hex(order=2)
        assert np.isclose(wts.sum(), 8.0)

    def test_order_3_weights_sum(self):
        """3x3x3 weights sum to 8."""
        pts, wts = gauss_points_hex(order=3)
        assert np.isclose(wts.sum(), 8.0)

    def test_order_1_weights_sum(self):
        """1x1x1 weights sum to 8."""
        pts, wts = gauss_points_hex(order=1)
        assert np.isclose(wts.sum(), 8.0)

    def test_weights_product_structure(self):
        """3-D weights are tensor products of 1-D weights (2^3=8 for order=2)."""
        pts_1d, wts_1d = gauss_points_1d(2)
        pts_3d, wts_3d = gauss_points_hex(order=2)
        # Each 3-D weight should be a product of three 1-D weights
        assert np.isclose(wts_1d.sum() ** 3, wts_3d.sum())

    def test_all_points_in_reference_cube(self):
        """All Gauss points lie within [-1,1]^3."""
        for order in [1, 2, 3]:
            pts, _ = gauss_points_hex(order=order)
            assert np.all(pts >= -1.0)
            assert np.all(pts <= 1.0)

    def test_default_order_is_2(self):
        """Default order parameter is 2."""
        pts_default, wts_default = gauss_points_hex()
        pts_explicit, wts_explicit = gauss_points_hex(order=2)
        np.testing.assert_array_equal(pts_default, pts_explicit)
        np.testing.assert_array_equal(wts_default, wts_explicit)

    def test_all_weights_positive(self):
        """All quadrature weights are strictly positive."""
        for order in [1, 2, 3]:
            _, wts = gauss_points_hex(order=order)
            assert np.all(wts > 0)

    def test_invalid_order_hex(self):
        """Invalid order raises ValueError for hex rule."""
        with pytest.raises(ValueError):
            gauss_points_hex(order=4)

    def test_integrate_constant_over_hex(self):
        """Integrating f=1 over the reference cube gives volume = 8."""
        pts, wts = gauss_points_hex(order=2)
        result = np.sum(wts)
        assert np.isclose(result, 8.0)

    def test_integrate_linear_over_hex(self):
        """Integrating f=x over the reference cube gives 0 by symmetry."""
        pts, wts = gauss_points_hex(order=2)
        result = np.sum(wts * pts[:, 0])
        assert np.isclose(result, 0.0, atol=1e-14)
