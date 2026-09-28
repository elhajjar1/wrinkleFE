"""Equivalence guards for the FE hot-path optimisations.

Every change these cover makes the same computation cheaper, never
different: caching pure functions, hoisting a loop-invariant rotation,
and reusing a Jacobian instead of rebuilding it. This project pins a
validation ledger to exact values, so "close enough" is not the bar —
each test below recomputes the result the slow way and demands a
**bit-identical** match, not an approximate one.

Two of the caches hand the same array to every caller. Those are
returned read-only on purpose: a shared buffer that something writes to
in place corrupts every later element silently, whereas a read-only one
fails loudly at the write. The tests pin that too.
"""

from __future__ import annotations

import numpy as np
import pytest

from wrinklefe.core.material import MaterialLibrary
from wrinklefe.core.transforms import (
    rotate_stiffness_3d,
    strain_transformation_3d,
    stress_transformation_3d,
)
from wrinklefe.elements.gauss import gauss_points_hex
from wrinklefe.elements.hex8 import Hex8Element, _b_from_dN_dx

# A deliberately irregular (non-cube) element: a unit cube makes the
# Jacobian constant and the identity trivial, which would hide exactly
# the kind of mistake these tests exist to catch.
_SKEWED_NODES = np.array([
    [0.00, 0.00, 0.00], [2.10, 0.05, -0.10], [2.30, 1.40, 0.10],
    [0.15, 1.30, -0.05], [-0.05, 0.10, 0.90], [2.00, -0.05, 1.10],
    [2.20, 1.35, 1.05], [0.10, 1.45, 0.95],
])


@pytest.fixture
def skewed_element():
    return Hex8Element(
        node_coords=_SKEWED_NODES,
        material=MaterialLibrary().get("IM7_8552"),
        ply_angle=37.0,
        wrinkle_angles=np.array(
            [0.02, 0.05, -0.03, 0.01, 0.04, -0.02, 0.06, 0.00]
        ),
    )


# ----------------------------------------------------------------------
# Shape-function caches
# ----------------------------------------------------------------------

class TestShapeFunctionCache:
    """Cached, but still a pure function of the natural coordinates."""

    @staticmethod
    def _n_ref(xi, eta, zeta):
        from wrinklefe.elements.hex8 import _NODE_COORDS
        N = np.empty(8)
        for i in range(8):
            N[i] = (
                0.125
                * (1.0 + _NODE_COORDS[i, 0] * xi)
                * (1.0 + _NODE_COORDS[i, 1] * eta)
                * (1.0 + _NODE_COORDS[i, 2] * zeta)
            )
        return N

    @staticmethod
    def _dn_ref(xi, eta, zeta):
        from wrinklefe.elements.hex8 import _NODE_COORDS
        dN = np.empty((3, 8))
        for j in range(8):
            xj, ej, zj = _NODE_COORDS[j]
            dN[0, j] = 0.125 * xj * (1.0 + ej * eta) * (1.0 + zj * zeta)
            dN[1, j] = 0.125 * (1.0 + xj * xi) * ej * (1.0 + zj * zeta)
            dN[2, j] = 0.125 * (1.0 + xj * xi) * (1.0 + ej * eta) * zj
        return dN

    @pytest.mark.parametrize(
        "pt",
        [(0.0, 0.0, 0.0), (0.3, -0.5, 0.7), (-1.0, -1.0, -1.0),
         (1.0, 1.0, 1.0), (-0.577350269189626, 0.577350269189626, -0.1)],
    )
    def test_shape_functions_bit_identical_to_direct_evaluation(self, pt):
        assert np.array_equal(Hex8Element.shape_functions(*pt), self._n_ref(*pt))

    @pytest.mark.parametrize(
        "pt",
        [(0.0, 0.0, 0.0), (0.3, -0.5, 0.7), (-1.0, -1.0, -1.0),
         (1.0, 1.0, 1.0), (-0.577350269189626, 0.577350269189626, -0.1)],
    )
    def test_shape_derivatives_bit_identical_to_direct_evaluation(self, pt):
        assert np.array_equal(
            Hex8Element.shape_derivatives(*pt), self._dn_ref(*pt)
        )

    def test_distinct_points_are_not_conflated(self):
        """The obvious way a memo goes wrong: a key collision."""
        seen = {}
        rng = np.random.default_rng(0)
        for _ in range(200):
            pt = tuple(rng.uniform(-1.0, 1.0, 3))
            seen[pt] = Hex8Element.shape_functions(*pt)
        for pt, cached in seen.items():
            assert np.array_equal(cached, self._n_ref(*pt))

    def test_repeat_calls_return_equal_values(self):
        a = Hex8Element.shape_derivatives(0.11, -0.22, 0.33)
        b = Hex8Element.shape_derivatives(0.11, -0.22, 0.33)
        assert np.array_equal(a, b)

    @pytest.mark.parametrize(
        "fn", [Hex8Element.shape_functions, Hex8Element.shape_derivatives],
    )
    def test_cached_arrays_are_read_only(self, fn):
        """Shared between every caller, so a write must fail, not spread."""
        out = fn(0.25, -0.25, 0.5)
        with pytest.raises(ValueError):
            out[...] = 0.0

    def test_integer_and_float_arguments_agree(self):
        """``0`` and ``0.0`` are the same point and must not split the memo."""
        assert np.array_equal(
            Hex8Element.shape_functions(0, 0, 0),
            Hex8Element.shape_functions(0.0, 0.0, 0.0),
        )


class TestGaussRuleCache:
    def test_repeat_calls_return_equal_rules(self):
        p1, w1 = gauss_points_hex(order=2)
        p2, w2 = gauss_points_hex(order=2)
        assert np.array_equal(p1, p2) and np.array_equal(w1, w2)

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_orders_are_not_conflated(self, order):
        pts, wts = gauss_points_hex(order=order)
        assert pts.shape == (order ** 3, 3)
        assert wts.shape == (order ** 3,)
        # Weights of the reference cube always sum to its volume.
        assert np.isclose(wts.sum(), 8.0)

    def test_rule_arrays_are_read_only(self):
        """Every element shares this rule; an in-place edit would change
        the quadrature of every element built afterwards."""
        pts, wts = gauss_points_hex(order=2)
        with pytest.raises(ValueError):
            pts[0, 0] = 0.0
        with pytest.raises(ValueError):
            wts[0] = 0.0


# ----------------------------------------------------------------------
# Transform identities
# ----------------------------------------------------------------------

class TestTransformEquivalence:
    @pytest.mark.parametrize("axis", ["y", "z"])
    def test_strain_transformation_matches_the_reuter_triple_product(
        self, axis
    ):
        """Hoisting the Reuter matrices out of the call must change nothing."""
        R = np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        R_inv = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
        for angle in np.linspace(-np.pi, np.pi, 401):
            expected = R @ stress_transformation_3d(angle, axis=axis) @ R_inv
            assert np.array_equal(
                strain_transformation_3d(angle, axis=axis), expected
            )

    @pytest.mark.parametrize("axis", ["y", "z"])
    def test_rotate_stiffness_matches_the_two_transform_formulation(self, axis):
        """``rotate_stiffness_3d`` derives T_epsilon from the T_sigma it
        already built instead of calling ``strain_transformation_3d``,
        which would construct a second identical T_sigma."""
        C = MaterialLibrary().get("IM7_8552").stiffness_matrix
        for angle in np.linspace(-np.pi, np.pi, 401):
            T_sigma = stress_transformation_3d(angle, axis=axis)
            T_eps = strain_transformation_3d(angle, axis=axis)
            expected = np.linalg.inv(T_sigma) @ C @ T_eps
            assert np.array_equal(
                rotate_stiffness_3d(C, angle, axis=axis), expected
            )


# ----------------------------------------------------------------------
# Element stiffness
# ----------------------------------------------------------------------

class TestStiffnessMatrixEquivalence:
    def test_ke_bit_identical_to_the_b_matrix_formulation(
        self, skewed_element
    ):
        """``stiffness_matrix`` builds B from the Jacobian it just formed
        rather than calling ``B_matrix`` (which would redo the shape
        derivatives, the Jacobian and its determinant). Recompute the old
        way and demand an exact match."""
        el = skewed_element
        expected = np.zeros((24, 24))
        for gp in range(len(el._gauss_weights)):
            xi, eta, zeta = el._gauss_points[gp]
            w = el._gauss_weights[gp]
            J = el.jacobian(xi, eta, zeta)
            detJ = el._check_detJ(float(np.linalg.det(J)), gp_index=gp)
            B = el.B_matrix(xi, eta, zeta)
            C_bar = el.rotated_stiffness(xi, eta, zeta)
            expected += (B.T @ C_bar @ B) * detJ * w

        assert np.array_equal(el.stiffness_matrix(), expected)

    def test_b_matrix_still_agrees_with_the_shared_assembler(
        self, skewed_element
    ):
        """``B_matrix`` and the integration path share one definition of
        the Voigt row layout; pin that they still produce the same B."""
        el = skewed_element
        xi, eta, zeta = 0.3, -0.4, 0.5
        dN_dxi = el.shape_derivatives(xi, eta, zeta)
        J = dN_dxi @ el.node_coords
        expected = _b_from_dN_dx(np.linalg.inv(J) @ dN_dxi)
        assert np.array_equal(el.B_matrix(xi, eta, zeta), expected)

    def test_ply_rotation_cache_matches_a_fresh_rotation(self, skewed_element):
        """The ply rotation is hoisted out of the Gauss loop because it
        cannot vary within an element. Check the cached value against the
        rotation done from scratch."""
        el = skewed_element
        expected = rotate_stiffness_3d(
            el.material.stiffness_matrix, np.radians(el.ply_angle), axis="z",
        )
        assert np.array_equal(el._ply_rotated_stiffness(), expected)
        # Still the same on the second call — a cache, not a one-shot.
        assert np.array_equal(el._ply_rotated_stiffness(), expected)

    def test_zero_ply_angle_returns_the_material_matrix_unrotated(self):
        el = Hex8Element(
            node_coords=_SKEWED_NODES,
            material=MaterialLibrary().get("IM7_8552"),
            ply_angle=0.0,
            wrinkle_angles=np.zeros(8),
        )
        assert np.array_equal(
            el._ply_rotated_stiffness(), el.material.stiffness_matrix
        )

    def test_elements_with_different_ply_angles_do_not_share_a_cache(self):
        """The cache is per element; two angles must not collide."""
        mat = MaterialLibrary().get("IM7_8552")
        kw = dict(node_coords=_SKEWED_NODES, wrinkle_angles=np.zeros(8))
        a = Hex8Element(material=mat, ply_angle=0.0, **kw)
        b = Hex8Element(material=mat, ply_angle=45.0, **kw)
        assert not np.array_equal(
            a._ply_rotated_stiffness(), b._ply_rotated_stiffness()
        )

    def test_degenerate_element_still_raises_with_the_gauss_point_index(self):
        """The inlined path kept the Gauss-point-aware detJ check, which
        is the informative one (issue #45)."""
        flat = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        ])
        el = Hex8Element(
            node_coords=flat,
            material=MaterialLibrary().get("IM7_8552"),
            ply_angle=0.0,
            wrinkle_angles=np.zeros(8),
        )
        with pytest.raises(ValueError, match="at gauss point 0"):
            el.stiffness_matrix()
