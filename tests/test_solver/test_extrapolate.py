"""Regression tests for hex8 Gauss-point to nodal extrapolation (issue #51).

The two orderings at play are:

- ``gauss_points_hex(order=2)`` -> lexicographic in (xi, eta, zeta), with
  zeta varying fastest (because of ``np.meshgrid(..., indexing="ij")``
  followed by ``ravel``).
- ``Hex8Element._NODE_COORDS`` -> VTK / Abaqus convention: bottom face
  CCW (zeta=-1) followed by top face CCW (zeta=+1).

The bug guarded against here is the silent assumption that "GP i pairs
with node i" — that is wrong, and gives a spatially-permuted nodal field
that is hard to detect after the fact.  These tests pin the correct
nearest-neighbour mapping and verify exact recovery of a tri-linear
field.
"""

from __future__ import annotations

import numpy as np
import pytest

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.core.laminate import Laminate
from wrinklefe.core.mesh import WrinkleMesh
from wrinklefe.elements.gauss import gauss_points_hex
from wrinklefe.elements.hex8 import _NODE_COORDS
from wrinklefe.solver.static import StaticSolver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_elem_solver():
    """A StaticSolver wrapping a single-hex8 mesh — only needed so that
    :meth:`StaticSolver._extrapolate_to_nodes` (an instance method) can be
    invoked.  The mesh itself is incidental to the extrapolation test.
    """
    material = OrthotropicMaterial()
    laminate = Laminate.from_angles([0.0], material=material, ply_thickness=1.0)
    mesh = WrinkleMesh(
        laminate=laminate,
        wrinkle_config=None,
        Lx=1.0, Ly=1.0,
        nx=1, ny=1, nz_per_ply=1,
    ).generate()
    return StaticSolver(mesh=mesh, laminate=laminate)


# ---------------------------------------------------------------------------
# Pinned ordering / mapping tests
# ---------------------------------------------------------------------------

class TestHex8Orderings:
    """Sanity-check the two orderings the extrapolation depends on."""

    def test_gauss_points_are_lex(self):
        """``gauss_points_hex(2)`` returns GPs in lex order with zeta fastest.

        This is the property that ``_extrapolate_to_nodes`` depends on; if
        it ever changes, the extrapolation matrix must be rebuilt.
        """
        gp_coords, _ = gauss_points_hex(order=2)
        g = 1.0 / np.sqrt(3.0)
        expected = np.array(
            [
                [-g, -g, -g],
                [-g, -g, +g],
                [-g, +g, -g],
                [-g, +g, +g],
                [+g, -g, -g],
                [+g, -g, +g],
                [+g, +g, -g],
                [+g, +g, +g],
            ]
        )
        np.testing.assert_allclose(gp_coords, expected, atol=1e-12)

    def test_node_coords_are_vtk(self):
        """``_NODE_COORDS`` follows the VTK / Abaqus hex8 convention."""
        expected = np.array(
            [
                [-1.0, -1.0, -1.0],
                [+1.0, -1.0, -1.0],
                [+1.0, +1.0, -1.0],
                [-1.0, +1.0, -1.0],
                [-1.0, -1.0, +1.0],
                [+1.0, -1.0, +1.0],
                [+1.0, +1.0, +1.0],
                [-1.0, +1.0, +1.0],
            ]
        )
        np.testing.assert_array_equal(_NODE_COORDS, expected)

    def test_node_to_gp_mapping_is_pinned(self):
        """The VTK-node -> lex-GP nearest mapping has a fixed permutation.

        Hard-coding the expected permutation guards against either
        ordering convention drifting silently.
        """
        _N_inv, node_to_gp = StaticSolver._build_hex8_extrapolation()
        # Expected pairing (each VTK node paired to the lex GP at the
        # same sign(xi, eta, zeta)):
        #   Node 0 (-,-,-) <-> GP 0 (-,-,-)
        #   Node 1 (+,-,-) <-> GP 4 (+,-,-)
        #   Node 2 (+,+,-) <-> GP 6 (+,+,-)
        #   Node 3 (-,+,-) <-> GP 2 (-,+,-)
        #   Node 4 (-,-,+) <-> GP 1 (-,-,+)
        #   Node 5 (+,-,+) <-> GP 5 (+,-,+)
        #   Node 6 (+,+,+) <-> GP 7 (+,+,+)
        #   Node 7 (-,+,+) <-> GP 3 (-,+,+)
        np.testing.assert_array_equal(
            node_to_gp, np.array([0, 4, 6, 2, 1, 5, 7, 3])
        )

    def test_node_to_gp_is_permutation(self):
        """The mapping must be a bijection across all 8 nodes/GPs."""
        _N_inv, node_to_gp = StaticSolver._build_hex8_extrapolation()
        assert sorted(node_to_gp.tolist()) == list(range(8))


# ---------------------------------------------------------------------------
# Extrapolation correctness
# ---------------------------------------------------------------------------

class TestExtrapolateToNodes:
    """Exercise :meth:`StaticSolver._extrapolate_to_nodes`."""

    def test_constant_field_recovers_exactly(self, single_elem_solver):
        """A constant Gauss-point field extrapolates to the same constant."""
        gp_values = np.full((8, 6), 3.14)
        nodal = single_elem_solver._extrapolate_to_nodes(gp_values)
        np.testing.assert_allclose(nodal, 3.14, atol=1e-12)

    def test_trilinear_field_recovers_exactly(self, single_elem_solver):
        """A tri-linear nodal field is recovered exactly after a round trip.

        Build a known nodal field, sample it at the lex Gauss points, then
        run the extrapolator and confirm we get the original nodal field
        back — verifying that GP-order / node-order alignment is correct
        end-to-end (issue #51).
        """
        from wrinklefe.elements.hex8 import Hex8Element

        gp_coords, _ = gauss_points_hex(order=2)
        # Distinct value per VTK node (non-uniform, asymmetric):
        f_nodes_true = np.arange(1.0, 9.0)  # [1, 2, ..., 8] in VTK order

        # Sample the tri-linear field at each lex Gauss point.
        f_gp = np.empty(8)
        for i in range(8):
            xi, eta, zeta = gp_coords[i]
            N = Hex8Element.shape_functions(xi, eta, zeta)  # cols = VTK nodes
            f_gp[i] = float(N @ f_nodes_true)

        f_nodes_back = single_elem_solver._extrapolate_to_nodes(f_gp)
        np.testing.assert_allclose(f_nodes_back, f_nodes_true, atol=1e-10)

    def test_gp_indicator_dominates_nearest_node(self, single_elem_solver):
        """A "delta" GP field — one GP at 1, others at 0 — yields the
        largest extrapolated nodal value at the **nearest** VTK node.

        The 2x2x2 extrapolation matrix is a true tri-linear extrapolant
        (it amplifies outward from the GP grid to the element corners),
        so the nodal value at the nearest corner is positive and larger
        in magnitude than at any other node — which is exactly the
        guarantee a caller relying on the node<->GP pairing needs.

        Wrong pairing (issue #51) would put the maximum at a different,
        spatially incorrect node.
        """
        _N_inv, node_to_gp = StaticSolver._build_hex8_extrapolation()
        for gp_idx in range(8):
            f_gp = np.zeros(8)
            f_gp[gp_idx] = 1.0
            nodal = single_elem_solver._extrapolate_to_nodes(f_gp)
            # The VTK node nearest to this GP must have the maximum
            # extrapolated value.
            nearest_node = int(np.flatnonzero(node_to_gp == gp_idx)[0])
            assert int(np.argmax(nodal)) == nearest_node, (
                f"GP {gp_idx}: expected max at node {nearest_node}, "
                f"got max at node {int(np.argmax(nodal))}; nodal={nodal}"
            )

    def test_input_shape_1d(self, single_elem_solver):
        """1-D input of shape (8,) returns 1-D output of shape (8,)."""
        f_gp = np.linspace(0.0, 1.0, 8)
        out = single_elem_solver._extrapolate_to_nodes(f_gp)
        assert out.shape == (8,)

    def test_input_shape_2d(self, single_elem_solver):
        """2-D input of shape (8, k) returns 2-D output of shape (8, k)."""
        f_gp = np.random.default_rng(0).standard_normal((8, 6))
        out = single_elem_solver._extrapolate_to_nodes(f_gp)
        assert out.shape == (8, 6)

    def test_wrong_input_length_raises(self, single_elem_solver):
        """Inputs with the wrong number of Gauss points are rejected."""
        with pytest.raises(ValueError, match="8 rows"):
            single_elem_solver._extrapolate_to_nodes(np.zeros((4, 6)))
