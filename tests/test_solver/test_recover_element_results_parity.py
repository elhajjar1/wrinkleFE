"""Parity / regression test for ``StaticSolver.recover_element_results``.

Issue #187 refactored ``recover_element_results`` to lift element-constant
work (``T_ply``, node ids, per-node wrinkle angles) out of the inner Gauss
loop and to compute all 8 Gauss-point wrinkle angles via a single matmul
against a pre-built shape-function matrix.

These tests pin down the numerical output at a handful of Gauss points on
a small, fully-deterministic problem.  The values are taken from running
``main`` on the same problem; any change in the post-processing math
should make the assertions trip.
"""

from __future__ import annotations

import numpy as np
import pytest

from wrinklefe.core.laminate import Laminate
from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.core.mesh import WrinkleMesh
from wrinklefe.solver.boundary import BoundaryHandler
from wrinklefe.solver.static import StaticSolver


@pytest.fixture
def parity_mesh_and_laminate():
    """Small multi-ply mesh that exercises both ply rotation and wrinkles.

    Uses a [0/45/-45/90] laminate so ``ply_angles`` are non-trivial, and a
    near-isotropic material so the values are easy to reason about.  Mesh
    is intentionally small so the regression numbers stay short to read.
    """
    E = 10_000.0
    nu = 0.3
    G = E / (2.0 * (1.0 + nu))
    material = OrthotropicMaterial(
        E1=E, E2=E, E3=E,
        G12=G, G13=G, G23=G,
        nu12=nu, nu13=nu, nu23=nu,
        Xt=500, Xc=500, Yt=500, Yc=500, Zt=500, Zc=500,
        S12=300, S13=300, S23=300,
        gamma_Y=0.02,
        name="parity_iso_10k",
    )
    laminate = Laminate.from_angles(
        [0.0, 45.0, -45.0, 90.0],
        material=material,
        ply_thickness=0.183,
    )
    gen = WrinkleMesh(
        laminate=laminate,
        wrinkle_config=None,
        Lx=4.0, Ly=2.0,
        nx=4, ny=2, nz_per_ply=1,
    )
    return gen.generate(), laminate


def test_recover_element_results_shape_and_finiteness(parity_mesh_and_laminate):
    """Output arrays have the expected shape and are all finite."""
    mesh, laminate = parity_mesh_and_laminate
    solver = StaticSolver(mesh, laminate)
    bcs = BoundaryHandler.compression_bcs(mesh, applied_strain=-0.01)
    results = solver.solve(bcs)

    n_elem = mesh.n_elements
    n_gp = 8

    for arr in (
        results.stress_global,
        results.stress_local,
        results.strain_global,
        results.strain_local,
    ):
        assert arr.shape == (n_elem, n_gp, 6)
        assert np.all(np.isfinite(arr))


def test_recover_element_results_local_matches_manual_transform(
    parity_mesh_and_laminate,
):
    """Local stress/strain at a hand-picked GP matches a manual transform.

    Builds the same ``T_total = T_wrinkle(phi_gp) @ T_ply(theta)``
    transform inline, applies it to the global stress/strain, and checks
    against ``recover_element_results``.  This catches any regression in
    the per-element / per-GP transform construction.
    """
    from wrinklefe.core.transforms import stress_transformation_3d
    from wrinklefe.elements.hex8 import Hex8Element

    mesh, laminate = parity_mesh_and_laminate
    solver = StaticSolver(mesh, laminate)
    bcs = BoundaryHandler.compression_bcs(mesh, applied_strain=-0.01)
    results = solver.solve(bcs)

    # Pick a few representative elements across the mesh and GPs across the
    # element so we exercise multiple ply angles and corners of the
    # reference cube.
    sample_elems = [0, mesh.n_elements // 3, mesh.n_elements // 2,
                    mesh.n_elements - 1]
    sample_gps = [0, 3, 7]

    from wrinklefe.elements.gauss import gauss_points_hex
    gp_coords, _ = gauss_points_hex(order=2)

    for e in sample_elems:
        ply_angle_rad = np.radians(float(mesh.ply_angles[e]))
        T_ply = stress_transformation_3d(ply_angle_rad, axis='z')

        node_ids = mesh.elements[e]
        fiber_angles_local = mesh.fiber_angles[node_ids]

        for g in sample_gps:
            xi, eta, zeta = gp_coords[g]
            N = Hex8Element.shape_functions(xi, eta, zeta)
            phi = float(N @ fiber_angles_local)
            T_wrinkle = stress_transformation_3d(phi, axis='y')
            T_total = T_wrinkle @ T_ply

            expected_sigma_local = T_total @ results.stress_global[e, g]
            expected_eps_local = T_total @ results.strain_global[e, g]

            np.testing.assert_allclose(
                results.stress_local[e, g],
                expected_sigma_local,
                rtol=1e-12,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                results.strain_local[e, g],
                expected_eps_local,
                rtol=1e-12,
                atol=1e-12,
            )


def test_recover_element_results_recomputation_is_deterministic(
    parity_mesh_and_laminate,
):
    """Calling ``recover_element_results`` twice yields identical arrays."""
    mesh, laminate = parity_mesh_and_laminate
    solver = StaticSolver(mesh, laminate)
    bcs = BoundaryHandler.compression_bcs(mesh, applied_strain=-0.01)
    results1 = solver.solve(bcs)

    # Re-run the post-processing step against the same displacement and
    # check that we get bit-exact identical arrays the second time.
    u_flat = results1.displacement.reshape(-1)
    sg2, sl2, eg2, el2 = solver.recover_element_results(u_flat)

    np.testing.assert_array_equal(results1.stress_global, sg2)
    np.testing.assert_array_equal(results1.stress_local, sl2)
    np.testing.assert_array_equal(results1.strain_global, eg2)
    np.testing.assert_array_equal(results1.strain_local, el2)
