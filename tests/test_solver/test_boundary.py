"""Tests for the CLT LoadState -> 3-D boundary-condition conversion.

Covers the two parallel converters identified in issue #96:

- ``BoundaryHandler.load_state_to_bcs`` (``solver/boundary.py``)
- ``StaticSolver._load_state_to_bcs`` (``solver/static.py``), the private
  path actually used by ``StaticSolver.run_clt_load``.

The assertions are physically grounded: total applied face force is
checked through the *consistent face integration* path established by
PR #137 / issue #50 (``BoundaryHandler.get_force_dofs``), so the sum of
nodal forces equals the prescribed CLT resultant.  Rigid-body
suppression and the structural divergence between the two converters are
pinned so future drift is caught.
"""

import numpy as np
import pytest

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.core.laminate import Laminate, LoadState
from wrinklefe.core.mesh import WrinkleMesh
from wrinklefe.solver.boundary import BoundaryHandler
from wrinklefe.solver.static import StaticSolver


# ======================================================================
# Fixtures (mirror tests/test_solver/test_static.py conventions)
# ======================================================================

@pytest.fixture
def x850_material():
    """Default CYCOM X850/T800 material."""
    return OrthotropicMaterial()


@pytest.fixture
def single_ply_laminate(x850_material):
    """Single-ply [0] laminate with 0.183 mm thickness."""
    return Laminate.from_angles([0.0], material=x850_material,
                                ply_thickness=0.183)


@pytest.fixture
def two_ply_laminate(x850_material):
    """Two-ply [0/0] laminate so the mesh has a true midplane node row."""
    return Laminate.from_angles([0.0, 0.0], material=x850_material,
                                ply_thickness=0.183)


@pytest.fixture
def small_mesh(single_ply_laminate):
    """Small 3x2x1 mesh, domain (Lx=3, Ly=2, Lz=0.183)."""
    gen = WrinkleMesh(
        laminate=single_ply_laminate,
        wrinkle_config=None,
        Lx=3.0, Ly=2.0,
        nx=3, ny=2, nz_per_ply=1,
    )
    return gen.generate()


@pytest.fixture
def bending_mesh(two_ply_laminate):
    """3x2x2 mesh (nz=2) so x_max has an exact midplane node (z=0)."""
    gen = WrinkleMesh(
        laminate=two_ply_laminate,
        wrinkle_config=None,
        Lx=3.0, Ly=2.0,
        nx=3, ny=2, nz_per_ply=1,
    )
    return gen.generate()


# ======================================================================
# Helpers
# ======================================================================

def _force_components(mesh, bcs):
    """Return (sum Fx, sum Fy, sum Fz) of the assembled global force."""
    F = BoundaryHandler(mesh).get_force_dofs(bcs)
    return F[0::3].sum(), F[1::3].sum(), F[2::3].sum()


def _bc_kinds(bcs):
    return [b.bc_type for b in bcs]


def _node_fix_bcs(bcs):
    """BCs that pin explicit node_ids (the rigid-body corner fixes)."""
    return [b for b in bcs if b.node_ids is not None and b.bc_type == "fixed"]


# ======================================================================
# Zero load / thermal-only -> no mechanical BCs
# ======================================================================

class TestNoLoad:
    """LoadState with no mechanical resultant maps to an empty BC list."""

    def test_zero_load_returns_empty(self, small_mesh):
        bcs = BoundaryHandler.load_state_to_bcs(LoadState(), small_mesh)
        assert bcs == []

    def test_thermal_only_returns_empty(self, small_mesh):
        """delta_T-only LoadState is mechanical-only here -> []. (#133:
        thermal expansion is handled through CLT, not these BCs.)"""
        bcs = BoundaryHandler.load_state_to_bcs(
            LoadState(delta_T=-100.0), small_mesh
        )
        assert bcs == []

    def test_zero_load_no_applied_force(self, small_mesh):
        bcs = BoundaryHandler.load_state_to_bcs(LoadState(), small_mesh)
        fx, fy, fz = _force_components(small_mesh, bcs)
        assert (fx, fy, fz) == (0.0, 0.0, 0.0)


# ======================================================================
# Pure Nx
# ======================================================================

class TestPureNx:
    """Uniaxial Nx -> pressure on x_max, light rigid-body suppression."""

    def test_pressure_bc_on_xmax(self, small_mesh):
        Lx, Ly, Lz = small_mesh.domain_size
        bcs = BoundaryHandler.load_state_to_bcs(LoadState(Nx=100.0), small_mesh)
        press = [b for b in bcs if b.bc_type == "pressure"]
        assert len(press) == 1
        bc = press[0]
        assert bc.face == "x_max"
        assert bc.dofs == [0]
        # CLT Nx is force/width: total face force = Nx * Ly.
        assert bc.value == pytest.approx(100.0 * Ly)

    def test_total_force_equals_resultant(self, small_mesh):
        """Consistent face integration (#137): sum of nodal forces == Nx*Ly."""
        Lx, Ly, Lz = small_mesh.domain_size
        bcs = BoundaryHandler.load_state_to_bcs(LoadState(Nx=100.0), small_mesh)
        fx, fy, fz = _force_components(small_mesh, bcs)
        assert fx == pytest.approx(100.0 * Ly)
        # No spurious force in unloaded directions.
        assert fy == pytest.approx(0.0)
        assert fz == pytest.approx(0.0)

    def test_rigidbody_suppression_not_overconstrained(self, small_mesh):
        """x_min fixed only in x (not fully clamped); y_min symmetry;
        exactly one corner node pins the remaining y/z rigid modes."""
        bcs = BoundaryHandler.load_state_to_bcs(LoadState(Nx=100.0), small_mesh)
        kinds = _bc_kinds(bcs)
        assert "fixed" in kinds and "symmetry_y" in kinds
        xmin_fix = [b for b in bcs
                    if b.bc_type == "fixed" and b.face == "x_min"]
        assert len(xmin_fix) == 1
        # Only ux constrained on x_min -> y/z free (no over-constraint).
        assert xmin_fix[0].dofs == [0]
        nodefix = _node_fix_bcs(bcs)
        assert len(nodefix) == 1
        assert sorted(nodefix[0].dofs) == [1, 2]

    def test_force_sign_follows_nx_sign(self, small_mesh):
        Lx, Ly, Lz = small_mesh.domain_size
        bcs = BoundaryHandler.load_state_to_bcs(LoadState(Nx=-250.0), small_mesh)
        fx, _, _ = _force_components(small_mesh, bcs)
        assert fx == pytest.approx(-250.0 * Ly)


# ======================================================================
# Pure Ny (symmetric to Nx on the y faces)
# ======================================================================

class TestPureNy:
    """Uniaxial Ny -> pressure on y_max with mirrored constraints."""

    def test_pressure_bc_on_ymax(self, small_mesh):
        Lx, Ly, Lz = small_mesh.domain_size
        bcs = BoundaryHandler.load_state_to_bcs(LoadState(Ny=50.0), small_mesh)
        press = [b for b in bcs if b.bc_type == "pressure"]
        assert len(press) == 1
        bc = press[0]
        assert bc.face == "y_max"
        assert bc.dofs == [1]
        assert bc.value == pytest.approx(50.0 * Lx)

    def test_total_force_equals_resultant(self, small_mesh):
        Lx, Ly, Lz = small_mesh.domain_size
        bcs = BoundaryHandler.load_state_to_bcs(LoadState(Ny=50.0), small_mesh)
        fx, fy, fz = _force_components(small_mesh, bcs)
        assert fy == pytest.approx(50.0 * Lx)
        assert fx == pytest.approx(0.0)
        assert fz == pytest.approx(0.0)

    def test_symmetry_x_added_when_nx_zero(self, small_mesh):
        """When Nx == 0 the Ny branch supplies the x-symmetry + corner
        fix itself (boundary.py:564-571)."""
        bcs = BoundaryHandler.load_state_to_bcs(LoadState(Ny=50.0), small_mesh)
        kinds = _bc_kinds(bcs)
        assert "symmetry_x" in kinds
        ymin_fix = [b for b in bcs
                    if b.bc_type == "fixed" and b.face == "y_min"]
        assert len(ymin_fix) == 1
        assert ymin_fix[0].dofs == [1]
        nodefix = _node_fix_bcs(bcs)
        assert len(nodefix) == 1
        assert sorted(nodefix[0].dofs) == [0, 2]


# ======================================================================
# Combined Nx + Ny: corner-fix added exactly once
# ======================================================================

class TestCombinedNxNy:
    """The Ny branch must NOT re-add the corner fix when Nx already did."""

    def test_corner_fix_added_once(self, small_mesh):
        bcs = BoundaryHandler.load_state_to_bcs(
            LoadState(Nx=100.0, Ny=50.0), small_mesh
        )
        nodefix = _node_fix_bcs(bcs)
        assert len(nodefix) == 1, (
            "Combined Nx+Ny must pin the rigid-body corner once, not twice"
        )
        # The single corner fix is the one from the Nx branch ([1, 2]).
        assert sorted(nodefix[0].dofs) == [1, 2]
        # No duplicate x-symmetry from the Ny branch when Nx != 0.
        assert "symmetry_x" not in _bc_kinds(bcs)

    def test_both_pressures_present(self, small_mesh):
        Lx, Ly, Lz = small_mesh.domain_size
        bcs = BoundaryHandler.load_state_to_bcs(
            LoadState(Nx=100.0, Ny=50.0), small_mesh
        )
        press = [b for b in bcs if b.bc_type == "pressure"]
        faces = sorted(b.face for b in press)
        assert faces == ["x_max", "y_max"]
        fx, fy, fz = _force_components(small_mesh, bcs)
        assert fx == pytest.approx(100.0 * Ly)
        assert fy == pytest.approx(50.0 * Lx)


# ======================================================================
# Pure Nxy (in-plane shear)
# ======================================================================

class TestPureNxy:
    """Shear Nxy -> tangential pressure on x_max in the y-direction."""

    def test_shear_pressure_bc(self, small_mesh):
        Lx, Ly, Lz = small_mesh.domain_size
        bcs = BoundaryHandler.load_state_to_bcs(LoadState(Nxy=30.0), small_mesh)
        press = [b for b in bcs if b.bc_type == "pressure"]
        assert len(press) == 1
        bc = press[0]
        assert bc.face == "x_max"
        assert bc.dofs == [1], "shear traction acts in y on the x_max face"
        assert bc.value == pytest.approx(30.0 * Ly)

    def test_shear_force_and_equilibrium(self, small_mesh):
        Lx, Ly, Lz = small_mesh.domain_size
        bcs = BoundaryHandler.load_state_to_bcs(LoadState(Nxy=30.0), small_mesh)
        fx, fy, fz = _force_components(small_mesh, bcs)
        # Applied shear resultant == Nxy * (loaded edge length Ly).
        assert fy == pytest.approx(30.0 * Ly)
        # No net x or z force: reactions balance through the x_min clamp.
        assert fx == pytest.approx(0.0)
        assert fz == pytest.approx(0.0)

    def test_shear_dof_constraint_pattern(self, small_mesh):
        """With only Nxy, x_min is pinned in ux & uy; one node pins uz."""
        bcs = BoundaryHandler.load_state_to_bcs(LoadState(Nxy=30.0), small_mesh)
        xmin_fix = [b for b in bcs
                    if b.bc_type == "fixed" and b.face == "x_min"]
        assert len(xmin_fix) == 1
        assert sorted(xmin_fix[0].dofs) == [0, 1]
        nodefix = _node_fix_bcs(bcs)
        assert len(nodefix) == 1
        assert nodefix[0].dofs == [2]


# ======================================================================
# Superposition: combined BCs == union of individual contributions
# ======================================================================

class TestSuperposition:
    """For (Nx, Ny, Nxy) the combined BC list is the union of the
    individually generated mechanical/force blocks; the assembled force
    vector is the linear sum of the per-resultant force vectors."""

    def test_force_vector_is_linear_sum(self, small_mesh):
        handler = BoundaryHandler(small_mesh)
        ls = LoadState(Nx=100.0, Ny=50.0, Nxy=30.0)
        combined = BoundaryHandler.load_state_to_bcs(ls, small_mesh)

        f_nx = handler.get_force_dofs(
            BoundaryHandler.load_state_to_bcs(LoadState(Nx=100.0), small_mesh))
        f_ny = handler.get_force_dofs(
            BoundaryHandler.load_state_to_bcs(LoadState(Ny=50.0), small_mesh))
        f_nxy = handler.get_force_dofs(
            BoundaryHandler.load_state_to_bcs(LoadState(Nxy=30.0), small_mesh))
        f_combined = handler.get_force_dofs(combined)

        np.testing.assert_allclose(
            f_combined, f_nx + f_ny + f_nxy, rtol=1e-12, atol=1e-12
        )

    def test_combined_totals(self, small_mesh):
        Lx, Ly, Lz = small_mesh.domain_size
        bcs = BoundaryHandler.load_state_to_bcs(
            LoadState(Nx=100.0, Ny=50.0, Nxy=30.0), small_mesh
        )
        fx, fy, fz = _force_components(small_mesh, bcs)
        assert fx == pytest.approx(100.0 * Ly)
        # y-force = Ny*Lx contribution + Nxy*Ly shear contribution.
        assert fy == pytest.approx(50.0 * Lx + 30.0 * Ly)
        assert fz == pytest.approx(0.0)


# ======================================================================
# Pure Mx bending
# ======================================================================

class TestPureMx:
    """Mx -> linear through-thickness ux prescribed on every x_max node."""

    def test_one_displacement_bc_per_xmax_node(self, bending_mesh):
        bcs = BoundaryHandler.load_state_to_bcs(LoadState(Mx=2.0), bending_mesh)
        disp = [b for b in bcs if b.bc_type == "displacement"]
        xmax = bending_mesh.nodes_on_face("x_max")
        assert len(disp) == len(xmax)
        assert all(b.dofs == [0] for b in disp)

    def test_linear_through_thickness_profile(self, bending_mesh,
                                              two_ply_laminate):
        """ux(z) = kappa_x * (z - z_mid) * Lx with kappa_x = Mx / D11
        (issue #149).  Midplane node ux == 0; top & bottom fibers equal
        and opposite."""
        Lx, Ly, Lz = bending_mesh.domain_size
        Mx = 2.0
        D11 = float(two_ply_laminate.D[0, 0])
        bcs = BoundaryHandler.load_state_to_bcs(LoadState(Mx=Mx), bending_mesh)
        disp = {int(b.node_ids[0]): b.value
                for b in bcs if b.bc_type == "displacement"}
        xmax = bending_mesh.nodes_on_face("x_max")
        z = bending_mesh.nodes[xmax, 2]
        z_mid = 0.5 * (z.min() + z.max())
        for nid in xmax:
            zz = float(bending_mesh.nodes[nid, 2])
            assert disp[nid] == pytest.approx((Mx / D11) * (zz - z_mid) * Lx)
        # Midplane node (z == z_mid) has zero prescribed displacement.
        mid = [nid for nid in xmax
               if abs(float(bending_mesh.nodes[nid, 2]) - z_mid) < 1e-9]
        assert mid, "expected an exact midplane node row on the bending mesh"
        for nid in mid:
            assert disp[nid] == pytest.approx(0.0)
        # Top and bottom fiber displacements are equal and opposite.
        top = max(xmax, key=lambda n: bending_mesh.nodes[n, 2])
        bot = min(xmax, key=lambda n: bending_mesh.nodes[n, 2])
        assert disp[top] == pytest.approx(-disp[bot])

    def test_curvature_should_scale_with_D11(self, bending_mesh,
                                             two_ply_laminate):
        """Physically kappa_x == Mx / D11 (issue #149); the top-fiber
        prescribed ux scales with the laminate bending stiffness."""
        Lx, Ly, Lz = bending_mesh.domain_size
        Mx = 2.0
        D11 = float(two_ply_laminate.D[0, 0])
        bcs = BoundaryHandler.load_state_to_bcs(LoadState(Mx=Mx), bending_mesh)
        disp = {int(b.node_ids[0]): b.value
                for b in bcs if b.bc_type == "displacement"}
        xmax = bending_mesh.nodes_on_face("x_max")
        z = bending_mesh.nodes[xmax, 2]
        z_mid = 0.5 * (z.min() + z.max())
        top = max(xmax, key=lambda n: bending_mesh.nodes[n, 2])
        z_top = float(bending_mesh.nodes[top, 2])
        expected = (Mx / D11) * (z_top - z_mid) * Lx
        assert disp[top] == pytest.approx(expected)

    def test_mx_adds_own_rigidbody_constraints(self, bending_mesh):
        """With Nx == 0 the Mx branch supplies x_min fix, y-symmetry and
        a single corner pin (boundary.py:610-619)."""
        bcs = BoundaryHandler.load_state_to_bcs(LoadState(Mx=2.0), bending_mesh)
        kinds = _bc_kinds(bcs)
        assert "symmetry_y" in kinds
        xmin_fix = [b for b in bcs
                    if b.bc_type == "fixed" and b.face == "x_min"]
        assert len(xmin_fix) == 1 and xmin_fix[0].dofs == [0]
        nodefix = _node_fix_bcs(bcs)
        assert len(nodefix) == 1 and sorted(nodefix[0].dofs) == [1, 2]


# ======================================================================
# Issue #149 regression: bending curvature uses laminate D, not 1.0
# ======================================================================

class TestBendingCurvatureUsesLaminateD:
    """Reproduction of issue #149: kappa_x == Mx / D11, kappa_y == My / D22."""

    def _make_mesh(self):
        mat = OrthotropicMaterial()
        lam = Laminate.from_angles([0.0, 0.0], material=mat,
                                   ply_thickness=0.183)
        mesh = WrinkleMesh(
            laminate=lam, wrinkle_config=None,
            Lx=3.0, Ly=2.0, nx=3, ny=2, nz_per_ply=1,
        ).generate()
        return lam, mesh

    def test_kappa_x_equals_mx_over_d11(self):
        lam, mesh = self._make_mesh()
        Mx = 2.0
        D11 = float(lam.D[0, 0])
        assert D11 != pytest.approx(1.0)  # guard: a real, non-unit divisor
        Lx, Ly, Lz = mesh.domain_size
        bcs = BoundaryHandler.load_state_to_bcs(LoadState(Mx=Mx), mesh)
        disp = {int(b.node_ids[0]): b.value
                for b in bcs if b.bc_type == "displacement"}
        xmax = mesh.nodes_on_face("x_max")
        z = mesh.nodes[xmax, 2]
        z_mid = 0.5 * (z.min() + z.max())
        for nid in xmax:
            zz = float(mesh.nodes[nid, 2])
            assert disp[nid] == pytest.approx((Mx / D11) * (zz - z_mid) * Lx)

    def test_kappa_y_equals_my_over_d22(self):
        lam, mesh = self._make_mesh()
        My = 3.0
        D22 = float(lam.D[1, 1])
        assert D22 != pytest.approx(1.0)  # guard: a real, non-unit divisor
        Lx, Ly, Lz = mesh.domain_size
        bcs = BoundaryHandler.load_state_to_bcs(LoadState(My=My), mesh)
        disp = {int(b.node_ids[0]): b.value
                for b in bcs if b.bc_type == "displacement"}
        ymax = mesh.nodes_on_face("y_max")
        z = mesh.nodes[ymax, 2]
        z_mid = 0.5 * (z.min() + z.max())
        for nid in ymax:
            zz = float(mesh.nodes[nid, 2])
            assert disp[nid] == pytest.approx((My / D22) * (zz - z_mid) * Ly)


# ======================================================================
# StaticSolver._load_state_to_bcs parity / divergence
# ======================================================================

class TestStaticSolverConverter:
    """The private converter used by StaticSolver.run_clt_load."""

    def test_nx_total_force_matches_resultant(self, small_mesh,
                                              single_ply_laminate):
        """Total applied x-force == Nx * Ly via consistent integration."""
        Lx, Ly, Lz = small_mesh.domain_size
        solver = StaticSolver(small_mesh, single_ply_laminate)
        bcs = solver._load_state_to_bcs(LoadState(Nx=100.0))
        fx, fy, fz = _force_components(small_mesh, bcs)
        assert fx == pytest.approx(100.0 * Ly)
        assert fy == pytest.approx(0.0)
        assert fz == pytest.approx(0.0)

    def test_nxy_total_force_matches_resultant(self, small_mesh,
                                               single_ply_laminate):
        Lx, Ly, Lz = small_mesh.domain_size
        solver = StaticSolver(small_mesh, single_ply_laminate)
        bcs = solver._load_state_to_bcs(LoadState(Nxy=30.0))
        fx, fy, fz = _force_components(small_mesh, bcs)
        assert fy == pytest.approx(30.0 * Ly)
        assert fx == pytest.approx(0.0)

    def test_zero_load_only_clamps_xmin(self, small_mesh,
                                        single_ply_laminate):
        """No mechanical load -> only the x_min clamp, no applied force."""
        solver = StaticSolver(small_mesh, single_ply_laminate)
        bcs = solver._load_state_to_bcs(LoadState())
        assert _bc_kinds(bcs) == ["fixed"]
        # x_min clamp is supplied as an explicit node_ids list (not a
        # face= BC) covering every x_min node in all 3 DOFs.
        assert bcs[0].face is None
        assert bcs[0].node_ids is not None
        assert sorted(bcs[0].node_ids.tolist()) == sorted(
            small_mesh.nodes_on_face("x_min").tolist()
        )
        assert sorted(bcs[0].dofs) == [0, 1, 2]
        fx, fy, fz = _force_components(small_mesh, bcs)
        assert (fx, fy, fz) == (0.0, 0.0, 0.0)

    def test_applied_force_agrees_with_boundaryhandler(self, small_mesh,
                                                       single_ply_laminate):
        """Parity guard (#96): the two converters must produce the SAME
        applied force vector for the same LoadState, even though they
        differ in rigid-body constraint structure."""
        handler = BoundaryHandler(small_mesh)
        solver = StaticSolver(small_mesh, single_ply_laminate)
        ls = LoadState(Nx=100.0, Nxy=30.0)
        f_bh = handler.get_force_dofs(
            BoundaryHandler.load_state_to_bcs(ls, small_mesh))
        f_ss = handler.get_force_dofs(solver._load_state_to_bcs(ls))
        np.testing.assert_allclose(f_bh, f_ss, rtol=1e-12, atol=1e-12)

    def test_converters_diverge_on_rigidbody_constraints(self, small_mesh,
                                                         single_ply_laminate):
        """Pins the KNOWN structural divergence the issue calls out:
        BoundaryHandler uses light symmetry + corner fix on x_min, while
        StaticSolver fully clamps the entire x_min face (ux=uy=uz=0).
        Equal applied force, different constraints -> different solves."""
        solver = StaticSolver(small_mesh, single_ply_laminate)
        ls = LoadState(Nx=100.0)

        bh_bcs = BoundaryHandler.load_state_to_bcs(ls, small_mesh)
        ss_bcs = solver._load_state_to_bcs(ls)

        bh_constr = BoundaryHandler(small_mesh).get_constrained_dofs(bh_bcs)
        ss_constr = BoundaryHandler(small_mesh).get_constrained_dofs(ss_bcs)

        # StaticSolver clamps every x_min node in all 3 DOFs.
        xmin = small_mesh.nodes_on_face("x_min")
        for nid in xmin:
            for d in (0, 1, 2):
                assert 3 * int(nid) + d in ss_constr

        # BoundaryHandler does NOT fully clamp x_min: an x_min node that
        # is not the rigid-body corner is free in uy/uz.
        corner = int(xmin[0])
        non_corner = [int(n) for n in xmin if int(n) != corner]
        assert non_corner, "fixture must have >1 node on x_min"
        n0 = non_corner[0]
        assert 3 * n0 + 0 in bh_constr      # ux fixed on x_min
        assert 3 * n0 + 1 not in bh_constr  # uy free (not over-constrained)
        assert 3 * n0 + 2 not in bh_constr  # uz free

        # Therefore the two converters are NOT interchangeable for the
        # constrained-DOF set even though their force vectors match.
        assert ss_constr != bh_constr

    def test_solve_load_state_uses_private_converter(self, small_mesh,
                                                     single_ply_laminate):
        """solve_load_state on a zero load still solves (clamp only) and
        returns a finite, ~zero displacement field."""
        solver = StaticSolver(small_mesh, single_ply_laminate)
        results = solver.solve_load_state(LoadState(), solver="direct")
        assert np.all(np.isfinite(results.displacement))
        assert np.allclose(results.displacement, 0.0, atol=1e-9)
