"""Tests for the linear static solver and supporting modules.

Covers:
- GlobalAssembler: element creation, DOF mapping, stiffness assembly
- BoundaryCondition / BoundaryHandler: BC creation, resolution, application
- StaticSolver: single-element compression, multi-element bar, API checks
- FieldResults: basic shape and property checks
"""

import warnings

import numpy as np
import pytest
from scipy import sparse

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.core.laminate import Laminate, Ply, LoadState
from wrinklefe.core.mesh import WrinkleMesh, MeshData
from wrinklefe.solver.assembler import GlobalAssembler
from wrinklefe.solver.boundary import BoundaryCondition, BoundaryHandler
from wrinklefe.solver.static import StaticSolver
from wrinklefe.solver.results import FieldResults


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def x850_material():
    """Default CYCOM X850/T800 material."""
    return OrthotropicMaterial()


@pytest.fixture
def isotropic_material():
    """Near-isotropic material for simplified verification.

    E=10000 MPa, nu=0.3, G=3846.15 MPa.
    """
    E = 10000.0
    nu = 0.3
    G = E / (2.0 * (1.0 + nu))
    return OrthotropicMaterial(
        E1=E, E2=E, E3=E,
        G12=G, G13=G, G23=G,
        nu12=nu, nu13=nu, nu23=nu,
        Xt=500, Xc=500, Yt=500, Yc=500, Zt=500, Zc=500,
        S12=300, S13=300, S23=300,
        gamma_Y=0.02,
        name="isotropic_10k",
    )


@pytest.fixture
def single_ply_laminate(x850_material):
    """Single-ply [0] laminate with 0.183 mm thickness."""
    return Laminate.from_angles([0.0], material=x850_material, ply_thickness=0.183)


@pytest.fixture
def single_ply_iso_laminate(isotropic_material):
    """Single-ply [0] laminate with isotropic-like material, 1 mm thick."""
    return Laminate.from_angles([0.0], material=isotropic_material, ply_thickness=1.0)


@pytest.fixture
def small_mesh(single_ply_laminate):
    """Small 3x2x1 mesh for single-ply laminate."""
    gen = WrinkleMesh(
        laminate=single_ply_laminate,
        wrinkle_config=None,
        Lx=3.0, Ly=2.0,
        nx=3, ny=2, nz_per_ply=1,
    )
    return gen.generate()


@pytest.fixture
def bar_mesh(single_ply_iso_laminate):
    """5x1x1 mesh (bar along x) with isotropic-like material, 1 mm ply."""
    gen = WrinkleMesh(
        laminate=single_ply_iso_laminate,
        wrinkle_config=None,
        Lx=5.0, Ly=1.0,
        nx=5, ny=1, nz_per_ply=1,
    )
    return gen.generate()


# ======================================================================
# GlobalAssembler tests
# ======================================================================

class TestGlobalAssembler:
    """Tests for the global stiffness matrix assembler."""

    def test_create_assembler(self, small_mesh, single_ply_laminate):
        """Assembler can be created with mesh and laminate."""
        assembler = GlobalAssembler(small_mesh, single_ply_laminate)
        assert assembler.mesh is small_mesh
        assert assembler.laminate is single_ply_laminate

    def test_invalid_element_type(self, small_mesh, single_ply_laminate):
        """Unsupported element type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported"):
            GlobalAssembler(small_mesh, single_ply_laminate, element_type="hex20")

    def test_create_element(self, small_mesh, single_ply_laminate):
        """create_element returns a Hex8Element for element 0."""
        from wrinklefe.elements.hex8 import Hex8Element
        assembler = GlobalAssembler(small_mesh, single_ply_laminate)
        elem = assembler.create_element(0)
        assert isinstance(elem, Hex8Element)

    def test_element_dof_indices_shape(self, small_mesh, single_ply_laminate):
        """element_dof_indices returns 24 DOF indices per element."""
        assembler = GlobalAssembler(small_mesh, single_ply_laminate)
        dofs = assembler.element_dof_indices(0)
        assert dofs.shape == (24,)

    def test_element_dof_indices_values(self, small_mesh, single_ply_laminate):
        """DOF indices are 3*node_id, 3*node_id+1, 3*node_id+2 per node."""
        assembler = GlobalAssembler(small_mesh, single_ply_laminate)
        dofs = assembler.element_dof_indices(0)
        node_ids = small_mesh.elements[0]
        for i, nid in enumerate(node_ids):
            assert dofs[3 * i] == 3 * nid
            assert dofs[3 * i + 1] == 3 * nid + 1
            assert dofs[3 * i + 2] == 3 * nid + 2

    def test_assemble_stiffness_shape(self, small_mesh, single_ply_laminate):
        """Assembled stiffness matrix has shape (n_dof, n_dof)."""
        assembler = GlobalAssembler(small_mesh, single_ply_laminate)
        K = assembler.assemble_stiffness()
        n_dof = small_mesh.n_dof
        assert K.shape == (n_dof, n_dof)

    def test_assemble_stiffness_is_sparse(self, small_mesh, single_ply_laminate):
        """Assembled stiffness is a scipy sparse CSC matrix."""
        assembler = GlobalAssembler(small_mesh, single_ply_laminate)
        K = assembler.assemble_stiffness()
        assert isinstance(K, sparse.csc_matrix)

    def test_assemble_stiffness_symmetric(self, small_mesh, single_ply_laminate):
        """Global stiffness matrix is symmetric."""
        assembler = GlobalAssembler(small_mesh, single_ply_laminate)
        K = assembler.assemble_stiffness()
        K_dense = K.toarray()
        np.testing.assert_allclose(K_dense, K_dense.T, atol=1e-6)

    def test_assemble_stiffness_nonzero(self, small_mesh, single_ply_laminate):
        """Global stiffness matrix has non-zero entries."""
        assembler = GlobalAssembler(small_mesh, single_ply_laminate)
        K = assembler.assemble_stiffness()
        assert K.nnz > 0

    def test_assemble_verbose(self, small_mesh, single_ply_laminate, capsys):
        """Verbose assembly prints progress information."""
        assembler = GlobalAssembler(small_mesh, single_ply_laminate)
        K = assembler.assemble_stiffness(verbose=True)
        captured = capsys.readouterr()
        assert "Assembling" in captured.out


# ======================================================================
# BoundaryCondition tests
# ======================================================================

class TestBoundaryCondition:
    """Tests for the BoundaryCondition dataclass."""

    def test_fixed_bc(self):
        """Create a fixed BC on a face."""
        bc = BoundaryCondition(bc_type="fixed", face="x_min", dofs=[0])
        assert bc.bc_type == "fixed"
        assert bc.face == "x_min"
        assert bc.dofs == [0]

    def test_displacement_bc(self):
        """Create a displacement BC with a value."""
        bc = BoundaryCondition(
            bc_type="displacement", face="x_max", dofs=[0], value=-0.5
        )
        assert bc.bc_type == "displacement"
        assert bc.value == -0.5

    def test_force_bc_with_node_ids(self):
        """Create a force BC on explicit nodes."""
        bc = BoundaryCondition(
            bc_type="force", node_ids=np.array([0, 1, 2]), dofs=[0], value=100.0
        )
        assert bc.bc_type == "force"
        assert len(bc.node_ids) == 3

    def test_invalid_bc_type(self):
        """Unknown bc_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown"):
            BoundaryCondition(bc_type="invalid", face="x_min")

    def test_no_face_no_nodes_raises(self):
        """Neither face nor node_ids raises ValueError."""
        with pytest.raises(ValueError, match="Either"):
            BoundaryCondition(bc_type="fixed")

    def test_symmetry_x_effective_dofs(self):
        """symmetry_x effective DOFs are [0]."""
        bc = BoundaryCondition(bc_type="symmetry_x", face="x_min")
        assert bc.effective_dofs() == [0]

    def test_symmetry_y_effective_dofs(self):
        """symmetry_y effective DOFs are [1]."""
        bc = BoundaryCondition(bc_type="symmetry_y", face="y_min")
        assert bc.effective_dofs() == [1]

    def test_symmetry_z_effective_dofs(self):
        """symmetry_z effective DOFs are [2]."""
        bc = BoundaryCondition(bc_type="symmetry_z", face="z_min")
        assert bc.effective_dofs() == [2]

    def test_resolve_nodes_from_face(self, small_mesh):
        """resolve_nodes returns node IDs from a face."""
        bc = BoundaryCondition(bc_type="fixed", face="x_min", dofs=[0])
        nodes = bc.resolve_nodes(small_mesh)
        assert len(nodes) > 0
        # All returned nodes should have x = x_min
        x_min = small_mesh.nodes[:, 0].min()
        for nid in nodes:
            assert np.isclose(small_mesh.nodes[nid, 0], x_min)

    def test_resolve_nodes_from_node_ids(self, small_mesh):
        """resolve_nodes returns explicit node_ids directly."""
        explicit = np.array([0, 5, 10])
        bc = BoundaryCondition(bc_type="fixed", node_ids=explicit, dofs=[0])
        nodes = bc.resolve_nodes(small_mesh)
        np.testing.assert_array_equal(nodes, explicit)

    def test_default_dofs_all_three(self):
        """Default dofs is [0, 1, 2]."""
        bc = BoundaryCondition(bc_type="fixed", face="x_min")
        assert bc.dofs == [0, 1, 2]

    def test_pressure_bc(self):
        """Create a pressure BC distributed over a face."""
        bc = BoundaryCondition(
            bc_type="pressure", face="x_max", dofs=[0], value=1000.0
        )
        assert bc.bc_type == "pressure"
        assert bc.value == 1000.0


# ======================================================================
# BoundaryHandler tests
# ======================================================================

class TestBoundaryHandler:
    """Tests for the BoundaryHandler class."""

    def test_create_handler(self, small_mesh):
        """BoundaryHandler can be created with a mesh."""
        handler = BoundaryHandler(small_mesh)
        assert handler.mesh is small_mesh

    def test_get_constrained_dofs(self, small_mesh):
        """get_constrained_dofs returns a dict of DOF -> value."""
        handler = BoundaryHandler(small_mesh)
        bcs = [
            BoundaryCondition(bc_type="fixed", face="x_min", dofs=[0]),
        ]
        constrained = handler.get_constrained_dofs(bcs)
        assert isinstance(constrained, dict)
        assert len(constrained) > 0
        # All values should be 0 for fixed BCs
        for dof, val in constrained.items():
            assert val == 0.0

    def test_get_constrained_dofs_displacement(self, small_mesh):
        """Displacement BCs produce non-zero prescribed values."""
        handler = BoundaryHandler(small_mesh)
        bcs = [
            BoundaryCondition(
                bc_type="displacement", face="x_max", dofs=[0], value=-0.01
            ),
        ]
        constrained = handler.get_constrained_dofs(bcs)
        assert len(constrained) > 0
        for dof, val in constrained.items():
            assert val == -0.01

    def test_get_force_dofs(self, small_mesh):
        """get_force_dofs returns a force vector of correct size."""
        handler = BoundaryHandler(small_mesh)
        bcs = [
            BoundaryCondition(
                bc_type="force",
                node_ids=np.array([0]),
                dofs=[0],
                value=100.0,
            ),
        ]
        F = handler.get_force_dofs(bcs)
        assert F.shape == (small_mesh.n_dof,)
        # Force at node 0, DOF 0
        assert np.isclose(F[0], 100.0)

    def test_pressure_bc_distribution_total_force(self, small_mesh):
        """Pressure BC: summed nodal forces equal the prescribed total.

        Regression for issue #50: the distribution is *not* equal per
        node (corners, edges, and interior face nodes get different
        weights), but the global sum must match the prescribed total
        force on the face.
        """
        handler = BoundaryHandler(small_mesh)
        total_force = 600.0
        bcs = [
            BoundaryCondition(
                bc_type="pressure", face="x_max", dofs=[0], value=total_force
            ),
        ]
        F = handler.get_force_dofs(bcs)
        xmax_nodes = small_mesh.nodes_on_face("x_max")
        # Global equilibrium: sum of nodal Fx equals the total prescribed
        applied_x = F[3 * xmax_nodes].sum()
        assert np.isclose(applied_x, total_force, rtol=1e-12, atol=1e-9)
        # And no Fx force leaks elsewhere
        other = np.ones(small_mesh.n_dof, dtype=bool)
        other[3 * xmax_nodes] = False
        assert np.allclose(F[other], 0.0)

    def test_pressure_bc_consistent_tributary_weights(self, small_mesh):
        """Pressure BC distribution follows corner-1/4, edge-1/2, interior-1.

        On a flat structured mesh with uniform face quads of area ``A``,
        consistent face integration produces nodal forces:

        - corner face nodes (in 1 quad): ``t * A / 4``
        - edge face nodes  (in 2 quads): ``t * A / 2``
        - interior nodes   (in 4 quads): ``t * A``

        where ``t = total_force / total_face_area``.
        """
        handler = BoundaryHandler(small_mesh)
        total_force = 600.0
        bcs = [
            BoundaryCondition(
                bc_type="pressure", face="x_max", dofs=[0], value=total_force
            ),
        ]
        F = handler.get_force_dofs(bcs)

        face_quads = small_mesh.face_elements("x_max")
        # Total face area (sum of quad areas)
        from wrinklefe.solver.boundary import _quad_areas
        areas = _quad_areas(small_mesh.nodes, face_quads)
        A_total = float(areas.sum())
        traction = total_force / A_total
        # Number of face quads each node belongs to
        from collections import Counter
        share_count = Counter(face_quads.ravel().tolist())

        # Each face node's expected Fx = traction * sum(quad_area)/4 over
        # all quads that touch it.  For a regular grid quads have equal
        # area so this reduces to traction * A_e * (n_quads_touching)/4.
        for nid in small_mesh.nodes_on_face("x_max"):
            mask = (face_quads == nid).any(axis=1)
            expected = traction * areas[mask].sum() / 4.0
            assert np.isclose(F[3 * nid], expected, rtol=1e-10, atol=1e-12), (
                f"node {nid} (n_quads={share_count[int(nid)]}): "
                f"F={F[3 * nid]}, expected={expected}"
            )

    def test_pressure_bc_uniform_face_buckets(self, small_mesh):
        """On a flat uniform grid, face nodes group into corner/edge/interior buckets.

        Verifies the canonical 1:2:4 ratio of consistent nodal forces.
        """
        # small_mesh is 3x2x1 (Lx=3, Ly=2), so x_max face has
        # (ny+1)*(nz+1) = 3*2 = 6 nodes and ny*nz = 2 quads.
        # With only 2 quads (one row in z, two columns in y), every
        # face node is either at a quad corner (touched by 1 quad) or
        # at the shared edge (touched by 2 quads).  Use a denser face
        # by inflating ny, nz.
        from wrinklefe.core.mesh import WrinkleMesh
        from wrinklefe.core.material import OrthotropicMaterial
        from wrinklefe.core.laminate import Laminate

        mat = OrthotropicMaterial()
        lam = Laminate.from_angles([0.0], material=mat, ply_thickness=1.0)
        gen = WrinkleMesh(
            laminate=lam, wrinkle_config=None,
            Lx=3.0, Ly=3.0, nx=3, ny=3, nz_per_ply=3,
        )
        mesh = gen.generate()

        handler = BoundaryHandler(mesh)
        total_force = 90.0  # nice round number with Lx not in the load
        bcs = [
            BoundaryCondition(
                bc_type="pressure", face="x_max", dofs=[0], value=total_force,
            ),
        ]
        F = handler.get_force_dofs(bcs)

        face_quads = mesh.face_elements("x_max")
        face_nodes = mesh.nodes_on_face("x_max")
        # On a regular grid all face quads have equal area:
        from wrinklefe.solver.boundary import _quad_areas
        areas = _quad_areas(mesh.nodes, face_quads)
        np.testing.assert_allclose(areas, areas[0], rtol=1e-12)
        A_total = float(areas.sum())
        A_e = float(areas[0])
        traction = total_force / A_total

        # Tally how many quads each face node belongs to and check the
        # nodal force matches t * A_e * count / 4.
        from collections import Counter
        counts = Counter(face_quads.ravel().tolist())

        # Sanity: we should see counts {1, 2, 4} for a uniform interior face
        unique_counts = set(counts[int(nid)] for nid in face_nodes)
        assert unique_counts == {1, 2, 4}, unique_counts

        for nid in face_nodes:
            expected = traction * A_e * counts[int(nid)] / 4.0
            assert np.isclose(F[3 * nid], expected, rtol=1e-10), (
                f"node {nid} (count={counts[int(nid)]}): "
                f"F={F[3 * nid]}, expected={expected}"
            )

        # Global equilibrium
        assert np.isclose(F[3 * face_nodes].sum(), total_force, rtol=1e-12)

    def test_pressure_bc_wrinkled_mesh_equilibrium(self, x850_material):
        """Consistent face integration preserves total force on a wrinkled mesh.

        Regression for issue #50: on a wrinkled mesh, face quads have
        unequal areas; the legacy equal-per-node distribution gave the
        wrong tributary weighting (corner over-loading).  With the
        consistent-integration fix, the *summed* face force still equals
        the prescribed total, and corners now get less than the
        legacy 1/N share.
        """
        from wrinklefe.core.mesh import WrinkleMesh
        from wrinklefe.core.laminate import Laminate
        from wrinklefe.core.wrinkle import GaussianSinusoidal
        from wrinklefe.core.morphology import WrinkleConfiguration

        lam = Laminate.from_angles(
            [0.0, 0.0, 0.0, 0.0],
            material=x850_material, ply_thickness=0.183,
        )
        wrinkle = GaussianSinusoidal(
            amplitude=0.2, wavelength=8.0, width=6.0, center=0.0,
        )
        cfg = WrinkleConfiguration.dual_wrinkle(
            profile=wrinkle, interface1=1, interface2=2, phase=0.0,
        )
        gen = WrinkleMesh(
            laminate=lam, wrinkle_config=cfg,
            Lx=10.0, Ly=6.0, nx=8, ny=4, nz_per_ply=1,
        )
        mesh = gen.generate()

        handler = BoundaryHandler(mesh)
        total_force = 1234.5
        bcs = [
            BoundaryCondition(
                bc_type="pressure", face="z_max", dofs=[2],
                value=total_force,
            ),
        ]
        F = handler.get_force_dofs(bcs)

        face_nodes = mesh.nodes_on_face("z_max")
        # Equilibrium: the integrated normal force matches the
        # prescribed total to numerical tolerance.
        assert np.isclose(F[3 * face_nodes + 2].sum(), total_force, rtol=1e-10)

        # On a wrinkled face the corner nodes should receive *less*
        # than the legacy equal-share value because they sit in one
        # quad (1/4 area weight) while interior nodes accumulate area
        # from four neighbouring quads.
        face_quads = mesh.face_elements("z_max")
        from collections import Counter
        counts = Counter(face_quads.ravel().tolist())
        corner_nodes = [n for n in face_nodes if counts[int(n)] == 1]
        interior_nodes = [n for n in face_nodes if counts[int(n)] == 4]
        assert corner_nodes and interior_nodes
        legacy_share = total_force / len(face_nodes)
        for n in corner_nodes:
            assert F[3 * n + 2] < legacy_share
        for n in interior_nodes:
            assert F[3 * n + 2] > legacy_share

    def test_pressure_bc_nodeid_fallback_legacy(self, small_mesh):
        """Pressure BC with explicit ``node_ids`` falls back to equal split.

        Without face topology there is no way to integrate over surface
        shape functions, so we keep the legacy equal-per-node behaviour
        as a documented fallback.
        """
        handler = BoundaryHandler(small_mesh)
        xmax_nodes = small_mesh.nodes_on_face("x_max")
        total_force = 1000.0
        bcs = [
            BoundaryCondition(
                bc_type="pressure",
                node_ids=xmax_nodes,
                dofs=[0],
                value=total_force,
            ),
        ]
        F = handler.get_force_dofs(bcs)
        per_node = total_force / len(xmax_nodes)
        for nid in xmax_nodes:
            assert np.isclose(F[3 * nid], per_node)

    def test_apply_penalty(self, small_mesh, single_ply_laminate):
        """Penalty method modifies diagonal and force vector."""
        assembler = GlobalAssembler(small_mesh, single_ply_laminate)
        K = assembler.assemble_stiffness()
        F = np.zeros(small_mesh.n_dof)
        handler = BoundaryHandler(small_mesh)

        constrained = {0: 0.0, 3: -0.01}
        K_mod, F_mod = handler.apply_penalty(K, F, constrained)

        # Diagonal entries should be very large for constrained DOFs
        K_dense = K_mod.toarray()
        assert K_dense[0, 0] > 1e15
        assert K_dense[3, 3] > 1e15
        # F should be modified for non-zero prescribed displacement
        assert np.isclose(F_mod[0], 0.0, atol=1e-5)
        assert F_mod[3] < 0  # negative prescribed displacement

    def test_apply_penalty_accumulates_force(
        self, small_mesh, single_ply_laminate
    ):
        """Regression test for issue #46.

        ``apply_penalty`` must accumulate (``+=``) into the force vector
        for constrained DOFs, not overwrite (``=``).  Otherwise any force
        applied at the same DOF by another BC (or pre-existing F entry)
        is silently wiped out.
        """
        assembler = GlobalAssembler(small_mesh, single_ply_laminate)
        K = assembler.assemble_stiffness()
        handler = BoundaryHandler(small_mesh)

        # Seed an applied force at DOF 3 and DOF 0
        F = np.zeros(small_mesh.n_dof)
        original_force_dof3 = 12.5
        original_force_dof0 = -7.25
        F[3] = original_force_dof3
        F[0] = original_force_dof0

        penalty = 1e20
        prescribed_val_dof3 = -0.01
        prescribed_val_dof0 = 0.0
        constrained = {0: prescribed_val_dof0, 3: prescribed_val_dof3}

        K_mod, F_mod = handler.apply_penalty(
            K, F, constrained, penalty=penalty
        )

        # The penalty assignment must accumulate, not overwrite
        expected_dof3 = original_force_dof3 + penalty * prescribed_val_dof3
        expected_dof0 = original_force_dof0 + penalty * prescribed_val_dof0
        assert F_mod[3] == pytest.approx(expected_dof3, rel=1e-12)
        assert F_mod[0] == pytest.approx(expected_dof0, rel=0, abs=1e-6)

        # Sanity: input F must not be mutated in place
        assert F[3] == original_force_dof3
        assert F[0] == original_force_dof0

    def test_apply_elimination(self, small_mesh, single_ply_laminate):
        """Elimination method reduces the system size."""
        assembler = GlobalAssembler(small_mesh, single_ply_laminate)
        K = assembler.assemble_stiffness()
        F = np.zeros(small_mesh.n_dof)
        handler = BoundaryHandler(small_mesh)

        constrained = {0: 0.0, 1: 0.0, 2: 0.0}
        K_ff, F_red, free_dofs, c_dofs, c_vals = handler.apply_elimination(
            K, F, constrained
        )
        n_dof = small_mesh.n_dof
        n_free = n_dof - 3
        assert K_ff.shape == (n_free, n_free)
        assert F_red.shape == (n_free,)
        assert len(free_dofs) == n_free
        assert len(c_dofs) == 3

    def test_compression_bcs(self, small_mesh):
        """compression_bcs creates a valid set of BCs."""
        bcs = BoundaryHandler.compression_bcs(small_mesh, applied_strain=-0.01)
        assert isinstance(bcs, list)
        assert len(bcs) > 0
        bc_types = [bc.bc_type for bc in bcs]
        assert "fixed" in bc_types
        assert "displacement" in bc_types

    def test_bending_bcs(self, small_mesh):
        """bending_bcs creates a valid set of BCs."""
        bcs = BoundaryHandler.bending_bcs(small_mesh, curvature=0.001)
        assert isinstance(bcs, list)
        assert len(bcs) > 0
        bc_types = [bc.bc_type for bc in bcs]
        assert "fixed" in bc_types
        assert "displacement" in bc_types


# ======================================================================
# StaticSolver creation and basic API
# ======================================================================

class TestStaticSolverCreation:
    """Tests for StaticSolver instantiation."""

    def test_create_solver(self, small_mesh, single_ply_laminate):
        """StaticSolver can be created with mesh and laminate."""
        solver = StaticSolver(small_mesh, single_ply_laminate)
        assert solver.mesh is small_mesh
        assert solver.laminate is single_ply_laminate

    def test_default_element_type(self, small_mesh, single_ply_laminate):
        """Default element type is hex8."""
        solver = StaticSolver(small_mesh, single_ply_laminate)
        assert solver.element_type == "hex8"

    def test_explicit_hex8(self, small_mesh, single_ply_laminate):
        """Can create solver with explicit hex8 element type."""
        solver = StaticSolver(small_mesh, single_ply_laminate, element_type="hex8")
        assert solver.element_type == "hex8"


# ======================================================================
# Single-element compression test
# ======================================================================

class TestSingleElementCompression:
    """Test compression of a single element using displacement BCs.

    Creates a 1x1x1 mesh, applies compression via displacement BCs,
    and solves directly using the assembler and boundary handler.
    """

    @pytest.fixture
    def single_element_mesh(self, x850_material):
        """1x1x1 mesh (single element)."""
        lam = Laminate.from_angles([0.0], material=x850_material, ply_thickness=1.0)
        gen = WrinkleMesh(
            laminate=lam, wrinkle_config=None,
            Lx=1.0, Ly=1.0, nx=1, ny=1, nz_per_ply=1,
        )
        return gen.generate(), lam

    def test_single_element_compression(self, single_element_mesh):
        """Single element under compression produces non-zero displacement."""
        mesh, lam = single_element_mesh

        # Assemble
        assembler = GlobalAssembler(mesh, lam)
        K = assembler.assemble_stiffness()
        F = np.zeros(mesh.n_dof)

        # Boundary conditions: fix x_min face ux=0, prescribe x_max ux=-0.001
        handler = BoundaryHandler(mesh)
        bcs = BoundaryHandler.compression_bcs(mesh, applied_strain=-0.001)
        constrained = handler.get_constrained_dofs(bcs)

        # Apply penalty BCs
        K_mod, F_mod = handler.apply_penalty(K, F, constrained)

        # Solve
        from scipy.sparse.linalg import spsolve
        u = spsolve(K_mod, F_mod)

        # Check displacement vector is non-zero
        assert np.linalg.norm(u) > 0

        # Check x_max face has negative ux (compression)
        xmax_nodes = mesh.nodes_on_face("x_max")
        for nid in xmax_nodes:
            ux = u[3 * nid]
            assert ux < 0, f"Node {nid} ux={ux} should be negative (compression)"

        # Check x_min face ux is approximately zero
        xmin_nodes = mesh.nodes_on_face("x_min")
        for nid in xmin_nodes:
            ux = u[3 * nid]
            assert abs(ux) < 1e-8, f"Node {nid} ux={ux} should be ~0 (fixed)"

    def test_single_element_stress_recovery(self, single_element_mesh, x850_material):
        """Recover stress from single element under compression."""
        mesh, lam = single_element_mesh

        assembler = GlobalAssembler(mesh, lam)
        K = assembler.assemble_stiffness()
        F = np.zeros(mesh.n_dof)

        handler = BoundaryHandler(mesh)
        bcs = BoundaryHandler.compression_bcs(mesh, applied_strain=-0.001)
        constrained = handler.get_constrained_dofs(bcs)
        K_mod, F_mod = handler.apply_penalty(K, F, constrained)

        from scipy.sparse.linalg import spsolve
        u = spsolve(K_mod, F_mod)

        # Create element and compute stress
        elem = assembler.create_element(0)
        dofs = assembler.element_dof_indices(0)
        u_elem = u[dofs]
        stresses = elem.stress_at_gauss_points(u_elem)

        assert stresses.shape == (8, 6)
        # sigma_11 should be negative (compression)
        assert np.all(stresses[:, 0] < 0), "sigma_11 should be compressive"


# ======================================================================
# Multi-element bar under tension
# ======================================================================

class TestMultiElementBar:
    """Multi-element bar under tension (5x1x1 mesh, isotropic material).

    BCs: fix ux=0 on x_min, prescribe ux=0.01*Lx on x_max,
         symmetry on y_min and one corner node fixed for rigid body.

    Expected: approximately uniform strain eps_11 ~ 0.01.
    """

    def test_bar_tension(self, isotropic_material):
        """5x1x1 bar under 1% tension gives approximately uniform strain."""
        lam = Laminate.from_angles(
            [0.0], material=isotropic_material, ply_thickness=1.0
        )
        gen = WrinkleMesh(
            laminate=lam, wrinkle_config=None,
            Lx=5.0, Ly=1.0, nx=5, ny=1, nz_per_ply=1,
        )
        mesh = gen.generate()

        # Assemble
        assembler = GlobalAssembler(mesh, lam)
        K = assembler.assemble_stiffness()
        F = np.zeros(mesh.n_dof)
        handler = BoundaryHandler(mesh)

        # Apply BCs manually using compression_bcs (with positive strain = tension)
        applied_strain = 0.01
        bcs = BoundaryHandler.compression_bcs(mesh, applied_strain=applied_strain)
        constrained = handler.get_constrained_dofs(bcs)

        K_mod, F_mod = handler.apply_penalty(K, F, constrained)

        from scipy.sparse.linalg import spsolve
        u = spsolve(K_mod, F_mod)

        # Check x_max face displacement is approximately applied_strain * Lx
        Lx = mesh.domain_size[0]
        xmax_nodes = mesh.nodes_on_face("x_max")
        for nid in xmax_nodes:
            ux = u[3 * nid]
            expected_ux = applied_strain * Lx
            assert np.isclose(ux, expected_ux, rtol=1e-3), (
                f"Node {nid}: ux={ux:.6f}, expected={expected_ux:.6f}"
            )

        # Check strain is approximately uniform by sampling interior elements
        # eps_11 ~ applied_strain = 0.01
        for e_idx in range(mesh.n_elements):
            elem = assembler.create_element(e_idx)
            dofs = assembler.element_dof_indices(e_idx)
            u_elem = u[dofs]
            strains = elem.strain_at_gauss_points(u_elem)
            # eps_11 at all Gauss points should be close to applied_strain
            for gp in range(8):
                eps11 = strains[gp, 0]
                assert np.isclose(eps11, applied_strain, rtol=0.05), (
                    f"Element {e_idx}, GP {gp}: eps_11={eps11:.6f}, "
                    f"expected ~{applied_strain}"
                )

    def test_bar_stress_approximately_correct(self, isotropic_material):
        """Bar stress sigma_11 is approximately E * eps for isotropic material.

        The compression_bcs uses symmetry_y on y_min but y_max and z faces
        are free, so the bar is largely unconstrained transversely.  For an
        isotropic material in uniaxial tension with free lateral surfaces,
        sigma_11 ~ E * eps (not C11 * eps).  However, the single-element
        y-thickness combined with the symmetry BC creates a mixed state.
        We use a generous tolerance to accommodate these effects.
        """
        lam = Laminate.from_angles(
            [0.0], material=isotropic_material, ply_thickness=1.0
        )
        gen = WrinkleMesh(
            laminate=lam, wrinkle_config=None,
            Lx=5.0, Ly=1.0, nx=5, ny=1, nz_per_ply=1,
        )
        mesh = gen.generate()

        assembler = GlobalAssembler(mesh, lam)
        K = assembler.assemble_stiffness()
        F = np.zeros(mesh.n_dof)
        handler = BoundaryHandler(mesh)

        applied_strain = 0.01
        bcs = BoundaryHandler.compression_bcs(mesh, applied_strain=applied_strain)
        constrained = handler.get_constrained_dofs(bcs)
        K_mod, F_mod = handler.apply_penalty(K, F, constrained)

        from scipy.sparse.linalg import spsolve
        u = spsolve(K_mod, F_mod)

        # For a bar with free lateral faces, sigma_11 ~ E * eps_11
        E = isotropic_material.E1
        expected_sigma11_uniaxial = E * applied_strain  # 100 MPa

        # For a fully confined bar, sigma_11 = C11 * eps_11
        C = isotropic_material.stiffness_matrix
        expected_sigma11_confined = C[0, 0] * applied_strain  # ~134.6 MPa

        # Check a middle element
        mid_elem = mesh.n_elements // 2
        elem = assembler.create_element(mid_elem)
        dofs = assembler.element_dof_indices(mid_elem)
        u_elem = u[dofs]
        stresses = elem.stress_at_gauss_points(u_elem)

        mean_sigma11 = stresses[:, 0].mean()
        # sigma_11 should be between uniaxial and confined estimates
        assert expected_sigma11_uniaxial * 0.8 < mean_sigma11 < expected_sigma11_confined * 1.1, (
            f"mean sigma_11={mean_sigma11:.2f} should be between "
            f"{expected_sigma11_uniaxial:.2f} (uniaxial) and "
            f"{expected_sigma11_confined:.2f} (confined)"
        )


# ======================================================================
# StaticSolver.solve() — tests for the full solve path
# ======================================================================

class TestStaticSolverSolve:
    """Tests for StaticSolver.solve() with BoundaryCondition list."""

    def test_solve_compression(self, small_mesh, single_ply_laminate):
        """solve() with compression BCs completes and returns FieldResults."""
        solver = StaticSolver(small_mesh, single_ply_laminate)
        bcs = BoundaryHandler.compression_bcs(small_mesh, applied_strain=-0.01)
        results = solver.solve(bcs)
        assert isinstance(results, FieldResults)
        assert results.displacement.shape == (small_mesh.n_nodes, 3)

    def test_solve_field_results_shapes(self, small_mesh, single_ply_laminate):
        """FieldResults has correct shapes for stress/strain arrays."""
        solver = StaticSolver(small_mesh, single_ply_laminate)
        bcs = BoundaryHandler.compression_bcs(small_mesh, applied_strain=-0.01)
        results = solver.solve(bcs)

        n_elem = small_mesh.n_elements
        n_gp = 8  # hex8 with 2x2x2 quadrature
        assert results.stress_global.shape == (n_elem, n_gp, 6)
        assert results.stress_local.shape == (n_elem, n_gp, 6)
        assert results.strain_global.shape == (n_elem, n_gp, 6)
        assert results.strain_local.shape == (n_elem, n_gp, 6)


# ======================================================================
# StaticSolver.solve_load_state() — tests for CLT-to-3D path
# ======================================================================

class TestSolveLoadState:
    """Tests for solve_load_state()."""

    def test_solve_load_state_compression(self, small_mesh, single_ply_laminate):
        """solve_load_state with Nx compression completes without error."""
        solver = StaticSolver(small_mesh, single_ply_laminate)
        load = LoadState(Nx=-1000.0)
        results = solver.solve_load_state(load)
        assert isinstance(results, FieldResults)


# ======================================================================
# FieldResults tests
# ======================================================================

class TestFieldResults:
    """Tests for the FieldResults dataclass."""

    @pytest.fixture
    def dummy_results(self, small_mesh, single_ply_laminate):
        """Create a FieldResults with synthetic data."""
        n_nodes = small_mesh.n_nodes
        n_elem = small_mesh.n_elements
        n_gp = 8
        return FieldResults(
            displacement=np.zeros((n_nodes, 3)),
            stress_global=np.ones((n_elem, n_gp, 6)),
            stress_local=np.ones((n_elem, n_gp, 6)),
            strain_global=np.ones((n_elem, n_gp, 6)) * 0.001,
            strain_local=np.ones((n_elem, n_gp, 6)) * 0.001,
            mesh=small_mesh,
            laminate=single_ply_laminate,
        )

    def test_displacement_shape(self, dummy_results, small_mesh):
        """Displacement has shape (n_nodes, 3)."""
        assert dummy_results.displacement.shape == (small_mesh.n_nodes, 3)

    def test_stress_shape(self, dummy_results, small_mesh):
        """Stress arrays have shape (n_elements, n_gp, 6)."""
        assert dummy_results.stress_global.shape[0] == small_mesh.n_elements
        assert dummy_results.stress_global.shape[2] == 6

    def test_max_displacement(self, dummy_results):
        """max_displacement returns (float, int) tuple."""
        mag, node_idx = dummy_results.max_displacement()
        assert isinstance(mag, float)
        assert isinstance(node_idx, int)

    def test_max_stress(self, dummy_results):
        """max_stress returns (float, int, int) tuple."""
        val, elem_idx, gp_idx = dummy_results.max_stress(component=0)
        assert isinstance(val, float)
        assert isinstance(elem_idx, int)
        assert isinstance(gp_idx, int)

    def test_von_mises(self, dummy_results):
        """Von Mises stress has shape (n_elements, n_gp)."""
        vm = dummy_results.von_mises
        assert vm.shape == (
            dummy_results.stress_global.shape[0],
            dummy_results.stress_global.shape[1],
        )
        assert np.all(vm >= 0)

    def test_summary(self, dummy_results):
        """summary() returns a non-empty string."""
        s = dummy_results.summary()
        assert isinstance(s, str)
        assert len(s) > 0
        assert "FE Analysis" in s

    def test_von_mises_zero_stress(self, small_mesh, single_ply_laminate):
        """Von Mises is zero when all stresses are zero."""
        n_elem = small_mesh.n_elements
        n_gp = 8
        results = FieldResults(
            displacement=np.zeros((small_mesh.n_nodes, 3)),
            stress_global=np.zeros((n_elem, n_gp, 6)),
            stress_local=np.zeros((n_elem, n_gp, 6)),
            strain_global=np.zeros((n_elem, n_gp, 6)),
            strain_local=np.zeros((n_elem, n_gp, 6)),
            mesh=small_mesh,
            laminate=single_ply_laminate,
        )
        vm = results.von_mises
        np.testing.assert_allclose(vm, 0.0)

    def test_max_principal_stress(self, dummy_results):
        """max_principal_stress has correct shape."""
        mp = dummy_results.max_principal_stress
        n_elem = dummy_results.stress_global.shape[0]
        n_gp = dummy_results.stress_global.shape[1]
        assert mp.shape == (n_elem, n_gp)

    def test_reaction_forces_shape_and_equilibrium(
        self, bar_mesh, single_ply_iso_laminate
    ):
        """Regression for issue #48: reaction_forces returns (n_nodes, 4).

        Each row is ``[node_id, Rx, Ry, Rz]``.  For a static problem with
        only displacement BCs and no external forces, the sum of reaction
        components must equal the (zero) external load per direction
        (Newton's third law / equilibrium).
        """
        # Solve a small compression problem to get a real K, u, and BC set.
        solver = StaticSolver(bar_mesh, single_ply_iso_laminate)
        bcs = BoundaryHandler.compression_bcs(bar_mesh, applied_strain=-0.005)
        results = solver.solve(bcs, solver="direct")

        # Reach into the solver's stored K and constrained DOFs map.
        K = solver._K
        constrained = solver._constrained_dofs
        assert len(constrained) > 0

        R = results.reaction_forces(K, constrained)

        # --- Shape contract (the core of issue #48) ---
        # One row per *node* that has any constrained DOF, with four columns.
        expected_nodes = sorted({dof // 3 for dof in constrained})
        assert R.ndim == 2
        assert R.shape == (len(expected_nodes), 4)

        # Column 0 is the node id; remaining columns are Rx, Ry, Rz.
        assert np.array_equal(R[:, 0].astype(int), np.array(expected_nodes))

        # --- Equilibrium: only displacement BCs and zero external load,
        # so global reaction sums must equal zero (the applied load).
        # Use a tolerance scaled by the magnitude of the reactions.
        R_total = R[:, 1:4].sum(axis=0)
        scale = max(np.abs(R[:, 1:4]).max(), 1.0)
        np.testing.assert_allclose(R_total, 0.0, atol=1e-6 * scale)


# ======================================================================
# Integration test: manual solve with assembler + handler (no StaticSolver)
# ======================================================================

class TestManualSolveIntegration:
    """End-to-end integration test using assembler and boundary handler directly.

    Assembles K, applies BCs via penalty, solves, and recovers stresses
    using the element-level API.
    """

    def test_full_workflow(self, x850_material):
        """Complete workflow: mesh -> assemble -> BCs -> solve -> stress."""
        # Create a small 2x1x1 mesh with a single ply
        lam = Laminate.from_angles([0.0], material=x850_material, ply_thickness=0.183)
        gen = WrinkleMesh(
            laminate=lam, wrinkle_config=None,
            Lx=2.0, Ly=1.0, nx=2, ny=1, nz_per_ply=1,
        )
        mesh = gen.generate()

        # Assemble global stiffness
        assembler = GlobalAssembler(mesh, lam)
        K = assembler.assemble_stiffness()
        assert K.shape[0] == mesh.n_dof

        # Create BCs
        handler = BoundaryHandler(mesh)
        bcs = BoundaryHandler.compression_bcs(mesh, applied_strain=-0.005)
        constrained = handler.get_constrained_dofs(bcs)
        F = handler.get_force_dofs(bcs)

        # Apply penalty
        K_mod, F_mod = handler.apply_penalty(K, F, constrained)

        # Solve
        from scipy.sparse.linalg import spsolve
        u = spsolve(K_mod, F_mod)
        assert u.shape == (mesh.n_dof,)

        # Verify non-trivial solution
        assert np.linalg.norm(u) > 0

        # Recover stress for first element
        elem = assembler.create_element(0)
        dofs = assembler.element_dof_indices(0)
        u_elem = u[dofs]
        stresses = elem.stress_at_gauss_points(u_elem)
        strains = elem.strain_at_gauss_points(u_elem)

        assert stresses.shape == (8, 6)
        assert strains.shape == (8, 6)

        # Under compression, sigma_11 should be negative for a [0] ply
        assert stresses[:, 0].mean() < 0

    def test_elimination_method_solve(self, x850_material):
        """Solve using elimination method instead of penalty."""
        lam = Laminate.from_angles([0.0], material=x850_material, ply_thickness=0.183)
        gen = WrinkleMesh(
            laminate=lam, wrinkle_config=None,
            Lx=2.0, Ly=1.0, nx=2, ny=1, nz_per_ply=1,
        )
        mesh = gen.generate()

        assembler = GlobalAssembler(mesh, lam)
        K = assembler.assemble_stiffness()
        handler = BoundaryHandler(mesh)

        bcs = BoundaryHandler.compression_bcs(mesh, applied_strain=-0.005)
        constrained = handler.get_constrained_dofs(bcs)
        F = handler.get_force_dofs(bcs)

        K_ff, F_red, free_dofs, c_dofs, c_vals = handler.apply_elimination(
            K, F, constrained
        )

        from scipy.sparse.linalg import spsolve
        u_free = spsolve(K_ff, F_red)

        # Reconstruct full displacement vector
        u_full = np.zeros(mesh.n_dof)
        u_full[free_dofs] = u_free
        for dof, val in constrained.items():
            u_full[dof] = val

        assert np.linalg.norm(u_full) > 0

        # x_min face should have zero ux
        xmin_nodes = mesh.nodes_on_face("x_min")
        for nid in xmin_nodes:
            assert abs(u_full[3 * nid]) < 1e-8

        # x_max face should have negative ux
        xmax_nodes = mesh.nodes_on_face("x_max")
        for nid in xmax_nodes:
            assert u_full[3 * nid] < 0


# ======================================================================
# Regression tests for scipy CG ``rtol=`` migration (issue #47)
# ======================================================================

class TestCGSolverScipyRtolKwarg:
    """Regression: ``scipy.sparse.linalg.cg`` must be called with ``rtol=``.

    SciPy >=1.12 deprecated the ``tol=`` kwarg on iterative Krylov solvers
    (``cg``, ``cgs``, ``bicg``, ``bicgstab``, ``gmres``, ``lgmres``, etc.)
    in favour of ``rtol=`` and ``atol=``. Using the old kwarg emits a
    ``DeprecationWarning`` on 1.12-1.13 and raises ``TypeError`` on 1.14+.
    """

    def test_iterative_solver_no_deprecation_warning(
        self, small_mesh, single_ply_laminate
    ):
        """The iterative CG solver must not emit a DeprecationWarning.

        Catches *all* warnings during a full ``solver='iterative'`` solve
        and asserts none of them relate to the deprecated ``tol=`` kwarg.
        """
        solver = StaticSolver(small_mesh, single_ply_laminate)
        bcs = BoundaryHandler.compression_bcs(small_mesh, applied_strain=-0.001)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = solver.solve(bcs, solver="iterative")

        # FieldResults must come back as usual.
        assert isinstance(results, FieldResults)
        assert results.displacement.shape == (small_mesh.n_nodes, 3)

        # No deprecation about the ``tol`` kwarg should bubble up.
        offenders = [
            w for w in caught
            if issubclass(w.category, DeprecationWarning)
            and "tol" in str(w.message).lower()
        ]
        assert not offenders, (
            "scipy CG emitted a tol-related DeprecationWarning: "
            + "; ".join(str(w.message) for w in offenders)
        )

    def test_cg_accepts_rtol_directly(self):
        """Sanity check that the installed scipy CG accepts ``rtol=``.

        Guards against accidental rollback to ``tol=`` in static.py.
        """
        from scipy.sparse import csc_matrix
        from scipy.sparse import linalg as spla

        # Trivial SPD system: identity, RHS = ones.
        n = 5
        K = csc_matrix(np.eye(n))
        F = np.ones(n)

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            u, info = spla.cg(K, F, rtol=1e-10, maxiter=100)

        assert info == 0
        np.testing.assert_allclose(u, np.ones(n), atol=1e-8)
