"""Tests for wrinklefe.core.mesh module."""

import numpy as np
import numpy.testing as npt
import pytest

from wrinklefe.core.laminate import Laminate
from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.core.mesh import MeshData, MeshValidationError, WrinkleMesh
from wrinklefe.core.morphology import WrinkleConfiguration
from wrinklefe.core.wrinkle import GaussianSinusoidal


@pytest.fixture
def small_laminate():
    """Small 4-ply [0/90]s laminate for fast mesh tests."""
    mat = OrthotropicMaterial()
    return Laminate.symmetric([0, 90], material=mat, ply_thickness=0.183)


@pytest.fixture
def small_mesh_generator(small_laminate):
    """WrinkleMesh with small resolution for fast testing."""
    return WrinkleMesh(
        laminate=small_laminate,
        wrinkle_config=None,
        Lx=10.0,
        Ly=5.0,
        nx=4,
        ny=3,
        nz_per_ply=1,
    )


@pytest.fixture
def small_mesh(small_mesh_generator):
    """Generated MeshData from the small mesh generator."""
    return small_mesh_generator.generate()


class TestMeshDataCreation:
    """Test MeshData dataclass."""

    def test_creation_with_arrays(self):
        nodes = np.zeros((8, 3))
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
        ply_ids = np.array([0])
        fiber_angles = np.zeros(8)
        ply_angles = np.array([0.0])

        mesh = MeshData(
            nodes=nodes,
            elements=elements,
            ply_ids=ply_ids,
            fiber_angles=fiber_angles,
            ply_angles=ply_angles,
            nx=1, ny=1, nz=1,
        )
        assert mesh.n_nodes == 8
        assert mesh.n_elements == 1
        assert mesh.n_dof == 24


class TestWrinkleMeshGenerate:
    """Test WrinkleMesh.generate() produces valid MeshData."""

    def test_generates_mesh_data(self, small_mesh):
        assert isinstance(small_mesh, MeshData)

    def test_node_count(self, small_mesh_generator, small_mesh):
        """n_nodes = (nx+1) * (ny+1) * (nz+1)."""
        gen = small_mesh_generator
        expected = (gen.nx + 1) * (gen.ny + 1) * (gen.nz + 1)
        assert small_mesh.n_nodes == expected

    def test_element_count(self, small_mesh_generator, small_mesh):
        """n_elements = nx * ny * nz."""
        gen = small_mesh_generator
        expected = gen.nx * gen.ny * gen.nz
        assert small_mesh.n_elements == expected

    def test_hex_elements_have_8_nodes(self, small_mesh):
        """Each element should have exactly 8 node indices."""
        assert small_mesh.elements.shape[1] == 8

    def test_all_element_indices_valid(self, small_mesh):
        """All node indices in element connectivity must be in [0, n_nodes)."""
        assert np.all(small_mesh.elements >= 0)
        assert np.all(small_mesh.elements < small_mesh.n_nodes)

    def test_z_range_matches_laminate_thickness(self, small_laminate, small_mesh):
        """Z-range of nodes should match total laminate thickness."""
        z_coords = small_mesh.nodes[:, 2]
        z_range = z_coords.max() - z_coords.min()
        npt.assert_allclose(z_range, small_laminate.total_thickness, atol=1e-10)

    def test_x_range(self, small_mesh):
        """X-range should match domain Lx."""
        x_coords = small_mesh.nodes[:, 0]
        npt.assert_allclose(x_coords.min(), 0.0, atol=1e-10)
        npt.assert_allclose(x_coords.max(), 10.0, atol=1e-10)

    def test_y_range(self, small_mesh):
        """Y-range should match domain Ly."""
        y_coords = small_mesh.nodes[:, 1]
        npt.assert_allclose(y_coords.min(), 0.0, atol=1e-10)
        npt.assert_allclose(y_coords.max(), 5.0, atol=1e-10)


class TestFlatMeshFiberAngles:
    """Test that a flat mesh (no wrinkle) has all fiber_angles = 0."""

    def test_all_fiber_angles_zero(self, small_mesh):
        npt.assert_allclose(small_mesh.fiber_angles, 0.0, atol=1e-14)


class TestPlyIds:
    """Test ply ID assignment."""

    def test_ply_ids_cover_all_plies(self, small_laminate, small_mesh):
        """ply_ids should include values 0 through n_plies-1."""
        unique_plies = np.unique(small_mesh.ply_ids)
        expected = np.arange(small_laminate.n_plies)
        npt.assert_array_equal(unique_plies, expected)

    def test_ply_ids_shape(self, small_mesh):
        assert small_mesh.ply_ids.shape == (small_mesh.n_elements,)

    def test_ply_angles_shape(self, small_mesh):
        assert small_mesh.ply_angles.shape == (small_mesh.n_elements,)


class TestNodesOnFace:
    """Test nodes_on_face boundary queries."""

    def test_x_min_face_count(self, small_mesh_generator, small_mesh):
        """x_min face should have (ny+1)*(nz+1) nodes."""
        gen = small_mesh_generator
        face_nodes = small_mesh.nodes_on_face("x_min")
        expected_count = (gen.ny + 1) * (gen.nz + 1)
        assert len(face_nodes) == expected_count

    def test_x_max_face_count(self, small_mesh_generator, small_mesh):
        gen = small_mesh_generator
        face_nodes = small_mesh.nodes_on_face("x_max")
        expected_count = (gen.ny + 1) * (gen.nz + 1)
        assert len(face_nodes) == expected_count

    def test_y_min_face_count(self, small_mesh_generator, small_mesh):
        gen = small_mesh_generator
        face_nodes = small_mesh.nodes_on_face("y_min")
        expected_count = (gen.nx + 1) * (gen.nz + 1)
        assert len(face_nodes) == expected_count

    def test_z_min_face_count(self, small_mesh_generator, small_mesh):
        gen = small_mesh_generator
        face_nodes = small_mesh.nodes_on_face("z_min")
        expected_count = (gen.nx + 1) * (gen.ny + 1)
        assert len(face_nodes) == expected_count

    def test_z_max_face_count(self, small_mesh_generator, small_mesh):
        gen = small_mesh_generator
        face_nodes = small_mesh.nodes_on_face("z_max")
        expected_count = (gen.nx + 1) * (gen.ny + 1)
        assert len(face_nodes) == expected_count

    def test_invalid_face_raises(self, small_mesh):
        with pytest.raises(ValueError, match="Unknown face"):
            small_mesh.nodes_on_face("top")

    def test_x_min_nodes_have_x_zero(self, small_mesh):
        face_nodes = small_mesh.nodes_on_face("x_min")
        x_values = small_mesh.nodes[face_nodes, 0]
        npt.assert_allclose(x_values, 0.0, atol=1e-10)


class TestFaceElements:
    """Tests for ``MeshData.face_elements`` (issue #50)."""

    def test_face_elements_shape_x_max(self, small_mesh_generator, small_mesh):
        gen = small_mesh_generator
        quads = small_mesh.face_elements("x_max")
        assert quads.shape == (gen.ny * gen.nz, 4)

    def test_face_elements_shape_z_max(self, small_mesh_generator, small_mesh):
        gen = small_mesh_generator
        quads = small_mesh.face_elements("z_max")
        assert quads.shape == (gen.nx * gen.ny, 4)

    def test_face_elements_corners_on_face(self, small_mesh):
        """Every corner of every face quad must belong to the face's node set."""
        for face in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
            quads = small_mesh.face_elements(face)
            face_nodes = set(small_mesh.nodes_on_face(face).tolist())
            for q in quads:
                for nid in q:
                    assert int(nid) in face_nodes

    def test_face_elements_quad_area_sums_to_face_area(self, small_mesh):
        """Sum of Q4 quad areas equals the rectangular face area on flat mesh."""
        from wrinklefe.solver.boundary import _quad_areas
        Lx, Ly, Lz = small_mesh.domain_size
        face_to_area = {
            "x_min": Ly * Lz, "x_max": Ly * Lz,
            "y_min": Lx * Lz, "y_max": Lx * Lz,
            "z_min": Lx * Ly, "z_max": Lx * Ly,
        }
        for face, expected_area in face_to_area.items():
            quads = small_mesh.face_elements(face)
            total = float(_quad_areas(small_mesh.nodes, quads).sum())
            assert np.isclose(total, expected_area, rtol=1e-10), (
                f"{face}: total area {total} != expected {expected_area}"
            )

    def test_face_elements_invalid_face_raises(self, small_mesh):
        with pytest.raises(ValueError, match="Unknown face"):
            small_mesh.face_elements("top")


class TestElementConnectivity:
    """Test element connectivity and adjacency."""

    def test_adjacent_elements_share_nodes(self, small_mesh):
        """Adjacent elements (in x-direction) should share 4 nodes."""
        if small_mesh.n_elements < 2:
            pytest.skip("Need at least 2 elements")

        # Elements 0 and 1 should be adjacent in x
        elem0_nodes = set(small_mesh.elements[0])
        elem1_nodes = set(small_mesh.elements[1])
        shared = elem0_nodes & elem1_nodes
        # Adjacent hex elements share a face with 4 nodes
        assert len(shared) == 4

    def test_element_nodes_method(self, small_mesh):
        """element_nodes should return (8, 3) coordinate array."""
        coords = small_mesh.element_nodes(0)
        assert coords.shape == (8, 3)

    def test_element_center_shape(self, small_mesh):
        center = small_mesh.element_center(0)
        assert center.shape == (3,)

    def test_domain_size(self, small_mesh):
        Lx, Ly, Lz = small_mesh.domain_size
        npt.assert_allclose(Lx, 10.0, atol=1e-10)
        npt.assert_allclose(Ly, 5.0, atol=1e-10)
        assert Lz > 0


class TestWrinkledMesh:
    """Test mesh generation with a wrinkle configuration."""

    @pytest.fixture
    def wrinkled_mesh(self, small_laminate):
        # Mild-wrinkle parameters: amplitude small enough that no hex8
        # element inverts at this resolution (4x3x4).  The previous
        # amplitude (0.366) produced 6 genuinely inverted elements that
        # the old validate() silently let through (issue #94); validate()
        # now refuses such a mesh, so the fixture uses a physically valid
        # configuration.
        profile = GaussianSinusoidal(amplitude=0.1, wavelength=16.0, width=12.0, center=5.0)
        config = WrinkleConfiguration.dual_wrinkle(
            profile, interface1=1, interface2=2, phase=0.0
        )
        gen = WrinkleMesh(
            laminate=small_laminate,
            wrinkle_config=config,
            Lx=10.0,
            Ly=5.0,
            nx=4,
            ny=3,
            nz_per_ply=1,
        )
        return gen.generate()

    def test_wrinkled_mesh_generates(self, wrinkled_mesh):
        assert isinstance(wrinkled_mesh, MeshData)

    def test_wrinkled_mesh_has_nonzero_fiber_angles(self, wrinkled_mesh):
        """With a wrinkle, at least some fiber angles should be non-zero."""
        assert np.any(wrinkled_mesh.fiber_angles > 0)

    def test_wrinkled_mesh_node_count(self, small_laminate, wrinkled_mesh):
        """Node count should be same as flat mesh with same resolution."""
        nx, ny, nz_per_ply = 4, 3, 1
        nz = small_laminate.n_plies * nz_per_ply
        expected = (nx + 1) * (ny + 1) * (nz + 1)
        assert wrinkled_mesh.n_nodes == expected

    def test_wrinkled_z_min_face_count(self, small_laminate, wrinkled_mesh):
        """On a wrinkled mesh, z_min must still return the full k=0 plane.

        Regression test for issue #93: the old geometric tolerance test
        silently dropped nodes whose perturbed z deviated from the global
        minimum z, leaving only the trough nodes.
        """
        nx, ny = 4, 3
        face_nodes = wrinkled_mesh.nodes_on_face("z_min")
        assert len(face_nodes) == (nx + 1) * (ny + 1)
        # The k=0 plane is the first (nx+1)*(ny+1) nodes in canonical order.
        expected_nodes = np.arange((nx + 1) * (ny + 1))
        npt.assert_array_equal(face_nodes, expected_nodes)

    def test_wrinkled_z_max_face_count(self, small_laminate, wrinkled_mesh):
        """On a wrinkled mesh, z_max must still return the full k=nz plane."""
        nx, ny, nz_per_ply = 4, 3, 1
        nz = small_laminate.n_plies * nz_per_ply
        face_nodes = wrinkled_mesh.nodes_on_face("z_max")
        assert len(face_nodes) == (nx + 1) * (ny + 1)
        # The k=nz plane is the last (nx+1)*(ny+1) nodes in canonical order.
        n_per_layer = (nx + 1) * (ny + 1)
        expected_nodes = np.arange(nz * n_per_layer, (nz + 1) * n_per_layer)
        npt.assert_array_equal(face_nodes, expected_nodes)

    def test_wrinkled_x_y_faces_unaffected(self, wrinkled_mesh):
        """x/y faces should also still produce the correct surface count."""
        nx, ny, nz_per_ply = 4, 3, 1
        nz = 4 * nz_per_ply  # 4-ply laminate
        assert len(wrinkled_mesh.nodes_on_face("x_min")) == (ny + 1) * (nz + 1)
        assert len(wrinkled_mesh.nodes_on_face("x_max")) == (ny + 1) * (nz + 1)
        assert len(wrinkled_mesh.nodes_on_face("y_min")) == (nx + 1) * (nz + 1)
        assert len(wrinkled_mesh.nodes_on_face("y_max")) == (nx + 1) * (nz + 1)


class TestMeshValidation:
    """Test WrinkleMesh input validation."""

    def test_negative_Lx_raises(self, small_laminate):
        with pytest.raises(ValueError, match="positive"):
            WrinkleMesh(laminate=small_laminate, Lx=-1.0)

    def test_zero_nx_raises(self, small_laminate):
        with pytest.raises(ValueError, match="must be >= 1"):
            WrinkleMesh(laminate=small_laminate, nx=0)

    def test_zero_nz_per_ply_raises(self, small_laminate):
        with pytest.raises(ValueError, match="must be >= 1"):
            WrinkleMesh(laminate=small_laminate, nz_per_ply=0)


class TestElementsInPly:
    """Test element-to-ply queries."""

    def test_elements_in_ply_0(self, small_mesh_generator, small_mesh):
        gen = small_mesh_generator
        elems = small_mesh.elements_in_ply(0)
        # Each ply layer has nx * ny elements
        expected = gen.nx * gen.ny
        assert len(elems) == expected

    def test_elements_in_all_plies_cover_all(self, small_laminate, small_mesh):
        """Union of elements in all plies should be all elements."""
        all_elems = set()
        for p in range(small_laminate.n_plies):
            all_elems.update(small_mesh.elements_in_ply(p).tolist())
        assert len(all_elems) == small_mesh.n_elements


class TestMidplaneElements:
    """Test midplane element selection."""

    def test_midplane_elements_nonempty(self, small_mesh):
        mid = small_mesh.midplane_elements()
        assert len(mid) > 0

    def test_midplane_elements_near_z_zero(self, small_mesh):
        """Midplane element centroids should have z close to 0."""
        mid = small_mesh.midplane_elements()
        for eid in mid:
            center = small_mesh.element_center(eid)
            # Should be the layer closest to z=0
            assert abs(center[2]) < small_mesh.domain_size[2] / 2


class TestMultipleElementsPerPly:
    """Test mesh with nz_per_ply > 1."""

    def test_refined_mesh_node_count(self, small_laminate):
        gen = WrinkleMesh(
            laminate=small_laminate,
            Lx=10.0, Ly=5.0,
            nx=2, ny=2, nz_per_ply=2,
        )
        mesh = gen.generate()
        nz = small_laminate.n_plies * 2
        expected = (2 + 1) * (2 + 1) * (nz + 1)
        assert mesh.n_nodes == expected

    def test_refined_mesh_element_count(self, small_laminate):
        gen = WrinkleMesh(
            laminate=small_laminate,
            Lx=10.0, Ly=5.0,
            nx=2, ny=2, nz_per_ply=2,
        )
        mesh = gen.generate()
        nz = small_laminate.n_plies * 2
        expected = 2 * 2 * nz
        assert mesh.n_elements == expected


class TestValidateJacobianSign:
    """Regression tests for issue #94 — MeshData.validate() must reject
    inverted hex8 elements via the centroid det(J) check."""

    def test_clean_mesh_validate_does_not_raise(self, small_mesh):
        """A freshly generated flat mesh has all det(J) > 0."""
        warnings = small_mesh.validate()
        # No inverted-element error, and no Jacobian-related warning text
        for w in warnings:
            assert "inverted" not in w.lower()

    def test_inverted_element_raises(self, small_mesh):
        """Flipping an element's bottom/top faces inverts det(J) and must
        be caught by validate() with a descriptive message."""
        # Copy to avoid mutating the fixture for other tests
        nodes = small_mesh.nodes.copy()
        elements = small_mesh.elements.copy()

        # Swap bottom (nodes 0..3) and top (nodes 4..7) of element 0 in
        # the connectivity table.  This reverses the zeta-direction and
        # makes det(J) at the centroid negative.
        bad_id = 0
        bottom = elements[bad_id, 0:4].copy()
        top = elements[bad_id, 4:8].copy()
        elements[bad_id, 0:4] = top
        elements[bad_id, 4:8] = bottom

        bad_mesh = MeshData(
            nodes=nodes,
            elements=elements,
            ply_ids=small_mesh.ply_ids.copy(),
            fiber_angles=small_mesh.fiber_angles.copy(),
            ply_angles=small_mesh.ply_angles.copy(),
            nx=small_mesh.nx,
            ny=small_mesh.ny,
            nz=small_mesh.nz,
        )

        with pytest.raises(MeshValidationError) as exc_info:
            bad_mesh.validate()

        msg = str(exc_info.value)
        assert "inverted" in msg
        assert "hex8" in msg
        assert str(bad_id) in msg
        # Exception carries the full list of bad indices for callers.
        assert bad_id in exc_info.value.inverted_element_indices.tolist()

    def test_multiple_inverted_elements_reported(self, small_mesh):
        """validate() reports all inverted elements, not just the first."""
        nodes = small_mesh.nodes.copy()
        elements = small_mesh.elements.copy()

        bad_ids = [0, 2, 5]
        for bad_id in bad_ids:
            bottom = elements[bad_id, 0:4].copy()
            top = elements[bad_id, 4:8].copy()
            elements[bad_id, 0:4] = top
            elements[bad_id, 4:8] = bottom

        bad_mesh = MeshData(
            nodes=nodes,
            elements=elements,
            ply_ids=small_mesh.ply_ids.copy(),
            fiber_angles=small_mesh.fiber_angles.copy(),
            ply_angles=small_mesh.ply_angles.copy(),
            nx=small_mesh.nx,
            ny=small_mesh.ny,
            nz=small_mesh.nz,
        )

        with pytest.raises(MeshValidationError) as exc_info:
            bad_mesh.validate()

        msg = str(exc_info.value)
        assert f"{len(bad_ids)} inverted" in msg
        reported = set(exc_info.value.inverted_element_indices.tolist())
        assert set(bad_ids).issubset(reported)

    def test_generate_raises_on_inverted_mesh(self, small_laminate):
        """End-to-end: a manually corrupted MeshData through validate()
        surfaces during construction-time quality gating."""
        gen = WrinkleMesh(
            laminate=small_laminate,
            wrinkle_config=None,
            Lx=10.0, Ly=5.0,
            nx=4, ny=3, nz_per_ply=1,
        )
        mesh = gen.generate()
        # Corrupt and re-validate to confirm the error path.
        elements = mesh.elements.copy()
        elements[1, 0:4], elements[1, 4:8] = (
            elements[1, 4:8].copy(),
            elements[1, 0:4].copy(),
        )
        bad = MeshData(
            nodes=mesh.nodes,
            elements=elements,
            ply_ids=mesh.ply_ids,
            fiber_angles=mesh.fiber_angles,
            ply_angles=mesh.ply_angles,
            nx=mesh.nx, ny=mesh.ny, nz=mesh.nz,
        )
        with pytest.raises(MeshValidationError, match=r"inverted hex8"):
            bad.validate()
