"""Tests for wrinklefe.core.mesh module."""

import numpy as np
import numpy.testing as npt
import pytest

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.core.laminate import Laminate
from wrinklefe.core.wrinkle import GaussianSinusoidal
from wrinklefe.core.morphology import WrinkleConfiguration
from wrinklefe.core.mesh import MeshData, WrinkleMesh


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
        profile = GaussianSinusoidal(amplitude=0.366, wavelength=16.0, width=12.0, center=5.0)
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
