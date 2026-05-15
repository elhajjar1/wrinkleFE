"""Tests for WrinkleFE export utilities (JSON, Abaqus .inp, VTK).

Validates:
- JSON export produces valid JSON with all expected keys and finite values.
- Abaqus .inp export has valid node/element sections.
- VTK export has correct structure for ParaView.
- Round-trip: export JSON -> load -> verify field values match.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary
from wrinklefe.io.export import export_abaqus_inp, export_results_json, export_vtk


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture(scope="module")
def analysis_result():
    """Run a small WrinkleAnalysis once for the entire module."""
    mat = MaterialLibrary().get("IM7_8552")
    # NOTE: amplitude kept below the inversion threshold for this 1-element-
    # per-ply mesh; the original 0.366 mm value produced inverted hex
    # elements (see issue #45) and was silently masked by the old
    # abs(detJ) behaviour.
    cfg = AnalysisConfig(
        amplitude=0.25,
        wavelength=16.0,
        width=12.0,
        morphology="stack",
        loading="compression",
        material=mat,
        angles=[0, 45, -45, 90, 0, 45, -45, 0, 0, -45, 45, 0, 90, -45, 45, 0],
        ply_thickness=0.183,
        nx=4, ny=3, nz_per_ply=1,
        domain_length=20.0,
        domain_width=8.0,
        applied_strain=-0.005,
        verbose=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WrinkleAnalysis(cfg).run()


# ======================================================================
# JSON export tests
# ======================================================================

class TestJSONExport:
    """Tests for export_results_json."""

    def test_produces_valid_json(self, analysis_result, tmp_path):
        """Exported file is valid JSON."""
        out = tmp_path / "results.json"
        export_results_json(analysis_result, out)
        data = json.loads(out.read_text())
        assert isinstance(data, dict)

    def test_has_required_keys(self, analysis_result, tmp_path):
        """JSON contains all top-level keys."""
        out = tmp_path / "results.json"
        export_results_json(analysis_result, out)
        data = json.loads(out.read_text())
        assert "wrinklefe_version" in data
        assert "configuration" in data
        assert "analytical_predictions" in data

    def test_configuration_fields(self, analysis_result, tmp_path):
        """Configuration section has expected fields."""
        out = tmp_path / "results.json"
        export_results_json(analysis_result, out)
        cfg = json.loads(out.read_text())["configuration"]
        assert cfg["amplitude_mm"] == pytest.approx(0.25)
        assert cfg["wavelength_mm"] == pytest.approx(16.0)
        assert cfg["morphology"] == "stack"
        assert cfg["loading"] == "compression"
        assert cfg["material"] == "IM7_8552"
        assert cfg["n_plies"] == 16

    def test_predictions_are_finite(self, analysis_result, tmp_path):
        """All analytical prediction values are finite numbers."""
        out = tmp_path / "results.json"
        export_results_json(analysis_result, out)
        preds = json.loads(out.read_text())["analytical_predictions"]
        for key, val in preds.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"

    def test_knockdown_in_range(self, analysis_result, tmp_path):
        """Analytical knockdown is between 0 and 1."""
        out = tmp_path / "results.json"
        export_results_json(analysis_result, out)
        preds = json.loads(out.read_text())["analytical_predictions"]
        kd = preds["analytical_knockdown"]
        assert 0.0 < kd <= 1.0

    def test_mesh_summary_present(self, analysis_result, tmp_path):
        """Mesh summary is present when mesh was generated."""
        out = tmp_path / "results.json"
        export_results_json(analysis_result, out)
        data = json.loads(out.read_text())
        if analysis_result.mesh is not None:
            assert "mesh" in data
            mesh_data = data["mesh"]
            assert mesh_data["n_nodes"] > 0
            assert mesh_data["n_elements"] > 0

    def test_roundtrip_values(self, analysis_result, tmp_path):
        """Export and reload: values match the original result."""
        out = tmp_path / "results.json"
        export_results_json(analysis_result, out)
        data = json.loads(out.read_text())
        preds = data["analytical_predictions"]

        assert preds["analytical_knockdown"] == pytest.approx(
            analysis_result.analytical_knockdown, rel=1e-6
        )
        assert preds["max_angle_rad"] == pytest.approx(
            analysis_result.max_angle_rad, rel=1e-6
        )
        assert preds["analytical_strength_MPa"] == pytest.approx(
            analysis_result.analytical_strength_MPa, rel=1e-6
        )

    def test_creates_parent_directories(self, analysis_result, tmp_path):
        """Export creates intermediate directories if needed."""
        out = tmp_path / "subdir" / "deep" / "results.json"
        export_results_json(analysis_result, out)
        assert out.exists()


# ======================================================================
# Abaqus .inp export tests
# ======================================================================

class TestAbaqusExport:
    """Tests for export_abaqus_inp."""

    def test_file_created(self, analysis_result, tmp_path):
        """Abaqus file is created."""
        if analysis_result.mesh is None:
            pytest.skip("No mesh in result")
        out = tmp_path / "model.inp"
        laminate = analysis_result.laminate
        export_abaqus_inp(analysis_result.mesh, laminate, out)
        assert out.exists()

    def test_has_heading(self, analysis_result, tmp_path):
        """File starts with *HEADING section."""
        if analysis_result.mesh is None:
            pytest.skip("No mesh in result")
        out = tmp_path / "model.inp"
        export_abaqus_inp(analysis_result.mesh, analysis_result.laminate, out)
        content = out.read_text()
        assert content.startswith("*HEADING")

    def test_has_node_section(self, analysis_result, tmp_path):
        """File contains *NODE section with correct count."""
        if analysis_result.mesh is None:
            pytest.skip("No mesh in result")
        out = tmp_path / "model.inp"
        export_abaqus_inp(analysis_result.mesh, analysis_result.laminate, out)
        content = out.read_text()
        assert "*NODE" in content
        # Count node lines (between *NODE and next *)
        in_nodes = False
        node_count = 0
        for line in content.splitlines():
            if line.strip() == "*NODE":
                in_nodes = True
                continue
            if in_nodes:
                if line.startswith("*"):
                    break
                node_count += 1
        assert node_count == analysis_result.mesh.n_nodes

    def test_has_element_section(self, analysis_result, tmp_path):
        """File contains *ELEMENT sections."""
        if analysis_result.mesh is None:
            pytest.skip("No mesh in result")
        out = tmp_path / "model.inp"
        export_abaqus_inp(analysis_result.mesh, analysis_result.laminate, out)
        content = out.read_text()
        assert "*ELEMENT" in content

    def test_has_boundary_nsets(self, analysis_result, tmp_path):
        """File contains boundary node sets."""
        if analysis_result.mesh is None:
            pytest.skip("No mesh in result")
        out = tmp_path / "model.inp"
        export_abaqus_inp(analysis_result.mesh, analysis_result.laminate, out)
        content = out.read_text()
        for face in ("X_MIN", "X_MAX", "Z_MIN", "Z_MAX"):
            assert f"*NSET, NSET={face}" in content

    def test_node_ids_one_based(self, analysis_result, tmp_path):
        """Node IDs start at 1 (Abaqus convention)."""
        if analysis_result.mesh is None:
            pytest.skip("No mesh in result")
        out = tmp_path / "model.inp"
        export_abaqus_inp(analysis_result.mesh, analysis_result.laminate, out)
        content = out.read_text()
        # First node line after *NODE
        for line in content.splitlines():
            if line.strip() == "*NODE":
                continue
            parts = line.strip().split(",")
            if len(parts) >= 4:
                first_id = int(parts[0].strip())
                assert first_id == 1, "First node ID should be 1"
                break


# ======================================================================
# VTK export tests
# ======================================================================

class TestVTKExport:
    """Tests for export_vtk."""

    def test_file_created(self, analysis_result, tmp_path):
        """VTK file is created."""
        if analysis_result.mesh is None:
            pytest.skip("No mesh in result")
        out = tmp_path / "mesh.vtk"
        export_vtk(analysis_result.mesh, analysis_result.field_results, out)
        assert out.exists()

    def test_has_vtk_header(self, analysis_result, tmp_path):
        """File has proper VTK header."""
        if analysis_result.mesh is None:
            pytest.skip("No mesh in result")
        out = tmp_path / "mesh.vtk"
        export_vtk(analysis_result.mesh, analysis_result.field_results, out)
        content = out.read_text()
        assert "# vtk DataFile Version" in content
        assert "UNSTRUCTURED_GRID" in content

    def test_correct_point_count(self, analysis_result, tmp_path):
        """POINTS count matches mesh."""
        if analysis_result.mesh is None:
            pytest.skip("No mesh in result")
        out = tmp_path / "mesh.vtk"
        export_vtk(analysis_result.mesh, analysis_result.field_results, out)
        content = out.read_text()
        n = analysis_result.mesh.n_nodes
        assert f"POINTS {n}" in content

    def test_correct_cell_count(self, analysis_result, tmp_path):
        """CELLS count matches mesh."""
        if analysis_result.mesh is None:
            pytest.skip("No mesh in result")
        out = tmp_path / "mesh.vtk"
        export_vtk(analysis_result.mesh, analysis_result.field_results, out)
        content = out.read_text()
        n = analysis_result.mesh.n_elements
        assert f"CELLS {n}" in content

    def test_has_fiber_angle_data(self, analysis_result, tmp_path):
        """File contains fiber_angle point data."""
        if analysis_result.mesh is None:
            pytest.skip("No mesh in result")
        out = tmp_path / "mesh.vtk"
        export_vtk(analysis_result.mesh, analysis_result.field_results, out)
        content = out.read_text()
        assert "SCALARS fiber_angle" in content

    def test_has_ply_id_data(self, analysis_result, tmp_path):
        """File contains ply_id cell data."""
        if analysis_result.mesh is None:
            pytest.skip("No mesh in result")
        out = tmp_path / "mesh.vtk"
        export_vtk(analysis_result.mesh, analysis_result.field_results, out)
        content = out.read_text()
        assert "SCALARS ply_id" in content

    def test_vtk_without_field_results(self, analysis_result, tmp_path):
        """VTK export works with field_results=None."""
        if analysis_result.mesh is None:
            pytest.skip("No mesh in result")
        out = tmp_path / "mesh_no_fields.vtk"
        export_vtk(analysis_result.mesh, None, out)
        content = out.read_text()
        assert "SCALARS fiber_angle" in content
        # Should not have displacement vectors
        assert "displacement" not in content

    def test_stress_fields_are_true_element_averages(self, tmp_path):
        """Regression for #52: exported stress_* is the mean over the
        element's Gauss points, not the first Gauss point only.

        Builds a tiny synthetic mesh + FieldResults where the Gauss-point
        stresses differ within each element, then parses the VTK cell-data
        block and asserts every exported component equals the per-element
        Gauss-point mean (not stress_global[eid, 0, comp]).
        """
        from wrinklefe.solver.results import FieldResults

        n_nodes = 8
        n_elem = 1
        n_gauss = 8

        class _Mesh:
            pass

        mesh = _Mesh()
        # Unit cube hex8
        mesh.nodes = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=float,
        )
        mesh.elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=int)
        mesh.n_nodes = n_nodes
        mesh.n_elements = n_elem
        mesh.fiber_angles = np.zeros(n_nodes)
        mesh.ply_ids = np.zeros(n_elem, dtype=int)
        mesh.ply_angles = np.zeros(n_elem)

        rng = np.random.default_rng(52)
        # Deliberately distinct stresses at every Gauss point/component so
        # the mean differs from the first Gauss point's value.
        stress_global = rng.normal(scale=100.0, size=(n_elem, n_gauss, 6))
        # Make GP0 a clear outlier so a "first GP" bug would be obvious.
        stress_global[0, 0, :] += 5000.0

        zeros = np.zeros((n_elem, n_gauss, 6))
        field = FieldResults(
            displacement=np.zeros((n_nodes, 3)),
            stress_global=stress_global,
            stress_local=zeros.copy(),
            strain_global=zeros.copy(),
            strain_local=zeros.copy(),
            mesh=mesh,
            laminate=None,
        )

        out = tmp_path / "synthetic.vtk"
        export_vtk(mesh, field, out)
        lines = out.read_text().splitlines()

        # Parse each "SCALARS <name> double 1" block (one value, n_elem==1).
        exported: dict[str, float] = {}
        for i, line in enumerate(lines):
            if line.startswith("SCALARS stress_"):
                name = line.split()[1]
                # line i+1 is LOOKUP_TABLE, line i+2 is the value
                exported[name] = float(lines[i + 2])

        # Voigt order of stress_global columns.
        comp_index = {
            "stress_11": 0,
            "stress_22": 1,
            "stress_33": 2,
            "stress_23": 3,
            "stress_13": 4,
            "stress_12": 5,
        }
        assert set(exported) == set(comp_index), (
            f"Expected all 6 stress components, got {sorted(exported)}"
        )

        for name, col in comp_index.items():
            expected_mean = stress_global[0, :, col].mean()
            first_gp = stress_global[0, 0, col]
            assert exported[name] == pytest.approx(expected_mean, rel=1e-6), (
                f"{name}: exported {exported[name]} != element mean "
                f"{expected_mean}"
            )
            # Guard: the mean must be distinguishable from the old
            # first-Gauss-point behaviour for this fixture.
            assert abs(expected_mean - first_gp) > 1.0
