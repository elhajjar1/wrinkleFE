"""Tests for :mod:`wrinklefe.io.results` (CSV / JSON export of results).

Covers the v1.0 structured-export pair added for issue #2:

- :func:`export_results_json` writes a deterministic, schema-versioned
  JSON file containing the analytical predictions, per-ply table, FPF
  summary, and knockdown factors.
- :func:`export_results_csv` writes the per-ply table as a Pandas-
  friendly CSV.

The tests round-trip both formats through the stdlib (``json.load``,
``csv.DictReader``) and assert the documented fields are present and
parse as plain Python types.
"""

from __future__ import annotations

import csv
import json
import warnings

import numpy as np
import pytest

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary
from wrinklefe.io.results import (
    SCHEMA_VERSION,
    export_results_csv,
    export_results_json,
    results_to_dict,
)

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def fe_result():
    """Run a tiny end-to-end FE analysis once for the module.

    Uses a 16-ply layup and a mesh small enough to keep the test fast
    (~a few seconds) while still producing a real
    ``LaminateFailureReport`` and FE field, so the per-ply / FE branches
    of the exporter are exercised.
    """
    mat = MaterialLibrary().get("IM7_8552")
    cfg = AnalysisConfig(
        amplitude=0.25,
        wavelength=16.0,
        width=12.0,
        morphology="stack",
        loading="compression",
        material=mat,
        angles=[0, 45, -45, 90, 0, 45, -45, 0,
                0, -45, 45, 0, 90, -45, 45, 0],
        ply_thickness=0.183,
        nx=4, ny=3, nz_per_ply=1,
        domain_length=20.0,
        domain_width=8.0,
        applied_strain=-0.005,
        analytical_only=False,
        verbose=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WrinkleAnalysis(cfg).run()


@pytest.fixture(scope="module")
def analytical_result():
    """A pure analytical run (no mesh, no FE) for the schema-stability
    tests that must work even when ``failure_report is None``."""
    cfg = AnalysisConfig(
        amplitude=0.3,
        wavelength=15.0,
        width=10.0,
        morphology="stack",
        loading="compression",
        angles=[0, 45, -45, 90, 90, -45, 45, 0],
        analytical_only=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WrinkleAnalysis(cfg).run(analytical_only=True)


# ----------------------------------------------------------------------
# JSON tests
# ----------------------------------------------------------------------

class TestExportResultsJSON:
    """Round-trip + schema tests for export_results_json."""

    def test_writes_valid_json(self, fe_result, tmp_path):
        """The file is parseable by json.load."""
        out = tmp_path / "results.json"
        export_results_json(fe_result, out)
        data = json.loads(out.read_text())
        assert isinstance(data, dict)

    def test_has_documented_top_level_fields(self, fe_result, tmp_path):
        """Every field documented in the issue schema is present."""
        out = tmp_path / "results.json"
        export_results_json(fe_result, out)
        data = json.loads(out.read_text())
        for key in (
            "schema_version",
            "config",
            "load_factor",
            "first_ply_failure",
            "per_ply",
            "knockdown_factors",
        ):
            assert key in data, f"missing documented field: {key}"

    def test_schema_version_is_set(self, fe_result, tmp_path):
        out = tmp_path / "r.json"
        export_results_json(fe_result, out)
        assert json.loads(out.read_text())["schema_version"] == SCHEMA_VERSION

    def test_per_ply_row_shape(self, fe_result, tmp_path):
        """One row per ply, with the documented columns."""
        out = tmp_path / "r.json"
        export_results_json(fe_result, out)
        data = json.loads(out.read_text())
        per_ply = data["per_ply"]
        assert isinstance(per_ply, list)
        assert len(per_ply) == len(fe_result.config.angles)
        for row in per_ply:
            for col in ("index", "angle_deg", "max_FI", "min_RF",
                        "critical_mode", "critical_criterion"):
                assert col in row

    def test_per_ply_indices_are_dense_and_sorted(self, fe_result, tmp_path):
        out = tmp_path / "r.json"
        export_results_json(fe_result, out)
        per_ply = json.loads(out.read_text())["per_ply"]
        assert [row["index"] for row in per_ply] == list(range(len(per_ply)))

    def test_first_ply_failure_payload(self, fe_result, tmp_path):
        """FPF block matches the documented shape when FE is present."""
        out = tmp_path / "r.json"
        export_results_json(fe_result, out)
        fpf = json.loads(out.read_text())["first_ply_failure"]
        assert fpf is not None
        for key in ("ply_index", "criterion", "mode", "load_factor"):
            assert key in fpf
        assert isinstance(fpf["ply_index"], int)
        assert isinstance(fpf["criterion"], str)

    def test_knockdown_factors_block(self, fe_result, tmp_path):
        out = tmp_path / "r.json"
        export_results_json(fe_result, out)
        kd = json.loads(out.read_text())["knockdown_factors"]
        assert "analytical" in kd
        assert 0.0 < kd["analytical"] <= 1.0
        # FE branch is populated whenever there are retention factors.
        if fe_result.retention_factors:
            assert "fe" in kd

    def test_load_factor_is_finite_float(self, fe_result, tmp_path):
        out = tmp_path / "r.json"
        export_results_json(fe_result, out)
        lf = json.loads(out.read_text())["load_factor"]
        assert isinstance(lf, float)
        assert np.isfinite(lf)

    def test_no_numpy_scalars_in_json(self, fe_result, tmp_path):
        """Round-trip cleanly through json.load: all leaves are plain
        Python types (int/float/str/bool/None/list/dict)."""
        out = tmp_path / "r.json"
        export_results_json(fe_result, out)
        data = json.loads(out.read_text())

        allowed = (bool, int, float, str, type(None))

        def _walk(node):
            if isinstance(node, dict):
                for v in node.values():
                    _walk(v)
            elif isinstance(node, list):
                for v in node:
                    _walk(v)
            else:
                assert isinstance(node, allowed), (
                    f"non-JSON-native leaf: {type(node).__name__}: {node!r}"
                )

        _walk(data)

    def test_output_is_deterministic(self, fe_result, tmp_path):
        """Same input -> byte-identical output (sort_keys=True)."""
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        export_results_json(fe_result, a)
        export_results_json(fe_result, b)
        assert a.read_bytes() == b.read_bytes()

    def test_stress_field_summarised_not_inlined(self, fe_result, tmp_path):
        """Per-Gauss-point stress arrays are reduced to summary stats so
        the JSON stays small."""
        out = tmp_path / "r.json"
        export_results_json(fe_result, out)
        data = json.loads(out.read_text())
        fe_block = data.get("fe")
        assert fe_block is not None
        ss = fe_block["stress_field_summary"]
        for comp in ("stress_11", "stress_22", "stress_33",
                     "stress_23", "stress_13", "stress_12"):
            assert comp in ss
            for stat in ("min", "max", "mean", "p95", "n"):
                assert stat in ss[comp]

    def test_analytical_only_run_has_stable_schema(
        self, analytical_result, tmp_path
    ):
        """A pure analytical run still produces all top-level fields,
        with FE/mesh sections omitted and FPF as null."""
        out = tmp_path / "r.json"
        export_results_json(analytical_result, out)
        data = json.loads(out.read_text())
        assert data["schema_version"] == SCHEMA_VERSION
        assert data["first_ply_failure"] is None
        assert "fe" not in data
        assert "mesh" not in data
        # per_ply rows are still there, FI columns are null
        assert len(data["per_ply"]) == len(analytical_result.config.angles)
        assert data["per_ply"][0]["max_FI"] is None

    def test_creates_parent_dirs(self, fe_result, tmp_path):
        out = tmp_path / "sub" / "deep" / "results.json"
        export_results_json(fe_result, out)
        assert out.exists()

    def test_results_to_dict_matches_file(self, fe_result, tmp_path):
        """results_to_dict() is the same payload as the file content."""
        out = tmp_path / "r.json"
        export_results_json(fe_result, out)
        from_file = json.loads(out.read_text())
        in_memory = json.loads(json.dumps(
            results_to_dict(fe_result),
            sort_keys=True,
            default=lambda o: float(o) if hasattr(o, "__float__") else str(o),
        ))
        assert from_file == in_memory


# ----------------------------------------------------------------------
# CSV tests
# ----------------------------------------------------------------------

class TestExportResultsCSV:
    """Round-trip + schema tests for export_results_csv."""

    def test_writes_file(self, fe_result, tmp_path):
        out = tmp_path / "per_ply.csv"
        export_results_csv(fe_result, out)
        assert out.exists()

    def test_row_count_matches_ply_count(self, fe_result, tmp_path):
        out = tmp_path / "per_ply.csv"
        export_results_csv(fe_result, out)
        with open(out, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == len(fe_result.config.angles)

    def test_columns_match_schema(self, fe_result, tmp_path):
        out = tmp_path / "per_ply.csv"
        export_results_csv(fe_result, out)
        with open(out, newline="") as fh:
            reader = csv.DictReader(fh)
            assert reader.fieldnames == [
                "ply_index",
                "angle_deg",
                "max_FI",
                "min_RF",
                "critical_mode",
                "critical_criterion",
            ]

    def test_dictreader_parses_cleanly(self, fe_result, tmp_path):
        """csv.DictReader returns one dict per ply with the right keys."""
        out = tmp_path / "per_ply.csv"
        export_results_csv(fe_result, out)
        with open(out, newline="") as fh:
            rows = list(csv.DictReader(fh))
        for i, row in enumerate(rows):
            assert int(row["ply_index"]) == i
            # Angle column is always populated.
            assert row["angle_deg"] != ""
            float(row["angle_deg"])

    def test_angle_values_match_config(self, fe_result, tmp_path):
        out = tmp_path / "per_ply.csv"
        export_results_csv(fe_result, out)
        with open(out, newline="") as fh:
            rows = list(csv.DictReader(fh))
        for i, ang in enumerate(fe_result.config.angles):
            assert float(rows[i]["angle_deg"]) == pytest.approx(ang)

    def test_csv_for_analytical_only_run(self, analytical_result, tmp_path):
        """CSV is still well-formed when no failure_report is attached."""
        out = tmp_path / "per_ply.csv"
        export_results_csv(analytical_result, out)
        with open(out, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == len(analytical_result.config.angles)
        # FI / mode columns are blank (the schema is stable).
        for row in rows:
            assert row["max_FI"] == ""
            assert row["critical_mode"] == ""

    def test_creates_parent_dirs(self, fe_result, tmp_path):
        out = tmp_path / "sub" / "deep" / "per_ply.csv"
        export_results_csv(fe_result, out)
        assert out.exists()
