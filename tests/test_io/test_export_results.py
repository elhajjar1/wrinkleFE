"""Tests for :mod:`wrinklefe.io.results` (CSV / JSON export of results).

Covers the structured-export pair added for issue #2:

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
from dataclasses import fields

import numpy as np
import pytest

from wrinklefe.analysis import AnalysisConfig, AnalysisResults, WrinkleAnalysis
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


# ----------------------------------------------------------------------
# Progressive-damage + modulus_retention_global export (issue #345)
# ----------------------------------------------------------------------

def _minimal_config():
    """A cheap analytical-only config for direct-construction tests."""
    return AnalysisConfig(
        angles=[0, 45, -45, 90, 90, -45, 45, 0],
        analytical_only=True,
    )


def _progressive_result():
    """AnalysisResults with the progressive-damage fields populated.

    Built by direct construction — the exporter only reads attributes, so
    no FE/progressive solve is needed to exercise the serialisation path.
    """
    return AnalysisResults(
        config=_minimal_config(),
        modulus_retention_global=0.87,
        tension_mechanisms={"kd_fiber": 0.91, "mode": "fiber"},
        retention_factors={"tsai_wu": 0.82, "hashin": 0.90},
        progressive_strength_MPa=812.5,
        progressive_pristine_strength_MPa=1050.0,
        progressive_knockdown=0.7738,
        progressive_history=[
            (0.0, 0.0),
            (0.001, 520.0),
            (0.002, 812.5),
            (0.003, 640.0),
        ],
    )


class TestProgressiveExport:
    """The progressive block appears only for progressive runs (#345)."""

    def test_progressive_block_present_with_values(self):
        payload = results_to_dict(_progressive_result())
        assert "progressive" in payload
        prog = payload["progressive"]
        assert prog["strength_MPa"] == pytest.approx(812.5)
        assert prog["pristine_strength_MPa"] == pytest.approx(1050.0)
        assert prog["knockdown"] == pytest.approx(0.7738)
        assert prog["n_increments"] == 4
        assert prog["history"][2] == [pytest.approx(0.002), pytest.approx(812.5)]

    def test_progressive_block_json_round_trips(self):
        payload = results_to_dict(_progressive_result())
        restored = json.loads(json.dumps(payload, sort_keys=True))
        assert restored["progressive"]["n_increments"] == 4
        assert restored["progressive"]["history"][1] == [0.001, 520.0]

    def test_no_progressive_block_for_analytical_run(self, analytical_result):
        """Analytical-only runs (history is None) carry no progressive key."""
        payload = results_to_dict(analytical_result)
        assert "progressive" not in payload

    def test_modulus_retention_global_in_knockdown_factors(self):
        payload = results_to_dict(_progressive_result())
        kd = payload["knockdown_factors"]
        assert kd["modulus_retention_global"] == pytest.approx(0.87)


# ----------------------------------------------------------------------
# Export drift guard — every AnalysisResults field is exported or
# explicitly allowlisted (prevents a recurrence of issue #345).
# ----------------------------------------------------------------------

#: AnalysisResults fields whose export key differs from the field name.
FIELD_TO_EXPORT_KEY = {
    "retention_factors": "knockdown_factors.fe_per_criterion",
    "progressive_strength_MPa": "progressive.strength_MPa",
    "progressive_pristine_strength_MPa": "progressive.pristine_strength_MPa",
    "progressive_knockdown": "progressive.knockdown",
    "progressive_history": "progressive.history",
}

#: Fields intentionally not serialised by results_to_dict, each with a
#: reason. Heavy objects / large arrays are summarised elsewhere or are
#: internal intermediates; CZM results are surfaced via the app's own
#: CZM payload rather than the structured results export.
INTENTIONALLY_UNEXPORTED = {
    "modulus_retention_failed": (
        "diagnostic flag; results_to_dict emits it only when the local "
        "modulus-retention computation failed, so it is absent for valid "
        "runs (keeps the export byte-identical / ledger zero-drift, #374)"
    ),
    "modulus_retention_global_failed": (
        "diagnostic flag; emitted only when the global modulus-retention "
        "computation failed, absent for valid runs (#374)"
    ),
    "czm_failure_diagnostics": (
        "Newton/CZM convergence-failure diagnostics; results_to_dict emits it "
        "only on a non-converged solve (None when converged), so it is absent "
        "for valid runs (keeps the export byte-identical / zero-drift, #262)"
    ),
    "czm_failure_hint": (
        "actionable tuning hint; emitted only on a non-converged CZM solve, "
        "absent for valid runs (#262)"
    ),
    "load_state_factor": (
        "proportional load factor under a general load state; "
        "results_to_dict emits it only when AnalysisConfig.load_state was "
        "set, so it is absent for every applied_strain run (keeps the "
        "export byte-identical / ledger zero-drift, #275)"
    ),
    "load_state_factor_pristine": (
        "flat-baseline load factor; emitted only alongside load_state_factor "
        "(#275)"
    ),
    "load_state_factor_knockdown": (
        "load_factor / load_state_factor_pristine; emitted only alongside "
        "load_factor (#275)"
    ),
    "mesh": "heavy MeshData; summarised as the top-level 'mesh' block when present",
    "wrinkle_config": "WrinkleConfiguration object; geometry captured under config",
    "laminate": "Laminate object; layup captured under config.angles",
    "field_results": "heavy FE fields; summarised as fe.stress_field_summary",
    "failure_report": "LaminateFailureReport; flattened into per_ply/first_ply_failure",
    "failure_indices": "per-criterion FE failure-index arrays (large)",
    "failure_modes": "per-criterion failure-mode string arrays (large)",
    "baseline_fi": "pristine per-criterion max FI; retention-calc intermediate",
    "mesh_max_angle_rad": "FE-mesh diagnostic; analytical max_angle_rad is exported",
    "czm_damage": "CZM result; surfaced via the app CZM payload, not results_to_dict",
    "czm_separation": "CZM result; surfaced via the app CZM payload",
    "czm_traction": "CZM result; surfaced via the app CZM payload",
    "czm_energy_dissipated": "CZM result; surfaced via the app CZM payload",
    "czm_energy_per_interface": "CZM result; surfaced via the app CZM payload",
    "czm_crack_length_per_interface": "CZM result; surfaced via the app CZM payload",
    "czm_load_displacement": "CZM result; surfaced via the app CZM payload",
    "czm_converged": "CZM result; surfaced via the app CZM payload",
    "czm_interfaces_used": "CZM result; surfaced via the app CZM payload",
    "czm_delamination_report": "CZM result; surfaced via the app CZM payload",
    "czm_element_centroids": "CZM result; surfaced via the app CZM payload",
}


def _collect_keys(node, acc):
    """Recursively gather every dict key appearing in a payload."""
    if isinstance(node, dict):
        for k, v in node.items():
            acc.add(k)
            _collect_keys(v, acc)
    elif isinstance(node, list):
        for v in node:
            _collect_keys(v, acc)


def _resolve(payload, dotted):
    """Follow a dotted export path (e.g. 'progressive.strength_MPa')."""
    node = payload
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def test_every_results_field_is_exported_or_allowlisted():
    """Walk AnalysisResults fields; each must be exported by
    results_to_dict (directly, or via a mapped key) or explicitly
    allowlisted. Building the representative payload needs no FE solve —
    the exporters only read attributes."""
    payload = results_to_dict(_progressive_result())
    keys: set[str] = set()
    _collect_keys(payload, keys)

    for f in fields(AnalysisResults):
        name = f.name
        if name in keys:
            continue
        if name in FIELD_TO_EXPORT_KEY:
            assert _resolve(payload, FIELD_TO_EXPORT_KEY[name]), (
                f"field {name!r} maps to {FIELD_TO_EXPORT_KEY[name]!r} but "
                "that path is missing from the results_to_dict payload"
            )
            continue
        assert name in INTENTIONALLY_UNEXPORTED, (
            f"field {name!r} is neither exported by results_to_dict nor "
            "allowlisted — wire it into io/results.py or add it to "
            "INTENTIONALLY_UNEXPORTED with a reason."
        )


# ----------------------------------------------------------------------
# Provenance (schema 1.2) and the relationship to the legacy exporter
# ----------------------------------------------------------------------

class TestProvenance:
    """The structured document must support a reproducibility claim.

    Before schema 1.2 the ``provenance`` block existed only in the
    legacy exporter, which left the schema-versioned document — the one
    :mod:`wrinklefe.io` points forward-looking consumers at — as the
    only export that could not be checked against the validation
    ledger.
    """

    def test_provenance_block_is_present(self, fe_result, tmp_path):
        out = tmp_path / "r.json"
        export_results_json(fe_result, out)
        assert "provenance" in json.loads(out.read_text())

    def test_provenance_records_the_real_installed_version(
        self, fe_result, tmp_path
    ):
        """Not a hardcoded literal (issue #261)."""
        from wrinklefe import __version__

        out = tmp_path / "r.json"
        export_results_json(fe_result, out)
        prov = json.loads(out.read_text())["provenance"]
        assert prov["wrinklefe"] == __version__

    def test_provenance_records_the_numerics_stack(self, fe_result, tmp_path):
        """A reproducibility claim needs the versions that produced it."""
        import numpy
        import scipy

        out = tmp_path / "r.json"
        export_results_json(fe_result, out)
        prov = json.loads(out.read_text())["provenance"]
        assert prov["numpy"] == numpy.__version__
        assert prov["scipy"] == scipy.__version__
        assert prov["python"] and prov["platform"]

    def test_provenance_carries_no_timestamp_here(self, fe_result, tmp_path):
        """Deliberately omitted so the document stays byte-deterministic.

        The legacy exporter stamps ``timestamp_utc`` and makes no
        determinism guarantee; this one does guarantee it, and a
        wall-clock field would break it for two writes of the same
        result. Nothing is lost: reproducing a result needs the version
        set, and the write time is already on the filesystem.
        """
        out = tmp_path / "r.json"
        export_results_json(fe_result, out)
        assert "timestamp_utc" not in json.loads(out.read_text())["provenance"]

    def test_provenance_records_the_solver_actually_configured(
        self, fe_result, tmp_path
    ):
        out = tmp_path / "r.json"
        export_results_json(fe_result, out)
        prov = json.loads(out.read_text())["provenance"]
        assert prov["solver"]["type"] == fe_result.config.solver

    def test_both_exporters_report_the_same_environment(
        self, fe_result, tmp_path
    ):
        """One shared builder, so the paths cannot disagree.

        ``timestamp_utc`` is excluded: the two files are written at
        different instants, and that field is meant to differ.
        """
        from wrinklefe.io.export import export_results_json as legacy

        structured_path = tmp_path / "structured.json"
        legacy_path = tmp_path / "legacy.json"
        export_results_json(fe_result, structured_path)
        legacy(fe_result, legacy_path)

        a = json.loads(structured_path.read_text())["provenance"]
        b = json.loads(legacy_path.read_text())["provenance"]
        a.pop("timestamp_utc", None)
        b.pop("timestamp_utc", None)
        assert a == b

    def test_schema_version_advanced_past_the_version_without_provenance(self):
        """1.2 is the additive bump that added the block."""
        major, minor = (int(part) for part in SCHEMA_VERSION.split("."))
        assert (major, minor) >= (1, 2)


class TestLegacyExporterIsADifferentDocument:
    """The name collision is deliberate, but it is not interchangeable.

    :mod:`wrinklefe.io` documents both exporters and re-exports the
    legacy one. These tests pin the consequence a caller has to know
    about, so the claim in that docstring stays true.
    """

    def _leaf_paths(self, obj, prefix=""):
        out = set()
        if isinstance(obj, dict):
            for key, value in obj.items():
                out |= self._leaf_paths(
                    value, f"{prefix}.{key}" if prefix else key
                )
        elif isinstance(obj, list):
            if obj and isinstance(obj[0], (dict, list)):
                out |= self._leaf_paths(obj[0], f"{prefix}[]")
            else:
                out.add(prefix)
        else:
            out.add(prefix)
        return out

    def test_the_two_documents_share_no_leaf_paths_but_provenance(
        self, fe_result, tmp_path
    ):
        """Zero overlap outside the one block they deliberately share."""
        from wrinklefe.io.export import export_results_json as legacy

        structured_path = tmp_path / "structured.json"
        legacy_path = tmp_path / "legacy.json"
        export_results_json(fe_result, structured_path)
        legacy(fe_result, legacy_path)

        a = self._leaf_paths(json.loads(structured_path.read_text()))
        b = self._leaf_paths(json.loads(legacy_path.read_text()))
        shared = {
            p for p in (a & b)
            if not p.startswith("provenance") and not p.startswith("mesh.")
        }
        assert shared == set(), (
            f"the two schemas have started to overlap outside provenance "
            f"and the mesh counts: {sorted(shared)} — the docstrings in "
            f"wrinklefe.io and wrinklefe.io.results state exactly what is "
            f"shared, so either they or the schema is now wrong"
        )

    def test_the_shared_mesh_counts_are_the_only_non_provenance_overlap(
        self, fe_result, tmp_path
    ):
        """Pin the overlap positively, not just its absence elsewhere.

        Stated in both docstrings, so a change to either side that adds
        or drops one of these should fail here and force the prose to be
        updated with it.
        """
        from wrinklefe.io.export import export_results_json as legacy

        structured_path = tmp_path / "structured.json"
        legacy_path = tmp_path / "legacy.json"
        export_results_json(fe_result, structured_path)
        legacy(fe_result, legacy_path)

        a = self._leaf_paths(json.loads(structured_path.read_text()))
        b = self._leaf_paths(json.loads(legacy_path.read_text()))
        shared = {p for p in (a & b) if not p.startswith("provenance")}
        assert shared == {"mesh.n_nodes", "mesh.n_elements", "mesh.n_dof"}

    def test_analytical_only_run_shares_nothing_outside_provenance(
        self, analytical_result, tmp_path
    ):
        """The other half of the claim in both docstrings.

        With no ``mesh`` block there is no overlap left at all, so an
        analytical-only consumer really can read nothing from the wrong
        document.
        """
        from wrinklefe.io.export import export_results_json as legacy

        structured_path = tmp_path / "structured.json"
        legacy_path = tmp_path / "legacy.json"
        export_results_json(analytical_result, structured_path)
        legacy(analytical_result, legacy_path)

        structured = json.loads(structured_path.read_text())
        assert "mesh" not in structured, "fixture is not analytical-only"

        a = self._leaf_paths(structured)
        b = self._leaf_paths(json.loads(legacy_path.read_text()))
        shared = {p for p in (a & b) if not p.startswith("provenance")}
        assert shared == set()

    def test_wrinklefe_io_reexports_the_legacy_exporter(self):
        """Documented back-compat guarantee: the default did not move."""
        import wrinklefe.io as io_pkg
        from wrinklefe.io.export import export_results_json as legacy

        assert io_pkg.export_results_json is legacy

    def test_the_documented_way_to_tell_the_files_apart_works(
        self, fe_result, tmp_path
    ):
        """``schema_version`` vs ``wrinklefe_version`` at the top level."""
        from wrinklefe.io.export import export_results_json as legacy

        structured_path = tmp_path / "structured.json"
        legacy_path = tmp_path / "legacy.json"
        export_results_json(fe_result, structured_path)
        legacy(fe_result, legacy_path)

        structured = json.loads(structured_path.read_text())
        legacy_doc = json.loads(legacy_path.read_text())
        assert "schema_version" in structured
        assert "schema_version" not in legacy_doc
        assert "wrinklefe_version" in legacy_doc
        assert "wrinklefe_version" not in structured
