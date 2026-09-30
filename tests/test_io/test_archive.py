"""Round-trip coverage for the full-result archive (issue #277).

The archive exists so an expensive run survives the session: an FE or CZM
solve holds per-Gauss-point stress and strain, failure indices and modes,
and the cohesive fields, none of which the report-tier JSON keeps (it
reduces large arrays to ``{min, max, mean, p95, n}`` on purpose).

What these tests hold the format to:

1. **Bit-exactness.** Every array identical, every scalar equal — not
   ``allclose``. An archive that loses the last few bits would still
   produce plausible plots, which is the failure mode worth preventing.
2. **Usability by existing code.** The reloaded object is a real
   ``AnalysisResults``: ``viz`` functions take it unchanged and the
   internal wiring (``field_results.mesh is results.mesh``) is restored
   rather than duplicated.
3. **No pickle.** Enforced at write time by numpy and at read time by
   ``allow_pickle=False``, and asserted here by reading the archive with
   nothing but numpy and json.
"""

from __future__ import annotations

import dataclasses as dc
import json
import pathlib
import warnings
import zipfile

import numpy as np
import pytest

from wrinklefe.analysis import AnalysisConfig, AnalysisResults, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary
from wrinklefe.io.archive import (
    ARCHIVE_FORMAT_VERSION,
    DEFAULT_SUFFIX,
    ArchiveFormatError,
    load_results,
    save_results,
)

_BASE = dict(
    amplitude=0.25, wavelength=16.0, width=12.0, morphology="stack",
    loading="compression", ply_thickness=0.183,
    angles=[0, 45, -45, 90, 90, -45, 45, 0],
    nx=6, ny=3, nz_per_ply=1, applied_strain=-0.005,
    analytical_only=False, verbose=False,
)


def _run(**over) -> AnalysisResults:
    kwargs = dict(_BASE)
    kwargs["material"] = MaterialLibrary().get("IM7_8552")
    kwargs.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WrinkleAnalysis(AnalysisConfig(**kwargs)).run()


@pytest.fixture(scope="module")
def fe_result() -> AnalysisResults:
    return _run()


@pytest.fixture(scope="module")
def czm_result() -> AnalysisResults:
    return _run(enable_czm=True, czm_n_load_increments=3)


@pytest.fixture(scope="module")
def analytical_result() -> AnalysisResults:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kwargs = dict(_BASE)
        kwargs["material"] = MaterialLibrary().get("IM7_8552")
        kwargs["analytical_only"] = True
        cfg = AnalysisConfig(**kwargs)
        return WrinkleAnalysis(cfg).run(analytical_only=True)


def _roundtrip(result, tmp_path, name="r"):
    path = save_results(result, tmp_path / f"{name}{DEFAULT_SUFFIX}")
    return path, load_results(path)


def _arrays(obj, prefix=""):
    """Every ndarray reachable through dataclass fields and dict values."""
    out: dict[str, np.ndarray] = {}
    if isinstance(obj, np.ndarray):
        out[prefix] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_arrays(v, f"{prefix}.{k}"))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(_arrays(v, f"{prefix}.{i}"))
    elif dc.is_dataclass(obj) and not isinstance(obj, type):
        for f in dc.fields(obj):
            out.update(_arrays(getattr(obj, f.name), f"{prefix}.{f.name}"))
    return out


# ----------------------------------------------------------------------
# AC1 — bit-exact round trip
# ----------------------------------------------------------------------

class TestRoundTripFE:
    def test_path_written_is_the_path_asked_for(self, fe_result, tmp_path):
        """``np.savez_compressed`` appends ``.npz`` to a path argument, which
        would make ``load_results`` fail on the name the caller just used."""
        asked = tmp_path / f"exact{DEFAULT_SUFFIX}"
        written = save_results(fe_result, asked)
        assert written == asked
        assert asked.exists()
        assert not asked.with_suffix(asked.suffix + ".npz").exists()

    def test_every_array_is_bit_identical(self, fe_result, tmp_path):
        _, back = _roundtrip(fe_result, tmp_path)
        before, after = _arrays(fe_result), _arrays(back)
        assert set(before) == set(after)
        assert before, "fixture produced no arrays — nothing was tested"
        mismatched = [
            k for k in before if not np.array_equal(before[k], after[k])
        ]
        assert not mismatched

    def test_array_dtypes_survive(self, fe_result, tmp_path):
        """Including the ``<U32`` failure-mode strings, which are the one
        non-numeric dtype in the graph."""
        _, back = _roundtrip(fe_result, tmp_path)
        before, after = _arrays(fe_result), _arrays(back)
        assert {k: before[k].dtype for k in before} == {
            k: after[k].dtype for k in after
        }
        modes = next(iter(back.failure_modes.values()))
        assert modes.dtype.kind == "U"

    def test_every_scalar_is_equal(self, fe_result, tmp_path):
        _, back = _roundtrip(fe_result, tmp_path)
        for f in dc.fields(fe_result):
            a = getattr(fe_result, f.name)
            if isinstance(a, (int, float, bool, str)) or a is None:
                assert getattr(back, f.name) == a, f.name

    def test_scalar_dicts_survive_with_their_values(self, fe_result, tmp_path):
        _, back = _roundtrip(fe_result, tmp_path)
        assert back.retention_factors == fe_result.retention_factors
        assert back.baseline_fi == fe_result.baseline_fi
        assert back.retention_degenerate == fe_result.retention_degenerate

    def test_failure_report_round_trips(self, fe_result, tmp_path):
        _, back = _roundtrip(fe_result, tmp_path)
        before, after = fe_result.failure_report, back.failure_report
        assert after.critical_ply == before.critical_ply
        assert after.critical_mode == before.critical_mode
        assert after.critical_criterion == before.critical_criterion
        assert after.fpf == before.fpf
        assert after.lpf == before.lpf
        for crit, arr in before.ply_failure_indices.items():
            assert np.array_equal(after.ply_failure_indices[crit], arr)

    def test_config_round_trips(self, fe_result, tmp_path):
        _, back = _roundtrip(fe_result, tmp_path)
        assert back.config.to_dict() == fe_result.config.to_dict()

    def test_laminate_round_trips_including_materials(
        self, fe_result, tmp_path
    ):
        _, back = _roundtrip(fe_result, tmp_path)
        assert len(back.laminate.plies) == len(fe_result.laminate.plies)
        for a, b in zip(fe_result.laminate.plies, back.laminate.plies):
            assert b.angle == a.angle
            assert b.thickness == a.thickness
            assert b.material.to_dict() == a.material.to_dict()

    def test_derived_laminate_stiffness_is_recomputed_not_stored(
        self, fe_result, tmp_path
    ):
        """A/B/D come from the plies. Storing them would let an archive
        contradict itself; recomputing keeps one source of truth."""
        _, back = _roundtrip(fe_result, tmp_path)
        assert np.allclose(back.laminate.A, fe_result.laminate.A)
        assert np.allclose(back.laminate.D, fe_result.laminate.D)


class TestRoundTripCZM:
    def test_cohesive_fields_are_bit_identical(self, czm_result, tmp_path):
        _, back = _roundtrip(czm_result, tmp_path, "czm")
        for name in (
            "czm_damage", "czm_separation", "czm_traction",
            "czm_load_displacement", "czm_element_centroids",
        ):
            a = getattr(czm_result, name)
            assert a is not None, f"fixture has no {name}"
            assert np.array_equal(getattr(back, name), a), name

    def test_integer_keyed_dicts_keep_integer_keys(
        self, czm_result, tmp_path
    ):
        """JSON object keys are strings; ``czm_energy_per_interface`` is
        keyed by interface index, and handing back "0" for 0 would break
        every lookup silently."""
        _, back = _roundtrip(czm_result, tmp_path, "czm")
        before = czm_result.czm_energy_per_interface
        assert before, "fixture has no per-interface energy"
        assert back.czm_energy_per_interface == before
        assert all(isinstance(k, int) for k in back.czm_energy_per_interface)

    def test_czm_scalars_and_lists(self, czm_result, tmp_path):
        _, back = _roundtrip(czm_result, tmp_path, "czm")
        assert back.czm_converged == czm_result.czm_converged
        assert back.czm_interfaces_used == czm_result.czm_interfaces_used
        assert back.czm_energy_dissipated == czm_result.czm_energy_dissipated

    def test_delamination_report_round_trips(self, czm_result, tmp_path):
        _, back = _roundtrip(czm_result, tmp_path, "czm")
        if czm_result.czm_delamination_report is None:
            pytest.skip("this CZM run produced no delamination report")
        assert back.czm_delamination_report is not None


class TestAnalyticalOnly:
    def test_an_analytical_result_archives_and_reloads(
        self, analytical_result, tmp_path
    ):
        """No mesh, no fields — the archive should still be valid, just
        smaller, rather than failing on the Nones."""
        _, back = _roundtrip(analytical_result, tmp_path, "an")
        assert back.field_results is None
        assert back.analytical_knockdown == (
            analytical_result.analytical_knockdown
        )


# ----------------------------------------------------------------------
# AC2 — the reloaded object is usable by existing code
# ----------------------------------------------------------------------

class TestObjectGraph:
    def test_shared_objects_are_rewired_not_duplicated(
        self, fe_result, tmp_path
    ):
        """At runtime these are the same object; a loader that duplicated
        them would triple the mesh and break code that compares identity."""
        _, back = _roundtrip(fe_result, tmp_path)
        assert back.field_results.mesh is back.mesh
        assert back.field_results.laminate is back.laminate
        assert back.mesh.laminate is back.laminate

    def test_reloaded_value_is_a_real_analysis_results(
        self, fe_result, tmp_path
    ):
        _, back = _roundtrip(fe_result, tmp_path)
        assert isinstance(back, AnalysisResults)
        assert isinstance(back.summary(), str)

    def test_wrinkle_configuration_and_its_profiles_round_trip(
        self, fe_result, tmp_path
    ):
        _, back = _roundtrip(fe_result, tmp_path)
        before, after = fe_result.wrinkle_config, back.wrinkle_config
        assert len(after.wrinkles) == len(before.wrinkles)
        for a, b in zip(before.wrinkles, after.wrinkles):
            assert type(b.profile) is type(a.profile)
            assert b.ply_interface == a.ply_interface
            assert b.phase_offset == a.phase_offset
        # The profile must reproduce its own geometry, not just its class.
        x = np.linspace(-20.0, 20.0, 97)
        for a, b in zip(before.wrinkles, after.wrinkles):
            assert np.array_equal(
                np.asarray(a.profile.displacement(x)),
                np.asarray(b.profile.displacement(x)),
            )

    def test_the_report_tier_export_still_works_on_a_reloaded_result(
        self, fe_result, tmp_path
    ):
        """The archive feeds the existing exporters, so an archived run can
        be turned into a report later without re-solving."""
        from wrinklefe.io.results import export_results_json

        _, back = _roundtrip(fe_result, tmp_path)
        out = tmp_path / "report.json"
        export_results_json(back, out)
        doc = json.loads(out.read_text())
        assert doc["analytical"]["analytical_knockdown"] == pytest.approx(
            fe_result.analytical_knockdown
        )


@pytest.mark.viz
class TestPlotsRenderIdentically:
    """AC2: an existing plot must come out the same from a reloaded result.

    Rendered to PNG and compared byte for byte. One warm-up render is
    discarded first: measured, ``plot_mesh_3d``'s *first* call in a process
    differs from every later one (matplotlib initialisation), so comparing
    a cold render against a warm one reports a difference that has nothing
    to do with the archive.
    """

    @staticmethod
    def _png(fn, args, three_d):
        import hashlib
        import io as _io

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(4, 3), dpi=60)
        ax = (
            fig.add_subplot(111, projection="3d") if three_d
            else fig.add_subplot(111)
        )
        fn(*args, ax=ax)
        buf = _io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return hashlib.sha256(buf.getvalue()).hexdigest()

    @pytest.mark.parametrize(
        "plot_name,pick,three_d",
        [
            ("plot_mesh_3d", lambda r: (r.mesh,), True),
            ("plot_displacement_3d", lambda r: (r.field_results,), True),
            (
                "plot_dual_wrinkle_profiles",
                lambda r: (r.wrinkle_config,),
                False,
            ),
        ],
    )
    def test_plot_is_pixel_identical(
        self, fe_result, tmp_path, plot_name, pick, three_d
    ):
        from wrinklefe import viz

        fn = getattr(viz, plot_name)
        _, back = _roundtrip(fe_result, tmp_path)
        self._png(fn, pick(fe_result), three_d)          # warm-up, discarded
        original = self._png(fn, pick(fe_result), three_d)
        reloaded = self._png(fn, pick(back), three_d)
        assert reloaded == original


# ----------------------------------------------------------------------
# AC3 — no pickle, reasonable size
# ----------------------------------------------------------------------

class TestFormat:
    def test_archive_contains_no_pickled_objects(self, fe_result, tmp_path):
        """An ``.npy`` member written from an object array carries the
        pickle opcodes; none of ours may."""
        path, _ = _roundtrip(fe_result, tmp_path)
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            assert names
            for name in names:
                head = zf.read(name)[:256]
                assert b"numpy.core.multiarray" not in head, name
                assert b"cnumpy" not in head, name

    def test_readable_with_only_numpy_and_json(self, fe_result, tmp_path):
        """No import of this package should be needed to inspect one."""
        path, _ = _roundtrip(fe_result, tmp_path)
        with np.load(path, allow_pickle=False) as npz:
            manifest = json.loads(
                bytes(npz["__manifest__"]).decode("utf-8")
            )
            assert manifest["format_version"] == ARCHIVE_FORMAT_VERSION
            assert "provenance" in manifest
            assert "results" in manifest

    def test_archive_of_a_coarse_run_is_compressed(self, fe_result, tmp_path):
        """Smaller than the raw arrays it carries, and not enormous."""
        path, _ = _roundtrip(fe_result, tmp_path)
        raw = sum(a.nbytes for a in _arrays(fe_result).values())
        assert path.stat().st_size < raw
        assert path.stat().st_size < 5 * 1024 * 1024

    def test_provenance_is_stamped(self, fe_result, tmp_path):
        path, _ = _roundtrip(fe_result, tmp_path)
        with np.load(path, allow_pickle=False) as npz:
            prov = json.loads(
                bytes(npz["__manifest__"]).decode("utf-8")
            )["provenance"]
        assert prov["wrinklefe"] and prov["numpy"] and prov["scipy"]


class TestRefusals:
    def test_a_non_archive_npz_is_refused_by_name(self, tmp_path):
        path = tmp_path / "not-ours.npz"
        np.savez_compressed(path, a=np.arange(3))
        with pytest.raises(ArchiveFormatError, match="not a WrinkleFE"):
            load_results(path)

    def test_a_newer_major_version_is_refused(self, fe_result, tmp_path):
        """Refusing beats half-reading: numbers from a partially understood
        archive would look plausible."""
        path, _ = _roundtrip(fe_result, tmp_path)
        with np.load(path, allow_pickle=False) as npz:
            payload = {k: npz[k] for k in npz.files}
        manifest = json.loads(bytes(payload["__manifest__"]).decode())
        manifest["format_version"] = "99.0"
        payload["__manifest__"] = np.frombuffer(
            json.dumps(manifest).encode(), dtype=np.uint8
        )
        future = tmp_path / "future.wfr"
        with open(future, "wb") as fh:
            np.savez_compressed(fh, **payload)
        with pytest.raises(ArchiveFormatError, match="newer than this"):
            load_results(future)

    def test_a_missing_format_version_is_refused(self, fe_result, tmp_path):
        path, _ = _roundtrip(fe_result, tmp_path)
        with np.load(path, allow_pickle=False) as npz:
            payload = {k: npz[k] for k in npz.files}
        manifest = json.loads(bytes(payload["__manifest__"]).decode())
        del manifest["format_version"]
        payload["__manifest__"] = np.frombuffer(
            json.dumps(manifest).encode(), dtype=np.uint8
        )
        broken = tmp_path / "broken.wfr"
        with open(broken, "wb") as fh:
            np.savez_compressed(fh, **payload)
        with pytest.raises(ArchiveFormatError, match="no format_version"):
            load_results(broken)

    def test_a_missing_required_field_is_reported_as_a_format_error(
        self, fe_result, tmp_path
    ):
        """An archive from an older WrinkleFE, written before a field
        existed, must not surface a bare TypeError about a keyword the
        caller never passed."""
        path, _ = _roundtrip(fe_result, tmp_path)
        with np.load(path, allow_pickle=False) as npz:
            payload = {k: npz[k] for k in npz.files}
        manifest = json.loads(bytes(payload["__manifest__"]).decode())
        del manifest["results"]["config"]
        payload["__manifest__"] = np.frombuffer(
            json.dumps(manifest).encode(), dtype=np.uint8
        )
        older = tmp_path / "older.wfr"
        with open(older, "wb") as fh:
            np.savez_compressed(fh, **payload)
        with pytest.raises(ArchiveFormatError, match="missing 1 field"):
            load_results(older)

    def test_an_unarchivable_object_is_refused_not_pickled(self):
        from wrinklefe.io.archive import _Encoder

        class Opaque:
            pass

        with pytest.raises(ArchiveFormatError, match="no handler"):
            _Encoder().encode(Opaque(), "somewhere")

    def test_an_object_dtype_array_is_refused(self):
        from wrinklefe.io.archive import _Encoder

        arr = np.array([{"a": 1}, {"b": 2}], dtype=object)
        with pytest.raises(ArchiveFormatError, match="dtype=object"):
            _Encoder().encode(arr, "somewhere")


# ----------------------------------------------------------------------
# Surfaces: the CLI flag and the app's download button (issue #277's
# fourth acceptance criterion). Wiring is what makes the format usable,
# and both surfaces have failed in ways the round-trip tests cannot see:
# ``np.savez_compressed`` silently renames ``run.wfr`` to ``run.wfr.npz``
# unless it is handed an open handle, so the CLI reported a path that
# ``load_results`` then could not open.
# ----------------------------------------------------------------------

class TestCommandLineSurface:
    def test_save_results_writes_the_exact_path_and_reloads(self, tmp_path):
        from wrinklefe.cli import main as cli_main

        dst = tmp_path / "run.wfr"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cli_main([
                "analyze", "--amplitude", "0.25", "--wavelength", "16",
                "--fe", "--nx", "6", "--ny", "3",
                "--save-results", str(dst),
            ])

        # The exact path, not ``run.wfr.npz`` — numpy appends the suffix
        # when given a filename rather than a handle, and the CLI prints
        # the path it was asked for either way.
        assert dst.is_file(), sorted(p.name for p in tmp_path.iterdir())
        assert not (tmp_path / "run.wfr.npz").exists()

        restored = load_results(dst)
        assert restored.field_results is not None
        assert restored.mesh.elements.shape[0] > 0

    def test_save_results_and_output_json_coexist(self, tmp_path):
        """The archive and the report are different tiers, not
        alternatives, and the archive is written first so a report
        failure cannot lose the expensive artifact."""
        from wrinklefe.cli import main as cli_main

        archive = tmp_path / "run.wfr"
        report = tmp_path / "run.json"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cli_main([
                "analyze", "--amplitude", "0.25", "--wavelength", "16",
                "--fe", "--nx", "6", "--ny", "3",
                "--save-results", str(archive), "--output-json", str(report),
            ])

        assert archive.is_file() and report.is_file()
        # The report is the small one, by two orders of magnitude: that
        # difference is the whole reason this format exists.
        assert report.stat().st_size * 20 < archive.stat().st_size

    def test_the_flag_is_optional(self, tmp_path, monkeypatch):
        """Omitting it writes nothing at all — checked from inside an empty
        cwd, so "no archive" means the working directory too and not just
        some directory the CLI was never pointed at."""
        from wrinklefe.cli import main as cli_main

        monkeypatch.chdir(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cli_main([
                "analyze", "--amplitude", "0.25", "--wavelength", "16",
                "--nx", "6", "--ny", "3",
            ])
        assert not list(tmp_path.iterdir()), sorted(
            p.name for p in tmp_path.iterdir()
        )


class TestStreamlitSurface:
    """The Export tab's archive button, driven through ``AppTest``.

    Streamlit serves a download button's bytes over a URL rather than
    putting them in the element proto, so the offered bytes cannot be
    read back through ``AppTest``. What is checkable — and what has
    actually broken — is that the button is rendered and that the object
    it serialises is the live result, archivable as-is.
    """

    @pytest.fixture(scope="class")
    def ran_app(self):
        pytest.importorskip("streamlit", reason="Streamlit not installed.")
        import sys

        import matplotlib

        matplotlib.use("Agg")
        from streamlit.testing.v1 import AppTest

        repo_root = pathlib.Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        at = AppTest.from_file(str(repo_root / "app.py"), default_timeout=600)
        at.session_state["_wf_acknowledged"] = True
        at.run()
        for checkbox in at.checkbox:
            if checkbox.key == "sb_analytical_only":
                checkbox.set_value(True)     # keep it cheap
        at.run()
        for button in at.button:
            if button.label == "Run analysis":
                button.click()
                break
        else:                                # pragma: no cover - label drift
            raise AssertionError("no 'Run analysis' button")
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        return at

    @pytest.mark.viz
    def test_the_export_tab_renders_the_archive_button(self, ran_app):
        labels = [b.label for b in ran_app.download_button]
        assert any("full result archive" in label for label in labels), labels

    @pytest.mark.viz
    def test_the_button_serialises_the_live_result(self, ran_app, tmp_path):
        """The run dict carries the real ``AnalysisResults``, and that
        object archives and reloads unchanged — so the bytes the button
        offers are a loadable ``.wfr``."""
        live = ran_app.session_state["results"].get("_result")
        assert isinstance(live, AnalysisResults)

        path, restored = _roundtrip(live, tmp_path, name="from_app")
        assert path.is_file()
        assert restored.analytical_knockdown == live.analytical_knockdown
        assert restored.config.morphology == live.config.morphology

    @pytest.mark.viz
    def test_the_live_result_is_hidden_from_the_json_export(self, ran_app):
        """The Export tab's ``_strip_arrays`` drops underscore-prefixed
        keys, which is the only thing keeping an ``AnalysisResults`` out
        of ``json.dumps``. Renaming the key to ``result`` would put it
        back in and break the report download, so pin the convention."""
        results = ran_app.session_state["results"]
        leaked = [
            key for key, value in results.items()
            if isinstance(value, AnalysisResults) and not key.startswith("_")
        ]
        assert not leaked, leaked
        assert "_result" in results

    @pytest.mark.viz
    def test_the_archive_is_encoded_once_not_once_per_rerun(self, ran_app):
        """Streamlit re-executes the script on every interaction. The bytes
        are cached against the result object (by ``is``, not ``id``, so a
        freed result's id cannot be reused to serve stale bytes)."""
        cached = ran_app.session_state.get("_archive_bytes")
        assert cached is not None, "archive bytes were not cached"
        held, data = cached
        assert held is ran_app.session_state["results"]["_result"]
        assert isinstance(data, bytes) and data

        ran_app.run()                      # a rerun with no new run
        again = ran_app.session_state["_archive_bytes"]
        assert again[1] is data, "the archive was re-encoded on a rerun"

    @pytest.mark.viz
    def test_the_report_download_is_still_offered(self, ran_app):
        """Both tiers coexist on the tab: the archive did not displace the
        report, and the report still serialises (the app would have raised
        otherwise)."""
        labels = [b.label for b in ran_app.download_button]
        assert "Download results as JSON" in labels, labels
