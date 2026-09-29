"""Coverage for the Streamlit app's morphology-comparison view (#263).

The engine already compares morphologies (``compare_morphologies``, and
``wrinklefe compare``); what was missing was an app path, so this module
covers the *app surface*:

1. Pure-function coverage of the row/CSV/figure builders and the
   morphology-aware payload override — no Streamlit, no solve.
2. ``AppTest`` runs on the analytical path (a few seconds) that pin what
   the UI actually does: the comparison runs, a morphology that rejects
   the current inputs is reported instead of aborting the rest, and the
   staleness rule ignores the one sidebar input the view deliberately
   varies.
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import pytest

# ``app.py`` lives at the repo root, not under ``src/``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pytest.importorskip("streamlit", reason="Streamlit not installed.")
pytest.importorskip(
    "streamlit.testing.v1", reason="Streamlit testing API not available."
)

import matplotlib  # noqa: E402

pytestmark = pytest.mark.viz

matplotlib.use("Agg")

_COMPARE_BUTTON = "Compare morphologies"
_ALL_MORPHOLOGIES = [
    "stack", "convex", "concave", "uniform", "graded", "tool_flat",
]


def _app_path() -> str:
    return str(_REPO_ROOT / "app.py")


@pytest.fixture(scope="module")
def app_module():
    import app as app_module  # noqa: WPS433 — test-time import.

    return app_module


def _fresh_app(timeout: float = 600.0):
    """An AppTest past the acknowledgement gate, on the analytical path."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_app_path(), default_timeout=timeout)
    at.session_state["_wf_acknowledged"] = True
    at.run()
    for checkbox in at.checkbox:
        if checkbox.key == "sb_analytical_only":
            checkbox.set_value(True)
    at.run()
    return at


def _click(at, label: str) -> None:
    for button in at.button:
        if button.label == label:
            button.click()
            return
    raise AssertionError(
        f"button {label!r} not found; have {[b.label for b in at.button]}"
    )


def _stub_result(knockdown: float, *, fe: dict | None = None) -> dict:
    """The subset of the app's result dict the comparison reads."""
    return {
        "analytical_knockdown": knockdown,
        "analytical_strength_MPa": knockdown * 1200.0,
        "max_angle_deg": 8.5,
        "morphology_factor": 1.0,
        "damage_index": 1.0 - knockdown,
        "fe": fe,
    }


# ----------------------------------------------------------------------
# The palette — every morphology the engine accepts needs its own hue
# ----------------------------------------------------------------------

class TestPalette:
    def test_every_valid_morphology_has_a_colour(self):
        """Otherwise a six-morphology chart collapses three series onto
        the single ``"gray"`` fallback and reads as one."""
        from wrinklefe.core.morphology import (
            MORPHOLOGY_PHASES,
            SINGLE_WRINKLE_MODES,
        )
        from wrinklefe.viz.style import MORPHOLOGY_COLORS

        valid = set(MORPHOLOGY_PHASES) | set(SINGLE_WRINKLE_MODES)
        missing = sorted(valid - set(MORPHOLOGY_COLORS))
        assert not missing, f"no colour for {missing}"

    def test_colours_are_distinct(self):
        from wrinklefe.viz.style import MORPHOLOGY_COLORS

        values = list(MORPHOLOGY_COLORS.values())
        assert len(set(values)) == len(values)

    def test_the_app_offers_only_morphologies_the_engine_accepts(
        self, app_module
    ):
        from wrinklefe.core.morphology import (
            MORPHOLOGY_PHASES,
            SINGLE_WRINKLE_MODES,
        )

        valid = set(MORPHOLOGY_PHASES) | set(SINGLE_WRINKLE_MODES)
        assert set(app_module.MORPHOLOGIES) <= valid


# ----------------------------------------------------------------------
# Pure builders
# ----------------------------------------------------------------------

class TestRowBuilder:
    def test_one_row_per_morphology_in_order(self, app_module):
        comparison = {
            "stack": _stub_result(0.61),
            "convex": _stub_result(0.65),
        }
        rows = app_module._comparison_rows(comparison)
        assert [r["morphology"] for r in rows] == ["stack", "convex"]
        assert rows[0]["knockdown"] == pytest.approx(0.61)
        assert rows[0]["strength_MPa"] == pytest.approx(0.61 * 1200.0)

    def test_analytical_run_is_flagged_and_has_no_max_fi(self, app_module):
        rows = app_module._comparison_rows({"stack": _stub_result(0.6)})
        assert rows[0]["analytical_only"] is True
        assert rows[0]["max_FI"] is None

    def test_fe_run_reports_max_fi_across_criteria_and_plies(
        self, app_module
    ):
        fe = {
            "ply_failure_indices": {
                "larc05": [0.2, 0.9, 0.4],
                "hashin": [0.3, 0.5, 0.95],
            },
            "critical_criterion": "hashin",
            "critical_mode": "matrix_tension",
        }
        rows = app_module._comparison_rows({"stack": _stub_result(0.6, fe=fe)})
        assert rows[0]["max_FI"] == pytest.approx(0.95)
        assert rows[0]["critical_criterion"] == "hashin"
        assert rows[0]["analytical_only"] is False

    def test_max_fi_ignores_non_finite_entries(self, app_module):
        fe = {"ply_failure_indices": {"larc05": [0.4, float("inf"), 0.7]}}
        assert app_module._comparison_max_fi(fe) == pytest.approx(0.7)

    def test_max_fi_is_none_when_nothing_is_reported(self, app_module):
        assert app_module._comparison_max_fi(None) is None
        assert app_module._comparison_max_fi({}) is None
        assert app_module._comparison_max_fi(
            {"ply_failure_indices": {}}
        ) is None


class TestCsv:
    @pytest.fixture
    def payload(self):
        return tuple(sorted({
            "amplitude": 0.366,
            "wavelength": 16.0,
            "width": 12.0,
            "loading": "compression",
            "applied_strain": -0.005,
            "ply_thickness": 0.183,
            "angles_tuple": (0.0, 45.0, -45.0, 90.0),
            "morphology": "stack",
        }.items()))

    def _read(self, app_module, payload):
        rows = app_module._comparison_rows({
            "stack": _stub_result(0.61),
            "concave": _stub_result(0.55),
        })
        text = app_module._comparison_csv(rows, payload).decode()
        return list(csv.DictReader(io.StringIO(text)))

    def test_one_row_per_morphology(self, app_module, payload):
        out = self._read(app_module, payload)
        assert [r["morphology"] for r in out] == ["stack", "concave"]

    def test_every_row_carries_config_provenance(self, app_module, payload):
        """A row copied out of context still says what produced it."""
        out = self._read(app_module, payload)
        for row in out:
            assert row["amplitude_mm"] == "0.366"
            assert row["wavelength_mm"] == "16.0"
            assert row["loading"] == "compression"
            assert row["layup_deg"] == "0.0;45.0;-45.0;90.0"
            assert row["wrinklefe_version"]

    def test_missing_values_are_blank_not_none(self, app_module, payload):
        """``None`` would land in the file as the literal text 'None'."""
        out = self._read(app_module, payload)
        assert out[0]["max_FI"] == ""
        assert out[0]["critical_criterion"] == ""


class TestFigure:
    def test_bars_use_the_central_palette(self, app_module):
        pytest.importorskip("plotly")
        from wrinklefe.viz.style import MORPHOLOGY_COLORS

        rows = app_module._comparison_rows({
            "stack": _stub_result(0.61),
            "graded": _stub_result(0.83),
        })
        fig = app_module._comparison_figure(rows)
        assert fig is not None
        colors = list(fig.data[0].marker.color)
        assert colors == [
            MORPHOLOGY_COLORS["stack"], MORPHOLOGY_COLORS["graded"]
        ]

    def test_y_axis_is_anchored_at_zero(self, app_module):
        """Knockdown is a fraction: a zoomed axis exaggerates the spread."""
        pytest.importorskip("plotly")
        rows = app_module._comparison_rows({
            "stack": _stub_result(0.61),
            "convex": _stub_result(0.65),
        })
        fig = app_module._comparison_figure(rows)
        assert fig.layout.yaxis.range[0] == pytest.approx(0.0)


# ----------------------------------------------------------------------
# App surface
# ----------------------------------------------------------------------

class TestSidebarControl:
    def test_button_and_default_selection(self):
        at = _fresh_app()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert _COMPARE_BUTTON in [b.label for b in at.button]
        assert at.session_state["sb_compare_morphologies"] == [
            "stack", "convex", "concave"
        ]

    def test_offers_every_app_morphology(self):
        at = _fresh_app()
        assert at.multiselect(
            key="sb_compare_morphologies"
        ).options == _ALL_MORPHOLOGIES

    def test_refuses_fewer_than_two_morphologies(self):
        at = _fresh_app()
        at.multiselect(key="sb_compare_morphologies").set_value(["stack"])
        at.run()
        _click(at, _COMPARE_BUTTON)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert any("at least 2" in str(e.value) for e in at.error)
        assert at.session_state.get("morph_comparison") is None


class TestComparisonRun:
    def test_default_selection_runs_and_reports_a_spread(self):
        at = _fresh_app()
        _click(at, _COMPARE_BUTTON)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]

        comparison = at.session_state["morph_comparison"]
        assert sorted(comparison) == ["concave", "convex", "stack"]

        labels = {m.label for m in at.metric}
        assert {"Most severe", "Least severe", "Spread"} <= labels

    def test_the_three_dual_modes_do_not_all_agree(self):
        """If they did, the view would have nothing to say."""
        at = _fresh_app()
        _click(at, _COMPARE_BUTTON)
        at.run()
        import app as app_module

        rows = app_module._comparison_rows(
            at.session_state["morph_comparison"]
        )
        knockdowns = {r["morphology"]: r["knockdown"] for r in rows}
        assert len(set(knockdowns.values())) > 1, knockdowns
        # concave is the adverse morphology, convex the favourable one.
        assert knockdowns["concave"] < knockdowns["convex"]

    def test_a_morphology_that_rejects_the_inputs_does_not_abort_the_rest(
        self,
    ):
        """``tool_flat`` refuses the app's own default amplitude (its
        crest-side transition elements would invert). Losing the other
        five answers to that would defeat the point of the view."""
        at = _fresh_app()
        at.multiselect(
            key="sb_compare_morphologies"
        ).set_value(_ALL_MORPHOLOGIES)
        at.run()
        _click(at, _COMPARE_BUTTON)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]

        comparison = at.session_state["morph_comparison"]
        failures = at.session_state["morph_comparison_failures"]
        assert "tool_flat" in failures
        assert sorted(comparison) == [
            "concave", "convex", "graded", "stack", "uniform"
        ]

    def test_the_skipped_morphology_is_reported_not_silently_dropped(self):
        at = _fresh_app()
        at.multiselect(
            key="sb_compare_morphologies"
        ).set_value(_ALL_MORPHOLOGIES)
        at.run()
        _click(at, _COMPARE_BUTTON)
        at.run()
        warnings_text = " ".join(str(w.value) for w in at.warning)
        assert "tool_flat" in warnings_text
        assert "Not comparable" in warnings_text

    def test_the_failure_reason_is_kept(self):
        at = _fresh_app()
        at.multiselect(
            key="sb_compare_morphologies"
        ).set_value(_ALL_MORPHOLOGIES)
        at.run()
        _click(at, _COMPARE_BUTTON)
        at.run()
        why = at.session_state["morph_comparison_failures"]["tool_flat"]
        assert "tool_flat" in why and "amplitude" in why


class TestStaleness:
    def test_changing_the_sidebar_morphology_does_not_make_it_stale(self):
        """The view deliberately spans morphologies, so the sidebar's own
        morphology selector is not part of what invalidates it."""
        at = _fresh_app()
        _click(at, _COMPARE_BUTTON)
        at.run()
        at.session_state["sb_morphology"] = "concave"
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        stale_text = " ".join(str(w.value) for w in at.warning)
        assert "Inputs have changed since this comparison" not in stale_text

    def test_changing_the_amplitude_does_make_it_stale(self):
        at = _fresh_app()
        _click(at, _COMPARE_BUTTON)
        at.run()
        at.session_state["sb_amplitude"] = 0.9
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        stale_text = " ".join(str(w.value) for w in at.warning)
        assert "Inputs have changed since this comparison" in stale_text


class TestResetAndState:
    def test_reset_clears_the_comparison(self):
        at = _fresh_app()
        _click(at, _COMPARE_BUTTON)
        at.run()
        assert at.session_state.get("morph_comparison")
        _click(at, "↻ Reset to defaults")
        at.run()
        assert at.session_state.get("morph_comparison") is None
        assert at.session_state.get("morph_comparison_failures") is None

    def test_reset_hands_out_a_fresh_list_not_the_shared_default(
        self, app_module
    ):
        """A bare assignment would alias the module constant, so editing
        the selection after a Reset would rewrite the default."""
        at = _fresh_app()
        _click(at, "↻ Reset to defaults")
        at.run()
        assert (
            at.session_state["sb_compare_morphologies"]
            is not app_module.COMPARE_DEFAULT_MORPHOLOGIES
        )
        assert (
            at.session_state["sb_compare_morphologies"]
            == app_module.COMPARE_DEFAULT_MORPHOLOGIES
        )


class TestExport:
    def test_comparison_exports_without_a_single_run(self):
        """The comparison is its own result; gating its download behind a
        separate Run would tell the user there is nothing to export."""
        at = _fresh_app()
        _click(at, _COMPARE_BUTTON)
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert at.session_state.get("results") is None
        labels = [b.label for b in at.download_button]
        assert "Download morphology comparison as CSV" in labels
        assert "Download morphology comparison as JSON" in labels
