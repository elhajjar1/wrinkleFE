"""Coverage for the Streamlit app's "Find acceptable limit" mode (#280).

The engine (:func:`wrinklefe.goalseek.find_critical_value`) has its own
tests in ``tests/test_goalseek.py``; this module covers the *app surface*
around it:

1. Pure-function coverage of the DEFAULTS registration, the Reset
   behaviour, and ``_critical_limit_block`` — the hash guard that decides
   whether an acceptance limit may ride along on an NCR attachment.
2. ``AppTest`` runs against a **stubbed** engine (converged and refused),
   so the smoke tests never pay for a real solve, plus one **real**
   analytical end-to-end search on the app's own defaults (~0.5 s) that
   pins the number the UI actually shows.
"""

from __future__ import annotations

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

_GS_BUTTON = "Find acceptable limit"
_APPLY_BUTTON = "Apply this amplitude to the sidebar"
_STALE_MARKER = "Inputs have changed since this search"


def _app_path() -> str:
    return str(_REPO_ROOT / "app.py")


@pytest.fixture(scope="module")
def app_module():
    import app as app_module  # noqa: WPS433 - test-time import.
    return app_module


def _click(at, label: str) -> None:
    for b in at.button:
        if b.label == label:
            b.click()
            return
    raise AssertionError(
        f"button {label!r} not found; have {[b.label for b in at.button]}"
    )


# ---------------------------------------------------------------------------
# Result builders (stubbed engine).
# ---------------------------------------------------------------------------


def _evaluation(index: int, value: float, objective: float, phase: str):
    from wrinklefe.goalseek import GoalSeekEvaluation

    return GoalSeekEvaluation(
        index=index,
        value=value,
        objective=objective,
        knockdown=objective,
        strength_MPa=objective * 1200.0,
        residual=objective - 0.85,
        phase=phase,
        runtime_s=0.02,
    )


def _make_result(**overrides):
    """A converged ``CriticalValueResult`` with every field populated."""
    from wrinklefe.goalseek import CriticalValueResult

    scan_values = [0.0, 0.05, 0.1, 0.4]
    scan_objectives = [1.0, 0.88, 0.80, 0.60]
    evaluations = [
        _evaluation(i, v, o, "scan")
        for i, (v, o) in enumerate(zip(scan_values, scan_objectives))
    ]
    evaluations.append(_evaluation(len(evaluations), 0.0665, 0.8501, "verify"))
    fields = {
        "parameter": "amplitude",
        "objective_name": "knockdown",
        "target": 0.85,
        "target_is_strength": False,
        "direction": "decreasing",
        "status": "converged",
        "message": "converged",
        "critical_value": 0.0665,
        "critical_value_root": 0.06653,
        "achieved_objective": 0.8501,
        "achieved_knockdown": 0.8501,
        "achieved_strength_MPa": 1020.0,
        "critical_config": None,
        "bounds": (0.0, 4.392),
        "bounds_source": ("zero", "laminate_thickness"),
        "scan_values": scan_values,
        "scan_objectives": scan_objectives,
        "sign_changes": 1,
        "log_spaced": False,
        "pristine_reference_MPa": 1200.0,
        "evaluations": evaluations,
        "n_evaluations": len(evaluations),
        "rtol": 1e-3,
        "analytical_only": True,
        "runtime_s": 0.5,
    }
    fields.update(overrides)
    return CriticalValueResult(**fields)


_NON_MONOTONIC_MESSAGE = (
    "the scan of 'wavelength' reversed direction 2 times, so the target "
    "is met on more than one interval and no single limit exists"
)


def _stub_engine(monkeypatch, result):
    """Patch the engine the app imports on every script execution."""
    def _fake(base_config, **kwargs):
        return result

    monkeypatch.setattr("wrinklefe.goalseek.find_critical_value", _fake)


# ---------------------------------------------------------------------------
# 1. DEFAULTS / Reset.
# ---------------------------------------------------------------------------


def test_defaults_include_goalseek_keys(app_module):
    """Every ``sb_gs_*`` widget key must round-trip through DEFAULTS."""
    d = app_module.DEFAULTS
    assert d["sb_gs_parameter"] == "amplitude"
    assert d["sb_gs_target_mode"] == app_module.GS_TARGET_KNOCKDOWN
    assert d["sb_gs_target"] == 0.85
    assert isinstance(d["sb_gs_target_strength"], float)
    assert d["sb_gs_bracket_lo"] == 0.0
    assert d["sb_gs_bracket_hi"] == 0.0
    assert d["sb_gs_scan_points"] >= 3
    assert 0.0 < d["sb_gs_rtol"] < 0.1


def test_goalseek_scan_points_default_matches_engine(app_module):
    """The UI default must not drift from the engine's own default."""
    from wrinklefe.goalseek import DEFAULT_SCAN_POINTS

    assert app_module.DEFAULTS["sb_gs_scan_points"] == DEFAULT_SCAN_POINTS


def test_reset_inputs_clears_goalseek_state(app_module, monkeypatch):
    """A Reset drops the stored search: the limit belongs to the config it
    was searched against, and Reset replaces that config wholesale."""
    fake_state: dict[str, object] = {
        "sb_gs_target": 0.5,
        "goalseek_result": object(),
        "goalseek_payload": (("amplitude", 0.4),),
        "unrelated": "keep me",
    }
    monkeypatch.setattr(app_module.st, "session_state", fake_state)

    app_module.reset_inputs()

    assert "goalseek_result" not in fake_state
    assert "goalseek_payload" not in fake_state
    assert fake_state["sb_gs_target"] == app_module.DEFAULTS["sb_gs_target"]
    assert fake_state["unrelated"] == "keep me"


# ---------------------------------------------------------------------------
# 2. The NCR hash guard.
# ---------------------------------------------------------------------------


def test_critical_limit_block_matching_payload(app_module):
    """A converged search against the run's own payload yields the block."""
    payload = (("amplitude", 0.366), ("loading", "compression"))
    block = app_module._critical_limit_block(_make_result(), payload, payload)

    assert block is not None
    assert block["parameter"] == "amplitude"
    assert block["parameter_units"] == "mm"
    assert block["direction"] == "decreasing"
    assert block["objective"] == "knockdown"
    assert block["target"] == pytest.approx(0.85)
    assert block["target_units"] is None
    assert block["critical_value"] == pytest.approx(0.0665)
    assert block["achieved_knockdown"] == pytest.approx(0.8501)
    assert block["method"] == "analytical"
    assert block["n_evaluations"] == 5
    assert block["rtol"] == pytest.approx(1e-3)
    # The conservative value, never the raw root.
    assert block["critical_value"] != pytest.approx(0.06653)


def test_critical_limit_block_refuses_mismatched_payload(app_module):
    """A limit searched against different inputs must never be attached."""
    searched = (("amplitude", 0.366), ("loading", "compression"))
    ran = (("amplitude", 0.366), ("loading", "tension"))

    assert app_module._critical_limit_block(_make_result(), searched, ran) is None


def test_critical_limit_block_refuses_non_converged(app_module):
    """Only a converged search carries a number worth quoting."""
    payload = (("amplitude", 0.366),)
    refused = _make_result(
        status="non_monotonic",
        message=_NON_MONOTONIC_MESSAGE,
        critical_value=None,
    )

    assert app_module._critical_limit_block(refused, payload, payload) is None


def test_critical_limit_block_refuses_missing_state(app_module):
    """No search, no run, or no payload -> no block (never a crash)."""
    payload = (("amplitude", 0.366),)
    assert app_module._critical_limit_block(None, payload, payload) is None
    assert app_module._critical_limit_block(_make_result(), None, payload) is None
    assert app_module._critical_limit_block(_make_result(), payload, None) is None


def test_critical_limit_block_strength_target(app_module):
    """A MPa target is recorded with its units so the NCR can print them."""
    payload = (("amplitude", 0.366),)
    block = app_module._critical_limit_block(
        _make_result(
            target=1020.0,
            target_is_strength=True,
            objective_name="strength_MPa",
            achieved_objective=1020.5,
        ),
        payload,
        payload,
    )

    assert block is not None
    assert block["target_units"] == "MPa"
    assert block["objective"] == "strength_MPa"


# ---------------------------------------------------------------------------
# 3. AppTest — stubbed engine.
# ---------------------------------------------------------------------------


def test_mode_controls_render_without_a_search():
    """The button and its target input exist before anything is clicked."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_app_path(), default_timeout=90)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    assert _GS_BUTTON in [b.label for b in at.button]
    assert at.number_input(key="sb_gs_target").value == pytest.approx(0.85)
    assert at.selectbox(key="sb_gs_parameter").value == "amplitude"


def test_converged_stub_renders_metric_row(monkeypatch):
    """A converged result renders the limit, the achieved objective and the
    safe-side statement."""
    from streamlit.testing.v1 import AppTest

    _stub_engine(monkeypatch, _make_result())

    at = AppTest.from_file(_app_path(), default_timeout=90)
    at.run()
    _click(at, _GS_BUTTON)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    labels = {m.label: m.value for m in at.metric}
    assert "Largest acceptable amplitude" in labels
    assert "0.0665" in labels["Largest acceptable amplitude"]
    assert labels["Achieved knockdown"].startswith("0.850")
    assert labels["Forward evaluations"] == "5"

    blob = " ".join(s.value for s in at.success)
    assert "verifies it with an extra forward run" in blob
    assert "conservative value is the one to quote" in blob


def test_refusal_message_is_shown_verbatim(monkeypatch):
    """``non_monotonic`` surfaces the engine's own message, unparaphrased —
    it quotes the measurements behind the refusal."""
    from streamlit.testing.v1 import AppTest

    _stub_engine(
        monkeypatch,
        _make_result(
            status="non_monotonic",
            message=_NON_MONOTONIC_MESSAGE,
            critical_value=None,
            critical_value_root=None,
            achieved_objective=None,
            achieved_knockdown=None,
            sign_changes=2,
        ),
    )

    at = AppTest.from_file(_app_path(), default_timeout=90)
    at.run()
    _click(at, _GS_BUTTON)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    errors = [e.value for e in at.error]
    assert any(_NON_MONOTONIC_MESSAGE == e for e in errors), errors
    # No limit is offered alongside a refusal.
    assert _APPLY_BUTTON not in [b.label for b in at.button]


def test_no_crossing_is_a_warning_not_an_error(monkeypatch):
    """"No crossing in range" can simply mean nothing in range fails the
    criterion, so it must not be dressed up as a failure."""
    from streamlit.testing.v1 import AppTest

    message = (
        "no value of 'amplitude' in [0, 4.392] drives knockdown below the "
        "target 0.4 (measured minimum 0.424 at 4.392)"
    )
    _stub_engine(
        monkeypatch,
        _make_result(
            status="no_crossing",
            message=message,
            critical_value=None,
            critical_value_root=None,
            achieved_objective=None,
            sign_changes=0,
        ),
    )

    at = AppTest.from_file(_app_path(), default_timeout=90)
    at.run()
    _click(at, _GS_BUTTON)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    assert any(message == w.value for w in at.warning), [
        w.value for w in at.warning
    ]
    assert not [e.value for e in at.error]


def test_apply_seeds_the_amplitude_widget(monkeypatch):
    """Apply seeds ``sb_amplitude`` with the limit so the user can run the
    full analysis at it."""
    from streamlit.testing.v1 import AppTest

    _stub_engine(monkeypatch, _make_result())

    at = AppTest.from_file(_app_path(), default_timeout=90)
    at.run()
    _click(at, _GS_BUTTON)
    at.run()
    _click(at, _APPLY_BUTTON)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    assert at.session_state["sb_amplitude"] == pytest.approx(0.0665)
    assert at.number_input(key="sb_amplitude").value == pytest.approx(0.0665)


def test_editing_an_input_marks_the_search_stale(monkeypatch):
    """A stored limit describes the inputs it was searched against; say so
    when they change (same hash-compare as the run staleness banner)."""
    from streamlit.testing.v1 import AppTest

    _stub_engine(monkeypatch, _make_result())

    at = AppTest.from_file(_app_path(), default_timeout=90)
    at.run()
    _click(at, _GS_BUTTON)
    at.run()
    assert not any(_STALE_MARKER in w.value for w in at.warning)

    at.number_input(key="sb_wavelength").set_value(24.0)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    assert any(_STALE_MARKER in w.value for w in at.warning), [
        w.value for w in at.warning
    ]


# ---------------------------------------------------------------------------
# 4. AppTest — the real engine, analytical path.
# ---------------------------------------------------------------------------


def test_real_analytical_search_on_app_defaults():
    """End-to-end against the real engine on the app's default sidebar
    (IM7/8552, [0/45/-45/90]_3s, 1 % compression, analytical path).

    The measured limit is ~0.0665 mm of amplitude for a 0.85 knockdown
    target; the tolerance is loose on purpose (this pins the wiring and the
    safe-side contract, not the forward model, which has its own tests).
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_app_path(), default_timeout=180)
    at.run()
    _click(at, _GS_BUTTON)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    result = at.session_state["goalseek_result"]
    assert result.status == "converged", result.message
    assert result.parameter == "amplitude"
    assert result.analytical_only is True
    assert result.critical_value == pytest.approx(0.0665, abs=5e-3)
    # The safe-side contract: the reported value satisfies the criterion
    # under a real forward evaluation, and is on the acceptable side of
    # the raw root (knockdown decreases with amplitude).
    assert result.achieved_knockdown >= 0.85
    assert result.critical_value <= result.critical_value_root

    labels = {m.label: m.value for m in at.metric}
    assert "Largest acceptable amplitude" in labels
    assert labels["Largest acceptable amplitude"].endswith(" mm")
