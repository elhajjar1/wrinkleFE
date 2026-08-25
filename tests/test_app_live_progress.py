"""Live-progress + manual result cache coverage for the app (issue #377).

The Streamlit app used to run the solve behind ``@st.cache_data``, which
made repeat-identical runs instant but froze the progress bar at a
hard-coded 10 %: Streamlit refuses widget calls made inside a
cache-decorated function against a layout block created outside it
(issue #242), so ``WrinkleAnalysis.run(progress_callback=...)`` could not
reach the bar.

The app now runs uncached (``_run_analysis``) and caches the *result dict*
by hand in ``session_state`` keyed on the same hashable ``cfg_payload``.
These tests guard the three halves of that change:

1. ``run_analysis_cached`` still short-circuits an identical re-run, and
   the manual cache is bounded (FE result dicts carry mesh / stress /
   failure-index arrays, so it must not grow for the life of a session).
2. ``reset_inputs()`` clears the cache.
3. The run handler threads a live callback into the engine, and that
   callback survives out-of-range fractions without killing the solve.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.viz

# ``app.py`` lives at the repo root, not under ``src/``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pytest.importorskip("streamlit", reason="Streamlit not installed.")

import matplotlib  # noqa: E402

matplotlib.use("Agg")


@pytest.fixture(scope="module")
def app_module():
    """Import ``app.py`` for its module-level helpers."""
    import app as app_module  # noqa: WPS433 - test-time import.
    return app_module


@pytest.fixture
def fake_state(app_module, monkeypatch):
    """Swap ``st.session_state`` for a plain dict for the duration of a test."""
    state: dict[str, object] = {}
    monkeypatch.setattr(app_module.st, "session_state", state)
    return state


def _analytical_payload(**overrides) -> tuple:
    """Build a hashable analytical-only ``cfg_payload``.

    Mirrors what ``_assemble_cfg_payload`` produces for the default sidebar
    (which is analytical-only, so the run is fast), without needing a live
    Streamlit session to read the widgets from.
    """
    from wrinklefe.core.material import MaterialLibrary

    material_dict = MaterialLibrary().get("IM7_8552").to_dict()
    items: dict = {
        "amplitude": 0.5,
        "wavelength": 10.0,
        "width": 5.0,
        "morphology": "stack",
        "decay_floor": 0.0,
        "amplitude_profile": "constant",
        "amplitude_profile_decay_length": 5.0,
        "amplitude_profile_axis": "x",
        "loading": "compression",
        "ply_thickness": 0.125,
        "angles_tuple": (0.0, 45.0, -45.0, 90.0),
        "applied_strain": 0.01,
        "material_tuple": tuple(sorted(material_dict.items())),
        "analytical_only": True,
    }
    items.update(overrides)
    return tuple(items.items())


def _stub_run_analysis(app_module, monkeypatch) -> list[tuple]:
    """Replace the solve with a counting stub; returns the call log."""
    calls: list[tuple] = []

    def _fake(cfg_payload, progress_callback=None):
        calls.append(cfg_payload)
        if progress_callback is not None:
            progress_callback("Stub phase", 0.5)
        return {"summary": "stub", "fe": None, "czm": None}

    monkeypatch.setattr(app_module, "_run_analysis", _fake)
    return calls


# ---------------------------------------------------------------------------
# 1. Manual result cache
# ---------------------------------------------------------------------------


def test_identical_rerun_is_a_cache_hit(app_module, fake_state, monkeypatch):
    """An identical payload must not re-solve (issue #377 criterion 2)."""
    calls = _stub_run_analysis(app_module, monkeypatch)
    payload = _analytical_payload()

    first = app_module.run_analysis_cached(payload)
    second = app_module.run_analysis_cached(payload)

    assert len(calls) == 1, "identical re-run re-solved instead of hitting cache"
    assert second is first, "cache hit returned a different object"


def test_changed_payload_resolves(app_module, fake_state, monkeypatch):
    """Changing any sidebar value must miss the cache and re-solve."""
    calls = _stub_run_analysis(app_module, monkeypatch)

    app_module.run_analysis_cached(_analytical_payload())
    app_module.run_analysis_cached(_analytical_payload(amplitude=0.75))

    assert len(calls) == 2


def test_cache_is_bounded_and_evicts_oldest(app_module, fake_state, monkeypatch):
    """FE results carry arrays, so the cache keeps only the newest entries."""
    _stub_run_analysis(app_module, monkeypatch)
    cap = app_module._RUN_CACHE_MAX
    payloads = [_analytical_payload(amplitude=0.1 * (i + 1)) for i in range(cap + 1)]

    for payload in payloads:
        app_module.run_analysis_cached(payload)

    cache = fake_state[app_module._RUN_CACHE_KEY]
    assert len(cache) == cap, f"cache grew past {cap}: {len(cache)} entries"
    assert payloads[0] not in cache, "oldest entry survived eviction"
    for payload in payloads[1:]:
        assert payload in cache


def test_cache_hit_refreshes_recency(app_module, fake_state, monkeypatch):
    """A payload the user keeps re-running must not be the one evicted."""
    calls = _stub_run_analysis(app_module, monkeypatch)
    cap = app_module._RUN_CACHE_MAX
    payloads = [_analytical_payload(amplitude=0.1 * (i + 1)) for i in range(cap + 1)]

    for payload in payloads[:cap]:
        app_module.run_analysis_cached(payload)
    # Touch the oldest entry, then overflow the cache by one.
    app_module.run_analysis_cached(payloads[0])
    app_module.run_analysis_cached(payloads[cap])

    cache = fake_state[app_module._RUN_CACHE_KEY]
    assert len(calls) == cap + 1, "the refresh touch re-solved"
    assert payloads[0] in cache, "the refreshed entry was evicted"
    assert payloads[1] not in cache, "the least-recently-used entry survived"


def test_reset_inputs_clears_the_run_cache(app_module, fake_state, monkeypatch):
    """After a Reset the same payload must genuinely re-solve."""
    calls = _stub_run_analysis(app_module, monkeypatch)
    payload = _analytical_payload()

    app_module.run_analysis_cached(payload)
    app_module.reset_inputs()
    assert app_module._RUN_CACHE_KEY not in fake_state

    app_module.run_analysis_cached(payload)
    assert len(calls) == 2, "Reset did not clear the manual result cache"


# ---------------------------------------------------------------------------
# 2. Callback wiring through the wrapper
# ---------------------------------------------------------------------------


def test_wrapper_forwards_the_progress_callback(app_module, fake_state, monkeypatch):
    """``run_analysis_cached`` must thread the callback down to the solve."""
    seen: list = []

    def _fake(cfg_payload, progress_callback=None):
        seen.append(progress_callback)
        return {"summary": "stub"}

    monkeypatch.setattr(app_module, "_run_analysis", _fake)

    def _callback(label: str, fraction: float) -> None:
        pass

    app_module.run_analysis_cached(_analytical_payload(), progress_callback=_callback)

    assert seen == [_callback]


def test_real_analytical_run_reports_progress_and_keeps_shape(
    app_module, fake_state
):
    """An un-stubbed analytical run must emit progress and return the same
    result-dict shape the cached path used to return.

    Guards the decorator-removal refactor: the keys the Analyze tab, the
    JSON export, and the CZM renderer read must all still be there.
    """
    events: list[tuple[str, float]] = []
    payload = _analytical_payload()

    result = app_module._run_analysis(
        payload, progress_callback=lambda label, frac: events.append((label, frac))
    )

    assert events, "no progress events were emitted"
    assert all(0.0 <= frac <= 1.0 for _, frac in events), events
    assert events[-1][1] == pytest.approx(1.0), "final event was not 100 %"
    assert all(isinstance(label, str) and label for label, _ in events)

    for key in (
        "summary",
        "loading",
        "applied_strain_abs",
        "max_angle_deg",
        "effective_angle_deg",
        "morphology_factor",
        "gamma_Y_eff",
        "analytical_knockdown",
        "analytical_modulus_knockdown",
        "analytical_strength_MPa",
        "damage_index",
        "tension_mechanisms",
        "fe",
        "czm",
        "progressive",
    ):
        assert key in result, f"result dict lost the {key!r} key"
    # Analytical-only: no FE / CZM / progressive sub-payloads.
    assert result["fe"] is None
    assert result["czm"] is None
    assert result["progressive"] is None


# ---------------------------------------------------------------------------
# 3. End-to-end through the real script
# ---------------------------------------------------------------------------


def _app_path() -> str:
    return str(_REPO_ROOT / "app.py")


def _click(at, label: str) -> None:
    for b in at.button:
        if b.label == label:
            b.click()
            return
    raise AssertionError(
        f"button {label!r} not found; have {[b.label for b in at.button]}"
    )


def test_run_handler_drives_a_live_callback(monkeypatch):
    """The run handler must hand the engine a real callback, and that
    callback must survive nonsense fractions rather than killing the solve.

    The app re-imports ``wrinklefe.analysis`` on every script execution, so
    patching the class method is enough to spy on what the app passes in.
    """
    pytest.importorskip(
        "streamlit.testing.v1", reason="Streamlit testing API not available."
    )
    from streamlit.testing.v1 import AppTest

    from wrinklefe.analysis import WrinkleAnalysis

    seen: dict[str, object] = {}
    real_run = WrinkleAnalysis.run

    def _spy(self, analytical_only=None, progress_callback=None):
        seen["callback"] = progress_callback
        if progress_callback is not None:
            # Values the engine would never emit — the app's guard must
            # clamp them instead of raising out of the solve.
            for bad in (-0.5, 42.0, float("nan")):
                progress_callback("Stub phase", bad)
        return real_run(
            self,
            analytical_only=analytical_only,
            progress_callback=progress_callback,
        )

    monkeypatch.setattr(WrinkleAnalysis, "run", _spy)

    at = AppTest.from_file(_app_path(), default_timeout=60)
    at.run()
    _click(at, "Run analysis")
    at.run()

    assert not at.exception, [str(e.value) for e in at.exception]
    assert callable(seen.get("callback")), (
        "the run handler did not pass a progress_callback to WrinkleAnalysis.run"
    )
    assert "results" in at.session_state


def test_app_run_populates_and_reuses_the_manual_cache():
    """A real run must fill the session cache, and a second identical run
    must be served from it instead of re-solving."""
    pytest.importorskip(
        "streamlit.testing.v1", reason="Streamlit testing API not available."
    )
    from streamlit.testing.v1 import AppTest

    import app as app_module

    at = AppTest.from_file(_app_path(), default_timeout=60)
    at.run()
    _click(at, "Run analysis")
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    cache = at.session_state[app_module._RUN_CACHE_KEY]
    assert len(cache) == 1
    cached_result = next(iter(cache.values()))
    assert at.session_state["results"] is cached_result

    # Re-run the same (untouched) configuration: same object back, no solve.
    _click(at, "Run analysis")
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert len(at.session_state[app_module._RUN_CACHE_KEY]) == 1
    assert at.session_state["results"] is cached_result
