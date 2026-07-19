"""AppTest coverage for issue #374 items 2: Reset clears rendered results,
and a prominent staleness banner appears when the live sidebar inputs no
longer match the configuration the displayed results were computed from.

These drive the real Streamlit script (analytical-only so the run is fast)
via ``AppTest`` and skip when Streamlit is not installed (it is under CI).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pytest.importorskip("streamlit", reason="Streamlit not installed.")

import matplotlib  # noqa: E402

pytestmark = pytest.mark.viz

matplotlib.use("Agg")

_STALE_MARKER = "Inputs have changed since this run"


def _app_path() -> str:
    return str(_REPO_ROOT / "app.py")


def _click(at, label: str) -> None:
    for b in at.button:
        if b.label == label:
            b.click()
            return
    raise AssertionError(f"button {label!r} not found; have {[b.label for b in at.button]}")


def _seed_analytical_run(at) -> None:
    """Run the app once via the sidebar Run button. The default (non-expert)
    path is analytical-only, so the run is fast and needs no FE solve."""
    _click(at, "Run analysis")
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert "results" in at.session_state


def test_run_then_change_input_shows_staleness_warning():
    """After a run, editing a sidebar value must surface the staleness
    banner over the (now previous-config) results."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_app_path(), default_timeout=60)
    at.run()
    _seed_analytical_run(at)

    # Immediately after the run the live inputs match — no staleness banner.
    assert not any(_STALE_MARKER in w.value for w in at.warning), (
        "staleness banner shown when inputs are unchanged"
    )

    # Change a geometry input without re-running the analysis.
    at.number_input(key="sb_amplitude").set_value(0.55)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    assert "results" in at.session_state  # results still present…
    assert any(_STALE_MARKER in w.value for w in at.warning), (
        "expected the staleness banner after editing a sidebar input; "
        f"warnings were {[w.value for w in at.warning]}"
    )


def test_reset_clears_rendered_results():
    """Clicking Reset must drop the results so the Analyze tab returns to
    its empty 'No results yet' state instead of rendering the old run."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_app_path(), default_timeout=60)
    at.run()
    _seed_analytical_run(at)

    _click(at, "↻ Reset to defaults")
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    assert "results" not in at.session_state
    assert "cfg_payload" not in at.session_state
    markdown_blob = " ".join(m.value for m in at.markdown)
    assert "No results yet" in markdown_blob
