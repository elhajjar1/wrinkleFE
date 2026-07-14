"""Smoke tests for the Streamlit app's Surface resin pocket controls.

The Streamlit ``app.py`` exposes a collapsed *Surface resin pockets*
expander in the sidebar — visible only in **Expert mode**, because the
pockets are an FE-only material effect and the FE solve is itself
expert-only (issue #361, Part 4 follow-up). These tests use
``streamlit.testing.v1.AppTest`` to drive the page without a browser and
verify:

1. In the default (novice) sidebar the toggle is *hidden*; turning on
   ``session_state['expert_mode']`` surfaces it with
   ``key='sb_enable_surface_pockets'`` (unchecked) plus the side
   selectbox ``key='sb_surface_pocket_side'``.
2. The defaults are exposed via ``DEFAULTS`` so *Reset to defaults*
   restores them, and mirror ``AnalysisConfig``.
3. Toggling the feature on with a *compatible* (tool-flat) morphology
   renders without raising — the Analyze-tab cross-section shades the
   pockets.
4. Toggling it on with an *incompatible* morphology (uniform / graded
   with a decay floor) does not raise and surfaces the explanatory
   sidebar note, so a run can never build an invalid config.
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
pytest.importorskip(
    "streamlit.testing.v1", reason="Streamlit testing API not available."
)


@pytest.fixture(scope="module")
def app_module():
    import app as app_module  # noqa: WPS433 - test-time import.
    return app_module


def _app_path() -> str:
    return str(_REPO_ROOT / "app.py")


# ----------------------------------------------------------------------
# 1. DEFAULTS round-trip
# ----------------------------------------------------------------------


def test_defaults_include_surface_pocket_entries(app_module):
    """``DEFAULTS`` must hold both widget keys so Reset works."""
    d = app_module.DEFAULTS
    assert "sb_enable_surface_pockets" in d
    assert d["sb_enable_surface_pockets"] is False
    assert "sb_surface_pocket_side" in d
    assert d["sb_surface_pocket_side"] in ("top", "bottom", "both")


def test_surface_pocket_defaults_match_analysis_config(app_module):
    """UI defaults must mirror ``AnalysisConfig`` so they cannot drift."""
    from wrinklefe.analysis import AnalysisConfig

    cfg = AnalysisConfig()
    assert (
        app_module.DEFAULT_ENABLE_SURFACE_POCKETS
        == cfg.enable_surface_resin_pockets
    )
    assert app_module.DEFAULT_SURFACE_POCKET_SIDE == cfg.surface_pocket_side


# ----------------------------------------------------------------------
# 2. AppTest-driven UI smoke tests
# ----------------------------------------------------------------------


def test_novice_mode_hides_surface_pocket_controls():
    """Default (novice) load: the surface-pocket toggle is hidden."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_app_path(), default_timeout=30)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    cbs = {cb.key for cb in at.checkbox if cb.key is not None}
    assert "sb_enable_surface_pockets" not in cbs, (
        f"surface-pocket toggle leaked into novice UI; saw {sorted(cbs)}"
    )
    sbs = {sb.key for sb in at.selectbox if sb.key is not None}
    assert "sb_surface_pocket_side" not in sbs


def test_expert_mode_shows_surface_pocket_toggle_unchecked():
    """Expert mode surfaces the toggle (unchecked) and the side selectbox."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_app_path(), default_timeout=30)
    at.session_state["expert_mode"] = True
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    cbs = {cb.key: cb for cb in at.checkbox if cb.key is not None}
    assert "sb_enable_surface_pockets" in cbs, (
        f"toggle missing in Expert mode; saw {sorted(cbs)}"
    )
    assert cbs["sb_enable_surface_pockets"].value is False
    sbs = {sb.key for sb in at.selectbox if sb.key is not None}
    assert "sb_surface_pocket_side" in sbs


def test_enabling_on_compatible_morphology_does_not_raise():
    """Toggling on with a tool-flat morphology (stack) renders cleanly and
    shades the cross-section — no exception, no sidebar warning."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_app_path(), default_timeout=30)
    at.session_state["expert_mode"] = True
    at.session_state["sb_morphology"] = "stack"
    at.session_state["sb_enable_surface_pockets"] = True
    at.session_state["sb_surface_pocket_side"] = "both"
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    # A compatible morphology raises no tool-flat warning.
    warnings = [w.value for w in at.warning]
    assert not any("tool-flat morphology" in w for w in warnings), (
        f"unexpected incompatibility warning for stack: {warnings}"
    )


@pytest.mark.parametrize(
    "morphology,decay_floor", [("uniform", 0.0), ("graded", 0.4)]
)
def test_enabling_on_incompatible_morphology_warns_not_raises(
    morphology, decay_floor
):
    """Toggling on with a wavy morphology must not raise; it surfaces the
    explanatory note so the run cannot build an invalid config."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_app_path(), default_timeout=30)
    at.session_state["expert_mode"] = True
    at.session_state["sb_morphology"] = morphology
    if morphology == "graded":
        at.session_state["sb_decay_floor"] = decay_floor
    at.session_state["sb_enable_surface_pockets"] = True
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    warnings = [w.value for w in at.warning]
    assert any("tool-flat morphology" in w for w in warnings), (
        f"missing tool-flat note for {morphology}; saw {warnings}"
    )
