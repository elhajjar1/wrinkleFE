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
    """``DEFAULTS`` must hold every surface-pocket widget key so Reset works."""
    d = app_module.DEFAULTS
    assert "sb_enable_surface_pockets" in d
    assert d["sb_enable_surface_pockets"] is False
    assert "sb_surface_pocket_side" in d
    assert d["sb_surface_pocket_side"] in ("top", "bottom", "both")
    # tool_flat morphology's transition-ply control (issue #371).
    assert "sb_surface_transition_plies" in d
    assert d["sb_surface_transition_plies"] >= 1


def test_transition_plies_default_matches_analysis_config(app_module):
    """The tool_flat transition-ply default must mirror ``AnalysisConfig``."""
    from wrinklefe.analysis import AnalysisConfig

    cfg = AnalysisConfig()
    assert (
        app_module.DEFAULT_SURFACE_TRANSITION_PLIES
        == cfg.surface_transition_plies
    )
    assert (
        app_module.DEFAULTS["sb_surface_transition_plies"]
        == cfg.surface_transition_plies
    )


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


# ----------------------------------------------------------------------
# 3. tool_flat morphology — controls live in the morphology definition
#    (issue #371). The side + transition controls appear directly with
#    the morphology; the standalone legacy checkbox is folded away.
# ----------------------------------------------------------------------


def test_tool_flat_is_selectable_in_expert_mode():
    """``tool_flat`` joins the Morphology selectbox in Expert mode."""
    assert "tool_flat" in __import__("app").MORPHOLOGIES


def test_selecting_tool_flat_reveals_controls_and_hides_checkbox():
    """Selecting tool_flat surfaces the pinned-side + transition-ply
    controls and hides the standalone surface-pocket checkbox."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_app_path(), default_timeout=30)
    at.session_state["expert_mode"] = True
    at.session_state["sb_morphology"] = "tool_flat"
    # A safe amplitude keeps the config valid on the FE path.
    at.session_state["sb_amplitude"] = 0.2
    at.session_state["sb_surface_transition_plies"] = 3
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    number_keys = {n.key for n in at.number_input if n.key is not None}
    assert "sb_surface_transition_plies" in number_keys, (
        f"transition-ply control missing for tool_flat; saw {number_keys}"
    )
    sel_keys = {s.key for s in at.selectbox if s.key is not None}
    assert "sb_surface_pocket_side" in sel_keys
    # The standalone legacy checkbox is folded away for tool_flat.
    cb_keys = {c.key for c in at.checkbox if c.key is not None}
    assert "sb_enable_surface_pockets" not in cb_keys, (
        f"standalone surface-pocket checkbox leaked for tool_flat: {cb_keys}"
    )


def test_legacy_opt_in_still_reachable_for_stack():
    """A legacy morphology keeps the advanced opt-in checkbox (and hides
    the tool_flat-only transition control)."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_app_path(), default_timeout=30)
    at.session_state["expert_mode"] = True
    at.session_state["sb_morphology"] = "stack"
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    cb_keys = {c.key for c in at.checkbox if c.key is not None}
    assert "sb_enable_surface_pockets" in cb_keys, (
        f"legacy opt-in checkbox missing for stack; saw {cb_keys}"
    )
    number_keys = {n.key for n in at.number_input if n.key is not None}
    assert "sb_surface_transition_plies" not in number_keys


def test_tool_flat_over_bound_amplitude_warns_before_run():
    """An amplitude above the inversion bound surfaces a pre-run warning
    (naming both remedies) so the config ValueError is never first feedback."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_app_path(), default_timeout=30)
    at.session_state["expert_mode"] = True
    at.session_state["sb_morphology"] = "tool_flat"
    # Default amplitude 0.366 mm exceeds 0.8*2*0.183/1 = 0.293 mm at S = 2.
    at.session_state["sb_amplitude"] = 0.366
    at.session_state["sb_surface_transition_plies"] = 2
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    warnings = [w.value for w in at.warning]
    assert any("inversion bound" in w for w in warnings), (
        f"missing pre-run inversion-bound warning; saw {warnings}"
    )
    # Both remedies are named (more transition plies OR smaller amplitude).
    joined = " ".join(warnings)
    assert "transition plies" in joined and "amplitude" in joined
