"""Smoke tests for the Streamlit app's Cohesive Zone Modeling controls.

The Streamlit ``app.py`` exposes a collapsed *Cohesive Zone Modeling*
expander in the sidebar — visible only in **Expert mode**, because CZM
requires the full nonlinear FE solve which is itself expert-only. When
checked, the expander reveals toughness / strength / interface / Newton
inputs and switches the FE solve to the Newton-Raphson path. These tests
use ``streamlit.testing.v1.AppTest`` to drive the page without a real
browser and verify:

1. In the default (novice) sidebar the CZM checkbox is *hidden*; turning
   on ``session_state['expert_mode']`` surfaces it with
   ``key='sb_enable_czm'``, unchecked.
2. In Expert mode, setting ``session_state['sb_enable_czm'] = True``
   reveals the full set of CZM widgets (toughness, strength, interface
   placement, load increments, Newton tolerance).
3. The CZM defaults are exposed via the module-level ``DEFAULTS`` dict
   so the *Reset to defaults* button restores them.
4. The CZM result renderer (``_render_czm_results``) gracefully
   handles the sentinel case where ``enable_czm=True`` but no damage
   data was populated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

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
    """Import ``app.py`` for module-level constants and helpers."""
    import app as app_module  # noqa: WPS433 - test-time import.
    return app_module


# ----------------------------------------------------------------------
# 1. Imports cleanly + DEFAULTS round-trip
# ----------------------------------------------------------------------


def test_defaults_include_czm_entries(app_module):
    """``DEFAULTS`` must hold every CZM widget key so reset works."""
    d = app_module.DEFAULTS
    assert "sb_enable_czm" in d
    assert d["sb_enable_czm"] is False
    assert d["sb_czm_interfaces"] in ("near_crest", "all")
    assert isinstance(d["sb_czm_load_increments"], int)
    assert isinstance(d["sb_czm_newton_tol"], float)


def test_czm_defaults_match_analysis_config(app_module):
    """The CZM defaults must mirror ``AnalysisConfig()`` so the UI cannot
    silently drift from the analysis contract."""
    from wrinklefe.analysis import AnalysisConfig

    cfg = AnalysisConfig()
    assert app_module.DEFAULT_ENABLE_CZM == cfg.enable_czm
    assert app_module.DEFAULT_CZM_LOAD_INCREMENTS == cfg.czm_n_load_increments
    assert app_module.DEFAULT_CZM_NEWTON_TOL == cfg.czm_newton_tol


# ----------------------------------------------------------------------
# 2. AppTest-driven UI smoke tests
# ----------------------------------------------------------------------


def _app_path() -> str:
    return str(_REPO_ROOT / "app.py")


def test_novice_mode_hides_czm_controls():
    """Default (novice) page load: no exceptions, and the CZM checkbox is
    hidden because CZM depends on the expert-only FE solve."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_app_path(), default_timeout=30)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    # CZM checkbox is not rendered in the simplified novice sidebar.
    cbs = {cb.key for cb in at.checkbox if cb.key is not None}
    assert "sb_enable_czm" not in cbs, (
        f"CZM checkbox leaked into the novice UI; saw keys: {sorted(cbs)}"
    )

    # None of the dependent czm_* widgets should render either.
    czm_inputs = [ni for ni in at.number_input if "czm" in (ni.key or "")]
    assert czm_inputs == [], (
        f"CZM inputs leaked into the novice UI: {[ni.key for ni in czm_inputs]}"
    )


def test_expert_mode_shows_czm_checkbox_unchecked():
    """Turning on Expert mode surfaces the CZM checkbox, unchecked, with
    no dependent inputs revealed until it is ticked."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_app_path(), default_timeout=30)
    at.session_state["expert_mode"] = True
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    cbs = {cb.key: cb for cb in at.checkbox if cb.key is not None}
    assert "sb_enable_czm" in cbs, (
        f"CZM checkbox missing in Expert mode; saw keys: {sorted(cbs)}"
    )
    assert cbs["sb_enable_czm"].value is False

    czm_inputs = [ni for ni in at.number_input if "czm" in (ni.key or "")]
    assert czm_inputs == [], (
        f"CZM inputs shown before enabling: {[ni.key for ni in czm_inputs]}"
    )


def test_enabling_czm_reveals_full_widget_set():
    """In Expert mode, toggling ``sb_enable_czm`` reveals every CZM input."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_app_path(), default_timeout=30)
    at.session_state["expert_mode"] = True
    at.session_state["sb_enable_czm"] = True
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    expected_inputs = {
        "sb_czm_GIc",
        "sb_czm_GIIc",
        "sb_czm_sigma_max",
        "sb_czm_tau_max",
        "sb_czm_load_increments",
        "sb_czm_newton_tol",
    }
    actual_inputs = {ni.key for ni in at.number_input if ni.key is not None}
    missing = expected_inputs - actual_inputs
    assert not missing, (
        f"CZM number_input widgets missing after enable: {missing}; "
        f"saw {sorted(actual_inputs)}"
    )

    # ``sb_czm_interfaces`` is a selectbox.
    selectboxes = {sb.key for sb in at.selectbox if sb.key is not None}
    assert "sb_czm_interfaces" in selectboxes


def test_czm_widgets_seed_from_material_defaults():
    """The toughness / strength fields default to the chosen material's
    library values (IM7/8552: GIc=0.28, GIIc=0.79, sigma_max=80, tau_max=90)."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_app_path(), default_timeout=30)
    at.session_state["expert_mode"] = True
    at.session_state["sb_enable_czm"] = True
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    by_key = {
        ni.key: ni.value for ni in at.number_input if ni.key is not None
    }
    assert by_key["sb_czm_GIc"] == pytest.approx(0.28, rel=1e-3)
    assert by_key["sb_czm_GIIc"] == pytest.approx(0.79, rel=1e-3)
    assert by_key["sb_czm_sigma_max"] == pytest.approx(80.0, rel=1e-3)
    assert by_key["sb_czm_tau_max"] == pytest.approx(90.0, rel=1e-3)


# ----------------------------------------------------------------------
# 3. Result-rendering sentinel: enable_czm=True but no damage data
# ----------------------------------------------------------------------


def test_render_czm_results_handles_missing_damage(app_module):
    """``_render_czm_results`` must not crash when the solve produced
    no damage data — it should surface a warning instead."""
    from streamlit.testing.v1 import AppTest

    # Drive ``_render_czm_results`` indirectly by stuffing a CZM payload
    # whose ``_result`` carries an empty damage array into the session
    # state and running the page. AppTest exposes the rendered warnings
    # via the ``at.warning`` collection.
    from wrinklefe.analysis import AnalysisConfig, AnalysisResults

    empty_result = AnalysisResults(
        config=AnalysisConfig(),
        morphology_factor=1.0,
        max_angle_rad=0.0,
        effective_angle_rad=0.0,
        damage_index=0.0,
        analytical_knockdown=1.0,
        analytical_strength_MPa=1000.0,
        gamma_Y_eff=0.05,
        czm_damage=np.empty((0, 0)),
    )
    fake_results = {
        "summary": "stub summary",
        "loading": "tension",
        "applied_strain_abs": 0.01,
        "max_angle_deg": 0.0,
        "effective_angle_deg": 0.0,
        "morphology_factor": 1.0,
        "gamma_Y_eff": 0.05,
        "analytical_knockdown": 1.0,
        "analytical_strength_MPa": 1000.0,
        "damage_index": 0.0,
        "tension_mechanisms": None,
        "fe": None,
        "czm": {
            "enabled": True,
            "converged": True,
            "interfaces_used": [3],
            "max_damage": 0.0,
            "n_elements_above_half": 0,
            "energy_dissipated": 0.0,
            "crack_length_per_interface": {},
            "energy_per_interface": {},
            "_result": empty_result,
        },
    }
    # The session-state cfg_payload is consumed by the Export tab; an
    # empty tuple is safe because the export logic re-parses material /
    # angles defensively.
    fake_payload = (
        ("amplitude", 0.3),
        ("angles_tuple", (0, 45, -45, 90)),
        ("material_tuple", tuple()),
        ("wavelength", 16.0),
    )

    at = AppTest.from_file(_app_path(), default_timeout=30)
    at.session_state["results"] = fake_results
    at.session_state["cfg_payload"] = fake_payload
    at.run()
    # The CZM section must render *something* warning-like, not crash.
    assert not at.exception, [str(e.value) for e in at.exception]
    warning_texts = [w.value for w in at.warning]
    assert any("CZM" in t and "no damage" in t.lower() for t in warning_texts), (
        f"missing CZM-no-damage warning; saw warnings: {warning_texts!r}"
    )
