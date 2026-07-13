"""Tests for the Streamlit sidebar "Reset to defaults" helper (issue #184).

The Streamlit ``app.py`` exposes a ``reset_inputs()`` helper and a
``DEFAULTS`` dict that the sidebar's reset button consumes. These tests
guard two invariants:

1. Every default that has a counterpart on :class:`AnalysisConfig` matches
   the dataclass value, so the UI cannot silently drift from the analysis
   contract.
2. ``reset_inputs()`` writes every ``DEFAULTS`` entry into a fresh
   session-state mapping and clears the dynamic ``custom_*`` editor keys.
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


@pytest.fixture(scope="module")
def app_module():
    """Import ``app.py`` with a stubbed Streamlit module.

    Streamlit's top-level ``st.set_page_config``, ``st.sidebar``, widget
    calls etc. all run at import time. We don't need a real session here —
    only the module-level constants and the ``reset_inputs`` helper — so a
    minimal stub keeps the test fast and headless.
    """
    pytest.importorskip("streamlit", reason="Streamlit not installed.")

    import app as app_module  # noqa: WPS433 – test-time import.
    return app_module


def test_defaults_match_analysis_config(app_module):
    """Numerical / categorical defaults must match ``AnalysisConfig()``."""
    from wrinklefe.analysis import AnalysisConfig

    cfg = AnalysisConfig()
    overlap = {
        "sb_amplitude": cfg.amplitude,
        "sb_wavelength": cfg.wavelength,
        "sb_width": cfg.width,
        "sb_ply_thickness": cfg.ply_thickness,
        "sb_morphology": cfg.morphology,
        "sb_decay_floor": cfg.decay_floor,
        "sb_analytical_only": cfg.analytical_only,
        "sb_nx": cfg.nx,
        "sb_ny": cfg.ny,
        "sb_nz_per_ply": cfg.nz_per_ply,
    }
    for key, expected in overlap.items():
        assert key in app_module.DEFAULTS, f"{key} missing from DEFAULTS"
        assert app_module.DEFAULTS[key] == expected, (
            f"DEFAULTS[{key!r}] = {app_module.DEFAULTS[key]!r} drifted from "
            f"AnalysisConfig().{key.removeprefix('sb_')} = {expected!r}"
        )


def test_defaults_include_ui_only_inputs(app_module):
    """UI-only inputs (layup string, material, loading) must have defaults."""
    d = app_module.DEFAULTS
    assert d["sb_layup"] == "[0/45/-45/90]_3s"
    assert d["sb_material"] == "IM7_8552"
    assert d["sb_loading"] == "compression"
    assert d["sb_custom_name"] == "custom"
    assert d["expert_mode"] is False
    assert d["sb_strain_mag_pct"] == 1.0


def test_defaults_include_amplitude_profile_controls(app_module):
    """The amplitude-profile sidebar group (issue #182) must round-trip
    through ``DEFAULTS`` so the Reset button can restore it."""
    from wrinklefe.analysis import AnalysisConfig

    cfg = AnalysisConfig()
    d = app_module.DEFAULTS
    # Categorical defaults track the dataclass exactly.
    assert d["sb_amplitude_profile"] == cfg.amplitude_profile
    assert d["sb_amplitude_profile_axis"] == cfg.amplitude_profile_axis
    # Numeric default is concrete (the dataclass default ``None`` falls
    # back to the wrinkle width at apply-time, and Streamlit's
    # ``number_input`` cannot represent ``None``). It must be a finite
    # positive float so the widget renders.
    decay = d["sb_amplitude_profile_decay_length"]
    assert isinstance(decay, float) and decay > 0.0


def test_reset_inputs_restores_every_default(app_module, monkeypatch):
    """``reset_inputs()`` must write every DEFAULTS entry into session_state."""
    fake_state: dict[str, object] = {
        # Pretend the user tweaked everything.
        "sb_amplitude": 999.0,
        "sb_wavelength": 1.0,
        "sb_layup": "[0]_2",
        "sb_material": "Something_else",
        "expert_mode": True,
        # Dynamic custom-material editor keys must also be cleared.
        "custom_E1": 1.23e5,
        "custom_Xt": 4567.0,
        # An unrelated key must be left alone.
        "unrelated": "keep me",
    }
    monkeypatch.setattr(app_module.st, "session_state", fake_state)

    app_module.reset_inputs()

    for key, expected in app_module.DEFAULTS.items():
        assert fake_state[key] == expected, (
            f"reset_inputs did not restore {key!r}: "
            f"got {fake_state[key]!r}, expected {expected!r}"
        )
    # ``custom_*`` keys must be gone after a reset.
    assert "custom_E1" not in fake_state
    assert "custom_Xt" not in fake_state
    # Unrelated keys are preserved.
    assert fake_state["unrelated"] == "keep me"


def test_defaults_cover_every_sidebar_key(app_module):
    """``DEFAULTS`` must cover every ``key=`` used by sidebar analysis inputs.

    Guards against new widgets being added without a matching default.
    """
    import ast

    src = (Path(app_module.__file__)).read_text()
    tree = ast.parse(src)

    sidebar_keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "key" or not isinstance(kw.value, ast.Constant):
                continue
            key = kw.value.value
            # Skip visualisation widgets (issue #184 scope: analysis inputs).
            if not isinstance(key, str) or key.startswith("viz_"):
                continue
            sidebar_keys.add(key)

    missing = sidebar_keys - set(app_module.DEFAULTS)
    assert not missing, (
        f"Sidebar widgets without DEFAULTS entries: {sorted(missing)}"
    )
