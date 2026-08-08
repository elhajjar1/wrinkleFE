"""Tests for the Streamlit app's config upload/download and the through-width
transverse controls (issue #375 app slice).

Two layers of coverage:

1. Pure-function tests import ``app`` and exercise the seeding / parsing
   helpers directly with plain dicts (the same headless pattern as
   ``test_app_reset_inputs``).
2. ``AppTest`` drives the real Streamlit script to confirm the download
   serialises a round-trippable config, an uploaded config seeds the
   sidebar widgets, the transverse controls appear in Expert mode and
   thread into the run config, and an incompatible combo warns without
   crashing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# ``app.py`` lives at the repo root, not under ``src/``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pytest.importorskip("streamlit", reason="Streamlit not installed.")

import matplotlib  # noqa: E402

pytestmark = pytest.mark.viz

matplotlib.use("Agg")


def _app_path() -> str:
    return str(_REPO_ROOT / "app.py")


@pytest.fixture(scope="module")
def app_module():
    import app as app_module  # noqa: WPS433 - test-time import.
    return app_module


# ---------------------------------------------------------------------------
# Pure-function coverage: parse + seed helpers.
# ---------------------------------------------------------------------------


def test_parse_uploaded_config_roundtrips_json(app_module):
    """A JSON dump from ``to_json`` parses back to an equal config."""
    from wrinklefe.analysis import AnalysisConfig

    cfg = AnalysisConfig(amplitude=0.42, wavelength=18.0, morphology="uniform")
    parsed = app_module._parse_uploaded_config(
        "case.json", cfg.to_json().encode("utf-8")
    )
    assert parsed.to_dict() == cfg.to_dict()


def test_parse_uploaded_config_bad_json_raises(app_module):
    """A malformed / unknown-key file raises ``ValueError`` (surfaced as
    ``st.error`` by the sidebar, never a crash)."""
    with pytest.raises(Exception):
        app_module._parse_uploaded_config("case.json", b"{not valid json")
    # Valid JSON but an unknown key -> AnalysisConfig.from_dict rejects it.
    payload = json.dumps(
        {"config_version": "1", "not_a_field": 3}
    ).encode("utf-8")
    with pytest.raises(ValueError):
        app_module._parse_uploaded_config("case.json", payload)


def test_seed_state_from_config_maps_fields(app_module):
    """``_seed_state_from_config`` is the inverse of the sidebar mapping."""
    from wrinklefe.analysis import AnalysisConfig
    from wrinklefe.core.layup import parse_layup

    cfg = AnalysisConfig(
        amplitude=0.5,
        wavelength=22.0,
        width=9.0,
        morphology="graded",
        decay_floor=0.3,
        loading="tension",
        applied_strain=0.02,
        angles=[0, 45, -45, 90, 90, -45, 45, 0],
        ply_thickness=0.2,
        transverse_mode="gaussian_decay",
        transverse_span=16.0,
        transverse_width=3.0,
        analytical_only=False,
        nx=10,
        ny=8,
        nz_per_ply=2,
    )
    state: dict[str, object] = {}
    app_module._seed_state_from_config(cfg, state)

    assert state["expert_mode"] is True
    assert state["sb_amplitude"] == 0.5
    assert state["sb_wavelength"] == 22.0
    assert state["sb_width"] == 9.0
    assert state["sb_morphology"] == "graded"
    assert state["sb_decay_floor"] == 0.3
    assert state["sb_loading"] == "tension"
    # Loading sign lives on the radio; magnitude on the number input.
    assert state["sb_strain_mag_pct"] == pytest.approx(2.0)
    assert parse_layup(state["sb_layup"]) == [0, 45, -45, 90, 90, -45, 45, 0]
    assert state["sb_ply_thickness"] == 0.2
    assert state["sb_transverse_mode"] == "gaussian_decay"
    assert state["sb_transverse_span"] == 16.0
    assert state["sb_transverse_width"] == 3.0
    assert state["sb_analytical_only"] is False
    assert state["sb_nx"] == 10
    assert state["sb_ny"] == 8
    assert state["sb_nz_per_ply"] == 2


def test_seed_state_auto_transverse_sentinels(app_module):
    """A ``None`` transverse span/width seeds the 0 = auto sentinel."""
    from wrinklefe.analysis import AnalysisConfig

    cfg = AnalysisConfig(
        morphology="uniform",
        transverse_mode="sinusoidal_y",
        analytical_only=False,
    )
    state: dict[str, object] = {}
    app_module._seed_state_from_config(cfg, state)
    assert state["sb_transverse_span"] == 0.0
    assert state["sb_transverse_width"] == 0.0


def test_seed_state_preset_material(app_module):
    """A library-preset material selects the matching dropdown entry."""
    from wrinklefe.analysis import AnalysisConfig
    from wrinklefe.core.material import MaterialLibrary

    name = app_module.MATERIAL_NAMES[0]
    cfg = AnalysisConfig(material=MaterialLibrary().get(name))
    state: dict[str, object] = {"custom_E1": 999.0}
    app_module._seed_state_from_config(cfg, state)
    assert state["sb_material"] == name
    # Stale custom editor keys are cleared on a preset load.
    assert "custom_E1" not in state


def test_seed_state_custom_material(app_module):
    """A tweaked (custom) material routes through the custom editor keys."""
    from wrinklefe.analysis import AnalysisConfig
    from wrinklefe.core.material import MaterialLibrary

    base = MaterialLibrary().get(app_module.MATERIAL_NAMES[0]).to_dict()
    base["E1"] = base["E1"] * 1.10  # tweak -> no longer a preset
    base["name"] = "my_custom"
    from wrinklefe.core.material import OrthotropicMaterial

    cfg = AnalysisConfig(material=OrthotropicMaterial.from_dict(base))
    state: dict[str, object] = {}
    app_module._seed_state_from_config(cfg, state)
    assert state["sb_material"] == app_module.CUSTOM_MATERIAL_LABEL
    assert state["sb_custom_name"] == "my_custom"
    assert state["custom_E1"] == pytest.approx(base["E1"])


def test_seed_state_clamps_out_of_range(app_module):
    """A physically valid but out-of-widget-range value is clamped so the
    seeding cannot raise a StreamlitAPIException on the rerun."""
    from wrinklefe.analysis import AnalysisConfig

    cfg = AnalysisConfig(amplitude=50.0, wavelength=500.0)  # exceed widget max
    state: dict[str, object] = {}
    app_module._seed_state_from_config(cfg, state)
    assert state["sb_amplitude"] == 5.0  # clamped to the number_input max
    assert state["sb_wavelength"] == 200.0


# ---------------------------------------------------------------------------
# AppTest coverage: real Streamlit script.
# ---------------------------------------------------------------------------


def test_download_config_button_present_and_roundtrips():
    """The sidebar exposes a config-download button whose serialised bytes
    round-trip through ``AnalysisConfig.from_dict`` (works before any run)."""
    from streamlit.testing.v1 import AppTest

    from wrinklefe.analysis import AnalysisConfig

    at = AppTest.from_file(_app_path(), default_timeout=90)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    labels = [b.label for b in at.button] + [
        getattr(b, "label", None) for b in at.get("download_button")
    ]
    assert "Download config (JSON)" in labels

    eff = at.session_state["_effective_config_json"]
    assert eff is not None
    # The default sidebar state serialises to a valid, parseable config.
    AnalysisConfig.from_dict(json.loads(eff))


def test_upload_seeds_widgets_and_roundtrips_config():
    """Staging an uploaded config seeds the sidebar widgets, and the
    re-serialised effective config equals the uploaded one (download ->
    upload -> rebuild round-trip)."""
    from streamlit.testing.v1 import AppTest

    from wrinklefe.analysis import AnalysisConfig

    cfg = AnalysisConfig(
        amplitude=0.44,
        wavelength=19.0,
        morphology="uniform",
        transverse_mode="gaussian_decay",
        transverse_span=17.0,
        transverse_width=4.0,
        analytical_only=False,
    )
    at = AppTest.from_file(_app_path(), default_timeout=90)
    at.run()
    # Simulate a parsed upload staged for the top-of-sidebar seeding block.
    at.session_state["_pending_config"] = cfg
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    # Widgets reflect the loaded values.
    assert at.session_state["sb_amplitude"] == pytest.approx(0.44)
    assert at.session_state["sb_morphology"] == "uniform"
    assert at.session_state["sb_transverse_mode"] == "gaussian_decay"
    assert at.session_state["expert_mode"] is True

    # The re-serialised effective config equals the uploaded one.
    rebuilt = AnalysisConfig.from_dict(
        json.loads(at.session_state["_effective_config_json"])
    )
    assert rebuilt.to_dict() == cfg.to_dict()


def test_transverse_controls_thread_into_run_config():
    """Selecting a non-uniform transverse mode in Expert mode threads
    ``transverse_mode`` into the effective run config and forces the FE
    path (analytical-only is overridden)."""
    from streamlit.testing.v1 import AppTest

    from wrinklefe.analysis import AnalysisConfig

    at = AppTest.from_file(_app_path(), default_timeout=90)
    at.run()
    at.toggle(key="expert_mode").set_value(True)
    at.run()
    at.selectbox(key="sb_morphology").set_value("uniform")
    at.selectbox(key="sb_transverse_mode").set_value("gaussian_decay")
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    eff = AnalysisConfig.from_dict(
        json.loads(at.session_state["_effective_config_json"])
    )
    assert eff.transverse_mode == "gaussian_decay"
    # Non-uniform transverse is FE-only.
    assert eff.analytical_only is False


def test_transverse_czm_incompatible_warns_without_crash():
    """A non-uniform transverse mode combined with CZM warns and does NOT
    thread the transverse flags into an invalid config (no crash)."""
    from streamlit.testing.v1 import AppTest

    from wrinklefe.analysis import AnalysisConfig

    at = AppTest.from_file(_app_path(), default_timeout=90)
    at.run()
    at.toggle(key="expert_mode").set_value(True)
    at.run()
    at.selectbox(key="sb_morphology").set_value("uniform")
    at.selectbox(key="sb_transverse_mode").set_value("gaussian_decay")
    at.checkbox(key="sb_enable_czm").set_value(True)
    at.run()
    # The incompatible combo must not crash the script.
    assert not at.exception, [str(e.value) for e in at.exception]
    warnings = " ".join(w.value for w in at.warning)
    assert "transverse" in warnings.lower()

    # The effective config must stay valid: transverse is dropped, CZM wins.
    eff = AnalysisConfig.from_dict(
        json.loads(at.session_state["_effective_config_json"])
    )
    assert eff.transverse_mode == "uniform"
    assert eff.enable_czm is True
