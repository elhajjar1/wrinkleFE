"""The app's permalink surface (issue #281).

The codec itself is covered by ``tests/test_io/test_permalink.py``; this is
the wiring: an inbound ``?cfg=`` reaching the sidebar exactly once, a bad one
reporting itself instead of crashing the load, and the Share expander
offering a link that round-trips back to the same sidebar state.

``AppTest`` has no way to set query parameters, so the inbound half is driven
through ``_consume_permalink`` with plain dicts -- which is why it takes
``params`` and ``state`` as arguments rather than reading globals. The
round-trip half goes through the real app, using the same staging slot the
sidebar's permalink branch writes to.
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


def _fresh_app(timeout: float = 120.0):
    """An ``AppTest`` past the acknowledgement gate.

    Without it the script stops before the lower sidebar blocks — which is
    where the Share expander and ``_effective_config_json`` live — so a test
    that forgets this passes or fails on nothing.
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_app_path(), default_timeout=timeout)
    at.session_state["_wf_acknowledged"] = True
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


@pytest.fixture(scope="module")
def app_module():
    import app as app_module  # noqa: WPS433 — test-time import.

    return app_module


# ----------------------------------------------------------------------
# Inbound: reading ?cfg= once, and never crashing on a bad one
# ----------------------------------------------------------------------

class TestConsumeInbound:
    def test_a_valid_payload_yields_the_config(self, app_module):
        from wrinklefe.analysis import AnalysisConfig
        from wrinklefe.io.permalink import QUERY_PARAM, encode_config

        cfg = AnalysisConfig(amplitude=0.44, wavelength=19.0)
        state: dict = {}
        config, warnings, error = app_module._consume_permalink(
            {QUERY_PARAM: encode_config(cfg)}, state
        )
        assert error is None and warnings == ()
        assert config is not None
        assert config.to_dict() == cfg.to_dict()

    def test_no_param_is_not_an_error(self, app_module):
        config, warnings, error = app_module._consume_permalink({}, {})
        assert (config, warnings, error) == (None, (), None)

    def test_a_bad_payload_reports_instead_of_raising(self, app_module):
        """AC2 at the app boundary: a hand-mangled link must not become an
        exception page on load."""
        config, warnings, error = app_module._consume_permalink(
            {"cfg": "!!!not a permalink!!!"}, {}
        )
        assert config is None and warnings == ()
        assert error and "base64" in error

    def test_it_is_consumed_once_per_session(self, app_module):
        """The parameter is deliberately left in the URL so the link stays
        bookmarkable, so it is still there on every rerun. Re-seeding from
        it would overwrite the visitor's edits each time they touched a
        widget."""
        from wrinklefe.analysis import AnalysisConfig
        from wrinklefe.io.permalink import QUERY_PARAM, encode_config

        params = {QUERY_PARAM: encode_config(AnalysisConfig(amplitude=0.3))}
        state: dict = {}
        first = app_module._consume_permalink(params, state)
        assert first[0] is not None
        # The parameter has NOT been removed ...
        assert QUERY_PARAM in params
        # ... but a second read (a rerun) yields nothing to apply.
        assert app_module._consume_permalink(params, state) == (None, (), None)

    def test_a_bad_payload_is_only_reported_once(self, app_module):
        """The consumed flag is set before the decode, so a broken link does
        not re-raise its error on every rerun of the session."""
        params = {"cfg": "garbage!!"}
        state: dict = {}
        assert app_module._consume_permalink(params, state)[2] is not None
        assert app_module._consume_permalink(params, state)[2] is None

    def test_a_repeated_parameter_takes_the_first(self, app_module):
        """Streamlit hands back a list when a parameter appears twice; the
        obvious reading beats an error page."""
        from wrinklefe.analysis import AnalysisConfig
        from wrinklefe.io.permalink import QUERY_PARAM, encode_config

        payload = encode_config(AnalysisConfig(amplitude=0.37))
        config, _, error = app_module._consume_permalink(
            {QUERY_PARAM: [payload, "junk"]}, {}
        )
        assert error is None
        assert config is not None and config.amplitude == pytest.approx(0.37)

    def test_an_empty_repeated_parameter_is_reported(self, app_module):
        config, _, error = app_module._consume_permalink({"cfg": []}, {})
        assert config is None and error


# ----------------------------------------------------------------------
# The base URL a link is built against
# ----------------------------------------------------------------------

class TestBaseUrl:
    def test_it_falls_back_to_the_deployment_outside_a_script_run(
        self, app_module
    ):
        """``st.context.url`` is unavailable here, and a relative link the
        user has to assemble by hand is not a shareable link."""
        assert app_module._app_base_url() == (
            app_module.PERMALINK_FALLBACK_BASE_URL
        )
        assert app_module._app_base_url().startswith("https://")


# ----------------------------------------------------------------------
# AC1 — round trip through the real app
# ----------------------------------------------------------------------

class TestRoundTripThroughTheApp:
    def test_sidebar_state_survives_a_permalink(self):
        """Configure a non-default case, take its permalink, decode it into
        a fresh session, and the effective config comes back identical."""

        from wrinklefe.analysis import AnalysisConfig
        from wrinklefe.io.permalink import decode_config, encode_config

        # A case built by driving the real widgets, not by hand.
        at = _fresh_app()
        at.toggle(key="expert_mode").set_value(True)
        at.run()
        at.number_input(key="sb_amplitude").set_value(0.47)
        at.number_input(key="sb_wavelength").set_value(21.0)
        at.selectbox(key="sb_morphology").set_value("convex")
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]

        shared = AnalysisConfig.from_dict(
            json.loads(at.session_state["_effective_config_json"])
        )
        assert shared.amplitude == pytest.approx(0.47)

        payload = encode_config(shared)
        decoded = decode_config(payload)
        assert decoded.warnings == ()

        # A fresh session, seeded the way the sidebar's permalink branch
        # seeds it.
        fresh = _fresh_app()
        fresh.session_state["_pending_config"] = decoded.config
        fresh.run()
        assert not fresh.exception, [str(e.value) for e in fresh.exception]

        reopened = AnalysisConfig.from_dict(
            json.loads(fresh.session_state["_effective_config_json"])
        )
        assert reopened.to_dict() == shared.to_dict()
        # And the widgets themselves, not just the derived config.
        assert fresh.session_state["sb_amplitude"] == pytest.approx(0.47)
        assert fresh.session_state["sb_wavelength"] == pytest.approx(21.0)
        assert fresh.session_state["sb_morphology"] == "convex"

    def test_a_custom_material_survives_the_app_round_trip(self):
        """AC3 through the app: a custom material must come back as the
        custom-material editor's state, not as a preset."""

        from wrinklefe.analysis import AnalysisConfig
        from wrinklefe.core.material import OrthotropicMaterial
        from wrinklefe.io.permalink import decode_config, encode_config

        # Interface properties included: a material without them picks
        # them up from the sidebar's CZM defaults, on this path and on the
        # config-file upload path alike, which would make this a test of
        # that behaviour rather than of the permalink.
        custom = OrthotropicMaterial(
            name="MyTape-X", E1=152000.0, E2=9100.0, E3=9100.0,
            G12=5300.0, G13=5300.0, G23=3400.0,
            nu12=0.31, nu13=0.31, nu23=0.46,
            Xt=2600.0, Xc=1700.0, Yt=70.0, Yc=260.0,
            S12=95.0, S13=95.0, S23=70.0,
            GIc=0.28, GIIc=0.79, sigma_max=80.0, tau_max=90.0,
        )
        cfg = AnalysisConfig(amplitude=0.4, material=custom)
        decoded = decode_config(encode_config(cfg))

        at = _fresh_app()
        at.session_state["_pending_config"] = decoded.config
        at.run()
        assert not at.exception, [str(e.value) for e in at.exception]

        assert at.session_state["sb_material"] == app_custom_label()
        assert at.session_state["sb_custom_name"] == "MyTape-X"
        assert at.session_state["custom_E1"] == pytest.approx(152000.0)

        rebuilt = AnalysisConfig.from_dict(
            json.loads(at.session_state["_effective_config_json"])
        )
        assert rebuilt.material is not None
        assert rebuilt.material.to_dict() == custom.to_dict()


def app_custom_label() -> str:
    import app as app_module

    return app_module.CUSTOM_MATERIAL_LABEL


# ----------------------------------------------------------------------
# The Share expander
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def ran_app():
    """One app run shared by the Share-expander assertions — booting the
    script is the expensive part and none of them mutate it."""
    return _fresh_app()


class TestShareUi:
    def test_the_sidebar_offers_a_link(self, ran_app):
        """The link is rendered as ``st.code``, which carries Streamlit's
        own copy button — no clipboard JavaScript to maintain."""
        from wrinklefe.io.permalink import QUERY_PARAM

        blocks = [c.value for c in ran_app.get("code")]
        links = [b for b in blocks if f"?{QUERY_PARAM}=" in str(b)]
        assert links, blocks

    def test_the_offered_link_decodes_to_the_current_config(self, ran_app):
        """The link and the config-file download come off one builder
        (``_current_config``), so a shared link and a saved file describe
        the same case."""
        from urllib.parse import parse_qs, urlparse

        from wrinklefe.analysis import AnalysisConfig
        from wrinklefe.io.permalink import QUERY_PARAM, decode_config

        link = next(
            str(c.value) for c in ran_app.get("code")
            if f"?{QUERY_PARAM}=" in str(c.value)
        )
        payload = parse_qs(urlparse(link).query)[QUERY_PARAM][0]
        from_link = decode_config(payload).config
        from_file = AnalysisConfig.from_dict(
            json.loads(ran_app.session_state["_effective_config_json"])
        )
        assert from_link.to_dict() == from_file.to_dict()

    def test_the_link_is_short_enough_to_paste(self, ran_app):
        from wrinklefe.io.permalink import QUERY_PARAM

        link = next(
            str(c.value) for c in ran_app.get("code")
            if f"?{QUERY_PARAM}=" in str(c.value)
        )
        assert len(link) < 2000, len(link)
