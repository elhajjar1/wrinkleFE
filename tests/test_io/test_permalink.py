"""Permalinks: encode a config into a URL, decode it back (issue #281).

Four things have to hold, and they map onto the issue's acceptance criteria:

1. **Round trip.** A configured case survives encode -> decode exactly,
   including a custom material, which is the case most likely to be dropped
   by a "just put the interesting numbers in the URL" shortcut.
2. **Hostile input is contained.** The payload comes from the URL bar, so
   every malformed, truncated, hand-edited or deliberately abusive value has
   to produce one ``PermalinkError`` -- never an exception page, and never a
   zlib bomb expanding into the server's memory.
3. **Length.** A full non-default config must stay inside practical URL
   limits.
4. **The delta is honest.** Only non-default fields travel, which means the
   link means "whatever the defaults are when it is read". When those
   defaults differ from the ones it was built against, the reader is told
   rather than handed a quietly different case.
"""

from __future__ import annotations

import base64
import json
import zlib

import pytest

from wrinklefe.analysis import CONFIG_VERSION, AnalysisConfig
from wrinklefe.core.material import MaterialLibrary, OrthotropicMaterial
from wrinklefe.io.permalink import (
    MAX_PAYLOAD_CHARS,
    PERMALINK_VERSION,
    QUERY_PARAM,
    PermalinkError,
    decode_config,
    encode_config,
    permalink_url,
)

#: The issue's bound: "URL length stays within practical limits (< ~2k chars)".
URL_LIMIT = 2000

_CUSTOM = OrthotropicMaterial(
    name="MyTape-X", E1=152000.0, E2=9100.0, E3=9100.0,
    G12=5300.0, G13=5300.0, G23=3400.0, nu12=0.31, nu13=0.31, nu23=0.46,
    Xt=2600.0, Xc=1700.0, Yt=70.0, Yc=260.0, S12=95.0, S13=95.0, S23=70.0,
)


def _everything_non_default() -> AnalysisConfig:
    """A config that differs from the defaults in as many fields as the
    sidebar can reach, with a custom material -- the worst case for URL
    length and for the delta encoding."""
    return AnalysisConfig(
        amplitude=0.62, wavelength=22.5, width=18.0, ply_thickness=0.191,
        angles=[0, 45, -45, 90, 90, -45, 45, 0] * 2,
        morphology="graded", loading="compression", applied_strain=-0.0085,
        delta_T=-155.0, nx=28, ny=12, nz_per_ply=2, analytical_only=False,
        material=_CUSTOM, enable_czm=True, czm_GIc=0.28, czm_GIIc=0.85,
        czm_sigma_max=62.0, czm_tau_max=92.0, czm_n_load_increments=40,
        enable_surface_resin_pockets=True,
        surface_pocket_side="both", surface_transition_plies=3,
    )


def _pack(envelope: object) -> str:
    """Hand-build a payload, for testing envelopes the encoder would not emit."""
    raw = json.dumps(envelope, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(zlib.compress(raw, 9)).decode().rstrip("=")


# ----------------------------------------------------------------------
# AC1 — round trip
# ----------------------------------------------------------------------

class TestRoundTrip:
    @pytest.mark.parametrize(
        "config",
        [
            pytest.param(AnalysisConfig(), id="defaults"),
            pytest.param(
                AnalysisConfig(amplitude=0.42, wavelength=18.0),
                id="two-parameter-case",
            ),
            pytest.param(
                AnalysisConfig(morphology="graded", loading="tension"),
                id="string-fields",
            ),
            pytest.param(
                AnalysisConfig(angles=[0, 90, 90, 0], ply_thickness=0.25),
                id="layup",
            ),
            pytest.param(
                AnalysisConfig(material=MaterialLibrary().get("T800S_M21")),
                id="library-preset",
            ),
            pytest.param(_everything_non_default(), id="everything"),
        ],
    )
    def test_config_survives_exactly(self, config):
        """``to_dict`` equality, not field spot-checks: a permalink that
        drops one of 69 fields is worse than one that fails, because the
        colleague who opens it analyses a slightly different defect."""
        restored = decode_config(encode_config(config)).config
        assert restored.to_dict() == config.to_dict()

    def test_a_custom_material_survives(self):
        """AC3. The sidebar's custom-material editor is the whole reason the
        payload cannot be a handful of readable query parameters."""
        config = AnalysisConfig(material=_CUSTOM)
        restored = decode_config(encode_config(config)).config
        assert restored.material is not None
        assert restored.material.to_dict() == _CUSTOM.to_dict()

    def test_a_clean_link_carries_no_warnings(self):
        assert decode_config(encode_config(AnalysisConfig())).warnings == ()

    def test_encoding_is_deterministic(self):
        """Two shares of the same case produce the same link, so a
        bookmark and a pasted link compare equal."""
        config = _everything_non_default()
        assert encode_config(config) == encode_config(config)

    def test_payload_needs_no_url_escaping(self):
        """URL-safe base64 with the padding stripped: no '+', '/' or '=' to
        be mangled by a chat client or a mail gateway."""
        payload = encode_config(_everything_non_default())
        assert set(payload) <= set(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz0123456789-_"
        )


# ----------------------------------------------------------------------
# AC4 — length
# ----------------------------------------------------------------------

class TestLength:
    def test_a_full_config_url_is_within_practical_limits(self):
        url = permalink_url(
            _everything_non_default(), "https://wrinklefe.streamlit.app"
        )
        assert len(url) < URL_LIMIT, len(url)

    def test_a_typical_case_is_short_enough_to_paste(self):
        """The delta is what earns this: a two-parameter case should not
        carry 69 fields' worth of characters."""
        url = permalink_url(
            AnalysisConfig(amplitude=0.5, wavelength=20.0),
            "https://wrinklefe.streamlit.app",
        )
        assert len(url) < 300, len(url)

    def test_the_delta_is_smaller_than_the_whole_config(self):
        full = len(
            base64.urlsafe_b64encode(
                zlib.compress(
                    json.dumps(
                        AnalysisConfig().to_dict(), separators=(",", ":")
                    ).encode(),
                    9,
                )
            )
        )
        assert len(encode_config(AnalysisConfig())) < full

    def test_the_url_drops_an_existing_query_and_fragment(self):
        """A permalink replaces the configuration rather than merging into
        whatever happened to be in the address bar."""
        url = permalink_url(
            AnalysisConfig(), "https://example.test/app/?cfg=stale#section"
        )
        assert url.startswith("https://example.test/app/?cfg=")
        assert "stale" not in url and "#" not in url
        assert url.count("?") == 1


# ----------------------------------------------------------------------
# AC2 — hostile input
# ----------------------------------------------------------------------

class TestHostileInput:
    @pytest.mark.parametrize(
        "payload,match",
        [
            pytest.param("", "empty", id="empty"),
            pytest.param("   ", "empty", id="whitespace"),
            pytest.param("!!!not base64!!!", "base64", id="not-base64"),
            pytest.param("aGVsbG8gd29ybGQ", "zlib", id="base64-but-not-zlib"),
            pytest.param("A" * (MAX_PAYLOAD_CHARS + 1), "limit", id="too-long"),
        ],
    )
    def test_malformed_payloads_raise_permalink_error(self, payload, match):
        with pytest.raises(PermalinkError, match=match):
            decode_config(payload)

    def test_a_non_string_payload_is_refused(self):
        with pytest.raises(PermalinkError):
            decode_config(None)  # type: ignore[arg-type]

    def test_a_zlib_bomb_is_refused_before_it_inflates(self):
        """A few KB of base64 that expands to megabytes. Caught by the
        inflation cap specifically -- the payload is inside the character
        limit, so this does not pass by accident."""
        packed = zlib.compress(b"\0" * (4 * 1024 * 1024), 9)
        payload = base64.urlsafe_b64encode(packed).decode().rstrip("=")
        assert len(payload) <= MAX_PAYLOAD_CHARS, "no longer tests the cap"
        with pytest.raises(PermalinkError, match="inflates past"):
            decode_config(payload)

    def test_json_that_is_not_an_object_is_refused(self):
        with pytest.raises(PermalinkError, match="not an object"):
            decode_config(_pack([1, 2, 3]))

    def test_a_newer_envelope_version_is_refused(self):
        with pytest.raises(PermalinkError, match="not supported"):
            decode_config(_pack({"v": PERMALINK_VERSION + 1, "cfg": {}}))

    def test_a_missing_cfg_object_is_refused(self):
        with pytest.raises(PermalinkError, match="no 'cfg' object"):
            decode_config(_pack({"v": PERMALINK_VERSION, "d": "x"}))

    def test_an_unknown_config_key_is_refused(self):
        """The payload goes through ``AnalysisConfig.from_dict``, so an
        injected key is rejected by the same validation an in-code config
        gets rather than being silently set as an attribute."""
        with pytest.raises(PermalinkError, match="valid config"):
            decode_config(_pack({
                "v": PERMALINK_VERSION, "d": "x",
                "cfg": {"config_version": CONFIG_VERSION, "rm_rf": "/"},
            }))

    def test_an_out_of_range_value_is_refused(self):
        with pytest.raises(PermalinkError, match="valid config"):
            decode_config(_pack({
                "v": PERMALINK_VERSION, "d": "x",
                "cfg": {"config_version": CONFIG_VERSION, "amplitude": -5.0},
            }))

    def test_a_wrong_config_version_is_refused(self):
        with pytest.raises(PermalinkError, match="valid config"):
            decode_config(_pack({
                "v": PERMALINK_VERSION, "d": "x",
                "cfg": {"config_version": CONFIG_VERSION + 99},
            }))

    def test_a_truncated_real_payload_is_refused(self):
        """What a link broken by a line-wrapping mail client looks like."""
        payload = encode_config(_everything_non_default())
        with pytest.raises(PermalinkError):
            decode_config(payload[: len(payload) // 2])


# ----------------------------------------------------------------------
# The delta's one liability, surfaced rather than hidden
# ----------------------------------------------------------------------

class TestDefaultsDrift:
    def test_a_link_from_different_defaults_still_decodes_but_warns(self):
        decoded = decode_config(_pack({
            "v": PERMALINK_VERSION, "d": "deadbeef",
            "cfg": {"config_version": CONFIG_VERSION, "amplitude": 0.5},
        }))
        # The fields it named are honoured exactly ...
        assert decoded.config.amplitude == 0.5
        # ... and the ones it did not are flagged, not silently defaulted.
        assert len(decoded.warnings) == 1
        assert "default" in decoded.warnings[0].lower()

    def test_the_digest_is_stable_across_calls(self):
        """Two encodes in one process must agree, or every link would warn."""
        first = decode_config(encode_config(AnalysisConfig()))
        second = decode_config(encode_config(AnalysisConfig()))
        assert first.warnings == second.warnings == ()


class TestQueryParam:
    def test_the_param_name_is_what_the_url_uses(self):
        url = permalink_url(AnalysisConfig(), "https://example.test")
        assert f"?{QUERY_PARAM}=" in url
