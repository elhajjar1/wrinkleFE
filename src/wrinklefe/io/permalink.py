"""Encode an :class:`~wrinklefe.analysis.AnalysisConfig` into a URL, and back.

An analysis set up in the hosted app used to be trapped in the browser tab
that built it (issue #281). The collaboration moves the tool's use case
implies had no support: a stress engineer could not send a colleague the
exact configured defect without dictating a dozen parameter values or
screenshotting the sidebar, a recurring team case could not be bookmarked,
and a bug report against the app could not carry a reproducing link.

This module is the codec, deliberately kept out of ``app.py``: it is pure,
testable without Streamlit, and reusable by anything else that needs a
config in a URL. The payload is the *same* dict
:meth:`AnalysisConfig.to_dict` writes to a config file, so config files,
permalinks and any future API share one schema rather than drifting apart.

Shape
-----
One query parameter, :data:`QUERY_PARAM`::

    https://wrinklefe.streamlit.app/?cfg=eJyNVFtv2jAU...

whose value is ``urlsafe_b64(zlib(json(envelope)))`` with the base64 padding
stripped. The envelope is::

    {"v": 1, "d": "<8 hex>", "cfg": {<delta against the defaults>}}

Why a delta, and why the digest
-------------------------------
Only the fields that differ from a default :class:`AnalysisConfig` are
carried. Measured on the 68-field config: a two-parameter case encodes to
132 characters as a delta against 979 in full, and a case with every sidebar
section changed plus a custom material to 700 against 1250. Both are inside
practical URL limits; the delta is what keeps the *typical* case short
enough to paste into a chat message without it wrapping into something that
looks broken.

The cost of a delta is that it means "whatever the defaults are **when it is
read**". If a release changes a default, an old link silently acquires the
new value for a field it never named. ``d`` is a short digest of the default
baseline the link was built against, so :func:`decode_config` can detect
exactly that case and say so (:attr:`DecodedPermalink.warnings`) instead of
handing back a config that quietly differs from the one that was shared. The
link still decodes -- refusing it would be worse -- it just stops being
silent about it.

Untrusted input
---------------
A payload arrives from the URL bar, so :func:`decode_config` treats it as
hostile: the encoded length is capped, inflation is bounded through an
incremental decompressor (a few hundred bytes of zlib can otherwise expand
to gigabytes), and every failure -- bad base64, bad zlib, bad JSON, wrong
envelope, unknown config key, invalid value -- surfaces as one
:class:`PermalinkError`. Nothing here constructs a config by any route other
than :meth:`AnalysisConfig.from_dict`, so a permalink gets the same
validation an in-code config does.

Examples
--------
>>> from wrinklefe.analysis import AnalysisConfig
>>> from wrinklefe.io.permalink import decode_config, encode_config
>>> payload = encode_config(AnalysisConfig(amplitude=0.42, wavelength=18.0))
>>> decoded = decode_config(payload)
>>> decoded.config.amplitude, decoded.config.wavelength
(0.42, 18.0)
>>> decoded.warnings
()
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import zlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from wrinklefe.analysis import AnalysisConfig

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_PAYLOAD_CHARS",
    "PERMALINK_VERSION",
    "QUERY_PARAM",
    "DecodedPermalink",
    "PermalinkError",
    "decode_config",
    "encode_config",
    "permalink_url",
]

#: Envelope version. Bump when the envelope's own shape changes; the config
#: dict inside it carries :data:`wrinklefe.analysis.CONFIG_VERSION`
#: separately and is validated by ``AnalysisConfig.from_dict``.
PERMALINK_VERSION = 1

#: The query parameter a permalink travels in.
QUERY_PARAM = "cfg"

#: Longest encoded payload :func:`decode_config` will look at. Even a full,
#: non-delta encoding of a config with a custom material is ~1250
#: characters, so this leaves generous headroom while refusing a megabyte of
#: base64 pasted into the URL bar before any of it is decompressed.
MAX_PAYLOAD_CHARS = 8192

#: Largest inflated JSON document accepted, in bytes. A full config is ~2 KB;
#: the bound exists because zlib is a compression bomb vector, not because a
#: real config could approach it.
_MAX_INFLATED_BYTES = 256 * 1024


class PermalinkError(ValueError):
    """A permalink payload could not be decoded into a config."""


@dataclass(frozen=True)
class DecodedPermalink:
    """What a permalink decoded to, plus anything the reader should know.

    Attributes
    ----------
    config : AnalysisConfig
        The reconstructed configuration, validated exactly as any other
        :class:`AnalysisConfig` construction is.
    warnings : tuple of str
        Human-readable notes that do not make the link unusable -- today,
        only the defaults-drift case described in the module docstring.
        Empty for a link written by this version of WrinkleFE.
    """

    config: AnalysisConfig
    warnings: tuple[str, ...] = ()


@lru_cache(maxsize=1)
def _defaults() -> tuple[dict[str, Any], str]:
    """The resolved default config dict and its digest.

    Cached: building it constructs an ``AnalysisConfig``, and encoding or
    decoding a handful of permalinks per rerun should not pay for that
    repeatedly. The returned dict is never handed out directly -- callers
    copy it -- so the cache cannot be mutated through them.
    """
    baseline = AnalysisConfig().to_dict()
    canonical = json.dumps(baseline, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
    return baseline, digest


def encode_config(config: AnalysisConfig) -> str:
    """Encode ``config`` as a URL-safe permalink payload.

    Parameters
    ----------
    config : AnalysisConfig
        The configuration to share. Its *resolved* values are encoded (the
        ones after ``__post_init__``), so the link is self-contained.

    Returns
    -------
    str
        The value for the :data:`QUERY_PARAM` query parameter: URL-safe
        base64 with padding stripped, so it needs no further escaping.

    See Also
    --------
    decode_config : the inverse.
    permalink_url : wraps this into a complete shareable URL.
    """
    baseline, digest = _defaults()
    full = config.to_dict()
    delta = {
        key: value
        for key, value in full.items()
        if key != "config_version" and baseline.get(key) != value
    }
    envelope = {
        "v": PERMALINK_VERSION,
        "d": digest,
        "cfg": {"config_version": full["config_version"], **delta},
    }
    raw = json.dumps(envelope, sort_keys=True, separators=(",", ":"))
    packed = zlib.compress(raw.encode("utf-8"), 9)
    return base64.urlsafe_b64encode(packed).decode("ascii").rstrip("=")


def permalink_url(config: AnalysisConfig, base_url: str) -> str:
    """Build a complete shareable URL for ``config``.

    Parameters
    ----------
    config : AnalysisConfig
        The configuration to encode.
    base_url : str
        Where the app is served, e.g.
        ``"https://wrinklefe.streamlit.app"``. Any existing query string or
        fragment is dropped -- a permalink replaces the configuration rather
        than merging into whatever was there.

    Returns
    -------
    str
        ``<base>?cfg=<payload>``.
    """
    base = base_url.split("#", 1)[0].split("?", 1)[0].rstrip("/")
    return f"{base}/?{QUERY_PARAM}={encode_config(config)}"


def decode_config(payload: str) -> DecodedPermalink:
    """Decode a permalink payload back into a configuration.

    Parameters
    ----------
    payload : str
        The :data:`QUERY_PARAM` value produced by :func:`encode_config`.

    Returns
    -------
    DecodedPermalink
        The config, plus any non-fatal notes (see :class:`DecodedPermalink`).

    Raises
    ------
    PermalinkError
        For anything that makes the payload unusable: too long, not base64,
        not zlib, inflating past the cap, not JSON, not this envelope, or a
        config dict ``AnalysisConfig.from_dict`` rejects. One exception type
        for every failure, because every caller does the same thing with
        them -- shows the message and falls back to defaults.
    """
    if not isinstance(payload, str) or not payload.strip():
        raise PermalinkError("permalink payload is empty.")
    text = payload.strip()
    if len(text) > MAX_PAYLOAD_CHARS:
        raise PermalinkError(
            f"permalink payload is {len(text)} characters, over the "
            f"{MAX_PAYLOAD_CHARS}-character limit."
        )

    # base64 → zlib → JSON, each step with its own message: "that link is
    # broken" is not worth much to someone who has to decide whether to ask
    # for a new one or report a bug.
    try:
        packed = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (binascii.Error, ValueError) as exc:
        raise PermalinkError(
            f"permalink payload is not valid URL-safe base64: {exc}"
        ) from exc

    try:
        inflater = zlib.decompressobj()
        raw = inflater.decompress(packed, _MAX_INFLATED_BYTES)
        if inflater.unconsumed_tail:
            raise PermalinkError(
                f"permalink payload inflates past the "
                f"{_MAX_INFLATED_BYTES}-byte limit."
            )
    except zlib.error as exc:
        raise PermalinkError(
            f"permalink payload is not valid zlib data: {exc}"
        ) from exc

    try:
        envelope = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PermalinkError(
            f"permalink payload does not contain JSON: {exc}"
        ) from exc

    if not isinstance(envelope, dict):
        raise PermalinkError(
            f"permalink payload is a {type(envelope).__name__}, not an object."
        )

    version = envelope.get("v")
    if version != PERMALINK_VERSION:
        raise PermalinkError(
            f"permalink format {version!r} is not supported by this version "
            f"of WrinkleFE (expected {PERMALINK_VERSION}). The link was "
            f"probably made by a newer release."
        )

    delta = envelope.get("cfg")
    if not isinstance(delta, dict):
        raise PermalinkError(
            "permalink payload has no 'cfg' object to build a config from."
        )

    baseline, digest = _defaults()
    warnings: list[str] = []
    their_digest = envelope.get("d")
    if isinstance(their_digest, str) and their_digest != digest:
        # Only the fields the link names are carried; everything else means
        # "the default". Say so rather than pretending the config is
        # byte-identical to the one that was shared.
        warnings.append(
            "This link was created against a different set of WrinkleFE "
            "defaults, so any input it does not name has been filled in "
            "with this version's default. Check the sidebar before relying "
            "on the numbers."
        )
        logger.info(
            "permalink defaults digest %s != current %s; "
            "unnamed fields take current defaults",
            their_digest, digest,
        )

    merged = {**baseline, **delta}
    try:
        config = AnalysisConfig.from_dict(merged)
    except (ValueError, TypeError, KeyError) as exc:
        raise PermalinkError(f"permalink does not describe a valid config: {exc}") from exc

    return DecodedPermalink(config=config, warnings=tuple(warnings))
