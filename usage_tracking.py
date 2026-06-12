"""Soft-gate acknowledgment + optional usage logging for the WrinkleFE app.

The hosted Streamlit app is public and runs on an *ephemeral* filesystem
(see ``DEPLOYMENT_STREAMLIT.md``), so this module does two things:

1. :func:`render_gate` shows a one-time acknowledgment screen (cite + star
   the repo) before the rest of the app renders, capturing an optional
   email so visitors can subscribe to release notes.
2. :func:`log_event` appends ``signup`` / ``run`` rows to a Google Sheet so
   the maintainer can see who is using the tool and what they analyse.

Both are deliberately **fail-soft**: if ``gspread`` / ``google-auth`` are
not installed, or no ``gcp_service_account`` secret is configured (e.g. a
local ``streamlit run`` with no secrets), the gate still works and logging
is silently skipped. The app never crashes because tracking is unavailable.

Configure on Streamlit Community Cloud via *Manage app -> Settings ->
Secrets* (see ``.streamlit/secrets.toml.example`` for the full template)::

    [gcp_service_account]
    type = "service_account"
    project_id = "..."
    private_key = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"
    client_email = "wrinklefe-logger@...iam.gserviceaccount.com"
    # ... rest of the downloaded service-account JSON ...

    [usage_log]
    sheet_key = "<google-sheet-id-from-its-url>"   # or: sheet_url = "https://..."

Share the target Google Sheet with the service account's ``client_email``
(give it *Editor*) so it can append rows.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import streamlit as st

# Read-write scopes for Sheets + Drive (Drive is needed to open by URL/key).
_SCOPES = (
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
)

REPO_URL = "https://github.com/elhajjar1/wrinklefe"


_DISABLE_ENV = "WRINKLEFE_DISABLE_GATE"


def _gate_disabled() -> bool:
    """True when the gate is explicitly switched off via env var.

    Set ``WRINKLEFE_DISABLE_GATE=1`` to suppress the gate entirely — used by
    the test suite (``tests/conftest.py``) so ``AppTest`` can drive the app,
    and available to operators self-hosting the app who don't want the gate.
    """
    return os.environ.get(_DISABLE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _running_in_served_app() -> bool:
    """True when a Streamlit script-run context is active.

    ``False`` for a bare ``import app`` (pytest importing the module for its
    constants), so :func:`render_gate` can't call ``st.stop()`` at import
    time. NOTE: this is ``True`` under ``streamlit.testing`` ``AppTest`` too
    (it starts a Runtime), so it does not gate out the test harness on its
    own — :func:`_gate_disabled` (set by ``tests/conftest.py``) does that.
    """
    try:
        from streamlit.runtime import exists

        return bool(exists())
    except Exception:
        return False


def _session_id() -> str:
    """Return a stable, anonymous per-session id (not tied to any identity)."""
    sid = st.session_state.get("_wf_session_id")
    if not sid:
        sid = uuid.uuid4().hex[:12]
        st.session_state["_wf_session_id"] = sid
    return str(sid)


@st.cache_resource(show_spinner=False)
def _worksheet() -> Any:
    """Authorize against Google Sheets and return the target worksheet.

    Cached for the lifetime of the server process so we authorise once, not
    on every rerun. Returns ``None`` (and never raises) when the optional
    dependencies or secrets are missing, which is the normal case for a
    local run without configured secrets.
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        return None

    try:
        sa_info = dict(st.secrets["gcp_service_account"])
        log_cfg = dict(st.secrets.get("usage_log", {}))
    except Exception:
        # No secrets file / section configured.
        return None

    sheet_key = log_cfg.get("sheet_key")
    sheet_url = log_cfg.get("sheet_url")
    if not (sheet_key or sheet_url):
        return None

    try:
        creds = Credentials.from_service_account_info(sa_info, scopes=list(_SCOPES))
        client = gspread.authorize(creds)
        book = client.open_by_key(sheet_key) if sheet_key else client.open_by_url(sheet_url)
        worksheet_name = log_cfg.get("worksheet")
        ws = book.worksheet(worksheet_name) if worksheet_name else book.sheet1
    except Exception:
        # Bad credentials, sheet not shared with the service account, network
        # hiccup, etc. Logging is best-effort; don't take the app down for it.
        return None

    # Best-effort header on a fresh sheet. Failure here is non-fatal.
    try:
        if not ws.row_values(1):
            ws.append_row(
                ["timestamp_utc", "event", "email", "session_id", "props"],
                value_input_option="RAW",
            )
    except Exception:
        pass
    return ws


def log_event(event: str, *, email: str = "", props: dict[str, Any] | None = None) -> None:
    """Append one row to the usage sheet. Silent no-op if logging is unconfigured.

    Parameters
    ----------
    event:
        Short event kind, e.g. ``"signup"`` or ``"run"``.
    email:
        Optional email captured at the gate (blank for anonymous events).
    props:
        JSON-serialisable extra context (material, layup size, morphology…).
    """
    ws = _worksheet()
    if ws is None:
        return
    try:
        ws.append_row(
            [
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                event,
                email or "",
                _session_id(),
                json.dumps(props or {}, default=str),
            ],
            value_input_option="RAW",
        )
    except Exception:
        # Never let a logging failure surface to the user mid-analysis.
        pass


def render_gate() -> None:
    """Render the one-time acknowledgment gate; halt the app until accepted.

    Once the visitor agrees, an ``_wf_acknowledged`` flag is set in
    ``session_state`` so the gate is skipped for the rest of the session.
    Calls :func:`streamlit.stop` while the gate is showing so nothing below
    it in ``app.py`` renders. No-op when disabled via ``WRINKLEFE_DISABLE_GATE``
    (how the test suite drives the app) or for a bare ``import app`` outside a
    Streamlit run, so it never halts test collection or the testing API.
    """
    if _gate_disabled():
        return
    if not _running_in_served_app():
        return
    if st.session_state.get("_wf_acknowledged"):
        return

    st.title("WrinkleFE")
    st.subheader("Free academic software — a quick acknowledgment before you start")
    st.markdown(
        f"""
**WrinkleFE** predicts strength and stiffness knockdown in composite
laminates containing fiber-waviness defects. It is built and maintained by
**Rani Elhajjar (University of Wisconsin–Milwaukee)** and released free
under the MIT license.

If WrinkleFE supports your published, academic, or commercial work, please:

- **Cite it** — use the *Cite this repository* button on
  [GitHub]({REPO_URL}) (driven by `CITATION.cff`).
- **⭐ Star the repository** so other engineers can find it.
"""
    )

    email = st.text_input(
        "Email (optional)",
        placeholder="you@university.edu",
        help="Only used to send major release notes. Leave blank to skip.",
    )
    agree = st.checkbox(
        "I'll acknowledge / cite WrinkleFE in any work that uses it."
    )
    st.caption(
        "We keep an anonymous usage log (and your email, if you provide one) "
        "to understand how the tool is used and improve it. Nothing else is "
        "collected, and you can use the tool without entering an email."
    )

    if st.button("Enter the app →", type="primary", disabled=not agree):
        st.session_state["_wf_acknowledged"] = True
        log_event("signup", email=email.strip())
        st.rerun()

    # Block the rest of app.py until the visitor acknowledges.
    st.stop()
