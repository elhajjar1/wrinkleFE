"""Tests for the acknowledgment-gate / usage-logging helpers.

``usage_tracking.render_gate()`` runs at the top of ``app.py`` and halts the
page with ``st.stop()`` until a visitor acknowledges. These tests guard the
contract that lets the rest of the suite (and Streamlit's ``AppTest``) drive
the app at all:

1. The gate honours the ``WRINKLEFE_DISABLE_GATE`` off-switch — set globally
   by ``tests/conftest.py`` — so it can never halt the test harness. This is
   the exact regression that earlier runtime-context guards failed to prevent
   (``AppTest`` starts a Runtime, so it looks like a served app).
2. ``log_event`` is fail-soft: with no logging dependency / secret configured
   it is a silent no-op and never raises into a running analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ``usage_tracking.py`` lives at the repo root, not under ``src/``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pytest.importorskip("streamlit", reason="Streamlit not installed.")

import usage_tracking  # noqa: E402  - test-time import after path/skip setup.

pytestmark = pytest.mark.viz


def test_gate_disabled_via_env_var(monkeypatch):
    """With the off-switch set, the gate is disabled and render_gate no-ops."""
    monkeypatch.setenv("WRINKLEFE_DISABLE_GATE", "1")
    assert usage_tracking._gate_disabled() is True
    # Would raise streamlit's StopException if the gate fired here.
    usage_tracking.render_gate()


def test_gate_off_switch_is_falsey_aware(monkeypatch):
    """A blank/0 value must NOT count as 'disabled' (avoid the 0-means-on trap)."""
    monkeypatch.setenv("WRINKLEFE_DISABLE_GATE", "0")
    assert usage_tracking._gate_disabled() is False
    monkeypatch.delenv("WRINKLEFE_DISABLE_GATE", raising=False)
    assert usage_tracking._gate_disabled() is False


def test_render_gate_noop_for_bare_import(monkeypatch):
    """Even without the off-switch, the gate must no-op outside a served app."""
    monkeypatch.delenv("WRINKLEFE_DISABLE_GATE", raising=False)
    # No Streamlit server Runtime in the bare pytest thread.
    assert usage_tracking._running_in_served_app() is False
    usage_tracking.render_gate()


def test_log_event_noop_without_configuration():
    """log_event swallows the missing dependency/secret case without raising."""
    usage_tracking.log_event("run", email="", props={"morphology": "stack"})
