"""Tests for the acknowledgment-gate / usage-logging helpers.

``usage_tracking.render_gate()`` runs at the top of ``app.py`` and halts the
page with ``st.stop()`` until a visitor acknowledges. These tests guard the
contract that lets the rest of the suite (and Streamlit's ``AppTest``) drive
the app at all:

1. The gate is *inactive* outside a real ``streamlit run`` server — both for a
   bare ``import app`` and under ``AppTest`` — so it can never halt the test
   harness. This is the exact regression that an earlier guard (keyed on the
   script-run context, which ``AppTest`` sets) caused.
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


def test_gate_inactive_outside_served_app():
    """No server Runtime exists under pytest/AppTest, so the gate must no-op."""
    assert usage_tracking._running_in_served_app() is False


def test_render_gate_is_noop_under_test():
    """render_gate() must return (not call st.stop()) when not served."""
    # Would raise streamlit's StopException if the gate fired here.
    usage_tracking.render_gate()


def test_log_event_noop_without_configuration():
    """log_event swallows the missing dependency/secret case without raising."""
    usage_tracking.log_event("run", email="", props={"morphology": "stack"})
