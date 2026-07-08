"""Guards the Streamlit app's top-level tab layout.

The app opens on **Configure** (Streamlit renders the first tab by
default), with **Results** and **Export** next and the intro/**Help**
tab last. These are AppTest smoke checks; they skip when Streamlit is
not installed (they run under CI, where it is).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pytest.importorskip("streamlit", reason="Streamlit not installed.")

import matplotlib  # noqa: E402

matplotlib.use("Agg")


def _app_path() -> str:
    return str(_REPO_ROOT / "app.py")


def test_tab_order_defaults_to_configure():
    """Tabs are Configure, Results, Export, Help in that order — Configure
    first so the app defaults to it."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_app_path(), default_timeout=30)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    labels = [t.label for t in at.tabs]
    assert labels[:4] == ["Configure", "Results", "Export", "Help"], (
        f"unexpected tab order: {labels}"
    )


def test_help_tab_carries_the_intro_and_demo_button():
    """The renamed Help tab still hosts the orientation copy and the
    one-click demo button."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_app_path(), default_timeout=30)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    # The demo button lives on the Help tab and drives an analysis.
    demo = [b for b in at.button if "demo" in (b.label or "").lower()]
    assert demo, "the 'Try a demo analysis' button is missing"

    markdown_blob = " ".join(m.value for m in at.markdown)
    assert "What to do next" in markdown_blob
