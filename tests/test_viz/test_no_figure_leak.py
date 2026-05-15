"""Regression tests for the matplotlib figure leak (issues #38 / #30).

``ensure_axes`` creates a new ``Figure`` when ``ax=None`` but the ``plot_*``
functions only return the ``Axes``. In batch / sweep / headless loops nothing
closed those figures, so they accumulated until matplotlib emitted the
``More than 20 figures have been opened`` warning (a real memory leak).

The fix adds opt-in :func:`wrinklefe.viz.save_figure` /
:func:`wrinklefe.viz.figure_context` helpers that close internally-created
figures. These tests assert that the number of open matplotlib figures stays
bounded across many iterations when those helpers are used, while the
interactive contract (the returned ``Axes`` keeps its live figure) is
preserved when they are not.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless; no display required

import matplotlib.pyplot as plt
import pytest

from wrinklefe.core.wrinkle import GaussianSinusoidal
from wrinklefe.viz import figure_context, plot_wrinkle_profile, save_figure

N_ITERATIONS = 30


@pytest.fixture
def profile() -> GaussianSinusoidal:
    return GaussianSinusoidal(amplitude=0.5, wavelength=10.0, width=5.0)


@pytest.fixture(autouse=True)
def _clean_figures():
    """Start and end each test with no open figures."""
    plt.close("all")
    yield
    plt.close("all")


def test_save_figure_does_not_leak_in_loop(profile, tmp_path):
    """The save-to-disk path must close figures: fignums stay bounded at ~0."""
    assert len(plt.get_fignums()) == 0

    for i in range(N_ITERATIONS):
        ax = plot_wrinkle_profile(profile)
        save_figure(ax, tmp_path / f"profile_{i}.png")
        # After each save+close exactly zero figures should remain open.
        assert len(plt.get_fignums()) == 0, (
            f"figure leaked at iteration {i}: "
            f"{len(plt.get_fignums())} open figures"
        )

    assert len(plt.get_fignums()) == 0
    # Sanity: files were actually written.
    assert len(list(tmp_path.glob("profile_*.png"))) == N_ITERATIONS


def test_figure_context_does_not_leak_in_loop(profile, tmp_path):
    """figure_context closes the internally-created figure on block exit."""
    assert len(plt.get_fignums()) == 0

    for i in range(N_ITERATIONS):
        with figure_context(plot_wrinkle_profile(profile)) as fig:
            fig.savefig(tmp_path / f"ctx_{i}.png")
        assert len(plt.get_fignums()) == 0, (
            f"figure leaked at iteration {i}"
        )

    assert len(plt.get_fignums()) == 0


@pytest.mark.filterwarnings("ignore:More than 20 figures:RuntimeWarning")
def test_plain_loop_without_helper_does_leak(profile):
    """Documents the original bug: without a helper, figures accumulate.

    This guards against a regression where ``ensure_axes`` stopped creating
    a figure at all (which would silently break the interactive contract).
    """
    for _ in range(N_ITERATIONS):
        plot_wrinkle_profile(profile)
    # Without an explicit close the leak is real and unbounded.
    assert len(plt.get_fignums()) == N_ITERATIONS


def test_save_figure_keeps_figure_when_close_false(profile, tmp_path):
    """Interactive contract: close=False leaves the live figure open."""
    ax = plot_wrinkle_profile(profile)
    fig = ax.figure
    save_figure(ax, tmp_path / "keep.png", close=False)
    assert fig.number in plt.get_fignums()
    plt.close(fig)


def test_returned_axes_still_has_live_figure(profile):
    """Non-breaking: plot_* still returns an Axes whose Figure is alive.

    Streamlit / interactive callers rely on ``ax.figure`` being a usable,
    not-yet-closed figure (e.g. ``st.pyplot(ax.figure)``).
    """
    ax = plot_wrinkle_profile(profile)
    assert ax.figure is not None
    assert ax.figure.number in plt.get_fignums()
    plt.close(ax.figure)


def test_save_figure_accepts_figure_directly(profile, tmp_path):
    """save_figure works whether handed an Axes or a Figure."""
    ax = plot_wrinkle_profile(profile)
    save_figure(ax.figure, tmp_path / "fig.png")
    assert len(plt.get_fignums()) == 0
    assert (tmp_path / "fig.png").exists()
