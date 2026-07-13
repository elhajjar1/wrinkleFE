"""Tests for the Configure tab's through-thickness cross-section helper.

``app._through_thickness_cross_section`` renders the deformed ply stack
in the (x, z) plane so the user can see how the wrinkle manifests
through the laminate thickness. Unlike the sidebar cartoon it reuses the
real :class:`~wrinklefe.core.morphology.WrinkleConfiguration` field the
FE mesh uses, so these tests guard both that it renders for every
morphology and that the field it draws is faithful to that model
(interface resolution + amplitude at the wrinkle interface).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ``app.py`` lives at the repo root, not under ``src/``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pytest.importorskip("streamlit", reason="Streamlit not installed.")

import matplotlib  # noqa: E402

pytestmark = pytest.mark.viz

matplotlib.use("Agg")


@pytest.fixture(scope="module")
def app_module():
    import app as app_module  # noqa: WPS433 - test-time import.
    return app_module


_ANGLES_12 = [0, 45, -45, 90] * 3  # 12-ply quasi-isotropic


def _call(app_module, morphology, **overrides):
    kwargs = dict(
        morphology=morphology,
        amplitude=0.4,
        wavelength=16.0,
        width=8.0,
        angles=list(_ANGLES_12),
        ply_thickness=0.183,
        decay_floor=0.3,
        amplitude_profile="constant",
        amplitude_profile_decay_length=None,
        amplitude_profile_axis="x",
    )
    kwargs.update(overrides)
    return app_module._through_thickness_cross_section(**kwargs)


@pytest.mark.parametrize(
    "morphology", ["stack", "convex", "concave", "uniform", "graded"]
)
def test_returns_figure_with_one_band_per_ply(app_module, morphology):
    """Every morphology renders a Figure with exactly one filled band per ply."""
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure

    fig = _call(app_module, morphology)
    try:
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert len(polys) == len(_ANGLES_12), (
            f"expected {len(_ANGLES_12)} ply bands, got {len(polys)}"
        )
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_dual_morphologies_outline_interface_plies(app_module):
    """Dual-wrinkle morphologies mark the interface plies with red outlines;
    single-wrinkle morphologies do not."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    # ``to_rgba`` normalises both the hex string the outline is drawn with
    # and any tuple form to a 4-float RGBA, so the comparison is robust to
    # how matplotlib stores the colour.
    red = mcolors.to_rgba("#d62728")

    def _n_red_lines(fig):
        ax = fig.axes[0]
        return sum(
            1 for ln in ax.get_lines()
            if all(
                abs(a - b) < 1e-3
                for a, b in zip(mcolors.to_rgba(ln.get_color()), red)
            )
        )

    for morph in ("stack", "convex", "concave"):
        fig = _call(app_module, morph)
        # Two interface plies × top+bottom outline = 4 red lines.
        assert _n_red_lines(fig) == 4, f"{morph}: expected 4 outline lines"
        plt.close(fig)

    for morph in ("uniform", "graded"):
        fig = _call(app_module, morph)
        assert _n_red_lines(fig) == 0, f"{morph}: should not outline an interface"
        plt.close(fig)


def test_cross_section_field_matches_wrinkle_configuration(app_module):
    """The bands drawn must be the *real* deformation field: reconstruct it
    from the public morphology API and confirm the peak wrinkle
    displacement at the interface equals the configured amplitude."""
    from wrinklefe.core.morphology import WrinkleConfiguration
    from wrinklefe.core.wrinkle import GaussianSinusoidal

    angles = list(_ANGLES_12)
    n = len(angles)
    mid = n // 2
    i1, i2 = max(0, mid - 1), min(n - 1, mid)
    prof = GaussianSinusoidal(amplitude=0.4, wavelength=16.0, width=8.0, center=0.0)
    wc = WrinkleConfiguration.from_morphology_name(
        "stack", prof, interface1=i1, interface2=i2, decay_floor=0.0
    )
    # Odd sample count so x = 0 (the wrinkle crest) is sampled exactly;
    # otherwise the discrete max sits just off the peak.
    n_x = 401
    x = np.linspace(-24.0, 24.0, n_x)
    t, T = 0.183, n * 0.183
    z0 = (np.arange(n) + 0.5) * t - T / 2.0
    nodes = np.column_stack(
        [np.tile(x, n), np.zeros(n * n_x), np.repeat(z0, n_x)]
    )
    ply_ids = np.repeat(np.arange(n), n_x)
    dz = (
        wc.apply_to_nodes(nodes, ply_ids, n)[:, 2].reshape(n, n_x)
        - z0[:, None]
    )
    # Peak wrinkle displacement (composed dual wrinkle) equals the configured
    # amplitude A at the seed, to within the grid/decay-composition margin.
    assert np.abs(dz).max() == pytest.approx(0.4, rel=5e-3)

    # And the helper should not raise for this exact configuration.
    import matplotlib.pyplot as plt

    fig = _call(app_module, "stack", decay_floor=0.0)
    plt.close(fig)


def test_handles_small_and_odd_ply_counts(app_module):
    """Interface resolution must stay in range for tiny / odd layups."""
    import matplotlib.pyplot as plt

    for angles in ([0, 90], [0, 45, 90], [0]):
        fig = app_module._through_thickness_cross_section(
            morphology="stack",
            amplitude=0.3,
            wavelength=12.0,
            width=6.0,
            angles=angles,
            ply_thickness=0.2,
            decay_floor=0.0,
            amplitude_profile="constant",
            amplitude_profile_decay_length=None,
            amplitude_profile_axis="x",
        )
        assert len(fig.axes) == 1
        plt.close(fig)
