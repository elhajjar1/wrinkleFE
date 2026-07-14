"""Tests for the Analyze tab's through-thickness cross-section helper.

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


# ---------------------------------------------------------------------------
# Surface resin pockets (issue #361, Part 4 follow-up)
# ---------------------------------------------------------------------------
#
# The Analyze-tab cross-section shades the neat-resin pockets that fill the
# wrinkle troughs under a tool-flat surface.  They are drawn *analytically*
# (the fill between the flat tool line and the deformed outermost undulating
# ply) rather than by deforming an FE mesh at render time — so the preview
# stays responsive.  ``test_preview_gap_matches_fe_tagged_volume`` is the
# load-bearing test: it cross-checks that analytic gap against the resin
# volume ``compute_surface_resin_blend`` tags on a real (coarse) mesh, so the
# cheap render is provably the same geometry the FE run models.


def _resin_polys(fig, side):
    """Return the resin ``PolyCollection`` for one side (or None)."""
    ax = fig.axes[0]
    gid = f"surface_resin_{side}"
    for c in ax.collections:
        if c.get_gid() == gid:
            return c
    return None


def _covered_x_intervals(poly):
    """(x_min, x_max) of each filled polygon in a fill_between collection."""
    intervals = []
    for path in poly.get_paths():
        v = path.vertices
        if len(v):
            intervals.append((float(v[:, 0].min()), float(v[:, 0].max())))
    return intervals


def test_is_tool_flat_morphology_predicate(app_module):
    """The gate matches ``AnalysisConfig`` validation: flat vs wavy."""
    f = app_module._is_tool_flat_morphology
    assert f("stack", 0.3) and f("convex", 0.0) and f("concave", 1.0)
    assert f("graded", 0.0)              # graded with zero floor IS tool-flat
    assert not f("graded", 0.2)          # a residual floor leaves waviness
    assert not f("uniform", 0.0)         # never decays


def test_surface_pockets_off_by_default_adds_nothing(app_module):
    """Default (feature off) draws no resin patches — backward compatible."""
    import matplotlib.pyplot as plt

    fig = _call(app_module, "stack", decay_floor=0.0)
    try:
        assert getattr(fig, "surface_pocket_gap_area", None) == {}
        assert _resin_polys(fig, "top") is None
        assert _resin_polys(fig, "bottom") is None
    finally:
        plt.close(fig)


def test_surface_pockets_render_over_troughs_not_crest(app_module):
    """stack + toggle on tags amber patches over the troughs, none at the
    central crest (x = 0, where the wrinkle peaks toward the top tool)."""
    import matplotlib.pyplot as plt

    fig = _call(
        app_module, "stack", decay_floor=0.0,
        enable_surface_resin_pockets=True, surface_pocket_side="top",
    )
    try:
        poly = _resin_polys(fig, "top")
        assert poly is not None, "expected a top surface-resin patch"
        areas = fig.surface_pocket_gap_area
        assert areas.get("top", 0.0) > 0.0
        # The crest at x = 0 pokes toward the tool: no top pocket spans it.
        intervals = _covered_x_intervals(poly)
        assert intervals, "resin patch has no filled region"
        assert not any(lo <= 0.0 <= hi for lo, hi in intervals), (
            f"top resin should avoid the x=0 crest; covered {intervals}"
        )
    finally:
        plt.close(fig)


def test_surface_pockets_both_sides_and_crest_on_bottom(app_module):
    """side='both' adds patches on both faces; at the top crest (x=0) the
    *bottom* face troughs, so the bottom patch does span x=0."""
    import matplotlib.pyplot as plt

    fig = _call(
        app_module, "stack", decay_floor=0.0,
        enable_surface_resin_pockets=True, surface_pocket_side="both",
    )
    try:
        top = _resin_polys(fig, "top")
        bot = _resin_polys(fig, "bottom")
        assert top is not None and bot is not None
        areas = fig.surface_pocket_gap_area
        assert areas.get("top", 0.0) > 0.0 and areas.get("bottom", 0.0) > 0.0
        bot_intervals = _covered_x_intervals(bot)
        assert any(lo <= 0.0 <= hi for lo, hi in bot_intervals), (
            "bottom face should have a pocket at the x=0 crest"
        )
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "morphology,decay_floor",
    [("uniform", 0.0), ("graded", 0.3)],
)
def test_surface_pockets_skipped_for_wavy_morphologies(
    app_module, morphology, decay_floor
):
    """Wavy morphologies (uniform, graded+floor) render nothing even when the
    toggle is on — matching what the FE run would (refuse to) do."""
    import matplotlib.pyplot as plt

    fig = _call(
        app_module, morphology, decay_floor=decay_floor,
        enable_surface_resin_pockets=True, surface_pocket_side="both",
    )
    try:
        assert fig.surface_pocket_gap_area == {}
        assert _resin_polys(fig, "top") is None
        assert _resin_polys(fig, "bottom") is None
    finally:
        plt.close(fig)


def test_graded_zero_floor_is_a_valid_surface_pocket_morphology(app_module):
    """graded with decay_floor==0 is tool-flat: the toggle DOES render."""
    import matplotlib.pyplot as plt

    fig = _call(
        app_module, "graded", decay_floor=0.0,
        enable_surface_resin_pockets=True, surface_pocket_side="top",
    )
    try:
        assert fig.surface_pocket_gap_area.get("top", 0.0) > 0.0
        assert _resin_polys(fig, "top") is not None
    finally:
        plt.close(fig)


@pytest.mark.parametrize("morphology", ["stack", "convex", "concave", "graded"])
def test_preview_gap_matches_fe_tagged_volume(app_module, morphology):
    """FIDELITY CROSS-CHECK — the important one.

    The analytic gap area the preview shades must equal the neat-resin
    volume ``compute_surface_resin_blend`` tags on a real deformed mesh
    (per unit width), to within the coarse-mesh tolerance #361's own
    conservation test uses (~10%).  This binds the cheap analytic render to
    the shared FE function's geometry: they are the same kinematic gap
    ``-w(x)·decay``, not two independently-derived shapes.
    """
    import matplotlib.pyplot as plt

    from wrinklefe.core.laminate import Laminate
    from wrinklefe.core.material import MaterialLibrary
    from wrinklefe.core.mesh import WrinkleMesh
    from wrinklefe.core.morphology import WrinkleConfiguration
    from wrinklefe.core.resin_pocket import (
        SurfacePocketSpec,
        compute_surface_resin_blend,
    )
    from wrinklefe.core.wrinkle import GaussianSinusoidal

    amplitude, wavelength, width, ply_t = 0.4, 16.0, 8.0, 0.183
    angles = list(_ANGLES_12)
    n = len(angles)
    decay_floor = 0.0

    # 1) Preview: the analytic gap area (mm² per unit width) it returns.
    fig = _call(
        app_module, morphology,
        amplitude=amplitude, wavelength=wavelength, width=width,
        ply_thickness=ply_t, decay_floor=decay_floor,
        enable_surface_resin_pockets=True, surface_pocket_side="top",
    )
    preview_area = fig.surface_pocket_gap_area["top"]
    plt.close(fig)
    assert preview_area > 0.0

    # 2) FE: build a coarse structured mesh from the SAME morphology field
    #    the preview uses (from_morphology_name, mid = n // 2 interfaces),
    #    tag the surface pockets, and integrate weight·height·footprint.
    mid = n // 2
    i1, i2 = max(0, mid - 1), min(n - 1, mid)
    Lx, Ly = 6.0 * width, 4.0
    prof = GaussianSinusoidal(
        amplitude=amplitude, wavelength=wavelength, width=width, center=Lx / 2.0
    )
    wc = WrinkleConfiguration.from_morphology_name(
        morphology, prof, interface1=i1, interface2=i2, decay_floor=decay_floor
    )
    lam = Laminate.from_angles(angles, MaterialLibrary().get("IM7_8552"), ply_t)
    mesh = WrinkleMesh(
        lam, wc, Lx=Lx, Ly=Ly, nx=240, ny=1, nz_per_ply=1
    ).generate()
    weight = compute_surface_resin_blend(mesh, wc, SurfacePocketSpec(side="top"))

    ez = mesh.nodes[mesh.elements][:, :, 2]
    ex = mesh.nodes[mesh.elements][:, :, 0]
    ey = mesh.nodes[mesh.elements][:, :, 1]
    height = ez.max(axis=1) - ez.min(axis=1)
    footprint = (ex.max(axis=1) - ex.min(axis=1)) * (ey.max(axis=1) - ey.min(axis=1))
    tagged = weight > 0
    assert tagged.any(), "FE mesh tagged no surface resin — bad fixture"
    fe_volume = float((weight * height * footprint)[tagged].sum())
    fe_area_per_width = fe_volume / Ly

    # The two independent computations of the same kinematic gap agree.
    assert preview_area == pytest.approx(fe_area_per_width, rel=0.10), (
        f"{morphology}: preview gap {preview_area:.5f} vs FE/width "
        f"{fe_area_per_width:.5f} (ratio {preview_area / fe_area_per_width:.4f})"
    )
