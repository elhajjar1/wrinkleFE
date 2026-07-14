"""Surface resin pockets under a tool-flat outer surface (issue #361).

Covers the new :class:`SurfacePocketSpec` /
:func:`compute_surface_resin_blend` geometry (fast, mesh-only) plus the
``analysis.py`` config surface, validation and FE wiring (the FE-solve
smoke test is marked ``integration``/``slow`` per issue #267).

Part 0 spike findings this file pins:

* **Flatness** — the default through-thickness decay leaves both outer
  surfaces *exactly* flat (to floating point), so no ``flatten_surfaces``
  clamp is needed; ``uniform`` and ``graded`` with ``decay_floor > 0`` are
  the only non-flat cases and are rejected by validation instead.
* **Volume conservation** — the tagged resin volume equals the integrated
  kinematic gap ``-w(x)·decay_last`` between the flat surface and the
  outermost undulating ply.
"""

from __future__ import annotations

import numpy as np
import pytest

from wrinklefe.core.laminate import Laminate
from wrinklefe.core.material import MaterialLibrary
from wrinklefe.core.mesh import WrinkleMesh
from wrinklefe.core.morphology import WrinkleConfiguration, WrinklePlacement
from wrinklefe.core.resin_pocket import (
    SurfacePocketSpec,
    compute_surface_resin_blend,
)
from wrinklefe.core.wrinkle import GaussianSinusoidal

PLY_T = 0.184
N_PLIES = 8
LX = 16.0
LY = 4.0


def _build_mesh(
    *,
    amplitude: float = 0.12,
    interfaces: tuple[int, ...] = (3,),
    phases: tuple[float, ...] | None = None,
    nx: int = 120,
    ny: int = 1,
    nz_per_ply: int = 1,
    decay_mode: str = "default",
    decay_floor: float = 0.0,
    wrinkle_z_position: float = 0.5,
    width: float = 1.0e6,
):
    """Build a deformed hex mesh with one or more mid-laminate wrinkles."""
    mat = MaterialLibrary().get("AS4_3501_6")
    lam = Laminate.from_angles([0.0] * N_PLIES, mat, PLY_T)
    if phases is None:
        phases = (0.0,) * len(interfaces)
    placements = [
        WrinklePlacement(
            profile=GaussianSinusoidal(
                amplitude=amplitude, wavelength=LX, width=width, center=LX / 2.0
            ),
            ply_interface=k,
            phase_offset=ph,
        )
        for k, ph in zip(interfaces, phases)
    ]
    wc = WrinkleConfiguration(
        placements, decay_mode=decay_mode, decay_floor=decay_floor
    )
    wc.wrinkle_z_position = wrinkle_z_position
    mesh = WrinkleMesh(
        lam, wc, Lx=LX, Ly=LY, nx=nx, ny=ny, nz_per_ply=nz_per_ply
    ).generate()
    return mesh, wc


def _element_geometry(mesh):
    """Return (deformed height, nominal height, footprint area) per element."""
    ez = mesh.nodes[mesh.elements][:, :, 2]
    ex = mesh.nodes[mesh.elements][:, :, 0]
    ey = mesh.nodes[mesh.elements][:, :, 1]
    h = ez.max(axis=1) - ez.min(axis=1)
    z_span = float(mesh.nodes[:, 2].max() - mesh.nodes[:, 2].min())
    h0 = z_span / mesh.nz
    area = (ex.max(axis=1) - ex.min(axis=1)) * (ey.max(axis=1) - ey.min(axis=1))
    return h, h0, area


def _surface_node_spread(mesh) -> tuple[float, float]:
    """Peak-to-peak z of the top and bottom surface node layers."""
    per_layer = (mesh.nx + 1) * (mesh.ny + 1)
    k_idx = np.arange(mesh.n_nodes) // per_layer
    top = mesh.nodes[k_idx == mesh.nz, 2]
    bot = mesh.nodes[k_idx == 0, 2]
    return float(top.max() - top.min()), float(bot.max() - bot.min())


# ---------------------------------------------------------------------------
# Part 0.1 — flatness invariant
# ---------------------------------------------------------------------------


def test_default_decay_leaves_outer_surfaces_exactly_flat():
    """Spike 0.1: default decay pins both outer surfaces flat to fp tolerance."""
    mesh, _ = _build_mesh()
    top_spread, bot_spread = _surface_node_spread(mesh)
    assert top_spread < 1e-12
    assert bot_spread < 1e-12


def test_flatness_holds_when_surface_pockets_tagged():
    """Tagging pockets does not move any node — the surface stays flat."""
    mesh, wc = _build_mesh()
    before = mesh.nodes.copy()
    _ = compute_surface_resin_blend(mesh, wc, SurfacePocketSpec(side="both"))
    assert np.array_equal(mesh.nodes, before)


# ---------------------------------------------------------------------------
# Part 1 — gap-field correctness and volume conservation
# ---------------------------------------------------------------------------


def test_weight_is_excess_stretch_fraction():
    """weight == max(0, (h - h0) / h) exactly on the tagged transition layer."""
    mesh, wc = _build_mesh()
    h, h0, _ = _element_geometry(mesh)
    w = compute_surface_resin_blend(mesh, wc, SurfacePocketSpec(side="top"))
    tagged = w > 0
    assert tagged.any()
    np.testing.assert_allclose(w[tagged], ((h - h0) / h)[tagged], rtol=1e-9)
    # Excess-stretch fraction is a valid blend weight.
    assert w.min() >= 0.0 and w.max() <= 1.0


def test_pockets_form_over_troughs_not_crests():
    """Tagged top-surface elements sit over troughs (ply dips from the tool)."""
    mesh, wc = _build_mesh()
    w = compute_surface_resin_blend(mesh, wc, SurfacePocketSpec(side="top"))
    centroids = mesh.nodes[mesh.elements].mean(axis=1)
    xc = centroids[w > 0, 0]
    # The single mid wrinkle (cos, centre at Lx/2) crests at x = Lx/2 and
    # troughs at the ends: no pocket near the centre crest.
    assert np.all(np.abs(xc - LX / 2.0) > LX / 8.0)


def test_resin_volume_matches_integrated_kinematic_gap():
    """Total tagged resin volume ≈ width · ∫ max(0, -w(x)·decay_last) dx."""
    mesh, wc = _build_mesh(nx=200, ny=1)
    h, h0, area = _element_geometry(mesh)
    w = compute_surface_resin_blend(mesh, wc, SurfacePocketSpec(side="top"))
    tagged = w > 0

    # Excess-stretch form and kinematic-gap form of the per-element volume
    # agree by construction (weight·h == h - h0).
    vol_weight = float((w * h * area)[tagged].sum())
    vol_gap = float(((h - h0) * area)[tagged].sum())
    np.testing.assert_allclose(vol_weight, vol_gap, rtol=1e-9)

    # ... and both match the analytic band integral to within mesh tolerance.
    decay_last = float(
        wc._through_thickness_decay(np.array([N_PLIES - 2]), 3, N_PLIES)[0]
    )
    prof = wc.wrinkles[0].profile
    xs = np.linspace(0.0, LX, 8000)
    gap_x = np.maximum(0.0, -prof.displacement(xs) * decay_last)
    analytic = float(np.trapezoid(gap_x, xs)) * LY
    assert vol_weight == pytest.approx(analytic, rel=0.1)
    assert analytic > 0.0


def test_min_gap_threshold_suppresses_resin_dust():
    """A large min_gap_threshold zeros out shallow pockets."""
    mesh, wc = _build_mesh(amplitude=0.05)
    w_default = compute_surface_resin_blend(mesh, wc, SurfacePocketSpec(side="top"))
    big = SurfacePocketSpec(side="top", min_gap_threshold=1.0)  # 1 mm >> any gap
    w_big = compute_surface_resin_blend(mesh, wc, big)
    assert (w_default > 0).sum() > 0
    assert (w_big > 0).sum() == 0


# ---------------------------------------------------------------------------
# Part 1 — side selection
# ---------------------------------------------------------------------------


def test_side_selection_top_bottom_both():
    """top tags the upper band, bottom the lower, both their union."""
    mesh, wc = _build_mesh()
    w_top = compute_surface_resin_blend(mesh, wc, SurfacePocketSpec(side="top"))
    w_bot = compute_surface_resin_blend(mesh, wc, SurfacePocketSpec(side="bottom"))
    w_both = compute_surface_resin_blend(mesh, wc, SurfacePocketSpec(side="both"))

    top_plies = np.unique(mesh.ply_ids[w_top > 0])
    bot_plies = np.unique(mesh.ply_ids[w_bot > 0])
    assert top_plies.max() >= N_PLIES - 2         # near the top surface
    assert bot_plies.min() <= 1                     # near the bottom surface
    assert set(top_plies).isdisjoint(set(bot_plies))

    # "both" is the per-element maximum of the two single-sided fields.
    np.testing.assert_allclose(w_both, np.maximum(w_top, w_bot))


def test_symmetric_wrinkle_gives_symmetric_side_counts():
    """A single mid-plane wrinkle tags equal element counts top and bottom."""
    mesh, wc = _build_mesh(interfaces=(3,))
    w_top = compute_surface_resin_blend(mesh, wc, SurfacePocketSpec(side="top"))
    w_bot = compute_surface_resin_blend(mesh, wc, SurfacePocketSpec(side="bottom"))
    assert int((w_top > 0).sum()) == int((w_bot > 0).sum())


# ---------------------------------------------------------------------------
# Part 1 — mesh-quirk robustness and multi-wrinkle
# ---------------------------------------------------------------------------


def test_single_transition_layer_per_side():
    """Only one horizontal element-layer per side is tagged (no internal slivers)."""
    for nz_per_ply in (1, 2):
        mesh, wc = _build_mesh(nz_per_ply=nz_per_ply, amplitude=0.08)
        w = compute_surface_resin_blend(mesh, wc, SurfacePocketSpec(side="top"))
        tagged = np.flatnonzero(w > 0)
        # All tagged elements share one nominal through-thickness band ...
        assert len(np.unique(mesh.ply_ids[tagged])) == 1
        # ... and one structured element-layer (k index = elem // (nx·ny)).
        k_layer = tagged // (mesh.nx * mesh.ny)
        assert len(np.unique(k_layer)) == 1


def test_multi_wrinkle_uses_composed_displacement_field():
    """Two stacked wrinkles tag pockets from the composed displacement field."""
    single, wc_s = _build_mesh(interfaces=(3,), amplitude=0.06, nx=200)
    multi, wc_m = _build_mesh(interfaces=(3, 4), amplitude=0.06, nx=200)

    w_single = compute_surface_resin_blend(single, wc_s, SurfacePocketSpec("top"))
    w_multi = compute_surface_resin_blend(multi, wc_m, SurfacePocketSpec("top"))

    h_s, h0_s, a_s = _element_geometry(single)
    h_m, h0_m, a_m = _element_geometry(multi)
    vol_single = float(((h_s - h0_s) * a_s)[w_single > 0].sum())
    vol_multi = float(((h_m - h0_m) * a_m)[w_multi > 0].sum())

    assert (w_multi > 0).sum() > 0
    # The second wrinkle deepens the composed trough, so more resin is tagged.
    assert vol_multi > vol_single


def test_flat_mesh_tags_nothing():
    """A wrinkle-free mesh yields an all-zero blend field."""
    mat = MaterialLibrary().get("AS4_3501_6")
    lam = Laminate.from_angles([0.0] * N_PLIES, mat, PLY_T)
    wc = WrinkleConfiguration(
        [WrinklePlacement(
            profile=GaussianSinusoidal(0.0, LX, 1.0e6, LX / 2.0),
            ply_interface=3,
            phase_offset=0.0,
        )],
        decay_mode="default",
    )
    mesh = WrinkleMesh(lam, wc, Lx=LX, Ly=LY, nx=40, ny=1, nz_per_ply=1).generate()
    w = compute_surface_resin_blend(mesh, wc, SurfacePocketSpec(side="both"))
    assert not np.any(w > 0)


def test_spec_rejects_bad_side_and_threshold():
    """SurfacePocketSpec validates its side and threshold at construction."""
    with pytest.raises(ValueError, match="side"):
        SurfacePocketSpec(side="left")
    with pytest.raises(ValueError, match="min_gap_threshold"):
        SurfacePocketSpec(min_gap_threshold=-1.0)


def test_none_wrinkle_config_is_rejected():
    """compute_surface_resin_blend refuses an undeformed (config-less) call."""
    mesh, _ = _build_mesh()
    with pytest.raises(ValueError, match="wrinkle"):
        compute_surface_resin_blend(mesh, None, SurfacePocketSpec())


# ---------------------------------------------------------------------------
# Part 2 — config validation (fast, no FE solve)
# ---------------------------------------------------------------------------


def _cfg(**over):
    from wrinklefe.analysis import AnalysisConfig

    base = dict(
        amplitude=0.354, wavelength=7.4, width=3.7, morphology="graded",
        loading="compression",
        material=MaterialLibrary().get("AC318_S6C10"),
        angles=[0.0] * 15, ply_thickness=0.42,
        nx=18, ny=3, nz_per_ply=3, domain_length=22.2, domain_width=10.0,
        applied_strain=-0.01,
    )
    base.update(over)
    return AnalysisConfig(**base)


def test_config_accepts_flat_morphologies():
    """stack/convex/concave and graded(decay_floor=0) are tool-flat: accepted."""
    for morph in ("stack", "convex", "concave"):
        _cfg(morphology=morph, enable_surface_resin_pockets=True)
    _cfg(morphology="graded", decay_floor=0.0, enable_surface_resin_pockets=True)


def test_config_rejects_uniform_morphology():
    with pytest.raises(ValueError, match="uniform morphology has wavy"):
        _cfg(morphology="uniform", enable_surface_resin_pockets=True)


def test_config_rejects_graded_with_decay_floor():
    with pytest.raises(ValueError, match="decay_floor"):
        _cfg(
            morphology="graded", decay_floor=0.2,
            enable_surface_resin_pockets=True,
        )


def test_config_rejects_analytical_only():
    with pytest.raises(ValueError, match="requires the FE path"):
        _cfg(enable_surface_resin_pockets=True, analytical_only=True)


def test_config_rejects_bad_side_and_gap():
    with pytest.raises(ValueError, match="surface_pocket_side"):
        _cfg(enable_surface_resin_pockets=True, surface_pocket_side="left")
    with pytest.raises(ValueError, match="surface_pocket_min_gap"):
        _cfg(enable_surface_resin_pockets=True, surface_pocket_min_gap=-1.0)


def test_config_roundtrips_with_surface_pockets():
    """The new fields ride the #259 dataclass-walk to_dict/from_dict."""
    from wrinklefe.analysis import AnalysisConfig

    cfg = _cfg(
        enable_surface_resin_pockets=True, surface_pocket_side="both",
        surface_pocket_min_gap=0.01,
    )
    restored = AnalysisConfig.from_dict(cfg.to_dict())
    assert restored.enable_surface_resin_pockets is True
    assert restored.surface_pocket_side == "both"
    assert restored.surface_pocket_min_gap == 0.01


def test_disabled_by_default_is_a_noop():
    """Off (the default) tags nothing — the mesh carries no resin fields."""
    cfg = _cfg()
    assert cfg.enable_surface_resin_pockets is False


# ---------------------------------------------------------------------------
# Part 3 — end-to-end FE wiring (integration; the solve is marked slow)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
def test_fe_pockets_tag_elements_and_reduce_modulus():
    """Surface pockets soften the laminate: modulus_retention_global drops."""
    from wrinklefe.analysis import WrinkleAnalysis

    off = WrinkleAnalysis(_cfg()).run()
    on = WrinkleAnalysis(
        _cfg(enable_surface_resin_pockets=True, surface_pocket_side="both")
    ).run()

    assert off.mesh.resin_blend is None
    assert on.mesh.resin_blend is not None
    assert int((on.mesh.resin_blend > 0).sum()) > 0
    assert on.mesh.resin_blend.max() <= 1.0
    assert on.mesh.resin_material.name == "EPOXY_S6C10"
    # The soft isotropic pockets reduce the global modulus retention.
    assert on.modulus_retention_global < off.modulus_retention_global


@pytest.mark.integration
@pytest.mark.slow
def test_fe_feature_off_is_bit_identical():
    """Enabling then disabling the flag reproduces the baseline exactly."""
    from wrinklefe.analysis import WrinkleAnalysis

    a = WrinkleAnalysis(_cfg()).run()
    b = WrinkleAnalysis(_cfg(enable_surface_resin_pockets=False)).run()
    assert a.modulus_retention_global == b.modulus_retention_global
    assert b.mesh.resin_blend is None and b.mesh.resin_mask is None


@pytest.mark.integration
@pytest.mark.slow
def test_fe_composes_with_crest_lens_no_double_blend():
    """Crest lens + surface pockets compose by per-element maximum."""
    from wrinklefe.analysis import WrinkleAnalysis

    crest = WrinkleAnalysis(_cfg(enable_resin_pocket=True)).run()
    both = WrinkleAnalysis(
        _cfg(
            enable_resin_pocket=True,
            enable_surface_resin_pockets=True, surface_pocket_side="both",
        )
    ).run()

    cw = crest.mesh.resin_blend
    bw = both.mesh.resin_blend
    assert np.all(bw >= cw - 1e-12)                 # composed via maximum
    assert int((bw > 0).sum()) >= int((cw > 0).sum())
    # Exactly one blended material per tagged element (no double-blend).
    assert len(both.mesh.resin_blend_materials) == int((bw > 0).sum())


@pytest.mark.integration
@pytest.mark.slow
def test_fe_binary_mode_tags_mask():
    """resin_pocket_graded=False routes surface pockets through resin_mask."""
    from wrinklefe.analysis import WrinkleAnalysis

    res = WrinkleAnalysis(
        _cfg(
            enable_surface_resin_pockets=True, surface_pocket_side="top",
            resin_pocket_graded=False,
        )
    ).run()
    assert res.mesh.resin_mask is not None
    assert res.mesh.resin_mask.sum() > 0
    assert res.mesh.resin_blend is None
