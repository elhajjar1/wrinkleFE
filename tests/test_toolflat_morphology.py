"""Acceptance tests for the ``tool_flat`` morphology (issue #371, Part A).

These encode the user's complaint and the fix's guarantees:

(a) **Significance** — toggling the surface pockets on/off moves
    ``modulus_retention_global`` by a clearly significant margin for
    ``tool_flat`` (>= 3 %), versus a negligible move for the legacy
    ``stack`` morphology (the original bug).
(b) **Preview-gap magnitude** — the trough pocket depth is >= ~1 ply
    thickness (vs ~0.25 for the linear-decay morphologies).
(c) **Flatness invariant** — the pinned surface(s) are exactly flat.
(d) **Positive Jacobians** at the validated amplitude bound; amplitudes
    beyond it are rejected at construction.
(e) **Analytical path equals uniform** (M_f = 1.0); zero drift for the
    existing morphologies is covered by the validation ledger.

The FE-solve cases are marked ``integration``/``slow`` per issue #267;
the geometry/validation cases run fast (no solve).
"""

from __future__ import annotations

import numpy as np
import pytest

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.laminate import Laminate
from wrinklefe.core.material import MaterialLibrary
from wrinklefe.core.mesh import WrinkleMesh
from wrinklefe.core.morphology import WrinkleConfiguration
from wrinklefe.core.wrinkle import GaussianSinusoidal

PLY_T = 0.183
N_PLIES = 24


def _common(**over):
    base = dict(
        material=MaterialLibrary().get("IM7_8552"),
        angles=[0.0] * N_PLIES,
        ply_thickness=PLY_T,
        wavelength=16.0,
        width=12.0,
        nx=36,
        ny=3,
        nz_per_ply=1,
        loading="compression",
    )
    base.update(over)
    return base


def _toolflat_mesh(amplitude, *, side="both", s_trans=3, nz_per_ply=1, nx=36):
    """Deformed tool_flat mesh only (no FE solve) — fast."""
    mat = MaterialLibrary().get("IM7_8552")
    lam = Laminate.from_angles([0.0] * N_PLIES, mat, PLY_T)
    Lx = 3.0 * 16.0
    prof = GaussianSinusoidal(
        amplitude=amplitude, wavelength=16.0, width=12.0, center=Lx / 2.0
    )
    wc = WrinkleConfiguration.from_morphology_name(
        "tool_flat", prof, interface1=11, interface2=12,
        tool_side=side, surface_transition_plies=s_trans,
    )
    mesh = WrinkleMesh(
        lam, wc, Lx=Lx, Ly=20.0, nx=nx, ny=3, nz_per_ply=nz_per_ply
    ).generate()
    return mesh, wc


# ---------------------------------------------------------------------------
# (e) Morphology / analytical path
# ---------------------------------------------------------------------------


def test_toolflat_is_single_wrinkle_mf_one():
    """tool_flat is a single mid-interface wrinkle with M_f = 1.0."""
    prof = GaussianSinusoidal(0.25, 16.0, 12.0, center=24.0)
    wc = WrinkleConfiguration.from_morphology_name(
        "tool_flat", prof, interface1=11, interface2=12
    )
    assert wc.n_wrinkles() == 1
    assert wc.decay_mode == "tool_flat"
    assert wc.aggregate_morphology_factor("compression") == 1.0


def test_toolflat_decay_core_full_surface_pinned():
    """Decay is 1 in the core, ramps over S plies, and 0 at the surface."""
    prof = GaussianSinusoidal(0.25, 16.0, 12.0, center=24.0)
    wc = WrinkleConfiguration.from_morphology_name(
        "tool_flat", prof, interface1=11, interface2=12,
        tool_side="both", surface_transition_plies=2,
    )
    ply = np.minimum(np.arange(N_PLIES + 1), N_PLIES - 1)
    decay = wc._through_thickness_decay(ply, 11, N_PLIES)
    assert decay[0] == 0.0 and decay[N_PLIES - 1] == 0.0  # both surfaces pinned
    assert decay[N_PLIES // 2] == 1.0                     # full-amplitude core
    assert decay[1] == pytest.approx(0.5)                 # linear ramp (S=2)


def test_toolflat_analytical_equals_uniform():
    """The closed-form knockdown equals uniform's (same full-amplitude core)."""
    tf = WrinkleAnalysis(
        AnalysisConfig(morphology="tool_flat", amplitude=0.25, analytical_only=True,
                       material=MaterialLibrary().get("IM7_8552"),
                       angles=[0.0] * N_PLIES, ply_thickness=PLY_T)
    ).run()
    uni = WrinkleAnalysis(
        AnalysisConfig(morphology="uniform", amplitude=0.25, analytical_only=True,
                       material=MaterialLibrary().get("IM7_8552"),
                       angles=[0.0] * N_PLIES, ply_thickness=PLY_T)
    ).run()
    assert tf.analytical_knockdown == pytest.approx(uni.analytical_knockdown)


# ---------------------------------------------------------------------------
# (b, c, d) geometry: gap magnitude, flatness, positive Jacobians
# ---------------------------------------------------------------------------


def test_pocket_gap_is_at_least_one_ply_thickness():
    """The trough pocket depth is >= ~1 ply thickness (vs ~0.25 for linear)."""
    A = 0.35
    mesh, _ = _toolflat_mesh(A, side="top", s_trans=3)
    # Pocket depth = the deepest the core plies pull away from the flat
    # surface = the full amplitude, since the core decay is 1.
    flat = WrinkleMesh(
        Laminate.from_angles([0.0] * N_PLIES,
                             MaterialLibrary().get("IM7_8552"), PLY_T),
        None, Lx=48.0, Ly=20.0, nx=36, ny=3, nz_per_ply=1,
    ).generate()
    dz = mesh.nodes[:, 2] - flat.nodes[:, 2]
    pocket_depth = float(np.max(np.abs(dz)))
    assert pocket_depth == pytest.approx(A, rel=0.02)
    assert pocket_depth / PLY_T > 1.0


def test_pinned_surface_is_exactly_flat():
    """The tool_side surface(s) carry no residual waviness."""
    mesh, _ = _toolflat_mesh(0.30, side="both", s_trans=2)
    per_layer = (mesh.nx + 1) * (mesh.ny + 1)
    k_idx = np.arange(mesh.n_nodes) // per_layer
    top = mesh.nodes[k_idx == mesh.nz, 2]
    bot = mesh.nodes[k_idx == 0, 2]
    assert float(top.max() - top.min()) < 1e-12
    assert float(bot.max() - bot.min()) < 1e-12


def test_multi_layer_pocket_volume_is_conserved():
    """Σ weight·h over the S-layer transition zone == the integrated gap."""
    from wrinklefe.core.resin_pocket import (
        SurfacePocketSpec,
        compute_surface_resin_blend,
    )

    A, S = 0.3, 3
    mesh, wc = _toolflat_mesh(A, side="top", s_trans=S, nx=48)
    w = compute_surface_resin_blend(mesh, wc, SurfacePocketSpec(side="top"))
    tagged = w > 0
    # Exactly S element-layers carry the pocket.
    k_layers = np.unique(np.flatnonzero(tagged) // (mesh.nx * mesh.ny))
    assert len(k_layers) == S

    ez = mesh.nodes[mesh.elements][:, :, 2]
    ex = mesh.nodes[mesh.elements][:, :, 0]
    ey = mesh.nodes[mesh.elements][:, :, 1]
    h = ez[:, 4:8].mean(axis=1) - ez[:, 0:4].mean(axis=1)
    area = (ex.max(1) - ex.min(1)) * (ey.max(1) - ey.min(1))
    zc = np.asarray(mesh.laminate.z_coords(), dtype=float)
    h0 = float(zc[-1] - zc[0]) / mesh.nz

    # weight·h == h − h0 per element (excess vertical stretch), so the tagged
    # volume equals the integrated kinematic gap through the whole zone.
    vol_weight = float((w * h * area)[tagged].sum())
    vol_gap = float(((h - h0) * area)[tagged].sum())
    assert np.isclose(vol_weight, vol_gap, rtol=1e-9)
    assert vol_gap > 0.0


def test_positive_jacobians_at_the_amplitude_bound():
    """At the max validated amplitude the mesh generates (positive Jacobians)."""
    for s_trans in (2, 3):
        a_bound = 0.8 * s_trans * PLY_T / 1  # nz_per_ply = 1
        # Mesh generation runs the strict det(J) > 0 check and would raise
        # MeshValidationError on inversion — reaching here means it passed.
        mesh, _ = _toolflat_mesh(a_bound, side="both", s_trans=s_trans)
        assert mesh.n_elements > 0


# ---------------------------------------------------------------------------
# (d) validation: inversion bound, S >= 1, combination rules, auto-enable
# ---------------------------------------------------------------------------


def test_amplitude_above_bound_is_rejected_with_both_remedies():
    a_bound = 0.8 * 2 * PLY_T / 1
    with pytest.raises(ValueError) as exc:
        AnalysisConfig(**_common(morphology="tool_flat",
                                 amplitude=a_bound * 1.2,
                                 surface_transition_plies=2))
    msg = str(exc.value)
    assert "surface_transition_plies" in msg  # remedy 1: more transition plies
    assert "amplitude" in msg                  # remedy 2: smaller amplitude


def test_surface_transition_plies_must_be_at_least_one():
    with pytest.raises(ValueError, match="surface_transition_plies"):
        AnalysisConfig(**_common(morphology="tool_flat", amplitude=0.2,
                                 surface_transition_plies=0))


def test_toolflat_auto_enables_pockets_on_fe_path():
    cfg = AnalysisConfig(**_common(morphology="tool_flat", amplitude=0.25))
    assert cfg.enable_surface_resin_pockets is True


def test_toolflat_analytical_only_does_not_auto_enable():
    cfg = AnalysisConfig(morphology="tool_flat", amplitude=0.25,
                         analytical_only=True,
                         material=MaterialLibrary().get("IM7_8552"),
                         angles=[0.0] * N_PLIES, ply_thickness=PLY_T)
    assert cfg.enable_surface_resin_pockets is False


def test_toolflat_rejects_unverified_combinations():
    from wrinklefe.analysis import WrinkleSpec

    with pytest.raises(NotImplementedError, match="multi-wrinkle"):
        AnalysisConfig(**_common(
            morphology="tool_flat", amplitude=0.2,
            wrinkles=[WrinkleSpec(amplitude=0.2, wavelength=16.0, width=12.0,
                                  ply_interface=11)],
        ))
    with pytest.raises(NotImplementedError, match="enable_czm"):
        AnalysisConfig(**_common(morphology="tool_flat", amplitude=0.2,
                                 enable_czm=True))
    with pytest.raises(NotImplementedError, match="transverse_mode"):
        AnalysisConfig(**_common(morphology="tool_flat", amplitude=0.2,
                                 transverse_mode="gaussian_decay"))


def test_toolflat_roundtrips_new_field():
    """surface_transition_plies rides the #259 dataclass-walk round-trip."""
    cfg = AnalysisConfig(**_common(morphology="tool_flat", amplitude=0.25,
                                   surface_transition_plies=3))
    restored = AnalysisConfig.from_dict(cfg.to_dict())
    assert restored.surface_transition_plies == 3
    assert restored.morphology == "tool_flat"


# ---------------------------------------------------------------------------
# (a) Headline significance — the user's complaint, encoded (FE solve; slow)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
def test_toggling_pockets_is_significant_for_toolflat_not_for_stack():
    """tool_flat pockets move the modulus by >= 3 %; the old stack barely moves.

    The toggle for tool_flat compares the auto-tagged pockets against the
    *identical* geometry with tagging suppressed (a huge ``min_gap``), so the
    only difference is the resin material — a clean on/off of the feature.
    """
    A, S = 0.55, 4

    def run(**kw):
        return WrinkleAnalysis(AnalysisConfig(**_common(**kw))).run()

    tf_on = run(morphology="tool_flat", amplitude=A, surface_pocket_side="both",
                surface_transition_plies=S)
    tf_off = run(morphology="tool_flat", amplitude=A, surface_pocket_side="both",
                 surface_transition_plies=S, surface_pocket_min_gap=1.0e9)
    st_on = run(morphology="stack", amplitude=A,
                enable_surface_resin_pockets=True, surface_pocket_side="both")
    st_off = run(morphology="stack", amplitude=A,
                 enable_surface_resin_pockets=False)

    tf_delta = tf_off.modulus_retention_global - tf_on.modulus_retention_global
    st_delta = st_off.modulus_retention_global - st_on.modulus_retention_global

    assert int((tf_on.mesh.resin_blend > 0).sum()) > 0
    assert tf_off.mesh.resin_blend is None or not (tf_off.mesh.resin_blend > 0).any()

    # tool_flat: a clearly significant, asserted change (>= 3 %).
    assert tf_delta >= 0.03
    # Old stack: negligible (the original bug — pockets insignificant).
    assert st_delta < 0.01
    # ... and the tool_flat toggle dwarfs the old one.
    assert tf_delta > 5.0 * st_delta


@pytest.mark.integration
@pytest.mark.slow
def test_toolflat_tags_multiple_transition_layers():
    """All S transition layers are tagged (not just the outermost one)."""
    res = WrinkleAnalysis(
        AnalysisConfig(**_common(morphology="tool_flat", amplitude=0.3,
                                 surface_pocket_side="top",
                                 surface_transition_plies=3))
    ).run()
    weight = res.mesh.resin_blend
    tagged = np.flatnonzero(weight > 0)
    k_layers = np.unique(tagged // (res.mesh.nx * res.mesh.ny))
    assert len(k_layers) == 3  # one per transition ply
