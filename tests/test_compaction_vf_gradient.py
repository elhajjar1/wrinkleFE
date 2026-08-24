"""Compaction-driven Vf / ply-thickness gradient (issue #379, Part B).

Covers the core physics of :mod:`wrinklefe.core.compaction` (fast,
mesh-only) plus the ``analysis.py`` config surface, validation and FE
wiring; the FE solves are marked ``integration``/``slow`` per issue #267.

The properties pinned here are the ones the design rests on:

* **Exactness at nominal** — an element that keeps its nominal height gets
  ratios of exactly 1.0 and therefore the preset material *object* itself.
  This is what makes the feature zero-drift when it is off and
  bit-identical outside the compacted band when it is on.
* **Fibre conservation** — ``sum(Vf_local * h)`` over a column equals
  ``Vf_nominal * sum(h0)`` to machine precision for ``tool_flat`` (before
  the ``vf_max`` clamp engages), which is the whole justification for the
  kinematic rule.
* **Direction** — stretched (trough) elements soften, compacted (crest)
  elements stiffen.
* **Saturation** — the clamp engages at a realistic amplitude and says so.
* **Positive-definiteness** — every quantized bin material across an
  amplitude sweep is a valid ``OrthotropicMaterial``.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from wrinklefe.analysis import AnalysisConfig
from wrinklefe.core.compaction import (
    CONSTITUENT_DEFAULTS,
    VfGradientSpec,
    build_vf_materials,
    compute_vf_field,
    resolve_constituents,
    scale_material_to_vf,
)
from wrinklefe.core.laminate import Laminate
from wrinklefe.core.material import MaterialLibrary, OrthotropicMaterial
from wrinklefe.core.mesh import WrinkleMesh
from wrinklefe.core.morphology import WrinkleConfiguration
from wrinklefe.core.resin_pocket import element_height_ratio, element_heights
from wrinklefe.core.wrinkle import GaussianSinusoidal

PLY_T = 0.183
N_PLIES = 24
LX = 40.0
LY = 4.0
MATERIAL = "IM7_8552"


def _build_tool_flat_mesh(
    *,
    amplitude: float = 0.25,
    tool_side: str = "both",
    transition_plies: int = 2,
    nx: int = 40,
    ny: int = 2,
):
    """A tool-flat wrinkled mesh: flat outer surface(s), undulating core."""
    material = MaterialLibrary().get(MATERIAL)
    laminate = Laminate.from_angles([0.0] * N_PLIES, material, PLY_T)
    profile = GaussianSinusoidal(
        amplitude=amplitude, wavelength=16.0, width=12.0, center=LX / 2.0
    )
    config = WrinkleConfiguration.from_morphology_name(
        "tool_flat", profile,
        interface1=N_PLIES // 2 - 1, interface2=N_PLIES // 2,
        tool_side=tool_side, surface_transition_plies=transition_plies,
    )
    mesh = WrinkleMesh(
        laminate=laminate, wrinkle_config=config,
        Lx=LX, Ly=LY, nx=nx, ny=ny, nz_per_ply=1,
    ).generate()
    return mesh, laminate


def _spec(**kwargs) -> VfGradientSpec:
    return VfGradientSpec.for_material(MATERIAL, **kwargs)


# ---------------------------------------------------------------------------
# Part 1 — the shared height metric
# ---------------------------------------------------------------------------


def test_height_helper_is_the_surface_pocket_metric():
    """``element_height_ratio`` is ``h / h0`` from the shared helper."""
    mesh, _ = _build_tool_flat_mesh()
    height, h0, thickness = element_heights(mesh)
    assert h0 == pytest.approx(thickness / mesh.nz)
    assert thickness == pytest.approx(N_PLIES * PLY_T)
    np.testing.assert_allclose(element_height_ratio(mesh), height / h0)


def test_tool_flat_column_heights_sum_to_the_nominal_thickness():
    """The flat outer envelope keeps every column at nominal thickness."""
    mesh, _ = _build_tool_flat_mesh()
    height, h0, _ = element_heights(mesh)
    columns = height.reshape(mesh.nz, mesh.nx * mesh.ny).sum(axis=0)
    np.testing.assert_allclose(columns, mesh.nz * h0, rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Part 2 — the Vf field
# ---------------------------------------------------------------------------


def test_vf_field_follows_the_kinematic_rule():
    """``Vf_local == Vf_nominal * h0 / h`` wherever the clamps are inactive."""
    mesh, _ = _build_tool_flat_mesh()
    spec = _spec()
    vf = compute_vf_field(mesh, spec)
    expected = spec.vf_nominal / element_height_ratio(mesh)
    free = (expected > spec.vf_min) & (expected < spec.vf_max)
    assert free.sum() > 0
    np.testing.assert_allclose(vf[free], expected[free], rtol=1e-12)
    assert vf.min() >= spec.vf_min and vf.max() <= spec.vf_max


def test_vf_direction_stretched_soft_compacted_stiff():
    """Stretched elements lose fibre, compacted elements gain it."""
    mesh, _ = _build_tool_flat_mesh()
    spec = _spec()
    vf = compute_vf_field(mesh, spec)
    ratio = element_height_ratio(mesh)
    stretched = ratio > 1.0 + 1e-9
    compacted = ratio < 1.0 - 1e-9
    assert stretched.sum() > 0 and compacted.sum() > 0
    assert np.all(vf[stretched] < spec.vf_nominal)
    assert np.all(vf[compacted] > spec.vf_nominal)
    # Untouched elements sit on the nominal value (to floating point; the
    # quantization in ``build_vf_materials`` snaps that dust back exactly).
    nominal = ~(stretched | compacted)
    np.testing.assert_allclose(vf[nominal], spec.vf_nominal, rtol=1e-12)


def test_fibre_conservation_per_column():
    """``sum(Vf_local * h) == Vf_nominal * sum(h0)`` per column, to 1e-9."""
    # Amplitude chosen so no element reaches the vf_max cap: the tool_flat
    # crest band compacts by ``A / surface_transition_plies`` per element,
    # so the clamp engages above A ~ 0.073 mm here.
    mesh, _ = _build_tool_flat_mesh(amplitude=0.05)
    spec = _spec()
    vf = compute_vf_field(mesh, spec)
    assert np.all(vf < spec.vf_max)      # no clamp: conservation is exact
    height, h0, _ = element_heights(mesh)
    nz, ncol = mesh.nz, mesh.nx * mesh.ny
    fibre = (vf.reshape(nz, ncol) * height.reshape(nz, ncol)).sum(axis=0)
    np.testing.assert_allclose(
        fibre, spec.vf_nominal * nz * h0, rtol=1e-9, atol=0.0
    )


def test_conservation_holds_for_a_single_tool_side():
    """The single-tool (one caul plate) case conserves fibre too."""
    mesh, _ = _build_tool_flat_mesh(amplitude=0.05, tool_side="top")
    spec = _spec()
    vf = compute_vf_field(mesh, spec)
    assert np.all(vf < spec.vf_max)
    height, h0, _ = element_heights(mesh)
    nz, ncol = mesh.nz, mesh.nx * mesh.ny
    fibre = (vf.reshape(nz, ncol) * height.reshape(nz, ncol)).sum(axis=0)
    np.testing.assert_allclose(
        fibre, spec.vf_nominal * nz * h0, rtol=1e-9, atol=0.0
    )


def test_saturation_is_clamped_and_warned(caplog):
    """At a large amplitude the compaction cap engages and says so."""
    mesh, _ = _build_tool_flat_mesh(amplitude=0.29)
    spec = _spec()
    with caplog.at_level(logging.WARNING, logger="wrinklefe.core.compaction"):
        vf = compute_vf_field(mesh, spec)
    n_saturated = int(np.count_nonzero(vf == spec.vf_max))
    assert n_saturated > 0
    assert vf.max() == spec.vf_max
    messages = [r.getMessage() for r in caplog.records]
    assert any("saturated" in m and "lateral resin flow" in m
               for m in messages)
    assert any(str(n_saturated) in m for m in messages)


def test_flat_mesh_leaves_vf_at_nominal():
    """A wrinkle-free mesh has no compaction: Vf is nominal everywhere."""
    mesh, _ = _build_tool_flat_mesh(amplitude=0.0)
    spec = _spec()
    vf = compute_vf_field(mesh, spec)
    np.testing.assert_allclose(vf, spec.vf_nominal, rtol=1e-12)


# ---------------------------------------------------------------------------
# Part 3 — ratio-anchored materials
# ---------------------------------------------------------------------------


def test_nominal_vf_returns_the_preset_object():
    """The zero-drift anchor: at nominal Vf the preset card is untouched."""
    base = MaterialLibrary().get(MATERIAL)
    spec = _spec()
    assert scale_material_to_vf(base, spec.vf_nominal, spec) is base


def test_ratio_anchoring_scales_stiffness_and_holds_strengths():
    """Stiffnesses/CTEs move with Vf; Poisson ratios and strengths do not."""
    base = MaterialLibrary().get(MATERIAL)
    spec = _spec()
    stiff = scale_material_to_vf(base, 0.70, spec)
    soft = scale_material_to_vf(base, 0.45, spec)

    for attr in ("E1", "E2", "E3", "G12", "G13", "G23"):
        assert getattr(stiff, attr) > getattr(base, attr)
        assert getattr(soft, attr) < getattr(base, attr)
    # Documented limitation: no Vf-strength model, so strengths and the
    # Poisson ratios are carried over verbatim.
    for attr in ("Xt", "Xc", "Yt", "Yc", "Zt", "Zc", "S12", "S13", "S23",
                 "nu12", "nu13", "nu23", "gamma_Y", "GIc", "GIIc"):
        assert getattr(stiff, attr) == getattr(base, attr)
        assert getattr(soft, attr) == getattr(base, attr)


def test_ratio_equals_the_micromechanics_ratio():
    """``P_local / P_preset`` is exactly ``P_micro(Vf) / P_micro(Vf_nom)``."""
    from wrinklefe.core.micromechanics import (
        e1_rule_of_mixtures,
        e2_halpin_tsai,
    )

    base = MaterialLibrary().get(MATERIAL)
    spec = _spec()
    fiber, matrix = spec.constituents
    local = scale_material_to_vf(base, 0.68, spec)
    assert local.E1 / base.E1 == pytest.approx(
        e1_rule_of_mixtures(0.68, fiber, matrix)
        / e1_rule_of_mixtures(spec.vf_nominal, fiber, matrix),
        rel=1e-12,
    )
    assert local.E2 / base.E2 == pytest.approx(
        e2_halpin_tsai(0.68, fiber, matrix)
        / e2_halpin_tsai(spec.vf_nominal, fiber, matrix),
        rel=1e-12,
    )


@pytest.mark.parametrize("preset", sorted(CONSTITUENT_DEFAULTS))
def test_every_documented_preset_stays_positive_definite(preset):
    """Every bin material over the full Vf range is a valid ply card."""
    base = MaterialLibrary().get(preset)
    spec = VfGradientSpec.for_material(preset)
    for vf in np.linspace(spec.vf_min, spec.vf_max, 40):
        material = scale_material_to_vf(base, float(vf), spec)
        material.validate()          # raises if not positive-definite
        assert isinstance(material, OrthotropicMaterial)


def test_materials_are_shared_across_elements():
    """Quantization yields far fewer material objects than elements."""
    mesh, _ = _build_tool_flat_mesh()
    spec = _spec()
    vf = compute_vf_field(mesh, spec)
    materials = build_vf_materials(MaterialLibrary().get(MATERIAL), vf, spec)
    assert len(materials) > 0
    distinct = {id(m) for m in materials.values()}
    assert len(distinct) <= spec.n_bins
    assert len(distinct) < len(materials)


def test_nominal_elements_are_omitted_from_the_mapping():
    """Elements at nominal Vf keep the ply material (no entry at all)."""
    mesh, _ = _build_tool_flat_mesh()
    spec = _spec()
    vf = compute_vf_field(mesh, spec)
    materials = build_vf_materials(MaterialLibrary().get(MATERIAL), vf, spec)
    at_nominal = np.flatnonzero(vf == spec.vf_nominal)
    assert at_nominal.size > 0
    assert not (set(at_nominal.tolist()) & set(materials))


def test_element_ids_restricts_the_mapping():
    """``element_ids`` scopes the build to one ply material's elements."""
    mesh, _ = _build_tool_flat_mesh()
    spec = _spec()
    vf = compute_vf_field(mesh, spec)
    subset = np.arange(0, mesh.n_elements, 2, dtype=np.int64)
    materials = build_vf_materials(
        MaterialLibrary().get(MATERIAL), vf, spec, element_ids=subset
    )
    assert set(materials) <= set(subset.tolist())


# ---------------------------------------------------------------------------
# Part 4 — spec resolution and errors
# ---------------------------------------------------------------------------


def test_constituent_defaults_all_resolve():
    """Every entry of the defaults map names real presets/cards."""
    for name, (fiber, matrix, vf) in CONSTITUENT_DEFAULTS.items():
        resolve_constituents(fiber, matrix)
        assert 0.0 < vf < 1.0
        assert MaterialLibrary().get(name) is not None


def test_unknown_material_names_the_missing_knobs():
    """An undocumented card is an actionable error, not a guess."""
    with pytest.raises(ValueError, match="vf_fiber"):
        VfGradientSpec.for_material("NOT_A_MATERIAL")


def test_explicit_constituents_override_the_defaults():
    """Config values always win over the documented defaults."""
    spec = VfGradientSpec.for_material(
        MATERIAL, fiber="AS4", matrix="EPOXY_3501_6", vf_nominal=0.55
    )
    assert spec.fiber == "AS4" and spec.matrix == "EPOXY_3501_6"
    assert spec.vf_nominal == 0.55


def test_unknown_constituents_are_rejected():
    with pytest.raises(ValueError, match="Unknown fibre preset"):
        resolve_constituents("NOPE", "EPOXY_8552")
    with pytest.raises(ValueError, match="Unknown matrix"):
        resolve_constituents("IM7", "NOPE")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vf_nominal": 0.8, "vf_max": 0.75},     # nominal above the cap
        {"vf_min": 0.0},                          # non-positive floor
        {"vf_max": 0.95},                         # beyond the packing limit
        {"n_bins": 1},                            # degenerate grid
    ],
)
def test_spec_bounds_are_validated(kwargs):
    with pytest.raises(ValueError):
        VfGradientSpec(fiber="IM7", matrix="EPOXY_8552", **kwargs)


def test_library_isotropic_card_works_as_a_matrix():
    """A material-library isotropic card resolves as the matrix."""
    _, matrix = resolve_constituents("S2_GLASS", "EPOXY_S6C10")
    assert matrix.Em == pytest.approx(3_500.0)


# ---------------------------------------------------------------------------
# Part 5 — AnalysisConfig surface
# ---------------------------------------------------------------------------


def _cfg(**kwargs) -> AnalysisConfig:
    base = dict(
        morphology="tool_flat",
        amplitude=0.25,
        angles=[0.0] * N_PLIES,
        ply_thickness=PLY_T,
        surface_pocket_side="both",
        domain_length=LX,
        domain_width=LY,
        nx=40, ny=2, nz_per_ply=1,
        analytical_only=False,
    )
    base.update(kwargs)
    return AnalysisConfig(**base)


def test_disabled_by_default():
    assert AnalysisConfig().enable_vf_gradient is False
    assert AnalysisConfig().vf_nominal is None
    assert AnalysisConfig().vf_max == 0.75


def test_non_tool_flat_morphology_is_rejected():
    """v1 restriction: only a flat outer envelope conserves resin mass."""
    with pytest.raises(ValueError, match="restricted to morphology='tool_flat'"):
        _cfg(morphology="stack", enable_vf_gradient=True)


def test_analytical_only_is_rejected():
    with pytest.raises(ValueError, match="requires the FE path"):
        _cfg(enable_vf_gradient=True, analytical_only=True)


def test_unresolvable_constituents_are_rejected():
    with pytest.raises(ValueError, match="cannot resolve the micromechanics"):
        _cfg(
            enable_vf_gradient=True,
            material=OrthotropicMaterial(name="unknown_card"),
        )


def test_explicit_constituents_accept_an_unknown_card():
    cfg = _cfg(
        enable_vf_gradient=True,
        material=OrthotropicMaterial(name="unknown_card"),
        vf_fiber="IM7", vf_matrix="EPOXY_8552", vf_nominal=0.58,
    )
    assert cfg.vf_nominal == 0.58


def test_czm_combination_is_rejected():
    with pytest.raises(NotImplementedError, match="enable_czm"):
        _cfg(enable_vf_gradient=True, enable_czm=True, czm_interfaces=[1])


def test_config_round_trip_carries_the_new_fields():
    cfg = _cfg(enable_vf_gradient=True, vf_max=0.72, vf_nominal=0.61)
    restored = AnalysisConfig.from_dict(cfg.to_dict())
    assert restored.enable_vf_gradient is True
    assert restored.vf_max == 0.72
    assert restored.vf_nominal == 0.61
    assert restored == cfg


# ---------------------------------------------------------------------------
# Part 6 — end-to-end FE wiring (integration; solves are slow)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
def test_fe_gradient_installs_shared_materials():
    """The FE path attaches local-Vf materials on the blend channel."""
    from wrinklefe.analysis import WrinkleAnalysis

    result = WrinkleAnalysis(_cfg(enable_vf_gradient=True)).run()
    mesh = result.mesh
    materials = mesh.resin_blend_materials
    assert materials
    # Shared, not one per element (memory + the id()-keyed caches).
    assert len({id(m) for m in materials.values()}) < len(materials)
    # The binary surface-pocket tag is superseded, not composed.
    assert mesh.resin_blend is None
    assert mesh.resin_mask is None
    # The progressive-damage channel is left free.
    assert mesh.element_material_override is None
    base = MaterialLibrary().get(MATERIAL)
    stiffer = [m for m in materials.values() if m.E1 > base.E1]
    softer = [m for m in materials.values() if m.E1 < base.E1]
    assert stiffer and softer


@pytest.mark.integration
@pytest.mark.slow
def test_fe_off_is_the_untouched_baseline():
    """Off (the default) leaves the tool_flat result exactly as it was."""
    from wrinklefe.analysis import WrinkleAnalysis

    a = WrinkleAnalysis(_cfg()).run()
    b = WrinkleAnalysis(_cfg(enable_vf_gradient=False)).run()
    assert a.modulus_retention_global == b.modulus_retention_global
    # Off, the binary surface pockets still own the trough.
    assert a.mesh.resin_blend is not None


@pytest.mark.integration
@pytest.mark.slow
def test_fe_two_caul_significance():
    """Two-caul-plate case: the gradient measurably moves the retention.

    MEASURED on exactly this configuration (24-ply UD, tool_flat,
    side="both", A = 0.25 mm, 40 x 4 mm domain): binary surface pockets
    give ``modulus_retention_global = 0.937591`` and the Vf gradient
    ``0.944565`` — the gradient is +0.006974 (+0.744 %) *stiffer*, because
    the compacted crest band (Vf up to the 0.75 cap) stiffens more than the
    resin-rich trough softens, and because the continuous field replaces a
    binary neat-resin tag.  Against a wrinkle with no trough treatment at
    all (0.957389) the gradient is 0.012824 *softer*, i.e. it lands between
    the two, which is the physically expected ordering.  64 of the 1920
    elements saturate at the ``vf_max`` cap here.

    The floor below is deliberately loose (a third of the measured delta)
    so mesh/solver noise cannot flake it while a regression that silently
    disables the feature still fails.
    """
    from wrinklefe.analysis import WrinkleAnalysis

    binary = WrinkleAnalysis(_cfg()).run()
    gradient = WrinkleAnalysis(_cfg(enable_vf_gradient=True)).run()

    delta = (
        gradient.modulus_retention_global - binary.modulus_retention_global
    )
    assert abs(delta) > 0.002, (
        f"Vf gradient changed modulus_retention_global by only {delta:+.5f}; "
        f"expected the measured ~+0.0070."
    )
    # Direction, as measured: the compacted band dominates.
    assert delta > 0.0


@pytest.mark.integration
@pytest.mark.slow
def test_fe_composes_with_the_crest_lens():
    """The machined crest lens still blends, on top of the local-Vf host."""
    from wrinklefe.analysis import WrinkleAnalysis

    result = WrinkleAnalysis(
        _cfg(enable_vf_gradient=True, enable_resin_pocket=True)
    ).run()
    mesh = result.mesh
    assert mesh.resin_blend is not None          # the lens weight survives
    assert int((mesh.resin_blend > 0).sum()) > 0
    assert mesh.resin_blend_materials
    # Lens elements are blended toward the soft isotropic resin card.
    resin = MaterialLibrary().get("EPOXY_S6C10")
    lens = np.flatnonzero(mesh.resin_blend > 0.5)
    assert lens.size > 0
    for elem in lens[:20].tolist():
        material = mesh.resin_blend_materials.get(elem)
        if material is not None:
            assert material.E1 < MaterialLibrary().get(MATERIAL).E1
            assert material.E1 >= resin.E1 * 0.5
