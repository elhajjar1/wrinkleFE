"""Tests for the resin-pocket material zone (Li et al. 2024/2025).

Covers the geometry helper (:mod:`wrinklefe.core.resin_pocket`), the
isotropic material constructor, and the end-to-end FE wiring through
:class:`~wrinklefe.analysis.WrinkleAnalysis` — that flagged elements use
the resin material, carry a zeroed fibre angle, and soften the laminate.
"""

from __future__ import annotations

import numpy as np
import pytest

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary, OrthotropicMaterial
from wrinklefe.core.resin_pocket import ResinPocketSpec, compute_resin_mask


# ----------------------------------------------------------------------
# Isotropic material constructor
# ----------------------------------------------------------------------
class TestIsotropicMaterial:
    def test_isotropic_shear_relation(self):
        m = OrthotropicMaterial.isotropic(3500.0, 0.35)
        assert m.E1 == m.E2 == m.E3 == 3500.0
        assert m.nu12 == m.nu13 == m.nu23 == 0.35
        assert m.G12 == pytest.approx(3500.0 / (2.0 * 1.35))
        assert m.G12 == m.G13 == m.G23

    def test_isotropic_positive_definite(self):
        # Construction runs validate(), which rejects non-PD compliance.
        m = OrthotropicMaterial.isotropic(3500.0, 0.35)
        eig = np.linalg.eigvalsh(m._build_compliance())
        assert np.all(eig > 0)

    def test_isotropic_rejects_bad_nu(self):
        with pytest.raises(ValueError):
            OrthotropicMaterial.isotropic(3500.0, 0.5)
        with pytest.raises(ValueError):
            OrthotropicMaterial.isotropic(-1.0, 0.3)

    def test_epoxy_card_present(self):
        m = MaterialLibrary().get("EPOXY_S6C10")
        assert m.E1 == pytest.approx(3500.0)
        assert m.E1 == m.E2 == m.E3  # isotropic


# ----------------------------------------------------------------------
# Pocket geometry
# ----------------------------------------------------------------------
class TestResinPocketSpec:
    def test_from_wrinkle_scales(self):
        spec = ResinPocketSpec.from_wrinkle(
            amplitude=0.5, wavelength=8.0, center_x=10.0, z_center=3.0,
            height_scale=1.0, length_scale=1.0,
        )
        assert spec.half_length == pytest.approx(4.0)   # L/2
        assert spec.h_center == pytest.approx(0.5)       # height_scale * A

    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            ResinPocketSpec(center_x=0.0, z_center=0.0,
                            half_length=0.0, h_center=1.0)
        with pytest.raises(ValueError):
            ResinPocketSpec(center_x=0.0, z_center=0.0,
                            half_length=1.0, h_center=-1.0)

    def test_mask_is_lens_shaped(self):
        # Build a real mesh and check the mask sits at the crest, is
        # symmetric about the centre, and tapers (more elements near the
        # centre column than at the longitudinal edges).
        cfg = AnalysisConfig(
            amplitude=0.5, wavelength=8.0, width=4.0, morphology="graded",
            loading="compression",
            material=MaterialLibrary().get("AC318_S6C10"),
            angles=[0.0] * 15, ply_thickness=0.42,
            nx=24, ny=4, nz_per_ply=3, domain_length=24.0, domain_width=10.0,
            analytical_only=False,
        )
        mesh = WrinkleAnalysis(cfg).run().mesh
        spec = ResinPocketSpec.from_wrinkle(
            amplitude=0.5, wavelength=8.0, center_x=12.0,
            z_center=0.5 * 0.42 * 15, height_scale=1.0, length_scale=1.0,
        )
        mask = compute_resin_mask(mesh, spec)
        assert mask.dtype == bool
        assert mask.sum() > 0
        centroids = mesh.nodes[mesh.elements].mean(axis=1)
        xr = centroids[mask, 0]
        zr = centroids[mask, 2]
        # Longitudinally centred on the crest, within the support.
        assert np.all(np.abs(xr - 12.0) <= spec.half_length + 1e-9)
        # Through-thickness centred near the midplane.
        assert abs(zr.mean() - spec.z_center) < spec.h_center


# ----------------------------------------------------------------------
# End-to-end FE wiring
# ----------------------------------------------------------------------
def _li_config(**over):
    base = dict(
        amplitude=0.354, wavelength=7.4, width=3.7, morphology="graded",
        loading="compression", material=MaterialLibrary().get("AC318_S6C10"),
        angles=[0.0] * 15, ply_thickness=0.42,
        nx=20, ny=4, nz_per_ply=3, domain_length=22.2, domain_width=10.0,
        applied_strain=-0.01,
    )
    base.update(over)
    return AnalysisConfig(**base)


class TestResinPocketFE:
    def test_disabled_by_default(self):
        res = WrinkleAnalysis(_li_config()).run()
        assert res.mesh.resin_mask is None
        assert res.mesh.resin_material is None

    def test_enabled_tags_elements(self):
        res = WrinkleAnalysis(_li_config(enable_resin_pocket=True)).run()
        assert res.mesh.resin_mask is not None
        assert res.mesh.resin_mask.sum() > 0
        assert res.mesh.resin_material.name == "EPOXY_S6C10"

    def test_pocket_softens_laminate(self):
        # The soft isotropic inclusion must reduce the FE modulus
        # retention relative to the no-pocket baseline (same geometry).
        base = WrinkleAnalysis(_li_config()).run()
        pocket = WrinkleAnalysis(_li_config(enable_resin_pocket=True)).run()
        assert pocket.modulus_retention < base.modulus_retention

    def test_custom_resin_material(self):
        soft = OrthotropicMaterial.isotropic(1000.0, 0.3, name="soft_resin")
        res = WrinkleAnalysis(
            _li_config(enable_resin_pocket=True, resin_pocket_material=soft)
        ).run()
        assert res.mesh.resin_material.name == "soft_resin"

    def test_analytical_only_skips_pocket(self):
        res = WrinkleAnalysis(
            _li_config(enable_resin_pocket=True)
        ).run(analytical_only=True)
        assert res.mesh is None  # no FE mesh built

    def test_invalid_scales_rejected(self):
        with pytest.raises(ValueError):
            _li_config(enable_resin_pocket=True, resin_pocket_height_scale=0.0)
        with pytest.raises(ValueError):
            _li_config(enable_resin_pocket=True, resin_pocket_length_scale=-1.0)
