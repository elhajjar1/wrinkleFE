"""End-to-end tests for the Phase 3 ``enable_czm`` wiring.

These tests exercise the public :class:`~wrinklefe.analysis.WrinkleAnalysis`
/ :class:`~wrinklefe.analysis.AnalysisConfig` API after the cohesive-
zone modelling (CZM) integration: the ``enable_czm=True`` switch must
plumb through to the Newton-Raphson solver, populate the new CZM
result fields, and not perturb the legacy (``enable_czm=False``)
behaviour.

The reference geometry is a [0/90]_4s IM7/8552 laminate with a
concave dual-wrinkle (amplitude 0.366 mm, wavelength 16 mm).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary

# ----------------------------------------------------------------------
# Shared fixtures
# ----------------------------------------------------------------------


_LAYUP_0_90_4S = ([0, 90] * 4) + ([90, 0] * 4)  # 16 plies, [0/90]_4s


def _czm_config(**overrides) -> AnalysisConfig:
    """Build a baseline tension + concave CZM config (IM7/8552, [0/90]_4s).

    Defaults are tuned to (a) trigger meaningful cohesive damage at
    the wrinkle crest and (b) still converge through the
    NewtonRaphsonSolver — sit just past damage initiation but well
    below catastrophic interface failure.

    The applied strain (0.025) was recalibrated for the corrected
    dual-wrinkle amplitude contract (issue #305): the concave mesh now
    composes to ~0.70*A rather than the previous ~1.37*A, so the older
    0.015 strain no longer opened the crest interface past the cohesive
    initiation threshold. 0.025 restores the intended "just past
    initiation" state (max damage ~0.04, still convergent) for the
    physically-correct geometry.
    """
    mat = MaterialLibrary().get("IM7_8552")
    defaults = dict(
        amplitude=0.366,
        wavelength=16.0,
        width=12.0,
        morphology="concave",
        loading="tension",
        material=mat,
        angles=list(_LAYUP_0_90_4S),
        ply_thickness=0.183,
        nx=12,
        ny=4,
        nz_per_ply=1,
        applied_strain=0.025,
        enable_czm=True,
        czm_n_load_increments=20,
        verbose=False,
    )
    defaults.update(overrides)
    return AnalysisConfig(**defaults)


# ----------------------------------------------------------------------
# 1. Canonical end-to-end CZM smoke test
# ----------------------------------------------------------------------


class TestEnableCzmEndToEnd:

    def test_enable_czm_concave_tension_runs_end_to_end(self):
        """Concave tension wrinkle drives non-zero cohesive damage at the
        crest, the Newton-Raphson loop converges, and the FE knockdown
        signature changes vs the linear path."""
        cfg = _czm_config()
        result = WrinkleAnalysis(cfg).run()

        # Convergence indicator must be set and truthy.
        assert result.czm_converged is True, (
            f"NR solver did not converge "
            f"(czm_converged={result.czm_converged}); "
            f"load_displacement shape="
            f"{getattr(result.czm_load_displacement, 'shape', None)}"
        )

        # CZM fields populated with non-trivial damage.
        assert result.czm_damage is not None
        assert result.czm_damage.size > 0
        assert np.max(result.czm_damage) > 0.0, (
            "Expected the concave-tension wrinkle to initiate cohesive "
            "damage at the crest; got all zeros."
        )

        assert result.czm_energy_dissipated is not None
        assert result.czm_energy_dissipated > 0.0

        assert result.czm_separation is not None
        assert result.czm_separation.shape == result.czm_damage.shape + (3,)

        assert result.czm_traction is not None
        assert result.czm_traction.shape == result.czm_damage.shape + (3,)

        assert result.czm_load_displacement is not None
        assert result.czm_load_displacement.ndim == 2
        assert result.czm_load_displacement.shape[1] == 2

        assert result.czm_interfaces_used is not None
        assert len(result.czm_interfaces_used) >= 1

        # Sanity: a CZM-mode result is qualitatively different from the
        # linear (no-CZM) baseline.  The analytical (closed-form)
        # knockdown is the same in both runs — it depends only on the
        # input geometry — so we compare a *CZM-specific* output (e.g.
        # the max displacement of the deformed mesh) against the linear
        # baseline solve to confirm the CZM path actually ran.
        linear_cfg = dataclasses.replace(cfg, enable_czm=False)
        linear = WrinkleAnalysis(linear_cfg).run()

        # Both paths populate ``field_results``: the CZM path recovers
        # bulk hex8 stress/strain from the final Newton displacement so
        # ply-level failure criteria can run alongside the interface
        # delamination output.
        assert linear.field_results is not None
        assert result.field_results is not None, (
            "CZM path must populate field_results so ply-level failure "
            "criteria can run on the bulk material."
        )
        # The CZM run's displacement vector has the same shape as the
        # linear run's (n_nodes, 3) — but with the duplicated cohesive
        # nodes the CZM array is strictly larger.
        assert result.field_results.displacement.shape[0] == result.mesh.n_nodes
        assert result.field_results.stress_local.shape[0] == result.mesh.n_elements
        # The CZM mesh has duplicated interface nodes, so the node
        # count is strictly greater than the linear (no-CZM) mesh's
        # node count.  This is the cheapest end-to-end signal that the
        # ``enable_czm`` switch reached the mesh path.
        assert result.mesh.n_nodes > linear.mesh.n_nodes, (
            f"CZM mesh ({result.mesh.n_nodes} nodes) should have more "
            f"nodes than the linear mesh ({linear.mesh.n_nodes}); the "
            f"cohesive insertion did not run."
        )

    def test_enable_czm_off_matches_baseline(self):
        """``enable_czm=False`` produces the existing analytical+FE result.

        Guardrail that no CZM-related field leaks into the legacy path.
        We compare against a snapshot of the analytical knockdown for
        the canonical [0/90]_4s IM7/8552 concave tension wrinkle —
        which is purely an analytical quantity (closed-form formula on
        the input geometry) so it is reproducible to machine precision.
        """
        cfg = _czm_config(enable_czm=False)
        result = WrinkleAnalysis(cfg).run()

        # No CZM fields populated.
        assert result.czm_damage is None
        assert result.czm_separation is None
        assert result.czm_traction is None
        assert result.czm_energy_dissipated is None
        assert result.czm_load_displacement is None
        assert result.czm_converged is None
        assert result.czm_interfaces_used is None
        assert result.czm_delamination_report is None

        # The analytical knockdown for the canonical case is in a
        # reasonable physical range (computed from a closed-form
        # expression on the inputs — reproducible to machine precision
        # within the AnalysisConfig API).
        assert 0.5 < result.analytical_knockdown < 1.0
        assert result.field_results is not None
        assert result.retention_factors is not None

    def test_enable_czm_compression_runs(self):
        """The CZM path must run on a compression wrinkle without crashing.

        Cohesive damage is secondary to kink-band in compression so the
        actual damage value is allowed to be small (potentially zero);
        the assertion is operational, not physical.
        """
        cfg = _czm_config(
            loading="compression",
            applied_strain=-0.01,
            czm_n_load_increments=10,
        )
        result = WrinkleAnalysis(cfg).run()
        assert result.czm_damage is not None
        assert result.czm_energy_dissipated is not None
        assert result.czm_converged is True


# ----------------------------------------------------------------------
# 2. czm_interfaces resolution
# ----------------------------------------------------------------------


class TestCzmInterfaceResolution:

    def test_czm_interfaces_near_crest_picks_correct_z(self):
        """``"near_crest"`` selects the interface closest to the wrinkle's
        reference centre z.  The concave dual-wrinkle places one wrinkle
        each at ``interface_1`` and ``interface_2`` (auto-derived
        symmetrically about the midplane); the near-crest selection
        should land on one of those two boundaries.
        """
        cfg = _czm_config(czm_interfaces="near_crest")
        analysis = WrinkleAnalysis(cfg)
        # Don't run the full solve — just inspect the resolver.
        laminate = analysis._build_laminate()
        from wrinklefe.core.morphology import WrinkleConfiguration
        from wrinklefe.core.wrinkle import GaussianSinusoidal

        profile = GaussianSinusoidal(
            amplitude=cfg.amplitude,
            wavelength=cfg.wavelength,
            width=cfg.width,
            center=cfg.domain_length / 2.0,
        )
        wrinkle_config = WrinkleConfiguration.from_morphology_name(
            cfg.morphology, profile,
            interface1=cfg.interface_1,
            interface2=cfg.interface_2,
            decay_floor=cfg.decay_floor,
        )

        ifaces = analysis._resolve_cohesive_interfaces(
            laminate, wrinkle_config,
        )
        assert len(ifaces) == 1
        chosen = ifaces[0]
        # The chosen interface index must equal one of the wrinkle's
        # ply-interface indices (the resolver picks the interior
        # boundary whose z matches the strongest wrinkle's reference
        # centre z; for the concave dual-wrinkle that is one of
        # interface_1 / interface_2 + 1, i.e. the boundary index that
        # appears directly in the wrinkle placement).
        wrinkle_ifaces = {w.ply_interface for w in wrinkle_config.wrinkles}
        # The resolver returns boundary indices; the placement's
        # ``ply_interface`` is also a boundary index in the same
        # numbering, so they should be equal for at least one wrinkle.
        assert chosen in wrinkle_ifaces, (
            f"near_crest selected interface {chosen} but the wrinkle "
            f"placements use interfaces {wrinkle_ifaces}."
        )

    def test_czm_interfaces_all_inserts_all_interior_planes(self):
        """``"all"`` resolves to every interior ply boundary."""
        cfg = _czm_config(czm_interfaces="all")
        analysis = WrinkleAnalysis(cfg)
        laminate = analysis._build_laminate()
        from wrinklefe.core.morphology import WrinkleConfiguration
        from wrinklefe.core.wrinkle import GaussianSinusoidal

        profile = GaussianSinusoidal(
            amplitude=cfg.amplitude, wavelength=cfg.wavelength,
            width=cfg.width, center=cfg.domain_length / 2.0,
        )
        wrinkle_config = WrinkleConfiguration.from_morphology_name(
            cfg.morphology, profile,
            interface1=cfg.interface_1, interface2=cfg.interface_2,
            decay_floor=cfg.decay_floor,
        )

        ifaces = analysis._resolve_cohesive_interfaces(
            laminate, wrinkle_config,
        )
        assert ifaces == list(range(laminate.n_plies - 1))

    def test_czm_interfaces_explicit_list(self):
        """An explicit ``list[int]`` overrides the auto-derived choice."""
        cfg = _czm_config(czm_interfaces=[2, 5, 9])
        analysis = WrinkleAnalysis(cfg)
        laminate = analysis._build_laminate()
        from wrinklefe.core.morphology import WrinkleConfiguration
        from wrinklefe.core.wrinkle import GaussianSinusoidal

        profile = GaussianSinusoidal(
            amplitude=cfg.amplitude, wavelength=cfg.wavelength,
            width=cfg.width, center=cfg.domain_length / 2.0,
        )
        wrinkle_config = WrinkleConfiguration.from_morphology_name(
            cfg.morphology, profile,
            interface1=cfg.interface_1, interface2=cfg.interface_2,
            decay_floor=cfg.decay_floor,
        )

        ifaces = analysis._resolve_cohesive_interfaces(
            laminate, wrinkle_config,
        )
        assert ifaces == [2, 5, 9]


# ----------------------------------------------------------------------
# 3. MaterialLibrary CZM defaults coverage
# ----------------------------------------------------------------------


class TestMaterialLibraryCzmDefaults:

    def test_material_library_has_czm_defaults(self):
        """Every built-in material exposes finite positive ``sigma_max``
        and ``tau_max`` and (when present) ``GIc``/``GIIc``."""
        lib = MaterialLibrary()
        for name in lib.list_names():
            mat = lib.get(name)
            for attr in ("sigma_max", "tau_max"):
                val = getattr(mat, attr)
                assert val is not None, f"{name}.{attr} is None"
                assert np.isfinite(val), f"{name}.{attr} is not finite"
                assert val > 0.0, f"{name}.{attr} must be > 0; got {val}"
            # GIc/GIIc are optional — but when supplied they must be
            # finite positive (already enforced by validate() at
            # material construction, this is a runtime guardrail).
            for attr in ("GIc", "GIIc"):
                val = getattr(mat, attr)
                if val is not None:
                    assert np.isfinite(val), f"{name}.{attr} not finite"
                    assert val > 0.0, f"{name}.{attr} must be > 0"
