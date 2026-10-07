"""Dedicated unit tests for the LaRC04/05 failure criterion.

These tests target the LaRC05-specific physics that ``test_criteria.py``
and ``test_evaluator.py`` do not exercise directly:

- fibre tension with the quadratic shear-interaction term,
- fibre kinking under compression: the Xc-calibrated intrinsic
  misalignment (a pristine ply kinks exactly at -Xc), the kink-plane
  search, and the agreement between a wrinkle expressed in the stress
  frame (the FE route) and one supplied as ``misalignment_angle``,
- matrix tension governed by the *in-situ* transverse strength,
- matrix compression resolved on the fracture plane at +/- alpha_0,
- monotonic / proportional-loading behaviour and reserve factor.

Material: default ``OrthotropicMaterial()`` == IM7/8552 (matching the
convention used by the other ``tests/test_failure/`` modules; this is
the same data as ``MaterialLibrary().get("IM7_8552")``).
"""

import numpy as np
import pytest
from scipy.optimize import brentq

from wrinklefe.core.material import MaterialLibrary, OrthotropicMaterial
from wrinklefe.core.transforms import stress_transformation_3d
from wrinklefe.failure.base import FailureResult
from wrinklefe.failure.larc05 import LaRC05Criterion

_LIBRARY = MaterialLibrary()


@pytest.fixture
def material():
    """Default IM7/8552 orthotropic material (same as MaterialLibrary)."""
    return OrthotropicMaterial()


@pytest.fixture
def criterion():
    return LaRC05Criterion()


# ======================================================================
# Zero stress
# ======================================================================

class TestLaRC05ZeroStress:

    def test_zero_stress_gives_fi_zero(self, criterion, material):
        result = criterion.evaluate(np.zeros(6), material)
        assert isinstance(result, FailureResult)
        assert result.index == pytest.approx(0.0, abs=1e-12)
        assert result.criterion_name == "larc05"

    def test_zero_stress_reserve_factor_infinite(self, criterion, material):
        result = criterion.evaluate(np.zeros(6), material)
        assert np.isinf(result.reserve_factor)


# ======================================================================
# Fibre tension (with shear interaction)
# ======================================================================

class TestLaRC05FibreTension:

    def test_pure_longitudinal_tension_at_Xt(self, criterion, material):
        """sigma_11 = Xt -> fibre-tension FI = 1.0, mode fiber_tension."""
        stress = np.array([material.Xt, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-10)
        assert result.mode == "fiber_tension"
        # Fibre sub-criterion governs; no matrix damage under pure sigma_11.
        assert result.detail["fi_fiber"] == pytest.approx(1.0, abs=1e-10)
        assert result.detail["fi_matrix"] == pytest.approx(0.0, abs=1e-10)

    def test_below_Xt_fi_less_than_one(self, criterion, material):
        stress = np.array([0.5 * material.Xt, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(0.5, abs=1e-10)
        assert result.index < 1.0

    def test_above_Xt_fi_greater_than_one(self, criterion, material):
        stress = np.array([1.5 * material.Xt, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.5, abs=1e-10)
        assert result.index > 1.0

    def test_shear_interaction_amplifies_fibre_tension(
        self, criterion, material
    ):
        """Adding tau_12 to a longitudinal-tension state must raise the
        fibre-tension FI (the LaRC05 quadratic shear-interaction term)."""
        base = np.array([0.5 * material.Xt, 0.0, 0.0, 0.0, 0.0, 0.0])
        with_shear = base.copy()
        with_shear[5] = 0.5 * material.S12
        fi_base = criterion.evaluate(base, material).index
        fi_shear = criterion.evaluate(with_shear, material).index
        assert fi_shear > fi_base


# ======================================================================
# Fibre kinking (compression)
# ======================================================================

class TestLaRC05FibreKinking:

    def test_past_the_instability_the_index_is_finite_and_one_over_rf(self):
        """Past the kinking instability the closed form has no value. The
        index must stay finite there (consumers drop non-finite values, so
        an inf would hide the worst point in a field), be at least 1, and
        agree between the scalar and field paths."""
        m = _LIBRARY.get("IM7_8552")
        crit = LaRC05Criterion()
        # |sigma_11| > G12: the misalignment denominator is non-positive.
        stress = np.array([-1.5 * m.G12, 0.0, 0.0, 0.0, 0.0, 0.0])
        res = crit.evaluate(stress, m)
        assert res.mode == "fiber_kinking"
        assert np.isfinite(res.index) and res.index >= 1.0
        assert res.index == pytest.approx(1.0 / res.reserve_factor, rel=1e-9)
        fi_field, _, rf_field = crit.evaluate_field(stress[None, :], m)
        assert fi_field[0] == res.index
        assert rf_field[0] == res.reserve_factor

    @pytest.mark.parametrize("name", _LIBRARY.list_names())
    def test_a_pristine_ply_kinks_exactly_at_Xc(self, name):
        """The defining property of LaRC kinking: with no wrinkle, pure
        fibre compression reaches FI = 1 at sigma_11 = -Xc. The previous
        model had no intrinsic misalignment and gave FI = 0 here, so a
        pristine ply could never fail in compression."""
        m = _LIBRARY.get(name)
        stress = np.array([-m.Xc, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = LaRC05Criterion().evaluate(stress, m)
        assert result.index == pytest.approx(1.0, abs=1e-12)
        assert result.detail["mode_fiber"] == "fiber_kinking"

    def test_the_intrinsic_misalignment_is_a_few_degrees(self, criterion):
        """phi_C follows from Xc, S_L and eta_L alone; for carbon and glass
        it lands in the few-degree range the literature reports."""
        for name in ("IM7_8552", "T700_2510", "AC318_S6C10"):
            phi_c = criterion.intrinsic_misalignment(_LIBRARY.get(name), 0.5)
            assert phi_c is not None
            assert 2.0 < np.degrees(phi_c) < 6.0

    def test_a_material_with_no_real_phi_c_falls_back_to_Xc(self, criterion):
        """Neat resin: shear strength too high relative to Xc for any angle
        to reproduce it, so the fibre-compression index is |s1| / Xc."""
        resin = _LIBRARY.get("EPOXY_S6C10")
        assert criterion.intrinsic_misalignment(resin, 0.5) is None
        stress = np.array([-0.7 * resin.Xc, 0.0, 0.0, 0.0, 0.0, 0.0])
        fi = LaRC05Criterion(ply_thickness=0.5).evaluate(stress, resin)
        assert fi.detail["fi_fiber"] == pytest.approx(0.7, abs=1e-12)

    def test_kinking_engages_and_grows_with_load(self, criterion, material):
        fi_lo = criterion.evaluate(
            np.array([-400.0, 0.0, 0.0, 0.0, 0.0, 0.0]), material
        ).index
        fi_hi = criterion.evaluate(
            np.array([-800.0, 0.0, 0.0, 0.0, 0.0, 0.0]), material
        ).index
        assert 0.0 < fi_lo < fi_hi

    @pytest.mark.parametrize("deg", [2, 5, 10])
    def test_a_wrinkle_in_the_stress_frame_matches_one_given_as_context(
        self, deg
    ):
        """Two ways to express the same out-of-plane misalignment must give
        the same compressive strength: the FE route (stress rotated into a
        fibre frame tilted about y, so the wrinkle appears as tau_13 and
        the kink-plane search must find it) and the API route (unrotated
        stress plus ``misalignment_angle``)."""
        m = _LIBRARY.get("AC318_S6C10")
        crit = LaRC05Criterion(ply_thickness=0.5)
        theta = np.radians(deg)
        T = stress_transformation_3d(theta, axis="y")

        def strength(fi_of_load):
            return brentq(lambda s0: fi_of_load(s0) - 1.0, 1.0, m.Xc)

        def fe_route(s0):
            return crit.evaluate(T @ np.array([-s0, 0, 0, 0, 0, 0.0]), m).index

        def api_route(s0):
            return crit.evaluate(
                np.array([-s0, 0, 0, 0, 0, 0.0]), m,
                {"misalignment_angle": theta},
            ).index

        assert strength(fe_route) == pytest.approx(strength(api_route), rel=2e-3)

    def test_strength_falls_with_misalignment(self):
        m = _LIBRARY.get("AC318_S6C10")
        crit = LaRC05Criterion(ply_thickness=0.5)
        strengths = [
            brentq(
                lambda s0, th=th: crit.evaluate(
                    np.array([-s0, 0, 0, 0, 0, 0.0]), m,
                    {"misalignment_angle": th},
                ).index - 1.0,
                1.0, 1.5 * m.Xc,
            )
            for th in np.radians([0.0, 2.0, 5.0, 10.0])
        ]
        assert strengths[0] == pytest.approx(m.Xc, rel=1e-9)
        assert all(a > b for a, b in zip(strengths, strengths[1:]))

    def test_higher_misalignment_increases_kinking_fi(
        self, criterion, material
    ):
        stress = np.array([-1000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        fi0 = criterion.evaluate(
            stress, material, {"misalignment_angle": 0.05}
        ).index
        fi1 = criterion.evaluate(
            stress, material, {"misalignment_angle": 0.15}
        ).index
        assert fi1 > fi0


# ======================================================================
# Matrix tension (Mode A) -- in-situ strength governs
# ======================================================================

class TestLaRC05MatrixTension:

    def test_transverse_tension_unity_at_in_situ_Yt(
        self, criterion, material
    ):
        """LaRC05 uses the *in-situ* transverse strength. For a thin ply
        without GIc/GIIc this is 1.12*sqrt(2)*Yt, so FI reaches 1.0 at
        sigma_22 = Yt_is (not at the raw Yt)."""
        Yt_is, _ = criterion._in_situ_strengths(material)
        assert Yt_is > material.Yt  # in-situ enhancement is real
        stress = np.array([0.0, Yt_is, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=0.01)
        assert result.mode == "matrix_tension"

    def test_transverse_tension_at_raw_Yt_below_unity(
        self, criterion, material
    ):
        """At the *raw* Yt the in-situ correction keeps FI below 1.0."""
        stress = np.array([0.0, material.Yt, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index < 1.0
        assert result.mode == "matrix_tension"


# ======================================================================
# Matrix compression (fracture-plane search) -- Mode B/C
# ======================================================================

class TestLaRC05MatrixCompression:

    def test_transverse_compression_unity_at_Yc(self, criterion, material):
        """sigma_22 = -Yc -> matrix-compression FI = 1.0 (the fracture
        plane angle calibration recovers the uniaxial allowable)."""
        stress = np.array([0.0, -material.Yc, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-6)
        assert result.mode == "matrix_compression"

    def test_transverse_compression_governing_fracture_angle(
        self, criterion, material
    ):
        """The fracture-plane search must select the angle |alpha| = alpha_0
        (~53 deg for CFRP) for pure transverse compression."""
        Yt_is, S12_is = criterion._in_situ_strengths(material)
        mu_L, mu_T = LaRC05Criterion._friction_coefficients(material)

        alpha_0_rad = np.radians(material.alpha_0)
        tan_2a = np.tan(2.0 * alpha_0_rad)
        S_T = material.Yc * np.cos(alpha_0_rad) * (
            np.sin(alpha_0_rad) + np.cos(alpha_0_rad) / tan_2a
        )

        thetas = np.linspace(-np.pi / 2, np.pi / 2, criterion.n_theta)
        ct, st = np.cos(thetas), np.sin(thetas)
        s2 = -material.Yc
        sigma_n = s2 * ct ** 2
        tau_nt = -s2 * st * ct
        fi = np.zeros(len(thetas))
        for i in range(len(thetas)):
            sn, tnt = sigma_n[i], tau_nt[i]
            if sn >= 0:
                fi[i] = (tnt / S_T) ** 2 + (sn / Yt_is) ** 2
            else:
                denom_t = S_T + mu_T * abs(sn)
                fi[i] = (tnt / denom_t) ** 2
        alpha_governing = abs(np.degrees(thetas[int(np.argmax(fi))]))
        assert alpha_governing == pytest.approx(material.alpha_0, abs=1.0)
        assert alpha_governing == pytest.approx(53.0, abs=1.0)

    def test_transverse_compression_below_Yc_under_unity(
        self, criterion, material
    ):
        stress = np.array([0.0, -0.75 * material.Yc, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.index < 1.0
        assert result.mode == "matrix_compression"


# ======================================================================
# In-plane shear
# ======================================================================

class TestLaRC05InPlaneShear:

    def test_pure_inplane_shear_at_S12(self, criterion, material):
        """Pure tau_12 = S12. With sigma_11 = 0 the fibre-tension branch is
        taken and its quadratic shear-interaction term gives
        FI = tau_12 / S12 = 1.0."""
        stress = np.array([0.0, 0.0, 0.0, 0.0, 0.0, material.S12])
        result = criterion.evaluate(stress, material)
        assert result.index == pytest.approx(1.0, abs=1e-10)

    def test_inplane_shear_scales_linearly(self, criterion, material):
        half = criterion.evaluate(
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * material.S12]),
            material,
        ).index
        assert half == pytest.approx(0.5, abs=1e-10)


# ======================================================================
# Monotonicity / proportional loading / reserve factor
# ======================================================================

class TestLaRC05MonotonicityAndReserve:

    def test_doubling_stress_increases_fi(self, criterion, material):
        stress = np.array([300.0, 20.0, 0.0, 0.0, 0.0, 30.0])
        fi1 = criterion.evaluate(stress, material).index
        fi2 = criterion.evaluate(2.0 * stress, material).index
        assert fi2 > fi1

    def test_proportional_scaling_is_linear_in_load(
        self, criterion, material
    ):
        """Every LaRC05 sub-FI is linear-in-load (issue #79), so a
        proportional load doubling exactly doubles FI."""
        stress = np.array([300.0, 20.0, 0.0, 0.0, 0.0, 30.0])
        fi1 = criterion.evaluate(stress, material).index
        fi2 = criterion.evaluate(2.0 * stress, material).index
        assert fi2 == pytest.approx(2.0 * fi1, rel=1e-9)

    def test_reserve_factor_scales_proportional_load_to_failure(
        self, criterion, material
    ):
        """Scaling a proportional load by its reserve factor must drive
        FI to exactly 1.0 (first failure)."""
        stress = np.array([300.0, 20.0, 0.0, 0.0, 0.0, 30.0])
        result = criterion.evaluate(stress, material)
        rf = result.reserve_factor
        at_failure = criterion.evaluate(rf * stress, material)
        assert at_failure.index == pytest.approx(1.0, rel=1e-9)

    def test_reserve_factor_is_inverse_of_index_for_linear_modes(
        self, criterion, material
    ):
        """Fibre tension and matrix tension keep rf = 1 / FI."""
        for stress in (
            np.array([0.5 * material.Xt, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.5 * material.Yt, 0.0, 0.0, 0.0, 0.0]),
        ):
            result = criterion.evaluate(stress, material)
            assert result.reserve_factor == pytest.approx(
                1.0 / result.index, rel=1e-12
            )

    def test_matrix_compression_reserve_factor_is_the_first_failure_load(
        self, criterion, material
    ):
        """Friction makes the matrix-compression index nonlinear in load,
        so 1 / FI is not the reserve factor. LaRC05 is calibrated so pure
        transverse compression fails at exactly Yc: at half of it the
        reserve is exactly 2 (1 / FI gives 1.77, under-reading it)."""
        stress = np.array([0.0, -0.5 * material.Yc, 0.0, 0.0, 0.0, 0.0])
        result = criterion.evaluate(stress, material)
        assert result.mode == "matrix_compression"
        assert result.reserve_factor == pytest.approx(2.0, rel=1e-9)
        assert 1.0 / result.index < 0.95 * result.reserve_factor
        at_rf = criterion.evaluate(result.reserve_factor * stress, material)
        assert at_rf.index == pytest.approx(1.0, rel=1e-9)

    def test_matrix_reserve_is_exact_under_friction(self, criterion, material):
        """Over random matrix-compression states: FI(rf * sigma) = 1, and
        the field path returns the same reserve factor bit for bit."""
        rng = np.random.default_rng(7)
        m = material
        checked = 0
        stresses = []
        for _ in range(200):
            s = np.array([
                rng.uniform(-0.1, 0.1) * m.Xc,
                -rng.uniform(0.0, 1.5) * m.Yc,
                -rng.uniform(0.0, 1.0) * m.Yc,
                rng.normal() * m.S23,
                rng.normal() * 0.5 * m.S12,
                rng.normal() * 0.5 * m.S12,
            ]) * rng.uniform(0.3, 2.0)
            r = criterion.evaluate(s, m)
            if r.mode != "matrix_compression":
                continue
            assert np.isfinite(r.reserve_factor)
            at_rf = criterion.evaluate(r.reserve_factor * s, m)
            assert at_rf.index == pytest.approx(1.0, rel=1e-8)
            stresses.append((s, r.reserve_factor))
            checked += 1
        assert checked > 30
        field = np.array([s for s, _ in stresses])
        _, _, rf_field = criterion.evaluate_field(field, m)
        np.testing.assert_array_equal(rf_field, [rf for _, rf in stresses])

    @pytest.mark.parametrize("load", [400.0, 1000.0, 3400.0])
    def test_kinking_reserve_factor_is_the_first_failure_load(
        self, criterion, material, load
    ):
        """Kinking is nonlinear in load, so its reserve factor is solved:
        FI(rf * sigma) = 1, and just below rf the ply has not failed.
        At 2.8 x Xc (3400 MPa) 1 / FI is not even close."""
        ctx = {"misalignment_angle": 0.05}
        stress = np.array([-load, 0.0, 0.0, 0.0, 0.0, 0.0])
        rf = criterion.evaluate(stress, material, ctx).reserve_factor
        assert criterion.evaluate(rf * stress, material, ctx).index == (
            pytest.approx(1.0, abs=1e-9)
        )
        assert criterion.evaluate(
            rf * (1.0 - 1e-6) * stress, material, ctx
        ).index < 1.0

    @staticmethod
    def _assert_never_unfails(names, angles, n_dirs, n_loads, seed=0):
        rng = np.random.default_rng(seed)
        crit = LaRC05Criterion(ply_thickness=0.5)
        for name in names:
            m = _LIBRARY.get(name)
            for deg in angles:
                T = stress_transformation_3d(np.radians(deg), axis="y")
                for _ in range(n_dirs):
                    g = np.array([-1.0, *rng.uniform(-0.15, 0.15, 5)])
                    loads = np.linspace(1.0, 8.0 * m.Xc, n_loads)
                    fi = crit.evaluate_field(
                        loads[:, None] * (T @ g)[None, :], m
                    )[0]
                    crossed = np.flatnonzero(fi >= 1.0)
                    if crossed.size:
                        assert np.all(fi[crossed[0]:] >= 1.0), (name, deg)

    def test_a_failed_state_never_reports_fi_below_one(self):
        """Safety property the reserve-factor solve relies on: under
        proportional loading, once FI reaches 1 it never drops back below 1
        (the closed-form misalignment is small-angle; past 45 degrees the
        band is reported kinked, FI = 1 / reserve factor). Representative subset here;
        the full sweep is the ``slow`` test below."""
        self._assert_never_unfails(
            ("IM7_8552", "AC318_S6C10"), (0, 20), n_dirs=1, n_loads=300
        )

    @pytest.mark.slow
    def test_a_failed_state_never_reports_fi_below_one_full_sweep(self):
        """Every library material, four wrinkle angles, random
        compression-dominated directions out to 8 x Xc."""
        self._assert_never_unfails(
            _LIBRARY.list_names(), (0, 5, 20, 40), n_dirs=2, n_loads=600
        )

    @pytest.mark.parametrize(
        "name, seed",
        [("IM7_8552", 2), ("AC318_S6C10", 5), ("T700_2510", 9)],
    )
    def test_the_plane_search_is_accurate_near_failure(self, name, seed):
        """The default kink-plane search (coarse grid + golden refinement of
        the two best planes) against a 1440-plane reference, on the
        governing index, for states near the failure threshold — where
        pass/fail and the reserve factor are decided.

        Measured: 99.9 % of states agree to ~4e-7; a rare narrow-lobe tail
        (transverse-shear-dominated states) under-reads by up to ~0.5 %.
        Both bounds are pinned so a regression in either shows up."""
        rng = np.random.default_rng(seed)
        m = _LIBRARY.get(name)
        stress = np.column_stack(
            [-rng.uniform(100, 1.3 * m.Xc, 6000), rng.normal(0, 60, (6000, 5))]
        )
        ref = LaRC05Criterion(n_psi=1440).evaluate_field_indices(stress, m)[0]
        fi = LaRC05Criterion().evaluate_field_indices(stress, m)[0]
        near = np.isfinite(ref) & (ref > 0.5) & (ref < 1.2)
        assert near.sum() > 500
        rel = np.abs(fi[near] - ref[near]) / ref[near]
        assert np.quantile(rel, 0.999) < 1e-5
        assert rel.max() < 1e-2

# ======================================================================
# Context override must not mutate criterion state (issue #192)
# ======================================================================

class TestLaRC05PlyThicknessOverrideIsLocal:
    """Pin the fix for #192: ``context['ply_thickness']`` must be treated
    as a local effective value.  It must not be written onto the criterion
    instance, otherwise the override silently leaks into subsequent calls
    that omit the override -- a call-order-dependent bug that also makes
    a single ``LaRC05Criterion`` instance unsafe to share across threads
    or across an ``evaluate_field`` sweep over a thick/thin ply mix.
    """

    def test_override_does_not_mutate_instance(self, criterion, material):
        original = criterion.ply_thickness
        stress = np.array([0.0, 0.5 * material.Yt, 0.0, 0.0, 0.0, 0.0])
        criterion.evaluate(stress, material, {"ply_thickness": 0.30})
        assert criterion.ply_thickness == original

    def test_override_then_no_context_matches_isolated_calls(
        self, criterion, material
    ):
        """A call with an override followed by a call without context must
        return the same FI as if each were called on a fresh criterion.
        Before the fix the second call inherited 0.30 mm via mutated
        ``self.ply_thickness``, so its in-situ strengths (and therefore
        the matrix FI) silently shifted."""
        stress = np.array([0.0, 0.5 * material.Yt, 0.0, 0.0, 0.0, 0.0])

        # Reference values from *isolated* (fresh-instance) calls.
        fi_with_override_ref = LaRC05Criterion().evaluate(
            stress, material, {"ply_thickness": 0.30}
        ).index
        fi_default_ref = LaRC05Criterion().evaluate(stress, material).index

        # Interleaved on a single shared instance.
        fi_with_override = criterion.evaluate(
            stress, material, {"ply_thickness": 0.30}
        ).index
        fi_default = criterion.evaluate(stress, material).index

        assert fi_with_override == pytest.approx(
            fi_with_override_ref, rel=1e-12
        )
        assert fi_default == pytest.approx(fi_default_ref, rel=1e-12)

    def test_override_does_not_corrupt_in_situ_branch_for_thick_ply(
        self, criterion, material
    ):
        """A thick-ply override (>= 2*t_ref) toggles the in-situ branch.
        Before the fix, calling with the thick override would leave
        ``self.ply_thickness`` thick, so a subsequent thin-ply call would
        misread the ``is_thin`` flag and skip the in-situ correction.
        """
        # 2 * t_ref triggers the thick-ply (no-correction) branch.
        t_thick = 2.0 * criterion.t_ref + 0.05
        stress = np.array([0.0, material.Yt, 0.0, 0.0, 0.0, 0.0])

        # Thick override -> uses raw Yt -> FI = 1.0 at sigma_22 = Yt.
        fi_thick = criterion.evaluate(
            stress, material, {"ply_thickness": t_thick}
        ).index
        assert fi_thick == pytest.approx(1.0, rel=1e-6)

        # Same criterion, no override -> default (thin) in-situ branch
        # must still apply -> FI strictly below 1.0 at sigma_22 = Yt.
        fi_default = criterion.evaluate(stress, material).index
        assert fi_default < 1.0

    def test_in_situ_strengths_accepts_thickness_parameter(
        self, criterion, material
    ):
        """The in-situ strengths must be derivable from an explicit
        effective-thickness argument, not only from instance state."""
        # Passing the instance value explicitly must reproduce the
        # default-argument result.
        Yt_default, S12_default = criterion._in_situ_strengths(material)
        Yt_param, S12_param = criterion._in_situ_strengths(
            material, criterion.ply_thickness
        )
        assert Yt_param == pytest.approx(Yt_default, rel=1e-12)
        assert S12_param == pytest.approx(S12_default, rel=1e-12)

        # Passing a thick override must switch off the thin-ply
        # correction (Yt_is collapses back to raw Yt).
        Yt_thick, _ = criterion._in_situ_strengths(
            material, 2.0 * criterion.t_ref + 0.05
        )
        assert Yt_thick == pytest.approx(material.Yt, rel=1e-12)
