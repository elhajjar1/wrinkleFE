"""Dedicated coverage for the two functions that physically inject a wrinkle
into the FE mesh (issue #86):

* :meth:`WrinkleConfiguration.apply_to_nodes` -- deforms node ``z`` by the
  wrinkle profile times a through-thickness decay factor.
* :meth:`WrinkleConfiguration.fiber_angles_at_nodes` -- per-node local fibre
  misalignment angle that feeds every downstream stress/strain/failure path.

These methods operate directly on ``(N, 3)`` node arrays and ``(N,)`` ply-id
arrays, so the tests build minimal arrays by hand (no FE solve -> < 1 s) and
assert *physically grounded* numbers (``assert_allclose`` / ``==``), never
shape/dtype only.

Conventions and the standard ``GaussianSinusoidal`` fixture (A=0.366,
lambda=16, w=12) are reused from ``tests/conftest.py`` to match the rest of
the suite.
"""

import numpy as np
import numpy.testing as npt
import pytest

from wrinklefe.core.morphology import (
    MORPHOLOGY_PHASES,
    WrinkleConfiguration,
    WrinklePlacement,
)
from wrinklefe.core.wrinkle import GaussianSinusoidal

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _strip(xs, n_plies):
    """Build a (n_plies * len(xs), 3) node strip + matching ply_ids.

    Node ``(p, x)`` sits at ``[x, 0, 0]`` with ply id ``p``; every ply
    sees the same x-stations so decay can be probed ply-by-ply.
    """
    nodes = []
    ply_ids = []
    for p in range(n_plies):
        for x in xs:
            nodes.append([x, 0.0, 0.0])
            ply_ids.append(p)
    return np.asarray(nodes, dtype=float), np.asarray(ply_ids, dtype=int)


def _single_config(profile, ply_interface, phase=0.0, **kw):
    return WrinkleConfiguration(
        [WrinklePlacement(profile, ply_interface=ply_interface, phase_offset=phase)],
        **kw,
    )


# ----------------------------------------------------------------------
# apply_to_nodes -- zero amplitude (identity)
# ----------------------------------------------------------------------

class TestApplyToNodesZeroAmplitude:
    """A zero-amplitude wrinkle must not move any node."""

    def test_zero_amplitude_is_identity(self):
        flat = GaussianSinusoidal(
            amplitude=0.0, wavelength=16.0, width=12.0, center=0.0
        )
        cfg = _single_config(flat, ply_interface=1)
        nodes, ply = _strip(np.linspace(-8.0, 8.0, 7), n_plies=4)
        out = cfg.apply_to_nodes(nodes, ply, n_plies=4)
        npt.assert_allclose(out, nodes, atol=1e-15)


# ----------------------------------------------------------------------
# apply_to_nodes -- known sinusoid profile at the interface
# ----------------------------------------------------------------------

class TestApplyToNodesKnownSinusoid:
    """At an interface ply the z-perturbation must equal the analytic
    wrinkle profile evaluated at that node's x, and the crest displacement
    must equal the amplitude."""

    def test_interface_z_matches_profile(self, gaussian_wrinkle):
        cfg = _single_config(gaussian_wrinkle, ply_interface=1)
        xs = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        nodes, ply = _strip(xs, n_plies=4)
        out = cfg.apply_to_nodes(nodes, ply, n_plies=4)
        dz = (out - nodes)[:, 2]
        expected = gaussian_wrinkle.displacement(xs)
        # Interface plies for k=1 in a 4-ply laminate are p=1 and p=2.
        for p in (1, 2):
            sel = ply == p
            npt.assert_allclose(dz[sel], expected, rtol=1e-12, atol=1e-14)

    def test_crest_displacement_equals_amplitude(self, gaussian_wrinkle):
        # x = center = 0 is the wrinkle crest; profile value there == A.
        cfg = _single_config(gaussian_wrinkle, ply_interface=1)
        nodes, ply = _strip(np.array([0.0]), n_plies=4)
        out = cfg.apply_to_nodes(nodes, ply, n_plies=4)
        dz = (out - nodes)[:, 2]
        crest = dz[ply == 1][0]
        npt.assert_allclose(crest, gaussian_wrinkle.amplitude, rtol=1e-12)
        npt.assert_allclose(
            crest,
            gaussian_wrinkle.displacement(np.array([0.0]))[0],
            rtol=1e-12,
        )

    def test_only_z_coordinate_modified(self, gaussian_wrinkle):
        cfg = _single_config(gaussian_wrinkle, ply_interface=1)
        nodes, ply = _strip(np.linspace(-6.0, 6.0, 5), n_plies=4)
        out = cfg.apply_to_nodes(nodes, ply, n_plies=4)
        npt.assert_array_equal(out[:, :2], nodes[:, :2])


# ----------------------------------------------------------------------
# apply_to_nodes -- decay / boundary condition through thickness
# ----------------------------------------------------------------------

class TestApplyToNodesDecayBC:
    """Default-mode decay: 1.0 at the two interface plies, 0.0 at the
    laminate outer surfaces, linear in between."""

    def test_outer_surfaces_undeflected(self, gaussian_wrinkle):
        cfg = _single_config(gaussian_wrinkle, ply_interface=3)  # k=3, 8 plies
        xs = np.linspace(-6.0, 6.0, 7)
        nodes, ply = _strip(xs, n_plies=8)
        out = cfg.apply_to_nodes(nodes, ply, n_plies=8)
        dz = (out - nodes)[:, 2]
        # Bottom (p=0) and top (p=n_plies-1=7) outer surfaces -> zero.
        npt.assert_allclose(dz[ply == 0], 0.0, atol=1e-15)
        npt.assert_allclose(dz[ply == 7], 0.0, atol=1e-15)

    def test_max_at_interface_plies(self, gaussian_wrinkle):
        cfg = _single_config(gaussian_wrinkle, ply_interface=3)
        nodes, ply = _strip(np.array([0.0]), n_plies=8)  # crest only
        out = cfg.apply_to_nodes(nodes, ply, n_plies=8)
        dz = (out - nodes)[:, 2]
        amp = gaussian_wrinkle.amplitude
        # Interface plies p=k=3 and p=k+1=4 carry full amplitude.
        npt.assert_allclose(dz[ply == 3][0], amp, rtol=1e-12)
        npt.assert_allclose(dz[ply == 4][0], amp, rtol=1e-12)
        # Every other ply must be strictly below the interface plies.
        for p in (0, 1, 2, 5, 6, 7):
            assert abs(dz[ply == p][0]) < amp

    def test_linear_decay_table(self, gaussian_wrinkle):
        """Intermediate plies follow the documented linear decay
        ``p/k`` (below) and ``(n-1-p)/(n-1-(k+1))`` (above)."""
        cfg = _single_config(gaussian_wrinkle, ply_interface=3)
        nodes, ply = _strip(np.array([0.0]), n_plies=8)  # crest, dz==A*decay
        out = cfg.apply_to_nodes(nodes, ply, n_plies=8)
        dz = (out - nodes)[:, 2]
        amp = gaussian_wrinkle.amplitude
        k, n = 3, 8
        for p in range(n):
            if p <= k:
                expected = p / k
            else:
                expected = (n - 1 - p) / ((n - 1) - (k + 1))
            npt.assert_allclose(
                dz[ply == p][0], amp * expected, rtol=1e-12, atol=1e-14
            )


# ----------------------------------------------------------------------
# _ply_decay -- direct unit coverage (issue #86 items 1-3)
# ----------------------------------------------------------------------

class TestPlyDecayHelper:
    """Direct tests of the shared ``_ply_decay`` table including its
    degenerate branches; these have real discriminating power (they fail
    if the n_plies<=1 guard is dropped or the top/bottom branches flip)."""

    def test_interior_table_n8_k3(self):
        n, k = 8, 3
        d = WrinkleConfiguration._ply_decay
        assert d(0, k, n) == 0.0
        assert d(n - 1, k, n) == 0.0
        assert d(k, k, n) == 1.0
        assert d(k + 1, k, n) == 1.0
        # Monotone linear on each side.
        below = [d(p, k, n) for p in range(0, k + 1)]
        above = [d(p, k, n) for p in range(k + 1, n)]
        assert below == sorted(below)
        assert above == sorted(above, reverse=True)
        npt.assert_allclose(d(1, k, n), 1 / 3, rtol=1e-12)
        npt.assert_allclose(d(2, k, n), 2 / 3, rtol=1e-12)

    def test_degenerate_bottom_surface(self):
        """Wrinkle at the bottom outer surface (k=0).

        Documented contract for the actual code (the `_ply_decay`
        docstring's "1.0 only on the interface ply, 0.0 elsewhere" wording
        only holds for the ``p <= k`` side): the special ``k <= 0`` branch
        gives ``d(0)=1``; for ``p > k`` the normal top-side ramp applies,
        ``(n-1-p)/((n-1)-(k+1))``, so the field decays linearly 1.0 -> 0.0
        and is *not* zero on the plies just above the interface.
        """
        n, k = 6, 0
        d = WrinkleConfiguration._ply_decay
        assert d(0, k, n) == 1.0  # interface ply (special branch)
        # p > k side is the normal linear ramp, NOT all-zero.
        expected = [1.0, 1.0, 0.75, 0.5, 0.25, 0.0]
        for p in range(n):
            npt.assert_allclose(d(p, k, n), expected[p], rtol=1e-12)
        # Outer surface still pinned to zero.
        assert d(n - 1, k, n) == 0.0

    def test_degenerate_top_surface(self):
        """Wrinkle at the top outer surface (k = n_plies - 2).

        Symmetric to the bottom case: the special ``denom <= 0`` branch
        gives ``d(k+1)=1`` and the ``p <= k`` side is the normal linear
        ramp ``p/k``; the field decays linearly 0.0 -> 1.0 and is not
        all-zero away from the interface ply.
        """
        n = 6
        k = n - 2  # k + 1 == n - 1 -> top outer surface
        d = WrinkleConfiguration._ply_decay
        assert d(k + 1, k, n) == 1.0  # top interface ply (special branch)
        expected = [0.0, 0.25, 0.5, 0.75, 1.0, 1.0]
        for p in range(n):
            npt.assert_allclose(d(p, k, n), expected[p], rtol=1e-12)
        # Bottom outer surface still pinned to zero.
        assert d(0, k, n) == 0.0

    def test_single_ply_no_divide_by_zero(self):
        """n_plies == 1 must short-circuit to full amplitude, not 0/0."""
        assert WrinkleConfiguration._ply_decay(0, 0, 1) == 1.0


# ----------------------------------------------------------------------
# apply_to_nodes -- uniform decay mode
# ----------------------------------------------------------------------

class TestApplyToNodesUniformMode:
    """``uniform`` mode: every ply displaced by the full profile,
    independent of p."""

    def test_uniform_mode_full_profile_all_plies(self, gaussian_wrinkle):
        cfg = _single_config(
            gaussian_wrinkle, ply_interface=1, decay_mode="uniform"
        )
        xs = np.array([-4.0, 0.0, 3.0])
        n_plies = 5
        nodes, ply = _strip(xs, n_plies=n_plies)
        out = cfg.apply_to_nodes(nodes, ply, n_plies=n_plies)
        dz = (out - nodes)[:, 2]
        expected = gaussian_wrinkle.displacement(xs)
        for p in range(n_plies):
            npt.assert_allclose(
                dz[ply == p], expected, rtol=1e-12, atol=1e-14
            )


# ----------------------------------------------------------------------
# graded mode -- displacement/angle decay parity (issues #32 / #40)
# ----------------------------------------------------------------------

class TestGradedDecayParity:
    """Regression for the asymmetric clamp-order drift flagged in
    #32 / #40: with a *fully spanning* ply_ids array, the decay implicit
    in ``apply_to_nodes`` (dz / profile(x)) must equal the decay implicit
    in ``fiber_angles_at_nodes`` across a sweep of ``decay_floor`` and
    ply index. Since #252 the angle is the slope of the composed field
    (``arctan(decay * |slope|)``), so the angle-side decay is recovered
    in slope space: ``tan(angle) / |slope|``."""

    @pytest.mark.parametrize("decay_floor", [0.0, 0.25, 0.5, 1.0])
    @pytest.mark.parametrize("n_plies", [4, 7, 8])
    def test_graded_disp_angle_decay_match(self, decay_floor, n_plies):
        prof = GaussianSinusoidal(
            amplitude=0.366, wavelength=16.0, width=12.0, center=0.0
        )
        cfg = _single_config(
            prof,
            ply_interface=n_plies // 2,
            decay_mode="graded",
            decay_floor=decay_floor,
        )
        x = 2.0  # non-zero slope and non-zero displacement here
        prof_val = prof.displacement(np.array([x]))[0]
        slope_base = np.abs(prof.slope(np.array([x]))[0])

        # ply_ids spans the full laminate so fiber_angles_at_nodes infers
        # the same n_plies that apply_to_nodes is given (see issue #146).
        nodes, ply = _strip(np.array([x]), n_plies=n_plies)
        dz = (cfg.apply_to_nodes(nodes, ply, n_plies=n_plies) - nodes)[:, 2]
        ang = cfg.fiber_angles_at_nodes(nodes, ply)

        decay_from_disp = dz / prof_val
        decay_from_slope = np.tan(ang) / slope_base
        npt.assert_allclose(
            decay_from_disp, decay_from_slope, rtol=1e-12, atol=1e-14
        )


# ----------------------------------------------------------------------
# fiber_angles_at_nodes -- baseline (zero amplitude -> zero misalignment)
# ----------------------------------------------------------------------

class TestFiberAnglesBaseline:
    """``fiber_angles_at_nodes`` returns the *misalignment* (deviation from
    the nominal ply direction). With zero amplitude there is no waviness,
    so every node's misalignment angle is exactly zero -- i.e. fibres lie
    along their nominal ply angle everywhere."""

    def test_zero_amplitude_zero_misalignment(self):
        flat = GaussianSinusoidal(
            amplitude=0.0, wavelength=16.0, width=12.0, center=0.0
        )
        cfg = _single_config(flat, ply_interface=2)
        nodes, ply = _strip(np.linspace(-8.0, 8.0, 9), n_plies=6)
        out = cfg.fiber_angles_at_nodes(nodes, ply)
        npt.assert_allclose(out, 0.0, atol=1e-15)
        assert out.shape == (nodes.shape[0],)


# ----------------------------------------------------------------------
# fiber_angles_at_nodes -- sign / magnitude vs analytic slope & max_angle
# ----------------------------------------------------------------------

class TestFiberAnglesSlopeMagnitude:
    """At an interface ply the angle equals ``arctan|dz/dx|``; its peak
    over the domain equals ``profile.max_angle()``; output is always
    non-negative (RSS of |arctan(slope)| contributions)."""

    def test_angle_equals_arctan_abs_slope(self, gaussian_wrinkle):
        cfg = _single_config(gaussian_wrinkle, ply_interface=1)
        xs = np.array([1.0, 2.0, 3.0, 5.0])
        nodes, ply = _strip(xs, n_plies=4)
        ang = cfg.fiber_angles_at_nodes(nodes, ply)
        expected = np.arctan(np.abs(gaussian_wrinkle.slope(xs)))
        npt.assert_allclose(ang[ply == 1], expected, rtol=1e-12, atol=1e-14)

    def test_peak_angle_matches_profile_max_angle(self, gaussian_wrinkle):
        """Sampling the steepest part of the wrinkle at an interface ply
        recovers ``WrinkleProfile.max_angle()`` to grid tolerance."""
        cfg = _single_config(gaussian_wrinkle, ply_interface=1)
        xlo, xhi = gaussian_wrinkle.domain()
        xs = np.linspace(xlo, xhi, 4097)
        nodes, ply = _strip(xs, n_plies=4)
        ang = cfg.fiber_angles_at_nodes(nodes, ply)
        peak = ang[ply == 1].max()
        npt.assert_allclose(
            peak, gaussian_wrinkle.max_angle(), rtol=1e-4, atol=1e-6
        )

    def test_all_angles_non_negative_fuzz(self):
        """Fuzz over profile / decay_floor / ply_ids: every returned angle
        must be >= 0 (pins the non-negativity invariant)."""
        rng = np.random.default_rng(20260516)
        for _ in range(25):
            amp = float(rng.uniform(0.0, 1.0))
            lam = float(rng.uniform(4.0, 24.0))
            w = float(rng.uniform(4.0, 20.0))
            prof = GaussianSinusoidal(
                amplitude=amp, wavelength=lam, width=w, center=0.0
            )
            n_plies = int(rng.integers(2, 12))
            k = int(rng.integers(0, n_plies - 1))
            floor = float(rng.uniform(0.0, 1.0))
            mode = rng.choice(["default", "uniform", "graded"])
            cfg = _single_config(
                prof, ply_interface=k, decay_mode=mode, decay_floor=floor
            )
            xs = rng.uniform(-3.0 * w, 3.0 * w, size=7)
            nodes, ply = _strip(xs, n_plies=n_plies)
            out = cfg.fiber_angles_at_nodes(nodes, ply)
            assert np.all(out >= 0.0), (
                f"negative angle for amp={amp}, lam={lam}, mode={mode}"
            )
            assert np.all(np.isfinite(out))


# ----------------------------------------------------------------------
# fiber_angles_at_nodes -- dual-wrinkle RSS combination
# ----------------------------------------------------------------------

class TestFiberAnglesDualWrinkleComposed:
    """Angles derive from the composed displacement field (#252).

    Per the amplitude contract (#305), ``dual_wrinkle`` places an ``A/2``
    clone of the profile at each interface so the in-phase composed mesh
    sums to exactly the configured amplitude ``A`` rather than ``2A``.
    Two identical co-located wrinkles (phase=0) therefore reproduce the
    single full-amplitude slope, and the co-located dual angle equals the
    single-wrinkle angle. With phase=pi the halved fields partly cancel.
    """

    def test_phase0_colocated_matches_single_full_amplitude(
        self, gaussian_wrinkle
    ):
        """Issue #305: two co-located in-phase A/2 wrinkles compose to the
        configured amplitude A, so the dual fibre angle equals the single
        full-amplitude wrinkle's angle (not twice its slope)."""
        single = _single_config(gaussian_wrinkle, ply_interface=2)
        dual = WrinkleConfiguration.dual_wrinkle(
            gaussian_wrinkle, interface1=2, interface2=2, phase=0.0
        )
        nodes, ply = _strip(np.array([2.0]), n_plies=6)
        a_single = single.fiber_angles_at_nodes(nodes, ply)[ply == 2][0]
        a_dual = dual.fiber_angles_at_nodes(nodes, ply)[ply == 2][0]
        npt.assert_allclose(a_dual, a_single, rtol=1e-12)

    def test_phase_pi_partially_cancels_displacement(self, gaussian_wrinkle):
        """Anti-stack (phase=pi) shifts the 2nd wrinkle by lambda/2, so the
        co-located dual displacement is the sum of the two A/2 clones,
        profile_half(x) + profile_half(x - lambda/2); at the crest these are
        opposite-sign and the net is strictly smaller than the single
        full-amplitude crest."""
        single = _single_config(gaussian_wrinkle, ply_interface=2)
        dual = WrinkleConfiguration.dual_wrinkle(
            gaussian_wrinkle, interface1=2, interface2=2, phase=np.pi
        )
        nodes, ply = _strip(np.array([0.0]), n_plies=6)  # crest
        dz_single = (single.apply_to_nodes(nodes, ply, 6) - nodes)[:, 2]
        dz_dual = (dual.apply_to_nodes(nodes, ply, 6) - nodes)[:, 2]
        s = dz_single[ply == 2][0]
        d = dz_dual[ply == 2][0]
        # Net dual displacement strictly below the single full crest
        # (partial cancellation of the two halved, phase-shifted fields).
        assert abs(d) < abs(s)
        # Closed-form: each clone carries A/2, so the composed crest is
        # 0.5 * (profile(0) + profile(0 - lambda/2)).
        lam = gaussian_wrinkle.wavelength
        expected = 0.5 * (
            gaussian_wrinkle.displacement(np.array([0.0]))[0]
            + gaussian_wrinkle.displacement(np.array([-lam / 2.0]))[0]
        )
        npt.assert_allclose(d, expected, rtol=1e-12)


# ----------------------------------------------------------------------
# determinism, shape, non-mutation contract
# ----------------------------------------------------------------------

class TestDeterminismShapeMutation:
    """Output shape == input shape; calling twice is bit-identical; the
    input array is NOT mutated in place (apply_to_nodes copies)."""

    def test_apply_to_nodes_shape_and_determinism(self, gaussian_wrinkle):
        cfg = _single_config(gaussian_wrinkle, ply_interface=1)
        nodes, ply = _strip(np.linspace(-5.0, 5.0, 6), n_plies=4)
        a = cfg.apply_to_nodes(nodes, ply, n_plies=4)
        b = cfg.apply_to_nodes(nodes, ply, n_plies=4)
        assert a.shape == nodes.shape
        npt.assert_array_equal(a, b)

    def test_apply_to_nodes_does_not_mutate_input(self, gaussian_wrinkle):
        cfg = _single_config(gaussian_wrinkle, ply_interface=1)
        nodes, ply = _strip(np.linspace(-5.0, 5.0, 6), n_plies=4)
        original = nodes.copy()
        cfg.apply_to_nodes(nodes, ply, n_plies=4)
        npt.assert_array_equal(nodes, original)

    def test_fiber_angles_shape_and_determinism(self, gaussian_wrinkle):
        cfg = _single_config(gaussian_wrinkle, ply_interface=1)
        nodes, ply = _strip(np.linspace(-5.0, 5.0, 6), n_plies=4)
        a = cfg.fiber_angles_at_nodes(nodes, ply)
        b = cfg.fiber_angles_at_nodes(nodes, ply)
        assert a.shape == (nodes.shape[0],)
        npt.assert_array_equal(a, b)


# ----------------------------------------------------------------------
# phase vs named morphology equivalence
# ----------------------------------------------------------------------

class TestPhaseVsNamedMorphology:
    """A numeric ``phase`` must reproduce the corresponding named
    morphology preset (the sweep-phase equivalence), and the displacement
    + angle fields must agree element-for-element."""

    @pytest.mark.parametrize("name", ["stack", "convex", "concave"])
    def test_numeric_phase_matches_named_preset(self, gaussian_wrinkle, name):
        phase = MORPHOLOGY_PHASES[name]
        named = WrinkleConfiguration.from_morphology_name(
            name, gaussian_wrinkle, interface1=1, interface2=2
        )
        numeric = WrinkleConfiguration.dual_wrinkle(
            gaussian_wrinkle, interface1=1, interface2=2, phase=phase
        )
        xs = np.linspace(-8.0, 8.0, 17)
        nodes, ply = _strip(xs, n_plies=4)

        npt.assert_allclose(
            numeric.apply_to_nodes(nodes, ply, 4),
            named.apply_to_nodes(nodes, ply, 4),
            rtol=1e-12,
            atol=1e-14,
        )
        npt.assert_allclose(
            numeric.fiber_angles_at_nodes(nodes, ply),
            named.fiber_angles_at_nodes(nodes, ply),
            rtol=1e-12,
            atol=1e-14,
        )
        # Pairwise phase recovered from the numeric config == preset phase.
        npt.assert_allclose(
            numeric.pairwise_phases()[0], phase, atol=1e-14
        )

    def test_phase_none_default_keeps_stack_behavior(self, gaussian_wrinkle):
        """`dual_wrinkle` default phase (0.0) == the 'stack' named preset."""
        default = WrinkleConfiguration.dual_wrinkle(
            gaussian_wrinkle, interface1=1, interface2=2
        )
        stack = WrinkleConfiguration.from_morphology_name(
            "stack", gaussian_wrinkle, interface1=1, interface2=2
        )
        nodes, ply = _strip(np.linspace(-6.0, 6.0, 9), n_plies=4)
        npt.assert_allclose(
            default.apply_to_nodes(nodes, ply, 4),
            stack.apply_to_nodes(nodes, ply, 4),
            rtol=1e-12,
            atol=1e-14,
        )
        npt.assert_allclose(default.pairwise_phases()[0], 0.0, atol=1e-14)


# ----------------------------------------------------------------------
# KNOWN BUG (issue #146): partial ply_ids desyncs the two decay fields
# ----------------------------------------------------------------------

def test_partial_ply_ids_decay_stays_synced():
    prof = GaussianSinusoidal(
        amplitude=0.366, wavelength=16.0, width=12.0, center=0.0
    )
    cfg = _single_config(prof, ply_interface=3)  # true laminate: 8 plies
    x = 2.0
    prof_val = prof.displacement(np.array([x]))[0]
    ang_base = np.arctan(np.abs(prof.slope(np.array([x]))[0]))

    # Caller passes nodes for plies 0..5 only; top ply (7) absent.
    ply = np.array([0, 1, 2, 3, 4, 5])
    nodes = np.array([[x, 0.0, 0.0] for _ in ply], float)

    dz = (cfg.apply_to_nodes(nodes, ply, n_plies=8) - nodes)[:, 2]
    ang = cfg.fiber_angles_at_nodes(nodes, ply, n_plies=8)

    decay_from_disp = dz / prof_val
    # Slope-space decay recovery (composed-field angles since #252).
    decay_from_angle = np.tan(ang) / np.tan(ang_base)
    # Once #146 is fixed these two decay fields will match.
    npt.assert_allclose(
        decay_from_disp, decay_from_angle, rtol=1e-12, atol=1e-14
    )


# ----------------------------------------------------------------------
# Issues #17 / #18: BC vanishes at outer surfaces AND displacement and
# fibre-angle fields share the same default-mode decay.
# ----------------------------------------------------------------------

# Valid (k, n_plies) pairs for the parametrize below: ``k`` is a ply-
# interface index and must satisfy ``k < n_plies - 1`` so that both
# interface plies (k, k+1) exist and the upper outer surface (n_plies-1)
# sits strictly above them. Filtering at collection time means the test
# count reflects real coverage instead of being inflated by skipped
# combinations (issue #205).
_K_VALUES_OUTER = [1, 2, 3, 5, 8]
_K_VALUES_INTERFACE = [1, 2, 3, 5]
_N_PLIES_VALUES = [4, 8, 12]

_VALID_K_N_OUTER = [
    (k, n) for n in _N_PLIES_VALUES for k in _K_VALUES_OUTER if k < n - 1
]
_VALID_K_N_INTERFACE = [
    (k, n) for n in _N_PLIES_VALUES for k in _K_VALUES_INTERFACE if k < n - 1
]


class TestIssue1718DefaultDecayBC:
    """Regression tests pinning the contract from issues #17 and #18.

    Issue #17: in default decay mode the through-thickness displacement
    must vanish at BOTH outer surfaces (``p == 0`` and ``p == n_plies-1``)
    and be unity at the interface plies (``p == k`` and ``p == k+1``).

    Issue #18: ``apply_to_nodes`` and ``fiber_angles_at_nodes`` must apply
    the SAME decay function in default mode -- so a ply whose geometric
    displacement is zero must also see zero misalignment-angle decay.
    """

    @pytest.mark.parametrize("k,n_plies", _VALID_K_N_OUTER)
    def test_outer_surfaces_zero_decay(self, k, n_plies):
        """#17: bottom (p=0) and top (p=n-1) outer surfaces -> decay=0."""
        prof = GaussianSinusoidal(
            amplitude=0.366, wavelength=16.0, width=12.0, center=0.0
        )
        cfg = _single_config(prof, ply_interface=k)
        # Crest -> dz at any ply == amplitude * decay.
        nodes, ply = _strip(np.array([0.0]), n_plies=n_plies)
        out = cfg.apply_to_nodes(nodes, ply, n_plies=n_plies)
        dz = (out - nodes)[:, 2]
        # The two surfaces are explicitly zero.
        if k > 0:
            npt.assert_allclose(dz[ply == 0], 0.0, atol=1e-15)
        if k + 1 < n_plies - 1:
            npt.assert_allclose(dz[ply == n_plies - 1], 0.0, atol=1e-15)

    @pytest.mark.parametrize("k,n_plies", _VALID_K_N_INTERFACE)
    def test_interface_plies_unit_decay(self, k, n_plies):
        """#17: interface plies (p=k and p=k+1) carry the full profile."""
        prof = GaussianSinusoidal(
            amplitude=0.366, wavelength=16.0, width=12.0, center=0.0
        )
        cfg = _single_config(prof, ply_interface=k)
        nodes, ply = _strip(np.array([0.0]), n_plies=n_plies)
        out = cfg.apply_to_nodes(nodes, ply, n_plies=n_plies)
        dz = (out - nodes)[:, 2]
        amp = prof.amplitude
        npt.assert_allclose(dz[ply == k][0], amp, rtol=1e-12)
        npt.assert_allclose(dz[ply == k + 1][0], amp, rtol=1e-12)

    @pytest.mark.parametrize("k,n_plies", _VALID_K_N_INTERFACE)
    def test_displacement_and_angle_share_decay_default_mode(
        self, k, n_plies
    ):
        """#18: at every ply, decay extracted from the displacement field
        must equal decay extracted from the fibre-angle field. In
        particular: the outer-surface plies that get zero displacement
        must also get zero angle (no inflated misalignment downstream)."""
        prof = GaussianSinusoidal(
            amplitude=0.366, wavelength=16.0, width=12.0, center=0.0
        )
        cfg = _single_config(prof, ply_interface=k)
        # Pick a non-symmetric x where both profile value and slope are
        # non-zero so we can divide cleanly.
        x = 2.5
        prof_val = prof.displacement(np.array([x]))[0]
        ang_base = np.arctan(np.abs(prof.slope(np.array([x]))[0]))

        nodes, ply = _strip(np.array([x]), n_plies=n_plies)
        dz = (cfg.apply_to_nodes(nodes, ply, n_plies=n_plies) - nodes)[:, 2]
        ang = cfg.fiber_angles_at_nodes(nodes, ply, n_plies=n_plies)

        decay_from_disp = dz / prof_val
        # Since #252 the angle is the slope of the composed field
        # (arctan(decay * |slope|)), so the angle-side decay is
        # recovered in slope space.
        decay_from_angle = np.tan(ang) / np.tan(ang_base)
        npt.assert_allclose(
            decay_from_disp,
            decay_from_angle,
            rtol=1e-12,
            atol=1e-14,
        )
        # Surface plies (when not degenerate) must be exactly zero in BOTH
        # fields -- this is what #18 was about.
        if k > 0:
            assert dz[ply == 0][0] == 0.0
            assert ang[ply == 0][0] == 0.0
        if k + 1 < n_plies - 1:
            assert dz[ply == n_plies - 1][0] == 0.0
            assert ang[ply == n_plies - 1][0] == 0.0


# ----------------------------------------------------------------------
# Issue #159: pin the behavioural difference between ``stack`` and
# ``uniform`` so they cannot be accidentally homogenised.
# ----------------------------------------------------------------------

class TestIssue159StackVsUniform:
    """``stack`` and ``uniform`` both report M_f = 1.0, but they model
    fundamentally different defects:

    * ``stack``  — DUAL wrinkle (φ = 0, two adjacent interfaces),
                   default through-thickness decay (linear taper from
                   the interface plies to zero at the outer surfaces).
    * ``uniform`` — SINGLE wrinkle, ``decay_mode="uniform"`` (every ply
                    carries the full profile, including surfaces).

    Both come from :meth:`WrinkleConfiguration.from_morphology_name`, so
    a regression that conflated either choice (wrinkle count *or* decay
    mode) would slip past the existing morphology tests. These checks
    pin both axes simultaneously.
    """

    def _build_pair(self, profile, interface1=2, interface2=3, n_plies=6):
        stack = WrinkleConfiguration.from_morphology_name(
            "stack", profile, interface1=interface1, interface2=interface2
        )
        uniform = WrinkleConfiguration.from_morphology_name(
            "uniform", profile, interface1=interface1, interface2=interface2
        )
        return stack, uniform

    def test_wrinkle_count_differs(self, gaussian_wrinkle):
        """The two morphologies must NOT have the same number of wrinkles."""
        stack, uniform = self._build_pair(gaussian_wrinkle)
        assert stack.n_wrinkles() == 2, "stack must be a dual-wrinkle preset"
        assert uniform.n_wrinkles() == 1, "uniform must be a single-wrinkle preset"

    def test_decay_mode_differs(self, gaussian_wrinkle):
        """The two morphologies must NOT share the same through-thickness mode."""
        stack, uniform = self._build_pair(gaussian_wrinkle)
        # ``from_morphology_name`` defaults dual-wrinkle modes to the
        # standard linear taper and explicitly tags ``uniform`` so the
        # decay-mode label is the canonical regression check.
        assert stack.decay_mode == "default"
        assert uniform.decay_mode == "uniform"

    def test_deformed_meshes_differ(self, gaussian_wrinkle):
        """``apply_to_nodes`` outputs must differ on a multi-ply mesh."""
        n_plies = 6
        stack, uniform = self._build_pair(
            gaussian_wrinkle, interface1=2, interface2=3, n_plies=n_plies
        )
        xs = np.linspace(-8.0, 8.0, 9)
        nodes, ply = _strip(xs, n_plies=n_plies)

        dz_stack = (stack.apply_to_nodes(nodes, ply, n_plies) - nodes)[:, 2]
        dz_uniform = (uniform.apply_to_nodes(nodes, ply, n_plies) - nodes)[:, 2]

        # Any reasonable mesh must register a non-trivial difference.
        diff = np.abs(dz_stack - dz_uniform)
        assert diff.max() > 1e-6, (
            "stack and uniform produced identical mesh displacements -- "
            "they should differ either in wrinkle count or decay mode."
        )

    def test_per_ply_decay_vectors_differ(self, gaussian_wrinkle):
        """The through-thickness decay vector that scales each ply must
        differ in *expected* ways: uniform = 1 on every ply (no decay);
        stack tapers to 0 at the outer surfaces."""
        n_plies = 6
        k = 2  # stack picks (interface1+interface2)//2 == (2+3)//2 == 2
        stack, uniform = self._build_pair(
            gaussian_wrinkle, interface1=2, interface2=3, n_plies=n_plies
        )
        ply_ids = np.arange(n_plies, dtype=np.int64)

        # ``stack`` is dual-wrinkle; query the decay for its first wrinkle
        # which sits at interface ``k=2`` (matches uniform's mid_interface).
        stack_decay = stack._through_thickness_decay(ply_ids, k, n_plies)
        uniform_decay = uniform._through_thickness_decay(ply_ids, k, n_plies)

        # Uniform: every ply carries the full profile.
        npt.assert_allclose(uniform_decay, np.ones(n_plies), atol=1e-15)

        # Stack (default linear taper): outer surfaces zero, interface
        # plies (p == k, p == k + 1) unity.
        assert stack_decay[0] == 0.0
        assert stack_decay[n_plies - 1] == 0.0
        npt.assert_allclose(stack_decay[k], 1.0, rtol=1e-12)
        npt.assert_allclose(stack_decay[k + 1], 1.0, rtol=1e-12)

        # And the vectors must NOT be elementwise equal.
        assert not np.allclose(stack_decay, uniform_decay), (
            "stack and uniform produced identical through-thickness decay "
            "vectors -- a regression has homogenised the two morphologies."
        )

    def test_outer_surface_signature(self, gaussian_wrinkle):
        """The cleanest physical fingerprint: in ``uniform`` the outer
        surface plies see the full profile, in ``stack`` they see zero.
        A regression that swapped decay modes would flip this."""
        n_plies = 6
        stack, uniform = self._build_pair(
            gaussian_wrinkle, interface1=2, interface2=3, n_plies=n_plies
        )
        # Sample the crest (x == center == 0) so dz == amplitude * decay.
        nodes, ply = _strip(np.array([0.0]), n_plies=n_plies)
        dz_stack = (stack.apply_to_nodes(nodes, ply, n_plies) - nodes)[:, 2]
        dz_uniform = (uniform.apply_to_nodes(nodes, ply, n_plies) - nodes)[:, 2]

        amp = gaussian_wrinkle.amplitude
        # Bottom outer surface (p == 0)
        npt.assert_allclose(dz_stack[ply == 0][0], 0.0, atol=1e-15)
        npt.assert_allclose(dz_uniform[ply == 0][0], amp, rtol=1e-12)
        # Top outer surface (p == n_plies - 1)
        npt.assert_allclose(dz_stack[ply == n_plies - 1][0], 0.0, atol=1e-15)
        npt.assert_allclose(dz_uniform[ply == n_plies - 1][0], amp, rtol=1e-12)
