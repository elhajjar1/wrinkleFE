"""Spatially varying in-plane amplitude profile tests (issue #3).

The :class:`WrinkleConfiguration` accepts an ``amplitude_profile`` kwarg
that multiplies the wrinkle's *A* by a position-dependent scale along the
chosen in-plane axis. Tests here cover:

- the three profile kinds (``constant``, ``gaussian``, ``linear``) at the
  centre, at one decay-length, and beyond,
- the #18 invariant -- where the displacement scale vanishes the local
  fibre-angle deviation also vanishes,
- validation of bad inputs,
- a smoke test that ``apply_to_nodes`` runs end-to-end for every kind.

The standard ``gaussian_wrinkle`` fixture (A=0.366, lambda=16, w=12, x0=0)
from ``tests/conftest.py`` is reused throughout.
"""

import math

import numpy as np
import numpy.testing as npt
import pytest

from wrinklefe.core.morphology import (
    WrinkleConfiguration,
    WrinklePlacement,
)
from wrinklefe.core.wrinkle import GaussianSinusoidal

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _strip(xs, n_plies, y=0.0):
    """Build a (n_plies * len(xs), 3) node strip + matching ply_ids."""
    nodes = []
    ply_ids = []
    for p in range(n_plies):
        for x in xs:
            nodes.append([x, y, 0.0])
            ply_ids.append(p)
    return np.asarray(nodes, dtype=float), np.asarray(ply_ids, dtype=int)


def _single_config(profile, ply_interface=1, phase=0.0, **kw):
    return WrinkleConfiguration(
        [WrinklePlacement(profile, ply_interface=ply_interface, phase_offset=phase)],
        **kw,
    )


# ----------------------------------------------------------------------
# Constant profile -- backwards-compat baseline
# ----------------------------------------------------------------------

class TestConstantProfileBaseline:
    """``amplitude_profile='constant'`` (the default) must leave the
    legacy displacement field untouched."""

    def test_default_matches_legacy(self, gaussian_wrinkle):
        cfg_default = _single_config(gaussian_wrinkle, ply_interface=1)
        cfg_const = _single_config(
            gaussian_wrinkle, ply_interface=1, amplitude_profile="constant"
        )
        xs = np.linspace(-8.0, 8.0, 9)
        nodes, ply = _strip(xs, n_plies=4)
        out_default = cfg_default.apply_to_nodes(nodes, ply, n_plies=4)
        out_const = cfg_const.apply_to_nodes(nodes, ply, n_plies=4)
        npt.assert_allclose(out_default, out_const, rtol=1e-14, atol=1e-15)

    def test_constant_scale_is_unity_everywhere(self, gaussian_wrinkle):
        """At every probed (x, y) the internal scale must be exactly 1.0."""
        cfg = _single_config(
            gaussian_wrinkle, ply_interface=1, amplitude_profile="constant"
        )
        wrinkle = cfg.wrinkles[0]
        for x in (-20.0, -5.0, 0.0, 3.7, 50.0):
            for y in (-5.0, 0.0, 5.0):
                assert cfg._amplitude_scale(wrinkle, x, y) == 1.0


# ----------------------------------------------------------------------
# Gaussian profile -- decay-law check
# ----------------------------------------------------------------------

class TestGaussianProfile:
    """``A_eff(s) = A * exp(-(s/d)**2)``."""

    def test_center_unity(self, gaussian_wrinkle):
        cfg = _single_config(
            gaussian_wrinkle,
            ply_interface=1,
            amplitude_profile="gaussian",
            amplitude_profile_decay_length=4.0,
            amplitude_profile_axis="x",
        )
        wrinkle = cfg.wrinkles[0]
        # Wrinkle centre is at profile.center=0 (no phase offset),
        # so s=0 -> scale = exp(0) = 1.
        npt.assert_allclose(
            cfg._amplitude_scale(wrinkle, 0.0, 0.0), 1.0, rtol=1e-14
        )

    def test_one_decay_length(self, gaussian_wrinkle):
        d = 5.0
        cfg = _single_config(
            gaussian_wrinkle,
            ply_interface=1,
            amplitude_profile="gaussian",
            amplitude_profile_decay_length=d,
        )
        wrinkle = cfg.wrinkles[0]
        npt.assert_allclose(
            cfg._amplitude_scale(wrinkle, d, 0.0),
            math.exp(-1.0),
            rtol=1e-14,
        )
        npt.assert_allclose(
            cfg._amplitude_scale(wrinkle, -d, 0.0),
            math.exp(-1.0),
            rtol=1e-14,
        )

    def test_two_decay_lengths(self, gaussian_wrinkle):
        d = 3.0
        cfg = _single_config(
            gaussian_wrinkle,
            ply_interface=1,
            amplitude_profile="gaussian",
            amplitude_profile_decay_length=d,
        )
        wrinkle = cfg.wrinkles[0]
        npt.assert_allclose(
            cfg._amplitude_scale(wrinkle, 2.0 * d, 0.0),
            math.exp(-4.0),
            rtol=1e-14,
        )

    def test_default_decay_length_falls_back_to_width(self):
        """When decay_length is None, the helper uses profile.width."""
        prof = GaussianSinusoidal(
            amplitude=0.5, wavelength=10.0, width=6.0, center=0.0
        )
        cfg = _single_config(
            prof, ply_interface=1, amplitude_profile="gaussian"
        )
        wrinkle = cfg.wrinkles[0]
        # At s = width=6, expect exp(-1).
        npt.assert_allclose(
            cfg._amplitude_scale(wrinkle, 6.0, 0.0),
            math.exp(-1.0),
            rtol=1e-14,
        )

    def test_y_axis_modulation(self, gaussian_wrinkle):
        """``axis='y'`` makes the scale a function of y only."""
        d = 4.0
        cfg = _single_config(
            gaussian_wrinkle,
            ply_interface=1,
            amplitude_profile="gaussian",
            amplitude_profile_decay_length=d,
            amplitude_profile_axis="y",
        )
        wrinkle = cfg.wrinkles[0]
        # Independent of x.
        for x in (-10.0, 0.0, 10.0):
            npt.assert_allclose(
                cfg._amplitude_scale(wrinkle, x, 0.0), 1.0, rtol=1e-14
            )
            npt.assert_allclose(
                cfg._amplitude_scale(wrinkle, x, d),
                math.exp(-1.0),
                rtol=1e-14,
            )


# ----------------------------------------------------------------------
# Linear profile -- triangular decay clipped at zero
# ----------------------------------------------------------------------

class TestLinearProfile:
    """``A_eff(s) = A * max(0, 1 - |s|/d)``."""

    def test_center_unity(self, gaussian_wrinkle):
        cfg = _single_config(
            gaussian_wrinkle,
            ply_interface=1,
            amplitude_profile="linear",
            amplitude_profile_decay_length=4.0,
        )
        wrinkle = cfg.wrinkles[0]
        npt.assert_allclose(
            cfg._amplitude_scale(wrinkle, 0.0, 0.0), 1.0, rtol=1e-14
        )

    def test_zero_at_one_decay_length(self, gaussian_wrinkle):
        d = 4.0
        cfg = _single_config(
            gaussian_wrinkle,
            ply_interface=1,
            amplitude_profile="linear",
            amplitude_profile_decay_length=d,
        )
        wrinkle = cfg.wrinkles[0]
        npt.assert_allclose(
            cfg._amplitude_scale(wrinkle, d, 0.0), 0.0, atol=1e-15
        )
        npt.assert_allclose(
            cfg._amplitude_scale(wrinkle, -d, 0.0), 0.0, atol=1e-15
        )

    def test_clipped_beyond_decay_length(self, gaussian_wrinkle):
        d = 2.0
        cfg = _single_config(
            gaussian_wrinkle,
            ply_interface=1,
            amplitude_profile="linear",
            amplitude_profile_decay_length=d,
        )
        wrinkle = cfg.wrinkles[0]
        for s in (1.5 * d, 3.0 * d, 100.0 * d):
            assert cfg._amplitude_scale(wrinkle, s, 0.0) == 0.0
            assert cfg._amplitude_scale(wrinkle, -s, 0.0) == 0.0

    def test_half_decay_length(self, gaussian_wrinkle):
        d = 8.0
        cfg = _single_config(
            gaussian_wrinkle,
            ply_interface=1,
            amplitude_profile="linear",
            amplitude_profile_decay_length=d,
        )
        wrinkle = cfg.wrinkles[0]
        npt.assert_allclose(
            cfg._amplitude_scale(wrinkle, 0.5 * d, 0.0), 0.5, rtol=1e-14
        )


# ----------------------------------------------------------------------
# Geometry / fibre-angle parity (#18 invariant)
# ----------------------------------------------------------------------

class TestGeometryAngleParity:
    """Where the amplitude scale vanishes the fibre-angle deviation
    must also vanish (the #18 invariant carried over to the new modulation)."""

    def test_linear_zero_zone_vanishes_in_both_fields(self, gaussian_wrinkle):
        # Use linear profile with d=2 so |x| >= 2 has zero amplitude.
        d = 2.0
        cfg = _single_config(
            gaussian_wrinkle,
            ply_interface=1,
            amplitude_profile="linear",
            amplitude_profile_decay_length=d,
            amplitude_profile_axis="x",
        )
        # Probe nodes well inside the zero zone.
        xs = np.array([-5.0, -3.0, 3.0, 5.0])
        nodes, ply = _strip(xs, n_plies=4)
        out = cfg.apply_to_nodes(nodes, ply, n_plies=4)
        dz = (out - nodes)[:, 2]
        npt.assert_allclose(dz, 0.0, atol=1e-15)
        # And the fibre angle field must agree at the same nodes.
        ang = cfg.fiber_angles_at_nodes(nodes, ply, n_plies=4)
        npt.assert_allclose(ang, 0.0, atol=1e-15)

    def test_gaussian_decays_displacement_and_angle_together(
        self, gaussian_wrinkle
    ):
        """Both fields share the same multiplicative scale, so their
        ratio (relative to the unmodulated baseline at the same node)
        must be identical."""
        d = 4.0
        cfg_mod = _single_config(
            gaussian_wrinkle,
            ply_interface=1,
            amplitude_profile="gaussian",
            amplitude_profile_decay_length=d,
            amplitude_profile_axis="x",
        )
        cfg_base = _single_config(gaussian_wrinkle, ply_interface=1)

        # Pick x stations off the crest so |slope| and |displacement| are
        # both nonzero (slope is zero at the crest).
        xs = np.array([1.0, 3.0, 5.0, 7.0])
        nodes, ply = _strip(xs, n_plies=4)
        dz_mod = (cfg_mod.apply_to_nodes(nodes, ply, n_plies=4) - nodes)[:, 2]
        dz_base = (cfg_base.apply_to_nodes(nodes, ply, n_plies=4) - nodes)[:, 2]
        ang_mod = cfg_mod.fiber_angles_at_nodes(nodes, ply, n_plies=4)
        ang_base = cfg_base.fiber_angles_at_nodes(nodes, ply, n_plies=4)

        # On the interface plies (p=1, p=2) the through-thickness decay
        # is unity, so the ratio is exactly the amplitude scale.
        mask = (ply == 1) | (ply == 2)
        nonzero = np.abs(dz_base[mask]) > 1e-12
        ratio_dz = dz_mod[mask][nonzero] / dz_base[mask][nonzero]
        nonzero_ang = np.abs(ang_base[mask]) > 1e-12
        # Composed-field angles (#252): the amplitude scale multiplies
        # the slope, so the ratio is recovered in tan space.
        ratio_ang = (
            np.tan(ang_mod[mask][nonzero_ang])
            / np.tan(ang_base[mask][nonzero_ang])
        )
        # Both ratios must equal the Gaussian scale at the same x.
        # Use the displacement-side ratio as ground truth and compare.
        npt.assert_allclose(ratio_dz.mean(), ratio_ang.mean(), rtol=1e-12)


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

class TestValidation:
    def test_bogus_profile_name_raises(self, gaussian_wrinkle):
        with pytest.raises(ValueError, match="Unknown amplitude_profile"):
            _single_config(
                gaussian_wrinkle,
                ply_interface=1,
                amplitude_profile="bogus",
            )

    def test_negative_decay_length_raises(self, gaussian_wrinkle):
        with pytest.raises(
            ValueError, match="amplitude_profile_decay_length must be positive"
        ):
            _single_config(
                gaussian_wrinkle,
                ply_interface=1,
                amplitude_profile="gaussian",
                amplitude_profile_decay_length=-1.0,
            )

    def test_zero_decay_length_raises(self, gaussian_wrinkle):
        with pytest.raises(
            ValueError, match="amplitude_profile_decay_length must be positive"
        ):
            _single_config(
                gaussian_wrinkle,
                ply_interface=1,
                amplitude_profile="linear",
                amplitude_profile_decay_length=0.0,
            )

    def test_bogus_axis_raises(self, gaussian_wrinkle):
        with pytest.raises(ValueError, match="Unknown amplitude_profile_axis"):
            _single_config(
                gaussian_wrinkle,
                ply_interface=1,
                amplitude_profile_axis="z",
            )


# ----------------------------------------------------------------------
# Smoke -- apply_to_nodes runs end-to-end for every profile kind
# ----------------------------------------------------------------------

class TestSmoke:
    @pytest.mark.parametrize("kind", ["constant", "gaussian", "linear"])
    def test_apply_to_nodes_runs(self, gaussian_wrinkle, kind):
        cfg = _single_config(
            gaussian_wrinkle, ply_interface=1, amplitude_profile=kind
        )
        nodes, ply = _strip(np.linspace(-6.0, 6.0, 5), n_plies=4)
        out = cfg.apply_to_nodes(nodes, ply, n_plies=4)
        assert out.shape == nodes.shape
        # X / Y unchanged.
        npt.assert_array_equal(out[:, :2], nodes[:, :2])

    @pytest.mark.parametrize("kind", ["constant", "gaussian", "linear"])
    def test_fiber_angles_runs(self, gaussian_wrinkle, kind):
        cfg = _single_config(
            gaussian_wrinkle, ply_interface=1, amplitude_profile=kind
        )
        nodes, ply = _strip(np.linspace(-6.0, 6.0, 5), n_plies=4)
        ang = cfg.fiber_angles_at_nodes(nodes, ply, n_plies=4)
        assert ang.shape == (nodes.shape[0],)
        assert np.all(ang >= 0.0)
