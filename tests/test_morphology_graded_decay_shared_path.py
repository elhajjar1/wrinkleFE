"""Regression guard for issues #40 and #32 (duplicate graded-decay drift).

Background
----------
Historically, :meth:`WrinkleConfiguration.apply_to_nodes` and
:meth:`WrinkleConfiguration.fiber_angles_at_nodes` each carried their own
copy of the through-thickness decay logic. If one copy was updated and the
other was not, the displacement field and the local fibre-angle field
would silently disagree, producing physically inconsistent results
(node z-shifts implying one envelope, fibre angles implying another).

Issue #185 / PR #225 vectorised both methods and extracted the shared
helper ``_through_thickness_decay`` that handles the ``default``,
``uniform``, and ``graded`` decay modes. This test pins that contract.

What we assert
--------------
1. **Shared call site** -- both methods route their per-ply decay through
   the single ``_through_thickness_decay`` helper. We patch that helper
   in place and observe that BOTH methods reflect the patched values; a
   future regression that forks one method off to a private duplicate
   would fail this test.

2. **Identical per-ply decay factors** -- for a representative ``graded``
   configuration we recover the implicit per-ply decay from each method
   (``apply_to_nodes`` via the z-displacement and ``fiber_angles_at_nodes``
   via the fibre angle) and assert they match to machine precision.

3. **Behavioural sanity** -- the recovered factors match the documented
   graded envelope (``decay_floor + (1 - decay_floor) * (1 - |p - p_mid|/p_mid)``).
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from wrinklefe.core.morphology import WrinkleConfiguration, WrinklePlacement
from wrinklefe.core.wrinkle import GaussianSinusoidal


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

AMPLITUDE = 0.366
WAVELENGTH = 16.0
WIDTH = 12.0
N_PLIES = 24
DECAY_FLOOR = 0.2


def _profile() -> GaussianSinusoidal:
    # ``center=0`` puts the crest at x=0 so we can sample the displacement
    # / slope on a single column and read off the per-ply decay directly.
    return GaussianSinusoidal(
        amplitude=AMPLITUDE,
        wavelength=WAVELENGTH,
        width=WIDTH,
        center=0.0,
    )


def _graded_config() -> WrinkleConfiguration:
    """Single wrinkle, graded decay -- the exact configuration #40/#32 call out."""
    placement = WrinklePlacement(
        profile=_profile(),
        ply_interface=N_PLIES // 2,
        phase_offset=0.0,
    )
    return WrinkleConfiguration(
        [placement],
        decay_mode="graded",
        decay_floor=DECAY_FLOOR,
    )


def _single_column_mesh():
    """One node per ply, all stacked at x=0, y=0 (the wrinkle crest)."""
    nodes = np.zeros((N_PLIES, 3), dtype=np.float64)
    ply_ids = np.arange(N_PLIES, dtype=np.int64)
    return nodes, ply_ids


# ----------------------------------------------------------------------
# 1. Shared call site -- patching the helper must affect BOTH methods.
# ----------------------------------------------------------------------

def test_through_thickness_decay_helper_drives_both_methods(monkeypatch):
    """If ``_through_thickness_decay`` is patched, both consumers see the patch.

    Forking either method off to a private decay implementation would
    cause this test to fail because the patched sentinel would only show
    up in one output.
    """
    cfg = _graded_config()
    nodes, ply_ids = _single_column_mesh()

    sentinel_value = 0.37  # arbitrary distinguishable factor

    def fake_decay(self, ply_ids, k, n_plies):
        return np.full(np.shape(ply_ids), sentinel_value, dtype=np.float64)

    monkeypatch.setattr(
        WrinkleConfiguration,
        "_through_thickness_decay",
        fake_decay,
    )

    # Reference un-decayed crest displacement and slope-derived angle.
    profile = cfg.wrinkles[0].profile
    x = nodes[:, 0]
    raw_dz = profile.displacement(x)

    expected_dz = raw_dz * sentinel_value
    # Composed-field angles (#252): decay scales the slope, not the angle.
    expected_angle = np.arctan(np.abs(profile.slope(x)) * sentinel_value)

    deformed = cfg.apply_to_nodes(nodes, ply_ids, N_PLIES)
    angles = cfg.fiber_angles_at_nodes(nodes, ply_ids, n_plies=N_PLIES)

    # Both methods routed through the patched helper.
    npt.assert_allclose(deformed[:, 2], expected_dz, atol=1e-12, rtol=1e-12)
    npt.assert_allclose(angles, expected_angle, atol=1e-12, rtol=1e-12)


# ----------------------------------------------------------------------
# 2. Per-ply decay parity between the two methods.
# ----------------------------------------------------------------------

def _expected_graded_factor(p: int, n_plies: int, decay_floor: float) -> float:
    p_mid = (n_plies - 1) / 2.0
    raw = 1.0 - abs(p - p_mid) / p_mid
    return max(0.0, decay_floor + (1.0 - decay_floor) * raw)


def test_apply_to_nodes_and_fiber_angles_share_graded_decay():
    """The per-ply decay implied by both methods is identical (and graded).

    We sample on a single (x=0, y=0) column where the crest displacement
    is exactly ``A`` and the crest slope is 0 -- but the SECOND derivative
    of cos is non-zero only off-crest. To recover the decay from the angle
    field we instead sample at the steepest-slope x = wavelength/4.
    """
    cfg = _graded_config()
    profile = cfg.wrinkles[0].profile

    # --- Recover decay from apply_to_nodes at x = 0 (crest displacement). --
    nodes_crest = np.zeros((N_PLIES, 3), dtype=np.float64)
    ply_ids = np.arange(N_PLIES, dtype=np.int64)
    deformed = cfg.apply_to_nodes(nodes_crest, ply_ids, N_PLIES)
    raw_dz = profile.displacement(nodes_crest[:, 0])
    # Avoid division by zero (raw_dz at crest is ``amplitude`` > 0).
    assert np.all(np.abs(raw_dz) > 1e-9), "crest displacement should be non-zero"
    decay_from_disp = deformed[:, 2] / raw_dz

    # --- Recover decay from fiber_angles_at_nodes at the steepest x. ------
    x_steep = WAVELENGTH / 4.0
    nodes_steep = np.zeros((N_PLIES, 3), dtype=np.float64)
    nodes_steep[:, 0] = x_steep
    angles = cfg.fiber_angles_at_nodes(nodes_steep, ply_ids, n_plies=N_PLIES)
    raw_slope = np.abs(profile.slope(nodes_steep[:, 0]))
    assert raw_slope[0] > 1e-9, "slope at quarter-wavelength should be non-zero"
    # Composed-field angles (#252): angle = arctan(decay * |slope|), so
    # the decay is recovered in slope space.
    decay_from_angle = np.tan(angles) / raw_slope

    # --- The two recovered decay vectors must agree exactly. --------------
    npt.assert_allclose(
        decay_from_disp,
        decay_from_angle,
        atol=1e-12,
        rtol=1e-12,
        err_msg=(
            "apply_to_nodes and fiber_angles_at_nodes produced different "
            "per-ply decay factors -- the graded-mode decay has drifted "
            "(see issues #40 / #32)."
        ),
    )

    # --- And both must equal the documented graded envelope. --------------
    expected = np.array(
        [_expected_graded_factor(p, N_PLIES, DECAY_FLOOR) for p in range(N_PLIES)]
    )
    npt.assert_allclose(decay_from_disp, expected, atol=1e-12, rtol=1e-12)
    npt.assert_allclose(decay_from_angle, expected, atol=1e-12, rtol=1e-12)


# ----------------------------------------------------------------------
# 3. Parity also holds across the other decay modes (defence in depth).
# ----------------------------------------------------------------------

@pytest.mark.parametrize("decay_mode", ["default", "uniform", "graded"])
def test_decay_parity_across_modes(decay_mode):
    """Both methods agree on the per-ply decay for every supported mode."""
    placement = WrinklePlacement(
        profile=_profile(),
        ply_interface=N_PLIES // 2,
        phase_offset=0.0,
    )
    cfg = WrinkleConfiguration(
        [placement],
        decay_mode=decay_mode,
        decay_floor=DECAY_FLOOR,
    )
    profile = cfg.wrinkles[0].profile
    ply_ids = np.arange(N_PLIES, dtype=np.int64)

    # Crest sample => recover decay from displacement.
    nodes_crest = np.zeros((N_PLIES, 3), dtype=np.float64)
    deformed = cfg.apply_to_nodes(nodes_crest, ply_ids, N_PLIES)
    raw_dz = profile.displacement(nodes_crest[:, 0])
    decay_from_disp = deformed[:, 2] / raw_dz

    # Quarter-wavelength sample => recover decay from angle.
    nodes_steep = np.zeros((N_PLIES, 3), dtype=np.float64)
    nodes_steep[:, 0] = WAVELENGTH / 4.0
    angles = cfg.fiber_angles_at_nodes(nodes_steep, ply_ids, n_plies=N_PLIES)
    raw_slope = np.abs(profile.slope(nodes_steep[:, 0]))
    decay_from_angle = np.tan(angles) / raw_slope

    npt.assert_allclose(
        decay_from_disp,
        decay_from_angle,
        atol=1e-12,
        rtol=1e-12,
        err_msg=(
            f"decay_mode={decay_mode!r}: apply_to_nodes and "
            "fiber_angles_at_nodes disagreed on the per-ply decay."
        ),
    )
