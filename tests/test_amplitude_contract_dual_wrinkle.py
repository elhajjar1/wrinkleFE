"""Regression tests for the dual-wrinkle amplitude contract (issue #305).

The configured amplitude ``A`` is defined (see ``analysis.py`` and
``tests/test_amplitude_definition.py``) as the peak deflection of the
generated wrinkle: for a single profile the mesh peak-to-trough is ``2A``
and the peak deflection is ``A``.

The dual-wrinkle morphologies (``stack``/``convex``/``concave``) build the
mesh by *summing* two decayed displacement fields. Before #305 each
constituent carried the full amplitude ``A``, so the in-phase ``stack``
mesh peaked at ``2A`` — double the geometry, fibre angle and knockdown the
caller (and the analytical model) intended. The fix places an ``A/2`` clone
of the profile at each interface so the composed peak equals ``A``.

These tests pin, directly on the FE node arrays (no solve — fast):

* the composed peak deflection ``max|dz|`` for all five morphologies, and
* that the ``stack`` mesh fibre angle equals the single-profile
  ``profile.max_angle()`` (the analytical ``arctan(2*pi*A/lambda)``),
  i.e. the mesh is self-consistent with the amplitude that was requested.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from wrinklefe.core.morphology import WrinkleConfiguration
from wrinklefe.core.wrinkle import GaussianSinusoidal

_A = 0.15
_LAMBDA = 4.0
_WIDTH = 8.0
_N_PLIES = 10
_INTERFACE1 = 4
_INTERFACE2 = 5


def _profile() -> GaussianSinusoidal:
    return GaussianSinusoidal(
        amplitude=_A, wavelength=_LAMBDA, width=_WIDTH, center=0.0
    )


def _composed_peak_deflection(cfg: WrinkleConfiguration) -> float:
    """Max |dz| over the whole laminate for a fine x-strip (no FE solve)."""
    xs = np.linspace(-_WIDTH, _WIDTH, 801)
    peak = 0.0
    for p in range(_N_PLIES):
        nodes = np.zeros((len(xs), 3))
        nodes[:, 0] = xs
        ply_ids = np.full(len(xs), p, dtype=int)
        dz = cfg.apply_to_nodes(nodes, ply_ids, _N_PLIES)[:, 2]
        peak = max(peak, float(np.max(np.abs(dz))))
    return peak


def _composed_max_angle(cfg: WrinkleConfiguration) -> float:
    """Max |arctan(d(dz)/dx)| over the laminate (radians)."""
    xs = np.linspace(-_WIDTH, _WIDTH, 4001)
    max_angle = 0.0
    for p in range(_N_PLIES):
        nodes = np.zeros((len(xs), 3))
        nodes[:, 0] = xs
        ply_ids = np.full(len(xs), p, dtype=int)
        dz = cfg.apply_to_nodes(nodes, ply_ids, _N_PLIES)[:, 2]
        slope = np.gradient(dz, xs)
        max_angle = max(max_angle, float(np.max(np.abs(np.arctan(slope)))))
    return max_angle


# Composed peak deflection as a multiple of the configured amplitude A.
# stack (in-phase) sums to exactly A; convex/concave partly cancel;
# uniform is a single full-amplitude wrinkle (peak == A); graded decays.
EXPECTED_PEAK_OVER_A = {
    "stack": 1.000,
    "convex": 0.704,
    "concave": 0.704,
    "uniform": 1.000,
    "graded": 0.889,
}


@pytest.mark.parametrize("name", sorted(EXPECTED_PEAK_OVER_A))
def test_composed_peak_deflection_honours_amplitude(name):
    """Issue #305: no morphology builds a mesh taller than the configured
    amplitude A; the in-phase stack composes to exactly A (not 2A)."""
    cfg = WrinkleConfiguration.from_morphology_name(
        name, _profile(), interface1=_INTERFACE1, interface2=_INTERFACE2
    )
    peak = _composed_peak_deflection(cfg)
    assert peak / _A == pytest.approx(EXPECTED_PEAK_OVER_A[name], abs=2e-3), (
        f"{name}: composed peak {peak:.5f} = {peak / _A:.3f}*A, "
        f"expected {EXPECTED_PEAK_OVER_A[name]:.3f}*A"
    )
    # Hard ceiling: the mesh must never exceed the requested amplitude.
    assert peak <= _A + 1e-9


def test_stack_never_doubles_amplitude():
    """The precise #305 defect: an in-phase dual wrinkle used to peak at 2A.
    Pin the composed peak at exactly A so the regression cannot return."""
    cfg = WrinkleConfiguration.from_morphology_name(
        "stack", _profile(), interface1=_INTERFACE1, interface2=_INTERFACE2
    )
    npt.assert_allclose(_composed_peak_deflection(cfg), _A, atol=1e-6)


def test_stack_mesh_angle_matches_analytical_profile():
    """The stack mesh fibre angle equals the single-profile max_angle
    (arctan(2*pi*A/lambda)), so the FE geometry is self-consistent with the
    analytical knockdown model rather than twice its slope (issue #305)."""
    prof = _profile()
    cfg = WrinkleConfiguration.from_morphology_name(
        "stack", prof, interface1=_INTERFACE1, interface2=_INTERFACE2
    )
    mesh_angle = _composed_max_angle(cfg)
    # Fine finite-difference gradient: match to ~0.1 deg.
    npt.assert_allclose(mesh_angle, prof.max_angle(), atol=2e-3)


def test_dual_wrinkle_constituents_carry_half_amplitude():
    """Each placed wrinkle in a dual morphology carries A/2; the original
    profile passed by the caller is left untouched."""
    prof = _profile()
    cfg = WrinkleConfiguration.dual_wrinkle(
        prof, interface1=_INTERFACE1, interface2=_INTERFACE2, phase=0.0
    )
    for placement in cfg.wrinkles:
        placed = placement.profile
        amp = getattr(placed, "profile", placed).amplitude
        assert amp == pytest.approx(_A / 2.0)
    # Caller's profile is not mutated.
    assert prof.amplitude == pytest.approx(_A)
