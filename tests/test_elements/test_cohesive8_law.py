"""Tests for the bilinear intrinsic CZM in Cohesive8Element.

Single-element T-vs-delta sweeps verify, in order of importance:
1. Area under the mode-I curve equals G_Ic * A (energy conservation).
2. Damage irreversibility on unload-reload.
3. Compression uses the penalty branch and accumulates no damage.
4. Mode-II area equals G_IIc * A.
5. Mixed-mode envelope follows the Benzeggagh-Kenane formula.
"""

from __future__ import annotations

import numpy as np
import pytest

from wrinklefe.elements.cohesive8 import (
    Cohesive8Element,
    CohesiveProperties,
    make_initial_state,
)

# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def unit_square_interface_nodes():
    """8 nodes: bottom 1x1 square at z=0, top 1x1 square at z=0 (zero gap).

    Bottom 0..3 CCW from (0,0); top 4..7 paired with bottom 0..3.
    """
    bottom = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    top = bottom.copy()
    return np.vstack([bottom, top])


@pytest.fixture
def props():
    return CohesiveProperties(
        K=1.0e5, sigma_max=50.0, tau_max=80.0,
        GIc=0.5, GIIc=1.5, eta_BK=1.45, beta=1.0,
    )


def _make_elem(nodes, props):
    return Cohesive8Element(node_coords=nodes, properties=props)


def _integrate_work(
    elem: Cohesive8Element,
    direction: np.ndarray,
    deltas: np.ndarray,
) -> float:
    """Total external work done by the cohesive tractions during the sweep,
    computed directly from the internal force at the top nodes (which
    equals the cohesive resistance).

    For displacement-controlled monotonic loading, this is exactly the
    energy dissipated by the law (positive when traction opposes motion).
    """
    state = make_initial_state(elem._n_gp)
    u = np.zeros(24)
    work = 0.0
    F_prev = np.zeros(24)
    for k, d in enumerate(deltas):
        top_disp = d * direction
        u_new = u.copy()
        u_new[12:] = np.tile(top_disp, 4)
        _, F_int, state = elem.tangent_and_force(u_new, state)
        # Trapezoidal: dW = 0.5 * (F_prev + F_int)^T @ (u_new - u)
        work += 0.5 * float((F_prev + F_int) @ (u_new - u))
        u = u_new
        F_prev = F_int
    return work


# ======================================================================
# Tests
# ======================================================================

def test_mode_I_area_equals_GIc(unit_square_interface_nodes, props):
    """Sweep delta_n from 0 to 2 delta_f; total dissipated work should
    match GIc * A within 1%."""
    elem = _make_elem(unit_square_interface_nodes, props)
    delta_f = 2.0 * props.GIc / props.sigma_max
    deltas = np.linspace(0.0, 2.0 * delta_f, 4001)
    direction = np.array([0.0, 0.0, 1.0])  # +z is the bottom->top normal
    work = _integrate_work(elem, direction, deltas)
    expected = props.GIc * elem.area
    rel = abs(work - expected) / expected
    assert rel < 0.01, (
        f"Mode-I energy mismatch: work={work:.4f} vs Gc*A={expected:.4f} "
        f"(rel err {rel:.3%})"
    )


def test_irreversibility_unload_reload(unit_square_interface_nodes, props):
    """Load to 1.5*delta_0, unload to 0, reload — d must not decrease;
    reload should follow secant (1-d)*K, not original K."""
    elem = _make_elem(unit_square_interface_nodes, props)
    delta_0 = props.sigma_max / props.K
    direction = np.array([0.0, 0.0, 1.0])

    # Phase 1: load to 1.5 delta_0.
    state = make_initial_state(elem._n_gp)
    u = np.zeros(24)

    def _set_top(d):
        u_new = u.copy()
        u_new[12:] = np.tile(d * direction, 4)
        return u_new

    u_peak = _set_top(1.5 * delta_0)
    _, _, state = elem.tangent_and_force(u_peak, state)
    d_peak = state[0].d
    assert 0.0 < d_peak < 1.0, f"expected partial damage, got {d_peak}"

    # Phase 2: unload to 0.
    u_zero = _set_top(0.0)
    _, F_zero, state = elem.tangent_and_force(u_zero, state)
    assert state[0].d == pytest.approx(d_peak, rel=1e-12), (
        "Damage decreased on unload!"
    )
    # Traction at zero opening should be zero (secant goes through origin).
    assert np.linalg.norm(F_zero) < 1e-8

    # Phase 3: reload to a small fraction of 1.5 delta_0 — slope should
    # be (1-d) * K, not K.
    d_probe = 0.5 * delta_0  # below delta_0, but on the secant line
    u_probe = _set_top(d_probe)
    _, F_probe, state2 = elem.tangent_and_force(u_probe, state)
    # Damage must not have changed (we're below the recorded max).
    assert state2[0].d == pytest.approx(d_peak, rel=1e-12)
    # Top-node normal-component total force = T_z * area.
    T_z = (1.0 - d_peak) * props.K * d_probe
    expected_force = T_z * elem.area
    # Sum z-component of F at top nodes.
    actual_force = float(np.sum(F_probe[14::3]))  # top z-DOFs: 14, 17, 20, 23
    assert np.isclose(actual_force, expected_force, rtol=1e-6), (
        f"Reload slope wrong: expected (1-d)*K*delta*A = {expected_force}, "
        f"got {actual_force}; d_peak={d_peak}"
    )


def test_compression_penalty_no_damage(unit_square_interface_nodes, props):
    """Push top nodes in -z: traction should be K*delta_n, no damage."""
    elem = _make_elem(unit_square_interface_nodes, props)
    state = make_initial_state(elem._n_gp)

    direction = np.array([0.0, 0.0, -1.0])
    d = 0.01  # interpenetration of 0.01 mm
    u = np.zeros(24)
    u[12:] = np.tile(d * direction, 4)
    _, F_int, state = elem.tangent_and_force(u, state)

    # Damage should remain zero.
    assert state[0].d == 0.0, f"d={state[0].d} != 0 under compression"
    # Top-node z force = K * delta_n * area (delta_n = -d, so force on top
    # nodes is -K * d * area, pushing them back up).
    expected_force_per_area = props.K * (-d)
    expected_total = expected_force_per_area * elem.area
    actual_force = float(np.sum(F_int[14::3]))
    assert np.isclose(actual_force, expected_total, rtol=1e-10), (
        f"Compression penalty force wrong: expected {expected_total}, "
        f"got {actual_force}"
    )


def test_pure_mode_II_area_equals_GIIc(unit_square_interface_nodes, props):
    """Sweep tangential delta_s; area should match GIIc * A within 1%."""
    elem = _make_elem(unit_square_interface_nodes, props)
    # Pure mode II: shear in +x (which is in-plane for an x-y interface).
    direction = np.array([1.0, 0.0, 0.0])
    delta_f_II = 2.0 * props.GIIc / props.tau_max
    deltas = np.linspace(0.0, 2.0 * delta_f_II, 4001)
    work = _integrate_work(elem, direction, deltas)
    expected = props.GIIc * elem.area
    rel = abs(work - expected) / expected
    assert rel < 0.01, (
        f"Mode-II energy mismatch: work={work:.4f} vs GIIc*A={expected:.4f} "
        f"(rel err {rel:.3%})"
    )


def test_node_ids_rejects_negative(unit_square_interface_nodes, props):
    """Negative node indices must be rejected at construction time.

    Without an explicit bounds check, a negative index silently wraps
    through numpy fancy indexing in the assembler's DOF map and couples
    the cohesive element to the wrong end of the global vector.  This
    test guards against that off-by-one footgun.
    """
    with pytest.raises(ValueError, match="non-negative"):
        Cohesive8Element(
            node_coords=unit_square_interface_nodes,
            properties=props,
            node_ids=np.array(
                [-1, 0, 1, 2, 3, 4, 5, 6], dtype=np.intp
            ),
        )


def test_mixed_mode_BK_envelope(unit_square_interface_nodes, props):
    """Apply combined (delta_n, delta_s) at a fixed mode-mixity angle;
    integrate energy until d -> 1; compare to the BK Gc(psi)."""
    elem = _make_elem(unit_square_interface_nodes, props)

    # Mode-mixity angle phi: tan(phi) = delta_s / delta_n in the local
    # frame.  Pick phi such that G_II / (G_I + G_II) is well inside (0, 1).
    phi = np.radians(45.0)
    direction_global = np.array([np.sin(phi), 0.0, np.cos(phi)])

    # Determine the BK expected Gc at this mode-mixity.  In the law we
    # weight by mode_ratio = G_II / (G_I + G_II) computed from squared
    # openings with beta = 1; for a unit-magnitude direction this is
    # sin^2(phi) / (sin^2 + cos^2) = sin^2(phi).
    mode_ratio = np.sin(phi) ** 2
    Gc_expected = props.GIc + (props.GIIc - props.GIc) * (
        mode_ratio ** props.eta_BK
    )
    # Effective delta_f at this mixity.
    sigma_mixed_sq = (
        (props.sigma_max ** 2)
        + ((props.tau_max ** 2) - (props.sigma_max ** 2))
        * (mode_ratio ** props.eta_BK)
    )
    sigma_mixed = float(np.sqrt(sigma_mixed_sq))
    delta_f_mixed = 2.0 * Gc_expected / sigma_mixed

    # Sweep along the direction up to 2 * delta_f_mixed.
    deltas = np.linspace(0.0, 2.0 * delta_f_mixed, 4001)
    work = _integrate_work(elem, direction_global, deltas)
    expected = Gc_expected * elem.area
    rel = abs(work - expected) / expected
    assert rel < 0.05, (
        f"BK mixed-mode area mismatch at phi={np.degrees(phi):.1f} deg: "
        f"work={work:.4f} vs Gc(psi)*A={expected:.4f} (rel err {rel:.3%})"
    )
