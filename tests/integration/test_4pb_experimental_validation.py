"""Four-Point-Bend experimental validation against NASA/TM-2020-220498
Section 4.13.

Mode-II companion to ``test_enf_experimental_validation.py``.  Same
panel, same calibrated ``h_arm = 2.02 mm``, same pre-crack treatment;
the only differences are the loading fixture (4-point instead of 3-
point bend), the slightly shorter specimen (9 in instead of 10 in)
and a slightly lower measured G_IIc (0.720 vs 0.777 N/mm).

Source data
-----------
NASA/TM-2020-220498 "Overview of Coupon Testing of an IM7/8552
Composite ...", Justusson et al. 2020, Section 4.13 (Four-Point
Bending) reports:

  Material: IM7/8552 unidirectional tape (Boeing/NASA panel)
  Coupon size: 9 in x 1 in = 228.6 mm x 25.4 mm
  Layup: [+/-2/0_9/-/+2/2/FEP/2/-/+2/0_9/+/-2] (13 plies per arm)
  Pre-crack: 3 in (76.2 mm) FEP at midplane from one end
  4-pt-bend geometry:
      outer (support) span = 8 in = 203.2 mm  -- bottom rollers
      inner (loading) span = 4 in = 101.6 mm  -- top rollers
      both spans centered on the specimen
      overhang per side = (228.6 - 203.2) / 2 = 12.7 mm
  Crack-tip location: 12.7 mm overhang + 63.5 mm into span =
      76.2 mm from the FEP end.  Crack tip falls in the constant-
      moment region (between the inner rollers at x in
      [63.5, 165.1] mm) -> pure mode-II loading.
  Compliance-derivative slope: m = dC/da = 0.000064 /lbf
  Critical energy release rate from G_IIc = m*P^2 / (2b):
      G_IIc = 4.11 +/- 0.335 in*lb/in^2 = 0.720 +/- 0.059 N/mm
  Peak loads: 350-385 lbf (~1555-1715 N) for 5 specimens.
  Peak displacements: 0.28-0.32 in (~7.1-8.1 mm).
  Loading: quasi-static displacement control at 0.002 in/s.
  Deflection measured with a deflectometer under the loading head ->
      directly comparable to the FE global load-point displacement
      (in contrast to the ENF test, where the published deflections
      appear to use a local DIC-style measurement).
  The 5 published curves are remarkably LINEAR up to peak with no
  visible softening (NASA TM Section 4.13.2: "did not give as
  consistent of a result as the ENF testing"; nevertheless each
  individual curve is essentially linear).

Beam-theory sanity check
------------------------
The Carlsson m parameter, m = 0.000064 /lbf = 1.4388e-5 /N, with
G_IIc = m P^2 / (2b) and b = 25.4 mm, gives

    P_c = sqrt(2 b G_IIc / m)
        = sqrt(2 * 25.4 * 0.720 / 1.4388e-5)
        = 1594 N

within 3 % of the experimental peak load average ~ 1645 N.  Initial
elastic compliance from the standard 4-point-bend uncracked-beam
formula

    delta / P = s (3 L^2 - 4 s^2) / (48 EI)

with L = outer span = 203.2 mm, s = (L - inner span)/2 = 50.8 mm,
EI = E1 b (2 h)^3 / 12 = 2.393e7 N*mm^2, gives delta/P ~
5.02e-3 mm/N -> initial slope dP/d delta ~ 199 N/mm, within ~8 %
of the experimental slope ~ 213 N/mm (1625 N / 7.62 mm).

Parameter rationale
-------------------
- ``h_arm = 2.02 mm`` (calibrated): the same value used in
  ``test_dcb_experimental_validation.py`` and
  ``test_enf_experimental_validation.py``.  NASA TM doesn't publish
  per-specimen thickness, so we reuse the back-calculated 0.156 mm/
  ply that closed the elastic-compliance gap on the DCB test of the
  same panel.

- ``GIIc = 0.720`` (4PB measured value from Section 4.13);
  ``GIc = 0.324`` (measured from DCB on the same panel; required by
  the bilinear law, irrelevant in pure mode II).

- ``tau_max = 80 MPa``: same value that worked for the ENF test on
  the same panel.  Gives a mode-II cohesive-zone length
  ``lambda_cz_II = E1 G_IIc / tau_max^2 ~ 19 mm`` -- a small fraction
  of the 152 mm bonded region.

- Mesh ``NX = 150`` (~1.52 mm elements), same scale as DCB validation
  NX = 150 (~1.69 mm) and ENF validation NX = 200 (~1.27 mm).

Pre-crack treatment
-------------------
Same as ENF: cohesive elements in the FEP region [0, FEP_END_X] are
pre-damaged to ``d = 1`` and act as frictionless contact (penalty in
compression, zero traction in opening, zero shear).

Loading strategy
----------------
Displacement-controlled at the two top rollers simultaneously, ramped
over 200 fixed equal increments from 0 to ``DELTA_MAX``.  No adaptive
sub-stepping (Phase 7 finding from DCB and ENF).  The total applied
load P is read back as the sum of the internal-force z-components on
*both* top-roller node sets (sign-corrected to a positive magnitude).

``DELTA_MAX = 10 mm`` is chosen to comfortably overshoot the
experimental peak displacement (~7.6 mm at P ~ 1625 N).

Validation strategy
-------------------
Four comparison metrics, computed up-front so a single diagnostic
print summarises the run before any assertion fires:

  (1) Initial elastic compliance ``dP/d delta`` (linear fit through
      origin in the first ~10 % of the ramp) vs experimental slope
      ~ 213 N/mm.  Tolerance: 25 % relative (looser than ENF because
      4PB compliance is more sensitive to crack-tip location relative
      to the constant-moment region).
  (2) Peak load ``max(P)`` vs experimental peak ~ 1645 N.  Tolerance:
      +/- 15 % (allow [1320, 1972] N), covering the ~9 % experimental
      scatter (1555-1715 N) plus the usual CZM-vs-experiment numerical
      gap.
  (3) Cohesive-zone existence: at least one *bonded* cohesive element
      reaches ``d > 0.5`` during the ramp.
  (4) Curve linearity (soft): the predicted P-delta curve should be
      approximately linear up to peak, matching the qualitative
      behaviour of the experimental curves (no visible softening or
      pre-peak nonlinearity in Figure 30).  We fit a line through the
      first 80 % of the rising branch and check that the max
      deviation is below 10 % of P_peak.

Anti-goals
----------
- No solver / element / mesh changes -- only a test file + a plot.
- Tolerances NOT loosened to fit; if FE can't match within +/- 15 %
  on peak the test xfails with a documented reason.
- Fixed 200 equal increments -- no adaptive sub-stepping.

References
----------
Justusson, B., Pankow, M., Heinrich, C., Rudolph, M., Neal, A.
(2020).  NASA/TM-2020-220498.  Section 4.13 (Four-Point Bending).
Carlsson, L.A. & Pipes, R.B. (1997).  Experimental Characterization
of Advanced Composite Materials, 2nd ed.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI/test runs.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from wrinklefe.core.cohesive_mesh import insert_cohesive_interface  # noqa: E402
from wrinklefe.core.laminate import Laminate, Ply  # noqa: E402
from wrinklefe.core.material import MaterialLibrary  # noqa: E402
from wrinklefe.core.mesh import MeshData, WrinkleMesh  # noqa: E402
from wrinklefe.elements.cohesive8 import (  # noqa: E402
    CohesiveProperties,
    CohesiveState,
)
from wrinklefe.solver.assembler import GlobalAssembler  # noqa: E402
from wrinklefe.solver.boundary import BoundaryCondition, BoundaryHandler  # noqa: E402
from wrinklefe.solver.nonlinear import NewtonRaphsonSolver  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# ----------------------------------------------------------------------
# Experimental data (NASA/TM-2020-220498 Section 4.13)
# ----------------------------------------------------------------------

# Reported G_IIc from the 4PB test (Section 4.13.2):
EXPERIMENTAL_4PB_GIIC_NMM: float = 0.720       # 4.11 in*lb/in^2
EXPERIMENTAL_4PB_GIIC_STD_NMM: float = 0.059   # +/- 0.335 in*lb/in^2

# Peak load averaged across the 5 specimens in Figure 30: peak loads
# span ~ 350-385 lbf (1555-1715 N) at displacements 0.28-0.32 in
# (7.1-8.1 mm).  We take the midpoint ~ 370 lbf -> 1645 N as the
# reference and apply a +/- 15 % band that comfortably covers both
# experimental scatter and numerical CZM-vs-experiment gap.
EXPERIMENTAL_4PB_P_PEAK_N: float = 1645.0      # midpoint of 1555-1715 N

# Averaged P-delta loading branch digitised from NASA TM Figure 30.
# The five published curves are nearly perfectly linear up to peak with
# no visible softening -- this set of points fits a straight line of
# slope ~ 213 N/mm through the origin to within a few percent.
EXPERIMENTAL_4PB_PD: list[tuple[float, float]] = [
    (0.000, 0.0),
    (1.27,   267.0),    # 0.05 in,  60 lbf
    (2.54,   534.0),    # 0.10 in, 120 lbf
    (3.81,   801.0),    # 0.15 in, 180 lbf
    (5.08,  1068.0),    # 0.20 in, 240 lbf
    (6.35,  1335.0),    # 0.25 in, 300 lbf
    (7.62,  1625.0),    # 0.30 in, 365 lbf -- near peak
]

# Wide band on the peak load: +/- 15 % covers the +/- 9 % scatter in
# measured peak (1555-1715 N) plus the usual CZM-vs-experiment
# numerical gap.
PEAK_TOLERANCE_REL: float = 0.15
PEAK_LOAD_LO_N: float = EXPERIMENTAL_4PB_P_PEAK_N * (1.0 - PEAK_TOLERANCE_REL)
PEAK_LOAD_HI_N: float = EXPERIMENTAL_4PB_P_PEAK_N * (1.0 + PEAK_TOLERANCE_REL)

# Initial elastic slope from the experimental loading branch.  A line
# through the origin fit to the first non-zero digitised points gives
# ~ 213 N/mm; we use that as the reference and apply a 25 % tolerance
# (looser than ENF because 4PB compliance is more sensitive to where
# the crack tip falls relative to the constant-moment region).
EXPERIMENTAL_4PB_SLOPE_NMM: float = 213.0
SLOPE_TOLERANCE_REL: float = 0.25


# ----------------------------------------------------------------------
# Geometry / material / cohesive parameters
# ----------------------------------------------------------------------

L_TOTAL = 228.6            # mm (9 in)
WIDTH = 25.4               # mm (1 in)

# Ply thickness: same calibrated value as DCB/ENF tests; see those
# docstrings for the rationale (NASA TM doesn't publish per-specimen
# thickness; back-calc from DCB compliance gives ~0.156 mm/ply).
PLY_THICKNESS = 0.1554     # mm (calibrated from DCB compliance)
N_PLIES_PER_ARM = 13
H_ARM = N_PLIES_PER_ARM * PLY_THICKNESS  # ~ 2.020 mm

OUTER_SPAN = 203.2         # mm (8 in) -- bottom rollers
INNER_SPAN = 101.6         # mm (4 in) -- top rollers
OUTER_OVERHANG = (L_TOTAL - OUTER_SPAN) / 2.0  # = 12.7 mm per side

LEFT_SUPPORT_X = OUTER_OVERHANG                            # 12.7
RIGHT_SUPPORT_X = L_TOTAL - OUTER_OVERHANG                 # 215.9
LEFT_LOAD_X = L_TOTAL / 2.0 - INNER_SPAN / 2.0             # 63.5
RIGHT_LOAD_X = L_TOTAL / 2.0 + INNER_SPAN / 2.0            # 165.1
INNER_OFFSET = (OUTER_SPAN - INNER_SPAN) / 2.0             # 50.8

# FEP pre-crack: 3 in from one end of the specimen.  With 0.5 in
# (12.7 mm) overhang, the crack tip lands at 76.2 mm in mesh-native x.
# Crack length from left support to crack tip = 76.2 - 12.7 = 63.5 mm
# = 2.5 in, which falls inside the constant-moment region [LEFT_LOAD_X,
# RIGHT_LOAD_X] = [63.5, 165.1] mm so the loading is pure mode II.
FEP_END_X = 76.2           # mm (3 in from x = 0)
A_FROM_LEFT_SUPPORT = FEP_END_X - LEFT_SUPPORT_X  # 63.5 mm = 2.5 in

# Mesh
# NX = 180 lands ALL five critical x-positions (LEFT_SUPPORT, LEFT_LOAD,
# FEP_END, RIGHT_LOAD, RIGHT_SUPPORT = 12.7, 63.5, 76.2, 165.1, 215.9
# mm) exactly on node planes: each is an integer multiple of dx =
# 228.6 / 180 = 1.27 mm.  This is the same dx as the ENF validation
# (NX = 200 on a 254 mm specimen).  Specimen-internal NX choices that
# do NOT land the support / load x-positions on nodes would either
# require interpolated BCs (not supported by BoundaryHandler) or
# silently drop the BCs (the tol-based node selectors would return
# empty sets).
NX = 180
NY = 1
NZ_PER_ARM = 2             # total nz = 4, interface at z = 0 (mid-plane)

# Measured cohesive toughness from NASA TM
GIIC_MEASURED = 0.720      # N/mm (4PB-measured mode II)
GIC_MEASURED = 0.324       # N/mm (from DCB on same panel; required by
                           # bilinear law, irrelevant in pure mode II)

# Penalty stiffness same as DCB / ENF / monotonic tests
K_PENALTY = 1.0e6          # N/mm^3

# tau_max same as the ENF validation: gives a mode-II cohesive-zone
# length lambda_cz_II = E1 * G_IIc / tau_max^2 = 171420 * 0.720 / 6400
# = 19.3 mm, ~13 % of the 152 mm bonded region.
TAU_MAX = 80.0             # MPa
SIGMA_MAX = 80.0           # MPa (mode-II irrelevant; cf. ENF monotonic)

# Loading: 200 fixed equal displacement increments, NO adaptive sub-
# stepping (Phase 7 lesson from DCB / ENF).
N_INCREMENTS = 200

# DELTA_MAX = 10 mm comfortably overshoots the experimental peak at
# ~ 7.6 mm.  Leaves room for any post-peak excursion if it occurs.
DELTA_MAX = 10.0           # mm


# ----------------------------------------------------------------------
# Material
# ----------------------------------------------------------------------


def _build_material():
    """Fetch the IM7_8552 elastic properties from the canonical library."""
    return MaterialLibrary().get("IM7_8552")


MAT = _build_material()


def _build_cohesive_properties() -> CohesiveProperties:
    """Construct the bilinear law parameters for mode-II 4PB validation."""
    return CohesiveProperties(
        K=K_PENALTY,
        sigma_max=SIGMA_MAX,
        tau_max=TAU_MAX,
        GIc=GIC_MEASURED,
        GIIc=GIIC_MEASURED,
        eta_BK=1.45,
        beta=1.0,
    )


# ----------------------------------------------------------------------
# Analytical helpers (4-point-bend beam theory)
# ----------------------------------------------------------------------


def _fourpb_initial_slope() -> float:
    """Beam-theory initial slope ``dP/d delta`` for the uncracked 4PB
    configuration with the deflectometer under the loading head.

    Standard formula for symmetric 4PB midspan deflection:

        delta = P * s * (3 L^2 - 4 s^2) / (48 EI)

    where L is the outer span, s = (L - inner span)/2 = distance from
    each support to the nearest inner roller, P is the *total* applied
    load, and EI is the bending stiffness of the (uncracked) full beam.

    Because the deflectometer is positioned under the loading head, the
    measured deflection is the same as the midspan deflection in a
    symmetric configuration -- so this is directly comparable to the FE
    load-point displacement.
    """
    EI_full = MAT.E1 * WIDTH * (2.0 * H_ARM) ** 3 / 12.0
    delta_per_P = INNER_OFFSET * (
        3.0 * OUTER_SPAN ** 2 - 4.0 * INNER_OFFSET ** 2
    ) / (48.0 * EI_full)
    return 1.0 / delta_per_P if delta_per_P > 0.0 else float("inf")


def _fourpb_carlsson_peak() -> float:
    """Critical (peak) load for the 4PB specimen from the reported
    compliance-derivative slope m = dC/da:

        G_IIc = m P^2 / (2 b)
        =>  P_c = sqrt(2 b G_IIc / m)

    The NASA TM reports m = 0.000064 / lbf = 1.4388e-5 / N.  Plugging
    in b = 25.4 mm and G_IIc = 0.720 N/mm gives P_c ~ 1594 N, within
    3 % of the experimental average ~ 1645 N.
    """
    m_per_N = 0.000064 / 4.4482216  # /lbf -> /N
    return float(np.sqrt(2.0 * WIDTH * GIIC_MEASURED / m_per_N))


# ----------------------------------------------------------------------
# Mesh / model construction
# ----------------------------------------------------------------------


def _build_mesh(
    coh_props: CohesiveProperties,
) -> tuple[MeshData, list, list[bool]]:
    """Build the 4PB mesh + cohesive list + bonded/pre-crack mask.

    The structured hex8 mesh is generated as two stacked plies (each
    one arm thick) so the interface plane z = 0 lands on the ply
    interface.  Midplane interface nodes are duplicated by
    :func:`insert_cohesive_interface`, then cohesive elements are
    partitioned into two groups by their mid-surface x:

      * Bonded region [FEP_END_X, L_TOTAL]: cohesive law active
        (d = 0 initially); these elements can damage and grow the
        crack under the constant moment in the inner-span region.
      * FEP pre-crack [0, FEP_END_X): cohesive elements kept but pre-
        damaged to ``d = 1`` in :func:`_build_assembler` to act as
        frictionless contact (penalty in compression, zero traction
        in opening, zero shear) -- exactly the treatment validated in
        ``test_enf_monotonic.py`` / ``test_enf_experimental_validation.py``.

    Returns
    -------
    mesh : MeshData
        Mesh with duplicated interface nodes.
    cohesive_elements : list
        All cohesive elements (bonded + FEP-contact).
    is_bonded : list[bool]
        Per-element flag, True for bonded elements (d = 0 initial),
        False for FEP-contact elements (must be pre-damaged to d = 1).
    """
    laminate = Laminate([
        Ply(material=MAT, angle=0.0, thickness=H_ARM),
        Ply(material=MAT, angle=0.0, thickness=H_ARM),
    ])
    wm = WrinkleMesh(
        laminate=laminate,
        wrinkle_config=None,
        Lx=L_TOTAL, Ly=WIDTH,
        nx=NX, ny=NY,
        nz_per_ply=NZ_PER_ARM,
    )
    base_mesh = wm.generate()

    z_mid = 0.5 * (
        float(base_mesh.nodes[:, 2].min())
        + float(base_mesh.nodes[:, 2].max())
    )

    new_mesh, all_coh = insert_cohesive_interface(
        base_mesh, z_interface=z_mid, cohesive_props=coh_props,
    )

    kept: list = []
    is_bonded: list[bool] = []
    for c in all_coh:
        x_mid = float(c.node_coords[:4, 0].mean())
        if x_mid >= FEP_END_X:
            kept.append(c)
            is_bonded.append(True)
        else:
            # FEP pre-crack -- keep as pre-damaged contact element.
            kept.append(c)
            is_bonded.append(False)

    for k, c in enumerate(kept):
        c.elem_id = k

    return new_mesh, kept, is_bonded


def _build_assembler(
    mesh: MeshData,
    cohesive_elements: list,
    is_bonded: list[bool],
) -> GlobalAssembler:
    """Build the assembler and pre-damage FEP cohesives to d = 1.

    Bonded cohesive elements keep the default initial state so they can
    evolve via the bilinear traction-separation law as the crack grows.
    FEP pre-crack cohesives are pre-damaged to ``d = 1`` with a frozen
    mode-II ratio so the cohesive law returns pure frictionless-contact
    tractions (penalty-in-compression, zero-in-opening, zero-in-shear).
    """
    laminate = Laminate([
        Ply(material=MAT, angle=0.0, thickness=H_ARM),
        Ply(material=MAT, angle=0.0, thickness=H_ARM),
    ])
    asm = GlobalAssembler(
        mesh=mesh,
        laminate=laminate,
        cohesive_elements=[(c.elem_id, c) for c in cohesive_elements],
    )
    for c, bonded in zip(cohesive_elements, is_bonded):
        if not bonded:
            n_gp = c.n_gp
            asm.cohesive_state[c.elem_id] = [
                CohesiveState(d=1.0, mode_ratio_init=1.0)
                for _ in range(n_gp)
            ]
            asm.cohesive_state_trial[c.elem_id] = [
                CohesiveState(d=1.0, mode_ratio_init=1.0)
                for _ in range(n_gp)
            ]
    return asm


def _support_and_load_nodes(
    mesh: MeshData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Identify the bottom-support and top-loading-roller node sets.

    Returns
    -------
    left_support_nodes : nodes on the bottom face at x = LEFT_SUPPORT_X
        (the FEP-side bottom roller).
    right_support_nodes : nodes on the bottom face at x = RIGHT_SUPPORT_X
        (the far-side bottom roller).
    left_load_nodes : nodes on the top face at x = LEFT_LOAD_X (left
        inner top roller, x = 63.5 mm = 0.5 in past the FEP boundary).
    right_load_nodes : nodes on the top face at x = RIGHT_LOAD_X
        (right inner top roller, x = 165.1 mm).
    """
    tol = 1e-6
    x = mesh.nodes[:, 0]
    z = mesh.nodes[:, 2]
    z_min = float(z.min())
    z_max = float(z.max())

    on_z_min = np.abs(z - z_min) <= tol
    on_z_max = np.abs(z - z_max) <= tol

    on_left_sup_x = np.abs(x - LEFT_SUPPORT_X) <= tol
    on_right_sup_x = np.abs(x - RIGHT_SUPPORT_X) <= tol
    on_left_load_x = np.abs(x - LEFT_LOAD_X) <= tol
    on_right_load_x = np.abs(x - RIGHT_LOAD_X) <= tol

    left_support = np.flatnonzero(on_left_sup_x & on_z_min).astype(np.intp)
    right_support = np.flatnonzero(on_right_sup_x & on_z_min).astype(np.intp)
    left_load = np.flatnonzero(on_left_load_x & on_z_max).astype(np.intp)
    right_load = np.flatnonzero(on_right_load_x & on_z_max).astype(np.intp)
    return left_support, right_support, left_load, right_load


def _build_bcs(
    mesh: MeshData,
    delta: float,
) -> list[BoundaryCondition]:
    """Four-point-bend BCs matching the NASA TM Section 4.13 fixture.

    - Bottom support 1 (FEP side) at x = LEFT_SUPPORT_X: pin u_z on the
      full bottom-face line.  One node also pinned in u_x / u_y to
      remove the in-plane rigid-body translations.
    - Bottom support 2 at x = RIGHT_SUPPORT_X: pin u_z on the full
      bottom-face line.
    - Top loading roller 1 at x = LEFT_LOAD_X: prescribe u_z = -delta
      on the full top-face line at that x.
    - Top loading roller 2 at x = RIGHT_LOAD_X: prescribe u_z = -delta
      on the full top-face line at that x (same delta -- symmetric
      4-point loading).
    """
    left_support, right_support, left_load, right_load = _support_and_load_nodes(
        mesh,
    )

    # Pick exactly one left-support node to also pin in x/y -- the one
    # with the smallest y-coordinate is a deterministic choice.
    y_at_left = mesh.nodes[left_support, 1]
    pin_node = np.array(
        [int(left_support[int(np.argmin(y_at_left))])], dtype=np.intp,
    )

    return [
        BoundaryCondition(
            bc_type="fixed", node_ids=left_support, dofs=[2],
        ),
        BoundaryCondition(
            bc_type="fixed", node_ids=right_support, dofs=[2],
        ),
        BoundaryCondition(
            bc_type="fixed", node_ids=pin_node, dofs=[0, 1],
        ),
        BoundaryCondition(
            bc_type="displacement", node_ids=left_load,
            dofs=[2], value=-float(delta),
        ),
        BoundaryCondition(
            bc_type="displacement", node_ids=right_load,
            dofs=[2], value=-float(delta),
        ),
    ]


# ----------------------------------------------------------------------
# Fixed-increment Newton driver (no adaptive sub-stepping)
# ----------------------------------------------------------------------


def _drive_4pb_fixed(
    mesh: MeshData,
    cohesive_elements: list,
    is_bonded: list[bool],
    delta_max: float,
    n_increments: int,
    verbose: bool = False,
) -> dict:
    """Drive the 4PB specimen through N fixed equal displacement
    increments.

    Matches the Phase 7 DCB / ENF drivers: no step halving, no step
    growth on success -- just N equal increments.  If a Newton step
    fails the increment is skipped (u and committed state unchanged)
    and the driver moves on; the failure count is returned for
    diagnostics.

    Per converged increment we record:
      * The total applied load P = sum of internal-force z-components
        at *both* top-roller load-node sets (absolute value, since the
        internal force points up at the loaded surfaces while delta
        is negative).
      * Max ``d`` across all *bonded* cohesive elements (for the damage
        existence assertion).
    """
    assembler = _build_assembler(mesh, cohesive_elements, is_bonded)
    bc_handler = BoundaryHandler(mesh)

    solver = NewtonRaphsonSolver(
        assembler=assembler,
        bc_handler=bc_handler,
        boundary_conditions=_build_bcs(mesh, 0.0),
        n_increments=1,
        max_newton_iter=200,
        tol_residual=1e-4,
        tol_absolute=1e-8,
        tol_displacement=1e-9,
        line_search=False,
    )

    _, _, left_load_nodes, right_load_nodes = _support_and_load_nodes(mesh)
    left_z_dofs = 3 * left_load_nodes + 2
    right_z_dofs = 3 * right_load_nodes + 2

    # Bonded cohesive ids -- used for the damage-existence tracker.
    bonded_ids = [
        c.elem_id for c, b in zip(cohesive_elements, is_bonded) if b
    ]

    u = np.zeros(mesh.n_dof)
    converged_deltas: list[float] = [0.0]
    converged_P: list[float] = [0.0]
    converged_dmax: list[float] = [0.0]
    total_fails = 0

    step = delta_max / n_increments
    for i in range(n_increments):
        delta_try = (i + 1) * step
        bcs_now = _build_bcs(mesh, delta_try)
        cons = bc_handler.get_constrained_dofs(bcs_now)
        F_ext = bc_handler.get_force_dofs(bcs_now)
        u_new, n_iter, ok = solver._newton_step(
            u, F_ext, cons, verbose=verbose, inc=i + 1,
        )
        if not ok:
            total_fails += 1
            # Skip this increment -- do NOT halve.  Diagnostic only.
            continue
        u = u_new
        solver._commit_state()

        F_int = assembler.assemble_internal_force(u)
        # Total applied load = sum of both inner-roller reactions.
        P_left = float(np.sum(F_int[left_z_dofs]))
        P_right = float(np.sum(F_int[right_z_dofs]))
        P_load = abs(P_left + P_right)

        # Max damage across all bonded cohesives.
        d_max = 0.0
        for cid in bonded_ids:
            for s in assembler.cohesive_state[cid]:
                if s.d > d_max:
                    d_max = s.d

        converged_deltas.append(delta_try)
        converged_P.append(P_load)
        converged_dmax.append(d_max)

        if verbose:
            print(
                f"  inc {i + 1:3d}: delta={delta_try:.4f}, P={P_load:8.2f} "
                f"(left={abs(P_left):.1f}, right={abs(P_right):.1f}), "
                f"d_max={d_max:.3f}, iters={n_iter}"
            )

    return {
        "deltas": np.asarray(converged_deltas),
        "P": np.asarray(converged_P),
        "d_max": np.asarray(converged_dmax),
        "n_fails": total_fails,
    }


# ----------------------------------------------------------------------
# Plot helper
# ----------------------------------------------------------------------


def _save_comparison_plot(
    result: dict,
    out_path: Path,
) -> None:
    """Write the predicted-vs-experimental P-delta comparison plot."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cd = result["deltas"]
    cP = result["P"]
    i_peak = int(np.argmax(cP))
    P_peak = float(cP[i_peak])
    delta_peak = float(cd[i_peak])

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # Experimental loading branch from NASA TM Figure 30 (digitised
    # averaged curve across 5 specimens).  Light shaded band +/- 10 %
    # around the experimental curve as a scatter proxy.
    exp_arr = np.asarray(EXPERIMENTAL_4PB_PD, dtype=float)
    exp_delta = exp_arr[:, 0]
    exp_P = exp_arr[:, 1]
    x_max = max(float(cd[-1]), float(exp_delta[-1])) * 1.02
    ax.fill_between(
        exp_delta, 0.9 * exp_P, 1.1 * exp_P,
        color="tab:orange", alpha=0.18,
        label="Experimental scatter band ($\\pm$10 %)",
    )
    ax.plot(
        exp_delta, exp_P,
        linestyle="--", color="tab:orange", linewidth=2.0,
        marker="o", markersize=4,
        label=(
            f"Experimental P-$\\delta$ (NASA/TM-2020-220498 §4.13, "
            f"$P_\\mathrm{{peak}}$ $\\approx$ "
            f"{EXPERIMENTAL_4PB_P_PEAK_N:.0f} N)"
        ),
    )

    # Predicted curve
    ax.plot(
        cd, cP,
        linestyle="-", color="tab:blue", linewidth=2.0,
        label=f"FE prediction (CZM, $\\tau_\\max$ = {TAU_MAX:.0f} MPa)",
    )

    # Annotation: peak load marker + vertical guide
    ax.axvline(
        delta_peak, color="tab:blue", linestyle=":", linewidth=1.2,
        alpha=0.7,
    )
    ax.plot(
        [delta_peak], [P_peak],
        marker="o", color="tab:blue", markersize=8, zorder=5,
    )
    ax.annotate(
        f"FE peak: {P_peak:.1f} N @ $\\delta$ = {delta_peak:.2f} mm",
        xy=(delta_peak, P_peak),
        xytext=(max(0.5, delta_peak - 4.0), P_peak + 100.0),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
    )

    ax.set_xlabel("Load-point displacement, $\\delta$ [mm]")
    ax.set_ylabel("Total applied load, $P$ [N]")
    ax.set_title(
        "NASA/TM-2020-220498 Four-Point Bend — Predicted vs Experimental\n"
        f"IM7/8552, h$_\\mathrm{{arm}}$ = {H_ARM:.2f} mm (calibrated), "
        f"FEP = 76.2 mm, "
        f"G$_\\mathrm{{IIc}}$ = {GIIC_MEASURED:.3f} N/mm"
    )
    ax.set_xlim(0.0, x_max)
    y_top = max(float(cP.max()), EXPERIMENTAL_4PB_P_PEAK_N) * 1.20
    ax.set_ylim(0.0, y_top)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# The validation test
# ----------------------------------------------------------------------


def _predicted_initial_slope(
    deltas: np.ndarray, P_arr: np.ndarray,
) -> float:
    """Linear-fit slope ``dP/d delta`` from the first elastic portion of
    the ramp (delta in (0, 10 % of DELTA_MAX] = (0, 1.0] mm).

    Fit constrained through the origin: m = sum(d * p) / sum(d * d).
    """
    if deltas.size < 4:
        return 0.0
    cutoff = 0.10 * DELTA_MAX
    mask = (deltas > 1e-9) & (deltas <= cutoff)
    if mask.sum() < 2:
        idx = int(np.argmax(deltas > 1e-9))
        return float(P_arr[idx] / deltas[idx]) if deltas[idx] > 0 else 0.0
    d = deltas[mask]
    p = P_arr[mask]
    return float(np.sum(d * p) / np.sum(d * d))


def _curve_linearity_deviation(
    deltas: np.ndarray, P_arr: np.ndarray, i_peak: int,
) -> float:
    """Max deviation of the rising branch (up to 80 % of peak) from the
    best-fit line through the origin.

    Returns the max absolute residual ``max(|P - m*delta|)`` over the
    rising branch where P >= 0.05 * P_peak and P <= 0.80 * P_peak.
    This excludes both the initial noise near zero and the curvature
    that may appear just before peak.
    """
    if i_peak < 3:
        return 0.0
    P_peak = float(P_arr[i_peak])
    if P_peak <= 0.0:
        return 0.0
    cP = P_arr[: i_peak + 1]
    cd = deltas[: i_peak + 1]
    mask = (cP >= 0.05 * P_peak) & (cP <= 0.80 * P_peak)
    if mask.sum() < 3:
        return 0.0
    d = cd[mask]
    p = cP[mask]
    # Fit through origin again, on the rising branch.
    m = float(np.sum(d * p) / np.sum(d * d))
    residual = np.abs(p - m * d)
    return float(residual.max())


def test_4pb_experimental_validation_nasa_tm():
    """Compare the CZM prediction to NASA/TM-2020-220498 4-point-bend
    data.

    Validates mode-II behaviour against the same NASA TM panel used by
    the DCB and ENF validation tests, but with a 4-point-bend fixture
    rather than a 3-point ENF.  The crack tip falls in the constant-
    moment region between the inner rollers, so the loading is pure
    mode II (no transverse-shear contribution at the crack tip).
    Compliance, peak load, damage existence, and curve linearity are
    the four comparison metrics.
    """
    coh_props = _build_cohesive_properties()
    mesh, cohesive_elements, is_bonded = _build_mesh(coh_props)

    # Element-count sanity check: the bonded region is [FEP_END_X,
    # L_TOTAL] of length 152.4 mm; element width is L_TOTAL / NX.
    n_bonded_expected = sum(
        1 for i in range(NX) if (i + 0.5) * (L_TOTAL / NX) >= FEP_END_X
    )
    n_total_expected = NX
    assert sum(is_bonded) == n_bonded_expected, (
        f"expected {n_bonded_expected} bonded cohesive elements, got "
        f"{sum(is_bonded)}"
    )
    assert len(cohesive_elements) == n_total_expected, (
        f"expected {n_total_expected} total cohesive elements, got "
        f"{len(cohesive_elements)}"
    )

    res = _drive_4pb_fixed(
        mesh, cohesive_elements, is_bonded,
        delta_max=DELTA_MAX,
        n_increments=N_INCREMENTS,
        verbose=False,
    )
    cd = res["deltas"]
    cP = res["P"]
    cdmax = res["d_max"]

    # ------------------------------------------------------------------
    # Diagnostics (computed up-front so the print summarises the run
    # *before* any assertion fires).
    # ------------------------------------------------------------------

    # (1) Initial elastic compliance / stiffness.
    slope_pred = _predicted_initial_slope(cd, cP)
    slope_analytical = _fourpb_initial_slope()
    slope_rel_to_exp = (
        abs(slope_pred - EXPERIMENTAL_4PB_SLOPE_NMM)
        / EXPERIMENTAL_4PB_SLOPE_NMM
        if EXPERIMENTAL_4PB_SLOPE_NMM > 0.0 else float("inf")
    )
    slope_in_band = slope_rel_to_exp < SLOPE_TOLERANCE_REL

    # (2) Peak load.
    P_peak_pred = float(cP.max())
    i_peak = int(np.argmax(cP))
    delta_peak = float(cd[i_peak])
    peak_in_band = PEAK_LOAD_LO_N <= P_peak_pred <= PEAK_LOAD_HI_N
    peak_rel = (
        abs(P_peak_pred - EXPERIMENTAL_4PB_P_PEAK_N)
        / EXPERIMENTAL_4PB_P_PEAK_N
    )

    # (3) Damage existence.
    d_max_final = float(cdmax.max())
    damage_exists = d_max_final > 0.5

    # (4) Curve linearity (soft check).  Compute max deviation from the
    # best-fit line through the origin over the 5-80 % rising branch;
    # require it to be less than 10 % of P_peak.
    linearity_dev = _curve_linearity_deviation(cd, cP, i_peak)
    linearity_threshold = 0.10 * P_peak_pred
    linearity_in_band = linearity_dev <= linearity_threshold

    # Analytical reference values for the print.
    P_c_analytical = _fourpb_carlsson_peak()
    lambda_cz_II = MAT.E1 * GIIC_MEASURED / (TAU_MAX ** 2)
    bonded_length = L_TOTAL - FEP_END_X

    # Write the plot regardless of assertion outcomes -- user-facing
    # deliverable.
    out_path = Path(__file__).resolve().parents[2] / "figures" / (
        "phase7_4pb_validation.png"
    )
    _save_comparison_plot(res, out_path)

    print(
        f"\nPhase 7 4PB validation (NX={NX}, tau_max={TAU_MAX:.1f} MPa, "
        f"GIIc={GIIC_MEASURED:.3f} N/mm, h_arm={H_ARM:.3f} mm):\n"
        f"  (1) slope     = {slope_pred:8.2f} N/mm "
        f"(exp {EXPERIMENTAL_4PB_SLOPE_NMM:.1f} N/mm, "
        f"beam-theory {slope_analytical:.2f} N/mm, "
        f"rel {slope_rel_to_exp:.2%}, tol {SLOPE_TOLERANCE_REL:.0%})  "
        f"{'PASS' if slope_in_band else 'FAIL'}\n"
        f"  (2) P_peak    = {P_peak_pred:8.2f} N "
        f"(exp {EXPERIMENTAL_4PB_P_PEAK_N:.0f} N "
        f"[{PEAK_LOAD_LO_N:.0f}, {PEAK_LOAD_HI_N:.0f}], rel "
        f"{peak_rel:.2%}, Carlsson m {P_c_analytical:.0f} N)  "
        f"{'PASS' if peak_in_band else 'FAIL'}\n"
        f"  (3) max d     = {d_max_final:8.3f} "
        f"(threshold > 0.50)  "
        f"{'PASS' if damage_exists else 'FAIL'}\n"
        f"  (4) lin dev   = {linearity_dev:8.2f} N "
        f"(limit {linearity_threshold:.2f} N, 10 % of P_peak; SOFT CHECK)  "
        f"{'PASS' if linearity_in_band else 'FAIL'}\n"
        f"  delta_peak = {delta_peak:.3f} mm "
        f"(exp ~ 7.6 mm @ peak)\n"
        f"  lambda_cz_II = {lambda_cz_II:.2f} mm "
        f"(bonded length = {bonded_length:.2f} mm)\n"
        f"  Newton failures = {res['n_fails']} / {N_INCREMENTS} increments\n"
        f"  Plot: {out_path}"
    )

    assert out_path.is_file(), f"Comparison plot was not written to {out_path}"

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    # (1) Initial elastic slope within 25 % of the experimental value.
    assert slope_in_band, (
        f"Initial elastic slope {slope_pred:.2f} N/mm off experimental "
        f"{EXPERIMENTAL_4PB_SLOPE_NMM:.1f} N/mm by {slope_rel_to_exp:.2%} "
        f"(tol {SLOPE_TOLERANCE_REL:.0%})"
    )

    # (2) Peak load within experimental scatter band (+/- 15 %)
    assert peak_in_band, (
        f"Predicted peak load {P_peak_pred:.2f} N outside band "
        f"[{PEAK_LOAD_LO_N:.2f}, {PEAK_LOAD_HI_N:.2f}] N (experimental "
        f"P_peak ~ {EXPERIMENTAL_4PB_P_PEAK_N:.0f} N, rel "
        f"{peak_rel:.2%})"
    )

    # (3) At least one cohesive element reached d > 0.5
    assert damage_exists, (
        f"No cohesive element reached d > 0.5 during the ramp "
        f"(max d = {d_max_final:.3f})."
    )

    # (4) Pre-peak linearity -- soft check.
    assert linearity_in_band, (
        f"Pre-peak deviation from linearity {linearity_dev:.2f} N exceeds "
        f"10 % of P_peak (= {linearity_threshold:.2f} N).  The "
        f"experimental 4PB curves are nearly perfectly linear up to "
        f"peak (Figure 30 in NASA/TM-2020-220498)."
    )


if __name__ == "__main__":
    # Allow ad-hoc invocation without pytest's capture.
    os.environ.setdefault("WRINKLEFE_4PB_VERBOSE", "1")
    test_4pb_experimental_validation_nasa_tm()
