"""ENF experimental validation against NASA/TM-2020-220498 Section 4.15.

Mode-II companion to ``test_dcb_experimental_validation.py``.  Validates
the CZM infrastructure (Cohesive8 + NewtonRaphson + cohesive-mesh
insertion) against the IM7/8552 End-Notched Flexure (ENF) coupon test
data reported by Justusson et al. (2020, NASA/TM-2020-220498, Section
4.15).  Same panel as the DCB test (calibrated ``h_arm = 2.02 mm``);
this is therefore a directly-compatible mode-II check on the same
elastic model that succeeded for mode I.

Source data
-----------
NASA/TM-2020-220498 "Overview of Coupon Testing of an IM7/8552
Composite ...", Justusson et al. 2020, Section 4.15 (ENF) reports:

  Material: IM7/8552 unidirectional tape (Boeing/NASA panel)
  Coupon size: 10 in x 1 in = 254.0 mm x 25.4 mm
  Layup: [+/-2/0_9/-/+2/2/FEP/2/-/+2/0_9/+/-2] (13 plies per arm)
  Pre-crack: 3 in (76.2 mm) FEP at midplane from one end
  3-pt-bend geometry: 2L = 8 in span; specimen overhangs each support
                       by 1 in; top roller at midspan.
  Crack tip: 2 in (50.8 mm) into the span from the nearest support.
  Measured G_IIc = 4.44 +/- 0.36 in*lb/in^2 = 0.777 +/- 0.063 N/mm
  Critical peak load: P_critical = 160 lbf = 712 N
  Test stopped at peak (load-controlled at 0.002 in/s) to avoid
  unstable propagation; only the peak is published per specimen.

Beam-theory sanity check
------------------------
Carlsson-Pipes ENF critical load
    P_c = sqrt(16 b^2 E h^3 G_IIc / (9 a^2))
with b = 25.4 mm, E1 = 171.42 GPa, h = 2.02 mm, G_IIc = 0.777 N/mm,
a = 50.8 mm gives P_c ~ 699 N -- within 2 % of the experimental 712 N.
Mode-II beam theory matches this specimen *very* well (in contrast to
the DCB case where beam theory was 30 % off because the cohesive zone
was a non-trivial fraction of the bonded length).  Our CZM prediction
should therefore land very close to both.

Parameter rationale
-------------------
- ``h_arm = 2.02 mm`` (calibrated): the same value used in
  ``test_dcb_experimental_validation.py``; the rationale is the same.
  NASA TM doesn't publish per-specimen thickness, so we use the back-
  calculated 0.156 mm/ply that closed the elastic-compliance gap on
  the DCB test of the same panel.  Re-using this calibration here is
  what makes the two tests directly comparable.

- ``GIIc = 0.777`` (measured); ``GIc = 0.324`` (measured from DCB on
  the same panel; required by the bilinear law but irrelevant in
  pure mode II).

- ``tau_max = 80 MPa``: matched to the working tau_max from the existing
  ENF monotonic benchmark, which gives a Mode-II cohesive-zone length
  ``lambda_cz_II = E1 * GIIc / tau_max**2 = 20.8 mm`` -- a small fraction
  of the 152 mm bonded region, in the "developed CZ" regime per
  Hillerborg.  ``sigma_max = 80 MPa`` set equal (the bilinear law
  requires it but it is irrelevant in mode II).

- Mesh ``NX = 200`` (~1.27 mm elements): same scale as the DCB
  validation NX=150 (~1.69 mm).  ``NZ_PER_ARM = 2`` so the total z
  count is 4, with the cohesive interface at z = 0 in the middle.

Pre-crack treatment
-------------------
Same as ``test_enf_monotonic.py``: rather than deleting cohesive
elements in the FEP region (which would let the upper arm float
free of the lower arm in the unsupported part of the pre-crack and
roughly halve the elastic stiffness), we keep them but pre-damage
them to ``d = 1``.  At ``d = 1`` the bilinear law degenerates to
penalty-contact-in-compression / zero-traction-in-opening / zero-
shear, i.e. frictionless contact -- exactly what beam theory tacitly
assumes for an ENF pre-crack closed by the bending moment.

Loading strategy
----------------
Displacement-controlled at the top roller, ramped over 200 fixed
equal increments from 0 to ``DELTA_MAX``.  This matches the lesson
from Phase 7 DCB: adaptive sub-stepping inflates peak overshoot by
overshooting equilibrium on each successful step, so we use fixed
equal increments instead.  Load is read back as the sum of internal-
force z-components on the top-roller nodes (sign-corrected to a
positive magnitude).

``DELTA_MAX = 10 mm`` is chosen so the ramp reaches well past the
experimental peak (~6.2 mm at P = 712 N per beam theory) and leaves
~3.8 mm for post-peak observation.

Validation strategy
-------------------
Four comparison metrics, computed up-front so a single diagnostic
print summarises the run before any assertion fires:

  (1) Initial elastic compliance ``dP/d delta`` (linear fit in the
      first ~10 % of the ramp) vs analytical Carlsson-Pipes ENF
      compliance.  Tolerance: 15 % relative.
  (2) Peak load ``max(P)`` vs experimental 712 N.  Tolerance: +/- 15 %
      (allow [605, 819] N).
  (3) Cohesive-zone existence: at least one *bonded* cohesive element
      reaches ``d > 0.5`` during the ramp.  Catches the mode-II-under-
      compression collapse mode where the entire interface stays at
      d = 0 and the test is no longer a CZ validation.
  (4) Post-peak stability (soft check): no upward spikes greater than
      10 % of P_peak after the peak.  Likely to fail due to known
      limitation #6.1 in CZM_PLAN.md -- mode-II damage growth is
      suppressed in normal compression so the post-peak branch can be
      noisy.

The CZM_PLAN.md known limitation #6.1 (mode-II-under-compression
suppression) means the post-peak behaviour, energy dissipated, and
crack-tip advance are likely *not* meaningful for ENF.  Compliance
and peak load are.  We therefore validate only those two
quantitatively + assert that *some* damage occurs, and mark
post-peak stability as a soft check.

Anti-goals
----------
- No solver / element / mesh changes -- only a test file + a plot.
- Tolerances NOT loosened to fit; if the model can't match
  experimental within +/- 15 % on peak the whole test xfails with a
  documented reason.
- Fixed 200 equal increments -- no adaptive sub-stepping (Phase 7
  finding from the DCB test).

References
----------
Justusson, B., Pankow, M., Heinrich, C., Rudolph, M., Neal, A.
(2020).  NASA/TM-2020-220498.  Sections 4.14 (DCB) and 4.15 (ENF).
Carlsson, L.A. & Pipes, R.B. (1997).  Experimental Characterization
of Advanced Composite Materials, 2nd ed., Chapter 6 (ENF derivation).
"""

from __future__ import annotations

import math
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


# ----------------------------------------------------------------------
# Experimental data (NASA/TM-2020-220498 Section 4.15)
# ----------------------------------------------------------------------

EXPERIMENTAL_P_CRITICAL_N: float = 712.0    # 160 lbf
EXPERIMENTAL_GIIC_MEAN_NMM: float = 0.777   # 4.44 in*lb/in^2
EXPERIMENTAL_GIIC_STD_NMM: float = 0.063    # +/- 0.36 in*lb/in^2

# Wide band on the peak load: +/- 15 % covers (a) the +/- 8 % scatter
# in measured G_IIc and (b) typical CZM-vs-experiment numerical gap
# from a non-tuned bilinear law.
PEAK_TOLERANCE_REL: float = 0.15
PEAK_LOAD_LO_N: float = EXPERIMENTAL_P_CRITICAL_N * (1.0 - PEAK_TOLERANCE_REL)
PEAK_LOAD_HI_N: float = EXPERIMENTAL_P_CRITICAL_N * (1.0 + PEAK_TOLERANCE_REL)


# ----------------------------------------------------------------------
# Geometry / material / cohesive parameters
# ----------------------------------------------------------------------

L_TOTAL = 254.0            # mm (10 in)
WIDTH = 25.4               # mm (1 in)

# Ply thickness: same calibrated value as the DCB test; see the
# docstring of ``test_dcb_experimental_validation.py`` for the rationale
# (NASA TM doesn't publish per-specimen thickness; the back-calc from
# experimental DCB compliance gives ~0.156 mm/ply).
PLY_THICKNESS = 0.1554     # mm (calibrated from DCB compliance)
N_PLIES_PER_ARM = 13
H_ARM = N_PLIES_PER_ARM * PLY_THICKNESS  # ~ 2.020 mm

SPAN_FULL = 203.2          # mm (2L = 8 in)
HALF_SPAN = SPAN_FULL / 2.0  # 101.6 mm (= L in Carlsson-Pipes notation)
OVERHANG = (L_TOTAL - SPAN_FULL) / 2.0  # 25.4 mm per side

# Crack-tip position measured from the *nearest support* (the FEP-end
# support) is 2 in = 50.8 mm -> the FEP-bonded boundary in mesh-native
# x-coordinates lies at OVERHANG + A0_FROM_SUPPORT = 76.2 mm.
A0_FROM_SUPPORT = 50.8     # mm (2 in into span from the nearest support)
A_FEP_END = OVERHANG + A0_FROM_SUPPORT  # 76.2 mm = 3 in (FEP length)

# Mesh
NX = 200                   # ~1.27 mm elements; cf. DCB NX=150 (~1.69 mm)
NY = 1
NZ_PER_ARM = 2             # total nz = 4, interface at z = 0 (mid-plane)

# Measured cohesive toughness from NASA TM
GIIC_MEASURED = 0.777      # N/mm (mode II from ENF)
GIC_MEASURED = 0.324       # N/mm (from DCB on same panel; required by
                           # bilinear law, irrelevant in pure mode II)

# Penalty stiffness same as DCB / ENF monotonic tests
K_PENALTY = 1.0e6          # N/mm^3

# tau_max chosen to give a developed Mode-II cohesive zone inside the
# bonded region.  With tau_max = 80 MPa,
#     lambda_cz_II = E1 * GIIc / tau_max^2 = 171420 * 0.777 / 6400
#                  = 20.81 mm
# which is ~14 % of the 152 mm bonded region length -- short enough
# for the CZ to fully develop before the strength criterion fires.
# sigma_max is irrelevant in mode II but the bilinear law needs a
# value; set equal to tau_max for symmetry.
TAU_MAX = 80.0             # MPa
SIGMA_MAX = 80.0           # MPa (mode-II irrelevant; cf. ENF monotonic)

# Loading: 200 fixed equal displacement increments, NO adaptive sub-
# stepping (Phase 7 DCB finding -- adaptive driver overshoots peak via
# its step-growth-on-success logic).
N_INCREMENTS = 200

# DELTA_MAX = 10 mm puts the ramp ~50 % past the analytical peak
# (delta_at_P_c ~ 6.2 mm via Carlsson-Pipes); leaves a comfortable
# tail for post-peak observation if any propagation occurs.
DELTA_MAX = 10.0           # mm


# ----------------------------------------------------------------------
# Material
# ----------------------------------------------------------------------


def _build_material():
    """Fetch the IM7_8552 elastic properties from the canonical library."""
    return MaterialLibrary().get("IM7_8552")


MAT = _build_material()


def _build_cohesive_properties() -> CohesiveProperties:
    """Construct the bilinear law parameters for mode-II ENF validation."""
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
# Analytical helpers (Carlsson-Pipes ENF beam theory)
# ----------------------------------------------------------------------


def _enf_compliance(a: float = A0_FROM_SUPPORT) -> float:
    """ENF beam-theory compliance (Carlsson-Pipes, Eq. 6.13):

        C = (2 L^3 + 3 a^3) / (8 b E1 h^3)

    where L = half-span, a = crack length, b = width, h = arm
    half-thickness.  Linear-elastic, pre-damage; ignores transverse
    shear (~10 % correction for this h/L ratio).
    """
    return (2.0 * HALF_SPAN ** 3 + 3.0 * a ** 3) / (
        8.0 * WIDTH * MAT.E1 * (H_ARM ** 3)
    )


def _enf_peak_load(a: float = A0_FROM_SUPPORT) -> float:
    """Critical (peak) load for the ENF specimen at crack length ``a``:

        P_c = sqrt(16 b^2 E1 h^3 G_IIc / (9 a^2))

    Derived from G_II = (P^2 / (2b)) * dC/da with the ENF beam-theory
    compliance.  Independent of half-span L.
    """
    num = 16.0 * (WIDTH ** 2) * MAT.E1 * (H_ARM ** 3) * GIIC_MEASURED
    den = 9.0 * (a ** 2)
    return math.sqrt(num / den)


# ----------------------------------------------------------------------
# Mesh / model construction
# ----------------------------------------------------------------------


def _build_mesh(
    coh_props: CohesiveProperties,
) -> tuple[MeshData, list, list[bool]]:
    """Build the NASA-TM ENF mesh + cohesive list + bonded/pre-crack mask.

    The structured hex8 mesh is generated as two stacked plies (each one
    arm thick) so the interface plane z = 0 lands on the ply interface.
    All midplane interface nodes are duplicated by
    :func:`insert_cohesive_interface`, then cohesive elements are
    partitioned into three groups by their mid-surface x:

      * Bonded region [A_FEP_END, L_TOTAL]: cohesive law active (d = 0
        initially); these elements can damage and grow the crack.
      * FEP pre-crack [0, A_FEP_END): cohesive elements kept but pre-
        damaged to ``d = 1`` in :func:`_build_assembler` to act as
        frictionless contact (resist closure, zero shear, zero tension).
        This is the same treatment as ``test_enf_monotonic.py``.

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
        if x_mid >= A_FEP_END:
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify the support nodes (bottom face) and load nodes (top face).

    Returns
    -------
    left_support_nodes : nodes on the bottom face at x = OVERHANG (the
        FEP-side support).
    right_support_nodes : nodes on the bottom face at x = L_TOTAL - OVERHANG
        (the far-side support).
    center_load_nodes : nodes on the top face at x = L_TOTAL / 2 (the
        centerline downward load).
    """
    tol = 1e-6
    x = mesh.nodes[:, 0]
    z = mesh.nodes[:, 2]
    z_min = float(z.min())
    z_max = float(z.max())

    on_z_min = np.abs(z - z_min) <= tol
    on_z_max = np.abs(z - z_max) <= tol

    on_left_x = np.abs(x - OVERHANG) <= tol
    on_right_x = np.abs(x - (L_TOTAL - OVERHANG)) <= tol
    on_center_x = np.abs(x - 0.5 * L_TOTAL) <= tol

    left_support = np.flatnonzero(on_left_x & on_z_min).astype(np.intp)
    right_support = np.flatnonzero(on_right_x & on_z_min).astype(np.intp)
    center_load = np.flatnonzero(on_center_x & on_z_max).astype(np.intp)
    return left_support, right_support, center_load


def _build_bcs(
    mesh: MeshData,
    delta: float,
) -> list[BoundaryCondition]:
    """Three-point-bend BCs matching the NASA TM ENF fixture.

    - Bottom support 1 (FEP side) at x = OVERHANG: pin u_z on the full
      bottom-face line.  One node also pinned in u_x to remove the x
      rigid-body translation.
    - Bottom support 2 at x = L_TOTAL - OVERHANG: pin u_z on the full
      bottom-face line.
    - Top center roller at x = L_TOTAL / 2: prescribe u_z = -delta on
      the full top-face line at the centerline (downward).
    - One node pinned in u_y to remove the y rigid-body translation
      (re-uses the same x-pin node).
    """
    left_support, right_support, center_load = _support_and_load_nodes(mesh)

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
            bc_type="displacement", node_ids=center_load,
            dofs=[2], value=-float(delta),
        ),
    ]


# ----------------------------------------------------------------------
# Fixed-increment Newton driver (no adaptive sub-stepping)
# ----------------------------------------------------------------------


def _drive_enf_fixed(
    mesh: MeshData,
    cohesive_elements: list,
    is_bonded: list[bool],
    delta_max: float,
    n_increments: int,
    verbose: bool = False,
) -> dict:
    """Drive the ENF through N fixed equal displacement increments.

    Matches the Phase 7 DCB driver: no step halving, no step growth on
    success -- just N equal increments.  If a Newton step fails, the
    increment is skipped (u and committed state unchanged) and the driver
    moves on; the failure count is returned for diagnostics.

    Per converged increment we record:
      * The centerline reaction load (sum of internal-force z-components
        at the top-roller load nodes, absolute value for magnitude).
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

    _, _, center_load_nodes = _support_and_load_nodes(mesh)
    center_z_dofs = 3 * center_load_nodes + 2

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
        P_center = float(np.sum(F_int[center_z_dofs]))
        P_load = abs(P_center)

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
                f"  inc {i + 1:3d}: delta={delta_try:.4f}, P={P_load:8.2f}, "
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

    # Experimental peak load with +/- 10 % shaded band (per-coupon
    # scatter estimate -- the TM only publishes the average so we use
    # the +/- 10 % bracket as a visual scatter proxy).
    exp_band_lo = EXPERIMENTAL_P_CRITICAL_N * 0.90
    exp_band_hi = EXPERIMENTAL_P_CRITICAL_N * 1.10
    x_max = float(cd[-1]) * 1.02
    ax.axhspan(
        exp_band_lo, exp_band_hi, color="tab:orange", alpha=0.15,
        label=(
            f"Experimental scatter band ($\\pm$10 %): "
            f"[{exp_band_lo:.0f}, {exp_band_hi:.0f}] N"
        ),
    )
    ax.axhline(
        EXPERIMENTAL_P_CRITICAL_N, color="tab:orange",
        linestyle="--", linewidth=2.0,
        label=(
            f"Experimental $P_\\mathrm{{critical}}$ = "
            f"{EXPERIMENTAL_P_CRITICAL_N:.0f} N "
            f"(NASA/TM-2020-220498 §4.15)"
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
        xytext=(delta_peak + 0.5, P_peak + 30.0),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
    )

    ax.set_xlabel("Top-roller displacement, $\\delta$ [mm]")
    ax.set_ylabel("Applied load, $P$ [N]")
    ax.set_title(
        "NASA/TM-2020-220498 ENF — Predicted vs Experimental\n"
        f"IM7/8552, h$_\\mathrm{{arm}}$ = {H_ARM:.2f} mm (calibrated), "
        f"a$_0$ = {A0_FROM_SUPPORT:.1f} mm, "
        f"G$_\\mathrm{{IIc}}$ = {GIIC_MEASURED:.3f} N/mm"
    )
    ax.set_xlim(0.0, x_max)
    y_top = max(float(cP.max()), EXPERIMENTAL_P_CRITICAL_N) * 1.20
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
    """Linear-fit slope ``dP/d delta`` from the first elastic part of the
    ramp (delta in (0, 10 % of the first damage onset]).

    We use a fraction of the first 10 % of the ramp by default; this
    captures elastic stiffness before any cohesive softening starts.
    """
    # Skip the (0, 0) point at index 0.
    if deltas.size < 4:
        return 0.0
    # Use the first sample with delta > 0 up to the first 10 % of the
    # full ramp -- comfortably below the experimental peak at delta
    # ~ 6.2 mm.
    cutoff = 0.10 * DELTA_MAX
    mask = (deltas > 1e-9) & (deltas <= cutoff)
    if mask.sum() < 2:
        # Fallback: use only the first non-zero point against origin.
        idx = int(np.argmax(deltas > 1e-9))
        return float(P_arr[idx] / deltas[idx]) if deltas[idx] > 0 else 0.0
    d = deltas[mask]
    p = P_arr[mask]
    # Constrain the fit through the origin: m = sum(d * p) / sum(d * d).
    return float(np.sum(d * p) / np.sum(d * d))


def test_enf_experimental_validation_nasa_tm():
    """Compare the CZM prediction to NASA/TM-2020-220498 ENF data.

    Validates mode-II behaviour against the same NASA TM panel used by
    the DCB validation test.  Compliance and peak load are the
    quantitative checks; damage existence + post-peak stability are
    structural sanity checks (the latter likely fails due to the
    known mode-II-under-compression suppression -- limitation #6.1
    in CZM_PLAN.md).
    """
    coh_props = _build_cohesive_properties()
    mesh, cohesive_elements, is_bonded = _build_mesh(coh_props)

    # Element-count sanity check: the bonded region is [A_FEP_END,
    # L_TOTAL] of length 177.8 mm; element width is L_TOTAL / NX.
    n_bonded_expected = sum(
        1 for i in range(NX) if (i + 0.5) * (L_TOTAL / NX) >= A_FEP_END
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

    res = _drive_enf_fixed(
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
    C_analytical = _enf_compliance(A0_FROM_SUPPORT)
    slope_analytical = 1.0 / C_analytical
    slope_rel = (
        abs(slope_pred - slope_analytical) / slope_analytical
        if slope_analytical > 0.0 else float("inf")
    )
    slope_in_band = slope_rel < 0.15

    # (2) Peak load.
    P_peak_pred = float(cP.max())
    i_peak = int(np.argmax(cP))
    delta_peak = float(cd[i_peak])
    peak_in_band = PEAK_LOAD_LO_N <= P_peak_pred <= PEAK_LOAD_HI_N
    peak_rel = (
        abs(P_peak_pred - EXPERIMENTAL_P_CRITICAL_N)
        / EXPERIMENTAL_P_CRITICAL_N
    )

    # (3) Damage existence.
    d_max_final = float(cdmax.max())
    damage_exists = d_max_final > 0.5

    # (4) Post-peak stability: max upward jump after the peak should be
    # less than 10 % of P_peak.  Soft check -- likely to fail due to
    # mode-II-under-compression suppression collapsing the cohesive zone
    # to one element wide.
    if i_peak < len(cP) - 2:
        cP_post = cP[i_peak:]
        diffs_post = np.diff(cP_post)
        max_up_jump = float(diffs_post.max()) if diffs_post.size else 0.0
    else:
        max_up_jump = 0.0
    stability_threshold = 0.10 * P_peak_pred
    stability_in_band = max_up_jump <= stability_threshold

    # Analytical peak from Carlsson-Pipes for reference.
    P_c_analytical = _enf_peak_load(A0_FROM_SUPPORT)

    # CZ length for the diagnostic print.
    lambda_cz_II = MAT.E1 * GIIC_MEASURED / (TAU_MAX ** 2)

    # Write the plot regardless of assertion outcomes -- user-facing
    # deliverable.
    out_path = Path(__file__).resolve().parents[2] / "figures" / (
        "phase7_enf_validation.png"
    )
    _save_comparison_plot(res, out_path)

    print(
        f"\nPhase 7 ENF validation (NX={NX}, tau_max={TAU_MAX:.1f} MPa, "
        f"GIIc={GIIC_MEASURED:.3f} N/mm, h_arm={H_ARM:.3f} mm):\n"
        f"  (1) slope     = {slope_pred:8.2f} N/mm "
        f"(analytical {slope_analytical:.2f} N/mm, rel {slope_rel:.2%}, "
        f"tol 15 %)  {'PASS' if slope_in_band else 'FAIL'}\n"
        f"  (2) P_peak    = {P_peak_pred:8.2f} N "
        f"(exp {EXPERIMENTAL_P_CRITICAL_N:.0f} N "
        f"[{PEAK_LOAD_LO_N:.0f}, {PEAK_LOAD_HI_N:.0f}], rel "
        f"{peak_rel:.2%}, beam-theory {P_c_analytical:.0f} N)  "
        f"{'PASS' if peak_in_band else 'FAIL'}\n"
        f"  (3) max d     = {d_max_final:8.3f} "
        f"(threshold > 0.50)  "
        f"{'PASS' if damage_exists else 'FAIL'}\n"
        f"  (4) max up    = {max_up_jump:8.2f} N "
        f"(limit {stability_threshold:.2f} N, 10 % of P_peak; SOFT CHECK)  "
        f"{'PASS' if stability_in_band else 'FAIL'}\n"
        f"  delta_peak = {delta_peak:.3f} mm "
        f"(beam-theory @ exp peak: {C_analytical * EXPERIMENTAL_P_CRITICAL_N:.3f} mm)\n"
        f"  lambda_cz_II = {lambda_cz_II:.2f} mm "
        f"(bonded length = {L_TOTAL - A_FEP_END:.2f} mm)\n"
        f"  Newton failures = {res['n_fails']} / {N_INCREMENTS} increments\n"
        f"  Plot: {out_path}"
    )

    assert out_path.is_file(), f"Comparison plot was not written to {out_path}"

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    # (1) Initial elastic compliance within 15 % of analytical
    assert slope_in_band, (
        f"Initial elastic slope {slope_pred:.2f} N/mm off analytical "
        f"{slope_analytical:.2f} N/mm by {slope_rel:.2%} (tol 15 %)"
    )

    # (2) Peak load within experimental scatter band
    assert peak_in_band, (
        f"Predicted peak load {P_peak_pred:.2f} N outside band "
        f"[{PEAK_LOAD_LO_N:.2f}, {PEAK_LOAD_HI_N:.2f}] N (experimental "
        f"P_critical = {EXPERIMENTAL_P_CRITICAL_N:.0f} N, rel "
        f"{peak_rel:.2%})"
    )

    # (3) At least one cohesive element reached d > 0.5
    assert damage_exists, (
        f"No cohesive element reached d > 0.5 during the ramp "
        f"(max d = {d_max_final:.3f}).  Likely the mode-II-under-"
        f"compression suppression (CZM_PLAN.md known limitation #6.1) "
        f"collapsed the cohesive zone."
    )

    # (4) Post-peak stability -- soft check, likely to fail due to
    # mode-II-under-compression suppression.
    assert stability_in_band, (
        f"Post-peak upward jump {max_up_jump:.2f} N exceeds 10 % of "
        f"P_peak (= {stability_threshold:.2f} N).  Indicates numerical "
        f"instability in the post-peak branch, consistent with the "
        f"known mode-II-under-compression damage suppression."
    )


if __name__ == "__main__":
    # Allow ad-hoc invocation without pytest's capture.
    os.environ.setdefault("WRINKLEFE_ENF_VERBOSE", "1")
    test_enf_experimental_validation_nasa_tm()
