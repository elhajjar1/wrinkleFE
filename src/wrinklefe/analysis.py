"""High-level analysis pipeline for wrinkled composite laminates.

Provides :class:`WrinkleAnalysis`, a one-stop orchestrator that chains:

1. Laminate definition (material + stacking sequence)
2. Wrinkle geometry and morphology configuration
3. Mesh generation
4. Static FE solve
5. Failure evaluation
6. Optional buckling analysis
7. Optional Monte Carlo / Jensen gap statistics

This module is the primary user-facing entry point for typical workflows.
All lower-level modules (core, elements, solver, failure, statistics) are
accessed through this pipeline.

Examples
--------
Minimal compression analysis::

    >>> from wrinklefe.analysis import WrinkleAnalysis, AnalysisConfig
    >>> config = AnalysisConfig(
    ...     amplitude=0.366, wavelength=16.0, width=12.0,
    ...     morphology="concave", loading="compression",
    ... )
    >>> analysis = WrinkleAnalysis(config)
    >>> result = analysis.run()
    >>> print(result.summary())

References
----------
- Elhajjar, R. (2025). Scientific Reports, 15:25977 (fat-tail statistics).
- Jin, L. et al. (2026). Thin-Walled Structures, 219:114237 (wrinkle geometry).
- Budiansky, B. & Fleck, N.A. (1993). J. Mech. Phys. Solids, 41(1), 183-211.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from wrinklefe.core.cohesive_mesh import insert_cohesive_interface
from wrinklefe.core.laminate import Laminate, LoadState
from wrinklefe.core.layup import validate_ply_angle
from wrinklefe.core.material import MaterialLibrary, OrthotropicMaterial
from wrinklefe.core.mesh import MeshData, WrinkleMesh
from wrinklefe.core.morphology import (
    MORPHOLOGY_PHASES,
    SINGLE_WRINKLE_MODES,
    WrinkleConfiguration,
    WrinklePlacement,
)
from wrinklefe.core.penetration_gate import (
    GATE_PRESETS,
    GateParameters,
    penetration_gate_kd,
)
from wrinklefe.core.transforms import rotate_stiffness_3d
from wrinklefe.core.wrinkle import (
    GaussianSinusoidal,
)
from wrinklefe.elements.cohesive8 import (
    Cohesive8Element,
    CohesiveProperties,
    make_initial_state,
)
from wrinklefe.failure.delamination import build_delamination_report
from wrinklefe.failure.evaluator import FailureEvaluator, LaminateFailureReport
from wrinklefe.solver.assembler import GlobalAssembler
from wrinklefe.solver.boundary import BoundaryHandler
from wrinklefe.solver.nonlinear import NewtonRaphsonSolver
from wrinklefe.solver.results import FieldResults
from wrinklefe.solver.static import StaticSolver

logger = logging.getLogger(__name__)

# Analytical damage model constants (Section 6 of CLAUDE.md)
_D0 = 0.15       # Base damage coefficient
_BETA_ANGLE = 3.0  # Angle sensitivity
_THETA_CRIT = 0.1  # Critical angle (rad)
_A_REF = 0.183    # Reference amplitude (1 ply thickness, mm)

# Number of x-integration points for profile-proportional knockdown
_N_PROFILE_PTS = 500

# Confinement model constants
# Calibrated with CLT-weighted BF against Elhajjar (2025), T700/2510, and
# Mukhopadhyay (2015) blocked-layup compression cases.
_GAMMA_Y_UD = 0.032   # UD matrix yield strain (no confinement)
_ALPHA_CONF = 0.050   # confinement boost coefficient (per off-axis-neighbour score)
# Block-size penalty: each additional 0-deg ply in a consecutive run beyond
# the first reduces gamma_Y_eff by this amount. Captures the empirical
# observation that blocked layups such as Mukhopadhyay's [0_2] (effectively
# [0_4] across the symmetry plane) kink more easily than the neighbour-
# counting confinement score alone predicts: inner 0-deg faces of a block
# are bracketed by another 0-deg ply that does not constrain lateral
# expansion of the kink band. Only applied when at least one off-axis ply
# exists so pure UD ([0]_n) remains at the UD calibration point.
_BETA_BLOCK = 0.010   # per-extra-ply block penalty on gamma_Y_eff
# Lower bound on gamma_Y_eff so a long 0-deg block cannot drive it negative
# or arbitrarily close to zero (which would otherwise produce a degenerate
# Budiansky-Fleck knockdown).
_GAMMA_Y_FLOOR = _GAMMA_Y_UD / 2.0  # = 0.016 (UD half-strain floor)


def _confined_fraction(angles: list[float], tol: float = 5.0) -> float:
    """Weighted confinement fraction for 0-degree plies.

    Each 0-degree ply is scored by how many of its neighbors are off-axis:
        - Both neighbors off-axis (or free surface): score = 1.0
        - One neighbor off-axis, one neighbor 0-deg:  score = 0.5
        - Both neighbors 0-deg (block interior):      score = 0.0

    This partial-confinement model correctly handles both dispersed
    layups (e.g., [0/45/90/-45]) and blocked layups (e.g., [0_2/90_2]).

    Used to compute the effective matrix yield strain:
        gamma_Y_eff = gamma_Y_UD + alpha * f_confined

    Calibrated (with CLT-weighted compression) against:
        - UD [0]_n:       f=0.000, gamma_Y=0.032
        - Mukhopadhyay (2015): f=0.417, gamma_Y=0.053  (blocked [0_2])
        - Elhajjar (2025): f=0.833, gamma_Y=0.074  (dispersed)
    """
    n_0 = sum(1 for a in angles if abs(a) < tol)
    if n_0 == 0:
        return 0.0
    score = 0.0
    for i, a in enumerate(angles):
        if abs(a) < tol:
            left_ok = (abs(angles[i - 1]) > tol) if i > 0 else True
            right_ok = (abs(angles[i + 1]) > tol) if i < len(angles) - 1 else True
            if left_ok and right_ok:
                score += 1.0
            elif left_ok or right_ok:
                score += 0.5
            # else: both neighbors are 0-deg → score += 0.0
    return score / n_0


def _effective_gamma_Y(angles: list[float]) -> float:
    """Compute layup-dependent effective matrix yield strain.

    Three-parameter model::

        gamma_Y_eff = max(
            gamma_Y_UD + alpha_conf * f_confined
                       - beta_block * max(n_block_max - 1, 0),
            gamma_Y_floor,
        )

    where:

    * ``f_confined`` is the weighted confinement fraction of 0-degree
      plies (0 = unconfined, 1 = fully interspersed; see
      :func:`_confined_fraction`).  The linear ``alpha_conf`` term
      captures the constraint that off-axis plies impose on kink-band
      lateral expansion in multidirectional laminates.
    * ``n_block_max`` is the longest run of consecutive 0-degree plies
      (see :func:`_max_consecutive_zero_plies`).  The ``beta_block``
      term penalises long 0-deg blocks: each additional ply inside a
      block beyond the first contributes another increment of
      lateral-expansion freedom that the neighbour-counting confinement
      score does not capture.  Inner 0-deg faces of a block are
      bracketed by another 0-deg ply that does not constrain kink-band
      lateral expansion, so the matrix yields at a lower applied shear
      strain in blocked layups than in dispersed layups with the same
      ``f_confined``.

    The block penalty is only applied when at least one off-axis ply
    exists in the layup.  Pure UD ``[0]_n`` would otherwise be driven
    below the calibration point by the penalty term; with the guard, UD
    retains ``gamma_Y_eff = gamma_Y_UD = 0.032`` regardless of ``n``.

    The result is floored at ``_GAMMA_Y_FLOOR`` (= gamma_Y_UD / 2) so
    very thick 0-blocks cannot drive ``gamma_Y_eff`` arbitrarily close
    to zero, which would otherwise produce a degenerate Budiansky-Fleck
    knockdown.

    With CLT-weighted compression (``KD_lam = f0 * KD_BF + (1 - f0)``),
    the confinement effect is separated from load redistribution.

    Calibration anchors (three-parameter model, beta_block = 0.010):

    ======================================  =======  ===============  ==========
    Layup                                   ``f``    ``n_block_max``  ``gamma_Y``
    ======================================  =======  ===============  ==========
    UD ``[0]_n``                            ~0.13    n (guard skips)  0.032
    Mukhopadhyay ``[..../0_2]_3s``          ~0.42    4 (block of 4    ~0.023
                                                     at symmetry
                                                     plane)
    Elhajjar ``[0/45/90/-45/0/45/-45/0]_s`` ~0.83    2 (only at the   ~0.064
                                                     symmetry plane)
    ======================================  =======  ===============  ==========
    """
    fc = _confined_fraction(angles)
    # Guard: pure UD has no off-axis plies; the block penalty would
    # otherwise drive its gamma_Y below the calibration point.
    n_off_axis = sum(1 for a in angles if abs(a) >= 5.0)
    if n_off_axis == 0:
        return _GAMMA_Y_UD
    n_block_max = _max_consecutive_zero_plies(angles)
    block_penalty = _BETA_BLOCK * max(n_block_max - 1, 0)
    gamma_Y = _GAMMA_Y_UD + _ALPHA_CONF * fc - block_penalty
    return max(gamma_Y, _GAMMA_Y_FLOOR)


def _profile_proportional_kd(
    amplitude: float,
    wavelength: float,
    width: float,
    domain_length: float,
    ply_thickness: float,
    n_plies: int,
    gamma_Y: float,
    theta_max: float,
    *,
    morphology_factor: float = 1.0,
    through_thickness_decay: bool = True,
    z_position_fraction: float = 0.5,
    decay_scale: float | None = None,
    decay_floor: float = 0.0,
    kink_band_quadratic_coeff: float = 0.0,
) -> float:
    """Budiansky-Fleck knockdown averaged over the wrinkle profile.

    Instead of applying a single peak-angle knockdown to all plies,
    this function computes the local fibre angle at every (x, z) point
    in the laminate and averages the Budiansky-Fleck response:

        KD_lam = (1/N) * sum_p [ (1/L_s) * int KD(x, z_p) dx ]

    where the local angle at position (x, z_p) is:

        theta(x, z_p) = |dz_w/dx| * M_f * Phi(z_p)

    with:
        |dz_w/dx|  = slope of the GaussianSinusoidal wrinkle profile
        M_f        = morphology factor (accounts for dual-wrinkle interaction)
        Phi(z_p)   = exp(-(z_p - z_c)^2 / (2 * sigma^2))  (through-thickness
                     decay, standard Gaussian convention with sigma =
                     ``decay_scale``).  This differs from the legacy form
                     ``exp(-(z_p - T/2)^2 / A^2)`` (no factor of 2 — implicit
                     ``sqrt(2)*A`` scale): the new form makes the decay
                     scale an explicit standard deviation.
        z_c        = z_position_fraction * T (laminate thickness)

    The decay scale defaults to ``max(wavelength / 2, amplitude)`` when
    not provided.  The wrinkle's longitudinal extent (set by the
    wavelength) is the physical scale over which a buried wrinkle
    perturbs the through-thickness fibre orientation field: a short-
    wavelength wrinkle decays through only a few plies; a long-wavelength
    wrinkle reaches further.  The legacy ``A``-based scale almost always
    falls inside this default for the calibrated datasets, but for thick
    UD laminates with long wavelengths (e.g. Li 2024) the legacy form
    confined the wrinkle effect to just the midplane plies, leaving
    the laminate KD near 1 even at high amplitudes.

    When *through_thickness_decay* is False, Phi(z_p) = 1 for all plies
    (all plies see the same longitudinal profile).  This is appropriate
    for dual-wrinkle morphologies (stack/convex/concave) where the wrinkle
    extends through the full thickness.

    The per-point knockdown uses the Argon-Fleck quadratic extension of
    the Budiansky-Fleck closed form::

        r = theta_eff / gamma_Y
        KD = 1.0 / (1.0 + r + c_AF * r^2)

    With ``c_AF = 0`` (default) the legacy linear form is recovered.
    Non-zero ``c_AF`` improves the high-angle response (theta > ~20 deg)
    where the linear form systematically over-predicts strength.

    Parameters
    ----------
    amplitude : float
        Wrinkle amplitude A [mm].
    wavelength : float
        Full sinusoidal wavelength lambda [mm].
    width : float
        Gaussian envelope half-width w [mm].
    domain_length : float
        Specimen / domain length L_s [mm].
    ply_thickness : float
        Ply thickness [mm].
    n_plies : int
        Total number of plies.
    gamma_Y : float
        Effective matrix yield shear strain.
    theta_max : float
        Maximum unattenuated fibre angle [rad].
    morphology_factor : float
        Morphology factor M_f that scales the effective angle to account
        for dual-wrinkle interaction (convex < 1.0, concave > 1.0,
        stack = 1.0, graded = 1.0).  Default 1.0.
    through_thickness_decay : bool
        If True (default), apply Gaussian through-thickness decay centred
        at ``z_position_fraction * T`` with scale ``decay_scale`` (see
        below).  If False, all plies see the full wrinkle angle profile
        (Phi = 1).
    z_position_fraction : float
        Fraction of the laminate thickness at which the wrinkle through-
        thickness decay is centred.  ``0.5`` (default) centres the decay
        at the midplane, reproducing the legacy behaviour; ``0.0`` and
        ``1.0`` place the decay centre at the bottom and top surfaces,
        respectively.  Only consulted when ``through_thickness_decay`` is
        True.
    decay_scale : float or None
        Through-thickness Gaussian standard deviation [mm].  When None
        (default) the auto formula ``max(wavelength / 2, amplitude)`` is
        used.  Must be strictly positive when provided.
    decay_floor : float
        Minimum fraction of the wrinkle angle retained at any ply
        (issue #254): the through-thickness term becomes
        ``decay_floor + (1 - decay_floor) * raw`` with ``raw`` the
        Gaussian above — the same floor semantics the tension graded
        path applies, so a sign-flipped load sees the same envelope.
        ``0.0`` (default) reproduces the legacy pure-Gaussian decay
        bit-for-bit; ``1.0`` disables the decay (every ply sees the
        full angle).  Caller is responsible for the [0, 1] range
        (``AnalysisConfig`` validates it).
    kink_band_quadratic_coeff : float
        Argon-Fleck quadratic coefficient ``c_AF`` (dimensionless).
        Default 0.0 (legacy linear BF).  Must be >= 0.

    Returns
    -------
    float
        Profile-averaged BF knockdown factor (0, 1].
    """
    T = n_plies * ply_thickness
    z_center = z_position_fraction * T
    L_s = domain_length

    # Resolve the through-thickness decay scale.  The auto default uses
    # the wrinkle's longitudinal extent (lambda / 2) so long-wavelength
    # wrinkles reach further through the thickness; falls back to A so
    # short-wavelength wrinkles do not collapse to an unphysically small
    # decay scale.
    if decay_scale is None:
        sigma = max(wavelength / 2.0, amplitude)
    else:
        sigma = float(decay_scale)
    sigma_sq2 = 2.0 * sigma * sigma
    c_AF = float(kink_band_quadratic_coeff)

    # Longitudinal profile: compute |dz/dx| at each x-point
    x = np.linspace(-L_s / 2.0, L_s / 2.0, _N_PROFILE_PTS)

    # z(x) = A * exp(-x^2/w^2) * cos(2*pi*x/lambda)
    gauss_env = np.exp(-(x ** 2) / (width ** 2))
    cos_term = np.cos(2.0 * np.pi * x / wavelength)

    # Slope via analytical derivative (more accurate than np.gradient)
    sin_term = np.sin(2.0 * np.pi * x / wavelength)
    dz_dx = amplitude * gauss_env * (
        (-2.0 * x / (width ** 2)) * cos_term
        - (2.0 * np.pi / wavelength) * sin_term
    )
    theta_x = np.abs(np.arctan(dz_dx)) * morphology_factor  # M_f-scaled angle

    # Average over plies and x-positions
    kd_sum = 0.0
    for p in range(n_plies):
        z_p = (p + 0.5) * ply_thickness
        if through_thickness_decay:
            raw_p = np.exp(-((z_p - z_center) ** 2) / sigma_sq2)
            phi_p = decay_floor + (1.0 - decay_floor) * raw_p
        else:
            phi_p = 1.0
        theta_xz = theta_x * phi_p  # local angle at (x, z_p)
        r = theta_xz / gamma_Y
        kd_xz = 1.0 / (1.0 + r + c_AF * r * r)
        kd_sum += np.mean(kd_xz)

    return kd_sum / n_plies


def _is_unidirectional(angles: Sequence[float], tol: float = 5.0) -> bool:
    """True when every ply is a 0-degree (axial) ply within ``tol`` degrees.

    0 and 180 degrees are equivalent fibre directions. Used to dispatch
    the analytical modulus knockdown to its scalar unidirectional fast path
    (:func:`_profile_modulus_knockdown`); multidirectional layups take the
    laminate generalization (:func:`_laminate_modulus_knockdown`) instead.
    """
    if not angles:
        return False
    for a in angles:
        off = abs(float(a)) % 180.0
        if min(off, 180.0 - off) > tol:
            return False
    return True


def _profile_modulus_knockdown(
    amplitude: float,
    wavelength: float,
    width: float,
    domain_length: float,
    ply_thickness: float,
    n_plies: int,
    E1: float,
    E2: float,
    G12: float,
    nu12: float,
    *,
    morphology_factor: float = 1.0,
    through_thickness_decay: bool = True,
    z_position_fraction: float = 0.5,
    decay_scale: float | None = None,
    decay_floor: float = 0.0,
) -> float:
    r"""Axial Young's-modulus knockdown ``E_x / E_x0`` for a wavy UD laminate.

    A Classical-Lamination-Theory series-average of the off-axis lamina
    modulus over the wrinkle profile — the same off-axis-compliance
    integration as Hsiao & Daniel (1996, Compos. Sci. Technol. 56:581).
    At every ``(x, z_p)`` the local fibre tilt is the same field used by
    :func:`_profile_proportional_kd`,
    ``theta(x, z_p) = |dz_w/dx| * M_f * Phi(z_p)``, and the off-axis axial
    modulus of a 0-degree ply tilted by ``theta`` is::

        1/E_x(theta) = cos^4/E1 + (1/G12 - 2 nu12/E1) cos^2 sin^2 + sin^4/E2

    The plies share the membrane strain, so the section modulus at a
    station is the through-thickness mean
    ``E_sec(x) = <E_x(theta(x, z_p))>_p``; along the load direction the
    compliances add, so the effective modulus is the series (harmonic)
    average ``E_eff = 1 / <1/E_sec(x)>_x``. The knockdown is
    ``E_eff / E1`` (the pristine UD axial modulus is ``E1``).

    Linear-elastic, hence loading-independent. This closed form assumes a
    0-degree base ply, so it covers single-wrinkle **unidirectional** axial
    layups exactly (see :func:`_is_unidirectional`). Multidirectional and
    multi-wrinkle configurations are handled by the laminate generalization
    :func:`_laminate_modulus_knockdown`, which reduces to this result for
    ``[0]_n``; this scalar form is retained as the UD fast path so the
    pinned UD baselines stay numerically identical.

    Returns
    -------
    float
        Axial-modulus knockdown in ``(0, 1]``.
    """
    T = n_plies * ply_thickness
    z_center = z_position_fraction * T
    L_s = domain_length
    if decay_scale is None:
        sigma = max(wavelength / 2.0, amplitude)
    else:
        sigma = float(decay_scale)
    sigma_sq2 = 2.0 * sigma * sigma

    # Same longitudinal angle field as the strength profile-average.
    x = np.linspace(-L_s / 2.0, L_s / 2.0, _N_PROFILE_PTS)
    gauss_env = np.exp(-(x ** 2) / (width ** 2))
    cos_term = np.cos(2.0 * np.pi * x / wavelength)
    sin_term = np.sin(2.0 * np.pi * x / wavelength)
    dz_dx = amplitude * gauss_env * (
        (-2.0 * x / (width ** 2)) * cos_term
        - (2.0 * np.pi / wavelength) * sin_term
    )
    theta_x = np.abs(np.arctan(dz_dx)) * morphology_factor

    shear_coupling = 1.0 / G12 - 2.0 * nu12 / E1
    Ex_sum = np.zeros_like(x)
    for p in range(n_plies):
        z_p = (p + 0.5) * ply_thickness
        if through_thickness_decay:
            raw_p = np.exp(-((z_p - z_center) ** 2) / sigma_sq2)
            phi_p = decay_floor + (1.0 - decay_floor) * raw_p
        else:
            phi_p = 1.0
        theta = theta_x * phi_p
        c2 = np.cos(theta) ** 2
        s2 = np.sin(theta) ** 2
        inv_Ex = c2 * c2 / E1 + shear_coupling * c2 * s2 + s2 * s2 / E2
        Ex_sum += 1.0 / inv_Ex
    E_sec = Ex_sum / n_plies              # through-thickness mean at each x
    E_eff = 1.0 / np.mean(1.0 / E_sec)    # series-average along x
    return float(E_eff / E1)


def _plane_stress_qbar_tilted(
    stiffness_3d: np.ndarray, phi_rad: float, theta_rad: float
) -> np.ndarray:
    """Plane-stress reduced stiffness for a ply rotated in-plane *and* tilted.

    The ply's full 3D stiffness ``[C]`` (material axes) is rotated by the
    in-plane orientation ``phi`` about the through-thickness ``z`` axis and
    by the out-of-plane wrinkle tilt ``theta`` about the transverse ``y``
    axis (the combination of the ply angle with the local wrinkle slope is
    a composition of these two principal-axis rotations,
    :func:`wrinklefe.core.transforms.rotate_stiffness_3d`). The rotated 3D
    stiffness is then statically condensed to plane stress
    (``sigma_33 = tau_23 = tau_13 = 0``) onto the in-plane Voigt indices
    ``(11, 22, 12)`` so it can enter the laminate membrane (A) matrix like a
    standard ``Q-bar``.

    For ``theta = 0`` this returns the ordinary CLT ``Q-bar(phi)`` and for
    ``phi = 0`` the axial term ``1/inv(Q-bar)[0, 0]`` equals the off-axis
    formula used by :func:`_profile_modulus_knockdown`, so the laminate
    knockdown reduces exactly to the UD scalar result for ``[0]_n``.
    """
    c_rot = rotate_stiffness_3d(stiffness_3d, phi_rad, axis="z")
    c_rot = rotate_stiffness_3d(c_rot, theta_rad, axis="y")
    # Voigt split: in-plane (11, 22, 12) vs out-of-plane (33, 23, 13).
    ip = [0, 1, 5]
    op = [2, 3, 4]
    c_ii = c_rot[np.ix_(ip, ip)]
    c_io = c_rot[np.ix_(ip, op)]
    c_oi = c_rot[np.ix_(op, ip)]
    c_oo = c_rot[np.ix_(op, op)]
    return np.asarray(c_ii - c_io @ np.linalg.solve(c_oo, c_oi))


def _laminate_modulus_knockdown(
    slope_field: np.ndarray,
    ply_decays: np.ndarray,
    angles: Sequence[float],
    stiffness_3d: np.ndarray,
    ply_thickness: float,
    E_x0: float,
) -> float:
    r"""Axial-modulus knockdown ``E_x / E_x0`` for an arbitrary wavy laminate.

    Generalizes :func:`_profile_modulus_knockdown` from a single 0-degree
    base ply to an arbitrary stacking sequence and to an already-composed
    multi-wrinkle slope field, via a CLT membrane (A-matrix) series-average
    — the laminate form of the Hsiao & Daniel (1996) off-axis-compliance
    integration.

    At every longitudinal station ``x`` each ply ``p`` carries a local fibre
    tilt ``theta(x, z_p) = arctan|sum_w (dz_w/dx) Phi_w(z_p)|`` (the composed
    slope field, decayed through the thickness exactly as the FE composes it
    in :meth:`WrinkleConfiguration.fiber_angles_at_nodes` — "compose then
    differentiate"). The ply's plane-stress stiffness is rotated to its
    in-plane angle ``phi_p`` plus that tilt
    (:func:`_plane_stress_qbar_tilted`) and summed into the membrane
    stiffness ``A(x) = sum_p Q-bar_p(phi_p, theta) * t``. The plies share the
    membrane strain, so the section axial modulus is
    ``E_x_section(x) = 1 / (a11(x) * T)`` with ``a11 = inv(A)[0, 0]`` and
    ``T`` the total thickness; along ``x`` the compliances add, giving the
    series (harmonic) average ``E_eff = 1 / <1/E_x_section(x)>_x``.

    The knockdown is ``E_eff / E_x0`` where ``E_x0`` is the **flat** laminate
    axial modulus (``Laminate.Ex``). Because the flat-laminate A-matrix is
    recovered exactly when every tilt is zero, the knockdown is exactly
    ``1.0`` for a degenerate (zero-amplitude) wrinkle and lies in ``(0, 1]``
    otherwise. Off-axis plies, already carrying little axial load, are
    insensitive to the axial misalignment, so a multidirectional layup is
    knocked down less than the same wrinkle in pure UD.

    Linear-elastic, hence loading-independent.

    The per-ply tilt is built as ``sum_w slope_field[w] * ply_decays[p, w]``
    — the signed per-wrinkle slopes, each scaled by that wrinkle's
    through-thickness decay, summed *before* the angle is taken. A single
    wrinkle is just the one-entry (``N_w == 1``) case of this composition.

    Parameters
    ----------
    slope_field : np.ndarray
        Shape ``(N_w, N_x)`` per-wrinkle longitudinal slope ``dz_w/dx``
        *before* the through-thickness decay, evaluated along the wrinkle.
    ply_decays : np.ndarray
        Shape ``(n_plies, N_w, N_x)`` through-thickness decay ``Phi_w(z_p)``
        applied to each wrinkle's slope for each ply.
    angles : Sequence[float]
        Ply in-plane orientations ``phi_p`` in **degrees**.
    stiffness_3d : np.ndarray
        The common ply ``6x6`` 3D stiffness ``[C]`` (material axes).
    ply_thickness : float
        Uniform ply thickness [mm].
    E_x0 : float
        Pristine flat-laminate axial modulus ``Laminate.Ex`` [MPa].

    Returns
    -------
    float
        Axial-modulus knockdown in ``(0, 1]``.
    """
    phis = [math.radians(float(a)) for a in angles]
    n_plies = len(phis)
    total_thickness = n_plies * ply_thickness

    # Compose the per-wrinkle slopes (each decayed for this ply) then
    # take the local tilt angle: theta(p, x) = arctan|sum_w dz_w/dx * Phi_w|.
    slope_px = np.einsum("wx,pwx->px", slope_field, ply_decays)
    theta_px = np.arctan(np.abs(slope_px))  # (n_plies, n_x)

    # The in-plane (z-axis) rotation depends only on the ply angle, so
    # rotate the base stiffness once per ply; the wrinkle tilt (y-axis)
    # rotation and the plane-stress condensation are batched over the
    # whole (ply, x) grid (issue #301: the previous per-(ply, x) Python
    # loop made every multidirectional analytical run take seconds).
    c_phi = np.stack(
        [rotate_stiffness_3d(stiffness_3d, phi, axis="z") for phi in phis]
    )  # (n_plies, 6, 6)

    # Batched y-axis stress-transformation matrices T_sigma(theta) and
    # the engineering-strain counterparts T_eps = R T_sigma R^-1
    # (row/column Reuter scaling), mirroring
    # wrinklefe.core.transforms.stress/strain_transformation_3d.
    def _tsigma_y(cos_t: np.ndarray, sin_t: np.ndarray) -> np.ndarray:
        c2, s2, sc = cos_t * cos_t, sin_t * sin_t, sin_t * cos_t
        t = np.zeros(cos_t.shape + (6, 6), dtype=float)
        t[..., 0, 0] = c2
        t[..., 0, 2] = s2
        t[..., 0, 4] = -2.0 * sc
        t[..., 1, 1] = 1.0
        t[..., 2, 0] = s2
        t[..., 2, 2] = c2
        t[..., 2, 4] = 2.0 * sc
        t[..., 3, 3] = cos_t
        t[..., 3, 5] = sin_t
        t[..., 4, 0] = sc
        t[..., 4, 2] = -sc
        t[..., 4, 4] = c2 - s2
        t[..., 5, 3] = -sin_t
        t[..., 5, 5] = cos_t
        return t

    cos_t = np.cos(theta_px)
    sin_t = np.sin(theta_px)
    t_sig = _tsigma_y(cos_t, sin_t)
    # Rotation transformations invert by angle negation:
    # T_sigma(theta)^-1 == T_sigma(-theta) — a matmul instead of a
    # batched 6x6 solve.
    t_sig_inv = _tsigma_y(cos_t, -sin_t)
    reuter = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    t_eps = t_sig * (reuter[:, None] / reuter[None, :])

    # C_rot = T_sigma^-1 @ C_phi @ T_eps, batched over (ply, x).
    c_rot = t_sig_inv @ (c_phi[:, None, :, :] @ t_eps)

    # Plane-stress condensation onto the in-plane Voigt indices
    # (11, 22, 12): Q-bar = C_ii - C_io @ C_oo^-1 @ C_oi.
    ip = np.array([0, 1, 5])
    op = np.array([2, 3, 4])
    c_ii = c_rot[..., ip[:, None], ip[None, :]]
    c_io = c_rot[..., ip[:, None], op[None, :]]
    c_oi = c_rot[..., op[:, None], ip[None, :]]
    c_oo = c_rot[..., op[:, None], op[None, :]]
    q_bar = c_ii - c_io @ np.linalg.solve(c_oo, c_oi)

    # Membrane A(x) = sum_p Q-bar_p * t; shared membrane strain gives
    # E_section(x) = 1 / (inv(A)[0,0] * T).
    a_mat = q_bar.sum(axis=0) * ply_thickness  # (n_x, 3, 3)
    inv_E_section = np.linalg.inv(a_mat)[:, 0, 0] * total_thickness

    # E_section(x) = 1 / inv_E_section; series-average the compliance.
    e_eff = 1.0 / float(np.mean(inv_E_section))
    return float(e_eff / E_x0)


def _max_consecutive_zero_plies(angles: list[float], tol: float = 5.0) -> int:
    """Find maximum number of consecutive 0-degree plies in a layup.

    Used by the tension OOP model to determine the effective curved-beam
    thickness h_eff = n_adj × t_ply for interlaminar stress prediction.

    Returns the true maximum consecutive count, which is ``0`` when the
    layup contains no 0-degree plies.  Callers must guard against the
    zero case (e.g. for the curved-beam OOP path, ``n_adj == 0`` means
    there is no continuous 0-degree block to develop interlaminar stress).
    """
    max_count = 0
    count = 0
    for a in angles:
        if abs(a) < tol:
            count += 1
            max_count = max(max_count, count)
        else:
            count = 0
    return max_count


# ======================================================================
# Public helper: amplitude → wavelength estimation
# ======================================================================

def estimate_wavelength_from_amplitude(
    amplitude: float,
    *,
    K_lambda: float = 19.9,
    lambda_min: float = 8.2,
    lambda_ref: float = 0.366,
    scaling: str = "sqrt",
) -> float:
    """Estimate wrinkle wavelength from amplitude when lambda is not measured.

    Some validation datasets report only the wrinkle amplitude ``A`` and
    require an external rule to recover the wavelength ``lambda`` needed
    by the Budiansky-Fleck peak-fibre-angle model
    ``theta_max = arctan(2*pi*A/lambda)``.  Two scaling rules are
    supported:

    * ``"linear"`` (legacy):
        ``lambda = K_lambda * A``.  Reproduces the original convention
        documented in §1.4 of ``VALIDATION_DATA.md`` and used by older
        validation harnesses.  Because ``theta_max`` reduces to
        ``arctan(2*pi/K_lambda)``, the predicted peak fibre angle is
        *constant* in ``A`` under this rule.  That is unphysical for
        severe wrinkles, where larger amplitudes produce steeper local
        fibre rotations and stronger compressive knockdowns.

    * ``"sqrt"`` (default, recommended):
        ``lambda = K_lambda * sqrt(A * lambda_ref)``.  At the reference
        amplitude ``A = lambda_ref`` the sqrt rule matches the legacy
        linear rule exactly (``lambda = K_lambda * lambda_ref``), so
        the calibration of mild wrinkles is preserved.  For
        ``A > lambda_ref`` the wavelength grows sub-linearly with
        amplitude, so ``theta_max = arctan(2*pi*A/lambda)`` increases
        monotonically with ``A`` and the model captures the experimental
        knockdown collapse seen at high D/T in the Elhajjar (2025)
        compression dataset.

    Both rules are clamped from below at ``lambda_min`` so that tiny
    amplitudes do not produce vanishingly small wavelengths.

    Parameters
    ----------
    amplitude : float
        Wrinkle amplitude *A* [mm].  Must be non-negative.
    K_lambda : float, optional
        Slope coefficient. Defaults to ``19.9`` (Elhajjar 2025
        T700/2510 calibration).
    lambda_min : float, optional
        Lower clamp on the returned wavelength [mm].  Defaults to
        ``8.2`` mm (Elhajjar 2025).
    lambda_ref : float, optional
        Reference amplitude [mm] at which the sqrt scaling is anchored
        to the legacy linear rule.  Defaults to ``0.366`` mm (two ply
        thicknesses in the Elhajjar T700/2510 layup).  Ignored when
        ``scaling == "linear"``.
    scaling : {"sqrt", "linear"}, optional
        Scaling rule selector.  Defaults to ``"sqrt"``.

    Returns
    -------
    float
        Wavelength ``lambda`` in mm, lower-bounded by ``lambda_min``.

    Raises
    ------
    ValueError
        If ``scaling`` is not one of ``"sqrt"`` or ``"linear"``.

    Examples
    --------
    Legacy linear rule (constant peak angle):

    >>> estimate_wavelength_from_amplitude(0.5, scaling="linear")
    9.95

    Sub-linear sqrt rule (default):  at the reference amplitude the
    raw rule matches the legacy linear rule exactly
    (``K_lambda * lambda_ref = 19.9 * 0.366 = 7.2834``), which then
    clamps to ``lambda_min = 8.2``:

    >>> estimate_wavelength_from_amplitude(0.366)
    8.2
    """
    if scaling not in ("sqrt", "linear"):
        raise ValueError(
            f"estimate_wavelength_from_amplitude: scaling must be "
            f"'sqrt' or 'linear', got {scaling!r}"
        )
    if scaling == "linear":
        lam = K_lambda * amplitude
    else:  # "sqrt"
        # lambda = K_lambda * sqrt(A * lambda_ref) -- matches linear at
        # A = lambda_ref, grows sub-linearly for A > lambda_ref.  Guard
        # against negative amplitude reaching the sqrt.
        lam = K_lambda * math.sqrt(max(amplitude, 0.0) * lambda_ref)
    return max(lam, lambda_min)


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class WrinkleSpec:
    """Single-wrinkle specification used to assemble multi-wrinkle configs.

    A list of :class:`WrinkleSpec` instances passed to
    :class:`AnalysisConfig` via the ``wrinkles`` field overrides the
    single/dual-wrinkle dispatch in :meth:`WrinkleAnalysis.run`, allowing
    arbitrary N-wrinkle layouts at arbitrary ply interfaces with arbitrary
    phase offsets to be analysed (see Dataset F / Li et al. 2025).

    Parameters
    ----------
    amplitude : float
        Wrinkle half-amplitude *A* [mm], strictly positive.
    wavelength : float
        Wavelength lambda [mm], strictly positive.
    width : float
        Gaussian envelope half-width *w* [mm], strictly positive.
    ply_interface : int
        Ply interface index passed to :class:`WrinklePlacement`. For a
        laminate with *N* plies valid indices are 0 through N-2 inclusive.
    phase_offset : float, optional
        Phase offset phi [rad] relative to the reference wrinkle.
        Default 0.0.
    """

    amplitude: float
    wavelength: float
    width: float
    ply_interface: int
    phase_offset: float = 0.0


#: Schema version stamped into :meth:`AnalysisConfig.to_dict` output and
#: verified by :meth:`AnalysisConfig.from_dict`.  Bump this whenever the
#: serialised config layout changes in a non-round-trippable way so old
#: files fail loudly rather than load into a mismatched schema.
CONFIG_VERSION = 1


def _material_to_jsonable(mat: OrthotropicMaterial | None) -> dict | None:
    """Serialise a material as a library-preset reference or inline custom.

    A material equal (field-for-field) to the like-named library preset is
    written as ``{"preset": name}`` so the file stays compact and tracks
    library updates; any other material — including one that reuses a
    library name but tweaks a property — is written as
    ``{"custom": material.to_dict()}`` so no information is lost.
    """
    if mat is None:
        return None
    try:
        preset = MaterialLibrary().get(mat.name)
    except KeyError:
        preset = None
    if preset is not None and preset.to_dict() == mat.to_dict():
        return {"preset": mat.name}
    return {"custom": mat.to_dict()}


def _material_from_jsonable(
    value: dict | None, *, field: str
) -> OrthotropicMaterial | None:
    """Inverse of :func:`_material_to_jsonable`."""
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or len(value) != 1
        or next(iter(value)) not in ("preset", "custom")
    ):
        raise ValueError(
            f"AnalysisConfig.from_dict: {field} must be null, "
            f"{{'preset': name}}, or {{'custom': {{...}}}}, got {value!r}"
        )
    if "preset" in value:
        try:
            return MaterialLibrary().get(value["preset"])
        except KeyError as exc:
            raise ValueError(
                f"AnalysisConfig.from_dict: {field} references unknown "
                f"material preset {value['preset']!r}"
            ) from exc
    return OrthotropicMaterial.from_dict(value["custom"])


def _gate_to_jsonable(gate: GateParameters | None) -> dict | None:
    """Serialise a penetration gate as a registered-preset reference.

    Only the calibrated presets in
    :data:`wrinklefe.core.penetration_gate.GATE_PRESETS` are serialisable
    (the parameters are material-realization specific and are not meant to
    be hand-authored inline).  An unregistered custom gate raises so the
    lossy write surfaces loudly instead of silently dropping data.
    """
    if gate is None:
        return None
    preset = GATE_PRESETS.get(gate.name)
    if preset is not None and preset == gate:
        return {"preset": gate.name}
    raise ValueError(
        f"AnalysisConfig.to_dict: penetration_gate {gate.name!r} is not a "
        f"registered preset and inline custom gates are not serialisable; "
        f"use one of {sorted(GATE_PRESETS)} or None"
    )


def _gate_from_jsonable(value: dict | None) -> GateParameters | None:
    """Inverse of :func:`_gate_to_jsonable`."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"preset"}:
        raise ValueError(
            "AnalysisConfig.from_dict: penetration_gate must be null or "
            f"{{'preset': name}}, got {value!r}"
        )
    name = value["preset"]
    if name not in GATE_PRESETS:
        raise ValueError(
            f"AnalysisConfig.from_dict: unknown penetration_gate preset "
            f"{name!r}; available {sorted(GATE_PRESETS)}"
        )
    return GATE_PRESETS[name]


@dataclass
class AnalysisConfig:
    """Configuration for a wrinkle analysis run.

    Collects all user-specified parameters in a single object that
    can be serialised, compared, and passed to :class:`WrinkleAnalysis`.

    Parameters
    ----------
    amplitude : float
        Wrinkle half-amplitude *A* [mm]: the peak displacement of the
        wrinkled mid-surface from the flat (unwrinkled) reference plane,
        so ``z(x) = A * cos(2*pi*(x - x0) / lambda)`` (modulated by the
        envelope) and the peak-to-trough height is ``2A``. For a
        measured wrinkle, ``A = (z_max - z_min) / 2``. Units: mm.
        Default 0.366 (two ply thicknesses).

        Effect on knockdown: A enters the maximum fibre misalignment
        angle through the closed-form ``theta_max = arctan(2*pi*A /
        lambda)`` used in ``_compute_analytical``, so for small A/lambda
        the peak fibre angle scales nearly linearly with A and amplifies
        the Budiansky-Fleck compressive knockdown.
    wavelength : float
        Spatial period of the cosine carrier *lambda* [mm]: the
        crest-to-crest distance of the underlying
        ``cos(2*pi*(x - x0) / lambda)`` carrier along the longitudinal
        x-direction.  The wavenumber is ``k = 2*pi/lambda`` (1/mm).
        Default 16.0.  Must be > 0.
    width : float
        Longitudinal envelope decay length *w* [mm] about the wrinkle
        centre ``x0``.  Exact meaning is profile-dependent: Gaussian
        1/e length scale in ``exp(-(x - x0)**2 / w**2)``, tapered
        flat-top extent (``|x - x0| < w/2``), or triangular half-base
        (``|x - x0| < w``).  Also used as the transverse (y-direction)
        extent of the wrinkle in the 3-D dual-wrinkle / graded mesh
        deformation.  Default 12.0.  Must be > 0.
    morphology : str
        Morphology name. Five values are accepted; the first three are
        *dual-wrinkle* modes distinguished by phase, the last two are
        *single-wrinkle* modes distinguished by their through-thickness
        amplitude profile:

        - ``'stack'`` (default) — two aligned wrinkles, φ = 0. Linear
          through-thickness decay from the interface plies to zero at
          the outer surfaces. M_f = 1.0 (dual-wrinkle baseline).
        - ``'convex'`` — two wrinkles, φ = +π/2 (interface bulges
          outward). Same through-thickness decay. M_f < 1, least
          damaging in compression.
        - ``'concave'`` — two wrinkles, φ = −π/2 (interface pinches
          inward). Same through-thickness decay. M_f > 1, most
          damaging in compression.
        - ``'uniform'`` — *one* wrinkle, full amplitude on every ply
          (no through-thickness decay). M_f = 1.0 because there is no
          pairwise interaction, but the deformed mesh and per-ply
          fibre-angle field differ from ``'stack'``: every ply
          (including the outer surfaces) carries the full profile.
        - ``'graded'`` — one wrinkle, linear decay from mid-ply to the
          surfaces with the ``decay_floor`` knob (0 = full decay to
          zero, 1 = same as ``'uniform'``).
    phase : float or None
        Explicit dual-wrinkle phase offset phi [radians] between the two
        wrinkle centrelines.  When ``None`` (default), the phase is
        derived from ``morphology`` via :data:`MORPHOLOGY_PHASES`
        (stack=0, convex=+pi/2, concave=-pi/2).  When set to a float, it
        overrides the named-morphology phase, allowing arbitrary
        dual-wrinkle phase offsets to be analysed or swept (e.g.
        between 0 and pi).  Ignored for single-wrinkle morphologies
        (``'uniform'``, ``'graded'``).  Must be finite when set.
    decay_floor : float
        Graded morphology only (dimensionless, in ``[0, 1]``): minimum
        fraction of the wrinkle amplitude retained at the laminate outer
        surfaces.  ``0.0`` (default) means full decay to zero amplitude
        at the surfaces (pure graded); ``1.0`` means no decay
        (equivalent to ``uniform``).  Values outside ``[0, 1]`` are
        rejected by ``__post_init__``.
    wrinkle_z_position : float
        Through-thickness position of the (single-wrinkle) decay centre,
        expressed as a fraction of the laminate thickness *T*.  ``0.0``
        places the wrinkle at the bottom surface, ``0.5`` (default) at
        the midplane (legacy behaviour), and ``1.0`` at the top surface.
        Used by the graded morphology path to shift the Gaussian through-
        thickness decay centre off the midplane and to bias the linear
        per-ply tension grading; mirrors the Above / Middle / Below
        wrinkle locations of Li et al. (2025) Dataset F.  Ignored for the
        ``stack``, ``convex``, ``concave`` and ``uniform`` morphologies,
        whose through-thickness behaviour is set by the morphology itself
        or by the ``interface_1`` / ``interface_2`` interface indices.
        Must be a finite float in ``[0.0, 1.0]``.
    amplitude_profile : {"constant", "gaussian", "linear"}
        Spatially varying in-plane amplitude modulation applied on top
        of the wrinkle's own longitudinal envelope (see
        :class:`~wrinklefe.core.morphology.WrinkleConfiguration`).
        ``"constant"`` (default) preserves the legacy behaviour --
        the wrinkle amplitude *A* is used uniformly across the in-plane
        domain.  ``"gaussian"`` multiplies *A* by ``exp(-(s/d)**2)`` and
        ``"linear"`` by ``max(0, 1 - |s|/d)`` (clipped), where *s* is
        the in-plane coordinate (relative to the wrinkle centre) along
        ``amplitude_profile_axis`` and *d* is
        ``amplitude_profile_decay_length``.
    amplitude_profile_decay_length : float or None
        Length scale *d* (mm) controlling the Gaussian sigma or the
        linear-decay extent.  ``None`` (default) falls back to the
        wrinkle profile's own ``width``, so the amplitude tapers on the
        same length scale as the envelope.  Must be positive when
        provided.  Ignored when ``amplitude_profile == "constant"``.
    amplitude_profile_axis : {"x", "y"}
        In-plane axis along which the amplitude modulation runs.
        Default ``"x"``.  Pick ``"y"`` (transverse axis) for an
        independent in-plane tapering of *A* that does not stack with
        the existing longitudinal envelope on *x*.
    loading : str
        Loading mode: ``'compression'`` or ``'tension'``.
        Default is ``'compression'``.
    material : OrthotropicMaterial or None
        Composite material.  ``None`` uses the default IM7/8552.
    angles : list[float] or None
        Ply angles in degrees.  ``None`` uses a quasi-isotropic
        ``[0/45/-45/90]_3s`` layup (24 plies).
    interface_1 : int or None
        Ply interface for the first wrinkle.  ``None`` (default) auto-
        derives an interior interface from the resolved layup so small
        laminates (< 13 plies) work out of the box: ``interface_1 =
        max(0, n_plies // 2 - 1)``.  For the default 24-ply layup this
        resolves to ``11`` (backwards-compatible).
    interface_2 : int or None
        Ply interface for the second wrinkle.  ``None`` (default) auto-
        derives ``interface_2 = min(n_plies - 1, n_plies // 2)``.  For
        the default 24-ply layup this resolves to ``12``
        (backwards-compatible).
    nx : int
        Mesh divisions in x.  Default 12.
    ny : int
        Mesh divisions in y.  Default 6.
    nz_per_ply : int
        Mesh divisions per ply in z.  Default 1.
    domain_length : float
        Domain length in x [mm].  Default ``3 * wavelength``.
    domain_width : float
        Domain width in y [mm].  Default 20.0.
    applied_strain : float
        Applied nominal strain for displacement-controlled loading.
        Default ``-0.01`` (1 % compression).
    solver : str
        Linear solver: ``'direct'`` or ``'iterative'``.  Default ``'direct'``.
    verbose : bool
        Print progress information.  Default ``False``.
    through_thickness_decay_scale : float or None
        Optional override (mm) for the through-thickness Gaussian
        standard deviation used by the graded morphology's profile-
        proportional KD path and the tension graded-averaging block.
        ``None`` (default) triggers the auto formula
        ``max(wavelength / 2, amplitude)``.  Must be > 0 when set.
    kink_band_quadratic_coeff : float
        Argon-Fleck quadratic coefficient ``c_AF`` (dimensionless) in
        the extended Budiansky-Fleck closed form
        ``KD = 1 / (1 + r + c_AF * r**2)`` with
        ``r = theta_eff / gamma_Y_eff``.  Default ``0.0`` recovers the
        legacy linear BF response.  Must be >= 0.
    """

    # Wrinkle geometry
    amplitude: float = 0.366
    wavelength: float = 16.0
    width: float = 12.0

    # Morphology
    morphology: str = "stack"
    # Explicit dual-wrinkle phase offset phi [rad]. None → derive from
    # `morphology` (stack=0, convex=+pi/2, concave=-pi/2). A float
    # overrides the named-morphology phase so arbitrary phases can be
    # analysed/swept. Ignored for single-wrinkle modes (uniform/graded).
    phase: float | None = None
    decay_floor: float = 0.0  # graded mode: min amplitude fraction at surfaces (0–1)
    # Single-wrinkle through-thickness position as a fraction of the laminate
    # thickness (0 = bottom surface, 0.5 = midplane, 1 = top surface). Only
    # consulted by the graded morphology path; ignored for stack/convex/
    # concave/uniform.
    wrinkle_z_position: float = 0.5

    # Spatially varying in-plane amplitude profile (#178 follow-up). Defaults
    # mirror WrinkleConfiguration so the legacy "constant" behaviour is
    # preserved when callers leave them unset.
    amplitude_profile: str = "constant"
    amplitude_profile_decay_length: float | None = None
    amplitude_profile_axis: str = "x"

    # Loading
    loading: str = "compression"

    # Material & laminate
    material: OrthotropicMaterial | None = None
    angles: list[float] | None = None

    # Ply thickness
    ply_thickness: float = 0.183  # mm (1 ply thickness for CYCOM X850/T800)

    # Wrinkle placement. ``None`` triggers auto-derivation in
    # ``__post_init__`` from ``len(angles)`` so small laminates work out
    # of the box (issues #154/#156). For the default 24-ply layup the
    # auto-derived pair is (11, 12), preserving backwards compatibility.
    interface_1: int | None = None
    interface_2: int | None = None

    # Multi-wrinkle override. When non-empty, this overrides the named
    # single/dual-wrinkle dispatch in WrinkleAnalysis.run, allowing
    # arbitrary N-wrinkle configurations (Li et al. 2025 Dataset F).
    # Each spec contributes one WrinklePlacement; FE solve is currently
    # out of scope for this path (set analytical_only=True).
    wrinkles: list[WrinkleSpec] | None = None

    # Mesh
    nx: int = 12
    ny: int = 6
    nz_per_ply: int = 1
    domain_length: float = 0.0  # 0 → auto = 3 * wavelength
    domain_width: float = 20.0

    # Loading parameters
    applied_strain: float = -0.01

    # Solver
    solver: str = "direct"

    # Analytical-only mode (skip FE assembly)
    analytical_only: bool = False

    # Verbosity
    verbose: bool = False

    # Through-thickness Gaussian decay scale [mm] used by the graded
    # morphology's profile-proportional KD path.  ``None`` (default)
    # triggers the auto formula ``max(wavelength / 2, amplitude)`` so the
    # decay scale tracks the wrinkle's longitudinal extent rather than
    # its (much smaller) amplitude.  Provide an explicit positive float
    # to pin the decay scale (e.g. to reproduce the pre-PR amplitude-
    # based behaviour for regression testing).  Must be > 0 when set.
    through_thickness_decay_scale: float | None = None

    # Argon-Fleck quadratic coefficient ``c_AF`` (dimensionless) for the
    # extended Budiansky-Fleck closed form ``KD = 1/(1 + r + c_AF*r^2)``
    # where ``r = theta_eff / gamma_Y_eff``.  Default 0.0 recovers the
    # legacy linear BF response; positive values are required to match
    # the high-angle response of thick UD wrinkled coupons (Li 2024 /
    # Li 2025 high-amplitude cases) where the linear form systematically
    # over-predicts strength.  Must be >= 0.
    kink_band_quadratic_coeff: float = 0.0

    # ------------------------------------------------------------------
    # Two-parameter (theta, D/T) penetration-gate knockdown (item D.3).
    # ------------------------------------------------------------------
    # When set, the analytical knockdown is computed from the
    # penetration-gate model instead of the Budiansky-Fleck kink-band
    # path: KD = 1 - (1 - KD_angle(theta)) * min(1, (D/T / dt0)**p), which
    # reproduces both the angle and the through-thickness penetration
    # dependence the Li UD grids expose (E MAE 2.8 %, F MAE 6.0 % vs the
    # angle-only/FE models' ~20-30 %).  Material-realization specific —
    # use ``wrinklefe.core.penetration_gate.GATE_LI2024_MOULDED`` /
    # ``GATE_LI2025_VACBAG`` or calibrate your own.  UD-scoped: do NOT set
    # for multidirectional/blocked laminates.
    penetration_gate: GateParameters | None = None

    # ------------------------------------------------------------------
    # Resin-pocket material zone (Li et al. 2024/2025 UD glass datasets).
    # ------------------------------------------------------------------
    # When ``enable_resin_pocket`` is True, the FE path tags the hex
    # elements inside the cosine resin lens at the wrinkle crest and
    # assigns them an isotropic epoxy material (``resin_pocket_material``,
    # default the built-in ``EPOXY_S6C10`` card) instead of the host ply's
    # fibre-direction material, with the fibre-misalignment angle zeroed.
    # This captures the soft, fibre-free inclusion the machined cosine
    # insert leaves at the crest — a real compressive-knockdown mechanism
    # the homogenised-ply mesh otherwise misses.  No effect on the
    # analytical path or when ``analytical_only=True``.
    enable_resin_pocket: bool = False
    resin_pocket_material: OrthotropicMaterial | None = None
    # Graded transition (default): the pocket modulus blends smoothly from
    # neat resin at the lens centre to the host fibre material at the
    # boundary, and the fibre-misalignment angle is scaled by (1 - weight),
    # so the wrinkle defect is counted once.  A binary fibre/resin jump
    # (``False``) over-weakens via a spurious stress concentration that
    # double-counts the misaligned-fibre crest knockdown.
    resin_pocket_graded: bool = True
    # Crest half-height of the lens as a multiple of the wrinkle
    # half-amplitude A (mm at center, tapering to 0 at the longitudinal
    # edges).  Must be > 0 when the pocket is enabled.
    resin_pocket_height_scale: float = 1.0
    # Longitudinal half-extent of the lens as a multiple of wavelength/2
    # (the cosine insert support).  Must be > 0 when enabled.
    resin_pocket_length_scale: float = 1.0

    # ------------------------------------------------------------------
    # Progressive-damage FE path (load-stepping ply-discount to ultimate
    # load).  When ``enable_progressive_damage`` is True the FE solve runs
    # the :class:`~wrinklefe.solver.progressive_damage.ProgressiveDamageSolver`
    # on the wrinkled mesh and a pristine baseline, populating
    # ``progressive_strength_MPa`` and ``progressive_knockdown`` on the
    # result.  This is the only path that carries UD compression past
    # first-ply failure (the linear LaRC05 index never activates for
    # pristine UD).  No effect on ``analytical_only`` runs; not combinable
    # with ``enable_czm``.
    enable_progressive_damage: bool = False
    progressive_n_increments: int = 15
    progressive_residual_factor: float = 0.1
    # Target nominal strain magnitude for the load-stepping ramp.  ``None``
    # auto-sizes it to ~1.8x the fibre failure strain (Xc / E1) so the
    # ramp brackets the peak load.  Must be > 0 when set.
    progressive_max_strain: float | None = None

    # ------------------------------------------------------------------
    # Cohesive zone modelling (delamination prediction).
    # ------------------------------------------------------------------
    # v1: bilinear intrinsic CZM with Benzeggagh-Kenane mode-mixity.
    # Off by default; when ``enable_czm=True`` the FE solve switches
    # from :class:`StaticSolver` (linear) to
    # :class:`NewtonRaphsonSolver` and inserts zero-thickness cohesive
    # elements at the requested ply interfaces.  None of the other
    # czm_* fields have any effect when ``enable_czm=False``.
    enable_czm: bool = False
    # Which ply interfaces to insert cohesive elements at:
    #   * ``"all"`` — every interior ply interface,
    #   * ``"near_crest"`` — for a scalar (named-morphology) config, the
    #     single interface whose z-coordinate is closest to the wrinkle
    #     peak; for a multi-wrinkle config (``wrinkles`` set), the
    #     interface nearest *each* wrinkle, deduplicated — wrinkles
    #     sharing an interface index get one continuous cohesive
    #     surface, enabling crest-to-crest delamination link-up
    #     (issue #283),
    #   * ``list[int]`` — explicit list of interface indices in
    #     ``[0, n_plies-1)`` (0 = bottom-most interior interface).
    czm_interfaces: list[int] | str = "near_crest"
    czm_law: str = "bilinear"
    czm_GIc: float | None = None      # N/mm; None -> material default
    czm_GIIc: float | None = None
    czm_sigma_max: float | None = None  # MPa; None -> material default
    czm_tau_max: float | None = None
    czm_penalty: float = 1.0e6           # N/mm^3 initial interface stiffness
    czm_BK_eta: float = 1.45
    # Default bumped 20 -> 100 after the Phase 7 NASA TM DCB validation
    # showed that the original 20-increment default was too coarse for
    # accurate post-peak crack-propagation tracking; coarse increments
    # also amplified the cohesive law's d=1 corner artefact (post-peak
    # see-saw oscillations) by ~33 %.  100 fixed equal increments lands
    # the predicted peak load within experimental scatter and integrated
    # energy within 7 % for the IM7/8552 DCB benchmark.  Users who need
    # faster turnaround can lower this; users running publication-grade
    # validations should bump to 200.
    czm_n_load_increments: int = 100
    czm_newton_tol: float = 1.0e-4

    def __post_init__(self) -> None:
        if self.domain_length <= 0:
            if self.wrinkles:
                # Multi-wrinkle: size the domain from the union of the
                # wrinkle extents (per-spec center offset from its phase
                # plus the 3*width Gaussian support), not from the
                # scalar wavelength field. The 3*wavelength floor keeps
                # a single centred spec consistent with the scalar path.
                half_span = max(
                    abs(s.phase_offset) * s.wavelength / (2.0 * math.pi)
                    + 3.0 * s.width
                    for s in self.wrinkles
                )
                self.domain_length = max(
                    2.0 * half_span, 3.0 * self.wavelength
                )
            else:
                self.domain_length = 3.0 * self.wavelength
        if self.material is None:
            self.material = MaterialLibrary().get("IM7_8552")
        if self.angles is None:
            # Quasi-isotropic [0/45/-45/90]_3s → 24 plies
            base: list[float] = [0, 45, -45, 90]
            self.angles = (base * 3) + list(reversed(base * 3))

        # Auto-derive interior interface indices when the user did not
        # specify them. The two interfaces sit symmetrically about the
        # mid-thickness, so for the default 24-ply layup this resolves
        # to (11, 12) — i.e. backwards-compatible with the previous
        # hard-coded dataclass defaults. For small layups (< 13 plies)
        # it picks valid in-range indices instead of crashing in
        # ``_validate`` (issues #154 / #156).
        n_plies = len(self.angles) if self.angles is not None else 0
        if n_plies > 0:
            mid = n_plies // 2
            if self.interface_1 is None:
                self.interface_1 = max(0, mid - 1)
            if self.interface_2 is None:
                self.interface_2 = min(n_plies - 1, mid)

        self._validate()

    def _validate(self) -> None:
        """Fail fast on physically invalid configuration.

        Called from :meth:`__post_init__` after defaults (``domain_length``,
        ``material``, ``angles``) have been resolved.  Each check raises a
        :class:`ValueError` naming the offending field and value so that
        misconfiguration surfaces at construction time rather than as an
        obscure traceback deep in the solver/mesh path.
        """
        # ``angles`` is filled in __post_init__ before _validate runs.
        assert self.angles is not None
        # --- Ply angles ------------------------------------------------
        # Enforce the canonical fibre-angle range (|angle| <= 90) using
        # the shared parser rule so a mis-typed layup (e.g. 900°) fails
        # loudly at construction instead of flowing into CLT trig and
        # being silently mis-classified by the tension-mechanism heuristic.
        for i, angle in enumerate(self.angles):
            validate_ply_angle(
                float(angle), context=f"AnalysisConfig.angles[{i}] = "
            )
        # --- Wrinkle geometry -----------------------------------------
        # amplitude == 0 is a legitimate "no wrinkle" (flat) case: the
        # mid-surface profile z(x) = A * envelope reduces to 0, so the
        # closed-form misalignment angle arctan(2*pi*A/lambda) is 0.
        # Only a negative amplitude is physically meaningless.
        if self.amplitude < 0:
            raise ValueError(
                f"AnalysisConfig.amplitude must be >= 0 "
                f"(0 = flat / no wrinkle), got {self.amplitude}"
            )
        if not (self.wavelength > 0):
            # Strictly positive: lambda divides into the closed-form
            # slope (2*pi*A/lambda) and the auto-derived domain_length.
            raise ValueError(
                f"AnalysisConfig.wavelength must be > 0, "
                f"got {self.wavelength}"
            )
        if not (self.width > 0):
            raise ValueError(
                f"AnalysisConfig.width must be > 0, got {self.width}"
            )
        if not (self.domain_length > 0):
            raise ValueError(
                f"AnalysisConfig.domain_length must be > 0, "
                f"got {self.domain_length}"
            )
        if not (self.domain_width > 0):
            raise ValueError(
                f"AnalysisConfig.domain_width must be > 0, "
                f"got {self.domain_width}"
            )
        if not (self.ply_thickness > 0):
            raise ValueError(
                f"AnalysisConfig.ply_thickness must be > 0, "
                f"got {self.ply_thickness}"
            )

        # --- Morphology / phase ---------------------------------------
        # run() resolves morphology by name (possibly lower/strip'd) via
        # WrinkleConfiguration.from_morphology_name; an explicit numeric
        # ``phase`` overrides the named-morphology phase for dual-wrinkle
        # modes but the name is still consumed (single-wrinkle modes and
        # the from_morphology_name fallback both require a known name).
        valid_morphologies = sorted(
            set(MORPHOLOGY_PHASES) | set(SINGLE_WRINKLE_MODES)
        )
        if (
            not isinstance(self.morphology, str)
            or self.morphology.lower().strip() not in valid_morphologies
        ):
            raise ValueError(
                f"AnalysisConfig.morphology must be one of "
                f"{valid_morphologies}, got {self.morphology!r}"
            )
        if self.phase is not None and not math.isfinite(float(self.phase)):
            raise ValueError(
                f"AnalysisConfig.phase must be finite when set, "
                f"got {self.phase}"
            )
        if not (0.0 <= self.decay_floor <= 1.0):
            raise ValueError(
                f"AnalysisConfig.decay_floor must be in [0, 1], "
                f"got {self.decay_floor}"
            )

        # Single-wrinkle through-thickness position as a fraction of T.
        # Must be a finite float in [0, 1]; NaN and infinities are rejected
        # so they cannot silently shift the decay centre off the laminate.
        try:
            wz_value = float(self.wrinkle_z_position)
        except (TypeError, ValueError):
            raise ValueError(
                f"AnalysisConfig.wrinkle_z_position must be a finite float "
                f"in [0, 1], got {self.wrinkle_z_position!r}"
            )
        if not math.isfinite(wz_value) or not (0.0 <= wz_value <= 1.0):
            raise ValueError(
                f"AnalysisConfig.wrinkle_z_position must be a finite float "
                f"in [0, 1], got {self.wrinkle_z_position!r}"
            )

        # --- Amplitude profile (spatially varying A modulation) -------
        valid_amplitude_profiles = ("constant", "gaussian", "linear")
        if (
            not isinstance(self.amplitude_profile, str)
            or self.amplitude_profile.lower().strip() not in valid_amplitude_profiles
        ):
            raise ValueError(
                f"AnalysisConfig.amplitude_profile must be one of "
                f"{list(valid_amplitude_profiles)}, got {self.amplitude_profile!r}"
            )
        valid_amplitude_profile_axes = ("x", "y")
        if (
            not isinstance(self.amplitude_profile_axis, str)
            or self.amplitude_profile_axis.lower().strip()
            not in valid_amplitude_profile_axes
        ):
            raise ValueError(
                f"AnalysisConfig.amplitude_profile_axis must be one of "
                f"{list(valid_amplitude_profile_axes)}, "
                f"got {self.amplitude_profile_axis!r}"
            )
        if self.amplitude_profile_decay_length is not None and not (
            self.amplitude_profile_decay_length > 0.0
            and math.isfinite(self.amplitude_profile_decay_length)
        ):
            raise ValueError(
                f"AnalysisConfig.amplitude_profile_decay_length must be "
                f"a finite positive float when set, "
                f"got {self.amplitude_profile_decay_length}"
            )

        # --- Loading --------------------------------------------------
        valid_loadings = ("compression", "tension")
        if (
            not isinstance(self.loading, str)
            or self.loading.lower().strip() not in valid_loadings
        ):
            raise ValueError(
                f"AnalysisConfig.loading must be one of "
                f"{list(valid_loadings)}, got {self.loading!r}"
            )
        if not math.isfinite(self.applied_strain):
            raise ValueError(
                f"AnalysisConfig.applied_strain must be finite, "
                f"got {self.applied_strain}"
            )

        # --- Wrinkle placement (interface indices) --------------------
        n_plies = len(self.angles)
        for name in ("interface_1", "interface_2"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"AnalysisConfig.{name} must be an int, got {value!r}"
                )
            if not (0 <= value < n_plies):
                raise ValueError(
                    f"AnalysisConfig.{name} must be in [0, {n_plies}) "
                    f"(0 <= interface < number of plies), got {value}"
                )

        # --- Mesh resolution (structural integers) --------------------
        for name in ("nx", "ny", "nz_per_ply"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"AnalysisConfig.{name} must be an int, got {value!r}"
                )
            if value < 1:
                raise ValueError(
                    f"AnalysisConfig.{name} must be >= 1, got {value}"
                )

        # --- Solver ---------------------------------------------------
        valid_solvers = ("direct", "iterative")
        if (
            not isinstance(self.solver, str)
            or self.solver.lower().strip() not in valid_solvers
        ):
            raise ValueError(
                f"AnalysisConfig.solver must be one of "
                f"{list(valid_solvers)}, got {self.solver!r}"
            )

        # --- Through-thickness decay scale ----------------------------
        # Optional positive float (mm) overriding the auto formula
        # ``max(wavelength / 2, amplitude)`` used by the graded
        # morphology profile-proportional path.  Must be strictly
        # positive and finite when provided.
        if self.through_thickness_decay_scale is not None and not (
            self.through_thickness_decay_scale > 0.0
            and math.isfinite(self.through_thickness_decay_scale)
        ):
            raise ValueError(
                f"AnalysisConfig.through_thickness_decay_scale must be a "
                f"finite positive float when set, "
                f"got {self.through_thickness_decay_scale}"
            )

        # --- Argon-Fleck quadratic coefficient ------------------------
        # Dimensionless non-negative float (0.0 = legacy linear BF).
        if (
            not isinstance(self.kink_band_quadratic_coeff, (int, float))
            or isinstance(self.kink_band_quadratic_coeff, bool)
            or not math.isfinite(self.kink_band_quadratic_coeff)
            or self.kink_band_quadratic_coeff < 0.0
        ):
            raise ValueError(
                f"AnalysisConfig.kink_band_quadratic_coeff must be a "
                f"finite float >= 0, "
                f"got {self.kink_band_quadratic_coeff!r}"
            )

        # --- Penetration gate -----------------------------------------
        if self.penetration_gate is not None and not isinstance(
            self.penetration_gate, GateParameters
        ):
            raise ValueError(
                "AnalysisConfig.penetration_gate must be a GateParameters "
                f"or None, got {type(self.penetration_gate).__name__}"
            )

        # --- Resin-pocket zone ----------------------------------------
        # Fields are no-ops when ``enable_resin_pocket`` is False; still
        # type-checked so misconfigurations surface at construction time.
        if not isinstance(self.enable_resin_pocket, bool):
            raise ValueError(
                f"AnalysisConfig.enable_resin_pocket must be a bool, "
                f"got {self.enable_resin_pocket!r}"
            )
        for name in ("resin_pocket_height_scale", "resin_pocket_length_scale"):
            val = getattr(self, name)
            if not (
                isinstance(val, (int, float))
                and not isinstance(val, bool)
                and math.isfinite(val)
                and val > 0.0
            ):
                raise ValueError(
                    f"AnalysisConfig.{name} must be a finite positive "
                    f"float, got {val!r}"
                )

        # --- Progressive-damage path ----------------------------------
        if not isinstance(self.enable_progressive_damage, bool):
            raise ValueError(
                f"AnalysisConfig.enable_progressive_damage must be a bool, "
                f"got {self.enable_progressive_damage!r}"
            )
        if self.enable_progressive_damage and self.enable_czm:
            raise ValueError(
                "AnalysisConfig: enable_progressive_damage and enable_czm "
                "cannot both be True (separate nonlinear FE paths)."
            )
        if (
            not isinstance(self.progressive_n_increments, int)
            or isinstance(self.progressive_n_increments, bool)
            or self.progressive_n_increments < 1
        ):
            raise ValueError(
                f"AnalysisConfig.progressive_n_increments must be an int "
                f">= 1, got {self.progressive_n_increments!r}"
            )
        if not (
            isinstance(self.progressive_residual_factor, (int, float))
            and not isinstance(self.progressive_residual_factor, bool)
            and 0.0 < self.progressive_residual_factor < 1.0
        ):
            raise ValueError(
                f"AnalysisConfig.progressive_residual_factor must be in "
                f"(0, 1), got {self.progressive_residual_factor!r}"
            )
        if self.progressive_max_strain is not None and not (
            isinstance(self.progressive_max_strain, (int, float))
            and not isinstance(self.progressive_max_strain, bool)
            and math.isfinite(self.progressive_max_strain)
            and self.progressive_max_strain > 0.0
        ):
            raise ValueError(
                f"AnalysisConfig.progressive_max_strain must be a finite "
                f"positive float when set, got {self.progressive_max_strain!r}"
            )

        # --- Cohesive zone modelling ----------------------------------
        # All CZM fields are no-ops when ``enable_czm`` is False; we
        # still type-check them so misconfigurations surface at
        # construction time rather than mid-solve.
        if not isinstance(self.enable_czm, bool):
            raise ValueError(
                f"AnalysisConfig.enable_czm must be a bool, "
                f"got {self.enable_czm!r}"
            )
        if self.czm_law != "bilinear":
            raise ValueError(
                f"AnalysisConfig.czm_law: only 'bilinear' is supported "
                f"in v1, got {self.czm_law!r}"
            )
        if isinstance(self.czm_interfaces, str):
            if self.czm_interfaces not in ("all", "near_crest"):
                raise ValueError(
                    "AnalysisConfig.czm_interfaces must be 'all', "
                    "'near_crest', or a list of int, got "
                    f"{self.czm_interfaces!r}"
                )
        elif isinstance(self.czm_interfaces, list):
            n_plies = len(self.angles)
            for i, idx in enumerate(self.czm_interfaces):
                if not isinstance(idx, int) or isinstance(idx, bool):
                    raise ValueError(
                        f"AnalysisConfig.czm_interfaces[{i}] must be an "
                        f"int, got {idx!r}"
                    )
                # Valid interior ply interfaces are 0 .. n_plies-2.
                if not (0 <= idx <= n_plies - 2):
                    raise ValueError(
                        f"AnalysisConfig.czm_interfaces[{i}] must be in "
                        f"[0, {n_plies - 2}], got {idx}"
                    )
        else:
            raise ValueError(
                "AnalysisConfig.czm_interfaces must be a string ('all' "
                "or 'near_crest') or a list[int], got "
                f"{type(self.czm_interfaces).__name__}"
            )
        for name in ("czm_GIc", "czm_GIIc", "czm_sigma_max", "czm_tau_max"):
            val = getattr(self, name)
            if val is None:
                continue
            if not (
                isinstance(val, (int, float))
                and not isinstance(val, bool)
                and math.isfinite(val)
                and val > 0.0
            ):
                raise ValueError(
                    f"AnalysisConfig.{name} must be a finite positive "
                    f"float when set, got {val!r}"
                )
        if not (
            isinstance(self.czm_penalty, (int, float))
            and not isinstance(self.czm_penalty, bool)
            and math.isfinite(self.czm_penalty)
            and self.czm_penalty > 0.0
        ):
            raise ValueError(
                f"AnalysisConfig.czm_penalty must be a finite positive "
                f"float, got {self.czm_penalty!r}"
            )
        if not (
            isinstance(self.czm_BK_eta, (int, float))
            and not isinstance(self.czm_BK_eta, bool)
            and math.isfinite(self.czm_BK_eta)
            and self.czm_BK_eta > 0.0
        ):
            raise ValueError(
                f"AnalysisConfig.czm_BK_eta must be a finite positive "
                f"float, got {self.czm_BK_eta!r}"
            )
        if (
            not isinstance(self.czm_n_load_increments, int)
            or isinstance(self.czm_n_load_increments, bool)
            or self.czm_n_load_increments < 1
        ):
            raise ValueError(
                f"AnalysisConfig.czm_n_load_increments must be an int "
                f">= 1, got {self.czm_n_load_increments!r}"
            )
        if not (
            isinstance(self.czm_newton_tol, (int, float))
            and not isinstance(self.czm_newton_tol, bool)
            and math.isfinite(self.czm_newton_tol)
            and self.czm_newton_tol > 0.0
        ):
            raise ValueError(
                f"AnalysisConfig.czm_newton_tol must be a finite positive "
                f"float, got {self.czm_newton_tol!r}"
            )

        # --- Multi-wrinkle override -----------------------------------
        # When ``wrinkles`` is provided, it overrides the named
        # single/dual-wrinkle dispatch.  An empty list is rejected
        # because the intent is ambiguous (use None for the default).
        if self.wrinkles is not None:
            if not isinstance(self.wrinkles, list):
                raise ValueError(
                    "AnalysisConfig.wrinkles must be a list of WrinkleSpec "
                    f"or None, got {type(self.wrinkles).__name__}"
                )
            if len(self.wrinkles) == 0:
                raise ValueError(
                    "AnalysisConfig.wrinkles must contain at least one "
                    "WrinkleSpec; pass None to use the default dispatch."
                )
            for i, spec in enumerate(self.wrinkles):
                if not isinstance(spec, WrinkleSpec):
                    raise ValueError(
                        f"AnalysisConfig.wrinkles[{i}] must be a "
                        f"WrinkleSpec, got {type(spec).__name__}"
                    )
                if not (spec.amplitude > 0 and math.isfinite(spec.amplitude)):
                    raise ValueError(
                        f"AnalysisConfig.wrinkles[{i}].amplitude must be "
                        f"a positive finite float, got {spec.amplitude}"
                    )
                if not (spec.wavelength > 0 and math.isfinite(spec.wavelength)):
                    raise ValueError(
                        f"AnalysisConfig.wrinkles[{i}].wavelength must be "
                        f"a positive finite float, got {spec.wavelength}"
                    )
                if not (spec.width > 0 and math.isfinite(spec.width)):
                    raise ValueError(
                        f"AnalysisConfig.wrinkles[{i}].width must be "
                        f"a positive finite float, got {spec.width}"
                    )
                if (
                    not isinstance(spec.ply_interface, int)
                    or isinstance(spec.ply_interface, bool)
                ):
                    raise ValueError(
                        f"AnalysisConfig.wrinkles[{i}].ply_interface must "
                        f"be an int, got {spec.ply_interface!r}"
                    )
                # WrinklePlacement valid range is [0, n_plies - 2]
                if not (0 <= spec.ply_interface <= n_plies - 2):
                    raise ValueError(
                        f"AnalysisConfig.wrinkles[{i}].ply_interface must "
                        f"be in [0, {n_plies - 2}] (n_plies={n_plies}), "
                        f"got {spec.ply_interface}"
                    )
                if not math.isfinite(spec.phase_offset):
                    raise ValueError(
                        f"AnalysisConfig.wrinkles[{i}].phase_offset must "
                        f"be finite, got {spec.phase_offset}"
                    )

    # ------------------------------------------------------------------
    # Serialisation (round-trippable save / load) — issue #259
    # ------------------------------------------------------------------
    #: Object-valued fields serialised through dedicated encoders rather
    #: than emitted as-is.  Every remaining field is a JSON-native scalar
    #: or list, so this list plus the plain fields covers the dataclass in
    #: full — the completeness guard in the test-suite pins that invariant.
    _NON_PRIMITIVE_FIELDS = frozenset(
        {"material", "resin_pocket_material", "angles", "wrinkles",
         "penetration_gate"}
    )

    def to_dict(self) -> dict:
        """Serialise this config to a plain, JSON-round-trippable dict.

        The dict carries a ``config_version`` key and one entry per
        dataclass field.  Non-primitive fields are handled explicitly:
        materials become ``{"preset": name}`` or ``{"custom": {...}}``,
        ``angles`` the resolved ply list, ``wrinkles`` a list of plain
        dicts, and ``penetration_gate`` a ``{"preset": name}`` reference.
        Because the values are the *resolved* ones (post ``__post_init__``),
        the file is self-contained and ``load`` -> ``to_dict`` is
        idempotent.

        See Also
        --------
        from_dict : Inverse constructor.
        save_json, load_json : File helpers.
        """
        out: dict = {"config_version": CONFIG_VERSION}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in ("material", "resin_pocket_material"):
                out[f.name] = _material_to_jsonable(value)
            elif f.name == "wrinkles":
                out[f.name] = (
                    None if value is None else [asdict(s) for s in value]
                )
            elif f.name == "penetration_gate":
                out[f.name] = _gate_to_jsonable(value)
            elif f.name == "angles":
                out[f.name] = (
                    None if value is None else [float(a) for a in value]
                )
            elif isinstance(value, np.ndarray):
                out[f.name] = value.tolist()
            elif isinstance(value, (list, tuple)):
                out[f.name] = [
                    float(v) if isinstance(v, (np.integer, np.floating)) else v
                    for v in value
                ]
            elif isinstance(value, np.integer):
                out[f.name] = int(value)
            elif isinstance(value, np.floating):
                out[f.name] = float(value)
            else:
                out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, data: dict) -> AnalysisConfig:
        """Reconstruct an :class:`AnalysisConfig` from :meth:`to_dict` output.

        The ``config_version`` must match :data:`CONFIG_VERSION` and every
        remaining key must name a dataclass field — an unknown key or a
        version mismatch raises :class:`ValueError` naming the offender
        rather than being silently ignored.  Construction runs the normal
        ``__post_init__`` / ``_validate`` path, so a malformed file fails
        with the same messages a bad in-code config would.

        Raises
        ------
        ValueError
            On version mismatch, unknown keys, or invalid field values.
        """
        if not isinstance(data, dict):
            raise ValueError(
                f"AnalysisConfig.from_dict expects a dict, got "
                f"{type(data).__name__}"
            )
        payload = dict(data)
        version = payload.pop("config_version", None)
        if version != CONFIG_VERSION:
            raise ValueError(
                f"AnalysisConfig.from_dict: unsupported config_version "
                f"{version!r} (expected {CONFIG_VERSION})"
            )
        valid = {f.name for f in fields(cls)}
        unknown = set(payload) - valid
        if unknown:
            raise ValueError(
                f"AnalysisConfig.from_dict: unknown key(s) "
                f"{sorted(unknown)}; valid fields are {sorted(valid)}"
            )
        kwargs = dict(payload)
        for mkey in ("material", "resin_pocket_material"):
            if mkey in kwargs:
                kwargs[mkey] = _material_from_jsonable(kwargs[mkey], field=mkey)
        if kwargs.get("wrinkles") is not None:
            specs = kwargs["wrinkles"]
            if not isinstance(specs, list):
                raise ValueError(
                    "AnalysisConfig.from_dict: wrinkles must be null or a "
                    f"list, got {type(specs).__name__}"
                )
            rebuilt: list[WrinkleSpec] = []
            for i, entry in enumerate(specs):
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"AnalysisConfig.from_dict: wrinkles[{i}] must be a "
                        f"dict, got {type(entry).__name__}"
                    )
                spec_fields = {f.name for f in fields(WrinkleSpec)}
                extra = set(entry) - spec_fields
                if extra:
                    raise ValueError(
                        f"AnalysisConfig.from_dict: wrinkles[{i}] has "
                        f"unknown key(s) {sorted(extra)}"
                    )
                rebuilt.append(WrinkleSpec(**entry))
            kwargs["wrinkles"] = rebuilt
        if "penetration_gate" in kwargs:
            kwargs["penetration_gate"] = _gate_from_jsonable(
                kwargs["penetration_gate"]
            )
        return cls(**kwargs)

    def to_json(self) -> str:
        """Return the config as a deterministic JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def save_json(self, path: str | Path) -> None:
        """Write the config to ``path`` as JSON (parent dirs created)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> AnalysisConfig:
        """Load a config from a JSON file written by :meth:`save_json`."""
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))

    def save_yaml(self, path: str | Path) -> None:
        """Write the config as YAML.

        Optional: requires PyYAML (not a hard dependency).  Raises
        :class:`ImportError` with an actionable message when it is absent.
        """
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "PyYAML is required for YAML config I/O; install it "
                "(`pip install pyyaml`) or use the JSON helpers."
            ) from exc
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load_yaml(cls, path: str | Path) -> AnalysisConfig:
        """Load a config from a YAML file (requires PyYAML)."""
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "PyYAML is required for YAML config I/O; install it "
                "(`pip install pyyaml`) or use the JSON helpers."
            ) from exc
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(yaml.safe_load(text))

    def save(self, path: str | Path) -> None:
        """Save the config, dispatching to YAML for ``.yaml`` / ``.yml``."""
        if Path(path).suffix.lower() in (".yaml", ".yml"):
            self.save_yaml(path)
        else:
            self.save_json(path)

    @classmethod
    def load(cls, path: str | Path) -> AnalysisConfig:
        """Load a config, dispatching to YAML for ``.yaml`` / ``.yml``."""
        if Path(path).suffix.lower() in (".yaml", ".yml"):
            return cls.load_yaml(path)
        return cls.load_json(path)


# ======================================================================
# Results
# ======================================================================

@dataclass
class AnalysisResults:
    """Aggregated results from a complete wrinkle analysis.

    Attributes
    ----------
    config : AnalysisConfig
        The configuration used for this analysis.
    mesh : MeshData
        The generated finite element mesh.
    wrinkle_config : WrinkleConfiguration
        The wrinkle configuration object.
    laminate : Laminate
        The laminate definition.
    morphology_factor : float
        Aggregate morphology factor M_f.
    max_angle_rad : float
        Maximum fibre misalignment angle (radians).
    effective_angle_rad : float
        Effective fibre angle theta_eff (radians).
    damage_index : float
        Interlaminar damage index D.
    analytical_knockdown : float
        Analytical combined knockdown factor (the *ultimate* fibre-failure
        KD for tension; the kink-band KD for compression).
    analytical_onset_knockdown : float or None
        Delamination-onset (first-load-drop) knockdown for tension
        loading, computed from a curved-beam mode-mixity criterion using
        ``material.GIc`` and ``material.GIIc``.  ``None`` when the
        material does not provide both fracture toughnesses or when the
        loading is compression.  Always strictly below
        ``analytical_knockdown``.
    analytical_strength_MPa : float
        Analytical predicted failure stress (MPa).
    field_results : FieldResults or None
        FE solution fields (displacement, stress, strain).
    failure_report : LaminateFailureReport or None
        Multi-criteria failure evaluation.
    failure_indices : dict or None
        Per-criterion FE failure index fields.
    """

    config: AnalysisConfig
    mesh: MeshData | None = None
    wrinkle_config: WrinkleConfiguration | None = None
    laminate: Laminate | None = None

    # Analytical predictions
    morphology_factor: float = 1.0
    max_angle_rad: float = 0.0
    effective_angle_rad: float = 0.0
    mesh_max_angle_rad: float = 0.0  # max fiber angle from FE mesh (accounts for decay)
    damage_index: float = 0.0
    analytical_knockdown: float = 1.0
    analytical_modulus_knockdown: float = 1.0
    """Analytical axial Young's-modulus (stiffness) knockdown
    ``E_x / E_x0`` — a closed-form CLT membrane series-average of the
    off-axis lamina modulus over the wrinkle profile. Populated for
    arbitrary layups and multi-wrinkle layouts: the unidirectional
    single-wrinkle case uses the scalar :func:`_profile_modulus_knockdown`,
    everything else the laminate generalization
    :func:`_laminate_modulus_knockdown` (which reduces to the UD result for
    ``[0]_n``). Stays ``1.0`` only for a degenerate (zero-amplitude)
    wrinkle. Loading-independent; the closed-form companion to the FE
    :attr:`modulus_retention`."""
    analytical_onset_knockdown: float | None = None
    analytical_strength_MPa: float = 0.0
    gamma_Y_eff: float = 0.02  # layup-dependent effective yield strain
    # Tension mechanism decomposition (only for tension loading)
    tension_mechanisms: dict | None = None  # {kd_fiber, kd_matrix, kd_oop, mode, ...}

    # FE results
    field_results: FieldResults | None = None
    failure_report: LaminateFailureReport | None = None
    failure_indices: dict | None = None
    failure_modes: dict | None = None  # {criterion: (n_elem, n_gauss) str array}

    # Retention factor (wrinkled / pristine)
    retention_factors: dict | None = None  # {criterion_name: float}
    baseline_fi: dict | None = None  # {criterion_name: float} pristine max FI

    # Modulus retention (E_wrinkled / E_pristine from FE)
    modulus_retention: float = 1.0
    """FE axial-modulus retention from the **local** fibre-direction-stress
    proxy: ``E_eff = <σ₁₁> / ε_applied`` (mean element-frame σ₁₁ over the
    coupon), wrinkled vs pristine.  Because it averages the *local* fibre
    stress rather than the coupon's global axial response, it over-predicts
    the modulus retention (flatter on the amplitude / penetration / position
    axes) than the measured ``E_x / E_x0``.  Kept for backward
    compatibility; prefer :attr:`modulus_retention_global` for the
    coupon-level stiffness knockdown."""

    modulus_retention_global: float = 1.0
    """FE axial-modulus retention from the **global** reaction response:
    ``E_eff = σ_nominal / ε_applied`` with
    ``σ_nominal = R / A`` — the total axial reaction on the loaded
    (``x_max``) face over the cross-section area ``Ly·Lz`` — computed
    wrinkled vs pristine to give a true coupon-level ``E_x / E_x0``.

    This reuses the same reaction-based nominal-stress pattern as the
    progressive-damage solver (``reaction = sum((K @ u)[xmax_dofs])`` over
    the loaded-face x-DOFs).  Unlike the local σ₁₁ proxy in
    :attr:`modulus_retention`, it captures load redistribution around the
    wrinkle, so it matches the measured modulus knockdown more closely (and
    is correspondingly lower for a wrinkled coupon)."""

    # ------------------------------------------------------------------
    # Progressive-damage results — populated when
    # AnalysisConfig.enable_progressive_damage = True.
    # ------------------------------------------------------------------
    progressive_strength_MPa: float = 0.0
    """Predicted ultimate compressive strength of the wrinkled coupon
    (peak carried nominal stress over the load history, MPa)."""

    progressive_pristine_strength_MPa: float = 0.0
    """Predicted ultimate strength of the pristine (flat) baseline, MPa."""

    progressive_knockdown: float = 1.0
    """Progressive-damage strength knockdown
    ``progressive_strength_MPa / progressive_pristine_strength_MPa``."""

    progressive_history: list | None = None
    """``(applied_strain, nominal_stress)`` samples for the wrinkled run."""

    # ------------------------------------------------------------------
    # CZM results — only populated when AnalysisConfig.enable_czm = True.
    # ------------------------------------------------------------------
    czm_damage: np.ndarray | None = None
    """Cohesive damage variable per (interface element, Gauss point).
    Shape ``(n_iface_elems, n_gauss)``."""

    czm_separation: np.ndarray | None = None
    """Displacement jump in the local cohesive frame per (interface
    element, Gauss point, component).  Shape ``(n_iface_elems, n_gauss,
    3)`` with components ``(delta_n, delta_s, delta_t)``."""

    czm_traction: np.ndarray | None = None
    """Cohesive traction in the local frame at each Gauss point.  Shape
    ``(n_iface_elems, n_gauss, 3)``."""

    czm_energy_dissipated: float | None = None
    """Total cohesive energy dissipated across all interfaces (N*mm)."""

    czm_energy_per_interface: dict[int, float] | None = None
    """Per-interface dissipated energy keyed by ply-interface index."""

    czm_crack_length_per_interface: dict[int, float] | None = None
    """Per-interface crack length (in mm) keyed by ply-interface index.
    Computed as the in-plane area of elements with ``damage > 0.99``,
    divided by the mesh width (so the reported quantity has units of
    length along the wrinkle/crack direction)."""

    czm_load_displacement: np.ndarray | None = None
    """``(n_inc, 2)`` array of ``(lambda, ||u||)`` samples from the
    incremental Newton-Raphson run."""

    czm_converged: bool | None = None
    """Whether every load increment converged."""

    czm_interfaces_used: list[int] | None = None
    """Ply-interface indices that actually received cohesive elements."""

    czm_delamination_report: LaminateFailureReport | None = None
    """Delamination failure report shaped like the other failure
    criteria, populated by :mod:`wrinklefe.failure.delamination`."""

    czm_element_centroids: np.ndarray | None = None
    """``(n_iface_elems, 2)`` array of in-plane ``(x, y)`` centroids of the
    cohesive interface elements, in the reference (undeformed) configuration.

    Populated alongside the other ``czm_*`` fields by ``_run_czm_path`` so
    that visualization wrappers (e.g.
    :func:`wrinklefe.viz.czm_overview_figure`) can colour the interface
    plane without needing access to the assembler / cohesive-element list.
    Same row order as ``czm_damage``."""

    def summary(self) -> str:
        """Generate a comprehensive text summary.

        Returns
        -------
        str
            Multi-line summary of all analysis results.
        """
        cfg = self.config
        lines = [
            "=" * 65,
            "  WrinkleFE Analysis Results",
            "=" * 65,
            "",
            "  Configuration:",
            f"    Morphology:      {cfg.morphology}",
            f"    Amplitude:       {cfg.amplitude:.3f} mm",
            f"    Wavelength:      {cfg.wavelength:.1f} mm",
            f"    Width:           {cfg.width:.1f} mm",
            f"    Amplitude profile: {cfg.amplitude_profile} "
            f"(d={cfg.amplitude_profile_decay_length}, "
            f"axis={cfg.amplitude_profile_axis})",
            f"    Loading:         {cfg.loading}",
            f"    Applied strain:  {cfg.applied_strain:.4f}",
            "",
            "  Analytical Predictions:",
            f"    Morphology factor M_f:  {self.morphology_factor:.4f}",
            f"    Max angle theta_max:    {np.degrees(self.max_angle_rad):.2f} deg "
            f"({self.max_angle_rad:.4f} rad)",
            f"    Effective angle:        {np.degrees(self.effective_angle_rad):.2f} deg "
            f"({self.effective_angle_rad:.4f} rad)",
            f"    Damage index D:         {self.damage_index:.4f}",
            f"    Combined knockdown:     {self.analytical_knockdown:.4f}",
            f"    Modulus knockdown:      {self.analytical_modulus_knockdown:.4f}",
            f"    Predicted strength:     {self.analytical_strength_MPa:.1f} MPa",
        ]

        if self.mesh is not None:
            lines.extend([
                "",
                "  Mesh:",
                f"    Nodes:    {self.mesh.n_nodes}",
                f"    Elements: {self.mesh.n_elements}",
                f"    DOFs:     {self.mesh.n_dof}",
            ])

        if self.field_results is not None:
            max_disp, _ = self.field_results.max_displacement()
            lines.extend([
                "",
                "  FE Results:",
                f"    Max displacement: {max_disp:.6e} mm",
                f"    Modulus retention (local σ₁₁):  {self.modulus_retention:.4f}",
                f"    Modulus retention (global E_x): "
                f"{self.modulus_retention_global:.4f}",
            ])

        if self.czm_damage is not None:
            max_d = float(np.max(self.czm_damage)) if self.czm_damage.size else 0.0
            mean_d = float(np.mean(self.czm_damage)) if self.czm_damage.size else 0.0
            energy = (
                self.czm_energy_dissipated
                if self.czm_energy_dissipated is not None
                else 0.0
            )
            iface_str = (
                ",".join(str(i) for i in self.czm_interfaces_used)
                if self.czm_interfaces_used else "(none)"
            )
            lines.extend([
                "",
                "  Cohesive Zone Modeling (delamination):",
                f"    Interfaces:        {iface_str}",
                f"    Max damage:        {max_d:.4f}",
                f"    Mean damage:       {mean_d:.4f}",
                f"    Energy dissipated: {energy:.4e} N*mm",
                f"    Converged:         {self.czm_converged}",
            ])

        lines.append("=" * 65)
        return "\n".join(lines)


# ======================================================================
# Sweep parallelism helpers (issue #260)
# ======================================================================


def _sweep_run_one(
    cfg: AnalysisConfig, analytical_only: bool
) -> AnalysisResults:
    """Run one sweep point.  Module-level so it pickles for
    ``ProcessPoolExecutor`` workers."""
    return WrinkleAnalysis(cfg).run(analytical_only=analytical_only)


def _resolve_sweep_workers(n_workers: int) -> int:
    """Validate and resolve a sweep worker count (``0`` -> all cores)."""
    if not isinstance(n_workers, int) or isinstance(n_workers, bool):
        raise ValueError(
            f"n_workers must be an int >= 0, got {n_workers!r}"
        )
    if n_workers < 0:
        raise ValueError(f"n_workers must be >= 0, got {n_workers}")
    if n_workers == 0:
        return os.cpu_count() or 1
    return n_workers


# ======================================================================
# Main analysis class
# ======================================================================

class WrinkleAnalysis:
    """High-level orchestrator for wrinkled laminate analysis.

    This class chains together all modules in the WrinkleFE framework:
    material → laminate → wrinkle → mesh → solve → failure → statistics.

    Parameters
    ----------
    config : AnalysisConfig
        Complete analysis configuration.

    Examples
    --------
    >>> config = AnalysisConfig(morphology="concave", amplitude=0.366)
    >>> analysis = WrinkleAnalysis(config)
    >>> result = analysis.run()
    >>> print(f"Strength = {result.analytical_strength_MPa:.1f} MPa")
    """

    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        analytical_only: bool | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> AnalysisResults:
        """Execute the complete analysis pipeline.

        Parameters
        ----------
        analytical_only : bool, optional
            If True, skip the FE assembly path (mesh generation, static
            solve, failure evaluation, retention factors) and return only
            the analytical predictions.  If None (default), the
            ``AnalysisConfig.analytical_only`` field is used.
        progress_callback : callable, optional
            Optional progress reporter invoked at each phase boundary as
            ``progress_callback(label, fraction)`` where ``label`` is a
            short human-readable phase name and ``fraction`` is the
            cumulative completion in ``[0, 1]``.  Defaults to ``None``
            (no reporting), so non-Streamlit callers (CLI, tests) are
            unaffected.  The callback always fires at least once with
            ``fraction == 1.0`` on successful completion.

        Steps
        -----
        1. Build laminate from material and ply angles.
        2. Create wrinkle profile and morphology configuration.
        3. Compute analytical predictions (knockdown, strength).
        4. Generate mesh with wrinkle deformation.
        5. Run static FE analysis.
        6. Evaluate failure criteria on the FE stress field.

        Returns
        -------
        AnalysisResults
            Complete results from all analysis steps.
        """
        cfg = self.config
        # interface_1 / interface_2 are filled in __post_init__.
        assert cfg.interface_1 is not None and cfg.interface_2 is not None
        # _validate constrained these to the literal sets that
        # WrinkleConfiguration expects.
        amp_profile = cast(
            Literal["constant", "gaussian", "linear"], cfg.amplitude_profile
        )
        amp_axis = cast(Literal["x", "y"], cfg.amplitude_profile_axis)
        if analytical_only is None:
            analytical_only = cfg.analytical_only

        # Multi-wrinkle FE solve (issue #252): overlapping and
        # non-overlapping layouts run through the linear FE path, and
        # (issue #283) through the CZM path — cohesive layers are
        # inserted along the full length of every interface a wrinkle
        # nominates, so delaminations can link up between adjacent
        # wrinkles sharing an interface.
        results = AnalysisResults(config=cfg)

        # Phase weights (sum to 1.0 on the full FE path).  The FE solve
        # dominates wall-clock for typical mesh densities, so it gets the
        # largest slice.  In analytical-only mode the analytical step is
        # rescaled to 1.0 below.
        #
        #   build mesh / wrinkle geom : 0.05
        #   analytical predictions    : 0.05
        #   FE assembly (mesh build)  : 0.15
        #   FE solve                  : 0.50
        #   failure evaluation        : 0.15  (stress recovery + criteria)
        #   retention factors         : 0.10
        def _report(label: str, fraction: float) -> None:
            if progress_callback is not None:
                progress_callback(label, max(0.0, min(1.0, float(fraction))))

        logger.info(
            "Starting analysis: morphology=%s A=%.4g lambda=%.4g "
            "analytical_only=%s czm=%s",
            cfg.morphology, cfg.amplitude, cfg.wavelength,
            analytical_only, cfg.enable_czm,
        )

        _report("Building laminate", 0.0)

        # 1. Build laminate
        laminate = self._build_laminate()
        results.laminate = laminate

        # 2. Create wrinkle configuration (centered in specimen)
        wrinkle_center = cfg.domain_length / 2.0

        # Multi-wrinkle override: build a WrinkleConfiguration directly
        # from the user-supplied WrinkleSpec list.  Each spec carries its
        # own geometry (A, lambda, w) and is placed at its own ply
        # interface with its own phase offset.  Bypasses the single/dual
        # name dispatch entirely.
        if cfg.wrinkles is not None:
            placements = []
            for spec in cfg.wrinkles:
                spec_profile = GaussianSinusoidal(
                    amplitude=spec.amplitude,
                    wavelength=spec.wavelength,
                    width=spec.width,
                    center=wrinkle_center,
                )
                placements.append(
                    WrinklePlacement(
                        profile=spec_profile,
                        ply_interface=spec.ply_interface,
                        phase_offset=spec.phase_offset,
                    )
                )
            is_graded = cfg.morphology.lower().strip() == "graded"
            wrinkle_config = WrinkleConfiguration(
                placements,
                decay_mode="graded" if is_graded else "default",
                decay_floor=cfg.decay_floor if is_graded else 0.0,
                amplitude_profile=amp_profile,
                amplitude_profile_decay_length=cfg.amplitude_profile_decay_length,
                amplitude_profile_axis=amp_axis,
            )
            results.wrinkle_config = wrinkle_config
        else:
            profile = GaussianSinusoidal(
                amplitude=cfg.amplitude,
                wavelength=cfg.wavelength,
                width=cfg.width,
                center=wrinkle_center,
            )
            if cfg.phase is not None and (
                cfg.morphology.lower().strip() not in SINGLE_WRINKLE_MODES
            ):
                # Explicit phase overrides the named-morphology phase so
                # arbitrary dual-wrinkle phase offsets can be analysed/swept.
                wrinkle_config = WrinkleConfiguration.dual_wrinkle(
                    profile,
                    interface1=cfg.interface_1,
                    interface2=cfg.interface_2,
                    phase=float(cfg.phase),
                    amplitude_profile=amp_profile,
                    amplitude_profile_decay_length=cfg.amplitude_profile_decay_length,
                    amplitude_profile_axis=amp_axis,
                )
                wrinkle_config.decay_floor = max(0.0, min(1.0, cfg.decay_floor))
            else:
                wrinkle_config = WrinkleConfiguration.from_morphology_name(
                    cfg.morphology, profile,
                    interface1=cfg.interface_1,
                    interface2=cfg.interface_2,
                    decay_floor=cfg.decay_floor,
                    amplitude_profile=amp_profile,
                    amplitude_profile_decay_length=cfg.amplitude_profile_decay_length,
                    amplitude_profile_axis=amp_axis,
                )
            results.wrinkle_config = wrinkle_config

        # Through-thickness wrinkle position (item D.5): the graded decay
        # is centred here (0.5 = mid-plane).  Off-mid values place the
        # wrinkle nearer a surface (Li 2025 S-A-2).
        wrinkle_config.wrinkle_z_position = float(cfg.wrinkle_z_position)
        results.wrinkle_config = wrinkle_config

        _report("Computing analytical predictions", 0.05)

        # 3. Analytical predictions
        self._compute_analytical(results, wrinkle_config)

        # In analytical-only mode, skip mesh, solve, failure, and retention.
        if analytical_only:
            _report("Analytical predictions complete", 1.0)
            logger.info(
                "Analysis complete (analytical only): knockdown=%s",
                results.analytical_knockdown,
            )
            return results

        _report("Assembling FE mesh", 0.10)

        # 4. Generate mesh
        mesh_gen = WrinkleMesh(
            laminate=laminate,
            wrinkle_config=wrinkle_config,
            Lx=cfg.domain_length,
            Ly=cfg.domain_width,
            nx=cfg.nx,
            ny=cfg.ny,
            nz_per_ply=cfg.nz_per_ply,
        )
        mesh = mesh_gen.generate()

        # 4a2. Resin-pocket material zone (Li et al. 2024/2025).
        if cfg.enable_resin_pocket:
            self._attach_resin_pocket(mesh, laminate)

        results.mesh = mesh

        # 4b. Mesh-based max fiber angle (accounts for decay mode)
        results.mesh_max_angle_rad = (
            float(np.max(mesh.fiber_angles))
            if mesh.fiber_angles.size > 0 else 0.0
        )

        _report("Solving FE system", 0.25)

        # 5. FE solve — branch on CZM mode.
        if cfg.enable_czm:
            self._run_czm_path(results, laminate, mesh, wrinkle_config)
            _report("Analysis complete", 1.0)
            logger.info(
                "Analysis complete (CZM path): knockdown=%s",
                results.analytical_knockdown,
            )
            return results

        # Progressive-damage path: carries the solve to ultimate load and
        # reports a strength knockdown (the only path that knocks down
        # pristine UD compression).  Still runs the linear solve below so
        # the usual field/failure/retention outputs remain populated.
        if cfg.enable_progressive_damage:
            self._run_progressive_path(results, laminate, mesh)

        # Linear (legacy) path.
        solver = StaticSolver(mesh, laminate)
        bcs = BoundaryHandler.compression_bcs(
            mesh, applied_strain=cfg.applied_strain
        )
        field_results = solver.solve(
            bcs, solver=cfg.solver, verbose=cfg.verbose
        )
        results.field_results = field_results

        _report("Evaluating failure criteria", 0.75)

        # 6. Failure evaluation on FE field
        self._evaluate_failure(results, laminate, field_results, mesh)

        _report("Computing retention factors", 0.90)

        # 6b. Retention factor (baseline pristine comparison)
        self._compute_retention_factors(results, laminate)

        _report("Analysis complete", 1.0)
        logger.info(
            "Analysis complete: knockdown=%s",
            results.analytical_knockdown,
        )

        return results

    # ------------------------------------------------------------------
    # Morphology comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare_morphologies(
        base_config: AnalysisConfig,
        morphologies: Sequence[str] = ("stack", "convex", "concave"),
        analytical_only: bool = False,
    ) -> dict[str, AnalysisResults]:
        """Run the full FE analysis for multiple morphologies and compare.

        Parameters
        ----------
        base_config : AnalysisConfig
            Base configuration.  The ``morphology`` field is overridden
            for each entry in *morphologies*.
        morphologies : sequence of str
            Morphology names to compare.
        analytical_only : bool, optional
            If True, skip the FE assembly path for each morphology and
            return only the analytical predictions.  Default ``False``.

        Returns
        -------
        dict[str, AnalysisResults]
            Mapping from morphology name to its results.
        """
        all_results: dict[str, AnalysisResults] = {}

        for morph in morphologies:
            cfg = replace(base_config, morphology=morph)
            all_results[morph] = WrinkleAnalysis(cfg).run(
                analytical_only=analytical_only
            )

        return all_results

    # ------------------------------------------------------------------
    # Parametric sweep
    # ------------------------------------------------------------------

    @staticmethod
    def parametric_sweep(
        base_config: AnalysisConfig,
        parameter: str,
        values: Sequence[float],
        analytical_only: bool = False,
        n_workers: int = 1,
    ) -> list[AnalysisResults]:
        """Sweep a single parameter over a range of values.

        Parameters
        ----------
        base_config : AnalysisConfig
            Base configuration to clone for each value.
        parameter : str
            Name of the parameter to vary.  Must be a numeric field of
            :class:`AnalysisConfig` (e.g. ``'amplitude'``, ``'wavelength'``,
            ``'width'``, ``'applied_strain'``).
        values : sequence of float
            Parameter values to evaluate.
        analytical_only : bool, optional
            If True, skip the FE assembly path for each sweep value and
            return only the analytical predictions.  Default ``False``.
        n_workers : int, optional
            Number of worker processes (issue #260).  ``1`` (default)
            keeps the sequential in-process path; ``0`` uses all CPU
            cores; ``> 1`` runs the independent per-value analyses on a
            ``ProcessPoolExecutor``.  Results come back in the order of
            *values* either way.  Peak memory scales with ``n_workers``
            x the per-solve footprint (each worker returns a full
            :class:`AnalysisResults`, including mesh and field arrays on
            the FE path) — size the worker count by available RAM for
            fine meshes.

        Returns
        -------
        list[AnalysisResults]
            One result per value, in the same order as *values*.

        Raises
        ------
        AttributeError
            If *parameter* is not a valid :class:`AnalysisConfig` field.
        """
        valid_field_names = {f.name for f in fields(base_config)}
        if parameter not in valid_field_names:
            raise AttributeError(
                f"AnalysisConfig has no field '{parameter}'"
            )
        n_workers = _resolve_sweep_workers(n_workers)

        # If domain_length was auto-derived from wavelength in the base
        # config (i.e. user left it at the sentinel 0.0), we must reset
        # it to the sentinel before each replace() so __post_init__
        # re-derives it from the swept value. ``replace`` only invokes
        # ``__post_init__``; it does not reset un-passed fields, so a
        # previously-derived ``domain_length`` would otherwise stick.
        reset_domain_length = (
            parameter == "wavelength"
            and "domain_length" in valid_field_names
            and base_config.domain_length == 3.0 * base_config.wavelength
        )

        configs: list[AnalysisConfig] = []
        for val in values:
            overrides: dict[str, Any] = {parameter: val}
            if reset_domain_length:
                overrides["domain_length"] = 0.0
            configs.append(replace(base_config, **overrides))

        if n_workers == 1:
            return [
                _sweep_run_one(cfg, analytical_only) for cfg in configs
            ]

        # Parallel path: each value is an independent analysis, so fan
        # the solves out over processes.  ``executor.map`` preserves the
        # submission order, so the returned list lines up with *values*
        # exactly like the sequential path.
        executor = ProcessPoolExecutor(max_workers=n_workers)
        try:
            results_list = list(
                executor.map(
                    _sweep_run_one,
                    configs,
                    itertools.repeat(analytical_only),
                )
            )
        except BaseException:
            # KeyboardInterrupt (or a worker failure): cancel queued
            # futures instead of letting the pool drain.
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
        return results_list

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_laminate(self) -> Laminate:
        """Build the Laminate from config."""
        # material / angles are filled in __post_init__.
        assert self.config.angles is not None
        assert self.config.material is not None
        return Laminate.from_angles(
            self.config.angles,
            self.config.material,
            ply_thickness=self.config.ply_thickness,
        )

    # ------------------------------------------------------------------
    # Cohesive-zone modelling (Phase 3 wiring)
    # ------------------------------------------------------------------

    def _resolve_cohesive_interfaces(
        self, laminate: Laminate, wrinkle_config: WrinkleConfiguration,
    ) -> list[int]:
        """Resolve ``cfg.czm_interfaces`` to an explicit ply-interface list.

        Parameters
        ----------
        laminate : Laminate
            The fully built laminate; supplies the ply z-coordinates.
        wrinkle_config : WrinkleConfiguration
            Used to locate the wrinkle peaks when
            ``cfg.czm_interfaces == "near_crest"``: scalar configs pick
            the single interface nearest the (largest-amplitude)
            wrinkle; multi-wrinkle configs pick the interface nearest
            *each* wrinkle, deduplicated (issue #283).

        Returns
        -------
        list[int]
            Sorted, de-duplicated list of ply-interface indices in
            ``[0, n_plies - 2]`` (0 = bottom-most interior interface).
        """
        cfg = self.config
        n_plies = laminate.n_plies
        if isinstance(cfg.czm_interfaces, list):
            return sorted({int(i) for i in cfg.czm_interfaces})
        if cfg.czm_interfaces == "all":
            return list(range(n_plies - 1))
        if cfg.czm_interfaces == "near_crest":
            # Pick the *interior* interface closest to the wrinkle peak.
            # The wrinkle peak amplitude in the un-deformed (flat) mesh
            # lies at the wrinkle's reference centre, i.e. at the
            # nominal z of its ply-interface index, shifted in z by the
            # wrinkle's peak displacement.  For the dual-wrinkle modes
            # the two interfaces straddle the laminate midplane; for
            # single-wrinkle modes there is one wrinkle interface.
            ply_z = laminate.z_coords()  # length n_plies + 1
            # Note: interface_z[i] is the midpoint between plies i+1 and
            # i+2; we want a z value associated with the boundary
            # between plies i and i+1, i.e. ply_z[i+1].  Use that
            # directly.
            boundary_z = ply_z[1:n_plies]  # internal boundaries
            # The wrinkle's reference centre z is the boundary z of its
            # ply_interface.  Take the wrinkle with the largest amplitude
            # (most-likely-to-delaminate driver) and pick the boundary
            # closest to that wrinkle's centre.
            wrinkles = list(getattr(wrinkle_config, "wrinkles", ()))

            def _nearest_boundary(placement) -> int:
                # The wrinkle's z reference is the ply boundary above ply
                # ``ply_interface``; ``ply_interface`` is in the [0,
                # n_plies-2] range and references the boundary at
                # ply_z[ply_interface + 1].
                k = int(placement.ply_interface)
                if 1 <= k + 1 <= n_plies - 1:
                    target_z = float(ply_z[k + 1])
                else:
                    target_z = 0.0
                return int(np.argmin(np.abs(boundary_z - target_z)))

            if not wrinkles:
                # No wrinkle placements (flat mesh) — default to the
                # interior boundary nearest the midplane.
                return [int(np.argmin(np.abs(boundary_z)))]
            if cfg.wrinkles is not None:
                # Multi-wrinkle configuration (issue #283): nominate the
                # interface nearest *each* wrinkle so a delamination can
                # initiate at any crest and, where wrinkles share an
                # interface index, run along one continuous cohesive
                # surface between them (crest-to-crest link-up).
                return sorted({_nearest_boundary(w) for w in wrinkles})
            # Scalar (named-morphology) configuration: keep the legacy
            # single-interface choice — the wrinkle with the largest
            # amplitude is the most-likely-to-delaminate driver.
            w_best = max(
                wrinkles,
                key=lambda w: abs(getattr(w.profile, "amplitude", 0.0)),
            )
            return [_nearest_boundary(w_best)]
        # Validation in ``__post_init__`` already constrains the values
        # this method sees; the catch-all keeps mypy happy.
        raise ValueError(
            f"Unrecognised czm_interfaces value: {cfg.czm_interfaces!r}"
        )

    def _build_cohesive_properties(
        self, laminate: Laminate,
    ) -> CohesiveProperties:
        """Build ``CohesiveProperties`` from config-or-material defaults.

        Falls back to ply-0's material when a CZM strength / toughness
        is not explicitly set on the config.  Raises ``ValueError`` when
        the laminate's first ply lacks ``GIc`` / ``GIIc`` and the user
        did not override them on the config.
        """
        cfg = self.config
        # Reference material: ply 0 (simpler than per-interface lookup,
        # matches the v1 spec).
        mat = laminate.plies[0].material

        def _coalesce(cfg_val: float | None, mat_val) -> float:
            if cfg_val is not None:
                return float(cfg_val)
            if mat_val is None:
                raise ValueError(
                    "Cohesive zone modelling requires GIc, GIIc, "
                    "sigma_max and tau_max either as explicit "
                    "AnalysisConfig.czm_* overrides or as material "
                    "defaults.  Material "
                    f"{mat.name!r} (ply 0) is missing one of them."
                )
            return float(mat_val)

        return CohesiveProperties(
            K=float(cfg.czm_penalty),
            sigma_max=_coalesce(cfg.czm_sigma_max, mat.sigma_max),
            tau_max=_coalesce(cfg.czm_tau_max, mat.tau_max),
            GIc=_coalesce(cfg.czm_GIc, mat.GIc),
            GIIc=_coalesce(cfg.czm_GIIc, mat.GIIc),
            eta_BK=float(cfg.czm_BK_eta),
            beta=1.0,
        )

    def _build_mesh_with_cohesive_interfaces(
        self,
        laminate: Laminate,
        wrinkle_config: WrinkleConfiguration,
        iface_indices: list[int],
        cohesive_props: CohesiveProperties,
    ) -> tuple[MeshData, list[tuple[int, Cohesive8Element]], dict[int, range]]:
        """Build a wrinkled mesh with cohesive elements inserted.

        The flat (un-deformed) mesh is built first so that ply-interface
        nodes lie on exact z-planes; cohesive elements are inserted on
        those flat planes; finally the wrinkle displacement field is
        applied to **all** nodes (including the duplicated interface
        nodes), so the cohesive layer sits at a curved surface in the
        deformed configuration.  This keeps the
        :func:`insert_cohesive_interface` topology check valid (which
        requires axis-aligned interface planes in the reference
        configuration) while letting the rest of the pipeline see the
        normal wrinkled geometry.

        Returns
        -------
        mesh : MeshData
            Final wrinkled mesh with duplicated interface nodes.
        cohesive_elements : list[tuple[int, Cohesive8Element]]
            Entries suitable for :class:`GlobalAssembler`.  The integer
            key is the *global* cohesive-element id, unique across
            interfaces.  Element ``node_coords`` are refreshed against
            the wrinkled ``mesh.nodes`` so the assembler's exact-equality
            check passes.
        elem_ranges : dict[int, range]
            Map from ply-interface index to the contiguous range of
            global cohesive-element ids belonging to that interface,
            used downstream for per-interface aggregation.
        """
        cfg = self.config

        # --- Step 1: flat mesh, no wrinkle deformation ---------------
        flat_mesh_gen = WrinkleMesh(
            laminate=laminate,
            wrinkle_config=None,
            Lx=cfg.domain_length,
            Ly=cfg.domain_width,
            nx=cfg.nx,
            ny=cfg.ny,
            nz_per_ply=cfg.nz_per_ply,
        )
        mesh = flat_mesh_gen.generate()

        # --- Step 2: insert cohesive layers at requested z-planes ----
        ply_z = laminate.z_coords()  # length n_plies + 1
        # ply-interface index ``i`` corresponds to the boundary between
        # plies ``i`` and ``i + 1`` => z = ply_z[i + 1].
        elem_ranges: dict[int, range] = {}
        cohesive_elements: list[tuple[int, Cohesive8Element]] = []
        next_global_id = 0
        for iface_idx in iface_indices:
            z_iface = float(ply_z[iface_idx + 1])
            mesh, coh_elems = insert_cohesive_interface(
                mesh, z_iface, cohesive_props,
            )
            start = next_global_id
            for k, c_elem in enumerate(coh_elems):
                # Reassign elem_id to be unique across interfaces.
                c_elem.elem_id = next_global_id
                cohesive_elements.append((next_global_id, c_elem))
                next_global_id += 1
            elem_ranges[iface_idx] = range(start, next_global_id)

        # --- Step 3: apply wrinkle deformation to the expanded mesh -
        # Rebuild the node_ply_ids array for the new (duplicated) node
        # set.  WrinkleMesh._node_to_element_ply assigns ply ids by the
        # z-layer index; for duplicated nodes the natural choice is to
        # carry the original ply id of the node they were duplicated
        # from.  The cleanest way to derive this without re-deriving
        # internal mesh state is to assign each node by the z-value of
        # its original (flat) coordinate, falling back to the original
        # ply id for nodes that exactly straddle a boundary.
        node_ply_ids = self._derive_node_ply_ids(mesh, laminate)
        deformed_nodes = wrinkle_config.apply_to_nodes(
            mesh.nodes, node_ply_ids, laminate.n_plies,
        )
        fiber_angles = wrinkle_config.fiber_angles_at_nodes(
            mesh.nodes, node_ply_ids, n_plies=laminate.n_plies,
        )
        mesh = MeshData(
            nodes=deformed_nodes,
            elements=mesh.elements,
            ply_ids=mesh.ply_ids,
            fiber_angles=fiber_angles,
            ply_angles=mesh.ply_angles,
            nx=mesh.nx,
            ny=mesh.ny,
            nz=mesh.nz,
            laminate=laminate,
        )

        # --- Step 4: refresh cohesive node_coords against the deformed
        # node array so the GlobalAssembler equality check passes -----
        refreshed: list[tuple[int, Cohesive8Element]] = []
        for gid, c_elem in cohesive_elements:
            new_coords = mesh.nodes[c_elem.node_ids]
            new_elem = Cohesive8Element(
                node_coords=new_coords,
                properties=c_elem.properties,
                node_ids=c_elem.node_ids,
                elem_id=c_elem.elem_id,
            )
            refreshed.append((gid, new_elem))
        return mesh, refreshed, elem_ranges

    @staticmethod
    def _derive_node_ply_ids(mesh: MeshData, laminate: Laminate) -> np.ndarray:
        """Assign a ply id to every mesh node from its z-coordinate.

        Used after :func:`insert_cohesive_interface` has duplicated
        interface nodes: those duplicates need the *same* ply id as the
        originals so the wrinkle through-thickness decay treats them as
        belonging to the same ply.  A node sitting exactly on a ply
        boundary z is assigned to the ply *below* (lower index); the
        wrinkle decay field is continuous across boundaries so this
        choice has no physical consequence.
        """
        ply_z = laminate.z_coords()  # n_plies + 1
        z = mesh.nodes[:, 2]
        # ``np.searchsorted(side='right') - 1`` puts a node sitting
        # exactly on a boundary into the ply below the boundary.
        ids = np.searchsorted(ply_z, z, side="right") - 1
        ids = np.clip(ids, 0, laminate.n_plies - 1)
        return ids.astype(np.intp)

    def _run_czm_path(
        self,
        results: AnalysisResults,
        laminate: Laminate,
        flat_mesh: MeshData,
        wrinkle_config: WrinkleConfiguration,
    ) -> None:
        """End-to-end CZM solve: insert interfaces, run Newton, populate results.

        The ``flat_mesh`` argument is the wrinkled hex8 mesh that the
        linear path would use; we rebuild a *different* mesh here that
        also carries cohesive layers (see
        :meth:`_build_mesh_with_cohesive_interfaces`) and use that for
        the solve.  The mesh passed in is otherwise unused — it is the
        sentinel value that the linear-path callers (``_evaluate_failure``,
        ``_compute_retention_factors``) consume; we overwrite
        ``results.mesh`` with our enriched mesh below.
        """
        cfg = self.config

        iface_indices = self._resolve_cohesive_interfaces(
            laminate, wrinkle_config,
        )
        results.czm_interfaces_used = list(iface_indices)

        cohesive_props = self._build_cohesive_properties(laminate)
        mesh, cohesive_elements, elem_ranges = (
            self._build_mesh_with_cohesive_interfaces(
                laminate, wrinkle_config, iface_indices, cohesive_props,
            )
        )
        results.mesh = mesh
        if mesh.fiber_angles.size > 0:
            results.mesh_max_angle_rad = float(np.max(mesh.fiber_angles))

        # Boundary conditions: ``compression_bcs`` is sign-agnostic (the
        # sign of ``applied_strain`` selects compression vs tension), so
        # we use it for both loading modes here, matching the linear
        # path.
        bcs = BoundaryHandler.compression_bcs(
            mesh, applied_strain=cfg.applied_strain
        )
        bc_handler = BoundaryHandler(mesh)
        assembler = GlobalAssembler(
            mesh, laminate, cohesive_elements=cohesive_elements,
        )

        solver = NewtonRaphsonSolver(
            assembler=assembler,
            bc_handler=bc_handler,
            boundary_conditions=bcs,
            n_increments=int(cfg.czm_n_load_increments),
            tol_residual=float(cfg.czm_newton_tol),
        )
        outcome = solver.solve(verbose=cfg.verbose)

        results.czm_converged = bool(outcome.get("converged", False))
        results.czm_load_displacement = outcome.get(
            "load_displacement", None,
        )

        # ----- Bulk hex8 stress / strain recovery from the final Newton u -----
        # The CZM path now populates ``field_results`` so users can run
        # ply-level failure criteria (LaRC05, Hashin, ...) on the bulk
        # material alongside the interface delamination output.  We reuse
        # StaticSolver's stress-recovery machinery: instantiate it on the
        # CZM-enriched mesh (without cohesive_elements registered, so the
        # assemble_stiffness guard does not fire) and call
        # recover_element_results on the converged displacement.
        u_final = outcome["displacement"]
        recovery_solver = StaticSolver(mesh, laminate)
        stress_g, stress_l, strain_g, strain_l = (
            recovery_solver.recover_element_results(u_final, verbose=False)
        )
        results.field_results = FieldResults(
            displacement=u_final.reshape(-1, 3),
            stress_global=stress_g,
            stress_local=stress_l,
            strain_global=strain_g,
            strain_local=strain_l,
            mesh=mesh,
            laminate=laminate,
        )
        self._evaluate_failure(results, laminate, results.field_results, mesh)

        # ----- Extract per-Gauss-point CZM state -----
        n_iface = len(cohesive_elements)
        if n_iface == 0:
            results.czm_damage = np.empty((0, 0))
            results.czm_separation = np.empty((0, 0, 3))
            results.czm_traction = np.empty((0, 0, 3))
            results.czm_energy_dissipated = 0.0
            results.czm_energy_per_interface = {}
            results.czm_crack_length_per_interface = {}
            results.czm_element_centroids = np.empty((0, 2))
            results.czm_delamination_report = build_delamination_report({})
            return

        n_gp = cohesive_elements[0][1].n_gp
        damage = np.zeros((n_iface, n_gp), dtype=float)
        separation = np.zeros((n_iface, n_gp, 3), dtype=float)
        traction = np.zeros((n_iface, n_gp, 3), dtype=float)

        # In-plane (x, y) centroid of each interface element in the
        # reference configuration; consumed by the viz layer to colour
        # damage on the interface plane without touching the assembler.
        centroids_xy = np.zeros((n_iface, 2), dtype=float)
        for row, (_gid, c_elem) in enumerate(cohesive_elements):
            bottom_xy = c_elem.node_coords[:4, :2]
            centroids_xy[row] = bottom_xy.mean(axis=0)
        results.czm_element_centroids = centroids_xy

        u = outcome["displacement"]
        # Iterate in the same order as `cohesive_elements` was built.
        for row, (gid, c_elem) in enumerate(cohesive_elements):
            # Damage from the assembler-committed state.  Fall back to
            # virgin Gauss-point states (d = 0) for elements the
            # assembler never committed; ``_law_local`` requires a real
            # ``CohesiveState`` (a ``None`` entry would crash it).
            state = assembler.cohesive_state.get(
                gid, make_initial_state(c_elem.n_gp)
            )
            for g in range(c_elem.n_gp):
                damage[row, g] = float(state[g].d)

            # Recompute the local separation and traction at each GP
            # from the final displacement.  ``tangent_and_force`` returns
            # the assembled force / tangent but does not expose
            # per-GP values; we recompute the kinematic / constitutive
            # pieces directly here.
            dofs = GlobalAssembler._cohesive_dof_indices(c_elem)
            u_e = u[dofs]
            for g in range(c_elem.n_gp):
                B = c_elem._B_jump(g)
                R = c_elem._R_gp[g]
                delta_global = B @ u_e
                delta_local = R @ delta_global
                separation[row, g] = delta_local
                # Traction via the committed-state law evaluation.
                T_local, _D_local, _new = c_elem._law_local(
                    delta_local, state[g],
                )
                traction[row, g] = T_local

        results.czm_damage = damage
        results.czm_separation = separation
        results.czm_traction = traction

        # ----- Per-interface energy + crack length -----
        # Per-element dissipated energy (approximate):
        # E_e ≈ 0.5 * sigma_max * delta_f * area_e * d_avg, where
        # delta_f is mode-dependent.  We use the simpler bound
        # GIc * area * d_avg which is accurate for mode-I-dominated
        # failure and gives the right total in the fully-failed limit.
        energy_per_iface: dict[int, float] = {}
        crack_len_per_iface: dict[int, float] = {}
        damage_per_iface: dict[int, np.ndarray] = {}

        # Mesh width in y for crack-length estimation.  Use the bounding
        # box of the *original* (flat) coordinates which the cohesive
        # element retains internally; falls back to the deformed mesh.
        y_min = float(mesh.nodes[:, 1].min())
        y_max = float(mesh.nodes[:, 1].max())
        Ly = max(y_max - y_min, 1e-12)

        for iface_idx, gid_range in elem_ranges.items():
            rows = list(range(gid_range.start, gid_range.stop))
            d_iface = damage[rows]
            damage_per_iface[iface_idx] = d_iface

            # Energy: sum per element of GIc * area * d_mean (mode-I
            # bound).  Mode-mixity refines this in principle but the
            # mode-I approximation suffices for the v1 reporter.
            e_iface = 0.0
            crack_area = 0.0
            for row, gid in enumerate(rows):
                _, c_elem = cohesive_elements[gid]
                area_e = c_elem.area
                d_avg = float(d_iface[row].mean())
                e_iface += cohesive_props.GIc * area_e * d_avg
                if d_iface[row].max() > 0.99:
                    crack_area += area_e
            energy_per_iface[iface_idx] = float(e_iface)
            crack_len_per_iface[iface_idx] = float(crack_area / Ly)

        results.czm_energy_per_interface = energy_per_iface
        results.czm_crack_length_per_interface = crack_len_per_iface
        results.czm_energy_dissipated = float(sum(energy_per_iface.values()))
        results.czm_delamination_report = build_delamination_report(
            damage_per_iface,
            energy_per_interface=energy_per_iface,
            crack_length_per_interface=crack_len_per_iface,
        )

    def _analytical_modulus_knockdown(
        self, angles: list[float], morphology_factor: float
    ) -> float:
        """Closed-form axial-modulus knockdown for a wavy laminate.

        Populated for arbitrary layups and multi-wrinkle layouts. The
        unidirectional single-wrinkle case is served by the scalar fast path
        :func:`_profile_modulus_knockdown` (so the pinned UD baselines stay
        bit-identical); every other case is routed through the CLT membrane
        series-average :func:`_laminate_modulus_knockdown`, which reduces
        exactly to the UD result for ``[0]_n``. Returns ``1.0`` only for a
        degenerate (zero-amplitude / zero-wavelength) wrinkle.
        """
        cfg = self.config
        mat = cfg.material
        assert mat is not None  # filled in __post_init__
        graded = cfg.morphology.lower().strip() == "graded"

        # --- Single-wrinkle UD fast path (unchanged, pins F/G baselines) ---
        if cfg.wrinkles is None and _is_unidirectional(angles):
            if cfg.wavelength <= 1e-12 or cfg.amplitude <= 0.0:
                return 1.0
            if cfg.through_thickness_decay_scale is not None:
                decay_scale = float(cfg.through_thickness_decay_scale)
            else:
                decay_scale = max(cfg.wavelength / 2.0, cfg.amplitude)
            return _profile_modulus_knockdown(
                amplitude=cfg.amplitude,
                wavelength=cfg.wavelength,
                width=cfg.width,
                domain_length=cfg.domain_length,
                ply_thickness=cfg.ply_thickness,
                n_plies=len(angles),
                E1=mat.E1, E2=mat.E2, G12=mat.G12, nu12=mat.nu12,
                morphology_factor=morphology_factor,
                through_thickness_decay=graded,
                z_position_fraction=float(cfg.wrinkle_z_position),
                decay_scale=decay_scale,
                decay_floor=float(cfg.decay_floor),
            )

        # --- Generalized laminate / multi-wrinkle path -------------------
        # Build the longitudinal slope field(s) and the per-ply
        # through-thickness decay, then hand to the CLT membrane
        # series-average. The composition mirrors the FE's
        # "compose then differentiate" multi-wrinkle field.
        n_plies = len(angles)
        if n_plies == 0:
            return 1.0
        ply_thickness = cfg.ply_thickness
        total_thickness = n_plies * ply_thickness
        decay_floor = float(cfg.decay_floor)
        x = np.linspace(
            -cfg.domain_length / 2.0, cfg.domain_length / 2.0, _N_PROFILE_PTS
        )

        # Collect (amplitude, wavelength, width, z_center, decay_scale,
        # phase_shift_x) for every wrinkle. The single (non-UD) wrinkle and
        # the multi-wrinkle list are normalized to the same representation.
        specs: list[tuple[float, float, float, float, float, float]] = []
        if cfg.wrinkles is not None:
            for spec in cfg.wrinkles:
                if spec.wavelength <= 1e-12 or spec.amplitude <= 0.0:
                    continue
                # Through-thickness decay centred on this wrinkle's interface.
                z_center = (spec.ply_interface + 1.0) * ply_thickness
                z_center = min(max(z_center, 0.0), total_thickness)
                ds = (
                    float(cfg.through_thickness_decay_scale)
                    if cfg.through_thickness_decay_scale is not None
                    else max(spec.wavelength / 2.0, spec.amplitude)
                )
                dx_shift = spec.phase_offset * spec.wavelength / (2.0 * np.pi)
                specs.append(
                    (spec.amplitude, spec.wavelength, spec.width,
                     z_center, ds, dx_shift)
                )
        else:
            if cfg.wavelength <= 1e-12 or cfg.amplitude <= 0.0:
                return 1.0
            z_center = float(cfg.wrinkle_z_position) * total_thickness
            ds = (
                float(cfg.through_thickness_decay_scale)
                if cfg.through_thickness_decay_scale is not None
                else max(cfg.wavelength / 2.0, cfg.amplitude)
            )
            specs.append(
                (cfg.amplitude, cfg.wavelength, cfg.width, z_center, ds, 0.0)
            )

        if not specs:
            return 1.0

        n_w = len(specs)
        # Per-wrinkle slope along x and per-(ply, wrinkle) decay.
        slope_field = np.empty((n_w, _N_PROFILE_PTS), dtype=float)
        ply_decays = np.empty((n_plies, n_w, _N_PROFILE_PTS), dtype=float)
        use_decay = graded or cfg.wrinkles is not None
        for w, (amp, lam, wid, z_center, ds, dx_shift) in enumerate(specs):
            dxw = x - dx_shift
            gauss_env = np.exp(-(dxw ** 2) / (wid ** 2))
            k = 2.0 * np.pi / lam
            slope = amp * gauss_env * (
                (-2.0 * dxw / (wid ** 2)) * np.cos(k * dxw)
                - k * np.sin(k * dxw)
            )
            slope_field[w] = slope * morphology_factor
            sigma_sq2 = 2.0 * ds * ds
            for p in range(n_plies):
                z_p = (p + 0.5) * ply_thickness
                if use_decay:
                    raw = math.exp(-((z_p - z_center) ** 2) / sigma_sq2)
                    ply_decays[p, w, :] = decay_floor + (1.0 - decay_floor) * raw
                else:
                    ply_decays[p, w, :] = 1.0

        laminate = Laminate.from_angles(list(angles), mat, ply_thickness)
        return _laminate_modulus_knockdown(
            slope_field=slope_field,
            ply_decays=ply_decays,
            angles=angles,
            stiffness_3d=mat.stiffness_matrix,
            ply_thickness=ply_thickness,
            E_x0=laminate.Ex,
        )

    def _compute_analytical(
        self,
        results: AnalysisResults,
        wrinkle_config: WrinkleConfiguration,
    ) -> None:
        """Fill analytical prediction fields in results.

        Uses the unattenuated sinusoidal angle theta = arctan(2*pi*A/lambda)
        rather than the Gaussian-envelope max angle, because D/T-based
        experimental data references the full wrinkle amplitude A.
        """
        cfg = self.config

        mf = wrinkle_config.aggregate_morphology_factor(cfg.loading)

        # Unattenuated sinusoidal angle: theta = arctan(2*pi*A/lambda)
        # This is the correct angle for D/T-based knockdown comparison,
        # since D/T uses the full amplitude A without Gaussian attenuation.
        if cfg.wrinkles is not None:
            # Multi-wrinkle analytical model: peak-angle over all wrinkles (initial implementation).
            # Each wrinkle gets its own theta_max,i = arctan(2*pi*A_i/lambda_i)
            # and we take the maximum, then scale by the aggregate morphology
            # factor. This is intentionally coarse; calibration against the
            # Li et al. (2025) Dataset F multi-wrinkle specimens is a
            # follow-up activity (D-AB-2, D-A-2, D-M-2, T-M-2).
            theta_max = 0.0
            for spec in cfg.wrinkles:
                if spec.wavelength > 1e-12:
                    theta_i = float(
                        np.arctan(2.0 * np.pi * spec.amplitude / spec.wavelength)
                    )
                    if theta_i > theta_max:
                        theta_max = theta_i
        elif cfg.wavelength > 1e-12:
            theta_max = float(np.arctan(2.0 * np.pi * cfg.amplitude / cfg.wavelength))
        else:
            theta_max = 0.0
        theta_eff = theta_max * mf

        # Analytical stiffness (axial-modulus) knockdown — generalized to
        # arbitrary layups and multi-wrinkle layouts, loading-independent,
        # set on both the gate and Budiansky-Fleck paths below. Resolve the
        # layup the same way each path does.
        if cfg.penetration_gate is not None:
            _mod_angles = cfg.angles if cfg.angles else [0.0]
        else:
            _mod_angles = (
                cfg.angles if cfg.angles else [0.0, 45.0, -45.0, 90.0] * 6
            )
        results.analytical_modulus_knockdown = (
            self._analytical_modulus_knockdown(_mod_angles, mf)
        )

        # Penetration-gate path (item D.3): when a calibrated gate is
        # configured, the analytical knockdown is the two-parameter
        # (theta, D/T) gate value instead of Budiansky-Fleck.  Uses the
        # peak angle and the penetration D/T = A / (t_ply * n_plies), both
        # on the section-2.7 conventions (config amplitude is the
        # half-amplitude).  Returns early — the BF / tension blocks below
        # are bypassed.
        if cfg.penetration_gate is not None:
            angles_g = cfg.angles if cfg.angles else [0]
            T = cfg.ply_thickness * len(angles_g)
            if cfg.wrinkles is not None:
                # Multi-wrinkle gate (issue #342): evaluate the gate per
                # spec — each wrinkle carries its own angle
                # theta_i = arctan(2*pi*A_i/lambda_i), penetration
                # D_i/T = A_i/T, and through-thickness position
                # z_i = (ply_interface + 1) / n_plies (the boundary the
                # spec nominates; ``cfg.wrinkle_z_position`` is a
                # scalar-path parameter and is ignored here) — and take
                # the weakest-link (minimum) knockdown.  Previously
                # ``dt`` silently read the leftover scalar
                # ``cfg.amplitude`` while theta came from the specs,
                # producing a plausible-looking wrong answer (e.g. kd
                # 0.98 instead of 0.64 on the issue's repro).
                n_plies_g = len(angles_g)
                kd_gate = 1.0
                for spec in cfg.wrinkles:
                    if spec.wavelength > 1e-12:
                        theta_i = float(np.arctan(
                            2.0 * np.pi * spec.amplitude / spec.wavelength
                        ))
                    else:
                        theta_i = 0.0
                    dt_i = (spec.amplitude / T) if T > 0 else 0.0
                    z_i = (spec.ply_interface + 1) / n_plies_g
                    kd_i = penetration_gate_kd(
                        math.degrees(theta_i), dt_i, cfg.penetration_gate,
                        z_position=float(z_i),
                    )
                    kd_gate = min(kd_gate, float(kd_i))
            else:
                dt = (cfg.amplitude / T) if T > 0 else 0.0
                kd_gate = penetration_gate_kd(
                    math.degrees(theta_max), dt, cfg.penetration_gate,
                    z_position=float(cfg.wrinkle_z_position),
                )
            results.morphology_factor = mf
            results.max_angle_rad = theta_max
            results.effective_angle_rad = theta_eff
            results.gamma_Y_eff = cfg.penetration_gate.gamma_Y
            results.analytical_knockdown = float(kd_gate)
            material = cfg.material
            assert material is not None  # set by AnalysisConfig.__post_init__
            ref = (material.Xt if cfg.loading == "tension"
                   else material.Xc)
            results.analytical_strength_MPa = float(ref) * float(kd_gate)
            return

        # Compute layup-dependent effective gamma_Y from confinement
        angles: list[float] = (
            cfg.angles if cfg.angles else [0.0, 45.0, -45.0, 90.0] * 6
        )
        gamma_Y_eff = _effective_gamma_Y(angles)

        # Compression KD (CLT-weighted Budiansky-Fleck) — computed for
        # both loading modes: used directly for compression, and as a
        # physical floor for tension (tension cannot be worse than compression).
        mat = cfg.material
        assert mat is not None  # filled in __post_init__
        E11 = mat.E1
        E22 = mat.E2
        G12 = mat.G12

        n_0 = sum(1 for a in angles if abs(a) < 5)
        n_45 = sum(1 for a in angles if 40 < abs(a) < 50)
        n_90 = sum(1 for a in angles if abs(a) > 85)

        Q11_0 = E11
        Q11_45 = E11 / 4.0 + E22 / 4.0 + G12 / 2.0
        Q11_90 = E22

        total_stiffness = n_0 * Q11_0 + n_45 * Q11_45 + n_90 * Q11_90
        f_0 = n_0 * Q11_0 / total_stiffness if total_stiffness > 0 else 1.0

        # For graded morphology (embedded wrinkle), use profile-proportional
        # knockdown: the BF knockdown is averaged over the wrinkle profile
        # in both x (local angle varies with dz/dx) and z (Gaussian decay).
        # For other morphologies (stack/convex/concave/uniform), the wrinkle
        # extends through the full thickness and fills the coupon, so failure
        # is governed by the peak-angle cross-section.
        #
        # KD_lam = (1/N) sum_p [ (1/L_s) int 1/(1 + theta(x)*Phi(z_p)/gY) dx ]
        # Force the non-graded peak-angle path when a multi-wrinkle
        # override is active: the graded profile-proportional helper
        # below uses cfg.amplitude / cfg.wavelength / cfg.width directly
        # and would silently ignore the per-spec geometry.
        is_graded = (
            cfg.morphology.lower().strip() == "graded"
            and cfg.wrinkles is None
        )
        n_plies = len(angles)

        # Resolve through-thickness Gaussian decay scale (mm).  The user
        # can override the auto formula via
        # ``cfg.through_thickness_decay_scale``; default is
        # ``max(wavelength / 2, amplitude)``.  Used for both the
        # compression profile-proportional KD and the tension graded-
        # averaging block.
        if cfg.through_thickness_decay_scale is not None:
            decay_scale_eff = float(cfg.through_thickness_decay_scale)
        else:
            decay_scale_eff = max(cfg.wavelength / 2.0, cfg.amplitude)
        c_AF = float(cfg.kink_band_quadratic_coeff)

        if is_graded and n_0 > 0 and n_plies > 1:
            # Profile-proportional compression knockdown (graded/embedded).
            # ``wrinkle_z_position`` shifts the through-thickness decay
            # centre off the midplane to model wrinkles closer to the
            # surface (Li et al. 2025 Dataset F: Above/Below positions).
            z_pos = float(cfg.wrinkle_z_position)
            kd_profile = _profile_proportional_kd(
                amplitude=cfg.amplitude,
                wavelength=cfg.wavelength,
                width=cfg.width,
                domain_length=cfg.domain_length,
                ply_thickness=cfg.ply_thickness,
                n_plies=n_plies,
                gamma_Y=gamma_Y_eff,
                theta_max=theta_max,
                morphology_factor=1.0,
                through_thickness_decay=True,
                z_position_fraction=z_pos,
                decay_scale=decay_scale_eff,
                decay_floor=cfg.decay_floor,
                kink_band_quadratic_coeff=c_AF,
            )
            kd_compression = f_0 * kd_profile + (1.0 - f_0)

            if cfg.loading == "tension":
                # Average tension knockdown over 0-deg plies at local angles
                # (profile-proportional, using stretched linear grading
                # on the new ``decay_scale_eff`` so the support tracks
                # the wrinkle's longitudinal extent rather than the
                # full half-thickness).  The grading centre ``p_mid`` is
                # shifted by ``wrinkle_z_position`` so the per-ply taper
                # peaks at the user-set through-thickness position
                # rather than the midplane (legacy: midplane).
                p_mid = z_pos * (n_plies - 1)
                zero_positions = [i for i, a in enumerate(angles) if abs(a) < 5]
                decay_floor = cfg.decay_floor
                # Stretched linear support: width = decay_scale / t_ply
                # plies.  Falls back to the legacy half-thickness norm if
                # the decay scale would exceed it (keeps backwards-
                # compatible behaviour for cases where the auto formula
                # is at least the legacy support).
                t_ply = cfg.ply_thickness
                p_support_decay = decay_scale_eff / max(t_ply, 1e-12)
                p_support_legacy = (n_plies - 1) / 2.0
                p_norm = max(min(p_support_decay, p_support_legacy), 1e-12)
                kd_0_sum = 0.0
                for p in zero_positions:
                    raw = max(0.0, 1.0 - abs(p - p_mid) / p_norm)
                    B_p = decay_floor + (1.0 - decay_floor) * raw
                    theta_p = theta_max * B_p
                    kd_p, _ = self._tension_knockdown_analytical(
                        theta_p, cfg, _return_kd0_only=True,
                    )
                    kd_0_sum += kd_p
                kd_0_avg = kd_0_sum / len(zero_positions)
                kd = f_0 * kd_0_avg + (1.0 - f_0)
                # Get mechanisms at peak angle for reporting
                _, mechanisms = self._tension_knockdown_analytical(theta_max, cfg)
                mechanisms["mode"] = mechanisms["mode"] + " (graded avg)"
                if kd < kd_compression:
                    kd = kd_compression
                    mechanisms["mode"] = mechanisms["mode"] + " (capped)"
                ref_strength = mat.Xt
                results.tension_mechanisms = mechanisms
            else:
                kd = kd_compression
                ref_strength = mat.Xc
                results.tension_mechanisms = None
        else:
            # Non-graded: peak-angle BF at the critical cross-section,
            # with the Argon-Fleck quadratic extension.
            r_bf = theta_eff / gamma_Y_eff
            kd_bf = 1.0 / (1.0 + r_bf + c_AF * r_bf * r_bf)
            kd_compression = f_0 * kd_bf + (1.0 - f_0)

            if cfg.loading == "tension":
                kd, mechanisms = self._tension_knockdown_analytical(theta_max, cfg)
                if kd < kd_compression:
                    kd = kd_compression
                    mechanisms["mode"] = mechanisms["mode"] + " (capped)"
                ref_strength = mat.Xt
                results.tension_mechanisms = mechanisms
            else:
                kd = kd_compression
                ref_strength = mat.Xc
                results.tension_mechanisms = None
        results.gamma_Y_eff = gamma_Y_eff

        # Damage index (for reporting; not used in knockdown computation)
        D = (_D0
             * (cfg.amplitude / _A_REF) ** 1.5
             * (1.0 + _BETA_ANGLE * max(theta_max - _THETA_CRIT, 0.0))
             * mf)
        D = min(D, 0.999)

        results.morphology_factor = mf
        results.max_angle_rad = theta_max
        results.effective_angle_rad = theta_eff
        results.damage_index = D
        results.analytical_knockdown = kd

        # Populate the delamination-onset KD from the tension mechanisms
        # dict (None for compression and for materials lacking GIc/GIIc).
        if (
            cfg.loading == "tension"
            and results.tension_mechanisms is not None
            and mat.GIc is not None
            and mat.GIIc is not None
        ):
            onset_val = results.tension_mechanisms.get("kd_onset")
            results.analytical_onset_knockdown = (
                float(onset_val) if onset_val is not None else None
            )
        else:
            results.analytical_onset_knockdown = None

        results.analytical_strength_MPa = ref_strength * kd

    # ------------------------------------------------------------------
    # Tension analytical model — three-mechanism knockdown
    # ------------------------------------------------------------------

    @staticmethod
    def _tension_knockdown_analytical(
        theta: float, cfg: AnalysisConfig,
        _return_kd0_only: bool = False,
    ) -> tuple[float, dict]:
        """Three-mechanism tension knockdown (LaRC04 + curved-beam OOP).

        Computes the laminate-level retention factor for tension loading
        by combining three competing failure mechanisms for the 0-degree
        plies, weighted by CLT axial stiffness fractions:

        1. **Fiber tension** (LaRC04 #3, Pinho Eq. 82): KD = cos²θ
        2. **Matrix tension** (LaRC04 #1, Pinho Eq. 40): Hashin σ₂₂/τ₁₂
           interaction with in-situ strengths (Yt_is, S12_is)
        3. **Out-of-plane delamination** (Timoshenko curved-beam):
           Combined σ₃₃ (mode I at crest) and τ₁₃ (mode II at inflection)

        The 0-degree ply knockdown is min(KD_fiber, KD_matrix, KD_oop).
        Off-axis plies (±45, 90) are assumed unaffected by the waviness
        for tension loading. The laminate knockdown is the CLT-weighted
        average: KD_lam = f_0 × KD_0 + (1 − f_0) × 1.0.

        References
        ----------
        - Pinho et al. (2005) NASA-TM-2005-213530, Eq. 40, 47, 57, 82
        - Timoshenko & Gere (1961), Theory of Elastic Stability (curved beam)
        - Elhajjar (2025) Scientific Reports 15:25977 (experimental data)
        """
        # Filled in __post_init__.  (The previous ``if mat is None:
        # return 1.0`` guard returned a bare float from a function whose
        # callers always unpack a (kd, mechanisms) tuple, so it could
        # never have worked anyway.)
        mat = cfg.material
        assert mat is not None
        angles: list[float] = (
            cfg.angles if cfg.angles else [0.0, 45.0, -45.0, 90.0] * 6
        )

        E11 = mat.E1
        E22 = mat.E2
        G12 = mat.G12
        Xt = mat.Xt
        Yt = mat.Yt if mat.Yt else 49.0
        S12 = mat.S12 if mat.S12 else 85.0
        S13 = mat.S13 if hasattr(mat, "S13") and mat.S13 else S12

        # --- Mechanism 1: Fiber tension  cos²θ ---
        kd_fiber = float(np.cos(theta) ** 2)

        # --- Mechanism 2: Matrix tension (Hashin with in-situ strengths) ---
        # In-situ transverse: Yt_is = 1.12·√2·Yt (Pinho Eq. 47, thin ply)
        # In-situ shear: thick-ply correction from Camanho (2006)
        #   S12_is = sqrt(8·GIIc / (pi·t_eff·Lambda_22))
        #   For typical carbon/epoxy (GIIc ≈ 1.0 N/mm), this gives
        #   S12_is ≈ 2.3·S12 for n_adj=2 adjacent 0-deg plies.
        #   Falls back to sqrt(2)·S12 for single-ply (thin) case.
        n_adj = _max_consecutive_zero_plies(angles)
        t_eff = n_adj * cfg.ply_thickness
        Yt_is = 1.12 * np.sqrt(2.0) * Yt
        # Thick-ply in-situ shear: GIIc ≈ 1.0 N/mm for carbon/epoxy
        _GIIc_typical = 1.0  # N/mm
        _Lambda22 = 2.0 * (1.0 / E22 - (mat.nu12 ** 2) / E11)
        if _Lambda22 > 0 and t_eff > 0:
            S12_is = np.sqrt(8.0 * _GIIc_typical / (np.pi * t_eff * _Lambda22))
        else:
            S12_is = np.sqrt(2.0) * S12

        if theta > 1e-10:
            sin_t = np.sin(theta)
            cos_t = np.cos(theta)
            term1 = (sin_t ** 2 / Yt_is) ** 2
            term2 = (sin_t * cos_t / S12_is) ** 2
            sigma_fail = 1.0 / np.sqrt(term1 + term2)
            kd_matrix = min(sigma_fail / Xt, 1.0)
        else:
            kd_matrix = 1.0

        # --- Mechanism 3: Out-of-plane delamination (curved-beam) ---
        amplitude = cfg.amplitude
        wavelength = cfg.wavelength

        # Effective thickness: max consecutive 0-degree plies. With no
        # continuous 0-degree block (n_adj == 0) the curved-beam model
        # has no fibrous load path to develop interlaminar σ₃₃ / τ₁₃,
        # so the OOP mechanism is inactive (kd_oop = 1.0).
        n_adj_oop = _max_consecutive_zero_plies(angles)

        # Interlaminar stresses at the 0-block interface (held for both
        # the stress-based OOP mechanism above AND the new energy-based
        # onset criterion below).  At ``λ = 1`` (applied stress = Xt)
        # these are the σ₃₃ at the crest and τ₁₃ at the inflection.
        sigma33 = 0.0
        tau13 = 0.0
        h_eff_oop = 0.0
        if n_adj_oop == 0 or amplitude <= 1e-12 or wavelength <= 1e-12:
            kd_oop = 1.0
        else:
            # Peak curvature at crest: κ = (2π/λ)² A
            kappa_max = (2.0 * np.pi / wavelength) ** 2 * amplitude
            # Max curvature gradient at inflection: |dκ/dx| = (2π/λ)³ A
            dkappa_dx_max = (2.0 * np.pi / wavelength) ** 3 * amplitude

            h_eff_oop = n_adj_oop * cfg.ply_thickness

            # σ₃₃ at crest (mode I) and τ₁₃ at inflection (mode II)
            sigma33 = Xt * h_eff_oop * kappa_max
            tau13 = Xt * h_eff_oop * dkappa_dx_max

            # Failure indices (peak at different spatial locations)
            FI_s33 = (sigma33 / Yt) ** 2
            FI_t13 = (tau13 / S13) ** 2
            FI_max = max(FI_s33, FI_t13)

            kd_oop = 1.0 / np.sqrt(1.0 + FI_max)

        # 0-degree ply knockdown: minimum of all three mechanisms
        kd_0 = min(kd_fiber, kd_matrix, kd_oop)

        # ----------------------------------------------------------------
        # Delamination-onset KD (Mukhopadhyay et al. 2015 first-load-drop)
        # ----------------------------------------------------------------
        # The three-mechanism kd_0 above is the *ultimate* fibre-failure
        # KD.  Embedded-wrinkle tests (Mukhopadhyay 2015) also exhibit an
        # earlier *first-load-drop* corresponding to delamination
        # initiation at the curved 0-block interface.  We predict that
        # initiation knockdown with a Benzeggagh-Kenane mode-mixity
        # criterion driven by the same σ₃₃ / τ₁₃ already computed for
        # the OOP mechanism, but compared to GIc / GIIc rather than to
        # the strength allowables Yt / S13.
        #
        # Derivation (notation: λ = applied stress / Xt):
        #   σ₃₃(λ) = λ · σ₃₃   (above, evaluated at λ = 1)
        #   τ₁₃(λ) = λ · τ₁₃
        # Energy release rate at a notional interfacial flaw of size a:
        #   G_I  = σ₃₃² · π · a / (2 · E_3)
        #   G_II = τ₁₃² · π · a / (2 · G_13)
        # Both scale as λ².  B-K-like mode-mixity initiation:
        #   λ²·(G_I/GIc) + (λ²·G_II/GIIc)^η = 1,   η = 1.45
        # Solved for λ_onset ∈ (0, 1].  If the criterion is satisfied
        # already at λ < 1, the onset KD is below the ultimate KD.
        #
        # Flaw size: a = t_eff (the 0-block thickness ``n_adj * t_ply``).
        # The spec proposed a = t_ply but with that single-ply scale the
        # criterion gives λ_onset > 1 for all the Mukhopadhyay cases —
        # i.e. no onset before fibre fracture, which is unphysical.  An
        # embedded delamination at the 0-block interface naturally spans
        # the block thickness, so a = t_eff is the correct local scale.
        kd_onset = None
        if (
            mat.GIc is not None
            and mat.GIIc is not None
            and n_adj_oop > 0
            and amplitude > 1e-12
            and wavelength > 1e-12
        ):
            E3 = mat.E3
            G13 = mat.G13
            GIc = mat.GIc
            GIIc = mat.GIIc
            a_flaw = h_eff_oop  # = t_eff = n_adj_oop * ply_thickness

            G_I_unit = (sigma33 ** 2) * np.pi * a_flaw / (2.0 * E3)
            G_II_unit = (tau13 ** 2) * np.pi * a_flaw / (2.0 * G13)

            R_I = G_I_unit / GIc
            R_II = G_II_unit / GIIc
            eta = 1.45

            # f(λ) = λ²·R_I + (λ²·R_II)^η  -  1.  Monotonically
            # increasing in λ ∈ (0, ∞).
            def _bk_criterion(lam: float) -> float:
                lam2 = lam * lam
                term_I = lam2 * R_I
                arg_II = lam2 * R_II
                term_II = arg_II ** eta if arg_II > 0.0 else 0.0
                return term_I + term_II - 1.0

            f_at_1 = _bk_criterion(1.0)
            if f_at_1 <= 0.0:
                # Energy criterion not met even at λ = 1.  No separate
                # initiation event before fibre fracture; report onset
                # equal to the ultimate.
                lam_onset = 1.0
            else:
                # Bisect in (1e-4, 1] for the unique root.
                lo, hi = 1.0e-4, 1.0
                f_lo = _bk_criterion(lo)
                if f_lo > 0.0:
                    # Even at vanishing load the criterion is satisfied,
                    # which would imply zero strength — guard against it.
                    lam_onset = lo
                else:
                    for _ in range(80):
                        mid = 0.5 * (lo + hi)
                        f_mid = _bk_criterion(mid)
                        if f_mid > 0.0:
                            hi = mid
                        else:
                            lo = mid
                        if hi - lo < 1.0e-6:
                            break
                    lam_onset = 0.5 * (lo + hi)

            kd_onset = float(lam_onset)

        # For graded averaging: return just the 0-deg ply KD without CLT
        if _return_kd0_only:
            return float(kd_0), {}

        # CLT-weighted laminate knockdown
        n_0 = sum(1 for a in angles if abs(a) < 5)
        n_45 = sum(1 for a in angles if 40 < abs(a) < 50)
        n_90 = sum(1 for a in angles if abs(a) > 85)

        Q11_0 = E11
        Q11_45 = E11 / 4.0 + E22 / 4.0 + G12 / 2.0
        Q11_90 = E22

        total_stiffness = n_0 * Q11_0 + n_45 * Q11_45 + n_90 * Q11_90
        if total_stiffness > 0:
            f_0 = n_0 * Q11_0 / total_stiffness
        else:
            f_0 = 1.0

        kd_lam = f_0 * kd_0 + (1.0 - f_0) * 1.0

        # CLT-weight the 0-ply onset KD to the laminate level, matching
        # the kd_0 \u2192 kd_lam pattern.  Then ensure the onset KD is
        # strictly less than the ultimate KD: any local interfacial
        # delamination event must precede (or coincide with) the
        # laminate ultimate, so we cap onset at ``kd_lam * 0.999``
        # whenever the raw energy-based onset is not already below it.
        # This guarantees the spec's requirement that onset KD < KD_oop
        # and onset KD < analytical_knockdown.
        kd_onset_lam: float | None = None
        if kd_onset is not None:
            kd_onset_lam_raw = f_0 * kd_onset + (1.0 - f_0) * 1.0
            kd_onset_lam = min(kd_onset_lam_raw, float(kd_lam) * 0.999)

        # Determine controlling mode
        if kd_oop <= kd_fiber and kd_oop <= kd_matrix:
            mode = "OOP \u03c3\u2083\u2083"
        elif kd_matrix <= kd_fiber:
            mode = "matrix"
        else:
            mode = "fiber"

        mechanisms = {
            "kd_fiber": kd_fiber,
            "kd_matrix": kd_matrix,
            "kd_oop": kd_oop,
            "kd_0": kd_0,
            "kd_lam": float(kd_lam),
            "kd_onset": kd_onset_lam,
            "f_0": f_0,
            "mode": mode,
        }

        return float(kd_lam), mechanisms

    def _build_flat_mesh(self, laminate: Laminate) -> MeshData:
        """Generate a pristine (zero-amplitude) mesh matching the config.

        Shared by the retention-factor baseline and the progressive-damage
        pristine reference so both compare against the same flat laminate.
        """
        cfg = self.config
        flat_profile = GaussianSinusoidal(
            amplitude=0.0,
            wavelength=cfg.wavelength,
            width=cfg.width,
            center=cfg.domain_length / 2.0,
        )
        # Both interface indices are populated by AnalysisConfig.__post_init__.
        assert cfg.interface_1 is not None and cfg.interface_2 is not None
        flat_config = WrinkleConfiguration.from_morphology_name(
            "stack", flat_profile,
            interface1=cfg.interface_1,
            interface2=cfg.interface_2,
        )
        return WrinkleMesh(
            laminate=laminate,
            wrinkle_config=flat_config,
            Lx=cfg.domain_length,
            Ly=cfg.domain_width,
            nx=cfg.nx,
            ny=cfg.ny,
            nz_per_ply=cfg.nz_per_ply,
        ).generate()

    def _run_progressive_path(
        self, results: AnalysisResults, laminate: Laminate, mesh: MeshData
    ) -> None:
        """Run the load-stepping progressive-damage solver to ultimate load.

        Solves the wrinkled mesh (with any resin pocket already attached)
        and a pristine flat baseline, then records the ultimate strengths
        and their ratio as the progressive-damage knockdown.  The wrinkled
        mesh's per-element material override is cleared afterwards so the
        subsequent linear field/failure/retention pass is unaffected.
        """
        from wrinklefe.solver.progressive_damage import (
            ProgressiveDamageResult,
            ProgressiveDamageSolver,
        )

        cfg = self.config
        # Auto-size the strain ramp to bracket fibre failure (~1.8x the
        # compressive failure strain Xc / E1 of the 0-degree material).
        if cfg.progressive_max_strain is not None:
            target = float(cfg.progressive_max_strain)
        else:
            mat0 = laminate.plies[0].material
            eps_f = mat0.Xc / mat0.E1
            target = 1.8 * eps_f
        sign = -1.0 if cfg.applied_strain <= 0 else 1.0
        applied = sign * abs(target)

        def _run(m: MeshData) -> ProgressiveDamageResult:
            return ProgressiveDamageSolver(
                m, laminate,
                applied_strain=applied,
                n_increments=cfg.progressive_n_increments,
                residual_factor=cfg.progressive_residual_factor,
                solver=cfg.solver,
                verbose=cfg.verbose,
            ).solve()

        # Wrinkled run — snapshot/restore the override so the later linear
        # pass sees the undamaged mesh.
        saved_override = mesh.element_material_override
        wr = _run(mesh)
        mesh.element_material_override = saved_override

        pristine = _run(self._build_flat_mesh(laminate))

        results.progressive_strength_MPa = wr.peak_stress
        results.progressive_pristine_strength_MPa = pristine.peak_stress
        results.progressive_history = wr.history
        if pristine.peak_stress > 0:
            results.progressive_knockdown = (
                wr.peak_stress / pristine.peak_stress
            )

    def _attach_resin_pocket(
        self, mesh: MeshData, laminate: Laminate
    ) -> None:
        """Tag the resin-lens elements and attach the resin material.

        Builds a :class:`~wrinklefe.core.resin_pocket.ResinPocketSpec`
        from the wrinkle geometry and the configured scale knobs, flags
        the hex elements whose centroids fall inside the lens, and stores
        the boolean mask plus the resin material on *mesh* so the
        assembler, stress-recovery and failure paths pick them up.

        The resin material defaults to the built-in ``EPOXY_S6C10``
        isotropic card when ``resin_pocket_material`` is unset.
        """
        from wrinklefe.core.resin_pocket import (
            ResinPocketSpec,
            compute_resin_blend,
            compute_resin_mask,
        )

        cfg = self.config
        # Place the pocket relative to the mesh's ACTUAL z-extent, which
        # is centred on the mid-plane (z in [-T/2, +T/2]) — not the
        # bottom-referenced [0, T].  ``wrinkle_z_position`` maps 0 -> bottom
        # surface, 0.5 -> mid-plane (where the graded morphology places the
        # wrinkle crest), 1 -> top surface.  Using ``z_frac * T`` (the old
        # form) put a mid-plane pocket at the top surface, mis-locating it
        # away from the high-angle crest entirely.
        z_lo = float(mesh.nodes[:, 2].min())
        z_hi = float(mesh.nodes[:, 2].max())
        z_center = z_lo + float(cfg.wrinkle_z_position) * (z_hi - z_lo)
        center_x = cfg.domain_length / 2.0

        spec = ResinPocketSpec.from_wrinkle(
            amplitude=cfg.amplitude,
            wavelength=cfg.wavelength,
            center_x=center_x,
            z_center=z_center,
            height_scale=cfg.resin_pocket_height_scale,
            length_scale=cfg.resin_pocket_length_scale,
        )

        resin_material = cfg.resin_pocket_material
        if resin_material is None:
            resin_material = MaterialLibrary().get("EPOXY_S6C10")
        mesh.resin_material = resin_material

        if cfg.resin_pocket_graded:
            # Graded pocket: per-element blend weight + precomputed blended
            # materials (host ply <-> resin), and the fibre angle scaled by
            # (1 - weight) downstream via ``resin_angle_scale``.
            weight = compute_resin_blend(mesh, spec)
            mesh.resin_blend = weight
            blend_mats: dict[int, OrthotropicMaterial] = {}
            for e_np in np.flatnonzero(weight > 0.0):
                e = int(e_np)
                ply_mat = laminate.plies[int(mesh.ply_ids[e])].material
                blend_mats[e] = ply_mat.blend(
                    resin_material, float(weight[e])
                )
            mesh.resin_blend_materials = blend_mats
            n_resin = int((weight > 0.0).sum())
        else:
            mesh.resin_mask = compute_resin_mask(mesh, spec)
            n_resin = int(mesh.resin_mask.sum())

        if cfg.verbose:
            import logging
            logging.getLogger(__name__).info(
                "Resin pocket (%s): %d/%d elements tagged "
                "(half_length=%.3g mm, h_center=%.3g mm, z_center=%.3g mm)",
                "graded" if cfg.resin_pocket_graded else "binary",
                n_resin, mesh.n_elements, spec.half_length,
                spec.h_center, z_center,
            )

    def _evaluate_failure(
        self,
        results: AnalysisResults,
        laminate: Laminate,
        field_results: FieldResults,
        mesh: MeshData,
    ) -> None:
        """Evaluate failure criteria on the FE stress field."""
        evaluator = FailureEvaluator.default_criteria()

        # Build material list for each ply
        materials = [ply.material for ply in laminate.plies]

        # Per-element fiber angles from wrinkle geometry (for LaRC05 kinking)
        elem_fiber_angles = mesh.element_fiber_angles_array()

        # Resin-pocket zone (Li et al. 2024/2025): route lens elements to
        # their pocket material (graded blend, or the binary resin card)
        # so failure is evaluated at the locally-softened strengths, and
        # scale the fibre angle by the retention factor so the LaRC05
        # kink-band path is not double-counted at the resin centre.
        eval_ply_ids = np.asarray(mesh.ply_ids)
        if mesh.resin_blend_materials:
            # Graded pocket: each blended element gets its own material.
            extra = list(mesh.resin_blend_materials.items())
            base = len(materials)
            mat_index = {e: base + i for i, (e, _m) in enumerate(extra)}
            materials = [*materials, *(m for _e, m in extra)]
            eval_ply_ids = eval_ply_ids.copy()
            for e, idx in mat_index.items():
                eval_ply_ids[e] = idx
            if mesh.resin_blend is not None:
                elem_fiber_angles = (
                    elem_fiber_angles * (1.0 - mesh.resin_blend)
                )
        elif mesh.resin_mask is not None and mesh.resin_material is not None:
            resin_idx = len(materials)
            materials = [*materials, mesh.resin_material]
            eval_ply_ids = np.where(
                mesh.resin_mask, resin_idx, mesh.ply_ids
            )
            elem_fiber_angles = np.where(
                mesh.resin_mask, 0.0, elem_fiber_angles
            )

        # Field-level evaluation
        fi_fields, mode_fields = evaluator.evaluate_field(
            field_results.stress_local,
            materials,
            eval_ply_ids,
            fiber_angles=elem_fiber_angles,
        )
        results.failure_indices = fi_fields
        results.failure_modes = mode_fields

        # CLT-level evaluation at applied load
        load = LoadState(Nx=self.config.applied_strain * 1000.0)  # approximate
        try:
            report = evaluator.evaluate_laminate(laminate, load)
            results.failure_report = report
        except Exception as exc:
            logger.warning("CLT evaluation skipped: %s", exc)

    def _compute_retention_factors(
        self,
        results: AnalysisResults,
        laminate: Laminate,
    ) -> None:
        """Compute retention factors by running a pristine (no-wrinkle) baseline.

        Retention = max_FI_pristine / max_FI_wrinkled

        A retention of 1.0 means no knockdown; 0.5 means 50% strength retained.
        """
        cfg = self.config
        # interface_1 / interface_2 are filled in __post_init__.
        assert cfg.interface_1 is not None and cfg.interface_2 is not None

        if results.failure_indices is None:
            return

        # Build a flat (no wrinkle) mesh with same dimensions
        flat_mesh = self._build_flat_mesh(laminate)

        # Solve with same BCs
        flat_solver = StaticSolver(flat_mesh, laminate)
        flat_bcs = BoundaryHandler.compression_bcs(
            flat_mesh, applied_strain=cfg.applied_strain
        )
        flat_field = flat_solver.solve(flat_bcs, solver=cfg.solver, verbose=False)

        # Evaluate failure on flat mesh (no fiber misalignment)
        evaluator = FailureEvaluator.default_criteria()
        materials = [ply.material for ply in laminate.plies]

        flat_fi_fields, _ = evaluator.evaluate_field(
            flat_field.stress_local,
            materials,
            flat_mesh.ply_ids,
            fiber_angles=None,  # no misalignment in pristine laminate
        )

        # Compute retention for each criterion
        retention = {}
        baseline = {}

        for crit_name in results.failure_indices:
            # Wrinkled max FI (interior elements)
            fi_w = results.failure_indices[crit_name]
            fi_w_mean = fi_w.mean(axis=-1)  # avg over Gauss pts
            finite_w = fi_w_mean[np.isfinite(fi_w_mean)]
            max_fi_w = float(finite_w.max()) if finite_w.size > 0 else 0.0

            # Pristine max FI
            fi_p = flat_fi_fields[crit_name]
            fi_p_mean = fi_p.mean(axis=-1)
            finite_p = fi_p_mean[np.isfinite(fi_p_mean)]
            max_fi_p = float(finite_p.max()) if finite_p.size > 0 else 0.0

            baseline[crit_name] = max_fi_p

            if max_fi_w > 0:
                # Retention = pristine_FI / wrinkled_FI
                # (how much of the pristine strength is retained)
                retention[crit_name] = min(max_fi_p / max_fi_w, 1.0)
            else:
                retention[crit_name] = 1.0

        results.retention_factors = retention
        results.baseline_fi = baseline

        # --- Modulus retention from FE ---
        # Two complementary estimators of the axial-modulus knockdown
        # ``E_x / E_x0`` (wrinkled vs pristine), both populated here:
        #
        #   modulus_retention        — LOCAL σ₁₁ proxy: E_eff = <σ₁₁> /
        #     ε_applied, the mean element-frame fibre-direction stress over
        #     the coupon divided by the applied strain.  A local proxy that
        #     over-predicts the retention (issue #328).
        #
        #   modulus_retention_global — GLOBAL reaction response: E_eff =
        #     σ_nominal / ε_applied with σ_nominal = R / A, the total axial
        #     reaction on the loaded (x_max) face over the cross-section
        #     area Ly·Lz.  A true coupon-level stiffness that captures load
        #     redistribution around the wrinkle, so it tracks the measured
        #     modulus knockdown more closely (and is lower than the local
        #     proxy for a wrinkled coupon).
        try:
            applied_strain = cfg.applied_strain
            if applied_strain == 0.0:
                results.modulus_retention = 1.0
            else:
                # Set by the FE path before retention factors run; if it
                # were ever None the except below restores the default.
                assert results.field_results is not None
                stress_w = results.field_results.stress_local  # (n_elem, n_gauss, 6)
                stress_p = flat_field.stress_local

                # Mean fiber-direction stress σ₁₁ (Voigt component 0)
                s11_w = stress_w[:, :, 0].mean()
                s11_p = stress_p[:, :, 0].mean()

                E_wrinkled = s11_w / applied_strain
                E_pristine = s11_p / applied_strain

                if abs(E_pristine) > 1e-6:
                    results.modulus_retention = float(abs(E_wrinkled) / abs(E_pristine))
                else:
                    results.modulus_retention = 1.0
        except Exception:
            results.modulus_retention = 1.0

        # --- Global (reaction-based) modulus retention ---
        try:
            applied_strain = cfg.applied_strain
            wrinkled_mesh = results.mesh
            if applied_strain == 0.0 or wrinkled_mesh is None:
                results.modulus_retention_global = 1.0
            else:
                E_w_global = self._reaction_modulus(
                    wrinkled_mesh, laminate, applied_strain
                )
                E_p_global = self._reaction_modulus(
                    flat_mesh, laminate, applied_strain
                )
                if E_w_global is not None and E_p_global is not None and (
                    abs(E_p_global) > 1e-12
                ):
                    results.modulus_retention_global = float(
                        abs(E_w_global) / abs(E_p_global)
                    )
                else:
                    results.modulus_retention_global = 1.0
        except Exception:
            logger.debug(
                "Global reaction-based modulus retention failed; "
                "falling back to 1.0", exc_info=True
            )
            results.modulus_retention_global = 1.0

    def _reaction_modulus(
        self,
        mesh: MeshData,
        laminate: Laminate,
        applied_strain: float,
    ) -> float | None:
        """Coupon-level axial modulus from the global reaction force.

        Solves the compression problem on ``mesh`` (retaining the
        unmodified stiffness ``K``) and returns the effective axial modulus
        ``E_eff = σ_nominal / ε_applied`` where the nominal stress is the
        total axial reaction on the loaded ``x_max`` face divided by the
        cross-section area ``Ly·Lz``.

        Reuses the exact reaction-extraction pattern of the
        progressive-damage solver: ``reaction = sum((K @ u)[xmax_dofs])``
        over the loaded-face x-DOFs (``3 * nodes_on_face("x_max")``), so the
        two agree.  Returns ``None`` if the reaction/area/strain cannot give
        a finite modulus.
        """
        if applied_strain == 0.0:
            return None

        solver = StaticSolver(mesh, laminate)
        bcs = BoundaryHandler.compression_bcs(
            mesh, applied_strain=applied_strain
        )
        field = solver.solve(
            bcs, solver=self.config.solver, verbose=False,
            keep_stiffness=True,
        )

        K = solver._K
        if K is None:
            return None

        xmax_nodes = mesh.nodes_on_face("x_max")
        xmax_dofs = 3 * xmax_nodes  # ux DOFs on the loaded face
        _Lx, Ly, Lz = mesh.domain_size
        area = Ly * Lz
        if area <= 0.0:
            return None

        u = field.displacement.ravel()
        reaction = float(np.sum((K @ u)[xmax_dofs]))
        sigma_nominal = reaction / area
        E_eff = sigma_nominal / applied_strain
        if not np.isfinite(E_eff):
            return None
        return float(E_eff)

