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

import math
from dataclasses import dataclass, field, fields, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from wrinklefe.core.material import OrthotropicMaterial, MaterialLibrary
from wrinklefe.core.laminate import Laminate, LoadState
from wrinklefe.core.wrinkle import (
    GaussianSinusoidal,
    WrinkleProfile,
)
from wrinklefe.core.morphology import (
    WrinkleConfiguration,
    WrinklePlacement,
    MORPHOLOGY_PHASES,
    SINGLE_WRINKLE_MODES,
)
from wrinklefe.core.mesh import WrinkleMesh, MeshData
from wrinklefe.solver.static import StaticSolver
from wrinklefe.solver.results import FieldResults
from wrinklefe.solver.boundary import BoundaryCondition, BoundaryHandler
from wrinklefe.failure.evaluator import FailureEvaluator, LaminateFailureReport
# Analytical damage model constants (Section 6 of CLAUDE.md)
_D0 = 0.15       # Base damage coefficient
_BETA_ANGLE = 3.0  # Angle sensitivity
_THETA_CRIT = 0.1  # Critical angle (rad)
_A_REF = 0.183    # Reference amplitude (1 ply thickness, mm)

# Number of x-integration points for profile-proportional knockdown
_N_PROFILE_PTS = 500

# Confinement model constants
# Calibrated with CLT-weighted BF against Elhajjar (2025), T700/2510.
_GAMMA_Y_UD = 0.032   # UD matrix yield strain (no confinement)
_ALPHA_CONF = 0.050   # confinement boost coefficient


def _confined_fraction(angles: List[float], tol: float = 5.0) -> float:
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


def _effective_gamma_Y(angles: List[float]) -> float:
    """Compute layup-dependent effective matrix yield strain.

    gamma_Y_eff = gamma_Y_UD + alpha * f_confined

    where f_confined is the weighted confinement fraction of 0-degree
    plies (0 = unconfined UD, 1 = fully interspersed). This captures
    the constraint that off-axis plies impose on kink-band lateral
    expansion in multidirectional laminates.

    With CLT-weighted compression (KD_lam = f0*KD_BF + 1-f0),
    the confinement effect is separated from load redistribution.
    Calibration points:
        UD (f=0.0):            gamma_Y = 0.032
        Mukhopadhyay (f≈0.42): gamma_Y = 0.053
        Elhajjar (f≈0.83):    gamma_Y = 0.074
    """
    fc = _confined_fraction(angles)
    return _GAMMA_Y_UD + _ALPHA_CONF * fc


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
        Phi(z_p)   = exp(-(z_p - z_c)^2 / A^2)   (through-thickness decay)
        z_c        = z_position_fraction * T (laminate thickness)

    When *through_thickness_decay* is False, Phi(z_p) = 1 for all plies
    (all plies see the same longitudinal profile).  This is appropriate
    for dual-wrinkle morphologies (stack/convex/concave) where the wrinkle
    extends through the full thickness.

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
        at ``z_position_fraction * T`` with scale = amplitude.  If False,
        all plies see the full wrinkle angle profile (Phi = 1).
    z_position_fraction : float
        Fraction of the laminate thickness at which the wrinkle through-
        thickness decay is centred.  ``0.5`` (default) centres the decay
        at the midplane, reproducing the legacy behaviour; ``0.0`` and
        ``1.0`` place the decay centre at the bottom and top surfaces,
        respectively.  Only consulted when ``through_thickness_decay`` is
        True.

    Returns
    -------
    float
        Profile-averaged BF knockdown factor (0, 1].
    """
    T = n_plies * ply_thickness
    z_center = z_position_fraction * T
    L_s = domain_length

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
            phi_p = np.exp(-((z_p - z_center) ** 2) / (amplitude ** 2))
        else:
            phi_p = 1.0
        theta_xz = theta_x * phi_p  # local angle at (x, z_p)
        kd_xz = 1.0 / (1.0 + theta_xz / gamma_Y)
        kd_sum += np.mean(kd_xz)

    return kd_sum / n_plies


def _max_consecutive_zero_plies(angles: List[float], tol: float = 5.0) -> int:
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
# Configuration
# ======================================================================

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
    phase: Optional[float] = None
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
    amplitude_profile_decay_length: Optional[float] = None
    amplitude_profile_axis: str = "x"

    # Loading
    loading: str = "compression"

    # Material & laminate
    material: Optional[OrthotropicMaterial] = None
    angles: Optional[List[float]] = None

    # Ply thickness
    ply_thickness: float = 0.183  # mm (1 ply thickness for CYCOM X850/T800)

    # Wrinkle placement. ``None`` triggers auto-derivation in
    # ``__post_init__`` from ``len(angles)`` so small laminates work out
    # of the box (issues #154/#156). For the default 24-ply layup the
    # auto-derived pair is (11, 12), preserving backwards compatibility.
    interface_1: Optional[int] = None
    interface_2: Optional[int] = None

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

    def __post_init__(self) -> None:
        if self.domain_length <= 0:
            self.domain_length = 3.0 * self.wavelength
        if self.material is None:
            self.material = MaterialLibrary().get("IM7_8552")
        if self.angles is None:
            # Quasi-isotropic [0/45/-45/90]_3s → 24 plies
            base = [0, 45, -45, 90]
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
        Analytical combined knockdown factor.
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
    mesh: Optional[MeshData] = None
    wrinkle_config: Optional[WrinkleConfiguration] = None
    laminate: Optional[Laminate] = None

    # Analytical predictions
    morphology_factor: float = 1.0
    max_angle_rad: float = 0.0
    effective_angle_rad: float = 0.0
    mesh_max_angle_rad: float = 0.0  # max fiber angle from FE mesh (accounts for decay)
    damage_index: float = 0.0
    analytical_knockdown: float = 1.0
    analytical_strength_MPa: float = 0.0
    gamma_Y_eff: float = 0.02  # layup-dependent effective yield strain
    tension_mechanisms: Optional[dict] = None  # {kd_fiber, kd_matrix, kd_oop, mode, ...}

    # FE results
    field_results: Optional[FieldResults] = None
    failure_report: Optional[LaminateFailureReport] = None
    failure_indices: Optional[dict] = None
    failure_modes: Optional[dict] = None  # {criterion: (n_elem, n_gauss) str array}

    # Retention factor (wrinkled / pristine)
    retention_factors: Optional[dict] = None  # {criterion_name: float}
    baseline_fi: Optional[dict] = None  # {criterion_name: float} pristine max FI

    # Modulus retention (E_wrinkled / E_pristine from FE)
    modulus_retention: float = 1.0

    # Tension mechanism decomposition (only for tension loading)
    tension_mechanisms: Optional[dict] = None  # {kd_fiber, kd_matrix, kd_oop, ...}

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
                f"    Modulus retention: {self.modulus_retention:.4f}",
            ])

        lines.append("=" * 65)
        return "\n".join(lines)


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
        analytical_only: Optional[bool] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
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
        if analytical_only is None:
            analytical_only = cfg.analytical_only
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

        _report("Building laminate", 0.0)

        # 1. Build laminate
        laminate = self._build_laminate()
        results.laminate = laminate

        # 2. Create wrinkle configuration (centered in specimen)
        wrinkle_center = cfg.domain_length / 2.0
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
                amplitude_profile=cfg.amplitude_profile,
                amplitude_profile_decay_length=cfg.amplitude_profile_decay_length,
                amplitude_profile_axis=cfg.amplitude_profile_axis,
            )
            wrinkle_config.decay_floor = max(0.0, min(1.0, cfg.decay_floor))
        else:
            wrinkle_config = WrinkleConfiguration.from_morphology_name(
                cfg.morphology, profile,
                interface1=cfg.interface_1,
                interface2=cfg.interface_2,
                decay_floor=cfg.decay_floor,
                amplitude_profile=cfg.amplitude_profile,
                amplitude_profile_decay_length=cfg.amplitude_profile_decay_length,
                amplitude_profile_axis=cfg.amplitude_profile_axis,
            )
        results.wrinkle_config = wrinkle_config

        _report("Computing analytical predictions", 0.05)

        # 3. Analytical predictions
        self._compute_analytical(results, wrinkle_config)

        # In analytical-only mode, skip mesh, solve, failure, and retention.
        if analytical_only:
            _report("Analytical predictions complete", 1.0)
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
        results.mesh = mesh

        # 4b. Mesh-based max fiber angle (accounts for decay mode)
        results.mesh_max_angle_rad = float(np.max(mesh.fiber_angles)) if mesh.fiber_angles.size > 0 else 0.0

        _report("Solving FE system", 0.25)

        # 5. Static FE solve
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

        return results

    # ------------------------------------------------------------------
    # Morphology comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare_morphologies(
        base_config: AnalysisConfig,
        morphologies: Sequence[str] = ("stack", "convex", "concave"),
        analytical_only: bool = False,
    ) -> Dict[str, AnalysisResults]:
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
        all_results: Dict[str, AnalysisResults] = {}

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
    ) -> List[AnalysisResults]:
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

        results_list: List[AnalysisResults] = []

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

        for val in values:
            overrides = {parameter: val}
            if reset_domain_length:
                overrides["domain_length"] = 0.0
            cfg = replace(base_config, **overrides)

            results_list.append(
                WrinkleAnalysis(cfg).run(analytical_only=analytical_only)
            )

        return results_list

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_laminate(self) -> Laminate:
        """Build the Laminate from config."""
        return Laminate.from_angles(
            self.config.angles,
            self.config.material,
            ply_thickness=self.config.ply_thickness,
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
        if cfg.wavelength > 1e-12:
            theta_max = float(np.arctan(2.0 * np.pi * cfg.amplitude / cfg.wavelength))
        else:
            theta_max = 0.0
        theta_eff = theta_max * mf

        # Compute layup-dependent effective gamma_Y from confinement
        angles = cfg.angles if cfg.angles else [0, 45, -45, 90] * 6
        gamma_Y_eff = _effective_gamma_Y(angles)

        # Compression KD (CLT-weighted Budiansky-Fleck) — computed for
        # both loading modes: used directly for compression, and as a
        # physical floor for tension (tension cannot be worse than compression).
        mat = cfg.material
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
        is_graded = cfg.morphology.lower().strip() == "graded"
        n_plies = len(angles)

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
            )
            kd_compression = f_0 * kd_profile + (1.0 - f_0)

            if cfg.loading == "tension":
                # Average tension knockdown over 0-deg plies at local angles
                # (profile-proportional, using linear grading as fallback
                # for the three-mechanism model).  The grading centre
                # ``p_mid`` is shifted by ``wrinkle_z_position`` so the
                # per-ply taper peaks at the user-set through-thickness
                # position rather than the midplane (legacy: midplane).
                p_mid = z_pos * (n_plies - 1)
                zero_positions = [i for i, a in enumerate(angles) if abs(a) < 5]
                decay_floor = cfg.decay_floor
                # Legacy normaliser ((n_plies - 1) / 2) for backwards-
                # compatible behaviour at z_pos = 0.5; the max() guards
                # the off-midplane case so we don't divide by a smaller
                # number when ``p_mid`` is closer to a surface.
                p_norm = max((n_plies - 1) / 2.0, 1e-12)
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
                ref_strength = cfg.material.Xt
                results.tension_mechanisms = mechanisms
            else:
                kd = kd_compression
                ref_strength = cfg.material.Xc
                results.tension_mechanisms = None
        else:
            # Non-graded: peak-angle BF at the critical cross-section
            kd_bf = 1.0 / (1.0 + theta_eff / gamma_Y_eff)
            kd_compression = f_0 * kd_bf + (1.0 - f_0)

            if cfg.loading == "tension":
                kd, mechanisms = self._tension_knockdown_analytical(theta_max, cfg)
                if kd < kd_compression:
                    kd = kd_compression
                    mechanisms["mode"] = mechanisms["mode"] + " (capped)"
                ref_strength = cfg.material.Xt
                results.tension_mechanisms = mechanisms
            else:
                kd = kd_compression
                ref_strength = cfg.material.Xc
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
        results.analytical_strength_MPa = ref_strength * kd

    # ------------------------------------------------------------------
    # Tension analytical model — three-mechanism knockdown
    # ------------------------------------------------------------------

    @staticmethod
    def _tension_knockdown_analytical(
        theta: float, cfg: "AnalysisConfig",
        _return_kd0_only: bool = False,
    ) -> Tuple[float, dict]:
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
        mat = cfg.material
        if mat is None:
            return 1.0
        angles = cfg.angles if cfg.angles else [0, 45, -45, 90] * 6

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
        lam_thickness = len(angles) * cfg.ply_thickness
        amplitude = cfg.amplitude
        wavelength = cfg.wavelength

        # Effective thickness: max consecutive 0-degree plies. With no
        # continuous 0-degree block (n_adj == 0) the curved-beam model
        # has no fibrous load path to develop interlaminar σ₃₃ / τ₁₃,
        # so the OOP mechanism is inactive (kd_oop = 1.0).
        n_adj_oop = _max_consecutive_zero_plies(angles)

        if n_adj_oop == 0 or amplitude <= 1e-12 or wavelength <= 1e-12:
            kd_oop = 1.0
        else:
            # Peak curvature at crest: κ = (2π/λ)² A
            kappa_max = (2.0 * np.pi / wavelength) ** 2 * amplitude
            # Max curvature gradient at inflection: |dκ/dx| = (2π/λ)³ A
            dkappa_dx_max = (2.0 * np.pi / wavelength) ** 3 * amplitude

            h_eff = n_adj_oop * cfg.ply_thickness

            # σ₃₃ at crest (mode I) and τ₁₃ at inflection (mode II)
            sigma33 = Xt * h_eff * kappa_max
            tau13 = Xt * h_eff * dkappa_dx_max

            # Failure indices (peak at different spatial locations)
            FI_s33 = (sigma33 / Yt) ** 2
            FI_t13 = (tau13 / S13) ** 2
            FI_max = max(FI_s33, FI_t13)

            kd_oop = 1.0 / np.sqrt(1.0 + FI_max)

        # 0-degree ply knockdown: minimum of all three mechanisms
        kd_0 = min(kd_fiber, kd_matrix, kd_oop)

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
            "f_0": f_0,
            "mode": mode,
        }

        return float(kd_lam), mechanisms

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

        # Field-level evaluation
        fi_fields, mode_fields = evaluator.evaluate_field(
            field_results.stress_local,
            materials,
            mesh.ply_ids,
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
            import logging
            logging.getLogger(__name__).warning(
                "CLT evaluation skipped: %s", exc
            )

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

        if results.failure_indices is None:
            return

        # Build a flat (no wrinkle) mesh with same dimensions
        flat_profile = GaussianSinusoidal(
            amplitude=0.0,
            wavelength=cfg.wavelength,
            width=cfg.width,
            center=cfg.domain_length / 2.0,
        )
        flat_config = WrinkleConfiguration.from_morphology_name(
            "stack", flat_profile,
            interface1=cfg.interface_1,
            interface2=cfg.interface_2,
        )

        flat_mesh_gen = WrinkleMesh(
            laminate=laminate,
            wrinkle_config=flat_config,
            Lx=cfg.domain_length,
            Ly=cfg.domain_width,
            nx=cfg.nx,
            ny=cfg.ny,
            nz_per_ply=cfg.nz_per_ply,
        )
        flat_mesh = flat_mesh_gen.generate()

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
        # E_eff = <σ₁₁> / ε_applied  where σ₁₁ is the local-coords fiber-direction stress
        # modulus_retention = E_wrinkled / E_pristine
        try:
            applied_strain = cfg.applied_strain
            if applied_strain == 0.0:
                results.modulus_retention = 1.0
            else:
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

