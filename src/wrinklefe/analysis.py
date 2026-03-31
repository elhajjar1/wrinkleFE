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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

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
        Phi(z_p)   = exp(-(z_p - T/2)^2 / A^2)   (through-thickness decay)

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
        at the midplane with scale = amplitude.  If False, all plies see
        the full wrinkle angle profile (Phi = 1).

    Returns
    -------
    float
        Profile-averaged BF knockdown factor (0, 1].
    """
    T = n_plies * ply_thickness
    z_mid = T / 2.0
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
            phi_p = np.exp(-((z_p - z_mid) ** 2) / (amplitude ** 2))
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
    """
    max_count = 0
    count = 0
    for a in angles:
        if abs(a) < tol:
            count += 1
            max_count = max(max_count, count)
        else:
            count = 0
    return max(max_count, 1)


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
        Wrinkle amplitude A [mm].  Default 0.366 (2 ply thicknesses).
    wavelength : float
        Wrinkle wavelength lambda [mm].  Default 16.0.
    width : float
        Gaussian envelope half-width [mm].  Default 12.0.
    morphology : str
        Morphology name: ``'stack'``, ``'convex'``, or ``'concave'``.
        Default is ``'stack'``.
    loading : str
        Loading mode: ``'compression'`` or ``'tension'``.
        Default is ``'compression'``.
    material : OrthotropicMaterial or None
        Composite material.  ``None`` uses the default IM7/8552.
    angles : list[float] or None
        Ply angles in degrees.  ``None`` uses a quasi-isotropic
        ``[0/45/-45/90]_3s`` layup (24 plies).
    interface_1 : int
        Ply interface for the first wrinkle.  Default 11.
    interface_2 : int
        Ply interface for the second wrinkle.  Default 12.
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
    run_buckling : bool
        Whether to perform buckling analysis.  Default ``False``.
    n_buckling_modes : int
        Number of buckling modes to extract.  Default 5.
    run_montecarlo : bool
        Whether to perform Monte Carlo analysis.  Default ``False``.
    mc_samples : int
        Number of Monte Carlo samples.  Default 5000.
    mc_seed : int or None
        Random seed for Monte Carlo reproducibility.  Default 42.
    verbose : bool
        Print progress information.  Default ``False``.
    """

    # Wrinkle geometry
    amplitude: float = 0.366
    wavelength: float = 16.0
    width: float = 12.0

    # Morphology
    morphology: str = "stack"
    decay_floor: float = 0.0  # graded mode: min amplitude fraction at surfaces (0–1)

    # Loading
    loading: str = "compression"

    # Material & laminate
    material: Optional[OrthotropicMaterial] = None
    angles: Optional[List[float]] = None

    # Ply thickness
    ply_thickness: float = 0.183  # mm (1 ply thickness for CYCOM X850/T800)

    # Wrinkle placement
    interface_1: int = 11
    interface_2: int = 12

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

    def run(self) -> AnalysisResults:
        """Execute the complete analysis pipeline.

        Steps
        -----
        1. Build laminate from material and ply angles.
        2. Create wrinkle profile and morphology configuration.
        3. Compute analytical predictions (knockdown, strength).
        4. Generate mesh with wrinkle deformation.
        5. Run static FE analysis.
        6. Evaluate failure criteria on the FE stress field.
        7. (Optional) Run buckling analysis.
        8. (Optional) Run Monte Carlo + Jensen gap analysis.

        Returns
        -------
        AnalysisResults
            Complete results from all analysis steps.
        """
        cfg = self.config
        results = AnalysisResults(config=cfg)

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
        wrinkle_config = WrinkleConfiguration.from_morphology_name(
            cfg.morphology, profile,
            interface1=cfg.interface_1,
            interface2=cfg.interface_2,
            decay_floor=cfg.decay_floor,
        )
        results.wrinkle_config = wrinkle_config

        # 3. Analytical predictions
        self._compute_analytical(results, wrinkle_config)

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

        # 5. Static FE solve
        solver = StaticSolver(mesh, laminate)
        bcs = BoundaryHandler.compression_bcs(
            mesh, applied_strain=cfg.applied_strain
        )
        field_results = solver.solve(
            bcs, solver=cfg.solver, verbose=cfg.verbose
        )
        results.field_results = field_results

        # 6. Failure evaluation on FE field
        self._evaluate_failure(results, laminate, field_results, mesh)

        # 6b. Retention factor (baseline pristine comparison)
        self._compute_retention_factors(results, laminate)

        return results

    # ------------------------------------------------------------------
    # Morphology comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare_morphologies(
        base_config: AnalysisConfig,
        morphologies: Sequence[str] = ("stack", "convex", "concave"),
    ) -> Dict[str, AnalysisResults]:
        """Run the full FE analysis for multiple morphologies and compare.

        Parameters
        ----------
        base_config : AnalysisConfig
            Base configuration.  The ``morphology`` field is overridden
            for each entry in *morphologies*.
        morphologies : sequence of str
            Morphology names to compare.

        Returns
        -------
        dict[str, AnalysisResults]
            Mapping from morphology name to its results.
        """
        all_results: Dict[str, AnalysisResults] = {}

        for morph in morphologies:
            cfg = AnalysisConfig(
                amplitude=base_config.amplitude,
                wavelength=base_config.wavelength,
                width=base_config.width,
                morphology=morph,
                loading=base_config.loading,
                material=base_config.material,
                angles=base_config.angles,
                ply_thickness=base_config.ply_thickness,
                interface_1=base_config.interface_1,
                interface_2=base_config.interface_2,
                nx=base_config.nx,
                ny=base_config.ny,
                nz_per_ply=base_config.nz_per_ply,
                domain_length=base_config.domain_length,
                domain_width=base_config.domain_width,
                applied_strain=base_config.applied_strain,
                solver=base_config.solver,
                verbose=base_config.verbose,
            )
            all_results[morph] = WrinkleAnalysis(cfg).run()

        return all_results

    # ------------------------------------------------------------------
    # Parametric sweep
    # ------------------------------------------------------------------

    @staticmethod
    def parametric_sweep(
        base_config: AnalysisConfig,
        parameter: str,
        values: Sequence[float],
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

        Returns
        -------
        list[AnalysisResults]
            One result per value, in the same order as *values*.

        Raises
        ------
        AttributeError
            If *parameter* is not a valid :class:`AnalysisConfig` field.
        """
        results_list: List[AnalysisResults] = []

        for val in values:
            cfg = AnalysisConfig(
                amplitude=base_config.amplitude,
                wavelength=base_config.wavelength,
                width=base_config.width,
                morphology=base_config.morphology,
                loading=base_config.loading,
                material=base_config.material,
                angles=base_config.angles,
                ply_thickness=base_config.ply_thickness,
                interface_1=base_config.interface_1,
                interface_2=base_config.interface_2,
                nx=base_config.nx,
                ny=base_config.ny,
                nz_per_ply=base_config.nz_per_ply,
                domain_length=base_config.domain_length,
                domain_width=base_config.domain_width,
                applied_strain=base_config.applied_strain,
                solver=base_config.solver,
                verbose=base_config.verbose,
            )
            if not hasattr(cfg, parameter):
                raise AttributeError(
                    f"AnalysisConfig has no field '{parameter}'"
                )
            setattr(cfg, parameter, val)
            # Rerun __post_init__ if domain_length depends on wavelength
            if parameter == "wavelength" and base_config.domain_length <= 0:
                cfg.domain_length = 3.0 * val

            results_list.append(WrinkleAnalysis(cfg).run())

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
            # Profile-proportional compression knockdown (graded/embedded)
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
            )
            kd_compression = f_0 * kd_profile + (1.0 - f_0)

            if cfg.loading == "tension":
                # Average tension knockdown over 0-deg plies at local angles
                # (profile-proportional, using linear grading as fallback
                # for the three-mechanism model)
                p_mid = (n_plies - 1) / 2.0
                zero_positions = [i for i, a in enumerate(angles) if abs(a) < 5]
                decay_floor = cfg.decay_floor
                kd_0_sum = 0.0
                for p in zero_positions:
                    raw = max(0.0, 1.0 - abs(p - p_mid) / p_mid)
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

        if amplitude > 1e-12 and wavelength > 1e-12:
            # Peak curvature at crest: κ = (2π/λ)² A
            kappa_max = (2.0 * np.pi / wavelength) ** 2 * amplitude
            # Max curvature gradient at inflection: |dκ/dx| = (2π/λ)³ A
            dkappa_dx_max = (2.0 * np.pi / wavelength) ** 3 * amplitude

            # Effective thickness: max consecutive 0-degree plies
            n_adj = _max_consecutive_zero_plies(angles)
            h_eff = n_adj * cfg.ply_thickness

            # σ₃₃ at crest (mode I) and τ₁₃ at inflection (mode II)
            sigma33 = Xt * h_eff * kappa_max
            tau13 = Xt * h_eff * dkappa_dx_max

            # Failure indices (peak at different spatial locations)
            FI_s33 = (sigma33 / Yt) ** 2
            FI_t13 = (tau13 / S13) ** 2
            FI_max = max(FI_s33, FI_t13)

            kd_oop = 1.0 / np.sqrt(1.0 + FI_max)
        else:
            kd_oop = 1.0

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

