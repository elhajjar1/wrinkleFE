"""Command-line interface for WrinkleFE.

Provides subcommands for running analyses, comparing morphologies,
performing parametric sweeps, and listing available materials.

Entry point registered in ``pyproject.toml`` as ``wrinklefe``.

Usage
-----
::

    wrinklefe analyze --amplitude 0.366 --morphology concave --verbose
    wrinklefe compare --amplitude 0.366 --wavelength 16.0
    wrinklefe sweep --parameter amplitude --min 0.183 --max 0.549 --steps 5
    wrinklefe materials
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from wrinklefe.analysis import AnalysisResults

from wrinklefe.core.layup import parse_layup
from wrinklefe.core.morphology import MORPHOLOGY_PHASES, SINGLE_WRINKLE_MODES

# Single source of truth for the morphology names the CLI accepts. Pulled
# straight from ``core.morphology`` so the CLI can never drift from the
# engine / Streamlit app (see issue #83).
MORPHOLOGY_CHOICES = sorted(set(MORPHOLOGY_PHASES.keys()) | SINGLE_WRINKLE_MODES)


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="wrinklefe",
        description=(
            "WrinkleFE -- Finite element analysis of wrinkled composite laminates "
            "with advanced failure theories."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ------------------------------------------------------------------ #
    # analyze
    # ------------------------------------------------------------------ #
    p_analyze = subparsers.add_parser(
        "analyze",
        help="Run a single wrinkle analysis",
        description="Run an analytical or full FE analysis for a single configuration.",
    )
    p_analyze.add_argument(
        "--amplitude", type=float, default=0.366,
        help=(
            "Wrinkle half-amplitude A in mm: peak displacement of the "
            "wrinkled mid-surface from the flat reference, so "
            "z(x) = A*cos(2*pi*x/lambda) and the peak-to-trough height "
            "is 2A (default: 0.366)"
        ),
    )
    p_analyze.add_argument(
        "--wavelength", type=float, default=16.0,
        help=(
            "Wrinkle wavelength lambda in mm: spatial period of the "
            "cos(2*pi*(x - x0)/lambda) carrier (crest-to-crest "
            "distance). Must be > 0 (default: 16.0)"
        ),
    )
    p_analyze.add_argument(
        "--width", type=float, default=12.0,
        help=(
            "Wrinkle envelope decay length w in mm about the centre x0 "
            "(Gaussian 1/e length in exp(-(x-x0)^2/w^2); also the "
            "transverse y-extent in 3-D dual-wrinkle / graded modes). "
            "Must be > 0 (default: 12.0)"
        ),
    )
    p_analyze.add_argument(
        "--morphology", type=str, default="stack",
        choices=MORPHOLOGY_CHOICES,
        help=(
            "Wrinkle morphology type (default: stack). "
            f"One of: {', '.join(MORPHOLOGY_CHOICES)}."
        ),
    )
    p_analyze.add_argument(
        "--amplitude-profile", type=str, default="constant",
        choices=["constant", "gaussian", "linear"],
        dest="amplitude_profile",
        help=(
            "Spatially varying in-plane modulation of the wrinkle "
            "amplitude A, applied on top of the wrinkle's own "
            "longitudinal envelope (default: constant). "
            "'gaussian' multiplies A by exp(-(s/d)^2); 'linear' by "
            "max(0, 1-|s|/d); 's' is the coordinate from the wrinkle "
            "centre along --amplitude-profile-axis and 'd' is "
            "--amplitude-profile-decay-length."
        ),
    )
    p_analyze.add_argument(
        "--amplitude-profile-decay-length", type=float, default=None,
        dest="amplitude_profile_decay_length",
        help=(
            "Length scale d in mm controlling the Gaussian sigma / "
            "linear-decay extent. Must be > 0 when set. Ignored when "
            "--amplitude-profile=constant. Defaults to the wrinkle's "
            "own envelope width."
        ),
    )
    p_analyze.add_argument(
        "--amplitude-profile-axis", type=str, default="x",
        choices=["x", "y"],
        dest="amplitude_profile_axis",
        help=(
            "In-plane axis along which the amplitude modulation runs "
            "(default: x). Use 'y' for an independent transverse "
            "tapering of A that does not stack with the existing "
            "longitudinal envelope on x."
        ),
    )
    p_analyze.add_argument(
        "--loading", type=str, default="compression",
        choices=["compression", "tension"],
        help="Loading mode (default: compression)",
    )
    p_analyze.add_argument(
        "--material", type=str, default=None,
        help="Material name from MaterialLibrary (default: IM7_8552)",
    )
    p_analyze.add_argument(
        "--angles", "--layup", type=str, default=None, dest="angles",
        help=(
            "Ply layup. Accepts an explicit comma-separated list "
            "(e.g. '0,45,-45,90') or contracted notation "
            "(e.g. '[0/45/-45/90]_3s', '[0/±45/90]s'). "
            "Default: quasi-isotropic [0/45/-45/90]_3s"
        ),
    )
    p_analyze.add_argument(
        "--interface-1", type=int, default=None, dest="interface_1",
        help=(
            "Zero-based ply interface index for the first wrinkle. "
            "Must satisfy 0 <= index < n_plies. When omitted, auto-"
            "derived from the layup (mid-thickness interior interface)."
        ),
    )
    p_analyze.add_argument(
        "--interface-2", type=int, default=None, dest="interface_2",
        help=(
            "Zero-based ply interface index for the second wrinkle. "
            "Must satisfy 0 <= index < n_plies. When omitted, auto-"
            "derived from the layup (mid-thickness interior interface)."
        ),
    )
    p_analyze.add_argument(
        "--nx", type=int, default=12,
        help="Mesh divisions in x (default: 12)",
    )
    p_analyze.add_argument(
        "--ny", type=int, default=6,
        help="Mesh divisions in y (default: 6)",
    )
    p_analyze.add_argument(
        "--strain", type=float, default=-0.01,
        help="Applied nominal strain (default: -0.01)",
    )
    p_analyze.add_argument(
        "--solver", type=str, default="direct",
        choices=["direct", "iterative"],
        help="Linear solver type (default: direct)",
    )
    p_analyze.add_argument(
        "--fe", action=argparse.BooleanOptionalAction, default=None,
        help=(
            "Force a full FE solve (--fe) or skip it (--no-fe). "
            "Default: FE on unless overridden."
        ),
    )
    p_analyze.add_argument(
        "--analytical-only", action="store_true", default=False,
        help=(
            "Run analytical predictions only, skipping the FE solve."
        ),
    )
    p_analyze.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help=(
            "Show detailed pipeline progress (sets the 'wrinklefe' "
            "logger to DEBUG with a stderr handler)"
        ),
    )
    p_analyze.add_argument(
        "--output-json", type=str, default=None,
        help="Export results to JSON file at specified path",
    )

    # ------------------------------------------------------------------ #
    # CZM (cohesive zone modelling) flags. Off by default; when
    # --enable-czm is absent every other czm flag is silently ignored so
    # the legacy ``analyze`` behaviour is bit-identical.
    # ------------------------------------------------------------------ #
    p_analyze.add_argument(
        "--enable-czm", action="store_true", default=False,
        dest="enable_czm",
        help=(
            "Enable cohesive zone modelling (delamination prediction). "
            "Inserts zero-thickness cohesive interface elements at ply "
            "boundaries and switches the FE solve from the linear "
            "StaticSolver to Newton-Raphson. Run time is ~5-20x longer; "
            "produces damage field output and energy dissipation per "
            "interface."
        ),
    )
    p_analyze.add_argument(
        "--czm-GIc", type=float, default=None, dest="czm_GIc",
        help=(
            "Mode-I fracture toughness G_Ic in N/mm. Defaults to the "
            "material's GIc when omitted."
        ),
    )
    p_analyze.add_argument(
        "--czm-GIIc", type=float, default=None, dest="czm_GIIc",
        help=(
            "Mode-II fracture toughness G_IIc in N/mm. Defaults to the "
            "material's GIIc when omitted."
        ),
    )
    p_analyze.add_argument(
        "--czm-sigma-max", type=float, default=None, dest="czm_sigma_max",
        help=(
            "Mode-I peak interface strength sigma_max in MPa. Defaults "
            "to the material's sigma_max when omitted."
        ),
    )
    p_analyze.add_argument(
        "--czm-tau-max", type=float, default=None, dest="czm_tau_max",
        help=(
            "Mode-II peak interface strength tau_max in MPa. Defaults "
            "to the material's tau_max when omitted."
        ),
    )
    p_analyze.add_argument(
        "--czm-interfaces", type=str, default="near_crest",
        dest="czm_interfaces",
        help=(
            "Which ply interfaces receive cohesive elements. Accepts "
            "'near_crest' (default), 'all', or a comma-separated list "
            "of interface indices (e.g. '11,12')."
        ),
    )
    p_analyze.add_argument(
        "--czm-load-increments", type=int, default=20,
        dest="czm_load_increments",
        help=(
            "Number of Newton-Raphson load increments for the CZM "
            "solve (default: 20)."
        ),
    )
    p_analyze.add_argument(
        "--czm-newton-tol", type=float, default=1.0e-4,
        dest="czm_newton_tol",
        help="Newton-Raphson residual tolerance (default: 1e-4).",
    )
    p_analyze.add_argument(
        "--save-czm-figure", type=str, default=None,
        dest="save_czm_figure",
        help=(
            "When --enable-czm is set, save the CZM overview figure "
            "(2x2 dashboard) to this path. Matplotlib infers the "
            "format from the file extension (.png, .pdf, .svg)."
        ),
    )

    # ------------------------------------------------------------------ #
    # compare
    # ------------------------------------------------------------------ #
    p_compare = subparsers.add_parser(
        "compare",
        help="Compare all three morphologies",
        description="Run analysis for stack, convex, and concave morphologies and compare.",
    )
    p_compare.add_argument(
        "--amplitude", type=float, default=0.366,
        help=(
            "Wrinkle half-amplitude A in mm: peak displacement of the "
            "wrinkled mid-surface from the flat reference, so "
            "z(x) = A*cos(2*pi*x/lambda) and the peak-to-trough height "
            "is 2A (default: 0.366)"
        ),
    )
    p_compare.add_argument(
        "--wavelength", type=float, default=16.0,
        help=(
            "Wrinkle wavelength lambda in mm: spatial period of the "
            "cos(2*pi*(x - x0)/lambda) carrier (crest-to-crest "
            "distance). Must be > 0 (default: 16.0)"
        ),
    )
    p_compare.add_argument(
        "--width", type=float, default=12.0,
        help=(
            "Wrinkle envelope decay length w in mm about the centre x0 "
            "(Gaussian 1/e length in exp(-(x-x0)^2/w^2); also the "
            "transverse y-extent in 3-D dual-wrinkle / graded modes). "
            "Must be > 0 (default: 12.0)"
        ),
    )
    p_compare.add_argument(
        "--amplitude-profile", type=str, default="constant",
        choices=["constant", "gaussian", "linear"],
        dest="amplitude_profile",
        help=(
            "Spatially varying in-plane amplitude modulation applied to "
            "every morphology compared (default: constant). See "
            "`wrinklefe analyze --help` for the full definition."
        ),
    )
    p_compare.add_argument(
        "--amplitude-profile-decay-length", type=float, default=None,
        dest="amplitude_profile_decay_length",
        help=(
            "Decay length d in mm for --amplitude-profile. Must be > 0 "
            "when set. Defaults to the wrinkle's own envelope width."
        ),
    )
    p_compare.add_argument(
        "--amplitude-profile-axis", type=str, default="x",
        choices=["x", "y"],
        dest="amplitude_profile_axis",
        help="Axis for --amplitude-profile (default: x).",
    )
    p_compare.add_argument(
        "--analytical-only", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Run analytical predictions only and skip the FE solve "
            "(default). Use --no-analytical-only to run the full FE "
            "comparison for all three morphologies."
        ),
    )
    p_compare.add_argument(
        "--output-json", type=str, default=None, dest="output_json",
        help=(
            "Write the comparison results to a JSON file (array of "
            "per-run objects matching 'analyze --output-json'); the "
            "stdout table is still printed"
        ),
    )
    p_compare.add_argument(
        "--output-csv", type=str, default=None, dest="output_csv",
        help=(
            "Write the comparison results to a tidy CSV (one row per "
            "morphology, full float precision); the stdout table is "
            "still printed"
        ),
    )
    p_compare.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help=(
            "Show detailed pipeline progress (sets the 'wrinklefe' "
            "logger to DEBUG with a stderr handler)"
        ),
    )

    # ------------------------------------------------------------------ #
    # sweep
    # ------------------------------------------------------------------ #
    p_sweep = subparsers.add_parser(
        "sweep",
        help="Parametric sweep over a single parameter",
        description="Sweep a wrinkle parameter and report strength vs. parameter.",
    )
    p_sweep.add_argument(
        "--parameter", type=str, required=True,
        help=(
            "Numeric AnalysisConfig field to sweep "
            "(e.g. amplitude, wavelength, width, applied_strain)"
        ),
    )
    p_sweep.add_argument(
        "--min", type=float, required=True, dest="sweep_min",
        help="Minimum parameter value",
    )
    p_sweep.add_argument(
        "--max", type=float, required=True, dest="sweep_max",
        help="Maximum parameter value",
    )
    p_sweep.add_argument(
        "--steps", type=int, default=10,
        help="Number of sweep steps (default: 10)",
    )
    p_sweep.add_argument(
        "--output-json", type=str, default=None, dest="output_json",
        help=(
            "Write the sweep results to a JSON file (array of per-run "
            "objects matching 'analyze --output-json'); the stdout "
            "table is still printed"
        ),
    )
    p_sweep.add_argument(
        "--output-csv", type=str, default=None, dest="output_csv",
        help=(
            "Write the sweep results to a tidy CSV (one row per run, "
            "full float precision); the stdout table is still printed"
        ),
    )
    p_sweep.add_argument(
        "--morphology", type=str, default="stack",
        choices=MORPHOLOGY_CHOICES,
        help=(
            "Morphology for the sweep (default: stack). "
            f"One of: {', '.join(MORPHOLOGY_CHOICES)}."
        ),
    )
    p_sweep.add_argument(
        "--amplitude-profile", type=str, default="constant",
        choices=["constant", "gaussian", "linear"],
        dest="amplitude_profile",
        help=(
            "Spatially varying in-plane amplitude modulation held "
            "fixed across the sweep (default: constant). See "
            "`wrinklefe analyze --help` for the full definition."
        ),
    )
    p_sweep.add_argument(
        "--amplitude-profile-decay-length", type=float, default=None,
        dest="amplitude_profile_decay_length",
        help=(
            "Decay length d in mm for --amplitude-profile. Must be > 0 "
            "when set. Defaults to the wrinkle's own envelope width."
        ),
    )
    p_sweep.add_argument(
        "--amplitude-profile-axis", type=str, default="x",
        choices=["x", "y"],
        dest="amplitude_profile_axis",
        help="Axis for --amplitude-profile (default: x).",
    )
    p_sweep.add_argument(
        "--analytical-only", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Run analytical predictions only for each sweep value "
            "(default). Use --no-analytical-only to run a full FE "
            "sweep (slow)."
        ),
    )
    p_sweep.add_argument(
        "--parallel", type=int, default=1, metavar="N",
        help=(
            "Number of worker processes for the per-value solves "
            "(default: 1 = sequential; 0 = all CPU cores). Each sweep "
            "value is an independent analysis, so a full FE sweep "
            "scales nearly linearly with workers. Peak memory scales "
            "with N x the per-solve footprint — size N by available "
            "RAM for fine meshes."
        ),
    )
    p_sweep.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help=(
            "Show detailed pipeline progress (sets the 'wrinklefe' "
            "logger to DEBUG with a stderr handler)"
        ),
    )

    # ------------------------------------------------------------------ #
    # converge
    # ------------------------------------------------------------------ #
    p_converge = subparsers.add_parser(
        "converge",
        help="Run a mesh-convergence study",
        description=(
            "Run the FE analysis at successively refined meshes, report "
            "the per-level QoI table, and recommend the coarsest mesh "
            "within a tolerance of the finest level."
        ),
    )
    p_converge.add_argument(
        "--amplitude", type=float, default=0.366,
        help="Wrinkle half-amplitude A in mm (default: 0.366)",
    )
    p_converge.add_argument(
        "--wavelength", type=float, default=16.0,
        help="Wrinkle wavelength lambda in mm (default: 16.0)",
    )
    p_converge.add_argument(
        "--width", type=float, default=12.0,
        help="Wrinkle envelope width w in mm (default: 12.0)",
    )
    p_converge.add_argument(
        "--morphology", type=str, default="stack",
        choices=MORPHOLOGY_CHOICES,
        help="Wrinkle morphology (default: stack)",
    )
    p_converge.add_argument(
        "--loading", type=str, default="compression",
        choices=["compression", "tension"],
        help="Loading condition (default: compression)",
    )
    p_converge.add_argument(
        "--material", type=str, default=None,
        help="Material name from MaterialLibrary (default: IM7/8552)",
    )
    p_converge.add_argument(
        "--angles", "--layup", type=str, default=None, dest="angles",
        help="Layup, contracted (e.g. '[0/45/-45/90]s') or comma-separated",
    )
    p_converge.add_argument(
        "--nx", type=int, default=12,
        help="Level-0 elements along the length (default: 12)",
    )
    p_converge.add_argument(
        "--ny", type=int, default=6,
        help="Level-0 elements across the width (default: 6)",
    )
    p_converge.add_argument(
        "--nz-per-ply", type=int, default=1, dest="nz_per_ply",
        help="Level-0 elements per ply through-thickness (default: 1)",
    )
    p_converge.add_argument(
        "--strain", type=float, default=-0.01,
        help="Applied strain (default: -0.01)",
    )
    p_converge.add_argument(
        "--levels", type=int, default=4,
        help="Number of refinement levels (default: 4)",
    )
    p_converge.add_argument(
        "--tolerance", type=float, default=0.01,
        help="Relative QoI tolerance for the recommendation (default: 0.01)",
    )
    p_converge.add_argument(
        "--refine", type=str, default="nx,nz_per_ply",
        help=(
            "Comma-separated mesh axes to refine, from nx, ny, "
            "nz_per_ply (default: nx,nz_per_ply)"
        ),
    )
    p_converge.add_argument(
        "--qoi", type=str, default="max_fi",
        choices=["max_fi", "modulus_retention", "strength_retention",
                 "max_damage"],
        help="Quantity of interest per level (default: max_fi)",
    )
    p_converge.add_argument(
        "--save-plot", type=str, default=None,
        help="Save the QoI-vs-DOF convergence plot to this path",
    )
    p_converge.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help=(
            "Show detailed pipeline progress (sets the 'wrinklefe' "
            "logger to DEBUG with a stderr handler)"
        ),
    )

    # ------------------------------------------------------------------ #
    # materials
    # ------------------------------------------------------------------ #
    subparsers.add_parser(
        "materials",
        help="List available materials in MaterialLibrary",
        description="Display all built-in and loaded materials with key properties.",
    )

    return parser


# ====================================================================== #
# Subcommand handlers
# ====================================================================== #

def _parse_czm_interfaces(value: str | None):
    """Parse --czm-interfaces into the form ``AnalysisConfig`` expects.

    Accepts the two sentinel strings ``"near_crest"`` / ``"all"`` (passed
    through verbatim) and a comma-separated list of non-negative
    integers (returned as a ``list[int]``). Anything else is a fatal
    CLI error so the user gets a clear message rather than a deep
    validation traceback.
    """
    if value is None:
        return "near_crest"
    s = value.strip()
    if s in ("near_crest", "all"):
        return s
    # Otherwise treat as comma-separated list of ints.
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        print(
            "error: --czm-interfaces must be 'near_crest', 'all', or "
            "a comma-separated list of interface indices",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        return [int(p) for p in parts]
    except ValueError:
        print(
            f"error: --czm-interfaces could not parse {value!r} as a "
            "list of integers",
            file=sys.stderr,
        )
        sys.exit(2)


def _parse_angles(angles_str: str | None) -> list[float] | None:
    """Parse a layup string into a list of ply angles (degrees).

    Accepts both an explicit comma/semicolon/newline-separated list
    (e.g. ``0,45,-45,90``) and contracted notation
    (e.g. ``[0/45/-45/90]_3s``) via the shared
    :func:`wrinklefe.core.layup.parse_layup` parser. On a malformed
    layup, prints a clear error and exits with a non-zero status.
    """
    if angles_str is None:
        return None
    try:
        return parse_layup(angles_str)
    except ValueError as exc:
        print(f"error: invalid layup: {exc}", file=sys.stderr)
        sys.exit(2)


def _cmd_analyze(args: argparse.Namespace) -> None:
    """Handle the ``analyze`` subcommand."""
    from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
    from wrinklefe.core.material import MaterialLibrary

    # Resolve material
    material = None
    if args.material is not None:
        lib = MaterialLibrary()
        material = lib.get(args.material)

    angles = _parse_angles(args.angles)

    # Resolve analytical-only vs. full FE intent.
    #
    # Precedence (highest first):
    #   1. --enable-czm      -> force full FE solve (CZM requires NR)
    #   2. --analytical-only -> skip FE
    #   3. --fe              -> full FE solve
    #   4. --no-fe           -> analytical-only
    #   5. default           -> full FE solve
    enable_czm = bool(getattr(args, "enable_czm", False))
    if enable_czm:
        analytical_only = False
    elif args.analytical_only:
        analytical_only = True
    elif args.fe is True:
        analytical_only = False
    elif args.fe is False:
        analytical_only = True
    else:
        analytical_only = False

    # CZM kwargs. When --enable-czm is absent these defaults exactly
    # match ``AnalysisConfig``'s class defaults so the resulting config
    # is bit-identical to the pre-CZM behaviour.
    czm_kwargs: dict = {}
    if enable_czm:
        czm_kwargs = dict(
            enable_czm=True,
            czm_interfaces=_parse_czm_interfaces(args.czm_interfaces),
            czm_GIc=args.czm_GIc,
            czm_GIIc=args.czm_GIIc,
            czm_sigma_max=args.czm_sigma_max,
            czm_tau_max=args.czm_tau_max,
            czm_n_load_increments=args.czm_load_increments,
            czm_newton_tol=args.czm_newton_tol,
        )

    config = AnalysisConfig(
        amplitude=args.amplitude,
        wavelength=args.wavelength,
        width=args.width,
        morphology=args.morphology,
        amplitude_profile=args.amplitude_profile,
        amplitude_profile_decay_length=args.amplitude_profile_decay_length,
        amplitude_profile_axis=args.amplitude_profile_axis,
        loading=args.loading,
        material=material,
        angles=angles,
        interface_1=args.interface_1,
        interface_2=args.interface_2,
        nx=args.nx,
        ny=args.ny,
        applied_strain=args.strain,
        solver=args.solver,
        analytical_only=analytical_only,
        verbose=args.verbose,
        **czm_kwargs,
    )

    analysis = WrinkleAnalysis(config)
    try:
        result = analysis.run(analytical_only=analytical_only)
    except Exception as exc:  # pragma: no cover - exercised via tests/CLI
        print(f"error: analysis failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(result.summary())

    # Extra CZM section: AnalysisResults.summary() already prints the
    # core CZM metrics (max/mean damage, energy, convergence) but it
    # does not break out crack length or # interfaces above the 0.5
    # threshold. Emit those here so the CLI surface matches the
    # Streamlit "Cohesive Zone Modeling Results" block.
    if enable_czm and result.czm_damage is not None and result.czm_damage.size:
        _print_czm_extras(result)

    # Optional CZM figure export.
    if enable_czm and args.save_czm_figure is not None:
        if result.czm_damage is None or not result.czm_damage.size:
            print(
                "warning: --save-czm-figure ignored, no CZM damage data "
                "was produced (the solve may have failed to insert "
                "cohesive elements).",
                file=sys.stderr,
            )
        else:
            try:
                from wrinklefe.viz import czm_overview_figure
                fig = czm_overview_figure(result)
                fig.savefig(args.save_czm_figure)
                print(f"\nCZM overview figure saved to: {args.save_czm_figure}")
                # Close to free memory and keep matplotlib's "too many open
                # figures" warning quiet when the CLI is scripted.
                import matplotlib.pyplot as _plt
                _plt.close(fig)
            except Exception as exc:  # pragma: no cover - exercised via tests
                print(
                    f"warning: failed to save CZM figure: {exc}",
                    file=sys.stderr,
                )

    # Export to JSON if requested
    if args.output_json is not None:
        from wrinklefe.io.export import export_results_json
        export_results_json(result, args.output_json)
        print(f"\nResults exported to: {args.output_json}")


def _print_czm_extras(result) -> None:
    """Print CZM crack length / damaged-interface counts after the summary.

    ``AnalysisResults.summary()`` already covers max/mean damage, energy,
    convergence, and the interface list. This helper appends the two
    extras the Streamlit UI also exposes so the CLI and web surfaces
    show the same set of CZM metrics.
    """
    damage = result.czm_damage
    # Per-element damage = mean across Gauss points. An interface "fails"
    # if any of its elements exceed the 0.5 threshold.
    damage_per_elem = np.asarray(damage).mean(axis=1) if damage.ndim == 2 else np.asarray(damage)
    n_above_half = int(np.sum(damage_per_elem > 0.5))

    crack_per_iface = result.czm_crack_length_per_interface or {}

    print()
    print("  Cohesive Zone Modeling extras:")
    print(f"    Elements with damage > 0.5: {n_above_half}")
    if crack_per_iface:
        for iface, length in sorted(crack_per_iface.items()):
            print(f"    Crack length, interface {iface}: {length:.4e} mm")
    else:
        print("    Crack length per interface: (none)")


def _cmd_compare(args: argparse.Namespace) -> None:
    """Handle the ``compare`` subcommand."""
    from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis

    config = AnalysisConfig(
        amplitude=args.amplitude,
        wavelength=args.wavelength,
        width=args.width,
        amplitude_profile=args.amplitude_profile,
        amplitude_profile_decay_length=args.amplitude_profile_decay_length,
        amplitude_profile_axis=args.amplitude_profile_axis,
        verbose=args.verbose,
    )

    all_results = WrinkleAnalysis.compare_morphologies(
        config,
        morphologies=("stack", "convex", "concave"),
        analytical_only=args.analytical_only,
    )

    # Print comparison table
    print("=" * 72)
    print("  WrinkleFE Morphology Comparison")
    print("=" * 72)
    print(f"  Amplitude:  {args.amplitude:.3f} mm")
    print(f"  Wavelength: {args.wavelength:.1f} mm")
    print(f"  Width:      {args.width:.1f} mm")
    print()
    print(f"  {'Morphology':<12} {'M_f':>8} {'theta_max':>10} {'theta_eff':>10} "
          f"{'Damage':>8} {'Knockdown':>10} {'Strength':>10}")
    print(f"  {'':.<12} {'':.<8} {'(deg)':>10} {'(deg)':>10} "
          f"{'D':>8} {'':>10} {'(MPa)':>10}")
    print("-" * 72)

    for morph in ("stack", "convex", "concave"):
        r = all_results[morph]
        print(
            f"  {morph:<12} {r.morphology_factor:>8.4f} "
            f"{np.degrees(r.max_angle_rad):>10.2f} "
            f"{np.degrees(r.effective_angle_rad):>10.2f} "
            f"{r.damage_index:>8.4f} "
            f"{r.analytical_knockdown:>10.4f} "
            f"{r.analytical_strength_MPa:>10.1f}"
        )

    print("=" * 72)

    # Ranking
    ranked = sorted(all_results.items(), key=lambda x: x[1].analytical_strength_MPa, reverse=True)
    print("\n  Ranking (strongest to weakest):")
    for i, (morph, r) in enumerate(ranked, 1):
        print(f"    {i}. {morph:<10} {r.analytical_strength_MPa:.1f} MPa")
    print()

    _write_batch_outputs(
        [
            ("morphology", morph, morph, all_results[morph])
            for morph in ("stack", "convex", "concave")
        ],
        args.output_json,
        args.output_csv,
    )


def _write_batch_outputs(
    rows: list[tuple[str, float | str, str, AnalysisResults]],
    output_json: str | None,
    output_csv: str | None,
) -> None:
    """Write sweep/compare batch results to JSON and/or CSV (issue #266).

    Parameters
    ----------
    rows : list of (parameter_name, parameter_value, morphology, results)
        One entry per run, in output order.
    output_json : str or None
        Path for a JSON array of per-run objects, each matching the
        ``analyze --output-json`` schema (:func:`analysis_results_to_dict`).
    output_csv : str or None
        Path for a tidy CSV (one row per run, full float precision):
        ``parameter_name, parameter_value, morphology, knockdown,
        predicted_strength_MPa, max_failure_index, governing_criterion``.
    """
    if output_json is None and output_csv is None:
        return

    import csv
    import json
    from pathlib import Path

    from wrinklefe.io.export import analysis_results_to_dict

    if output_json is not None:
        payload = [analysis_results_to_dict(r) for _, _, _, r in rows]
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nResults written to: {path}")

    if output_csv is not None:
        fieldnames = [
            "parameter_name", "parameter_value", "morphology",
            "knockdown", "predicted_strength_MPa",
            "max_failure_index", "governing_criterion",
        ]
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name, value, morphology, r in rows:
                fi = None
                if r.failure_indices:
                    fi = max(
                        float(np.max(arr))
                        for arr in r.failure_indices.values()
                    )
                crit = getattr(
                    r.failure_report, "critical_criterion", None
                ) if r.failure_report is not None else None
                writer.writerow({
                    "parameter_name": name,
                    "parameter_value": value,
                    "morphology": morphology,
                    # repr() keeps full float precision; rounding stays a
                    # display-only concern of the stdout table.
                    "knockdown": repr(float(r.analytical_knockdown)),
                    "predicted_strength_MPa": repr(
                        float(r.analytical_strength_MPa)
                    ),
                    "max_failure_index": "" if fi is None else repr(fi),
                    "governing_criterion": crit or "",
                })
        print(f"\nResults written to: {path}")


def _cmd_sweep(args: argparse.Namespace) -> None:
    """Handle the ``sweep`` subcommand."""
    from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis

    # Validate the range/steps before burning any compute (issue #298).
    if args.sweep_min >= args.sweep_max:
        print(
            f"error: --min ({args.sweep_min}) must be less than --max "
            f"({args.sweep_max})",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.steps < 2:
        print(
            f"error: --steps must be >= 2 (got {args.steps})",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.parallel < 0:
        print(
            f"error: --parallel must be >= 0 (got {args.parallel}; "
            "0 = all CPU cores)",
            file=sys.stderr,
        )
        sys.exit(2)

    values = np.linspace(args.sweep_min, args.sweep_max, args.steps)

    config = AnalysisConfig(
        morphology=args.morphology,
        amplitude_profile=args.amplitude_profile,
        amplitude_profile_decay_length=args.amplitude_profile_decay_length,
        amplitude_profile_axis=args.amplitude_profile_axis,
        verbose=args.verbose,
    )

    # parametric_sweep raises AttributeError for an unknown field name;
    # catch it (and any config/solve error) and exit cleanly with a
    # message instead of a raw traceback, matching _cmd_analyze.
    try:
        results = WrinkleAnalysis.parametric_sweep(
            config,
            parameter=args.parameter,
            values=values.tolist(),
            analytical_only=args.analytical_only,
            n_workers=args.parallel,
        )
    except (AttributeError, ValueError, NotImplementedError) as exc:
        print(f"error: sweep failed: {exc}", file=sys.stderr)
        sys.exit(2)

    print("=" * 60)
    print(f"  WrinkleFE Parametric Sweep: {args.parameter}")
    print(f"  Morphology: {args.morphology}")
    print("=" * 60)
    print()
    print(f"  {args.parameter:>14} {'Knockdown':>12} {'Strength (MPa)':>16} {'Damage':>10}")
    print("-" * 60)

    for val, r in zip(values, results):
        print(
            f"  {val:>14.4f} {r.analytical_knockdown:>12.4f} "
            f"{r.analytical_strength_MPa:>16.1f} {r.damage_index:>10.4f}"
        )

    print("=" * 60)

    _write_batch_outputs(
        [
            (args.parameter, float(val), args.morphology, r)
            for val, r in zip(values, results)
        ],
        args.output_json,
        args.output_csv,
    )


def _cmd_materials(args: argparse.Namespace) -> None:
    """Handle the ``materials`` subcommand."""
    from wrinklefe.core.material import MaterialLibrary

    lib = MaterialLibrary()
    names = lib.list_names()

    print("=" * 78)
    print("  WrinkleFE Material Library")
    print("=" * 78)
    print()
    print(f"  {'Name':<20} {'E1 (MPa)':>10} {'E2 (MPa)':>10} {'G12 (MPa)':>10} "
          f"{'Xc (MPa)':>10} {'gamma_Y':>8}")
    print("-" * 78)

    for name in names:
        mat = lib.get(name)
        print(
            f"  {mat.name:<20} {mat.E1:>10.0f} {mat.E2:>10.0f} {mat.G12:>10.0f} "
            f"{mat.Xc:>10.0f} {mat.gamma_Y:>8.4f}"
        )

    print("-" * 78)
    print(f"  Total: {len(names)} materials available")
    print()


# ====================================================================== #
# Entry point
# ====================================================================== #

def _cmd_converge(args: argparse.Namespace) -> None:
    """Handle the ``converge`` subcommand."""
    from wrinklefe.analysis import AnalysisConfig
    from wrinklefe.convergence import mesh_convergence_study
    from wrinklefe.core.material import MaterialLibrary

    material = None
    if args.material is not None:
        material = MaterialLibrary().get(args.material)

    config = AnalysisConfig(
        amplitude=args.amplitude,
        wavelength=args.wavelength,
        width=args.width,
        morphology=args.morphology,
        loading=args.loading,
        material=material,
        angles=_parse_angles(args.angles),
        nx=args.nx,
        ny=args.ny,
        nz_per_ply=args.nz_per_ply,
        applied_strain=args.strain,
    )

    refine = tuple(a.strip() for a in args.refine.split(",") if a.strip())
    try:
        study = mesh_convergence_study(
            config,
            levels=args.levels,
            refine=refine,
            qoi=args.qoi,
            tolerance=args.tolerance,
        )
    except (ValueError, NotImplementedError) as exc:
        print(f"error: convergence study failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(study.summary())

    if args.save_plot is not None:
        ax = study.plot()
        ax.figure.savefig(args.save_plot)
        print(f"\nConvergence plot saved to: {args.save_plot}")


def _configure_logging(verbose: bool) -> None:
    """Attach a stderr handler to the package logger for ``--verbose``.

    The library itself never configures handlers (standard logging
    etiquette); the CLI is the application, so handler setup lives here.
    Without ``--verbose`` the logger is left untouched and the default
    output is unchanged.
    """
    if not verbose:
        return
    pkg_logger = logging.getLogger("wrinklefe")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    pkg_logger.addHandler(handler)
    pkg_logger.setLevel(logging.DEBUG)


def main(argv: Sequence[str] | None = None) -> None:
    """Main CLI entry point.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments.  If ``None``, uses ``sys.argv[1:]``.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    _configure_logging(getattr(args, "verbose", False))

    handlers = {
        "analyze": _cmd_analyze,
        "compare": _cmd_compare,
        "sweep": _cmd_sweep,
        "converge": _cmd_converge,
        "materials": _cmd_materials,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
