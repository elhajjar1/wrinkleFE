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
import json
import sys
from typing import List, Optional, Sequence

import numpy as np

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
        help="Wrinkle wavelength in mm (default: 16.0)",
    )
    p_analyze.add_argument(
        "--width", type=float, default=12.0,
        help="Gaussian envelope half-width in mm (default: 12.0)",
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
        "--verbose", action="store_true", default=False,
        help="Print detailed progress information",
    )
    p_analyze.add_argument(
        "--output-json", type=str, default=None,
        help="Export results to JSON file at specified path",
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
        help="Wrinkle wavelength in mm (default: 16.0)",
    )
    p_compare.add_argument(
        "--width", type=float, default=12.0,
        help="Gaussian envelope half-width in mm (default: 12.0)",
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
        "--verbose", action="store_true", default=False,
        help="Print detailed progress information",
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
        help="Parameter to sweep: amplitude, wavelength, or width",
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
        "--morphology", type=str, default="stack",
        choices=MORPHOLOGY_CHOICES,
        help=(
            "Morphology for the sweep (default: stack). "
            f"One of: {', '.join(MORPHOLOGY_CHOICES)}."
        ),
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
        "--verbose", action="store_true", default=False,
        help="Print detailed progress information",
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

def _parse_angles(angles_str: Optional[str]) -> Optional[List[float]]:
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
    #   1. --analytical-only -> skip FE
    #   2. --fe              -> full FE solve
    #   3. --no-fe           -> analytical-only
    #   4. default           -> full FE solve
    if args.analytical_only:
        analytical_only = True
    elif args.fe is True:
        analytical_only = False
    elif args.fe is False:
        analytical_only = True
    else:
        analytical_only = False

    config = AnalysisConfig(
        amplitude=args.amplitude,
        wavelength=args.wavelength,
        width=args.width,
        morphology=args.morphology,
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
    )

    analysis = WrinkleAnalysis(config)
    try:
        result = analysis.run(analytical_only=analytical_only)
    except Exception as exc:  # pragma: no cover - exercised via tests/CLI
        print(f"error: analysis failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(result.summary())

    # Export to JSON if requested
    if args.output_json is not None:
        from wrinklefe.io.export import export_results_json
        export_results_json(result, args.output_json)
        print(f"\nResults exported to: {args.output_json}")


def _cmd_compare(args: argparse.Namespace) -> None:
    """Handle the ``compare`` subcommand."""
    from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis

    config = AnalysisConfig(
        amplitude=args.amplitude,
        wavelength=args.wavelength,
        width=args.width,
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


def _cmd_sweep(args: argparse.Namespace) -> None:
    """Handle the ``sweep`` subcommand."""
    from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis

    values = np.linspace(args.sweep_min, args.sweep_max, args.steps)

    config = AnalysisConfig(
        morphology=args.morphology,
        verbose=args.verbose,
    )

    results = WrinkleAnalysis.parametric_sweep(
        config,
        parameter=args.parameter,
        values=values,
        analytical_only=args.analytical_only,
    )

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

def main(argv: Optional[Sequence[str]] = None) -> None:
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

    handlers = {
        "analyze": _cmd_analyze,
        "compare": _cmd_compare,
        "sweep": _cmd_sweep,
        "materials": _cmd_materials,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
