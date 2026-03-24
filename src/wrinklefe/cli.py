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
        version="%(prog)s 0.1.0",
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
        help="Wrinkle amplitude A in mm (default: 0.366)",
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
        choices=["stack", "convex", "concave"],
        help="Wrinkle morphology type (default: stack)",
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
        "--angles", type=str, default=None,
        help=(
            "Ply angles as comma-separated values, e.g. '0,45,-45,90'. "
            "Default: quasi-isotropic [0/45/-45/90]_3s"
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
        "--buckling", action="store_true", default=False,
        help="Run linear buckling analysis",
    )
    p_analyze.add_argument(
        "--montecarlo", action="store_true", default=False,
        help="Run Monte Carlo simulation",
    )
    p_analyze.add_argument(
        "--mc-samples", type=int, default=5000,
        help="Number of Monte Carlo samples (default: 5000)",
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
        help="Wrinkle amplitude A in mm (default: 0.366)",
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
        "--analytical-only", action="store_true", default=True,
        help="Run analytical predictions only, no FE (default: True)",
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
        choices=["stack", "convex", "concave"],
        help="Morphology for the sweep (default: stack)",
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
    """Parse a comma-separated angle string into a list of floats."""
    if angles_str is None:
        return None
    return [float(a.strip()) for a in angles_str.split(",")]


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

    config = AnalysisConfig(
        amplitude=args.amplitude,
        wavelength=args.wavelength,
        width=args.width,
        morphology=args.morphology,
        loading=args.loading,
        material=material,
        angles=angles,
        nx=args.nx,
        ny=args.ny,
        applied_strain=args.strain,
        solver=args.solver,
        run_buckling=args.buckling,
        run_montecarlo=args.montecarlo,
        mc_samples=args.mc_samples,
        verbose=args.verbose,
    )

    analysis = WrinkleAnalysis(config)
    result = analysis.run_analytical_only()

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
        analytical_only=True,
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
