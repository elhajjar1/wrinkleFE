#!/usr/bin/env python3
"""
PARAMETRIC SWEEP FOR DUAL-WRINKLE ANALYSIS
============================================

Runs systematic parameter sweeps over wrinkle geometry parameters
(amplitude, wavelength, phase) using WrinkleAnalysis from the core module.

Usage:
    # Single-parameter sweep
    python parametric_sweep.py --sweep amplitude --range 0.183 0.549 5

    # Two-parameter sweep
    python parametric_sweep.py --sweep amplitude wavelength \\
        --range 0.183 0.549 3 --range 8.0 24.0 3

    # Fine mesh
    python parametric_sweep.py --sweep amplitude --range 0.183 0.549 5 --fine

Dependencies:
    numpy, matplotlib, wrinklefe core package
"""

import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis

# Valid sweep parameter names
VALID_PARAMS = {'amplitude', 'wavelength', 'phase'}

# Default parameter values (Jin et al. typical, 2A baseline).
# The value type is heterogeneous: geometry params are floats, ``phase`` is
# None-or-float, and callers also stash the swept ``morphology`` string in a
# copy of this dict — so annotate the union explicitly rather than letting
# mypy narrow it to ``float | None`` from the literal (issue #374).
DEFAULTS: dict[str, float | str | None] = {
    'amplitude': 0.366,   # mm (2A)
    'wavelength': 16.0,   # mm
    'width': 12.0,        # mm
    'phase': None,         # swept via morphology, not directly
    'applied_stress': -100.0,  # MPa
}

# Morphologies to analyze
MORPHOLOGIES = ['stack', 'convex', 'concave']


def validate_args(sweep_params, ranges):
    """Validate CLI arguments. Exits with clear error on failure."""
    # Check parameter names
    for p in sweep_params:
        if p not in VALID_PARAMS:
            print(f"Error: '{p}' is not a valid sweep parameter. "
                  f"Choose from: {', '.join(sorted(VALID_PARAMS))}")
            sys.exit(1)

    # Check for duplicates
    if len(sweep_params) != len(set(sweep_params)):
        print(f"Error: duplicate sweep parameters: {sweep_params}")
        sys.exit(1)

    # Check range count matches sweep count
    if len(ranges) != len(sweep_params):
        print(f"Error: {len(sweep_params)} sweep parameter(s) but "
              f"{len(ranges)} --range flag(s). Counts must match.")
        sys.exit(1)

    # Validate each range
    for i, r in enumerate(ranges):
        if len(r) != 3:
            print(f"Error: --range #{i+1} needs exactly 3 values (min max num_points), "
                  f"got {len(r)}")
            sys.exit(1)
        rmin, rmax, npts = r[0], r[1], int(r[2])
        if rmin >= rmax:
            print(f"Error: --range #{i+1} has min ({rmin}) >= max ({rmax})")
            sys.exit(1)
        if npts < 2:
            print(f"Error: --range #{i+1} has num_points ({npts}) < 2")
            sys.exit(1)


def _make_config(params, fine_mesh=False):
    """Build an AnalysisConfig from a parameter dict.

    A ``phase`` entry (when not ``None``) is plumbed through to
    ``AnalysisConfig.phase`` so phase sweeps actually change the
    dual-wrinkle geometry instead of being silently dropped (issue #49).
    """
    nx, ny = (50, 20) if fine_mesh else (30, 15)
    phase = params.get('phase')
    return AnalysisConfig(
        amplitude=params['amplitude'],
        wavelength=params['wavelength'],
        width=params.get('width', DEFAULTS['width']),
        morphology=params.get('morphology', 'stack'),
        phase=None if phase is None else float(phase),
        nx=nx,
        ny=ny,
    )


def _result_to_metrics(ar):
    """Extract scalar metrics dict from an AnalysisResults object."""
    return {
        'knockdown_factor': float(ar.analytical_knockdown),
        'failure_stress_MPa': float(ar.analytical_strength_MPa),
        'theta_max_rad': float(ar.max_angle_rad),
        'theta_max_deg': float(np.degrees(ar.max_angle_rad)),
        'theta_eff_rad': float(ar.effective_angle_rad),
        'theta_eff_deg': float(np.degrees(ar.effective_angle_rad)),
        'morphology_factor': float(ar.morphology_factor),
    }


def _evaluate_point(swept_params, param_vals, fine_mesh, is_phase_sweep):
    """Evaluate one grid point: the full set of solves for one parameter
    combination.

    Module-level (not a closure) so it pickles for
    ``ProcessPoolExecutor`` workers (issue #260). Only primitives cross
    the process boundary: the parameter tuple in, the scalar metrics
    dict out.
    """
    params = dict(DEFAULTS)
    for p, v in zip(swept_params, param_vals):
        params[p] = v

    if is_phase_sweep:
        # Phase sweep: the swept phase value is plumbed straight into
        # AnalysisConfig.phase (issue #49), which overrides the
        # named-morphology phase so each point uses a distinct
        # dual-wrinkle geometry instead of an identical 'stack' one.
        phase_val = params['phase']
        assert phase_val is not None, "phase sweep requires a phase value"
        phase = float(phase_val)
        params['morphology'] = 'stack'  # named phase is overridden anyway
        cfg = _make_config(params, fine_mesh)
        ar = WrinkleAnalysis(cfg).run()
        return {
            'custom': {
                **_result_to_metrics(ar),
                'phase_rad': float(phase),
                'phase_deg': float(np.degrees(phase)),
            }
        }
    # Standard sweep: run all 3 morphologies via WrinkleAnalysis
    point_results = {}
    for morph in MORPHOLOGIES:
        params['morphology'] = morph
        cfg = _make_config(params, fine_mesh)
        ar = WrinkleAnalysis(cfg).run()
        point_results[morph] = _result_to_metrics(ar)
    return point_results


def _resolve_n_workers(n_workers) -> int:
    """Validate and resolve the ``n_workers`` count (0 -> all cores)."""
    if not isinstance(n_workers, int) or isinstance(n_workers, bool):
        raise ValueError(f"n_workers must be an int >= 0, got {n_workers!r}")
    if n_workers < 0:
        raise ValueError(f"n_workers must be >= 0, got {n_workers}")
    if n_workers == 0:
        return os.cpu_count() or 1
    return n_workers


def run_sweep(sweep_config, fine_mesh=False, n_workers=1):
    """
    Run parametric sweep over wrinkle parameters.

    Parameters
    ----------
    sweep_config : dict
        Maps parameter name -> array of values to sweep.
        Example: {'amplitude': np.linspace(0.183, 0.549, 5)}
    fine_mesh : bool
        If True, use fine mesh (50x20). Default coarse (30x15).
    n_workers : int
        Number of worker processes for the per-point solves
        (issue #260). ``1`` (default) keeps the sequential in-process
        path; ``0`` uses all CPU cores; ``> 1`` fans the independent
        grid points out over a ``ProcessPoolExecutor``. Results are
        returned in grid order regardless. Note peak memory scales with
        ``n_workers`` x the per-solve footprint — size the worker count
        by available RAM for fine meshes.

    Returns
    -------
    dict : Sweep results with structure:
        {
            'swept_params': [...],
            'param_values': {param: [values]},
            'defaults': {param: value},
            'results': {param_key: {morph: {metrics}}},
            'elapsed_seconds': float
        }
    """
    n_workers = _resolve_n_workers(n_workers)
    swept_params = sorted(sweep_config.keys())
    is_phase_sweep = 'phase' in sweep_config

    # Build parameter grid
    param_arrays = [sweep_config[p] for p in swept_params]
    grid = list(itertools.product(*param_arrays))
    total = len(grid)

    print("\nParametric Sweep Configuration:")
    print(f"  Parameters: {', '.join(swept_params)}")
    for p in swept_params:
        vals = sweep_config[p]
        print(f"    {p}: {vals[0]:.4f} to {vals[-1]:.4f} ({len(vals)} points)")
    if is_phase_sweep:
        print("  Mode: phase sweep (single morphology per point, arbitrary phase)")
    else:
        print(f"  Total configurations: {total} x {len(MORPHOLOGIES)} morphologies"
              f" = {total * len(MORPHOLOGIES)} runs")
    if n_workers > 1:
        print(f"  Workers: {n_workers} processes")
    print()

    def _param_key(param_vals):
        if len(swept_params) == 1:
            return float(param_vals[0])
        return str(tuple(float(v) for v in param_vals))

    def _param_str(param_vals):
        return ', '.join(
            f'{p}={v:.4f}' for p, v in zip(swept_params, param_vals)
        )

    t_start = time.time()

    if n_workers == 1:
        # Sequential path — unchanged behaviour and output.
        results = {}
        for idx, param_vals in enumerate(grid):
            t_point = time.time()
            print(f"  [{idx+1}/{total}] {_param_str(param_vals)} ...",
                  end='', flush=True)
            results[_param_key(param_vals)] = _evaluate_point(
                swept_params, param_vals, fine_mesh, is_phase_sweep
            )
            print(f" done ({time.time() - t_point:.1f}s)")
    else:
        # Parallel path: every grid point is an independent analysis
        # (separate mesh, separate solve, no shared state), so fan them
        # out over processes. Progress is completion-based; the results
        # dict is assembled in grid order afterwards so ordering is
        # deterministic and identical to the sequential path.
        point_results: list = [None] * total
        executor = ProcessPoolExecutor(max_workers=n_workers)
        try:
            future_to_idx = {
                executor.submit(
                    _evaluate_point, swept_params, param_vals,
                    fine_mesh, is_phase_sweep,
                ): idx
                for idx, param_vals in enumerate(grid)
            }
            n_done = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                point_results[idx] = future.result()
                n_done += 1
                print(f"  [{n_done}/{total} done] {_param_str(grid[idx])}",
                      flush=True)
        except BaseException:
            # KeyboardInterrupt (or a worker failure): cancel everything
            # still queued instead of letting the pool drain.
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
        results = {
            _param_key(param_vals): point_results[idx]
            for idx, param_vals in enumerate(grid)
        }

    total_elapsed = time.time() - t_start
    print(f"\nSweep complete: {total} points in {total_elapsed:.1f}s")

    return {
        'swept_params': swept_params,
        'param_values': {p: [float(v) for v in sweep_config[p]] for p in swept_params},
        'defaults': {k: v for k, v in DEFAULTS.items() if k not in swept_params and v is not None},
        'results': results,
        'elapsed_seconds': total_elapsed,
    }


def save_sweep_results(sweep_results, output_dir):
    """Save sweep results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    param_name = '_'.join(sweep_results['swept_params'])
    filepath = os.path.join(output_dir, f'sweep_results_{param_name}.json')

    # Convert keys to strings for JSON
    serializable = dict(sweep_results)
    serializable['results'] = {str(k): v for k, v in sweep_results['results'].items()}

    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"Saved: {filepath}")
    return filepath


def plot_sweep_results(sweep_results, output_dir='./sweep_output/'):
    """
    Generate plots for sweep results.

    1D sweeps: line plots (knockdown and angle vs parameter).
    2D sweeps: heatmaps (knockdown per morphology).
    """
    os.makedirs(output_dir, exist_ok=True)
    swept = sweep_results['swept_params']

    if len(swept) == 1:
        _plot_1d_sweep(sweep_results, output_dir)
    elif len(swept) == 2:
        _plot_2d_sweep(sweep_results, output_dir)
    else:
        print(f"Warning: no plotting support for {len(swept)}-parameter sweeps")


def _plot_1d_sweep(sweep_results, output_dir):
    """Generate 1D sweep plots."""
    param = sweep_results['swept_params'][0]
    values = sweep_results['param_values'][param]
    results = sweep_results['results']

    first_result = results[values[0]]
    morphs = list(first_result.keys())

    colors = {'stack': '#666666', 'convex': '#2196F3', 'concave': '#F44336', 'custom': '#9C27B0'}
    markers = {'stack': 's', 'convex': '^', 'concave': 'v', 'custom': 'o'}

    # Plot 1: Knockdown factor vs parameter
    fig, ax = plt.subplots(figsize=(8, 5))
    for morph in morphs:
        knockdowns = [results[v][morph]['knockdown_factor'] for v in values]
        label = 'Phase sweep' if morph == 'custom' else morph.capitalize()
        ax.plot(values, knockdowns, '-o', color=colors[morph], marker=markers[morph],
                label=label, linewidth=2, markersize=8)

    ax.set_xlabel(_param_label(param), fontsize=12, fontweight='bold')
    ax.set_ylabel('Knockdown Factor', fontsize=12, fontweight='bold')
    ax.set_title(f'Compression Strength Knockdown vs {param.capitalize()}',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    path = os.path.join(output_dir, f'sweep_{param}_knockdown.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")

    # Plot 2: Effective angle vs parameter
    fig, ax = plt.subplots(figsize=(8, 5))
    for morph in morphs:
        angles = [results[v][morph]['theta_eff_deg'] for v in values]
        label = 'Phase sweep' if morph == 'custom' else morph.capitalize()
        ax.plot(values, angles, '-o', color=colors[morph], marker=markers[morph],
                label=label, linewidth=2, markersize=8)

    ax.set_xlabel(_param_label(param), fontsize=12, fontweight='bold')
    ax.set_ylabel('Effective Angle (deg)', fontsize=12, fontweight='bold')
    ax.set_title(f'Effective Fiber Angle vs {param.capitalize()}',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    path = os.path.join(output_dir, f'sweep_{param}_angle.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def _plot_2d_sweep(sweep_results, output_dir):
    """Generate 2D sweep heatmaps."""
    params = sweep_results['swept_params']
    p1_vals = np.array(sweep_results['param_values'][params[0]])
    p2_vals = np.array(sweep_results['param_values'][params[1]])
    results = sweep_results['results']

    for morph in MORPHOLOGIES:
        fig, ax = plt.subplots(figsize=(8, 6))

        # Build 2D grid of knockdown values
        grid = np.zeros((len(p2_vals), len(p1_vals)))
        for i, v1 in enumerate(p1_vals):
            for j, v2 in enumerate(p2_vals):
                key = str((float(v1), float(v2)))
                grid[j, i] = results[key][morph]['knockdown_factor']

        im = ax.pcolormesh(p1_vals, p2_vals, grid, cmap='plasma', shading='auto')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Knockdown Factor', fontsize=11)

        ax.set_xlabel(_param_label(params[0]), fontsize=12, fontweight='bold')
        ax.set_ylabel(_param_label(params[1]), fontsize=12, fontweight='bold')
        ax.set_title(f'{morph.capitalize()} - Knockdown Factor',
                    fontsize=13, fontweight='bold')
        ax.tick_params(labelsize=11)

        path = os.path.join(output_dir, f'sweep_{params[0]}_{params[1]}_knockdown_{morph}.png')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {path}")


def _param_label(param):
    """Human-readable axis label for a parameter."""
    labels = {
        'amplitude': 'Amplitude A (mm)',
        'wavelength': 'Wavelength lambda (mm)',
        'phase': 'Phase Offset phi (rad)',
    }
    return labels.get(param, param)


def main():
    parser = argparse.ArgumentParser(
        description='Parametric sweep for dual-wrinkle analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python parametric_sweep.py --sweep amplitude --range 0.183 0.549 5
  python parametric_sweep.py --sweep wavelength --range 8.0 24.0 5
  python parametric_sweep.py --sweep amplitude wavelength --range 0.183 0.549 3 --range 8.0 24.0 3
  python parametric_sweep.py --sweep amplitude --range 0.183 0.549 5 --fine
        """
    )
    parser.add_argument('--sweep', nargs='+', required=True,
                       help='Parameter(s) to sweep: amplitude, wavelength, phase')
    parser.add_argument('--range', nargs=3, action='append', dest='ranges',
                       type=float, metavar=('MIN', 'MAX', 'NUM_POINTS'),
                       help='Range for each swept parameter: min max num_points')
    parser.add_argument('--fine', action='store_true',
                       help='Use fine mesh (50x20) instead of coarse (30x15)')
    parser.add_argument('--output-dir', default='./sweep_output/',
                       help='Output directory for results and plots')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation, JSON only')

    args = parser.parse_args()

    if args.ranges is None:
        parser.error("--range is required for each --sweep parameter")

    validate_args(args.sweep, args.ranges)

    # Build sweep config
    sweep_config = {}
    for param, r in zip(args.sweep, args.ranges):
        rmin, rmax, npts = r[0], r[1], int(r[2])
        sweep_config[param] = np.linspace(rmin, rmax, npts)

    # Run sweep
    results = run_sweep(sweep_config, fine_mesh=args.fine)

    # Save JSON
    save_sweep_results(results, args.output_dir)

    # Generate plots
    if not args.no_plots:
        plot_sweep_results(results, args.output_dir)

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
