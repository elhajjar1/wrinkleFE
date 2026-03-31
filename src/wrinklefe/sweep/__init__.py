"""Parametric sweep module for dual-wrinkle analysis.

Provides systematic parameter sweeps over wrinkle geometry parameters
(amplitude, wavelength, phase) using the core WrinkleAnalysis pipeline.

Usage::

    from wrinklefe.sweep import run_sweep, plot_sweep_results
    results = run_sweep({'amplitude': np.linspace(0.183, 0.549, 5)})
"""

from wrinklefe.sweep.parametric_sweep import (
    run_sweep,
    plot_sweep_results,
    save_sweep_results,
)

__all__ = [
    "run_sweep",
    "plot_sweep_results",
    "save_sweep_results",
]
