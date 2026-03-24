"""Visualization: 2D plots (matplotlib) and 3D views (matplotlib mplot3d).

This package provides publication-quality plotting functions for wrinkle
analysis results. All functions use matplotlib only (no PyVista dependency).

Submodules
----------
style
    Publication styling, morphology color/marker/linestyle dictionaries,
    and helper utilities.
plots_2d
    2D plotting functions for profiles, morphology factors, distributions,
    Jensen gap, failure envelopes, through-thickness stress, and damage.
plots_3d
    3D plotting functions for mesh wireframes, displacement contours,
    stress contours, and buckling mode shapes.
"""

# -- Style configuration and helpers --
from wrinklefe.viz.style import (
    FIGSIZE_DOUBLE_COLUMN,
    FIGSIZE_DOUBLE_WIDE,
    FIGSIZE_SINGLE_COLUMN,
    FIGSIZE_SINGLE_TALL,
    MORPHOLOGY_COLORS,
    MORPHOLOGY_LABELS,
    MORPHOLOGY_LINESTYLES,
    MORPHOLOGY_MARKERS,
    colorbar_setup,
    ensure_axes,
    get_morphology_style,
    set_publication_style,
)

# -- 2D plotting functions --
from wrinklefe.viz.plots_2d import (
    plot_damage_contour,
    plot_dual_wrinkle_profiles,
    plot_failure_envelope,
    plot_jensen_gap,
    plot_kinkband_concavity,
    plot_morphology_factor,
    plot_strength_distribution,
    plot_strength_vs_amplitude,
    plot_stress_through_thickness,
    plot_wrinkle_profile,
)

# -- 3D plotting functions --
from wrinklefe.viz.plots_3d import (
    plot_buckling_mode,
    plot_displacement_3d,
    plot_mesh_3d,
    plot_stress_contour_3d,
)

__all__ = [
    # Style
    "set_publication_style",
    "MORPHOLOGY_COLORS",
    "MORPHOLOGY_MARKERS",
    "MORPHOLOGY_LINESTYLES",
    "MORPHOLOGY_LABELS",
    "FIGSIZE_SINGLE_COLUMN",
    "FIGSIZE_DOUBLE_COLUMN",
    "FIGSIZE_SINGLE_TALL",
    "FIGSIZE_DOUBLE_WIDE",
    "colorbar_setup",
    "ensure_axes",
    "get_morphology_style",
    # 2D plots
    "plot_wrinkle_profile",
    "plot_dual_wrinkle_profiles",
    "plot_morphology_factor",
    "plot_kinkband_concavity",
    "plot_strength_vs_amplitude",
    "plot_strength_distribution",
    "plot_jensen_gap",
    "plot_failure_envelope",
    "plot_stress_through_thickness",
    "plot_damage_contour",
    # 3D plots
    "plot_mesh_3d",
    "plot_displacement_3d",
    "plot_stress_contour_3d",
    "plot_buckling_mode",
]
