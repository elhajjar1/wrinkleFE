"""Visualization: 2D plots (matplotlib) and 3D views (matplotlib mplot3d / PyVista).

This package provides publication-quality plotting functions for wrinkle
analysis results.  Matplotlib is the primary 3D backend.  A small group of
cohesive-zone-modelling plots in :mod:`plots_3d` uses PyVista internally
(imported lazily) for high-quality interface rendering.

Submodules
----------
style
    Publication styling, morphology color/marker/linestyle dictionaries,
    and helper utilities.
plots_2d
    2D plotting functions for profiles, morphology factors, distributions,
    Jensen gap, failure envelopes, through-thickness stress, damage, and
    cohesive-zone-modelling (CZM) outputs.
plots_3d
    3D plotting functions for mesh wireframes, displacement contours,
    stress contours, buckling mode shapes, and PyVista-backed CZM
    interface visualizations.
"""

# -- Style configuration and helpers --
from wrinklefe.viz.style import (
    ACCENT_GRAY,
    FIGSIZE_DOUBLE_COLUMN,
    FIGSIZE_DOUBLE_WIDE,
    FIGSIZE_SINGLE_COLUMN,
    FIGSIZE_SINGLE_TALL,
    MORPHOLOGY_COLORS,
    MORPHOLOGY_LABELS,
    MORPHOLOGY_LINESTYLES,
    MORPHOLOGY_MARKERS,
    TENSION_MECHANISM_COLORS,
    colorbar_setup,
    ensure_axes,
    figure_context,
    get_morphology_style,
    save_figure,
    set_axes_equal_aspect,
    set_publication_style,
)

# -- 2D plotting functions --
from wrinklefe.viz.plots_2d import (
    czm_overview_figure,
    plot_damage_contour,
    plot_damage_histogram,
    plot_dual_wrinkle_profiles,
    plot_energy_per_interface,
    plot_failure_envelope,
    plot_interface_damage_field,
    plot_jensen_gap,
    plot_kinkband_concavity,
    plot_load_displacement,
    plot_morphology_factor,
    plot_strength_distribution,
    plot_strength_vs_amplitude,
    plot_stress_through_thickness,
    plot_traction_separation,
    plot_wrinkle_profile,
)

# -- 3D plotting functions --
from wrinklefe.viz.plots_3d import (
    plot_buckling_mode,
    plot_crack_front_3d,
    plot_displacement_3d,
    plot_interface_damage_3d,
    plot_mesh_3d,
    plot_stress_contour_3d,
)

__all__ = [
    # Style
    "set_publication_style",
    "ACCENT_GRAY",
    "MORPHOLOGY_COLORS",
    "MORPHOLOGY_MARKERS",
    "MORPHOLOGY_LINESTYLES",
    "MORPHOLOGY_LABELS",
    "TENSION_MECHANISM_COLORS",
    "FIGSIZE_SINGLE_COLUMN",
    "FIGSIZE_DOUBLE_COLUMN",
    "FIGSIZE_SINGLE_TALL",
    "FIGSIZE_DOUBLE_WIDE",
    "colorbar_setup",
    "ensure_axes",
    "save_figure",
    "figure_context",
    "get_morphology_style",
    "set_axes_equal_aspect",
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
    # 2D plots — CZM
    "plot_traction_separation",
    "plot_load_displacement",
    "plot_damage_histogram",
    "plot_interface_damage_field",
    "plot_energy_per_interface",
    "czm_overview_figure",
    # 3D plots
    "plot_mesh_3d",
    "plot_displacement_3d",
    "plot_stress_contour_3d",
    "plot_buckling_mode",
    # 3D plots — CZM (PyVista)
    "plot_interface_damage_3d",
    "plot_crack_front_3d",
]
