"""Publication-quality plot styling for wrinkle analysis figures.

Provides a consistent visual style for journal-quality figures across all
visualization functions. Configures matplotlib for high-resolution output
with appropriate font sizes, tick parameters, and frame styling.

The color scheme uses perceptually distinct colors for the three canonical
dual-wrinkle morphologies (stack, convex, concave) and is designed to be
distinguishable in both color and grayscale print.

References
----------
Jin, L. et al. (2026). Thin-Walled Structures, 219, 114237.
Elhajjar, R. (2025). Scientific Reports, 15, 25977.
"""

from __future__ import annotations

from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.image import AxesImage


# ======================================================================
# Morphology visual encoding
# ======================================================================

MORPHOLOGY_COLORS: dict[str, str] = {
    "stack": "#1f77b4",      # muted blue
    "convex": "#2ca02c",     # muted green
    "concave": "#d62728",    # muted red
    "anti-stack": "#9467bd", # muted purple
}
"""Color mapping for morphology types.

Stack (blue) is the baseline, convex (green) is the favorable morphology,
and concave (red) is the adverse morphology. Anti-stack (purple) is used
in Monte Carlo results where a fourth classification is present.
"""

MORPHOLOGY_MARKERS: dict[str, str] = {
    "stack": "o",      # circle
    "convex": "^",     # triangle up
    "concave": "v",    # triangle down
    "anti-stack": "s", # square
}
"""Scatter plot markers for morphology types."""

MORPHOLOGY_LINESTYLES: dict[str, str] = {
    "stack": "-",       # solid
    "convex": "--",     # dashed
    "concave": "-.",    # dash-dot
    "anti-stack": ":",  # dotted
}
"""Line styles for morphology types."""

MORPHOLOGY_LABELS: dict[str, str] = {
    "stack": "Stack ($\\phi=0$)",
    "convex": "Convex ($\\phi=\\pi/2$)",
    "concave": "Concave ($\\phi=-\\pi/2$)",
    "anti-stack": "Anti-stack ($\\phi=\\pi$)",
}
"""Display labels for morphology types with phase offset notation."""


# ======================================================================
# Standard figure sizes
# ======================================================================

FIGSIZE_SINGLE_COLUMN: tuple[float, float] = (3.5, 2.8)
"""Single-column figure size (inches) for typical journal format (~90 mm)."""

FIGSIZE_DOUBLE_COLUMN: tuple[float, float] = (7.0, 4.5)
"""Double-column figure size (inches) for typical journal format (~180 mm)."""

FIGSIZE_SINGLE_TALL: tuple[float, float] = (3.5, 4.5)
"""Single-column tall figure for through-thickness or multi-panel plots."""

FIGSIZE_DOUBLE_WIDE: tuple[float, float] = (7.0, 2.8)
"""Double-column wide figure for panoramic plots."""


# ======================================================================
# Publication style configuration
# ======================================================================

def set_publication_style() -> None:
    """Configure matplotlib for publication-quality figures.

    Sets font family, sizes, tick parameters, frame styles, and line widths
    consistent with typical journal requirements. Call this once before
    creating any figures.

    The style uses:

    - Serif font family for text (matches LaTeX documents)
    - 9 pt base font size (suitable for single-column figures)
    - Outward-facing ticks on all axes
    - Thicker axis frames for visibility
    - High-resolution savefig defaults (300 DPI, tight bounding box)
    """
    params = {
        # Font
        "font.family": "serif",
        "font.size": 9,
        "mathtext.fontset": "dejavuserif",

        # Axes
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "axes.linewidth": 0.8,
        "axes.labelpad": 4,
        "axes.formatter.use_mathtext": True,

        # Ticks
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.direction": "out",
        "ytick.direction": "out",

        # Legend
        "legend.fontsize": 8,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",

        # Lines
        "lines.linewidth": 1.5,
        "lines.markersize": 5,

        # Figure
        "figure.dpi": 150,
        "figure.constrained_layout.use": True,

        # Savefig
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
    mpl.rcParams.update(params)


# ======================================================================
# Helper utilities
# ======================================================================

def colorbar_setup(
    ax: Axes,
    mappable: AxesImage,
    label: str,
    orientation: str = "vertical",
) -> Colorbar:
    """Add a well-formatted colorbar to an axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes containing the mappable.
    mappable : matplotlib.image.AxesImage
        The image or contour set to create the colorbar for.
    label : str
        Colorbar label text.
    orientation : str, optional
        ``'vertical'`` (default) or ``'horizontal'``.

    Returns
    -------
    matplotlib.colorbar.Colorbar
        The created colorbar instance.
    """
    fig = ax.get_figure()
    cbar = fig.colorbar(mappable, ax=ax, orientation=orientation, pad=0.02)
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    return cbar


def get_morphology_style(morph: str) -> dict:
    """Return a dictionary of plot keyword arguments for a given morphology.

    Parameters
    ----------
    morph : str
        Morphology name (``'stack'``, ``'convex'``, ``'concave'``,
        or ``'anti-stack'``).

    Returns
    -------
    dict
        Keyword arguments suitable for ``ax.plot()`` or ``ax.scatter()``
        including color, marker, linestyle, and label.
    """
    return {
        "color": MORPHOLOGY_COLORS.get(morph, "gray"),
        "marker": MORPHOLOGY_MARKERS.get(morph, "o"),
        "linestyle": MORPHOLOGY_LINESTYLES.get(morph, "-"),
        "label": MORPHOLOGY_LABELS.get(morph, morph),
    }


def ensure_axes(
    ax: Optional[Axes] = None,
    figsize: tuple[float, float] = FIGSIZE_SINGLE_COLUMN,
    projection: Optional[str] = None,
) -> Axes:
    """Return the given Axes or create a new figure with one.

    Parameters
    ----------
    ax : Axes or None
        If provided, return it directly. If ``None``, create a new figure.
    figsize : tuple[float, float]
        Figure size (only used when creating a new figure).
    projection : str or None
        Axes projection (e.g. ``'3d'``). Only used when creating new axes.

    Returns
    -------
    Axes
        The matplotlib Axes to draw on.
    """
    if ax is not None:
        return ax
    fig = plt.figure(figsize=figsize)
    if projection is not None:
        return fig.add_subplot(111, projection=projection)
    return fig.add_subplot(111)
