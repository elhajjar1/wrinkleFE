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

import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
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

ACCENT_GRAY: str = "#333333"
"""Neutral dark gray accent color for non-morphology emphasis elements.

Used for overall/aggregate series in plots where a morphology color would
be misleading (e.g. the ``Overall`` bar in a Jensen-gap bar chart that sits
alongside per-morphology bars).
"""


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


def save_figure(
    ax_or_fig: Union[Axes, Figure],
    path: Union[str, "os.PathLike[str]"],
    *,
    close: bool = True,
    dpi: int = 300,
    **savefig_kwargs: Any,
) -> None:
    """Save the figure backing ``ax_or_fig`` to ``path`` and (by default) close it.

    This is the recommended way to persist plots produced by the ``plot_*``
    functions in batch / sweep / CLI / headless code. Those functions return an
    :class:`~matplotlib.axes.Axes` whose parent :class:`~matplotlib.figure.Figure`
    is created internally by :func:`ensure_axes`; if such loops never call
    :func:`matplotlib.pyplot.close`, the figures accumulate (memory leak and the
    ``More than 20 figures have been opened`` warning).

    Parameters
    ----------
    ax_or_fig : Axes or Figure
        The object returned by a ``plot_*`` function (an ``Axes``), or a
        ``Figure`` directly.
    path : str or os.PathLike
        Destination file path for :meth:`~matplotlib.figure.Figure.savefig`.
    close : bool, optional
        If ``True`` (default), close the figure after writing so it does not
        leak. Pass ``False`` only if the caller still needs the live figure
        (e.g. to display it interactively afterwards).
    dpi : int, optional
        Resolution passed to ``savefig``. Default is 300.
    **savefig_kwargs
        Extra keyword arguments forwarded to
        :meth:`~matplotlib.figure.Figure.savefig` (e.g. ``bbox_inches='tight'``).
    """
    fig = ax_or_fig if isinstance(ax_or_fig, Figure) else ax_or_fig.figure
    savefig_kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(path, dpi=dpi, **savefig_kwargs)
    if close:
        plt.close(fig)


@contextmanager
def figure_context(ax_or_fig: Union[Axes, Figure]) -> Iterator[Figure]:
    """Context manager that closes an internally-created figure on exit.

    Use this to wrap a ``plot_*`` call in leak-prone batch loops when you are
    not going through :func:`save_figure`::

        for case in cases:
            with figure_context(plot_wrinkle_profile(case.profile)) as fig:
                fig.savefig(case.out_path)
        # every figure is closed; plt.get_fignums() does not grow

    Interactive / Streamlit callers should simply not use this helper (and not
    pass ``close=True`` to :func:`save_figure`); the figure they receive stays
    alive exactly as before.

    Parameters
    ----------
    ax_or_fig : Axes or Figure
        The object returned by a ``plot_*`` function, or a ``Figure``.

    Yields
    ------
    Figure
        The parent figure, guaranteed closed when the ``with`` block exits.
    """
    fig = ax_or_fig if isinstance(ax_or_fig, Figure) else ax_or_fig.figure
    try:
        yield fig
    finally:
        plt.close(fig)
