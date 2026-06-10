"""2D plotting functions for wrinkle analysis and post-processing.

Provides publication-quality 2D visualizations including wrinkle profiles,
morphology factor sweeps, kink-band concavity diagrams, strength
distributions, Jensen gap charts, failure envelopes, through-thickness
stress profiles, and damage contour plots.

All plot functions follow a consistent interface:

- Accept an optional ``ax`` parameter; if ``None``, a new figure is created.
- Return the :class:`~matplotlib.axes.Axes` object for further customization.
- Apply publication styling from :mod:`wrinklefe.viz.style`.

.. note::
   When ``ax=None`` the returned ``Axes`` has a *new* parent ``Figure`` that
   the caller owns. Interactive / Streamlit callers keep it (e.g.
   ``st.pyplot(ax.figure)``). Batch / sweep / headless code that calls these
   in a loop **must** close the figure to avoid leaking it; use
   :func:`wrinklefe.viz.save_figure` (saves then closes by default) or
   :func:`wrinklefe.viz.figure_context`.

References
----------
Jin, L. et al. (2026). Thin-Walled Structures, 219, 114237.
Elhajjar, R. (2025). Scientific Reports, 15, 25977.
Budiansky, B. & Fleck, N.A. (1993). J. Mech. Phys. Solids, 41(1), 183-211.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from wrinklefe.viz.style import (
    ACCENT_GRAY,
    FIGSIZE_DOUBLE_COLUMN,
    FIGSIZE_SINGLE_COLUMN,
    FIGSIZE_SINGLE_TALL,
    MORPHOLOGY_COLORS,
    MORPHOLOGY_LABELS,
    MORPHOLOGY_LINESTYLES,
    colorbar_setup,
    ensure_axes,
    get_morphology_style,
    set_publication_style,
)

if TYPE_CHECKING:
    from wrinklefe.analysis import AnalysisResults
    from wrinklefe.core.mesh import MeshData
    from wrinklefe.core.morphology import WrinkleConfiguration
    from wrinklefe.core.wrinkle import WrinkleProfile
    from wrinklefe.solver.results import FieldResults
    from wrinklefe.statistics.jensen import JensenGapResult
    from wrinklefe.statistics.montecarlo import MonteCarloResults


# ======================================================================
# Wrinkle geometry plots
# ======================================================================

def plot_wrinkle_profile(
    profile: "WrinkleProfile",
    ax: Optional[Axes] = None,
    n_points: int = 500,
) -> Axes:
    """Plot the out-of-plane displacement z(x) of a single wrinkle profile.

    Parameters
    ----------
    profile : WrinkleProfile
        Wrinkle profile instance providing ``displacement(x)`` and ``domain()``.
    ax : Axes, optional
        Matplotlib axes. A new figure is created if ``None``.
    n_points : int, optional
        Number of evaluation points along x. Default is 500.

    Returns
    -------
    Axes
        The axes with the wrinkle profile plotted.
    """
    set_publication_style()
    ax = ensure_axes(ax)

    xlo, xhi = profile.domain()
    x = np.linspace(xlo, xhi, n_points)
    z = profile.displacement(x)

    ax.plot(x, z, color=MORPHOLOGY_COLORS["stack"], linewidth=1.5)
    ax.axhline(0, color="0.6", linewidth=0.5, zorder=0)
    ax.set_xlabel("$x$ (mm)")
    ax.set_ylabel("$z$ (mm)")
    ax.set_title("Wrinkle Profile $z(x)$")

    return ax


def plot_dual_wrinkle_profiles(
    config: "WrinkleConfiguration",
    ax: Optional[Axes] = None,
    n_points: int = 500,
    show_gap: bool = True,
) -> Axes:
    """Plot upper and lower wrinkle profiles from a dual-wrinkle configuration.

    For a configuration with two wrinkles, the first is labeled "Upper"
    and the second "Lower". If ``show_gap`` is True, the interface gap
    (difference between profiles) is shown as a shaded region.

    Parameters
    ----------
    config : WrinkleConfiguration
        Wrinkle configuration with at least two wrinkles.
    ax : Axes, optional
        Matplotlib axes. A new figure is created if ``None``.
    n_points : int, optional
        Number of evaluation points along x. Default is 500.
    show_gap : bool, optional
        Whether to shade the gap between profiles. Default is True.

    Returns
    -------
    Axes
        The axes with dual wrinkle profiles plotted.
    """
    set_publication_style()
    ax = ensure_axes(ax, figsize=FIGSIZE_DOUBLE_COLUMN)

    if config.n_wrinkles() < 2:
        # Fall back to single profile
        wrinkle = config.wrinkles[0]
        profile = wrinkle.profile
        xlo, xhi = profile.domain()
        x = np.linspace(xlo, xhi, n_points)
        z = profile.displacement(x)
        ax.plot(x, z, label="Wrinkle 1")
        ax.set_xlabel("$x$ (mm)")
        ax.set_ylabel("$z$ (mm)")
        ax.set_title("Wrinkle Profile")
        ax.legend()
        return ax

    w_upper = config.wrinkles[0]
    w_lower = config.wrinkles[1]
    p_upper = w_upper.profile
    p_lower = w_lower.profile

    # Use union of domains
    xlo_u, xhi_u = p_upper.domain()
    xlo_l, xhi_l = p_lower.domain()
    xlo = min(xlo_u, xlo_l)
    xhi = max(xhi_u, xhi_l)
    x = np.linspace(xlo, xhi, n_points)

    z_upper = p_upper.displacement(x)
    z_lower = p_lower.displacement(x)

    ax.plot(x, z_upper, color=MORPHOLOGY_COLORS["stack"], linewidth=1.5, label="Upper wrinkle")
    ax.plot(x, z_lower, color=MORPHOLOGY_COLORS["concave"], linewidth=1.5, label="Lower wrinkle")

    if show_gap:
        gap = z_upper - z_lower
        ax.fill_between(
            x, z_lower, z_upper,
            alpha=0.15, color=MORPHOLOGY_COLORS["convex"], label="Interface gap",
        )

    ax.axhline(0, color="0.6", linewidth=0.5, zorder=0)
    ax.set_xlabel("$x$ (mm)")
    ax.set_ylabel("$z$ (mm)")
    ax.set_title("Dual-Wrinkle Profiles")
    ax.legend(loc="upper right", fontsize=8)

    return ax


# ======================================================================
# Morphology and mechanics plots
# ======================================================================

def plot_morphology_factor(
    loading: str = "compression",
    ax: Optional[Axes] = None,
    n_points: int = 361,
) -> Axes:
    """Plot morphology factor M_f as a function of phase offset phi.

    Sweeps phi from -pi to pi and evaluates the analytical morphology
    factor for the specified loading mode. Vertical dashed lines mark
    the three canonical morphologies (stack, convex, concave).

    Parameters
    ----------
    loading : str, optional
        Loading mode: ``'compression'`` (default) or ``'tension'``.
    ax : Axes, optional
        Matplotlib axes. A new figure is created if ``None``.
    n_points : int, optional
        Number of phi evaluation points. Default is 361.

    Returns
    -------
    Axes
        The axes with the morphology factor curve.
    """
    from wrinklefe.core.morphology import MORPHOLOGY_PHASES, WrinkleConfiguration

    set_publication_style()
    ax = ensure_axes(ax)

    phi = np.linspace(-np.pi, np.pi, n_points)
    mf = np.array([
        WrinkleConfiguration.morphology_factor_analytical(p, loading)
        for p in phi
    ])

    ax.plot(phi, mf, color="k", linewidth=1.5)

    # Mark canonical morphologies
    for morph, phase in MORPHOLOGY_PHASES.items():
        mf_val = WrinkleConfiguration.morphology_factor_analytical(phase, loading)
        color = MORPHOLOGY_COLORS.get(morph, "gray")
        label = MORPHOLOGY_LABELS.get(morph, morph)
        ax.axvline(phase, color=color, linestyle="--", linewidth=0.8, alpha=0.7)
        ax.plot(phase, mf_val, "o", color=color, markersize=7, label=label)

    ax.set_xlabel("Phase offset $\\phi$ (rad)")
    ax.set_ylabel("Morphology factor $M_f$")
    ax.set_title(f"Morphology Factor ({loading.capitalize()})")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels(
        ["$-\\pi$", "$-\\pi/2$", "$0$", "$\\pi/2$", "$\\pi$"]
    )
    ax.legend(loc="best", fontsize=7)
    ax.axhline(1.0, color="0.7", linewidth=0.5, linestyle=":", zorder=0)

    return ax


def plot_kinkband_concavity(
    gamma_Y: float = 0.02,
    ax: Optional[Axes] = None,
    theta_max_deg: float = 25.0,
    n_points: int = 500,
) -> Axes:
    """Plot the Budiansky-Fleck kink-band knockdown showing concavity.

    Illustrates why the kink-band function creates fat-tailed distributions:
    the knockdown KD = 1/(1 + theta/gamma_Y) is concave in theta, so by
    Jensen's inequality E[KD(theta)] < KD(E[theta]).

    Parameters
    ----------
    gamma_Y : float, optional
        Matrix yield shear strain. Default is 0.02.
    ax : Axes, optional
        Matplotlib axes. A new figure is created if ``None``.
    theta_max_deg : float, optional
        Maximum theta to plot (degrees). Default is 25.
    n_points : int, optional
        Number of evaluation points. Default is 500.

    Returns
    -------
    Axes
        The axes with the concavity diagram.
    """
    set_publication_style()
    ax = ensure_axes(ax)

    theta_deg = np.linspace(0, theta_max_deg, n_points)
    theta_rad = np.radians(theta_deg)
    kd = 1.0 / (1.0 + theta_rad / gamma_Y)

    ax.plot(theta_deg, kd, color="k", linewidth=1.5)

    # Highlight concavity with a chord
    t1_deg, t2_deg = 5.0, 15.0
    t1_rad, t2_rad = np.radians(t1_deg), np.radians(t2_deg)
    kd1 = 1.0 / (1.0 + t1_rad / gamma_Y)
    kd2 = 1.0 / (1.0 + t2_rad / gamma_Y)
    ax.plot([t1_deg, t2_deg], [kd1, kd2], "r--", linewidth=1.0, label="Chord (linear interp.)")

    # Midpoint comparison
    t_mid_deg = (t1_deg + t2_deg) / 2.0
    t_mid_rad = np.radians(t_mid_deg)
    kd_mid_curve = 1.0 / (1.0 + t_mid_rad / gamma_Y)
    kd_mid_chord = (kd1 + kd2) / 2.0

    ax.plot(t_mid_deg, kd_mid_curve, "go", markersize=6, label="$f(E[\\theta])$")
    ax.plot(t_mid_deg, kd_mid_chord, "rs", markersize=6, label="$E[f(\\theta)]$")

    # Jensen gap annotation
    gap = kd_mid_curve - kd_mid_chord
    ax.annotate(
        f"Jensen gap\n$\\Delta = {gap:.3f}$",
        xy=(t_mid_deg, kd_mid_chord),
        xytext=(t_mid_deg + 3, kd_mid_chord + 0.05),
        fontsize=7,
        arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8),
    )

    ax.set_xlabel("Fiber misalignment angle $\\theta$ (deg)")
    ax.set_ylabel("Knockdown factor $KD$")
    ax.set_title("Kink-Band Concavity (Jensen Inequality)")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlim(0, theta_max_deg)
    ax.set_ylim(0, 1.05)

    return ax


# ======================================================================
# Parametric and distribution plots
# ======================================================================

def plot_strength_vs_amplitude(
    results_list: Sequence[dict],
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot predicted strength versus wrinkle amplitude from a parametric sweep.

    Parameters
    ----------
    results_list : sequence of dict
        Each dict must contain at least:

        - ``'amplitude'`` (float): wrinkle amplitude in mm.
        - ``'strength'`` (float): predicted failure stress in MPa.
        - ``'morphology'`` (str): morphology name.

        Additional keys are ignored.
    ax : Axes, optional
        Matplotlib axes. A new figure is created if ``None``.

    Returns
    -------
    Axes
        The axes with the amplitude-strength curves.
    """
    set_publication_style()
    ax = ensure_axes(ax)

    if not results_list:
        ax.text(
            0.5, 0.5, "No data", transform=ax.transAxes,
            ha="center", va="center", fontsize=10, color="0.5",
        )
        return ax

    # Group by morphology
    morph_data: dict[str, tuple[list[float], list[float]]] = {}
    for r in results_list:
        morph = r.get("morphology", "unknown")
        if morph not in morph_data:
            morph_data[morph] = ([], [])
        morph_data[morph][0].append(r["amplitude"])
        morph_data[morph][1].append(r["strength"])

    for morph, (amps, strengths) in morph_data.items():
        order = np.argsort(amps)
        amps_sorted = np.array(amps)[order]
        strengths_sorted = np.array(strengths)[order]

        style = get_morphology_style(morph)
        ax.plot(
            amps_sorted, strengths_sorted,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=4,
            label=style["label"],
        )

    ax.set_xlabel("Wrinkle amplitude $A$ (mm)")
    ax.set_ylabel("Predicted strength $\\sigma$ (MPa)")
    ax.set_title("Strength vs. Amplitude")
    ax.legend(loc="best", fontsize=7)

    return ax


def plot_strength_distribution(
    mc_results: "MonteCarloResults",
    ax: Optional[Axes] = None,
    n_bins: int = 50,
    show_kde: bool = True,
    by_morphology: bool = False,
) -> Axes:
    """Plot histogram and optional KDE of Monte Carlo strength results.

    Parameters
    ----------
    mc_results : MonteCarloResults
        Monte Carlo simulation results.
    ax : Axes, optional
        Matplotlib axes. A new figure is created if ``None``.
    n_bins : int, optional
        Number of histogram bins. Default is 50.
    show_kde : bool, optional
        If True, overlay a Gaussian KDE curve. Default is True.
    by_morphology : bool, optional
        If True, stack histograms by morphology. Default is False.

    Returns
    -------
    Axes
        The axes with the strength distribution plot.
    """
    set_publication_style()
    ax = ensure_axes(ax, figsize=FIGSIZE_DOUBLE_COLUMN)

    strengths = mc_results.strengths
    if strengths.size == 0:
        ax.text(
            0.5, 0.5, "No data", transform=ax.transAxes,
            ha="center", va="center", fontsize=10, color="0.5",
        )
        return ax

    if by_morphology:
        morphs = ["stack", "convex", "concave", "anti-stack"]
        data_list = []
        colors = []
        labels = []
        for morph in morphs:
            mask = mc_results.morphologies == morph
            if np.any(mask):
                data_list.append(strengths[mask])
                colors.append(MORPHOLOGY_COLORS.get(morph, "gray"))
                labels.append(MORPHOLOGY_LABELS.get(morph, morph))

        if data_list:
            ax.hist(
                data_list, bins=n_bins, stacked=True,
                color=colors, label=labels, alpha=0.8,
                edgecolor="white", linewidth=0.3,
            )
    else:
        ax.hist(
            strengths, bins=n_bins, density=True,
            color=MORPHOLOGY_COLORS["stack"], alpha=0.7, edgecolor="white", linewidth=0.3,
            label="Histogram",
        )

    if show_kde and not by_morphology:
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(strengths)
            x_kde = np.linspace(strengths.min(), strengths.max(), 300)
            ax.plot(x_kde, kde(x_kde), color=MORPHOLOGY_COLORS["concave"],
                    linewidth=1.5, label="KDE")
        except ImportError:
            pass

    # Add vertical lines for statistics
    mean_s = mc_results.mean_strength
    ax.axvline(mean_s, color="k", linestyle="--", linewidth=1.0,
               label=f"Mean = {mean_s:.0f} MPa")

    p5 = mc_results.percentile_5
    ax.axvline(p5, color="0.5", linestyle=":", linewidth=1.0,
               label=f"5th pctl = {p5:.0f} MPa")

    ax.set_xlabel("Predicted strength (MPa)")
    ax.set_ylabel("Density" if not by_morphology else "Count")
    ax.set_title("Strength Distribution (Monte Carlo)")
    ax.legend(loc="upper left", fontsize=7)

    return ax


def plot_jensen_gap(
    jensen_result: "JensenGapResult",
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot a bar chart of the Jensen gap broken down by morphology.

    The overall gap is shown as the first bar, followed by per-morphology
    values from the ``gap_by_morphology`` dictionary.

    Parameters
    ----------
    jensen_result : JensenGapResult
        Results from a Jensen gap analysis.
    ax : Axes, optional
        Matplotlib axes. A new figure is created if ``None``.

    Returns
    -------
    Axes
        The axes with the Jensen gap bar chart.
    """
    set_publication_style()
    ax = ensure_axes(ax)

    labels = ["Overall"]
    values = [jensen_result.jensen_gap]
    colors = [ACCENT_GRAY]

    for morph, gap in sorted(jensen_result.gap_by_morphology.items()):
        labels.append(MORPHOLOGY_LABELS.get(morph, morph))
        values.append(gap)
        colors.append(MORPHOLOGY_COLORS.get(morph, "gray"))

    if not values:
        ax.text(
            0.5, 0.5, "No data", transform=ax.transAxes,
            ha="center", va="center", fontsize=10, color="0.5",
        )
        return ax

    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, values, color=colors, edgecolor="white", linewidth=0.5)

    # Add value labels on bars
    for bar_obj, val in zip(bars, values):
        ax.text(
            bar_obj.get_x() + bar_obj.get_width() / 2.0,
            bar_obj.get_height() + max(values) * 0.02,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=7,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=7, rotation=15, ha="right")
    ax.set_ylabel("Jensen Gap (MPa)")
    ax.set_title(
        f"Jensen Gap: {jensen_result.jensen_gap:.1f} MPa "
        f"({jensen_result.jensen_gap_percent:.1f}%)"
    )
    ax.axhline(0, color="0.7", linewidth=0.5)

    return ax


def plot_failure_envelope(
    envelope_data: Sequence[dict],
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot a 2D failure envelope (e.g., amplitude vs. wavelength).

    Parameters
    ----------
    envelope_data : sequence of dict
        Each dict must contain:

        - ``'x'`` (float): x-axis parameter value.
        - ``'y'`` (float): y-axis parameter value.
        - ``'failed'`` (bool): whether the configuration fails.

        Optional keys:

        - ``'x_label'`` (str): x-axis label.
        - ``'y_label'`` (str): y-axis label.
        - ``'knockdown'`` (float): knockdown factor for color coding.

    ax : Axes, optional
        Matplotlib axes. A new figure is created if ``None``.

    Returns
    -------
    Axes
        The axes with the failure envelope plot.
    """
    set_publication_style()
    ax = ensure_axes(ax)

    if not envelope_data:
        ax.text(
            0.5, 0.5, "No data", transform=ax.transAxes,
            ha="center", va="center", fontsize=10, color="0.5",
        )
        return ax

    x_vals = np.array([d["x"] for d in envelope_data])
    y_vals = np.array([d["y"] for d in envelope_data])
    failed = np.array([d.get("failed", False) for d in envelope_data])

    # Check for knockdown data for color-coded scatter
    has_kd = all("knockdown" in d for d in envelope_data)

    if has_kd:
        kd_vals = np.array([d["knockdown"] for d in envelope_data])
        sc = ax.scatter(
            x_vals, y_vals, c=kd_vals, cmap="RdYlGn",
            s=25, edgecolors="0.3", linewidths=0.3,
        )
        colorbar_setup(ax, sc, "Knockdown factor")
    else:
        safe_mask = ~failed
        ax.scatter(
            x_vals[safe_mask], y_vals[safe_mask],
            c=MORPHOLOGY_COLORS["convex"], s=20, marker="o", label="Safe",
            edgecolors="0.3", linewidths=0.3,
        )
        ax.scatter(
            x_vals[failed], y_vals[failed],
            c=MORPHOLOGY_COLORS["concave"], s=20, marker="x", label="Failed",
            linewidths=1.0,
        )
        ax.legend(loc="best", fontsize=7)

    # Use labels from data if available
    x_label = envelope_data[0].get("x_label", "Parameter 1")
    y_label = envelope_data[0].get("y_label", "Parameter 2")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Failure Envelope")

    return ax


# ======================================================================
# FE post-processing plots
# ======================================================================

def plot_stress_through_thickness(
    field_results: "FieldResults",
    x: float,
    y: float,
    component: int = 0,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot a stress component through the laminate thickness.

    Extracts stress at the element column nearest to ``(x, y)`` and
    plots the selected Voigt component versus the z-coordinate.

    Parameters
    ----------
    field_results : FieldResults
        FE solution field results.
    x : float
        Longitudinal coordinate (mm).
    y : float
        Transverse coordinate (mm).
    component : int, optional
        Voigt stress component (0-5). Default is 0 (sigma_11).
    ax : Axes, optional
        Matplotlib axes. A new figure is created if ``None``.

    Returns
    -------
    Axes
        The axes with the through-thickness stress profile.
    """
    set_publication_style()
    ax = ensure_axes(ax, figsize=FIGSIZE_SINGLE_TALL)

    voigt_labels = [
        "$\\sigma_{11}$", "$\\sigma_{22}$", "$\\sigma_{33}$",
        "$\\tau_{23}$", "$\\tau_{13}$", "$\\tau_{12}$",
    ]
    comp_label = voigt_labels[component] if 0 <= component < 6 else f"Comp {component}"

    z_vals, stress_vals = field_results.stress_through_thickness(x, y, component)

    if z_vals.size == 0:
        ax.text(
            0.5, 0.5, "No data at this location",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="0.5",
        )
        return ax

    ax.plot(stress_vals, z_vals, "k-", linewidth=1.5)
    ax.axvline(0, color="0.7", linewidth=0.5, zorder=0)
    ax.set_xlabel(f"{comp_label} (MPa)")
    ax.set_ylabel("$z$ (mm)")
    ax.set_title(f"Through-Thickness {comp_label} at $x$={x:.1f}, $y$={y:.1f} mm")

    return ax


def plot_damage_contour(
    mesh: "MeshData",
    damage_field: np.ndarray,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot a 2D contour of damage at the laminate midplane.

    Extracts elements near the midplane and creates a scatter or
    contour plot of the damage field in the x-z plane (midplane).

    Parameters
    ----------
    mesh : MeshData
        Finite element mesh.
    damage_field : np.ndarray
        Shape ``(n_elements,)`` damage index per element. Values
        typically range from 0 (undamaged) to 1 (fully damaged).
    ax : Axes, optional
        Matplotlib axes. A new figure is created if ``None``.

    Returns
    -------
    Axes
        The axes with the damage contour plot.
    """
    set_publication_style()
    ax = ensure_axes(ax, figsize=FIGSIZE_DOUBLE_COLUMN)

    if damage_field.size == 0:
        ax.text(
            0.5, 0.5, "No damage data", transform=ax.transAxes,
            ha="center", va="center", fontsize=10, color="0.5",
        )
        return ax

    # Get midplane elements
    mid_elems = mesh.midplane_elements()
    if mid_elems.size == 0:
        ax.text(
            0.5, 0.5, "No midplane elements found", transform=ax.transAxes,
            ha="center", va="center", fontsize=10, color="0.5",
        )
        return ax

    # Compute element centroids at midplane
    x_centers = np.empty(len(mid_elems))
    y_centers = np.empty(len(mid_elems))
    for i, eid in enumerate(mid_elems):
        center = mesh.element_center(eid)
        x_centers[i] = center[0]
        y_centers[i] = center[1]

    damage_mid = damage_field[mid_elems]

    # Try to create a structured grid contour; fall back to scatter
    ux = np.unique(np.round(x_centers, 6))
    uy = np.unique(np.round(y_centers, 6))

    if len(ux) > 2 and len(uy) > 2 and len(ux) * len(uy) == len(mid_elems):
        # Reshape into grid for contourf
        nx_grid = len(ux)
        ny_grid = len(uy)
        # Sort centroids into grid
        order = np.lexsort((x_centers, y_centers))
        X = x_centers[order].reshape(ny_grid, nx_grid)
        Y = y_centers[order].reshape(ny_grid, nx_grid)
        D = damage_mid[order].reshape(ny_grid, nx_grid)

        cf = ax.contourf(X, Y, D, levels=20, cmap="hot_r")
        colorbar_setup(ax, cf, "Damage index $D$")
    else:
        # Fall back to scatter plot
        sc = ax.scatter(
            x_centers, y_centers, c=damage_mid, cmap="hot_r",
            s=8, edgecolors="none", vmin=0, vmax=max(damage_mid.max(), 0.01),
        )
        colorbar_setup(ax, sc, "Damage index $D$")

    ax.set_xlabel("$x$ (mm)")
    ax.set_ylabel("$y$ (mm)")
    ax.set_title("Damage Contour (Midplane)")
    ax.set_aspect("equal", adjustable="box")

    return ax


# ======================================================================
# Cohesive Zone Modeling (CZM) post-processing plots
# ======================================================================
#
# These plot functions operate on raw arrays / dicts extracted from an
# :class:`~wrinklefe.analysis.AnalysisResults` object whose CZM fields
# were populated by ``enable_czm=True``.  They intentionally do **not**
# depend on ``AnalysisResults`` directly so that they remain trivially
# unit-testable on synthetic inputs.  The :func:`czm_overview_figure`
# wrapper below assembles them into a single 2x2 dashboard.


def plot_traction_separation(
    separation_history: np.ndarray,
    traction_history: np.ndarray,
    ax: Optional[Axes] = None,
    label: Optional[str] = None,
    beta: float = 1.0,
) -> Axes:
    """Plot the traction-separation trajectory at a single cohesive Gauss point.

    The effective opening and effective traction are reduced from the
    three-component (normal, shear-s, shear-t) histories using the
    standard mixed-mode definitions::

        delta_eff = sqrt(<delta_n>^2 + beta^2 * (delta_s^2 + delta_t^2))
        T_eff    = sqrt(<T_n>^2     +           T_s^2     + T_t^2     )

    where ``<.>`` denotes the Macaulay bracket (negative normal opening
    is compression and is clamped to zero).  This is the same effective
    quantity that the bilinear damage law uses internally, so the curve
    can be read as a damage trajectory along the bilinear envelope.

    Parameters
    ----------
    separation_history : np.ndarray
        Shape ``(n_inc, 3)`` array of ``(delta_n, delta_s, delta_t)``
        per load increment at the Gauss point of interest.
    traction_history : np.ndarray
        Shape ``(n_inc, 3)`` array of ``(T_n, T_s, T_t)``.
    ax : Axes, optional
        Matplotlib axes. A new figure is created if ``None``.
    label : str, optional
        Curve label (forwarded to ``ax.plot``).
    beta : float, optional
        Mode-mixity weighting on the shear opening.  Default is 1.0.

    Returns
    -------
    Axes
        The axes with the traction-separation trajectory.
    """
    set_publication_style()
    ax = ensure_axes(ax)

    sep = np.asarray(separation_history, dtype=float)
    trc = np.asarray(traction_history, dtype=float)
    if sep.ndim != 2 or sep.shape[1] != 3:
        raise ValueError(
            "separation_history must have shape (n_inc, 3), got "
            f"{sep.shape}"
        )
    if trc.shape != sep.shape:
        raise ValueError(
            "traction_history must have the same shape as "
            f"separation_history; got {trc.shape} vs {sep.shape}"
        )

    delta_n = np.maximum(sep[:, 0], 0.0)
    delta_s = sep[:, 1]
    delta_t = sep[:, 2]
    delta_eff = np.sqrt(delta_n**2 + beta**2 * (delta_s**2 + delta_t**2))

    T_n = np.maximum(trc[:, 0], 0.0)
    T_s = trc[:, 1]
    T_t = trc[:, 2]
    T_eff = np.sqrt(T_n**2 + T_s**2 + T_t**2)

    ax.plot(
        delta_eff, T_eff,
        color=MORPHOLOGY_COLORS["concave"], linewidth=1.5,
        marker="o", markersize=3, label=label,
    )
    ax.axhline(0, color="0.7", linewidth=0.5, zorder=0)
    ax.axvline(0, color="0.7", linewidth=0.5, zorder=0)
    ax.set_xlabel("Effective separation $\\delta_{eff}$ (mm)")
    ax.set_ylabel("Effective traction $T_{eff}$ (MPa)")
    ax.set_title("Cohesive Traction-Separation")
    if label is not None:
        ax.legend(loc="best", fontsize=8)

    return ax


def plot_load_displacement(
    load_displacement: np.ndarray,
    ax: Optional[Axes] = None,
    label: Optional[str] = None,
) -> Axes:
    """Plot the load-displacement response from a CZM Newton-Raphson run.

    The convention here is x = load factor ``lambda`` (a clean control
    parameter in ``[0, 1]``) and y = norm of the displacement vector
    ``||u||``.  Plotting against ``lambda`` (rather than ``||u||``) keeps
    the abscissa monotone even when the global response softens, which
    is important for snap-back / snap-through visualization on a CZM
    run that has passed peak load.

    Parameters
    ----------
    load_displacement : np.ndarray
        Shape ``(n_inc, 2)`` array of ``(lambda, ||u||)`` samples per
        load increment (i.e. ``results.czm_load_displacement``).
    ax : Axes, optional
        Matplotlib axes. A new figure is created if ``None``.
    label : str, optional
        Curve label (forwarded to ``ax.plot``).

    Returns
    -------
    Axes
        The axes with the load-displacement curve.
    """
    set_publication_style()
    ax = ensure_axes(ax)

    ld = np.asarray(load_displacement, dtype=float)
    if ld.ndim != 2 or ld.shape[1] != 2:
        raise ValueError(
            "load_displacement must have shape (n_inc, 2), got "
            f"{ld.shape}"
        )

    if ld.shape[0] == 0:
        ax.text(
            0.5, 0.5, "No load increments", transform=ax.transAxes,
            ha="center", va="center", fontsize=10, color="0.5",
        )
        return ax

    lam = ld[:, 0]
    u_norm = ld[:, 1]

    ax.plot(
        lam, u_norm,
        color=MORPHOLOGY_COLORS["stack"], linewidth=1.5,
        marker="o", markersize=3, label=label,
    )
    ax.set_xlabel("Load factor $\\lambda$")
    ax.set_ylabel("$\\Vert u \\Vert$ (mm)")
    ax.set_title("Load-Displacement")
    if label is not None:
        ax.legend(loc="best", fontsize=8)

    return ax


def plot_damage_histogram(
    damage: np.ndarray,
    ax: Optional[Axes] = None,
    bins: int = 20,
) -> Axes:
    """Plot a histogram of the cohesive damage variable.

    Vertical reference lines mark three physically meaningful thresholds:

    * ``d = 0`` : intact cohesive zone.
    * ``d = 0.5`` : approximate location of the cohesive-zone front.
    * ``d = 1`` : fully delaminated.

    Parameters
    ----------
    damage : np.ndarray
        Shape ``(n_iface_elems, n_gauss)`` damage array, i.e.
        ``results.czm_damage``.  A 1-D array is also accepted.
    ax : Axes, optional
        Matplotlib axes. A new figure is created if ``None``.
    bins : int, optional
        Histogram bin count. Default is 20.

    Returns
    -------
    Axes
        The axes with the damage histogram.
    """
    set_publication_style()
    ax = ensure_axes(ax)

    d = np.asarray(damage, dtype=float).ravel()
    if d.size == 0:
        ax.text(
            0.5, 0.5, "No cohesive data", transform=ax.transAxes,
            ha="center", va="center", fontsize=10, color="0.5",
        )
        return ax

    ax.hist(
        d, bins=bins, range=(0.0, 1.0),
        color=MORPHOLOGY_COLORS["stack"], alpha=0.75,
        edgecolor="white", linewidth=0.4,
    )
    ax.axvline(0.0, color="0.4", linestyle=":", linewidth=0.8,
               label="intact ($d=0$)")
    ax.axvline(0.5, color=MORPHOLOGY_COLORS["convex"], linestyle="--",
               linewidth=0.8, label="cohesive front ($d=0.5$)")
    ax.axvline(1.0, color=MORPHOLOGY_COLORS["concave"], linestyle="--",
               linewidth=0.8, label="failed ($d=1$)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Damage $d$")
    ax.set_ylabel("Gauss-point count")
    ax.set_title("Damage Distribution")
    ax.legend(loc="best", fontsize=7)

    return ax


def plot_interface_damage_field(
    damage_per_elem: np.ndarray,
    xy_centroids: np.ndarray,
    ax: Optional[Axes] = None,
    cmap: str = "viridis",
) -> Axes:
    """Scatter the per-element damage on the interface (x, y) plane.

    A scatter plot rather than a contour is used because cohesive
    elements on a wrinkled interface are not guaranteed to form a
    regular ``(nx, ny)`` grid (the wrinkled surface is parameterised
    in 3D, projected here onto the in-plane axes).

    Parameters
    ----------
    damage_per_elem : np.ndarray
        Shape ``(n_iface_elems,)`` damage value per cohesive element
        (typically the Gauss-point mean of ``results.czm_damage``).
    xy_centroids : np.ndarray
        Shape ``(n_iface_elems, 2)`` element-centroid in-plane
        coordinates (``results.czm_element_centroids``).
    ax : Axes, optional
        Matplotlib axes. A new figure is created if ``None``.
    cmap : str, optional
        Matplotlib colormap name. Default is ``'viridis'`` (perceptually
        uniform; damage is unsigned in ``[0, 1]``).

    Returns
    -------
    Axes
        The axes with the interface damage scatter.
    """
    set_publication_style()
    ax = ensure_axes(ax, figsize=FIGSIZE_DOUBLE_COLUMN)

    d = np.asarray(damage_per_elem, dtype=float).ravel()
    xy = np.asarray(xy_centroids, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(
            f"xy_centroids must have shape (n_elem, 2), got {xy.shape}"
        )
    if d.size != xy.shape[0]:
        raise ValueError(
            "damage_per_elem and xy_centroids must agree on n_elem; "
            f"got {d.size} damage values vs {xy.shape[0]} centroids."
        )

    if d.size == 0:
        ax.text(
            0.5, 0.5, "No cohesive elements", transform=ax.transAxes,
            ha="center", va="center", fontsize=10, color="0.5",
        )
        return ax

    sc = ax.scatter(
        xy[:, 0], xy[:, 1], c=d, cmap=cmap,
        vmin=0.0, vmax=1.0,
        s=30, edgecolors="0.3", linewidths=0.2,
    )
    colorbar_setup(ax, sc, "Damage $d$")
    ax.set_xlabel("$x$ (mm)")
    ax.set_ylabel("$y$ (mm)")
    ax.set_title("Interface Damage Field")
    ax.set_aspect("equal", adjustable="box")

    return ax


def plot_energy_per_interface(
    energy_per_interface: Mapping[int, float],
    ax: Optional[Axes] = None,
) -> Axes:
    """Bar chart of cohesive energy dissipated, broken down by interface.

    Parameters
    ----------
    energy_per_interface : Mapping[int, float]
        ``{interface_index: energy}`` mapping (i.e.
        ``results.czm_energy_per_interface``).
    ax : Axes, optional
        Matplotlib axes. A new figure is created if ``None``.

    Returns
    -------
    Axes
        The axes with the energy bar chart.
    """
    set_publication_style()
    ax = ensure_axes(ax)

    if not energy_per_interface:
        ax.text(
            0.5, 0.5, "No interface energy data", transform=ax.transAxes,
            ha="center", va="center", fontsize=10, color="0.5",
        )
        return ax

    items = sorted(energy_per_interface.items(), key=lambda kv: kv[0])
    labels = [f"{idx}" for idx, _ in items]
    values = [float(v) for _, v in items]

    x_pos = np.arange(len(items))
    bars = ax.bar(
        x_pos, values,
        color=ACCENT_GRAY, edgecolor="white", linewidth=0.5,
    )

    # Value labels above each bar (skip when only one bar so the figure
    # stays uncluttered for the single-interface case).
    if len(items) > 1:
        vmax = max(values) if max(values) > 0 else 1.0
        for bar_obj, val in zip(bars, values):
            ax.text(
                bar_obj.get_x() + bar_obj.get_width() / 2.0,
                bar_obj.get_height() + 0.02 * vmax,
                f"{val:.2e}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Interface index")
    ax.set_ylabel("Dissipated energy (N$\\cdot$mm)")
    ax.set_title("Energy by Interface")
    ax.axhline(0, color="0.7", linewidth=0.5)

    return ax


def czm_overview_figure(results: "AnalysisResults") -> Figure:
    """Build a 2x2 dashboard of CZM outputs from an ``AnalysisResults``.

    Panels:

    * top-left      : load-displacement curve.
    * top-right     : damage histogram across all Gauss points.
    * bottom-left   : interface damage field (in-plane scatter).
    * bottom-right  : dissipated-energy bar chart per interface.

    Parameters
    ----------
    results : AnalysisResults
        Output of ``WrinkleAnalysis(cfg).run()`` with
        ``cfg.enable_czm=True``.

    Returns
    -------
    matplotlib.figure.Figure
        The composed 2x2 figure.  The caller owns the figure and must
        close it when done.

    Raises
    ------
    ValueError
        If CZM was not enabled (``results.czm_damage is None``) or
        the centroid bookkeeping was not populated.
    """
    set_publication_style()

    if results.czm_damage is None:
        raise ValueError(
            "czm_overview_figure requires AnalysisResults populated by "
            "WrinkleAnalysis(enable_czm=True); results.czm_damage is None."
        )

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)

    # Top-left: load-displacement
    if results.czm_load_displacement is not None:
        plot_load_displacement(results.czm_load_displacement, ax=axes[0, 0])
    else:
        axes[0, 0].text(
            0.5, 0.5, "No load-displacement data",
            transform=axes[0, 0].transAxes,
            ha="center", va="center", fontsize=10, color="0.5",
        )

    # Top-right: damage histogram
    plot_damage_histogram(results.czm_damage, ax=axes[0, 1])

    # Bottom-left: interface damage field scatter
    if (
        results.czm_element_centroids is not None
        and results.czm_damage.size > 0
    ):
        damage_per_elem = results.czm_damage.mean(axis=1)
        plot_interface_damage_field(
            damage_per_elem,
            results.czm_element_centroids,
            ax=axes[1, 0],
        )
    else:
        axes[1, 0].text(
            0.5, 0.5, "No interface damage field",
            transform=axes[1, 0].transAxes,
            ha="center", va="center", fontsize=10, color="0.5",
        )

    # Bottom-right: per-interface energy
    plot_energy_per_interface(
        results.czm_energy_per_interface or {},
        ax=axes[1, 1],
    )

    fig.suptitle("Cohesive Zone Modeling — Overview", fontsize=12)

    return fig
