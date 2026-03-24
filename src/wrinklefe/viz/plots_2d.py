"""2D plotting functions for wrinkle analysis and post-processing.

Provides publication-quality 2D visualizations including wrinkle profiles,
morphology factor sweeps, kink-band concavity diagrams, strength
distributions, Jensen gap charts, failure envelopes, through-thickness
stress profiles, and damage contour plots.

All plot functions follow a consistent interface:

- Accept an optional ``ax`` parameter; if ``None``, a new figure is created.
- Return the :class:`~matplotlib.axes.Axes` object for further customization.
- Apply publication styling from :mod:`wrinklefe.viz.style`.

References
----------
Jin, L. et al. (2026). Thin-Walled Structures, 219, 114237.
Elhajjar, R. (2025). Scientific Reports, 15, 25977.
Budiansky, B. & Fleck, N.A. (1993). J. Mech. Phys. Solids, 41(1), 183-211.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from wrinklefe.viz.style import (
    FIGSIZE_SINGLE_COLUMN,
    FIGSIZE_DOUBLE_COLUMN,
    MORPHOLOGY_COLORS,
    MORPHOLOGY_LABELS,
    MORPHOLOGY_LINESTYLES,
    colorbar_setup,
    ensure_axes,
    get_morphology_style,
    set_publication_style,
)

if TYPE_CHECKING:
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

    ax.plot(x, z, color="#1f77b4", linewidth=1.5)
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

    ax.plot(x, z_upper, color="#1f77b4", linewidth=1.5, label="Upper wrinkle")
    ax.plot(x, z_lower, color="#d62728", linewidth=1.5, label="Lower wrinkle")

    if show_gap:
        gap = z_upper - z_lower
        ax.fill_between(
            x, z_lower, z_upper,
            alpha=0.15, color="#2ca02c", label="Interface gap",
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
    from wrinklefe.core.morphology import WrinkleConfiguration, MORPHOLOGY_PHASES

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
            color="#1f77b4", alpha=0.7, edgecolor="white", linewidth=0.3,
            label="Histogram",
        )

    if show_kde and not by_morphology:
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(strengths)
            x_kde = np.linspace(strengths.min(), strengths.max(), 300)
            ax.plot(x_kde, kde(x_kde), color="#d62728", linewidth=1.5, label="KDE")
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
    colors = ["#333333"]

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
            c="#2ca02c", s=20, marker="o", label="Safe",
            edgecolors="0.3", linewidths=0.3,
        )
        ax.scatter(
            x_vals[failed], y_vals[failed],
            c="#d62728", s=20, marker="x", label="Failed",
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
