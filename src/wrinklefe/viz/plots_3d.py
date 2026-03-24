"""3D plotting functions for mesh and field visualization.

Provides 3D wireframe mesh plots, displacement and stress contour
visualizations on deformed meshes, and buckling mode shape rendering
using matplotlib's mplot3d toolkit.

All plot functions follow the same interface as :mod:`wrinklefe.viz.plots_2d`:

- Accept an optional ``ax`` parameter (must be a 3D projection axes if
  provided); if ``None``, a new figure with 3D axes is created.
- Return the :class:`~matplotlib.axes.Axes` object.
- Apply publication styling from :mod:`wrinklefe.viz.style`.

References
----------
Jin, L. et al. (2026). Thin-Walled Structures, 219, 114237.
Elhajjar, R. (2025). Scientific Reports, 15, 25977.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore[import-untyped]

from wrinklefe.viz.style import (
    FIGSIZE_DOUBLE_COLUMN,
    colorbar_setup,
    ensure_axes,
    set_publication_style,
)

if TYPE_CHECKING:
    from wrinklefe.core.mesh import MeshData
    from wrinklefe.solver.buckling import BucklingResult
    from wrinklefe.solver.results import FieldResults


# ======================================================================
# Hex8 face definitions (VTK/Abaqus node ordering)
# ======================================================================

# Each hex8 element has 6 faces, each defined by 4 node local indices.
# Node ordering: bottom CCW (0,1,2,3), top CCW (4,5,6,7).
_HEX_FACES = [
    [0, 1, 2, 3],  # bottom (-z)
    [4, 5, 6, 7],  # top (+z)
    [0, 1, 5, 4],  # front (-y)
    [2, 3, 7, 6],  # back (+y)
    [0, 3, 7, 4],  # left (-x)
    [1, 2, 6, 5],  # right (+x)
]


# ======================================================================
# Helper functions
# ======================================================================

def _sample_elements(mesh: "MeshData", max_elements: int = 2000) -> np.ndarray:
    """Select a subset of element indices for rendering.

    For large meshes, rendering every element is prohibitively slow.
    This function uniformly samples element indices.

    Parameters
    ----------
    mesh : MeshData
        The finite element mesh.
    max_elements : int
        Maximum number of elements to render.

    Returns
    -------
    np.ndarray
        1-D array of selected element indices.
    """
    n_elem = mesh.n_elements
    if n_elem <= max_elements:
        return np.arange(n_elem)
    step = max(1, n_elem // max_elements)
    return np.arange(0, n_elem, step)


def _collect_faces(
    mesh: "MeshData",
    elem_indices: np.ndarray,
    nodes: Optional[np.ndarray] = None,
) -> list[np.ndarray]:
    """Collect face vertex arrays for a set of elements.

    Parameters
    ----------
    mesh : MeshData
        The finite element mesh.
    elem_indices : np.ndarray
        Element indices to include.
    nodes : np.ndarray, optional
        Node coordinates to use. If ``None``, uses ``mesh.nodes``.

    Returns
    -------
    list of np.ndarray
        Each entry is shape ``(4, 3)`` -- the four vertices of a face.
    """
    if nodes is None:
        nodes = mesh.nodes
    faces = []
    for eid in elem_indices:
        conn = mesh.elements[eid]
        for face_local in _HEX_FACES:
            verts = nodes[conn[face_local]]  # (4, 3)
            faces.append(verts)
    return faces


# ======================================================================
# 3D Mesh Visualization
# ======================================================================

def plot_mesh_3d(
    mesh: "MeshData",
    ax: Optional[Axes] = None,
    max_elements: int = 2000,
    show_edges: bool = True,
    face_alpha: float = 0.15,
) -> Axes:
    """Plot a 3D wireframe mesh with wrinkle deformation.

    Renders the hex8 mesh as semi-transparent faces with black edges,
    showing the wrinkle-induced out-of-plane deformation.

    Parameters
    ----------
    mesh : MeshData
        Finite element mesh with (possibly deformed) node coordinates.
    ax : Axes, optional
        A 3D matplotlib axes. A new figure is created if ``None``.
    max_elements : int, optional
        Maximum number of elements to render. Default is 2000.
    show_edges : bool, optional
        Whether to draw element edges. Default is True.
    face_alpha : float, optional
        Face transparency (0 = invisible, 1 = opaque). Default is 0.15.

    Returns
    -------
    Axes
        The 3D axes with the mesh plot.
    """
    set_publication_style()
    ax = ensure_axes(ax, figsize=FIGSIZE_DOUBLE_COLUMN, projection="3d")

    if mesh.n_elements == 0:
        return ax

    elem_idx = _sample_elements(mesh, max_elements)
    faces = _collect_faces(mesh, elem_idx)

    edge_color = "0.3" if show_edges else "none"
    poly = Poly3DCollection(
        faces,
        facecolors=(0.7, 0.85, 1.0, face_alpha),
        edgecolors=edge_color,
        linewidths=0.2,
    )
    ax.add_collection3d(poly)

    # Set axis limits from node extents
    mins = mesh.nodes.min(axis=0)
    maxs = mesh.nodes.max(axis=0)
    pad = 0.05 * (maxs - mins + 1e-10)
    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])

    ax.set_xlabel("$x$ (mm)", labelpad=8)
    ax.set_ylabel("$y$ (mm)", labelpad=8)
    ax.set_zlabel("$z$ (mm)", labelpad=8)
    ax.set_title("3D Mesh")
    ax.view_init(elev=25, azim=-60)

    return ax


# ======================================================================
# Displacement Visualization
# ======================================================================

def plot_displacement_3d(
    field_results: "FieldResults",
    component: int = 2,
    ax: Optional[Axes] = None,
    max_elements: int = 2000,
    scale: float = 1.0,
) -> Axes:
    """Plot the deformed mesh colored by a displacement component.

    Parameters
    ----------
    field_results : FieldResults
        FE solution containing displacement and mesh reference.
    component : int, optional
        Displacement component: 0 = ux, 1 = uy, 2 = uz. Default is 2.
    ax : Axes, optional
        A 3D matplotlib axes. A new figure is created if ``None``.
    max_elements : int, optional
        Maximum elements to render. Default is 2000.
    scale : float, optional
        Displacement magnification factor. Default is 1.0.

    Returns
    -------
    Axes
        The 3D axes with the displacement contour.
    """
    set_publication_style()
    ax = ensure_axes(ax, figsize=FIGSIZE_DOUBLE_COLUMN, projection="3d")

    mesh = field_results.mesh
    disp = field_results.displacement

    if mesh.n_elements == 0 or disp.size == 0:
        return ax

    # Deformed node positions
    deformed_nodes = mesh.nodes + disp * scale

    # Compute per-node scalar for coloring
    comp_labels = ["$u_x$", "$u_y$", "$u_z$"]
    comp_label = comp_labels[component] if 0 <= component < 3 else f"$u_{component}$"
    node_scalar = disp[:, component]

    elem_idx = _sample_elements(mesh, max_elements)

    # Compute face colors from average node values
    face_verts = []
    face_values = []
    for eid in elem_idx:
        conn = mesh.elements[eid]
        for face_local in _HEX_FACES:
            face_nodes = conn[face_local]
            verts = deformed_nodes[face_nodes]
            face_verts.append(verts)
            face_values.append(np.mean(node_scalar[face_nodes]))

    face_values = np.array(face_values)
    vmin = face_values.min()
    vmax = face_values.max()
    if abs(vmax - vmin) < 1e-15:
        vmax = vmin + 1e-10

    # Normalize and map to colormap
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.coolwarm  # type: ignore[attr-defined]
    face_colors = cmap(norm(face_values))

    poly = Poly3DCollection(
        face_verts,
        facecolors=face_colors,
        edgecolors="0.5",
        linewidths=0.1,
    )
    ax.add_collection3d(poly)

    # Axis limits from deformed mesh
    mins = deformed_nodes.min(axis=0)
    maxs = deformed_nodes.max(axis=0)
    pad = 0.05 * (maxs - mins + 1e-10)
    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])

    ax.set_xlabel("$x$ (mm)", labelpad=8)
    ax.set_ylabel("$y$ (mm)", labelpad=8)
    ax.set_zlabel("$z$ (mm)", labelpad=8)

    scale_str = f" (x{scale:.0f})" if scale != 1.0 else ""
    ax.set_title(f"Displacement {comp_label}{scale_str}")

    # Add colorbar via a ScalarMappable
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig = ax.get_figure()
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(f"{comp_label} (mm)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.view_init(elev=25, azim=-60)

    return ax


# ======================================================================
# Stress Contour Visualization
# ======================================================================

def plot_stress_contour_3d(
    field_results: "FieldResults",
    component: int = 0,
    ax: Optional[Axes] = None,
    max_elements: int = 2000,
    coord: str = "global",
) -> Axes:
    """Plot stress contour on the deformed mesh.

    Colors each element face by the average stress value at its nodes
    (interpolated from Gauss-point data to element centroid).

    Parameters
    ----------
    field_results : FieldResults
        FE solution containing stress and mesh reference.
    component : int, optional
        Voigt stress component (0-5). Default is 0 (sigma_11).
    ax : Axes, optional
        A 3D matplotlib axes. A new figure is created if ``None``.
    max_elements : int, optional
        Maximum elements to render. Default is 2000.
    coord : str, optional
        Coordinate system: ``'global'`` (default) or ``'local'``.

    Returns
    -------
    Axes
        The 3D axes with the stress contour.
    """
    set_publication_style()
    ax = ensure_axes(ax, figsize=FIGSIZE_DOUBLE_COLUMN, projection="3d")

    mesh = field_results.mesh

    if coord == "local":
        stress_data = field_results.stress_local
    else:
        stress_data = field_results.stress_global

    if mesh.n_elements == 0 or stress_data.size == 0:
        return ax

    voigt_labels = [
        "$\\sigma_{11}$", "$\\sigma_{22}$", "$\\sigma_{33}$",
        "$\\tau_{23}$", "$\\tau_{13}$", "$\\tau_{12}$",
    ]
    comp_label = voigt_labels[component] if 0 <= component < 6 else f"Comp {component}"

    # Average stress over Gauss points for each element
    elem_stress = stress_data[:, :, component].mean(axis=1)  # (n_elem,)

    # Deformed nodes (small deformation visualization)
    disp = field_results.displacement
    deformed_nodes = mesh.nodes + disp

    elem_idx = _sample_elements(mesh, max_elements)

    face_verts = []
    face_values = []
    for eid in elem_idx:
        conn = mesh.elements[eid]
        s_val = elem_stress[eid]
        for face_local in _HEX_FACES:
            face_nodes = conn[face_local]
            verts = deformed_nodes[face_nodes]
            face_verts.append(verts)
            face_values.append(s_val)

    face_values = np.array(face_values)
    vmin = face_values.min()
    vmax = face_values.max()
    if abs(vmax - vmin) < 1e-15:
        vmax = vmin + 1e-10

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.jet  # type: ignore[attr-defined]
    face_colors = cmap(norm(face_values))

    poly = Poly3DCollection(
        face_verts,
        facecolors=face_colors,
        edgecolors="0.5",
        linewidths=0.1,
    )
    ax.add_collection3d(poly)

    mins = deformed_nodes.min(axis=0)
    maxs = deformed_nodes.max(axis=0)
    pad = 0.05 * (maxs - mins + 1e-10)
    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])

    ax.set_xlabel("$x$ (mm)", labelpad=8)
    ax.set_ylabel("$y$ (mm)", labelpad=8)
    ax.set_zlabel("$z$ (mm)", labelpad=8)
    ax.set_title(f"Stress {comp_label} ({coord})")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig = ax.get_figure()
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(f"{comp_label} (MPa)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.view_init(elev=25, azim=-60)

    return ax


# ======================================================================
# Buckling Mode Shape Visualization
# ======================================================================

def plot_buckling_mode(
    buckling_result: "BucklingResult",
    mode: int = 0,
    scale: float = 1.0,
    ax: Optional[Axes] = None,
    max_elements: int = 2000,
) -> Axes:
    """Plot a buckling mode shape on the deformed mesh.

    The mode shape eigenvector is scaled and added to the original
    mesh coordinates. Faces are colored by the out-of-plane (uz)
    component of the mode displacement.

    Parameters
    ----------
    buckling_result : BucklingResult
        Results from a linear buckling analysis.
    mode : int, optional
        Mode index (0-based). Default is 0 (critical mode).
    scale : float, optional
        Scaling factor for mode shape visualization. Default is 1.0.
    ax : Axes, optional
        A 3D matplotlib axes. A new figure is created if ``None``.
    max_elements : int, optional
        Maximum elements to render. Default is 2000.

    Returns
    -------
    Axes
        The 3D axes with the buckling mode shape.
    """
    set_publication_style()
    ax = ensure_axes(ax, figsize=FIGSIZE_DOUBLE_COLUMN, projection="3d")

    mesh = buckling_result.mesh

    if mesh.n_elements == 0 or buckling_result.n_modes == 0:
        ax.text2D(
            0.5, 0.5, "No buckling modes available",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="0.5",
        )
        return ax

    if mode < 0 or mode >= buckling_result.n_modes:
        ax.text2D(
            0.5, 0.5, f"Mode {mode} out of range (0-{buckling_result.n_modes - 1})",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="0.5",
        )
        return ax

    # Get mode displacement
    mode_disp = buckling_result.mode_displacement(mode, scale)  # (n_nodes, 3)
    deformed_nodes = mesh.nodes + mode_disp

    # Color by uz component
    uz = mode_disp[:, 2]

    elem_idx = _sample_elements(mesh, max_elements)

    face_verts = []
    face_values = []
    for eid in elem_idx:
        conn = mesh.elements[eid]
        for face_local in _HEX_FACES:
            face_nodes = conn[face_local]
            verts = deformed_nodes[face_nodes]
            face_verts.append(verts)
            face_values.append(np.mean(uz[face_nodes]))

    face_values = np.array(face_values)
    vmin = face_values.min()
    vmax = face_values.max()
    if abs(vmax - vmin) < 1e-15:
        vmax = vmin + 1e-10

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.coolwarm  # type: ignore[attr-defined]
    face_colors = cmap(norm(face_values))

    poly = Poly3DCollection(
        face_verts,
        facecolors=face_colors,
        edgecolors="0.5",
        linewidths=0.1,
    )
    ax.add_collection3d(poly)

    mins = deformed_nodes.min(axis=0)
    maxs = deformed_nodes.max(axis=0)
    pad = 0.05 * (maxs - mins + 1e-10)
    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])

    ax.set_xlabel("$x$ (mm)", labelpad=8)
    ax.set_ylabel("$y$ (mm)", labelpad=8)
    ax.set_zlabel("$z$ (mm)", labelpad=8)

    eigenvalue = buckling_result.eigenvalues[mode]
    ax.set_title(
        f"Buckling Mode {mode + 1} ($\\lambda$ = {eigenvalue:.4f}, scale = {scale:.1f})"
    )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig = ax.get_figure()
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("$u_z$ (mm, scaled)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.view_init(elev=25, azim=-60)

    return ax
