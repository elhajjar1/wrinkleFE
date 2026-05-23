"""3D plotting functions for mesh and field visualization.

Provides 3D wireframe mesh plots, displacement and stress contour
visualizations on deformed meshes, and buckling mode shape rendering
using matplotlib's mplot3d toolkit.

All plot functions follow the same interface as :mod:`wrinklefe.viz.plots_2d`:

- Accept an optional ``ax`` parameter (must be a 3D projection axes if
  provided); if ``None``, a new figure with 3D axes is created.
- Return the :class:`~matplotlib.axes.Axes` object.
- Apply publication styling from :mod:`wrinklefe.viz.style`.

.. note::
   As with :mod:`wrinklefe.viz.plots_2d`, batch / sweep / headless loops must
   close internally-created figures; use :func:`wrinklefe.viz.save_figure`
   or :func:`wrinklefe.viz.figure_context`. Interactive callers are unaffected.

Colormap convention
-------------------
Use perceptually uniform colormaps (``jet`` is deprecated for scientific viz):

- Sequential / unsigned data (e.g. magnitudes): ``viridis``.
- Diverging / signed data that crosses zero (e.g. signed stress, mode
  shape displacements): ``RdBu_r``.

Each public ``plot_*`` function accepts an optional ``cmap`` keyword for
overrides; the defaults follow the convention above.

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
    set_axes_equal_aspect,
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
#
# Face index convention used throughout this module (must stay aligned
# with ``_HEX_FACE_AXIS`` and ``_HEX_FACE_SIDE`` below):
#   0: -z (bottom), 1: +z (top), 2: -y (front), 3: +y (back),
#   4: -x (left),   5: +x (right)
_HEX_FACES = np.array(
    [
        [0, 1, 2, 3],  # 0: bottom (-z)
        [4, 5, 6, 7],  # 1: top (+z)
        [0, 1, 5, 4],  # 2: front (-y)
        [2, 3, 7, 6],  # 3: back (+y)
        [0, 3, 7, 4],  # 4: left (-x)
        [1, 2, 6, 5],  # 5: right (+x)
    ],
    dtype=np.int64,
)

# For each face index above: which structured-grid axis it lies on
# (0 = i / x, 1 = j / y, 2 = k / z) and which side ("min" = -, "max" = +).
_HEX_FACE_AXIS = np.array([2, 2, 1, 1, 0, 0], dtype=np.int64)
_HEX_FACE_SIDE = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)  # 0 = min, 1 = max


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


def _is_full_structured_set(mesh: "MeshData", elem_indices: np.ndarray) -> bool:
    """Return True if ``elem_indices`` covers the full structured mesh.

    The fast structured-mesh boundary-culling shortcut is only valid when
    we are rendering every element in the natural ``(i, j, k)`` order; any
    sampling / cropping creates "holes" that the structured rule cannot
    detect, so the caller must fall back to the general algorithm.
    """
    nx = getattr(mesh, "nx", None)
    ny = getattr(mesh, "ny", None)
    nz = getattr(mesh, "nz", None)
    if nx is None or ny is None or nz is None:
        return False
    expected = int(nx) * int(ny) * int(nz)
    if expected != mesh.n_elements:
        return False
    if elem_indices.size != expected:
        return False
    # _create_hex_connectivity ravels (k, j, i) so flat index 0..N-1 is the
    # natural order; we accept the trivial arange(N) covering the full mesh.
    if elem_indices[0] != 0 or elem_indices[-1] != expected - 1:
        return False
    # Cheap monotonicity check (avoid full equality on large arrays).
    return bool(np.array_equal(elem_indices, np.arange(expected)))


def _structured_boundary_mask(mesh: "MeshData") -> np.ndarray:
    """Boolean mask over ``(n_elements, 6)`` marking boundary faces.

    Uses the structured-grid rule: face ``f`` of element ``(i, j, k)`` is
    on the domain boundary iff its grid index along the face's axis is
    ``0`` (for ``-`` faces) or ``n_axis - 1`` (for ``+`` faces).

    Returns
    -------
    np.ndarray
        Shape ``(n_elements, 6)`` boolean array.
    """
    nx, ny, nz = int(mesh.nx), int(mesh.ny), int(mesh.nz)
    # Element flat index = k*(ny*nx) + j*nx + i  (see _create_hex_connectivity)
    flat = np.arange(nx * ny * nz, dtype=np.int64)
    ii = flat % nx
    jj = (flat // nx) % ny
    kk = flat // (nx * ny)

    grid = np.stack([ii, jj, kk], axis=1)            # (N, 3)
    axis_size = np.array([nx, ny, nz], dtype=np.int64)

    # For each of the 6 faces, look up the element's index on that face's
    # axis and compare against either 0 (min side) or n-1 (max side).
    coord_on_axis = grid[:, _HEX_FACE_AXIS]           # (N, 6)
    max_on_axis = axis_size[_HEX_FACE_AXIS] - 1       # (6,)
    is_min_face = (_HEX_FACE_SIDE == 0)               # (6,) bool
    boundary = np.where(
        is_min_face[np.newaxis, :],
        coord_on_axis == 0,
        coord_on_axis == max_on_axis[np.newaxis, :],
    )
    return boundary


def _general_boundary_mask(
    mesh: "MeshData", elem_indices: np.ndarray
) -> np.ndarray:
    """Boolean mask over ``(len(elem_indices), 6)`` for arbitrary subsets.

    A face is on the boundary iff its sorted node-id tuple appears in the
    subset exactly once.  This handles sampled / cropped / cutaway meshes
    where the structured shortcut would be incorrect.
    """
    if elem_indices.size == 0:
        return np.zeros((0, 6), dtype=bool)

    # (n_sub, 6, 4) node ids per face for each selected element.
    conn = mesh.elements[elem_indices]                # (n_sub, 8)
    face_nodes = conn[:, _HEX_FACES]                  # (n_sub, 6, 4)

    # Sort node ids within each face so shared faces hash identically.
    face_keys = np.sort(face_nodes, axis=2)           # (n_sub, 6, 4)
    flat_keys = face_keys.reshape(-1, 4)              # (n_sub*6, 4)

    # Identify unique faces and how many elements claim each one.
    _, inverse, counts = np.unique(
        flat_keys, axis=0, return_inverse=True, return_counts=True
    )
    boundary_flat = counts[inverse] == 1
    return boundary_flat.reshape(elem_indices.size, 6)


def _gather_boundary_faces(
    mesh: "MeshData",
    elem_indices: np.ndarray,
    nodes: np.ndarray,
    elem_scalar: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Return boundary-face vertices (and optional per-face scalars).

    Combines structured-mesh culling (when the full mesh is selected) with
    a general face-counting fallback, and vectorizes the face-vertex
    gather so we never iterate Python-side over elements.

    Parameters
    ----------
    mesh : MeshData
        The finite element mesh.
    elem_indices : np.ndarray
        Element indices to include (output of :func:`_sample_elements`).
    nodes : np.ndarray
        Node coordinates of shape ``(n_nodes, 3)`` — typically the
        (possibly deformed) coordinates used for rendering.
    elem_scalar : np.ndarray, optional
        Shape ``(n_elements,)`` per-element scalar (e.g. mean stress).
        If given, the returned scalar array is per-boundary-face,
        broadcasting the element value onto all of its boundary faces.

    Returns
    -------
    face_verts : np.ndarray
        Shape ``(n_boundary_faces, 4, 3)``.
    face_scalar : np.ndarray or None
        Shape ``(n_boundary_faces,)`` if ``elem_scalar`` was provided.
    """
    if elem_indices.size == 0:
        return np.empty((0, 4, 3), dtype=nodes.dtype), (
            None if elem_scalar is None else np.empty((0,), dtype=float)
        )

    if _is_full_structured_set(mesh, elem_indices):
        mask = _structured_boundary_mask(mesh)        # (n_elem, 6)
        # Sub-selection follows elem_indices (which equals arange(N)).
        sub_conn = mesh.elements                      # (n_elem, 8)
    else:
        mask = _general_boundary_mask(mesh, elem_indices)
        sub_conn = mesh.elements[elem_indices]        # (n_sub, 8)

    # Vectorized vertex gather: (n_sub, 6, 4) node ids -> (n_sub, 6, 4, 3) coords
    face_nodes = sub_conn[:, _HEX_FACES]              # (n_sub, 6, 4)
    face_verts_all = nodes[face_nodes]                # (n_sub, 6, 4, 3)

    face_verts = face_verts_all[mask]                 # (n_bnd, 4, 3)

    face_scalar: Optional[np.ndarray] = None
    if elem_scalar is not None:
        if _is_full_structured_set(mesh, elem_indices):
            elem_vals = elem_scalar
        else:
            elem_vals = elem_scalar[elem_indices]
        # Broadcast each element's value over its 6 faces, then mask.
        scalar_per_face = np.broadcast_to(
            elem_vals[:, np.newaxis], (elem_vals.shape[0], 6)
        )
        face_scalar = scalar_per_face[mask].astype(float, copy=False)

    return face_verts, face_scalar


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
    # Cull interior (shared) faces and assemble vertex array vectorized.
    face_verts, _ = _gather_boundary_faces(mesh, elem_idx, mesh.nodes)

    edge_color = "0.3" if show_edges else "none"
    poly = Poly3DCollection(
        face_verts,
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
    set_axes_equal_aspect(ax, mins, maxs)

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
    cmap: Optional[str] = None,
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
    cmap : str, optional
        Matplotlib colormap name. Defaults to ``'RdBu_r'`` (diverging,
        appropriate for signed displacement that crosses zero).

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

    # Cull interior faces and gather boundary-face vertices vectorized.
    face_verts, _ = _gather_boundary_faces(mesh, elem_idx, deformed_nodes)

    # For the per-face scalar we average the node scalar over the 4 face
    # nodes.  Reconstruct the same boundary mask used inside
    # ``_gather_boundary_faces`` and apply it to the (n_sub, 6) array of
    # mean-node-scalar values.
    if _is_full_structured_set(mesh, elem_idx):
        _mask = _structured_boundary_mask(mesh)
        _sub_conn = mesh.elements
    else:
        _mask = _general_boundary_mask(mesh, elem_idx)
        _sub_conn = mesh.elements[elem_idx]
    _face_node_ids = _sub_conn[:, _HEX_FACES]            # (n_sub, 6, 4)
    _face_mean = node_scalar[_face_node_ids].mean(axis=2)  # (n_sub, 6)
    face_values = _face_mean[_mask].astype(float, copy=False)
    vmin = face_values.min()
    vmax = face_values.max()
    if abs(vmax - vmin) < 1e-15:
        vmax = vmin + 1e-10

    # Normalize and map to colormap (diverging default for signed data)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap if cmap is not None else "RdBu_r")
    face_colors = cmap_obj(norm(face_values))

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
    set_axes_equal_aspect(ax, mins, maxs)

    ax.set_xlabel("$x$ (mm)", labelpad=8)
    ax.set_ylabel("$y$ (mm)", labelpad=8)
    ax.set_zlabel("$z$ (mm)", labelpad=8)

    scale_str = f" (x{scale:.0f})" if scale != 1.0 else ""
    ax.set_title(f"Displacement {comp_label}{scale_str}")

    # Add colorbar via a ScalarMappable
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
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
    cmap: Optional[str] = None,
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
    cmap : str, optional
        Matplotlib colormap name. Defaults to ``'RdBu_r'`` (diverging,
        appropriate for signed stress that crosses zero). Use
        ``'viridis'`` for unsigned stress measures (e.g. von Mises).

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

    # Cull interior (shared) faces and broadcast each element's stress onto
    # its surviving boundary faces in a single vectorized pass.
    face_verts, face_values = _gather_boundary_faces(
        mesh, elem_idx, deformed_nodes, elem_scalar=elem_stress
    )
    # ``_gather_boundary_faces`` returns ``None`` only when no elem_scalar was
    # supplied; here we always supplied one.
    assert face_values is not None
    vmin = face_values.min()
    vmax = face_values.max()
    if abs(vmax - vmin) < 1e-15:
        vmax = vmin + 1e-10

    # Diverging colormap default: signed stress components typically
    # cross zero. ``jet`` (previously used here) is perceptually
    # non-uniform and deprecated for scientific visualization.
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap if cmap is not None else "RdBu_r")
    face_colors = cmap_obj(norm(face_values))

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
    set_axes_equal_aspect(ax, mins, maxs)

    ax.set_xlabel("$x$ (mm)", labelpad=8)
    ax.set_ylabel("$y$ (mm)", labelpad=8)
    ax.set_zlabel("$z$ (mm)", labelpad=8)
    ax.set_title(f"Stress {comp_label} ({coord})")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
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
    cmap: Optional[str] = None,
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
    cmap : str, optional
        Matplotlib colormap name. Defaults to ``'RdBu_r'`` (diverging,
        appropriate for signed mode-shape displacement that crosses zero).

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

    # Cull interior faces and gather boundary-face vertices vectorized.
    face_verts, _ = _gather_boundary_faces(mesh, elem_idx, deformed_nodes)

    # Per-face scalar = average of uz at the 4 face nodes, computed on the
    # same (n_sub, 6) lattice and then masked down to boundary faces.
    if _is_full_structured_set(mesh, elem_idx):
        _mask = _structured_boundary_mask(mesh)
        _sub_conn = mesh.elements
    else:
        _mask = _general_boundary_mask(mesh, elem_idx)
        _sub_conn = mesh.elements[elem_idx]
    _face_node_ids = _sub_conn[:, _HEX_FACES]
    _face_mean = uz[_face_node_ids].mean(axis=2)
    face_values = _face_mean[_mask].astype(float, copy=False)
    vmin = face_values.min()
    vmax = face_values.max()
    if abs(vmax - vmin) < 1e-15:
        vmax = vmin + 1e-10

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap if cmap is not None else "RdBu_r")
    face_colors = cmap_obj(norm(face_values))

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
    set_axes_equal_aspect(ax, mins, maxs)

    ax.set_xlabel("$x$ (mm)", labelpad=8)
    ax.set_ylabel("$y$ (mm)", labelpad=8)
    ax.set_zlabel("$z$ (mm)", labelpad=8)

    eigenvalue = buckling_result.eigenvalues[mode]
    ax.set_title(
        f"Buckling Mode {mode + 1} ($\\lambda$ = {eigenvalue:.4f}, scale = {scale:.1f})"
    )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    fig = ax.get_figure()
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("$u_z$ (mm, scaled)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.view_init(elev=25, azim=-60)

    return ax
