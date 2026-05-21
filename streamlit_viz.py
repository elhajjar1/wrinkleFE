"""Plotly viz helpers for the WrinkleFE Streamlit app.

Renders the boundary surface of a structured hex mesh as a Plotly Mesh3d,
optionally deformed by the FE displacement field, coloured by a
per-element scalar (e.g. sigma_33 or max failure index). Also provides
a 2D y-slice scatter and a per-criterion failure-index heatmap.

All functions are pure: they take numpy arrays + parameters and return
a plotly.graph_objects.Figure. The Streamlit app is responsible for
caching the FE arrays and wiring sliders to function arguments.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from wrinklefe.viz.style import MORPHOLOGY_COLORS


# Local node indices for the 6 faces of a hex element following the
# CGNS / VTK_HEXAHEDRON convention used by wrinklefe.core.mesh:
#   nodes 0-3: bottom face (z low) CCW from below
#   nodes 4-7: top face (z high) CCW from above, parallel to bottom
HEX_FACES: np.ndarray = np.array(
    [
        [0, 1, 2, 3],  # z-low
        [4, 5, 6, 7],  # z-high
        [0, 1, 5, 4],  # y-low
        [3, 2, 6, 7],  # y-high
        [0, 3, 7, 4],  # x-low
        [1, 2, 6, 5],  # x-high
    ],
    dtype=np.int64,
)


def boundary_faces(elements: np.ndarray) -> np.ndarray:
    """Boundary faces of a hex mesh, with parent element id.

    A face is on the boundary iff its sorted node-id tuple appears
    exactly once across the mesh — i.e. it is not shared with a
    neighbouring element. Implemented vectorized with NumPy: no Python
    loops over elements, which keeps the cost dominated by a single
    ``np.unique`` over ``6 * n_elements`` rows.

    Parameters
    ----------
    elements : np.ndarray
        Shape ``(n_elements, 8)`` hex8 connectivity.

    Returns
    -------
    np.ndarray
        ``(n_boundary_faces, 5)`` array.  Columns 0-3 are global node
        indices in the element's local face ordering (so quad winding
        is preserved for downstream triangulation); column 4 is the
        parent element id.
    """
    elements = np.asarray(elements, dtype=np.int64)
    n_elem = elements.shape[0]
    if n_elem == 0:
        return np.empty((0, 5), dtype=np.int64)

    # (n_elem, 6, 4) global node ids per face in local winding order.
    face_nodes = elements[:, HEX_FACES]
    # Sort along the 4-node axis so shared faces hash identically
    # regardless of the two parent elements' local winding.
    face_keys = np.sort(face_nodes, axis=2).reshape(-1, 4)

    # np.unique on axis=0 sorts rows lexicographically; counts==1 picks
    # out the unshared (boundary) faces.
    _, inverse, counts = np.unique(
        face_keys, axis=0, return_inverse=True, return_counts=True
    )
    boundary_flat = counts[inverse] == 1            # (n_elem*6,)

    # Parent element id for each flat face row.
    elem_ids = np.repeat(np.arange(n_elem, dtype=np.int64), HEX_FACES.shape[0])

    # Keep the original (non-sorted) winding for the boundary faces.
    flat_nodes = face_nodes.reshape(-1, 4)
    out = np.empty((int(boundary_flat.sum()), 5), dtype=np.int64)
    out[:, :4] = flat_nodes[boundary_flat]
    out[:, 4] = elem_ids[boundary_flat]
    return out


def quads_to_triangles(quad_faces: np.ndarray) -> np.ndarray:
    """Split each quad (4 nodes + parent elem id) into two triangles.

    Each input row is [n0, n1, n2, n3, ei]. Output has 2 rows per input,
    each [na, nb, nc, ei], split along the n0-n2 diagonal.
    """
    n = quad_faces.shape[0]
    tri = np.empty((n * 2, 4), dtype=np.int64)
    tri[0::2, 0] = quad_faces[:, 0]
    tri[0::2, 1] = quad_faces[:, 1]
    tri[0::2, 2] = quad_faces[:, 2]
    tri[0::2, 3] = quad_faces[:, 4]
    tri[1::2, 0] = quad_faces[:, 0]
    tri[1::2, 1] = quad_faces[:, 2]
    tri[1::2, 2] = quad_faces[:, 3]
    tri[1::2, 3] = quad_faces[:, 4]
    return tri


def _scene_layout() -> dict:
    return dict(
        xaxis_title="x [mm]",
        yaxis_title="y [mm]",
        zaxis_title="z [mm]",
        aspectmode="data",
    )


def compute_mesh3d_geometry(elements: np.ndarray) -> dict:
    """Precompute the connectivity-derived geometry used by Mesh3d plots.

    The expensive parts of building a Plotly Mesh3d for a hex mesh
    (``boundary_faces`` + ``quads_to_triangles`` + the boundary-node
    ``np.unique`` remap) depend only on the connectivity, not on the
    nodal coordinates.  Streamlit slider re-renders keep ``elements``
    constant between solves, so the FE app can call this once after
    each solve and pass the result back into :func:`mesh3d_figure` via
    ``precomputed_geometry=`` to skip the boundary cull on every redraw.

    Returns a dict with:

    - ``tri``: ``(n_tri, 4)`` array of [na, nb, nc, parent_elem_id]
      from :func:`quads_to_triangles`. Owners (column 3) drive per-cell
      intensity broadcasting.
    - ``kept_nodes``: ``(n_kept,)`` unique global node ids referenced by
      the boundary triangles, in ascending order.  Used to slice the
      ``vertices`` / ``vertex_scalar`` arrays down to the surface.
    - ``tri_ijk``: ``(n_tri, 3)`` triangle node indices remapped into
      the compact ``kept_nodes`` index space — the i/j/k Plotly Mesh3d
      expects.
    """
    bf = boundary_faces(elements)
    tri = quads_to_triangles(bf)
    if tri.shape[0] > 0:
        kept_nodes, remap = np.unique(tri[:, :3].ravel(), return_inverse=True)
        tri_ijk = remap.reshape(-1, 3).astype(np.int64, copy=False)
    else:
        kept_nodes = np.empty(0, dtype=np.int64)
        tri_ijk = np.empty((0, 3), dtype=np.int64)
    return {"tri": tri, "kept_nodes": kept_nodes, "tri_ijk": tri_ijk}


def mesh3d_figure(
    vertices: np.ndarray,
    elements: np.ndarray,
    *,
    cell_scalar: np.ndarray | None = None,
    vertex_scalar: np.ndarray | None = None,
    colorscale: str = "Viridis",
    colorbar_title: str = "",
    title: str = "",
    symmetric: bool = False,
    height: int = 480,
    precomputed_geometry: dict | None = None,
) -> go.Figure:
    """Render the boundary surface of a hex mesh as a Plotly Mesh3d.

    Pass ``cell_scalar`` (one value per element) for FE field data like
    sigma_33 or max FI; pass ``vertex_scalar`` (one value per node) for
    per-node fields like displacement magnitude.

    ``precomputed_geometry`` is the dict returned by
    :func:`compute_mesh3d_geometry`.  When provided, the (expensive)
    boundary cull + triangulation are skipped — this is the hot path for
    Streamlit slider re-renders where ``elements`` is unchanged between
    calls.  When ``None`` (default), the geometry is computed inline,
    preserving the original single-argument API.
    """
    if precomputed_geometry is None:
        precomputed_geometry = compute_mesh3d_geometry(elements)
    tri = precomputed_geometry["tri"]
    kept_nodes = precomputed_geometry["kept_nodes"]
    tri_ijk = precomputed_geometry["tri_ijk"]

    # Slice the vertex payload to the surface nodes.  ``vertices`` can
    # change every call (e.g. deformed-mesh view re-applies the scaled
    # displacement) so this lookup stays per-call even when the
    # connectivity-derived geometry is cached.
    if kept_nodes.size:
        verts = np.asarray(vertices)[kept_nodes]
    else:
        verts = np.asarray(vertices)[:0]

    kwargs: dict = dict(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=tri_ijk[:, 0],
        j=tri_ijk[:, 1],
        k=tri_ijk[:, 2],
        flatshading=True,
    )
    if cell_scalar is not None:
        intensity = np.asarray(cell_scalar)[tri[:, 3]]
        kwargs.update(
            intensity=intensity,
            intensitymode="cell",
            colorscale=colorscale,
            colorbar=dict(title=colorbar_title, len=0.7),
            showscale=True,
        )
        if symmetric:
            vmax = float(np.nanmax(np.abs(intensity)))
            if vmax > 0:
                kwargs.update(cmin=-vmax, cmax=vmax)
    elif vertex_scalar is not None:
        # Slice the per-node scalar to match the trimmed vertex array.
        vs = np.asarray(vertex_scalar)
        kwargs.update(
            intensity=vs[kept_nodes] if kept_nodes.size else vs[:0],
            intensitymode="vertex",
            colorscale=colorscale,
            colorbar=dict(title=colorbar_title, len=0.7),
            showscale=True,
        )
    else:
        kwargs.update(color=MORPHOLOGY_COLORS["stack"], showscale=False)

    fig = go.Figure(go.Mesh3d(**kwargs))
    fig.update_layout(
        title=title,
        scene=_scene_layout(),
        margin=dict(l=0, r=0, t=40, b=0),
        height=height,
    )
    return fig


def stress_contour_figure(
    nodes: np.ndarray,
    elements: np.ndarray,
    stress_per_elem: np.ndarray,
    component_index: int = 2,
    *,
    component_label: str = "σ₃₃",
    title: str | None = None,
    precomputed_geometry: dict | None = None,
) -> go.Figure:
    """3D surface mesh coloured by a single Voigt stress component."""
    scalar = stress_per_elem[:, component_index]
    return mesh3d_figure(
        nodes,
        elements,
        cell_scalar=scalar,
        colorscale="RdBu_r",
        colorbar_title=f"{component_label} [MPa]",
        title=title or f"{component_label} surface contour",
        symmetric=True,
        precomputed_geometry=precomputed_geometry,
    )


def deformed_mesh_figure(
    nodes: np.ndarray,
    elements: np.ndarray,
    displacement: np.ndarray,
    *,
    scale: float = 10.0,
    precomputed_geometry: dict | None = None,
) -> go.Figure:
    """3D deformed mesh coloured by displacement magnitude."""
    deformed = nodes + scale * displacement
    disp_mag = np.linalg.norm(displacement, axis=1)
    return mesh3d_figure(
        deformed,
        elements,
        vertex_scalar=disp_mag,
        colorscale="Viridis",
        colorbar_title="|u| [mm]",
        title=f"Deformed mesh (×{scale:g} exaggeration)",
        precomputed_geometry=precomputed_geometry,
    )


def fi_3d_figure(
    nodes: np.ndarray,
    elements: np.ndarray,
    fi_per_gauss: np.ndarray,
    criterion: str,
    *,
    precomputed_geometry: dict | None = None,
) -> go.Figure:
    """3D surface mesh coloured by per-element max failure index."""
    fi_max = np.asarray(fi_per_gauss).max(axis=1)
    return mesh3d_figure(
        nodes,
        elements,
        cell_scalar=fi_max,
        colorscale="Reds",
        colorbar_title=f"FI ({criterion})",
        title=f"Failure index — {criterion}",
        precomputed_geometry=precomputed_geometry,
    )


def y_slice_figure(
    element_centers: np.ndarray,
    elements: np.ndarray,
    nodes: np.ndarray,
    stress_per_elem: np.ndarray,
    component_index: int,
    y_station: float,
    *,
    component_label: str = "σ₃₃",
) -> go.Figure | None:
    """Filter elements at the given y-station and render a 2D scatter
    in the (x, z) plane coloured by the chosen stress component.

    The 'slice' is the layer of elements whose y-row matches the station;
    we pick the elements whose centre-y is closest to the station to
    avoid empty slices on coarse meshes.
    """
    yc = element_centers[:, 1]
    unique_y = np.unique(yc)
    if unique_y.size == 0:
        return None
    nearest_y = unique_y[np.argmin(np.abs(unique_y - y_station))]
    mask = yc == nearest_y
    if not mask.any():
        return None

    xs = element_centers[mask, 0]
    zs = element_centers[mask, 2]
    vals = stress_per_elem[mask, component_index]
    vmax = float(np.nanmax(np.abs(vals))) if vals.size else 1.0

    # Approximate per-element x and z extents from the bounding box of
    # its 8 nodes so the rectangular markers tile cleanly.
    elem_node_xyz = nodes[elements[mask]]
    dx = float(np.ptp(elem_node_xyz[:, :, 0], axis=1).mean()) if mask.sum() else 1.0
    dz = float(np.ptp(elem_node_xyz[:, :, 2], axis=1).mean()) if mask.sum() else 1.0

    fig = go.Figure(
        go.Scatter(
            x=xs,
            y=zs,
            mode="markers",
            marker=dict(
                size=14,
                color=vals,
                colorscale="RdBu_r",
                cmin=-vmax,
                cmax=vmax,
                symbol="square",
                line=dict(width=0),
                colorbar=dict(title=f"{component_label} [MPa]"),
            ),
            hovertemplate=(
                f"x = %{{x:.2f}} mm<br>z = %{{y:.3f}} mm"
                f"<br>{component_label} = %{{marker.color:.1f}} MPa<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"{component_label} at y ≈ {nearest_y:.2f} mm  ·  Δx≈{dx:.2f} mm  Δz≈{dz:.3f} mm",
        xaxis_title="x [mm]",
        yaxis_title="z [mm]",
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def fi_y_slice_figure(
    element_centers: np.ndarray,
    elements: np.ndarray,
    nodes: np.ndarray,
    fi_per_gauss: np.ndarray,
    y_station: float,
    *,
    criterion: str = "FI",
) -> go.Figure | None:
    """Filter elements at the given y-station and render a 2D scatter in
    the (x, z) plane coloured by the per-element max failure index for
    the given criterion.

    Mirrors :func:`y_slice_figure` but uses a sequential (Reds) colour
    scale starting at 0 since FI is non-negative.
    """
    yc = element_centers[:, 1]
    unique_y = np.unique(yc)
    if unique_y.size == 0:
        return None
    nearest_y = unique_y[np.argmin(np.abs(unique_y - y_station))]
    mask = yc == nearest_y
    if not mask.any():
        return None

    xs = element_centers[mask, 0]
    zs = element_centers[mask, 2]
    fi_max_per_elem = np.asarray(fi_per_gauss).max(axis=1)
    vals = fi_max_per_elem[mask]
    vmax = float(np.nanmax(vals)) if vals.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0

    elem_node_xyz = nodes[elements[mask]]
    dx = float(np.ptp(elem_node_xyz[:, :, 0], axis=1).mean()) if mask.sum() else 1.0
    dz = float(np.ptp(elem_node_xyz[:, :, 2], axis=1).mean()) if mask.sum() else 1.0

    fig = go.Figure(
        go.Scatter(
            x=xs,
            y=zs,
            mode="markers",
            marker=dict(
                size=14,
                color=vals,
                colorscale="Reds",
                cmin=0.0,
                cmax=vmax,
                symbol="square",
                line=dict(width=0),
                colorbar=dict(title=f"FI ({criterion})"),
            ),
            hovertemplate=(
                f"x = %{{x:.2f}} mm<br>z = %{{y:.3f}} mm"
                f"<br>FI ({criterion}) = %{{marker.color:.3f}}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=(
            f"FI ({criterion}) at y ≈ {nearest_y:.2f} mm  ·  "
            f"Δx≈{dx:.2f} mm  Δz≈{dz:.3f} mm"
        ),
        xaxis_title="x [mm]",
        yaxis_title="z [mm]",
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig
