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

    Returns
    -------
    (n_boundary_faces, 5) array. Columns 0-3 are global node indices in
    the element's local face ordering; column 4 is the parent element id.
    """
    face_owners: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for ei in range(elements.shape[0]):
        conn = elements[ei]
        for fi, face in enumerate(HEX_FACES):
            key = tuple(sorted(int(v) for v in conn[face]))
            face_owners.setdefault(key, []).append((ei, fi))

    out: list[list[int]] = []
    for owners in face_owners.values():
        if len(owners) == 1:
            ei, fi = owners[0]
            face = HEX_FACES[fi]
            out.append([int(elements[ei][n]) for n in face] + [ei])
    return np.asarray(out, dtype=np.int64)


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
) -> go.Figure:
    """Render the boundary surface of a hex mesh as a Plotly Mesh3d.

    Pass ``cell_scalar`` (one value per element) for FE field data like
    sigma_33 or max FI; pass ``vertex_scalar`` (one value per node) for
    per-node fields like displacement magnitude.
    """
    bf = boundary_faces(elements)
    tri = quads_to_triangles(bf)

    kwargs: dict = dict(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=tri[:, 0],
        j=tri[:, 1],
        k=tri[:, 2],
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
        kwargs.update(
            intensity=np.asarray(vertex_scalar),
            intensitymode="vertex",
            colorscale=colorscale,
            colorbar=dict(title=colorbar_title, len=0.7),
            showscale=True,
        )
    else:
        kwargs.update(color="#9ec5fe", showscale=False)

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
    )


def deformed_mesh_figure(
    nodes: np.ndarray,
    elements: np.ndarray,
    displacement: np.ndarray,
    *,
    scale: float = 10.0,
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
    )


def fi_3d_figure(
    nodes: np.ndarray,
    elements: np.ndarray,
    fi_per_gauss: np.ndarray,
    criterion: str,
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
