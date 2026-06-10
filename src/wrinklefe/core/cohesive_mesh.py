"""Generic mesh utility for inserting zero-thickness cohesive interfaces.

Given an existing structured hex8 :class:`~wrinklefe.core.mesh.MeshData`
and a z-plane, :func:`insert_cohesive_interface` returns a new mesh with
duplicated nodes along the interface plane and a list of fully wired
:class:`~wrinklefe.elements.cohesive8.Cohesive8Element` instances ready
to be passed to :class:`~wrinklefe.solver.assembler.GlobalAssembler`.

The algorithm is purely topological and works on any structured-hex mesh
whose elements either have **0 nodes** or **exactly 4 nodes** on the
interface plane (this is the natural case for a horizontal interior
plane in a stacked structured grid).  Any other count signals a topology
the routine cannot handle (e.g. an element spans the plane with a side
face on it, or the interface plane intersects mid-element).

References
----------
Camanho, P.P. & Davila, C.G. (2002). NASA/TM-2002-211737, section 3
("inserted cohesive elements").
"""

from __future__ import annotations

import numpy as np

from wrinklefe.core.mesh import MeshData
from wrinklefe.elements.cohesive8 import (
    Cohesive8Element,
    CohesiveProperties,
)

# Hex8 face definitions following the VTK / Abaqus convention used by
# :class:`~wrinklefe.core.mesh.WrinkleMesh`:
#
#   Bottom face (k):   n0(i,j,k), n1(i+1,j,k), n2(i+1,j+1,k), n3(i,j+1,k)
#   Top face    (k+1): n4(i,j,k+1), n5(i+1,j,k+1), n6(i+1,j+1,k+1), n7(i,j+1,k+1)
#
# When an element sits **above** a horizontal interface plane its bottom
# face (local indices 0..3) lies on the plane; an element **below** has its
# top face (local indices 4..7) on the plane.  These local-index quartets
# are the only orderings the routine ever produces.
_HEX_BOTTOM_FACE = (0, 1, 2, 3)
_HEX_TOP_FACE = (4, 5, 6, 7)


def insert_cohesive_interface(
    mesh: MeshData,
    z_interface: float,
    cohesive_props: CohesiveProperties,
    tolerance: float = 1e-9,
) -> tuple[MeshData, list[Cohesive8Element]]:
    """Insert a zero-thickness cohesive layer at ``z = z_interface``.

    The original mesh is not modified; a new :class:`MeshData` is built
    with expanded ``nodes`` array and modified ``elements`` connectivity.
    Elements whose centroid is above ``z_interface`` are re-wired to use
    duplicated nodes for the 4 face nodes on the interface plane; the
    cohesive elements connect each original (bottom) quad to its
    duplicated (top) counterpart with the standard 8-node ordering.

    Parameters
    ----------
    mesh : MeshData
        Source mesh.  Must use hex8 elements (8-node connectivity).
    z_interface : float
        Z-coordinate of the interface plane.
    cohesive_props : CohesiveProperties
        Material properties for the bilinear traction-separation law.
        The same instance is shared by every generated cohesive element.
    tolerance : float, optional
        Absolute tolerance for matching node z-coordinates to
        ``z_interface``.  Default 1e-9 mm.

    Returns
    -------
    new_mesh : MeshData
        Mesh with the duplicated interface nodes and updated hex8
        connectivity.  ``nodes`` has ``n_interface`` extra rows appended
        after the original node block; the duplicates carry the same
        ``fiber_angle`` as the originals.  ``elements``, ``ply_ids``,
        ``ply_angles`` arrays have the same shape as the input — only
        the connectivity of "above" elements changes.
    cohesive_elements : list[Cohesive8Element]
        One element per quad on the interface plane, with ``node_ids``
        set to ``[bottom_quad..., top_quad...]`` (8-tuple) following the
        :class:`Cohesive8Element` convention.  Each element's
        ``elem_id`` is its 0-based index in this list.

    Raises
    ------
    ValueError
        If no nodes are found within ``tolerance`` of ``z_interface``,
        or if any element has 1, 2, 3, 5, 6, 7, or 8 nodes on the
        interface plane (only 0 or 4 are valid).
    """
    nodes = np.asarray(mesh.nodes, dtype=float)
    elements = np.asarray(mesh.elements, dtype=np.intp)
    if elements.ndim != 2 or elements.shape[1] != 8:
        raise ValueError(
            "insert_cohesive_interface requires hex8 elements "
            f"(shape (n_elem, 8)); got shape {elements.shape}."
        )

    # ------------------------------------------------------------------
    # 1. Find interface nodes
    # ------------------------------------------------------------------
    on_plane_mask = np.abs(nodes[:, 2] - z_interface) <= tolerance
    interface_node_ids = np.flatnonzero(on_plane_mask)
    if interface_node_ids.size == 0:
        raise ValueError(
            f"insert_cohesive_interface: no mesh nodes found within "
            f"tolerance={tolerance:g} of z_interface={z_interface:g}. "
            f"Node z-range is "
            f"[{float(nodes[:, 2].min()):g}, {float(nodes[:, 2].max()):g}]."
        )

    # ------------------------------------------------------------------
    # 2. Build the original -> duplicate node-id map
    # ------------------------------------------------------------------
    n_nodes_orig = nodes.shape[0]
    n_interface = interface_node_ids.size
    duplicate_ids = np.arange(
        n_nodes_orig, n_nodes_orig + n_interface, dtype=np.intp,
    )
    # Sparse "original -> duplicate" lookup; -1 for nodes not on the plane.
    orig_to_dup = np.full(n_nodes_orig, -1, dtype=np.intp)
    orig_to_dup[interface_node_ids] = duplicate_ids

    new_nodes = np.vstack([nodes, nodes[interface_node_ids]])

    # ------------------------------------------------------------------
    # 3. Classify each element + rewire "above" elements
    # ------------------------------------------------------------------
    new_elements = elements.copy()
    # Vectorised tally of interface-plane node count per element.
    on_plane_per_elem = on_plane_mask[elements]   # (n_elem, 8) bool
    count_per_elem = on_plane_per_elem.sum(axis=1)  # (n_elem,)

    valid_counts = (count_per_elem == 0) | (count_per_elem == 4)
    if not bool(np.all(valid_counts)):
        bad_indices = np.flatnonzero(~valid_counts)
        # Report up to 10 to keep error messages bounded.
        bad_show = bad_indices[:10].tolist()
        tail = "" if bad_indices.size <= 10 else (
            f", ... ({int(bad_indices.size) - 10} more)"
        )
        bad_counts = count_per_elem[bad_indices[:10]].tolist()
        raise ValueError(
            "insert_cohesive_interface: mesh topology not "
            "structured-hex compatible at the interface plane.  "
            f"Elements with an invalid number of interface-plane nodes "
            f"(expected 0 or 4): indices {bad_show} have counts "
            f"{bad_counts}{tail}."
        )

    # Element classification:
    #   * count = 0: untouched
    #   * count = 4: element has one face on the interface plane.  Use the
    #     element centroid to decide whether it sits ABOVE (top face is
    #     interior, bottom face on plane -> rewire bottom face to
    #     duplicates) or BELOW (top face on plane, bottom interior -> keep
    #     the originals).
    elem_centroids_z = nodes[elements][:, :, 2].mean(axis=1)  # (n_elem,)
    above_mask = (count_per_elem == 4) & (elem_centroids_z > z_interface)
    below_mask = (count_per_elem == 4) & (elem_centroids_z < z_interface)

    # Catch the rare-but-possible "centroid exactly on the plane" case:
    # if it happens for an element with a non-zero interface-node count
    # the topology is ambiguous (a 2D-thick degenerate slab) and we can't
    # decide above/below by centroid alone.
    ambiguous = (count_per_elem == 4) & ~(above_mask | below_mask)
    if bool(np.any(ambiguous)):
        bad = np.flatnonzero(ambiguous)[:10].tolist()
        raise ValueError(
            "insert_cohesive_interface: cannot classify elements as "
            "above/below the interface plane (centroid on plane).  "
            f"Offending element indices: {bad}."
        )

    # For "above" elements, replace the 4 interface-plane node-ids with
    # their duplicates.  on_plane_per_elem is a bool mask aligned with
    # `elements`; we use np.where for a clean vectorised swap.
    for eid in np.flatnonzero(above_mask):
        conn = elements[eid]
        mask = on_plane_per_elem[eid]
        new_elements[eid, mask] = orig_to_dup[conn[mask]]

    # ------------------------------------------------------------------
    # 4. Build cohesive elements
    # ------------------------------------------------------------------
    # Each below-element contributes one cohesive quad (its top face, local
    # indices 4..7 = _HEX_TOP_FACE).  Pair the original interface nodes
    # (bottom of cohesive) with the corresponding duplicates (top of
    # cohesive).  The (x, y) ordering of the bottom quad is preserved by
    # the original hex8 connectivity, so the resulting 8-node cohesive
    # element naturally satisfies the documented convention "node i+4
    # sits on node i".
    cohesive_elements: list[Cohesive8Element] = []
    below_indices = np.flatnonzero(below_mask)
    for k, eid in enumerate(below_indices):
        # Top face of a below-element is on the interface plane.  Use the
        # canonical _HEX_TOP_FACE ordering (CCW from +x normal viewing
        # toward -z); the corresponding duplicates sit at orig_to_dup[id].
        bottom_quad = elements[eid, list(_HEX_TOP_FACE)]
        # Sanity: every node should be on the plane.
        if not bool(np.all(on_plane_mask[bottom_quad])):
            # Element passed the count test but the top face is not
            # entirely on the plane — happens when the 4 interface nodes
            # are split across the bottom and top faces of the hex.  We
            # don't yet support that orientation.
            raise ValueError(
                f"insert_cohesive_interface: element {int(eid)} has 4 "
                "interface-plane nodes but they are not all on the hex "
                "top face.  This element's interface face is not "
                "axis-aligned with z; unsupported topology."
            )
        top_quad = orig_to_dup[bottom_quad]

        coh_node_ids = np.concatenate([bottom_quad, top_quad]).astype(
            np.intp
        )
        coh_coords = new_nodes[coh_node_ids]

        cohesive_elements.append(
            Cohesive8Element(
                node_coords=coh_coords,
                properties=cohesive_props,
                node_ids=coh_node_ids,
                elem_id=int(k),
            )
        )

    # Also verify: every "above" element exposes its bottom face on the
    # plane.  Same check as for "below"; catches mis-aligned interfaces.
    for eid in np.flatnonzero(above_mask):
        bottom_quad = elements[eid, list(_HEX_BOTTOM_FACE)]
        if not bool(np.all(on_plane_mask[bottom_quad])):
            raise ValueError(
                f"insert_cohesive_interface: element {int(eid)} has 4 "
                "interface-plane nodes but they are not all on the hex "
                "bottom face.  This element's interface face is not "
                "axis-aligned with z; unsupported topology."
            )

    # ------------------------------------------------------------------
    # 5. Build the new MeshData (preserve everything else)
    # ------------------------------------------------------------------
    fiber_angles = np.asarray(mesh.fiber_angles, dtype=float)
    new_fiber_angles = np.concatenate(
        [fiber_angles, fiber_angles[interface_node_ids]]
    )

    new_mesh = MeshData(
        nodes=new_nodes,
        elements=new_elements,
        ply_ids=np.asarray(mesh.ply_ids, dtype=np.intp).copy(),
        fiber_angles=new_fiber_angles,
        ply_angles=np.asarray(mesh.ply_angles, dtype=float).copy(),
        nx=int(mesh.nx),
        ny=int(mesh.ny),
        nz=int(mesh.nz),
        laminate=mesh.laminate,
    )
    return new_mesh, cohesive_elements
