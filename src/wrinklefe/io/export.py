"""Export utilities for WrinkleFE results, meshes, and field data.

Provides functions to export:

- Analysis results to JSON for post-processing and archiving.
- Mesh and laminate data to Abaqus ``.inp`` format for commercial FE solvers.
- Mesh and field results to VTK legacy format for ParaView visualisation.

References
----------
Abaqus Analysis User's Manual, Dassault Systemes (node/element format).
VTK File Formats, Kitware (legacy unstructured grid format).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from wrinklefe.analysis import AnalysisResults
    from wrinklefe.core.laminate import Laminate
    from wrinklefe.core.mesh import MeshData
    from wrinklefe.solver.results import FieldResults


# ====================================================================== #
# JSON export
# ====================================================================== #

def export_results_json(
    results: "AnalysisResults",
    filepath: Union[str, Path],
) -> None:
    """Export an :class:`~wrinklefe.analysis.AnalysisResults` object to JSON.

    Serialises the configuration, analytical predictions, and summary
    statistics.  Large array data (FE fields, Monte Carlo samples) is
    summarised rather than written in full.

    Parameters
    ----------
    results : AnalysisResults
        The analysis results to export.
    filepath : str or Path
        Output JSON file path.
    """
    cfg = results.config
    data: dict = {
        "wrinklefe_version": "0.1.0",
        "configuration": {
            "amplitude_mm": cfg.amplitude,
            "wavelength_mm": cfg.wavelength,
            "width_mm": cfg.width,
            "morphology": cfg.morphology,
            "loading": cfg.loading,
            "material": cfg.material.name if cfg.material else None,
            "ply_thickness_mm": cfg.ply_thickness,
            "n_plies": len(cfg.angles) if cfg.angles else 0,
            "applied_strain": cfg.applied_strain,
            "solver": cfg.solver,
        },
        "analytical_predictions": {
            "morphology_factor": float(results.morphology_factor),
            "max_angle_rad": float(results.max_angle_rad),
            "max_angle_deg": float(np.degrees(results.max_angle_rad)),
            "effective_angle_rad": float(results.effective_angle_rad),
            "effective_angle_deg": float(np.degrees(results.effective_angle_rad)),
            "damage_index": float(results.damage_index),
            "analytical_knockdown": float(results.analytical_knockdown),
            "analytical_strength_MPa": float(results.analytical_strength_MPa),
        },
    }

    # Mesh summary
    if results.mesh is not None:
        data["mesh"] = {
            "n_nodes": results.mesh.n_nodes,
            "n_elements": results.mesh.n_elements,
            "n_dof": results.mesh.n_dof,
            "domain_size_mm": list(results.mesh.domain_size),
        }

    # FE field summary
    if results.field_results is not None:
        max_disp, max_disp_node = results.field_results.max_displacement()
        data["fe_results"] = {
            "max_displacement_mm": float(max_disp),
            "max_displacement_node": int(max_disp_node),
        }

    # Buckling summary (optional attribute)
    buckling = getattr(results, "buckling_result", None)
    if buckling is not None:
        data["buckling"] = {
            "critical_load_factor": float(buckling.critical_load_factor),
        }

    # Monte Carlo summary (optional attribute)
    mc = getattr(results, "mc_results", None)
    if mc is not None:
        data["monte_carlo"] = {
            "n_samples": mc.n_samples,
            "mean_strength_MPa": float(mc.mean_strength),
            "std_strength_MPa": float(mc.std_strength),
            "cov_strength": float(mc.cov_strength),
            "min_strength_MPa": float(mc.min_strength),
            "percentile_5_MPa": float(mc.percentile_5),
            "percentile_1_MPa": float(mc.percentile_1),
        }

    # Jensen gap summary (optional attribute)
    jg = getattr(results, "jensen_gap", None)
    if jg is not None:
        data["jensen_gap"] = {
            "strength_at_mean_MPa": float(jg.strength_at_mean),
            "mean_of_strengths_MPa": float(jg.mean_of_strengths),
            "jensen_gap_MPa": float(jg.jensen_gap),
            "jensen_gap_percent": float(jg.jensen_gap_percent),
            "mean_amplitude_mm": float(jg.mean_amplitude),
            "mean_wavelength_mm": float(jg.mean_wavelength),
            "mean_angle_rad": float(jg.mean_angle),
        }

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ====================================================================== #
# Abaqus .inp export
# ====================================================================== #

def export_abaqus_inp(
    mesh: "MeshData",
    laminate: "Laminate",
    filepath: Union[str, Path],
) -> None:
    """Export mesh and laminate data to Abaqus ``.inp`` format.

    Writes nodes (1-based), hex8 elements grouped by ply, and boundary
    face node sets.  This provides a starting point for Abaqus analysis;
    material sections, loads, and step definitions must be added by the
    user.

    Parameters
    ----------
    mesh : MeshData
        The finite element mesh.
    laminate : Laminate
        The laminate definition (used for ply count and grouping).
    filepath : str or Path
        Output file path (should end with ``.inp``).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    n_plies = laminate.n_plies

    with open(filepath, "w") as f:
        # Header
        f.write("*HEADING\n")
        f.write("WrinkleFE - Wrinkled composite laminate mesh\n")
        f.write(
            f"** {mesh.n_nodes} nodes, {mesh.n_elements} C3D8 elements, "
            f"{n_plies} plies\n"
        )
        f.write("**\n")

        # Nodes (1-based indexing)
        f.write("*NODE\n")
        for nid in range(mesh.n_nodes):
            x, y, z = mesh.nodes[nid]
            f.write(f"{nid + 1:8d}, {x:14.6e}, {y:14.6e}, {z:14.6e}\n")

        # Elements grouped by ply
        for ply_idx in range(n_plies):
            elem_ids = mesh.elements_in_ply(ply_idx)
            if elem_ids.size == 0:
                continue
            f.write(f"**\n")
            f.write(f"*ELEMENT, TYPE=C3D8, ELSET=PLY_{ply_idx}\n")
            for eid in elem_ids:
                conn = mesh.elements[eid] + 1  # Convert to 1-based
                conn_str = ", ".join(str(int(n)) for n in conn)
                f.write(f"{eid + 1:8d}, {conn_str}\n")

        # Node sets for boundary faces
        f.write("**\n")
        f.write("** BOUNDARY NODE SETS\n")
        for face in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
            face_nodes = mesh.nodes_on_face(face) + 1  # 1-based
            f.write(f"*NSET, NSET={face.upper()}\n")
            for line_start in range(0, len(face_nodes), 16):
                chunk = face_nodes[line_start: line_start + 16]
                f.write(", ".join(str(int(n)) for n in chunk) + "\n")

        f.write("**\n")
        f.write("** END OF MESH DATA\n")
        f.write("** Add material sections, loads, and step definitions below.\n")


# ====================================================================== #
# VTK export
# ====================================================================== #

def export_vtk(
    mesh: "MeshData",
    field_results: Optional["FieldResults"],
    filepath: Union[str, Path],
) -> None:
    """Export mesh and field data to VTK legacy format for ParaView.

    Writes an unstructured grid (``.vtk``) file with hex8 cells.

    Point data written:

    - ``fiber_angle`` (radians) from mesh geometry
    - ``displacement`` (3-component vector, mm) if *field_results* is provided

    Cell data written:

    - ``ply_id`` (integer)
    - ``ply_angle`` (degrees)
    - ``stress_11`` (MPa, element average at first Gauss point) if
      *field_results* is provided

    Parameters
    ----------
    mesh : MeshData
        The finite element mesh.
    field_results : FieldResults or None
        FE solution fields.  If ``None``, only mesh geometry and fiber
        angles are written.
    filepath : str or Path
        Output file path (should end with ``.vtk``).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    n_nodes = mesh.n_nodes
    n_elem = mesh.n_elements

    with open(filepath, "w") as f:
        # VTK header
        f.write("# vtk DataFile Version 3.0\n")
        f.write("WrinkleFE - Wrinkled composite laminate\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        # Points
        f.write(f"POINTS {n_nodes} double\n")
        for nid in range(n_nodes):
            x, y, z = mesh.nodes[nid]
            f.write(f"{x:.8e} {y:.8e} {z:.8e}\n")

        # Cells
        total_ints = n_elem * 9  # 8 nodes + 1 count per element
        f.write(f"\nCELLS {n_elem} {total_ints}\n")
        for eid in range(n_elem):
            conn = mesh.elements[eid]
            conn_str = " ".join(str(int(n)) for n in conn)
            f.write(f"8 {conn_str}\n")

        # Cell types (12 = VTK_HEXAHEDRON)
        f.write(f"\nCELL_TYPES {n_elem}\n")
        for _ in range(n_elem):
            f.write("12\n")

        # --- Point data ---
        f.write(f"\nPOINT_DATA {n_nodes}\n")

        # Fiber angle
        f.write("SCALARS fiber_angle double 1\n")
        f.write("LOOKUP_TABLE default\n")
        for nid in range(n_nodes):
            f.write(f"{mesh.fiber_angles[nid]:.8e}\n")

        # Displacement (if available)
        if field_results is not None:
            f.write("VECTORS displacement double\n")
            for nid in range(n_nodes):
                ux, uy, uz = field_results.displacement[nid]
                f.write(f"{ux:.8e} {uy:.8e} {uz:.8e}\n")

        # --- Cell data ---
        f.write(f"\nCELL_DATA {n_elem}\n")

        # Ply ID
        f.write("SCALARS ply_id int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for eid in range(n_elem):
            f.write(f"{int(mesh.ply_ids[eid])}\n")

        # Ply angle
        f.write("SCALARS ply_angle double 1\n")
        f.write("LOOKUP_TABLE default\n")
        for eid in range(n_elem):
            f.write(f"{mesh.ply_angles[eid]:.4f}\n")

        # Stress sigma_11 (if available) -- element average at first Gauss point
        if field_results is not None:
            f.write("SCALARS stress_11 double 1\n")
            f.write("LOOKUP_TABLE default\n")
            for eid in range(n_elem):
                s11 = field_results.stress_global[eid, 0, 0]
                f.write(f"{s11:.6e}\n")
