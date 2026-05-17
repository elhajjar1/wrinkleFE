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
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

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
    - ``stress_11``, ``stress_22``, ``stress_33``, ``stress_23``,
      ``stress_13``, ``stress_12`` (MPa, true element average over the
      element's Gauss points) if *field_results* is provided.  Each value
      is the mean of the corresponding global-frame stress component over
      all Gauss points in the element, so the field is a genuine
      per-element representative value rather than a single sampled point.

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

        # Stress components (if available) -- true element average over the
        # element's Gauss points.  Voigt ordering of stress_global is
        # [sigma_11, sigma_22, sigma_33, tau_23, tau_13, tau_12].
        if field_results is not None and field_results.stress_global.size > 0:
            stress = field_results.stress_global  # (n_elem, n_gauss, 6)
            # Mean over Gauss-point axis -> (n_elem, 6)
            elem_avg = stress.mean(axis=1)
            component_names = (
                "stress_11",
                "stress_22",
                "stress_33",
                "stress_23",
                "stress_13",
                "stress_12",
            )
            for comp_idx, name in enumerate(component_names):
                f.write(f"SCALARS {name} double 1\n")
                f.write("LOOKUP_TABLE default\n")
                for eid in range(n_elem):
                    f.write(f"{elem_avg[eid, comp_idx]:.6e}\n")


# ====================================================================== #
# Nonconformance Report (NCR) — MRB decision-support export
# ====================================================================== #
#
# These helpers turn a WrinkleFE analysis of an out-of-plane fibre wrinkle
# into a structured Nonconformance Report for a field engineer to raise and
# for a Material Review Board (MRB) to review.
#
# Scope/authority note: the recommendation produced here is *decision
# support only*.  It does not constitute a final disposition.  Severity
# thresholds below are generic engineering guidance and MUST be superseded
# by the program-specific allowables, drawing requirements, and process
# specifications that the MRB applies.  The qualified MRB reviews, may
# modify, and approves the final disposition.

# Residual-strength fraction (= analytical knockdown) and damage-index
# severity bands.  The worst (lowest) tier from either metric governs.
_NCR_SEVERITY_BANDS = (
    # (label, min_knockdown, max_damage_index, recommended_path, approvals)
    (
        "Negligible",
        0.97,
        0.05,
        "Candidate for USE-AS-IS, contingent on confirming residual "
        "strength ≥ design allowable for the affected location.",
        ("Originating/Design Engineering",),
    ),
    (
        "Minor",
        0.90,
        0.20,
        "USE-AS-IS with documented stress justification, or a cosmetic "
        "blend/local rework if the wrinkle is surface-accessible.",
        ("Design Engineering", "Quality"),
    ),
    (
        "Moderate",
        0.75,
        0.40,
        "Engineering disposition required: REPAIR per an approved scheme, "
        "or USE-AS-IS only if a positive margin of safety is demonstrated "
        "by substantiating analysis or test.",
        ("Design Engineering", "Stress", "Quality"),
    ),
    (
        "Major",
        0.50,
        0.65,
        "REPAIR or REWORK per a qualified procedure with full MRB "
        "substantiation. Customer/DER concurrence is likely required.",
        ("Design Engineering", "Stress", "Quality", "Customer/DER"),
    ),
    (
        "Severe",
        0.0,
        1.01,
        "REJECT — SCRAP, or major REPAIR only under an engineering-"
        "approved, fully substantiated scheme. Mandatory customer/DER "
        "review.",
        ("Design Engineering", "Stress", "Quality", "Customer/DER",
         "Program Management"),
    ),
)


def recommend_disposition(
    knockdown: float,
    damage_index: float,
    *,
    loading: Optional[str] = None,
) -> dict:
    """Classify wrinkle severity and recommend a (non-binding) MRB path.

    The recommendation is decision support for a Material Review Board.
    It is **not** a final disposition: the MRB reviews, may modify, and
    approves the outcome against program-specific allowables.

    Parameters
    ----------
    knockdown : float
        Residual strength fraction of the wrinkled laminate (0–1,
        where 1.0 = no strength loss).
    damage_index : float
        WrinkleFE damage index D (0 = pristine, 1 = full loss of
        load-carrying capacity).
    loading : str, optional
        ``'compression'`` or ``'tension'``.  Used only to annotate the
        rationale; compression-dominated wrinkles are generally the more
        severe and less tolerant case.

    Returns
    -------
    dict
        Keys: ``severity``, ``recommended_path``, ``required_approvals``,
        ``rationale``, ``governed_by``, and ``is_final`` (always
        ``False``).
    """
    kd = float(knockdown)
    di = float(damage_index)

    # Pick the worst (latest) band each metric falls into, then take the
    # more conservative of the two.
    by_knockdown = len(_NCR_SEVERITY_BANDS) - 1
    for idx, (_, min_kd, _, _, _) in enumerate(_NCR_SEVERITY_BANDS):
        if kd >= min_kd:
            by_knockdown = idx
            break

    by_damage = 0
    for idx, (_, _, max_di, _, _) in enumerate(_NCR_SEVERITY_BANDS):
        if di < max_di:
            by_damage = idx
            break
    else:
        by_damage = len(_NCR_SEVERITY_BANDS) - 1

    band_idx = max(by_knockdown, by_damage)
    label, _, _, path, approvals = _NCR_SEVERITY_BANDS[band_idx]

    if by_knockdown >= by_damage:
        governed_by = "residual strength (analytical knockdown)"
    else:
        governed_by = "damage index D"

    strength_loss_pct = max(0.0, (1.0 - kd) * 100.0)
    rationale = (
        f"Predicted residual strength is {kd * 100.0:.1f}% of pristine "
        f"(≈{strength_loss_pct:.1f}% strength loss); damage index "
        f"D = {di:.3f}. Severity is governed by {governed_by}."
    )
    if loading:
        lo = str(loading).lower()
        if "compress" in lo:
            rationale += (
                " Loading is compression-dominated: fibre wrinkles are "
                "least tolerant under compression (kink-band / micro-"
                "buckling driven), so treat the recommendation "
                "conservatively."
            )
        elif "tens" in lo:
            rationale += (
                " Loading is tension-dominated: wrinkles are generally "
                "more tolerant in tension, but verify the governing "
                "failure mode against the drawing requirement."
            )

    return {
        "severity": label,
        "recommended_path": path,
        "required_approvals": list(approvals),
        "rationale": rationale,
        "governed_by": governed_by,
        "is_final": False,
    }


def build_ncr(
    *,
    metadata: Optional[dict] = None,
    defect: dict,
    engineering: dict,
) -> dict:
    """Assemble a structured Nonconformance Report (NCR) for an MRB.

    Combines field-engineer metadata, the as-found wrinkle defect, and
    the WrinkleFE engineering analysis into a single structured report,
    including the cited criteria and a *recommended* (non-binding)
    disposition path for the Material Review Board.

    Parameters
    ----------
    metadata : dict, optional
        Field-engineer / QMS fields.  Recognised keys (all optional):
        ``ncr_number``, ``originator``, ``date``, ``part_number``,
        ``part_name``, ``serial_lot``, ``work_order``, ``program``,
        ``quantity``, ``defect_location``, ``detection_method``,
        ``drawing_spec``, ``remarks``.
    defect : dict
        As-found wrinkle + laminate.  Recognised keys: ``amplitude_mm``,
        ``wavelength_mm``, ``width_mm``, ``morphology``, ``loading``,
        ``ply_thickness_mm``, ``n_plies``, ``layup``, ``material_name``.
    engineering : dict
        WrinkleFE results.  Recognised keys: ``analytical_knockdown``,
        ``analytical_strength_MPa``, ``damage_index``, ``max_angle_deg``,
        ``effective_angle_deg``, ``morphology_factor``, and an optional
        ``fe`` sub-dict (``modulus_retention``, ``retention_factors``,
        ``critical_criterion``, ``critical_mode``, ``critical_ply``).

    Returns
    -------
    dict
        The structured NCR.  JSON-serialisable.
    """
    meta = dict(metadata or {})

    def _m(key: str, default: str = "(to be assigned)") -> Any:
        val = meta.get(key)
        if val is None or (isinstance(val, str) and not val.strip()):
            return default
        return val

    date = _m("date", datetime.now(timezone.utc).date().isoformat())

    kd = float(engineering.get("analytical_knockdown", 1.0))
    di = float(engineering.get("damage_index", 0.0))
    loading = defect.get("loading")

    disposition = recommend_disposition(kd, di, loading=loading)

    fe = engineering.get("fe") or None
    fe_block: Optional[dict] = None
    criteria = [
        "WrinkleFE closed-form knockdown model (effective fibre-"
        "misalignment / morphology-factor formulation) — analytical "
        "residual strength and damage index.",
    ]
    if fe:
        retention = fe.get("retention_factors") or {}
        min_ret = min(retention.values()) if retention else None
        fe_block = {
            "modulus_retention": fe.get("modulus_retention"),
            "min_strength_retention": min_ret,
            "retention_factors": retention,
            "critical_criterion": fe.get("critical_criterion"),
            "critical_mode": fe.get("critical_mode"),
            "critical_ply": fe.get("critical_ply"),
        }
        crit = fe.get("critical_criterion")
        if crit:
            criteria.append(
                f"3D finite-element ply failure evaluation — governing "
                f"criterion: {crit}"
                + (f" (mode: {fe.get('critical_mode')})"
                   if fe.get("critical_mode") else "")
                + "."
            )
    criteria.append(
        "Generic severity banding in this report is advisory only and is "
        "superseded by the program-specific allowables, drawing "
        "requirements, and process specifications applied by the MRB."
    )

    ncr = {
        "report_type": "Nonconformance Report (NCR)",
        "tool": "WrinkleFE MRB decision-support export",
        "header": {
            "ncr_number": _m("ncr_number"),
            "date": date,
            "originator": _m("originator", "(field engineer)"),
            "program": _m("program"),
            "part_number": _m("part_number"),
            "part_name": _m("part_name", "(not specified)"),
            "serial_or_lot": _m("serial_lot"),
            "work_order": _m("work_order"),
            "quantity_affected": _m("quantity", "1"),
        },
        "nonconformance": {
            "defect_type": "Out-of-plane fibre wrinkle (ply waviness)",
            "defect_location": _m("defect_location", "(not specified)"),
            "detection_method": _m("detection_method", "(not specified)"),
            "violated_requirement": _m(
                "drawing_spec",
                "(drawing/specification reference not supplied — MRB "
                "to confirm the controlling requirement)",
            ),
            "as_found_geometry": {
                "amplitude_mm": defect.get("amplitude_mm"),
                "wavelength_mm": defect.get("wavelength_mm"),
                "envelope_half_width_mm": defect.get("width_mm"),
                "morphology": defect.get("morphology"),
                "loading_condition": loading,
            },
            "laminate": {
                "material": defect.get("material_name"),
                "ply_thickness_mm": defect.get("ply_thickness_mm"),
                "n_plies": defect.get("n_plies"),
                "layup_deg": defect.get("layup"),
            },
            "originator_remarks": _m("remarks", "(none)"),
        },
        "engineering_analysis": {
            "analytical_knockdown": kd,
            "residual_strength_pct": round(kd * 100.0, 2),
            "predicted_strength_MPa": engineering.get(
                "analytical_strength_MPa"
            ),
            "damage_index_D": di,
            "max_fibre_misalignment_deg": engineering.get("max_angle_deg"),
            "effective_fibre_angle_deg": engineering.get(
                "effective_angle_deg"
            ),
            "morphology_factor": engineering.get("morphology_factor"),
            "finite_element": fe_block,
        },
        "criteria_cited": criteria,
        "disposition_recommendation": {
            "severity": disposition["severity"],
            "recommended_path": disposition["recommended_path"],
            "rationale": disposition["rationale"],
            "required_approvals": disposition["required_approvals"],
            "is_final_disposition": False,
            "note": (
                "Decision support only. The Material Review Board reviews, "
                "may modify, and approves the final disposition against "
                "the controlling drawing and program allowables."
            ),
        },
        "mrb_disposition": {
            "final_disposition": "",  # MRB to complete
            "mrb_rationale": "",
            "approvals": [
                {"role": role, "name": "", "signature": "", "date": ""}
                for role in disposition["required_approvals"]
            ],
        },
        "disclaimer": (
            "This NCR was prepared with WrinkleFE decision-support "
            "tooling. The analysis and recommendation are advisory and do "
            "not constitute a final material disposition. A qualified "
            "Material Review Board must review, may modify, and formally "
            "approve the disposition."
        ),
    }
    return ncr


def _fmt(value: Any, default: str = "—") -> str:
    if value is None or value == "":
        return default
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(str(v) for v in value) + "]"
    return str(value)


def render_ncr_markdown(ncr: dict) -> str:
    """Render a :func:`build_ncr` report as a human-readable Markdown form.

    Suitable for a field engineer to print/attach to the MRB package.

    Parameters
    ----------
    ncr : dict
        A report produced by :func:`build_ncr`.

    Returns
    -------
    str
        Markdown text.
    """
    h = ncr["header"]
    nc = ncr["nonconformance"]
    ea = ncr["engineering_analysis"]
    dr = ncr["disposition_recommendation"]
    geo = nc["as_found_geometry"]
    lam = nc["laminate"]

    lines: list[str] = []
    lines.append("# Nonconformance Report (NCR)")
    lines.append("")
    lines.append(f"**NCR No.:** {_fmt(h['ncr_number'])}  ")
    lines.append(f"**Date:** {_fmt(h['date'])}  ")
    lines.append(f"**Originator:** {_fmt(h['originator'])}  ")
    lines.append(f"**Program:** {_fmt(h['program'])}")
    lines.append("")
    lines.append("## 1. Part Identification")
    lines.append("")
    lines.append(f"- **Part number:** {_fmt(h['part_number'])}")
    lines.append(f"- **Part name:** {_fmt(h['part_name'])}")
    lines.append(f"- **Serial / lot:** {_fmt(h['serial_or_lot'])}")
    lines.append(f"- **Work order:** {_fmt(h['work_order'])}")
    lines.append(
        f"- **Quantity affected:** {_fmt(h['quantity_affected'])}"
    )
    lines.append("")
    lines.append("## 2. Nonconformance")
    lines.append("")
    lines.append(f"- **Defect type:** {_fmt(nc['defect_type'])}")
    lines.append(f"- **Location on part:** {_fmt(nc['defect_location'])}")
    lines.append(
        f"- **Detection method:** {_fmt(nc['detection_method'])}"
    )
    lines.append(
        f"- **Violated requirement:** {_fmt(nc['violated_requirement'])}"
    )
    lines.append("")
    lines.append("**As-found wrinkle geometry**")
    lines.append("")
    lines.append(f"- Amplitude: {_fmt(geo['amplitude_mm'])} mm")
    lines.append(f"- Wavelength: {_fmt(geo['wavelength_mm'])} mm")
    lines.append(
        f"- Envelope half-width: {_fmt(geo['envelope_half_width_mm'])} mm"
    )
    lines.append(f"- Morphology: {_fmt(geo['morphology'])}")
    lines.append(f"- Loading condition: {_fmt(geo['loading_condition'])}")
    lines.append("")
    lines.append("**Affected laminate**")
    lines.append("")
    lines.append(f"- Material: {_fmt(lam['material'])}")
    lines.append(
        f"- Ply thickness: {_fmt(lam['ply_thickness_mm'])} mm  "
        f"·  Plies: {_fmt(lam['n_plies'])}"
    )
    lines.append(f"- Layup (deg): {_fmt(lam['layup_deg'])}")
    lines.append("")
    lines.append(f"**Originator remarks:** {_fmt(nc['originator_remarks'])}")
    lines.append("")
    lines.append("## 3. Engineering Analysis (WrinkleFE)")
    lines.append("")
    lines.append(
        f"- Analytical knockdown: **{_fmt(ea['analytical_knockdown'])}** "
        f"({_fmt(ea['residual_strength_pct'])}% residual strength)"
    )
    lines.append(
        f"- Predicted strength: {_fmt(ea['predicted_strength_MPa'])} MPa"
    )
    lines.append(f"- Damage index D: **{_fmt(ea['damage_index_D'])}**")
    lines.append(
        f"- Max fibre misalignment: "
        f"{_fmt(ea['max_fibre_misalignment_deg'])}°"
    )
    lines.append(
        f"- Effective fibre angle: "
        f"{_fmt(ea['effective_fibre_angle_deg'])}°"
    )
    lines.append(
        f"- Morphology factor: {_fmt(ea['morphology_factor'])}"
    )
    fe_block = ea.get("finite_element")
    if fe_block:
        lines.append("")
        lines.append("**Finite-element evaluation**")
        lines.append("")
        lines.append(
            f"- Modulus retention: {_fmt(fe_block['modulus_retention'])}"
        )
        lines.append(
            f"- Min strength retention: "
            f"{_fmt(fe_block['min_strength_retention'])}"
        )
        lines.append(
            f"- Governing criterion: "
            f"{_fmt(fe_block['critical_criterion'])}"
            f" (mode: {_fmt(fe_block['critical_mode'])}, "
            f"ply: {_fmt(fe_block['critical_ply'])})"
        )
    lines.append("")
    lines.append("## 4. Criteria Cited")
    lines.append("")
    for c in ncr["criteria_cited"]:
        lines.append(f"- {c}")
    lines.append("")
    lines.append("## 5. Recommended Disposition (NON-BINDING)")
    lines.append("")
    lines.append(f"- **Severity:** {_fmt(dr['severity'])}")
    lines.append(f"- **Recommended path:** {_fmt(dr['recommended_path'])}")
    lines.append(f"- **Rationale:** {_fmt(dr['rationale'])}")
    lines.append(
        f"- **Required approvals:** "
        f"{_fmt(dr['required_approvals'])}"
    )
    lines.append("")
    lines.append(f"> {dr['note']}")
    lines.append("")
    lines.append("## 6. MRB Disposition (to be completed by the Board)")
    lines.append("")
    lines.append("- Final disposition: ___________________________________")
    lines.append("- MRB rationale: ______________________________________")
    lines.append("")
    lines.append("| Role | Name | Signature | Date |")
    lines.append("| --- | --- | --- | --- |")
    for ap in ncr["mrb_disposition"]["approvals"]:
        lines.append(f"| {ap['role']} | | | |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"_{ncr['disclaimer']}_")
    lines.append("")
    return "\n".join(lines)


def export_ncr(
    ncr: dict,
    filepath: Union[str, Path],
    *,
    fmt: str = "md",
) -> None:
    """Write an NCR (from :func:`build_ncr`) to disk.

    Parameters
    ----------
    ncr : dict
        The structured NCR.
    filepath : str or Path
        Output path.
    fmt : {'md', 'json'}
        ``'md'`` writes the rendered Markdown form (default); ``'json'``
        writes the structured report.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        filepath.write_text(json.dumps(ncr, indent=2), encoding="utf-8")
    elif fmt == "md":
        filepath.write_text(render_ncr_markdown(ncr), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported fmt {fmt!r}; use 'md' or 'json'.")
