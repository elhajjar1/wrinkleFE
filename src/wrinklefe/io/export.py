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
# Wrinkle analysis validation summary — NCR attachment
# ====================================================================== #
#
# These helpers turn a WrinkleFE analysis of an out-of-plane fibre wrinkle
# into a concise engineering validation summary that an engineer can
# attach to a Nonconformance Report (NCR).  It deliberately carries no
# QMS/admin fields (NCR number, part/serial, work order, MRB sign-off):
# that paperwork lives on the NCR itself.  This artefact is the technical
# validation only.
#
# Scope/authority note: the recommendation produced here is *decision
# support only*.  It does not constitute a final disposition.  Severity
# thresholds below are generic engineering guidance and MUST be superseded
# by the program-specific allowables, drawing requirements, and process
# specifications that the Material Review Board (MRB) applies.  The
# qualified MRB reviews, may modify, and approves the final disposition.

# Residual-strength fraction (= analytical knockdown) and damage-index
# severity bands.  The worst (lowest) tier from either metric governs.
_SEVERITY_BANDS = (
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
    by_knockdown = len(_SEVERITY_BANDS) - 1
    for idx, (_, min_kd, _, _, _) in enumerate(_SEVERITY_BANDS):
        if kd >= min_kd:
            by_knockdown = idx
            break

    by_damage = 0
    for idx, (_, _, max_di, _, _) in enumerate(_SEVERITY_BANDS):
        if di < max_di:
            by_damage = idx
            break
    else:
        by_damage = len(_SEVERITY_BANDS) - 1

    band_idx = max(by_knockdown, by_damage)
    label, _, _, path, approvals = _SEVERITY_BANDS[band_idx]

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


def build_analysis_summary(
    *,
    defect: dict,
    engineering: dict,
    reference: Optional[str] = None,
    prepared_by: Optional[str] = None,
    notes: Optional[str] = None,
    tool_version: Optional[str] = None,
) -> dict:
    """Assemble a wrinkle-analysis validation summary for an NCR.

    Produces the *technical validation only* — the wrinkle geometry, the
    affected laminate, the WrinkleFE engineering analysis, the cited
    criteria, and a *recommended* (non-binding) disposition path. It
    carries no QMS/admin fields (NCR number, part/serial, work order, MRB
    sign-off): that paperwork lives on the NCR this is attached to.

    Parameters
    ----------
    defect : dict
        As-analyzed wrinkle + laminate.  Recognised keys:
        ``amplitude_mm``, ``wavelength_mm``, ``width_mm``, ``morphology``,
        ``loading``, ``ply_thickness_mm``, ``n_plies``, ``layup``,
        ``material_name``.
    engineering : dict
        WrinkleFE results.  Recognised keys: ``analytical_knockdown``,
        ``analytical_strength_MPa``, ``damage_index``, ``max_angle_deg``,
        ``effective_angle_deg``, ``morphology_factor``, and an optional
        ``fe`` sub-dict (``modulus_retention``, ``retention_factors``,
        ``critical_criterion``, ``critical_mode``, ``critical_ply``).
    reference : str, optional
        Free-text label tying this attachment to its NCR (e.g. the NCR
        number or part reference).  Optional by design.
    prepared_by : str, optional
        Who ran the analysis.
    notes : str, optional
        Free-text engineering notes.
    tool_version : str, optional
        WrinkleFE version string, recorded for traceability.

    Returns
    -------
    dict
        The structured validation summary.  JSON-serialisable.
    """

    def _clean(val: Optional[str], default: str) -> str:
        if val is None or (isinstance(val, str) and not val.strip()):
            return default
        return str(val).strip()

    date = datetime.now(timezone.utc).date().isoformat()

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
        "Generic severity banding in this summary is advisory only and is "
        "superseded by the program-specific allowables, drawing "
        "requirements, and process specifications applied by the MRB."
    )

    summary = {
        "report_type": "Wrinkle Analysis Validation Summary",
        "intended_use": (
            "Engineering validation attachment to a Nonconformance "
            "Report (NCR). Not a standalone NCR."
        ),
        "tool": "WrinkleFE",
        "header": {
            "date": date,
            "reference": _clean(reference, "(not specified)"),
            "prepared_by": _clean(prepared_by, "(not specified)"),
            "tool_version": _clean(tool_version, "(unspecified)"),
        },
        "wrinkle_geometry": {
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
        "notes": _clean(notes, "(none)"),
        "disclaimer": (
            "This validation summary was prepared with WrinkleFE "
            "decision-support tooling and is intended as an attachment to "
            "a Nonconformance Report. The analysis and recommendation are "
            "advisory and do not constitute a final material disposition. "
            "A qualified Material Review Board must review, may modify, "
            "and formally approve the disposition."
        ),
    }
    return summary


def _fmt(value: Any, default: str = "—") -> str:
    if value is None or value == "":
        return default
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(str(v) for v in value) + "]"
    return str(value)


def render_summary_markdown(summary: dict) -> str:
    """Render a :func:`build_analysis_summary` as Markdown.

    Produces a concise validation attachment an engineer can staple to
    an NCR — geometry, laminate, analysis, criteria, and the non-binding
    recommended disposition. No QMS/admin or MRB sign-off block.

    Parameters
    ----------
    summary : dict
        A report produced by :func:`build_analysis_summary`.

    Returns
    -------
    str
        Markdown text.
    """
    h = summary["header"]
    geo = summary["wrinkle_geometry"]
    lam = summary["laminate"]
    ea = summary["engineering_analysis"]
    dr = summary["disposition_recommendation"]

    lines: list[str] = []
    lines.append("# Wrinkle Analysis Validation Summary")
    lines.append("")
    lines.append(f"_{summary['intended_use']}_")
    lines.append("")
    lines.append(f"**Date:** {_fmt(h['date'])}  ")
    lines.append(f"**Reference:** {_fmt(h['reference'])}  ")
    lines.append(f"**Prepared by:** {_fmt(h['prepared_by'])}  ")
    lines.append(f"**WrinkleFE version:** {_fmt(h['tool_version'])}")
    lines.append("")
    lines.append("## 1. As-analyzed wrinkle geometry")
    lines.append("")
    lines.append(f"- Amplitude: {_fmt(geo['amplitude_mm'])} mm")
    lines.append(f"- Wavelength: {_fmt(geo['wavelength_mm'])} mm")
    lines.append(
        f"- Envelope half-width: {_fmt(geo['envelope_half_width_mm'])} mm"
    )
    lines.append(f"- Morphology: {_fmt(geo['morphology'])}")
    lines.append(f"- Loading condition: {_fmt(geo['loading_condition'])}")
    lines.append("")
    lines.append("## 2. Affected laminate")
    lines.append("")
    lines.append(f"- Material: {_fmt(lam['material'])}")
    lines.append(
        f"- Ply thickness: {_fmt(lam['ply_thickness_mm'])} mm  "
        f"·  Plies: {_fmt(lam['n_plies'])}"
    )
    lines.append(f"- Layup (deg): {_fmt(lam['layup_deg'])}")
    lines.append("")
    lines.append("## 3. Engineering analysis (WrinkleFE)")
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
    lines.append("## 4. Criteria cited")
    lines.append("")
    for c in summary["criteria_cited"]:
        lines.append(f"- {c}")
    lines.append("")
    lines.append("## 5. Recommended disposition (NON-BINDING)")
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
    lines.append("## 6. Notes")
    lines.append("")
    lines.append(_fmt(summary["notes"]))
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"_{summary['disclaimer']}_")
    lines.append("")
    return "\n".join(lines)


# A4 portrait page geometry (inches) and 1-inch margins.
_PDF_PAGE_W = 8.27
_PDF_PAGE_H = 11.69
_PDF_MARGIN = 0.9
# Per-line style: fontsize (pt), bold, italic, x-indent (in), gap-before (in).
_PDF_STYLES = {
    "title": (17.0, True, False, 0.0, 0.0),
    "h2": (12.5, True, False, 0.0, 0.22),
    "subhead": (10.0, True, False, 0.0, 0.10),
    "bullet": (9.5, False, False, 0.18, 0.0),
    "quote": (9.5, False, True, 0.22, 0.06),
    "table": (8.5, False, False, 0.10, 0.0),
    "rule": (9.5, False, False, 0.0, 0.06),
    "normal": (9.5, False, False, 0.0, 0.0),
    "disclaimer": (8.0, False, True, 0.0, 0.10),
}


def _classify_md_line(line: str) -> tuple[str, str]:
    """Map one Markdown line to a (style-kind, plain-text) pair."""
    raw = line.rstrip()
    stripped = raw.strip()
    if stripped == "":
        return "blank", ""
    if stripped.startswith("# "):
        return "title", stripped[2:].strip()
    if stripped.startswith("## "):
        return "h2", stripped[3:].strip()
    if stripped.startswith(("---", "***", "___")) and set(
        stripped
    ) <= {"-", "*", "_"}:
        return "rule", ""
    if stripped.startswith(">"):
        return "quote", _strip_md_inline(stripped[1:].strip())
    if stripped.startswith("|"):
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        non_empty = [c for c in cells if c]
        if non_empty and all(set(c) <= {"-", ":"} for c in non_empty):
            return "rule", ""
        widths = (26, 16, 16, 12)
        out = []
        for i, c in enumerate(cells):
            w = widths[i] if i < len(widths) else 14
            out.append(c.ljust(w))
        return "table", " ".join(out).rstrip()
    if stripped.startswith("- "):
        return "bullet", _strip_md_inline(stripped[2:].strip())
    if (
        stripped.startswith("_")
        and stripped.endswith("_")
        and len(stripped) > 2
    ):
        return "disclaimer", _strip_md_inline(stripped)
    if (
        stripped.startswith("**")
        and stripped.endswith("**")
        and "**" not in stripped[2:-2]
    ):
        return "subhead", _strip_md_inline(stripped)
    return "normal", _strip_md_inline(stripped)


def _strip_md_inline(text: str) -> str:
    """Drop inline Markdown emphasis markers for flat PDF text."""
    return (
        text.replace("**", "")
        .replace("`", "")
        .replace("__", "")
        .strip("_")
    )


def render_summary_pdf(summary: dict) -> bytes:
    """Render a validation summary as a paginated PDF.

    Uses Matplotlib's PDF backend (already a project dependency, headless-
    safe) so no extra packages are required. The structured Markdown form
    is laid out in a monospaced font with automatic word-wrap and
    pagination.

    Parameters
    ----------
    summary : dict
        A report produced by :func:`build_analysis_summary`.

    Returns
    -------
    bytes
        The PDF document.
    """
    import io as _io
    import textwrap

    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    lines = render_summary_markdown(summary).split("\n")
    usable_w = _PDF_PAGE_W - 2 * _PDF_MARGIN
    usable_h = _PDF_PAGE_H - 2 * _PDF_MARGIN

    # Pre-compute wrapped, styled physical lines: (kind, text, fs, bold,
    # italic, indent_in, gap_in, line_h_in).
    items: list[tuple] = []
    for line in lines:
        kind, text = _classify_md_line(line)
        if kind == "blank":
            items.append(("blank", "", 9.5, False, False, 0.0, 0.0, 0.11))
            continue
        fs, bold, italic, indent, gap = _PDF_STYLES[kind]
        line_h = fs / 72.0 * 1.55
        if kind == "rule":
            items.append(
                ("rule", "", fs, False, False, 0.0, gap, line_h)
            )
            continue
        char_w = fs * 0.60 / 72.0  # monospace advance, inches
        max_chars = max(8, int((usable_w - indent) / char_w))
        if kind == "table":
            wrapped = [text]  # keep table rows intact (monospaced)
        else:
            wrapped = textwrap.wrap(
                text, width=max_chars, break_long_words=True
            ) or [""]
        for j, seg in enumerate(wrapped):
            items.append(
                (
                    kind,
                    seg,
                    fs,
                    bold,
                    italic,
                    indent,
                    gap if j == 0 else 0.0,
                    line_h,
                )
            )

    buf = _io.BytesIO()
    fig: Optional[Any] = None
    cursor = 0.0  # inches consumed from the top of the usable area

    def _new_page(pdf: Any) -> Any:
        f = Figure(figsize=(_PDF_PAGE_W, _PDF_PAGE_H))
        return f

    with PdfPages(buf) as pdf:
        fig = _new_page(pdf)
        for kind, text, fs, bold, italic, indent, gap, line_h in items:
            need = gap + line_h
            if cursor + need > usable_h:
                pdf.savefig(fig)
                fig = _new_page(pdf)
                cursor = 0.0
            cursor += gap
            y = 1.0 - (_PDF_MARGIN + cursor + line_h * 0.5) / _PDF_PAGE_H
            if kind == "rule":
                xa = _PDF_MARGIN / _PDF_PAGE_W
                xb = (_PDF_PAGE_W - _PDF_MARGIN) / _PDF_PAGE_W
                fig.add_artist(
                    Line2D(
                        [xa, xb],
                        [y, y],
                        transform=fig.transFigure,
                        color="0.4",
                        linewidth=0.6,
                    )
                )
            elif text:
                x = (_PDF_MARGIN + indent) / _PDF_PAGE_W
                fig.text(
                    x,
                    y,
                    text,
                    fontsize=fs,
                    fontweight="bold" if bold else "normal",
                    fontstyle="italic" if italic else "normal",
                    family="monospace",
                    va="center",
                    ha="left",
                )
            cursor += line_h
        pdf.savefig(fig)

    return buf.getvalue()


def export_summary(
    summary: dict,
    filepath: Union[str, Path],
    *,
    fmt: str = "md",
) -> None:
    """Write a validation summary (from :func:`build_analysis_summary`).

    Parameters
    ----------
    summary : dict
        The structured validation summary.
    filepath : str or Path
        Output path.
    fmt : {'md', 'json', 'pdf'}
        ``'md'`` writes the rendered Markdown form (default); ``'json'``
        writes the structured report; ``'pdf'`` writes a paginated PDF.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        filepath.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    elif fmt == "md":
        filepath.write_text(
            render_summary_markdown(summary), encoding="utf-8"
        )
    elif fmt == "pdf":
        filepath.write_bytes(render_summary_pdf(summary))
    else:
        raise ValueError(
            f"Unsupported fmt {fmt!r}; use 'md', 'json', or 'pdf'."
        )
