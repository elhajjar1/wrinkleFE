"""Streamlit web app for WrinkleFE composite-laminate wrinkle analysis.

Wraps the existing wrinklefe.analysis pipeline in a browser UI: sidebar
inputs feed an AnalysisConfig, the main area shows the wrinkle profile
plot and analysis results.

Run locally:
    streamlit run app.py

Deploy: see DEPLOYMENT_STREAMLIT.md.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")  # headless backend; required on Streamlit Cloud

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Make the src-layout package importable on Streamlit Cloud, which does not
# pip-install the local repo.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary
from wrinklefe.core.wrinkle import GaussianSinusoidal

import streamlit_viz


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="WrinkleFE",
    page_icon=":dna:",
    layout="wide",
)

st.title("WrinkleFE — Composite Laminate Wrinkle Analysis")
st.caption(
    "Predicts strength and stiffness knockdown for composite laminates "
    "containing fiber-waviness defects. Configure parameters in the sidebar "
    "and click **Run analysis**."
)

LIB = MaterialLibrary()
MATERIAL_NAMES = sorted(LIB.list_names())
MORPHOLOGIES = ["stack", "convex", "concave", "uniform", "graded"]

DEFAULT_LAYUP = (
    "0,45,-45,90,0,45,-45,90,0,45,-45,90,"
    "90,-45,45,0,90,-45,45,0,90,-45,45,0"
)


# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("**Wrinkle geometry**")
    amplitude = st.number_input(
        "Amplitude A [mm]",
        min_value=0.0, max_value=5.0, value=0.366, step=0.01,
        help=(
            "Peak displacement of the wrinkled mid-surface from the flat "
            "reference (crest height, NOT peak-to-peak). Measure as "
            "A = (z_max − z_min) / 2 from a cross-section micrograph."
        ),
    )
    wavelength = st.number_input(
        "Wavelength λ [mm]",
        min_value=1.0, max_value=200.0, value=16.0, step=0.5,
        help=(
            "Spatial period of the cosine carrier. Larger λ at fixed A "
            "lowers the peak fibre angle θ_max ≈ arctan(2πA/λ)."
        ),
    )
    width = st.number_input(
        "Envelope width w [mm]",
        min_value=1.0, max_value=200.0, value=12.0, step=0.5,
        help=(
            "Half-width of the Gaussian envelope multiplying the cosine. "
            "Smaller w localises the wrinkle to a few wavelengths near x = 0."
        ),
    )

    st.markdown("**Morphology & loading**")
    morphology = st.selectbox("Morphology", MORPHOLOGIES, index=0)
    with st.popover("What do these morphologies mean?", use_container_width=True):
        st.markdown(
            "**Dual-wrinkle morphologies** — two adjacent wrinkles across the "
            "laminate thickness, classified by the relative phase offset φ "
            "between their centrelines (Jin et al. 2026):\n\n"
            "- **Stack** (φ = 0): peaks and troughs aligned vertically. "
            "Aggregate morphology factor M_f = 1.0 — the baseline case.\n"
            "- **Convex** (φ = π/2): the interface between the two wrinkles "
            "bulges outward. M_f < 1 — the *least* damaging configuration "
            "under compression.\n"
            "- **Concave** (φ = −π/2): the interface pinches inward. "
            "M_f > 1 — the *most* damaging configuration; concave pinching "
            "amplifies kink-band formation under compression.\n\n"
            "**Single-wrinkle through-thickness modes** — one wrinkle, "
            "varying how its amplitude propagates from the wrinkle core "
            "outward to the laminate surfaces:\n\n"
            "- **Uniform**: full amplitude on every ply — no through-thickness "
            "decay. Worst case for through-thickness uniformity of damage.\n"
            "- **Graded**: linear decay from the wrinkle interface to the "
            "outer surfaces. The **Decay floor** slider sets the minimum "
            "amplitude fraction retained at the surface plies "
            "(0 = full decay to zero, 1 = uniform)."
        )
    decay_floor = 0.0
    if morphology == "graded":
        decay_floor = st.slider(
            "Decay floor", 0.0, 1.0, 0.0, 0.05,
            help="Minimum amplitude fraction at the outer surfaces.",
        )

    loading = st.radio("Loading mode", ["compression", "tension"], horizontal=True)
    strain_mag_pct = st.number_input(
        "Applied strain magnitude [%]",
        min_value=0.0, max_value=5.0, value=1.0, step=0.1,
        help=(
            "Magnitude only — the sign is taken from the loading mode "
            "(compression → negative, tension → positive). Editing this "
            "value is preserved when you toggle the loading mode."
        ),
    )
    applied_strain_pct = -strain_mag_pct if loading == "compression" else strain_mag_pct

    st.markdown("**Material & layup**")
    default_idx = MATERIAL_NAMES.index("IM7_8552") if "IM7_8552" in MATERIAL_NAMES else 0
    material_name = st.selectbox("Material", MATERIAL_NAMES, index=default_idx)
    ply_thickness = st.number_input(
        "Ply thickness [mm]",
        min_value=0.05, max_value=1.0, value=0.183, step=0.01,
        help=(
            "Thickness of one ply in mm. Default 0.183 mm matches "
            "CYCOM X850/T800. Total laminate thickness = ply_thickness × "
            "number of plies in the layup."
        ),
    )
    layup_str = st.text_area(
        "Layup (comma-separated angles in degrees)",
        value=DEFAULT_LAYUP, height=80,
        help="Default: quasi-isotropic [0/45/-45/90]_3s (24 plies).",
    )

    with st.expander("Advanced — mesh & solver", expanded=True):
        analytical_only = st.checkbox(
            "Analytical only (skip FE solve)", value=False,
            help=(
                "Default off: the full FE solve runs (mesh, static "
                "displacement-controlled solve, multi-criterion failure "
                "evaluation) and yields stress fields, modulus retention, "
                "and per-criterion strength retention. Tick this to fall "
                "back to the closed-form analytical knockdown only — "
                "much faster but no FE outputs."
            ),
        )
        nx = st.number_input(
            "Mesh divisions in x", 4, 64, 12, 2,
            help=(
                "Hex elements along the wrinkle (x) direction across the "
                "domain length. More elements resolve the curvature but "
                "scale solve time roughly linearly."
            ),
        )
        ny = st.number_input(
            "Mesh divisions in y", 4, 32, 6, 2,
            help=(
                "Hex elements across the laminate width (y). Wrinkle is "
                "uniform in y, so a coarse mesh is usually adequate."
            ),
        )
        nz_per_ply = st.number_input(
            "Mesh divisions per ply (z)", 1, 4, 1,
            help=(
                "Hex elements through the thickness of every individual "
                "ply. Increase to capture interlaminar stress gradients."
            ),
        )
        if not analytical_only:
            st.caption(
                ":hourglass_flowing_sand: Full FE solve. On Streamlit "
                "Cloud's CPU this typically takes 30–90 s for the default "
                "mesh; reducing nx, ny, or nz_per_ply speeds it up."
            )

    run_clicked = st.button("Run analysis", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_layup(s: str) -> List[float]:
    """Parse '0,45,-45,90' (or whitespace/semicolon-separated) into floats."""
    out: List[float] = []
    for tok in s.replace(";", ",").replace("\n", ",").split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    if not out:
        raise ValueError("Layup is empty.")
    return out


@st.cache_data(show_spinner=False)
def run_analysis_cached(cfg_payload: tuple) -> dict:
    """Cached analysis run. cfg_payload is a hashable tuple of config items."""
    cfg_dict = dict(cfg_payload)
    angles = list(cfg_dict.pop("angles_tuple"))
    cfg = AnalysisConfig(
        amplitude=cfg_dict["amplitude"],
        wavelength=cfg_dict["wavelength"],
        width=cfg_dict["width"],
        morphology=cfg_dict["morphology"],
        decay_floor=cfg_dict["decay_floor"],
        loading=cfg_dict["loading"],
        material=LIB.get(cfg_dict["material_name"]),
        angles=angles,
        ply_thickness=cfg_dict["ply_thickness"],
        applied_strain=cfg_dict["applied_strain"],
        nx=cfg_dict["nx"], ny=cfg_dict["ny"], nz_per_ply=cfg_dict["nz_per_ply"],
        analytical_only=cfg_dict["analytical_only"],
    )
    result = WrinkleAnalysis(cfg).run(analytical_only=cfg_dict["analytical_only"])

    fe: dict | None = None
    if not cfg_dict["analytical_only"] and result.field_results is not None:
        max_disp_mm, _ = result.field_results.max_displacement()
        rep = result.failure_report
        ply_fi: dict[str, list[float]] = {}
        if rep is not None and getattr(rep, "ply_failure_indices", None):
            ply_fi = {
                k: [float(v) for v in np.asarray(arr).ravel().tolist()]
                for k, arr in rep.ply_failure_indices.items()
            }
        mesh = result.mesh
        fr = result.field_results
        nodes_arr = np.asarray(mesh.nodes, dtype=np.float64)
        elements_arr = np.asarray(mesh.elements, dtype=np.int64)
        stress_per_elem = np.asarray(fr.stress_local).mean(axis=1)
        element_centers = nodes_arr[elements_arr].mean(axis=1)
        fi_per_gauss = {
            k: np.asarray(v, dtype=np.float64)
            for k, v in (result.failure_indices or {}).items()
        }
        fe = {
            "modulus_retention": float(result.modulus_retention),
            "retention_factors": {
                k: float(v) for k, v in (result.retention_factors or {}).items()
            },
            "baseline_fi": {
                k: float(v) for k, v in (result.baseline_fi or {}).items()
            },
            "n_nodes": int(mesh.n_nodes),
            "n_elements": int(mesh.n_elements),
            "max_displacement_mm": float(max_disp_mm),
            "critical_criterion": getattr(rep, "critical_criterion", None),
            "critical_mode": getattr(rep, "critical_mode", None),
            "critical_ply": (
                int(rep.critical_ply)
                if rep is not None and getattr(rep, "critical_ply", None) is not None
                else None
            ),
            "ply_failure_indices": ply_fi,
            # Numpy arrays for the 3D viz. Stripped from the JSON export.
            "nodes": nodes_arr,
            "elements": elements_arr,
            "ply_ids": np.asarray(mesh.ply_ids, dtype=np.int64),
            "displacement": np.asarray(fr.displacement, dtype=np.float64),
            "stress_per_elem": stress_per_elem,
            "element_centers": element_centers,
            "fi_per_gauss": fi_per_gauss,
        }

    return {
        "summary": result.summary(),
        "loading": cfg_dict["loading"],
        "applied_strain_abs": float(cfg_dict["applied_strain"]),
        "max_angle_deg": float(np.degrees(result.max_angle_rad)),
        "effective_angle_deg": float(np.degrees(result.effective_angle_rad)),
        "morphology_factor": float(result.morphology_factor),
        "gamma_Y_eff": float(result.gamma_Y_eff),
        "analytical_knockdown": float(result.analytical_knockdown),
        "analytical_strength_MPa": float(result.analytical_strength_MPa),
        "damage_index": float(result.damage_index),
        "tension_mechanisms": (
            {k: float(v) if isinstance(v, (int, float, np.floating)) else v
             for k, v in result.tension_mechanisms.items()}
            if result.tension_mechanisms else None
        ),
        "fe": fe,
    }


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

profile = GaussianSinusoidal(amplitude=amplitude, wavelength=wavelength, width=width)


# Run handler — execute BEFORE tabs render so the Results tab sees the
# updated session_state and the empty-state placeholder doesn't render
# alongside the running indicator.
if run_clicked:
    try:
        layup = parse_layup(layup_str)
    except ValueError as e:
        st.error(f"Could not parse layup: {e}")
        st.stop()

    cfg_payload = tuple(sorted({
        "amplitude": amplitude,
        "wavelength": wavelength,
        "width": width,
        "morphology": morphology,
        "decay_floor": decay_floor,
        "loading": loading,
        "ply_thickness": ply_thickness,
        "angles_tuple": tuple(layup),
        "applied_strain": applied_strain_pct / 100.0,
        "nx": int(nx), "ny": int(ny), "nz_per_ply": int(nz_per_ply),
        "material_name": material_name,
        "analytical_only": bool(analytical_only),
    }.items()))

    with st.status("Running analysis…", expanded=True) as status:
        st.write("Building laminate, wrinkle geometry, and mesh…")
        if not analytical_only:
            st.caption(
                "Use the **Stop** button in the top-right toolbar to "
                "cancel a long FE solve."
            )
        try:
            results = run_analysis_cached(cfg_payload)
        except ValueError as exc:
            status.update(label="Invalid input", state="error", expanded=True)
            st.error(f"Invalid input: {exc}")
            st.stop()
        except np.linalg.LinAlgError as exc:
            status.update(label="FE solve failed", state="error", expanded=True)
            st.error(
                "FE matrix is singular. Try increasing the mesh density "
                "(nx, ny, or nz_per_ply) or check that the layup is "
                f"symmetric. Underlying error: {exc}"
            )
            st.stop()
        except MemoryError:
            status.update(label="Out of memory", state="error", expanded=True)
            st.error(
                "Out of memory while assembling the FE system. Reduce nx, "
                "ny, or nz_per_ply, or tick *Analytical only* to skip the "
                "FE solve."
            )
            st.stop()
        except Exception as exc:
            status.update(label="Analysis failed", state="error", expanded=True)
            st.error(f"Analysis failed: {exc}")
            with st.expander("Traceback"):
                st.exception(exc)
            st.stop()
        st.write("Done.")
        status.update(label="Analysis complete", state="complete", expanded=False)

    st.session_state["results"] = results
    st.session_state["cfg_payload"] = cfg_payload


tab_geom, tab_results, tab_export = st.tabs(["Geometry", "Results", "Export"])

with tab_geom:
    st.subheader("Wrinkle mid-surface profile")
    x = np.linspace(-3 * width, 3 * width, 800)
    z = profile.displacement(x)
    theta_deg = np.degrees(np.arctan(profile.slope(x)))
    fig, (ax_z, ax_theta) = plt.subplots(
        2, 1, figsize=(8, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1.0]},
    )
    ax_z.plot(x, z, color="#1f77b4", linewidth=1.5)
    ax_z.axhline(0.0, color="grey", linewidth=0.5)
    ax_z.set_ylabel("z(x) [mm]")
    ax_z.set_title(r"$z(x) = A \cdot \exp(-x^2 / w^2) \cdot \cos(2\pi x / \lambda)$")
    ax_z.grid(alpha=0.3)

    ax_theta.plot(x, theta_deg, color="#d62728", linewidth=1.2)
    theta_max_deg = np.degrees(profile.max_angle())
    ax_theta.axhline(theta_max_deg, color="grey", linestyle="--", linewidth=0.6)
    ax_theta.axhline(-theta_max_deg, color="grey", linestyle="--", linewidth=0.6)
    ax_theta.set_xlabel("x [mm]")
    ax_theta.set_ylabel("θ(x) [deg]")
    ax_theta.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    st.caption(
        f"Numerical θ_max = {theta_max_deg:.2f}°  ·  "
        f"Closed-form approx arctan(2πA/λ) = "
        f"{np.degrees(profile.max_angle_approx()):.2f}°"
    )

    if morphology == "graded":
        try:
            _angles_preview = parse_layup(layup_str)
            _n_plies = len(_angles_preview)
        except Exception:
            _n_plies = 24
        with st.expander("Through-thickness amplitude profile", expanded=False):
            ply_idx = np.arange(_n_plies)
            p_mid = (_n_plies - 1) / 2.0
            raw = np.maximum(0.0, 1.0 - np.abs(ply_idx - p_mid) / max(p_mid, 1e-9))
            decay = decay_floor + (1.0 - decay_floor) * raw
            fig_d, ax_d = plt.subplots(figsize=(4, 3))
            ax_d.barh(ply_idx, decay, color="#9467bd")
            ax_d.set_xlabel("Amplitude fraction")
            ax_d.set_ylabel("Ply index")
            ax_d.set_xlim(0, 1.05)
            ax_d.invert_yaxis()
            fig_d.tight_layout()
            st.pyplot(fig_d, clear_figure=True)
            st.caption(
                f"Decay floor = {decay_floor:.2f}; surface plies retain "
                f"{decay_floor*100:.0f}% of the wrinkle-core amplitude."
            )

    with st.expander("Layup stack visualizer", expanded=False):
        try:
            _angles = parse_layup(layup_str)
        except Exception as e:
            st.warning(f"Could not parse layup: {e}")
        else:
            cmap = plt.get_cmap("twilight")
            n = len(_angles)
            fig_l, ax_l = plt.subplots(figsize=(5, max(2.5, 0.18 * n)))
            for i, a in enumerate(_angles):
                colour = cmap(((a % 180) + 180) % 180 / 180)
                ax_l.barh(i, 1.0, color=colour, edgecolor="white", linewidth=0.4)
                ax_l.text(
                    0.5, i, f"{int(a):+d}°",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(a) % 180 not in (0, 90) else "black",
                )
            ax_l.set_xlim(0, 1)
            ax_l.set_xticks([])
            ax_l.set_ylabel("Ply index (top ↓ bottom)")
            ax_l.set_yticks(range(n))
            ax_l.invert_yaxis()
            fig_l.tight_layout()
            st.pyplot(fig_l, clear_figure=True)
            st.caption(f"{n} plies · colours hue-mapped on ply angle modulo 180°.")


with tab_results:
    if "results" not in st.session_state:
        st.info("Click **Run analysis** in the sidebar to compute results.")
    else:
        r = st.session_state["results"]

        st.subheader("Analytical predictions")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Max fibre misalignment", f"{r['max_angle_deg']:.2f}°")
        c2.metric("Analytical knockdown", f"{r['analytical_knockdown']:.3f}")
        c3.metric("Predicted strength", f"{r['analytical_strength_MPa']:.1f} MPa")
        c4.metric("Damage index D", f"{r['damage_index']:.3f}")

        d1, d2, d3 = st.columns(3)
        d1.metric("Morphology factor M_f", f"{r['morphology_factor']:.3f}")
        d2.metric("Effective fibre angle θ_eff", f"{r['effective_angle_deg']:.2f}°")
        d3.metric("Yield strain γ_Y eff", f"{r['gamma_Y_eff']:.4f}")

        if r.get("tension_mechanisms"):
            with st.expander(
                "Tension mechanism breakdown", expanded=True
            ):
                tm = r["tension_mechanisms"]
                bar_keys = [k for k in ("kd_fiber", "kd_matrix", "kd_oop")
                            if isinstance(tm.get(k), (int, float))]
                if bar_keys:
                    fig_tm, ax_tm = plt.subplots(figsize=(5, 2.6))
                    ax_tm.bar(
                        [k.replace("kd_", "") for k in bar_keys],
                        [tm[k] for k in bar_keys],
                        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
                    )
                    ax_tm.set_ylabel("Knockdown contribution")
                    ax_tm.set_ylim(0, max(1.05, max(tm[k] for k in bar_keys) * 1.1))
                    ax_tm.axhline(1.0, color="grey", linestyle="--", linewidth=0.6)
                    ax_tm.grid(axis="y", alpha=0.3)
                    fig_tm.tight_layout()
                    st.pyplot(fig_tm, clear_figure=True)
                if tm.get("mode"):
                    st.markdown(f"Controlling mechanism: **{tm['mode']}**")
                rest = {k: v for k, v in tm.items()
                        if k not in {"kd_fiber", "kd_matrix", "kd_oop", "mode"}}
                if rest:
                    st.json(rest)

        fe = r.get("fe")
        if fe is not None:
            st.subheader("FE solution")
            f1, f2, f3 = st.columns(3)
            f1.metric("Modulus retention", f"{fe['modulus_retention']:.3f}")
            worst = (
                min(fe["retention_factors"].values())
                if fe["retention_factors"] else None
            )
            f2.metric(
                "Strength retention (worst criterion)",
                f"{worst:.3f}" if worst is not None else "—",
            )
            f3.metric("Max displacement", f"{fe['max_displacement_mm']:.4f} mm")

            if fe["retention_factors"]:
                st.markdown("**Strength retention by failure criterion**")
                crits = list(fe["retention_factors"].keys())
                ret_vals = [fe["retention_factors"][c] for c in crits]
                base_vals = [fe["baseline_fi"].get(c) for c in crits]
                fig_ret, ax_ret = plt.subplots(
                    figsize=(7, max(1.6, 0.55 * len(crits) + 1.0))
                )
                y = np.arange(len(crits))
                ax_ret.barh(y, ret_vals, color="#2ca02c",
                            label="Retention (wrinkled / pristine)")
                ax_ret.axvline(1.0, color="grey", linestyle="--", linewidth=0.8)
                ax_ret.set_yticks(y)
                ax_ret.set_yticklabels(crits)
                ax_ret.set_xlabel("Retention")
                ax_ret.set_xlim(0, max(1.05, max(ret_vals) * 1.1))
                ax_ret.grid(axis="x", alpha=0.3)
                fig_ret.tight_layout()
                st.pyplot(fig_ret, clear_figure=True)
                if any(b is not None for b in base_vals):
                    base_table = {
                        c: {
                            "retention": ret_vals[i],
                            "pristine max FI": base_vals[i],
                        }
                        for i, c in enumerate(crits)
                    }
                    with st.expander("Per-criterion details", expanded=False):
                        st.table(base_table)

            if fe.get("ply_failure_indices"):
                st.markdown("**Per-ply failure index**")
                pf = fe["ply_failure_indices"]
                crit_for_plot = (
                    fe.get("critical_criterion")
                    if fe.get("critical_criterion") in pf
                    else next(iter(pf))
                )
                fi_arr = np.asarray(pf[crit_for_plot], dtype=float)
                colours = [
                    "#d62728" if i == fe.get("critical_ply") else "#1f77b4"
                    for i in range(len(fi_arr))
                ]
                fig_fi, ax_fi = plt.subplots(figsize=(7, 3))
                ax_fi.bar(range(len(fi_arr)), fi_arr, color=colours)
                ax_fi.axhline(1.0, color="grey", linestyle="--", linewidth=0.6)
                ax_fi.set_xlabel("Ply index")
                ax_fi.set_ylabel(f"FI ({crit_for_plot})")
                fig_fi.tight_layout()
                st.pyplot(fig_fi, clear_figure=True)
                st.caption(
                    "Critical ply highlighted in red. FI ≥ 1 indicates "
                    "failure of that ply under the applied strain."
                )

            crit = fe.get("critical_criterion")
            if crit:
                pieces = [f"criterion `{crit}`"]
                if fe.get("critical_mode"):
                    pieces.append(f"mode `{fe['critical_mode']}`")
                if fe.get("critical_ply") is not None:
                    pieces.append(f"first failing ply index `{fe['critical_ply']}`")
                st.markdown("**Critical failure:** " + ", ".join(pieces))

            if (
                isinstance(fe.get("nodes"), np.ndarray)
                and isinstance(fe.get("elements"), np.ndarray)
            ):
                st.subheader("3D field viz")
                st.caption(
                    "Drag to rotate, scroll to zoom. Sliders below the plot "
                    "re-render in milliseconds (the FE solve is cached)."
                )
                STRESS_COMPONENTS = [
                    (0, "σ₁₁"), (1, "σ₂₂"), (2, "σ₃₃"),
                    (3, "τ₂₃"), (4, "τ₁₃"), (5, "τ₁₂"),
                ]
                view_mode = st.radio(
                    "View mode",
                    ["Stress contour", "Deformed mesh", "Failure index"],
                    horizontal=True,
                    key="viz_view_mode",
                )
                if view_mode == "Stress contour":
                    comp = st.selectbox(
                        "Stress component (Voigt)",
                        STRESS_COMPONENTS,
                        index=2,
                        format_func=lambda t: t[1],
                        key="viz_stress_component",
                    )
                    fig_3d = streamlit_viz.stress_contour_figure(
                        fe["nodes"], fe["elements"], fe["stress_per_elem"],
                        component_index=comp[0],
                        component_label=comp[1],
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)
                elif view_mode == "Deformed mesh":
                    scale = st.slider(
                        "Deformation exaggeration",
                        1.0, 200.0, 50.0, 1.0,
                        key="viz_deformation_scale",
                        help=(
                            "FE displacements are ~10⁻³ mm, far below the "
                            "mesh dimensions. Multiply by this factor to "
                            "make the deformation visible."
                        ),
                    )
                    fig_3d = streamlit_viz.deformed_mesh_figure(
                        fe["nodes"], fe["elements"], fe["displacement"],
                        scale=scale,
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)
                else:
                    fi_dict = fe.get("fi_per_gauss", {})
                    fi_keys = list(fi_dict.keys())
                    if not fi_keys:
                        st.info("No per-gauss failure indices were captured.")
                    else:
                        crit_for_3d = st.selectbox(
                            "Failure criterion",
                            fi_keys,
                            index=(
                                fi_keys.index(fe.get("critical_criterion"))
                                if fe.get("critical_criterion") in fi_keys else 0
                            ),
                            key="viz_fi_criterion",
                        )
                        fig_3d = streamlit_viz.fi_3d_figure(
                            fe["nodes"], fe["elements"],
                            fi_dict[crit_for_3d], crit_for_3d,
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)

                st.markdown("**y-slice scrubber**")
                y_centers = fe["element_centers"][:, 1]
                y_unique = np.unique(y_centers)
                if y_unique.size > 1:
                    y_station = st.select_slider(
                        "y-station [mm]",
                        options=[float(y) for y in y_unique],
                        value=float(y_unique[len(y_unique) // 2]),
                        key="viz_y_station",
                    )
                    slice_comp = st.selectbox(
                        "Slice stress component",
                        STRESS_COMPONENTS,
                        index=2,
                        format_func=lambda t: t[1],
                        key="viz_slice_component",
                    )
                    fig_slice = streamlit_viz.y_slice_figure(
                        fe["element_centers"], fe["elements"], fe["nodes"],
                        fe["stress_per_elem"], slice_comp[0], y_station,
                        component_label=slice_comp[1],
                    )
                    if fig_slice is not None:
                        st.plotly_chart(fig_slice, use_container_width=True)

            with st.expander("Mesh statistics"):
                st.write(
                    f"Nodes: **{fe['n_nodes']:,}**  ·  "
                    f"Elements: **{fe['n_elements']:,}**"
                )
        else:
            st.info(
                "FE outputs are hidden because the run used **Analytical "
                "only**. Untick that option in the sidebar's *Advanced* "
                "panel and re-run to see modulus and strength retention."
            )

        st.subheader("Full text summary")
        st.text(r["summary"])

with tab_export:
    if "results" not in st.session_state:
        st.info("Run an analysis to enable export.")
    else:
        try:
            _wrinklefe_version = version("wrinklefe")
        except PackageNotFoundError:
            _wrinklefe_version = "0.0.0+unknown"

        def _strip_arrays(obj):
            """Drop numpy arrays so the JSON export stays small. Arrays
            are only useful for the in-app 3D viz."""
            if isinstance(obj, dict):
                return {
                    k: _strip_arrays(v)
                    for k, v in obj.items()
                    if not isinstance(v, np.ndarray)
                }
            if isinstance(obj, list):
                return [_strip_arrays(v) for v in obj]
            return obj

        payload = {
            "meta": {
                "wrinklefe_version": _wrinklefe_version,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
            "config": dict(st.session_state["cfg_payload"]),
            "results": _strip_arrays(st.session_state["results"]),
        }
        payload["config"]["angles"] = list(payload["config"].pop("angles_tuple"))

        st.download_button(
            "Download results as JSON",
            data=json.dumps(payload, indent=2).encode(),
            file_name="wrinklefe_results.json",
            mime="application/json",
        )

        fe = st.session_state["results"].get("fe")
        if fe and fe.get("retention_factors"):
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(
                ["criterion", "retention_factor",
                 "pristine_max_fi", "wrinkled_max_fi"]
            )
            for crit, ret in fe["retention_factors"].items():
                base = fe.get("baseline_fi", {}).get(crit)
                wrinkled = (base / ret) if (base is not None and ret) else None
                writer.writerow([
                    crit,
                    f"{ret:.6g}",
                    f"{base:.6g}" if base is not None else "",
                    f"{wrinkled:.6g}" if wrinkled is not None else "",
                ])
            st.download_button(
                "Download retention factors as CSV",
                data=buf.getvalue().encode(),
                file_name="wrinklefe_retention.csv",
                mime="text/csv",
            )

        st.caption(
            f"WrinkleFE {_wrinklefe_version} · "
            f"export timestamp {payload['meta']['timestamp_utc']}"
        )
