"""Streamlit web app for WrinkleFE composite-laminate wrinkle analysis.

Wraps the existing wrinklefe.analysis pipeline in a browser UI: sidebar
inputs feed an AnalysisConfig, the main area shows the wrinkle profile
plot and analysis results.

Run locally:
    streamlit run app.py

Deploy: see DEPLOYMENT_STREAMLIT.md.
"""

from __future__ import annotations

import json
import sys
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
    st.header("Wrinkle geometry")
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
    )
    width = st.number_input(
        "Envelope width w [mm]",
        min_value=1.0, max_value=200.0, value=12.0, step=0.5,
    )

    st.header("Morphology & loading")
    morphology = st.selectbox(
        "Morphology",
        MORPHOLOGIES,
        index=0,
        help=(
            "**Dual-wrinkle morphologies** (phase offset φ between two "
            "adjacent wrinkle centrelines):\n"
            "- **Stack** (φ = 0): peaks/troughs aligned, M_f = 1 — baseline.\n"
            "- **Convex** (φ = π/2): outward bulge, M_f < 1 — *best* under "
            "compression.\n"
            "- **Concave** (φ = −π/2): inward pinch, M_f > 1 — *worst* "
            "under compression (amplifies kink-band formation).\n\n"
            "**Single-wrinkle through-thickness modes:**\n"
            "- **Uniform**: full amplitude on every ply (no decay).\n"
            "- **Graded**: linear decay from the wrinkle core to the outer "
            "surfaces; the *Decay floor* slider sets the minimum amplitude "
            "fraction at the surface plies."
        ),
    )
    decay_floor = 0.0
    if morphology == "graded":
        decay_floor = st.slider(
            "Decay floor", 0.0, 1.0, 0.0, 0.05,
            help="Minimum amplitude fraction at the outer surfaces.",
        )

    loading = st.radio("Loading mode", ["compression", "tension"], horizontal=True)
    applied_strain_pct = st.number_input(
        "Applied strain [%]",
        min_value=-5.0, max_value=5.0,
        value=-1.0 if loading == "compression" else 1.0,
        step=0.1,
    )

    st.header("Material & layup")
    default_idx = MATERIAL_NAMES.index("IM7_8552") if "IM7_8552" in MATERIAL_NAMES else 0
    material_name = st.selectbox("Material", MATERIAL_NAMES, index=default_idx)
    ply_thickness = st.number_input(
        "Ply thickness [mm]",
        min_value=0.05, max_value=1.0, value=0.183, step=0.01,
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
        nx = st.number_input("Mesh divisions in x", 4, 64, 12, 2)
        ny = st.number_input("Mesh divisions in y", 4, 32, 6, 2)
        nz_per_ply = st.number_input("Mesh divisions per ply (z)", 1, 4, 1)
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
        fe = {
            "modulus_retention": float(result.modulus_retention),
            "retention_factors": {
                k: float(v) for k, v in (result.retention_factors or {}).items()
            },
            "baseline_fi": {
                k: float(v) for k, v in (result.baseline_fi or {}).items()
            },
            "n_nodes": int(result.mesh.n_nodes) if result.mesh else None,
            "n_elements": int(result.mesh.n_elements) if result.mesh else None,
            "max_displacement_mm": float(max_disp_mm),
            "critical_criterion": getattr(rep, "critical_criterion", None),
            "critical_mode": getattr(rep, "critical_mode", None),
            "critical_ply": getattr(rep, "critical_ply", None),
        }

    return {
        "summary": result.summary(),
        "max_angle_deg": float(np.degrees(result.max_angle_rad)),
        "analytical_knockdown": float(result.analytical_knockdown),
        "analytical_strength_MPa": float(result.analytical_strength_MPa),
        "damage_index": float(result.damage_index),
        "fe": fe,
    }


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

profile = GaussianSinusoidal(amplitude=amplitude, wavelength=wavelength, width=width)

tab_geom, tab_results, tab_export = st.tabs(["Geometry", "Results", "Export"])

with tab_geom:
    st.subheader("Wrinkle mid-surface profile")
    x = np.linspace(-3 * width, 3 * width, 800)
    z = profile.displacement(x)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(x, z, color="#1f77b4", linewidth=1.5)
    ax.axhline(0.0, color="grey", linewidth=0.5)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z(x) [mm]")
    ax.set_title(r"$z(x) = A \cdot \exp(-x^2 / w^2) \cdot \cos(2\pi x / \lambda)$")
    ax.grid(alpha=0.3)
    st.pyplot(fig, clear_figure=True)
    st.caption(
        f"Numerical θ_max = {np.degrees(profile.max_angle()):.2f}°  ·  "
        f"Closed-form approx arctan(2πA/λ) = "
        f"{np.degrees(profile.max_angle_approx()):.2f}°"
    )


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

    with st.spinner("Running analysis..."):
        try:
            results = run_analysis_cached(cfg_payload)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

    st.session_state["results"] = results
    st.session_state["cfg_payload"] = cfg_payload
    st.success("Analysis complete — see the **Results** tab.")


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
                vals = [fe["retention_factors"][c] for c in crits]
                fig_ret, ax_ret = plt.subplots(
                    figsize=(7, max(1.4, 0.45 * len(crits) + 1.0))
                )
                ax_ret.barh(crits, vals, color="#2ca02c")
                ax_ret.axvline(1.0, color="grey", linestyle="--", linewidth=0.8)
                ax_ret.set_xlabel("Retention (wrinkled / pristine)")
                ax_ret.set_xlim(0, max(1.05, max(vals) * 1.1))
                ax_ret.grid(axis="x", alpha=0.3)
                st.pyplot(fig_ret, clear_figure=True)
                st.caption(
                    "Per-criterion strength retention = applied strain at "
                    "first ply failure (wrinkled) / first ply failure "
                    "(pristine). Values < 1 indicate knockdown."
                )

            crit = fe.get("critical_criterion")
            if crit:
                st.markdown(
                    f"**Critical failure:** criterion `{crit}`"
                    + (
                        f", mode `{fe['critical_mode']}`"
                        if fe.get("critical_mode") else ""
                    )
                    + (
                        f", first failing ply index `{fe['critical_ply']}`"
                        if fe.get("critical_ply") is not None else ""
                    )
                )

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
        payload = {
            "config": dict(st.session_state["cfg_payload"]),
            "results": st.session_state["results"],
        }
        payload["config"]["angles"] = list(payload["config"].pop("angles_tuple"))
        st.download_button(
            "Download results as JSON",
            data=json.dumps(payload, indent=2).encode(),
            file_name="wrinklefe_results.json",
            mime="application/json",
        )
