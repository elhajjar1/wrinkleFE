"""Streamlit deployment entry point for WrinkleFE.

Provides a web UI around ``wrinklefe.analysis.WrinkleAnalysis`` so the
package can be exercised on Streamlit Cloud without the PyQt6 desktop GUI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary
from wrinklefe.core.wrinkle import GaussianSinusoidal


st.set_page_config(page_title="WrinkleFE", layout="wide")
st.title("WrinkleFE")
st.caption(
    "Finite element analysis of wrinkled composite laminates. "
    "Configure a wrinkle, run the analytical pipeline, and inspect the "
    "predicted knockdown."
)

@st.cache_resource
def _material_library() -> MaterialLibrary:
    return MaterialLibrary()


library = _material_library()
material_names = library.list_names()

DEFAULT_LAYUPS: dict[str, list[float]] = {
    "Quasi-isotropic [0/45/-45/90]_3s": (
        [0, 45, -45, 90] * 3 + list(reversed([0, 45, -45, 90] * 3))
    ),
    "Cross-ply [0/90]_6s": [0, 90] * 6 + list(reversed([0, 90] * 6)),
    "Unidirectional [0]_24": [0.0] * 24,
    "Custom": [],
}

with st.sidebar:
    st.header("Configuration")

    st.subheader("Material")
    material_name = st.selectbox(
        "Material system",
        material_names,
        index=material_names.index("IM7_8552") if "IM7_8552" in material_names else 0,
    )
    material = library.get(material_name)

    st.subheader("Layup")
    layup_name = st.selectbox("Stacking sequence", list(DEFAULT_LAYUPS.keys()))
    if layup_name == "Custom":
        layup_text = st.text_input(
            "Comma-separated ply angles (deg)",
            value="0,45,-45,90,0,45,-45,90,0,45,-45,90,90,-45,45,0,90,-45,45,0,90,-45,45,0",
        )
        try:
            angles = [float(x.strip()) for x in layup_text.split(",") if x.strip()]
        except ValueError:
            st.error("Could not parse ply angles. Use comma-separated numbers.")
            st.stop()
        if not angles:
            st.error("Layup must contain at least one ply.")
            st.stop()
    else:
        angles = [float(a) for a in DEFAULT_LAYUPS[layup_name]]
    st.caption(f"{len(angles)} plies")

    st.subheader("Wrinkle geometry")
    amplitude = st.slider("Amplitude A (mm)", 0.0, 2.0, 0.366, 0.001)
    wavelength = st.slider("Wavelength λ (mm)", 2.0, 60.0, 16.0, 0.5)
    width = st.slider("Gaussian half-width w (mm)", 1.0, 60.0, 12.0, 0.5)

    st.subheader("Morphology & loading")
    morphology = st.selectbox("Morphology", ["stack", "convex", "concave"])
    loading = st.selectbox("Loading mode", ["compression", "tension"])
    applied_strain = st.number_input(
        "Applied nominal strain",
        value=-0.01 if loading == "compression" else 0.01,
        step=0.001,
        format="%.4f",
    )

    st.subheader("Solver options")
    analytical_only = st.checkbox(
        "Analytical only (skip FE assembly)",
        value=True,
        help="Recommended on Streamlit Cloud; the full FE solve is memory-intensive.",
    )

    run = st.button("Run analysis", type="primary", use_container_width=True)


@st.cache_data
def _plot_profile(amp: float, lam: float, w: float) -> plt.Figure:
    profile = GaussianSinusoidal(amplitude=amp, wavelength=lam, width=w, center=0.0)
    x = np.linspace(-1.5 * lam, 1.5 * lam, 400)
    z = profile.displacement(x)
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(x, z, color="#1f77b4")
    ax.axhline(0.0, color="0.7", linewidth=0.8)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title("Mid-surface profile")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return fig


col_geom, col_mat = st.columns([2, 1])
with col_geom:
    st.subheader("Wrinkle profile")
    st.pyplot(_plot_profile(amplitude, wavelength, width), clear_figure=True)
with col_mat:
    st.subheader("Material card")
    st.write(
        {
            "Name": material.name,
            "E1 (MPa)": material.E1,
            "E2 (MPa)": material.E2,
            "G12 (MPa)": material.G12,
            "Xt (MPa)": material.Xt,
            "Xc (MPa)": material.Xc,
        }
    )

if not run:
    st.info("Set parameters in the sidebar and click **Run analysis**.")
    st.stop()

config = AnalysisConfig(
    amplitude=amplitude,
    wavelength=wavelength,
    width=width,
    morphology=morphology,
    loading=loading,
    material=material,
    angles=angles,
    applied_strain=applied_strain,
    analytical_only=analytical_only,
)

with st.spinner("Running WrinkleFE analysis…"):
    result = WrinkleAnalysis(config).run()

st.subheader("Results")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Knockdown", f"{result.analytical_knockdown:.3f}")
m2.metric("Predicted strength", f"{result.analytical_strength_MPa:.1f} MPa")
m3.metric("Max fiber angle", f"{np.degrees(result.max_angle_rad):.2f}°")
m4.metric("Damage index D", f"{result.damage_index:.3f}")

with st.expander("Full text summary", expanded=True):
    st.code(result.summary(), language="text")

if result.tension_mechanisms:
    with st.expander("Tension mechanism decomposition"):
        st.json(result.tension_mechanisms)
