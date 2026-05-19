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
from wrinklefe.core.layup import parse_layup
from wrinklefe.core.material import MaterialLibrary, OrthotropicMaterial
from wrinklefe.core.wrinkle import GaussianSinusoidal
from wrinklefe.io.export import (
    build_analysis_summary,
    render_summary_markdown,
    render_summary_pdf,
)

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
MATERIAL_NAMES = sorted(LIB.list_names())
MORPHOLOGIES = ["stack", "convex", "concave", "uniform", "graded"]

CUSTOM_MATERIAL_LABEL = "Custom…"
MATERIAL_OPTIONS = MATERIAL_NAMES + [CUSTOM_MATERIAL_LABEL]

# Field name -> (label, format, step, help) for the custom-material editor.
# Limited to the elastic + strength allowables that drive most analyses;
# hygrothermal/kink-band parameters fall back to the IM7/8552 defaults.
CUSTOM_MAT_ELASTIC_FIELDS = (
    ("E1", "E1 [MPa]", "%.0f"),
    ("E2", "E2 [MPa]", "%.0f"),
    ("E3", "E3 [MPa]", "%.0f"),
    ("G12", "G12 [MPa]", "%.0f"),
    ("G13", "G13 [MPa]", "%.0f"),
    ("G23", "G23 [MPa]", "%.0f"),
    ("nu12", "ν12", "%.3f"),
    ("nu13", "ν13", "%.3f"),
    ("nu23", "ν23", "%.3f"),
)
CUSTOM_MAT_STRENGTH_FIELDS = (
    ("Xt", "Xt [MPa]", "%.1f"),
    ("Xc", "Xc [MPa]", "%.1f"),
    ("Yt", "Yt [MPa]", "%.1f"),
    ("Yc", "Yc [MPa]", "%.1f"),
    ("Zt", "Zt [MPa]", "%.1f"),
    ("Zc", "Zc [MPa]", "%.1f"),
    ("S12", "S12 [MPa]", "%.1f"),
    ("S13", "S13 [MPa]", "%.1f"),
    ("S23", "S23 [MPa]", "%.1f"),
)

DEFAULT_LAYUP = "[0/45/-45/90]_3s"
DEFAULT_EXPERT_MODE = False
DEFAULT_MATERIAL = "IM7_8552"
DEFAULT_CUSTOM_NAME = "custom"
DEFAULT_PLY_THICKNESS = 0.183
DEFAULT_AMPLITUDE = 0.366
DEFAULT_WAVELENGTH = 16.0
DEFAULT_WIDTH = 12.0
DEFAULT_MORPHOLOGY = "stack"
DEFAULT_DECAY_FLOOR = 0.0
DEFAULT_LOADING = "compression"
DEFAULT_STRAIN_MAG_PCT = 1.0
DEFAULT_ANALYTICAL_ONLY = False
DEFAULT_NX = 12
DEFAULT_NY = 6
DEFAULT_NZ_PER_PLY = 1

# Keys for sidebar input widgets. Used by the "Reset to defaults" button to
# clear modified values from st.session_state so widgets fall back to their
# default= argument on the next rerun.
SIDEBAR_INPUT_KEYS = (
    "expert_mode",
    "sb_material",
    "sb_custom_name",
    "sb_ply_thickness",
    "sb_layup",
    "sb_amplitude",
    "sb_wavelength",
    "sb_width",
    "sb_morphology",
    "sb_decay_floor",
    "sb_loading",
    "sb_strain_mag_pct",
    "sb_analytical_only",
    "sb_nx",
    "sb_ny",
    "sb_nz_per_ply",
)


def _reset_sidebar_defaults() -> None:
    """Clear sidebar input keys from session_state so widgets reload defaults.

    Also clears the dynamic custom-material editor keys (``custom_*``), which
    are seeded from the chosen material on each rerun.
    """
    for k in SIDEBAR_INPUT_KEYS:
        st.session_state.pop(k, None)
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith("custom_"):
            st.session_state.pop(k, None)


# ---------------------------------------------------------------------------
# Sidebar helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _hero_schematic() -> bytes:
    """Three-panel cartoon: pristine laminate → wrinkled laminate → strength loss.

    Rendered above the tabs as a 60-second orientation for first-time
    visitors. Static — no parameters, cached forever per session.
    """
    fig, axes = plt.subplots(
        1, 3, figsize=(8.4, 1.95), dpi=110,
        gridspec_kw={"width_ratios": [1.0, 1.3, 0.7]},
    )
    n_plies = 7
    band_h = 0.10
    band_gap = 0.02
    pitch = band_h + band_gap
    ply_colors = [
        "#1e3a8a", "#cbd5e0", "#1e3a8a", "#cbd5e0",
        "#1e3a8a", "#cbd5e0", "#1e3a8a",
    ]

    # Panel 1 — pristine: stacked horizontal bands.
    ax = axes[0]
    for i in range(n_plies):
        y = (i - (n_plies - 1) / 2) * pitch
        ax.add_patch(plt.Rectangle(
            (-0.85, y - band_h / 2), 1.7, band_h,
            color=ply_colors[i], lw=0,
        ))
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.9, 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Pristine laminate", fontsize=9, pad=4)

    # Panel 2 — wrinkled: same bands deformed by a Gaussian-windowed cosine.
    ax = axes[1]
    x = np.linspace(-0.85, 0.85, 240)
    env = np.exp(-(x ** 2) / 0.28 ** 2)
    carrier = 0.10 * env * np.cos(2 * np.pi * x / 0.45)
    p_mid = (n_plies - 1) / 2.0
    for i in range(n_plies):
        y0 = (i - p_mid) * pitch
        # Slight through-thickness decay so the wrinkle "fades" outward.
        decay = 1.0 - 0.40 * abs(i - p_mid) / p_mid
        z_centre = y0 + decay * carrier
        ax.fill_between(
            x, z_centre - band_h / 2, z_centre + band_h / 2,
            color=ply_colors[i], lw=0,
        )
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.9, 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Wrinkled laminate", fontsize=9, pad=4)

    # Panel 3 — predicted knockdown: two-bar strength comparison.
    ax = axes[2]
    ax.bar([0], [1.00], color="#94a3b8", width=0.55)
    ax.bar([1], [0.62], color="#dc2626", width=0.55)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 1.25)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pristine", "wrinkled"], fontsize=8)
    ax.set_yticks([])
    ax.text(0, 1.04, "100%", ha="center", fontsize=8)
    ax.text(1, 0.66, "62%", ha="center", fontsize=8)
    ax.text(
        0.5, 1.18, "−38% strength",
        ha="center", fontsize=9, color="#dc2626", fontweight="bold",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Predicted knockdown", fontsize=9, pad=4)

    fig.subplots_adjust(
        left=0.02, right=0.98, top=0.85, bottom=0.05, wspace=0.30,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def _morphology_schematic(morphology: str) -> bytes:
    """Render a small cartoon schematic of a morphology as PNG bytes.

    Used inside the in-sidebar help popover so users can see at a glance
    what each morphology looks like before selecting one.
    """
    fig, ax = plt.subplots(figsize=(2.6, 1.6), dpi=110)
    x = np.linspace(-1.0, 1.0, 400)
    env = np.exp(-(x ** 2) / 0.32 ** 2)
    carrier = np.cos(2 * np.pi * x / 0.55)
    amp = 0.18
    blue, red = "#1f77b4", "#d62728"

    if morphology == "stack":
        z = amp * env * carrier
        ax.plot(x, z + 0.34, color=blue, lw=2.2)
        ax.plot(x, z - 0.34, color=blue, lw=2.2)
    elif morphology == "convex":
        z1 = amp * env * carrier
        z2 = amp * env * np.cos(2 * np.pi * x / 0.55 + np.pi / 2)
        ax.plot(x, z1 + 0.34, color=blue, lw=2.2)
        ax.plot(x, z2 - 0.34, color=blue, lw=2.2)
    elif morphology == "concave":
        z1 = amp * env * carrier
        z2 = amp * env * np.cos(2 * np.pi * x / 0.55 - np.pi / 2)
        ax.plot(x, z1 + 0.34, color=red, lw=2.2)
        ax.plot(x, z2 - 0.34, color=red, lw=2.2)
    elif morphology == "uniform":
        z = 0.14 * env * carrier
        for offset in np.linspace(-0.55, 0.55, 7):
            ax.plot(x, z + offset, color=blue, lw=1.4)
    elif morphology == "graded":
        offsets = np.linspace(-0.55, 0.55, 7)
        for offset in offsets:
            decay = 1.0 - abs(offset) / 0.55  # 1 at core, 0 at surfaces
            z = 0.18 * decay * env * carrier
            ax.plot(x, z + offset, color=blue, lw=1.4)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.85, 0.85)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.1)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Hero / orientation content — rendered inside the Overview tab below.
# Defined here as small helpers so the demo-button click handler can run
# before the tabs declaration (it needs to st.rerun() the whole script).
# ---------------------------------------------------------------------------

_HERO_INTRO_MD = (
    "Carbon-fibre composite parts — aircraft skins, wind blades, pressure "
    "vessels — develop tiny ripples (\"wrinkles\") during layup or cure. "
    "Even a 0.5 mm wrinkle can cut compressive strength by **30–60 %**. "
    "WrinkleFE predicts that loss in seconds so engineers and inspectors "
    "can decide: **scrap, repair, or accept.**"
)


# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------

with st.sidebar:
    expert_mode = st.toggle(
        "Expert mode", value=DEFAULT_EXPERT_MODE, key="expert_mode",
        help=(
            "**Off (default)** — simplified sidebar with the essentials: "
            "material, layup, amplitude, wavelength, loading, strain.\n\n"
            "**On** — full controls: ply thickness, envelope width, "
            "morphology selector, custom-material editor, decay floor, "
            "mesh density, and the full FE solve toggle."
        ),
    )

    st.markdown("**Material & layup**")
    _material_options = (
        MATERIAL_OPTIONS if expert_mode else MATERIAL_NAMES
    )
    default_idx = (
        _material_options.index(DEFAULT_MATERIAL)
        if DEFAULT_MATERIAL in _material_options else 0
    )
    material_choice = st.selectbox(
        "Material", _material_options, index=default_idx,
        key="sb_material",
        help=(
            "Pick a built-in carbon/epoxy or glass/epoxy system."
            + (
                f" Choose **{CUSTOM_MATERIAL_LABEL}** to enter your own "
                "ply properties."
                if expert_mode else
                " Switch on **Expert mode** at the top of the sidebar to "
                "enter custom ply properties."
            )
        ),
    )

    if expert_mode and material_choice == CUSTOM_MATERIAL_LABEL:
        # Seed the custom editor from a sensible default so users only have
        # to override the values they care about.
        _seed_name = (
            "IM7_8552" if "IM7_8552" in MATERIAL_NAMES else MATERIAL_NAMES[0]
        )
        _seed = LIB.get(_seed_name)
        with st.expander("Custom material properties", expanded=True):
            st.caption(
                "Defaults are seeded from "
                f"**{_seed_name}**. Hygrothermal coefficients, gamma_Y and "
                "LaRC parameters fall back to those defaults — only the "
                "elastic constants and strength allowables below are "
                "exposed here."
            )
            custom_name = st.text_input(
                "Material name", value=DEFAULT_CUSTOM_NAME, max_chars=64,
                key="sb_custom_name",
                help="Free-text label used in exports and plots.",
            )
            st.markdown("*Elastic constants*")
            elastic_vals: dict[str, float] = {}
            elastic_cols = st.columns(3)
            for i, (key, label, fmt) in enumerate(CUSTOM_MAT_ELASTIC_FIELDS):
                col = elastic_cols[i % 3]
                seed_val = float(getattr(_seed, key))
                step = 0.01 if key.startswith("nu") else 100.0
                elastic_vals[key] = col.number_input(
                    label, min_value=0.0, value=seed_val,
                    step=step, format=fmt, key=f"custom_{key}",
                )
            st.markdown("*Strength allowables*")
            strength_vals: dict[str, float] = {}
            strength_cols = st.columns(3)
            for i, (key, label, fmt) in enumerate(CUSTOM_MAT_STRENGTH_FIELDS):
                col = strength_cols[i % 3]
                seed_val = float(getattr(_seed, key))
                strength_vals[key] = col.number_input(
                    label, min_value=0.0, value=seed_val,
                    step=1.0, format=fmt, key=f"custom_{key}",
                )

        # Inherit non-exposed fields from the seed material.
        material_dict = _seed.to_dict()
        material_dict.update(elastic_vals)
        material_dict.update(strength_vals)
        material_dict["name"] = custom_name.strip() or "custom"
    else:
        material_dict = LIB.get(material_choice).to_dict()

    if expert_mode:
        ply_thickness = st.number_input(
            "Ply thickness [mm]",
            min_value=0.05, max_value=1.0,
            value=DEFAULT_PLY_THICKNESS, step=0.01,
            key="sb_ply_thickness",
            help=(
                "Thickness of one ply in mm. Default 0.183 mm matches "
                "CYCOM X850/T800. Total laminate thickness = ply_thickness × "
                "number of plies in the layup."
            ),
        )
    else:
        ply_thickness = DEFAULT_PLY_THICKNESS
    layup_str = st.text_area(
        "Layup",
        value=DEFAULT_LAYUP, height=80,
        key="sb_layup",
        help=(
            "Accepts contracted notation like `[0/45/-45/90]_3s` "
            "or an explicit comma-separated list of angles in degrees.\n\n"
            "**Contracted notation** — sublaminate in square brackets, "
            "plies separated by `/`, optional repeat count and trailing "
            "`s` for symmetry:\n\n"
            "| Input | Expanded plies |\n"
            "|---|---|\n"
            "| `[0/45/-45/90]_3s` | `0/45/-45/90` × 3 then mirrored — **24 plies** (default quasi-isotropic) |\n"
            "| `[0/±45/90]s` | `0,45,-45,90` mirrored — **8 plies** |\n"
            "| `[0_2/90]_2` | `0,0,90` × 2 — **6 plies** |\n"
            "| `[±30]_2` | `30,-30` × 2 — **4 plies** |\n\n"
            "Modifiers:\n"
            "- `±θ` expands to `θ, -θ` (two plies).\n"
            "- `θ_n` repeats a single ply *n* times (e.g. `0_4` ⇒ four 0° plies).\n"
            "- `_n` after the bracket repeats the whole sublaminate *n* times.\n"
            "- Trailing `s` mirrors the stack to enforce symmetry.\n\n"
            "**Explicit list** — comma-, semicolon-, or newline-separated "
            "angles, e.g. `0, 45, -45, 90, 90, -45, 45, 0`. Both forms can be "
            "edited freely in the textarea above."
        ),
    )

    st.markdown("**Wrinkle geometry**")
    amplitude = st.number_input(
        "Amplitude A [mm]",
        min_value=0.0, max_value=5.0,
        value=DEFAULT_AMPLITUDE, step=0.01,
        key="sb_amplitude",
        help=(
            "Peak displacement of the wrinkled mid-surface from the flat "
            "reference (crest height, NOT peak-to-peak). Measure as "
            "A = (z_max − z_min) / 2 from a cross-section micrograph."
        ),
    )
    wavelength = st.number_input(
        "Wavelength λ [mm]",
        min_value=1.0, max_value=200.0,
        value=DEFAULT_WAVELENGTH, step=0.5,
        key="sb_wavelength",
        help=(
            "Spatial period of the cosine carrier. Larger λ at fixed A "
            "lowers the peak fibre angle θ_max ≈ arctan(2πA/λ)."
        ),
    )
    if expert_mode:
        width = st.number_input(
            "Envelope width w [mm]",
            min_value=1.0, max_value=200.0,
            value=DEFAULT_WIDTH, step=0.5,
            key="sb_width",
            help=(
                "Half-width of the Gaussian envelope multiplying the cosine. "
                "Smaller w localises the wrinkle to a few wavelengths near x = 0."
            ),
        )
    else:
        width = DEFAULT_WIDTH

    st.markdown(
        "**Morphology & loading**" if expert_mode else "**Loading**"
    )
    if expert_mode:
        _morph_default_idx = (
            MORPHOLOGIES.index(DEFAULT_MORPHOLOGY)
            if DEFAULT_MORPHOLOGY in MORPHOLOGIES else 0
        )
        morphology = st.selectbox(
            "Morphology", MORPHOLOGIES, index=_morph_default_idx,
            key="sb_morphology",
            help=(
                "Wrinkle shape pattern through the laminate thickness. The cartoon "
                "below the dropdown shows the active choice; switch values to see "
                "the others.\n\n"
                "**Dual-wrinkle modes** — two adjacent wrinkles offset by phase φ "
                "between their centrelines (Jin et al. 2026):\n"
                "- *stack* (φ = 0): peaks aligned. M_f = 1.0 — baseline.\n"
                "- *convex* (φ = π/2): interface bulges outward. M_f < 1 — "
                "*least* damaging in compression.\n"
                "- *concave* (φ = −π/2): interface pinches inward. M_f > 1 — "
                "*most* damaging in compression.\n\n"
                "**Single-wrinkle modes** — one wrinkle, varying through-thickness "
                "amplitude:\n"
                "- *uniform*: full amplitude on every ply.\n"
                "- *graded*: linear decay from wrinkle core to surfaces, controlled "
                "by the **Decay floor** slider (0 = full decay, 1 = uniform)."
            ),
        )
        decay_floor = DEFAULT_DECAY_FLOOR
        if morphology == "graded":
            decay_floor = st.slider(
                "Decay floor", 0.0, 1.0, DEFAULT_DECAY_FLOOR, 0.05,
                key="sb_decay_floor",
                help="Minimum amplitude fraction at the outer surfaces.",
            )

        st.image(
            _morphology_schematic(morphology),
            caption=f"{morphology.capitalize()} morphology",
            width="stretch",
        )
    else:
        # Sensible defaults; the cartoon and morphology selector are exposed
        # in Expert mode for users who want to compare phase offsets.
        morphology = DEFAULT_MORPHOLOGY
        decay_floor = DEFAULT_DECAY_FLOOR

    _loading_options = ["compression", "tension"]
    _loading_default_idx = (
        _loading_options.index(DEFAULT_LOADING)
        if DEFAULT_LOADING in _loading_options else 0
    )
    loading = st.radio(
        "Loading mode", _loading_options, horizontal=True,
        index=_loading_default_idx, key="sb_loading",
    )
    strain_mag_pct = st.number_input(
        "Applied strain magnitude [%]",
        min_value=0.0, max_value=5.0,
        value=DEFAULT_STRAIN_MAG_PCT, step=0.1,
        key="sb_strain_mag_pct",
        help=(
            "Magnitude only — the sign is taken from the loading mode "
            "(compression → negative, tension → positive). Editing this "
            "value is preserved when you toggle the loading mode."
        ),
    )
    applied_strain_pct = -strain_mag_pct if loading == "compression" else strain_mag_pct

    if expert_mode:
        with st.expander("Advanced — mesh & solver", expanded=False):
            analytical_only = st.checkbox(
                "Analytical only (skip FE solve)",
                value=DEFAULT_ANALYTICAL_ONLY,
                key="sb_analytical_only",
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
                "Mesh divisions in x", 4, 64, DEFAULT_NX, 2,
                disabled=analytical_only,
                key="sb_nx",
                help=(
                    "Hex elements along the wrinkle (x) direction across the "
                    "domain length. More elements resolve the curvature but "
                    "scale solve time roughly linearly."
                ),
            )
            ny = st.number_input(
                "Mesh divisions in y", 4, 32, DEFAULT_NY, 2,
                disabled=analytical_only,
                key="sb_ny",
                help=(
                    "Hex elements across the laminate width (y). Wrinkle is "
                    "uniform in y, so a coarse mesh is usually adequate."
                ),
            )
            nz_per_ply = st.number_input(
                "Mesh divisions per ply (z)", 1, 4, DEFAULT_NZ_PER_PLY,
                disabled=analytical_only,
                key="sb_nz_per_ply",
                help=(
                    "Hex elements through the thickness of every individual "
                    "ply. Increase to capture interlaminar stress gradients."
                ),
            )
            if analytical_only:
                st.caption(
                    "Mesh inputs are inactive — the closed-form analytical "
                    "knockdown skips the FE solve."
                )
            else:
                st.caption(
                    ":hourglass_flowing_sand: Full FE solve. On Streamlit "
                    "Cloud's CPU this typically takes 30–90 s for the default "
                    "mesh; reducing nx, ny, or nz_per_ply speeds it up."
                )
    else:
        # Novice path: fast analytical-only run, no FE. Switch on Expert
        # mode to expose mesh density and the FE toggle.
        analytical_only = True
        nx = DEFAULT_NX
        ny = DEFAULT_NY
        nz_per_ply = DEFAULT_NZ_PER_PLY
        st.caption(
            "Quick analytical estimate. Switch on **Expert mode** above "
            "for the full FE solve, stress fields, and per-ply failure indices."
        )

    run_clicked = st.button("Run analysis", type="primary", width="stretch")

    st.divider()
    reset_clicked = st.button(
        "↻ Reset to defaults",
        width="stretch",
        help=(
            "Restore every sidebar input (material, layup, wrinkle geometry, "
            "loading, mesh) to its original default value. Modified entries "
            "are discarded; the page reruns immediately."
        ),
    )
    if reset_clicked:
        _reset_sidebar_defaults()
        st.rerun()

    with st.expander("What do these terms mean?", expanded=False):
        st.markdown(
            "**Wrinkle** — a ripple in the otherwise flat plies of a "
            "composite part, introduced during layup or cure.\n\n"
            "**Layup** — the stacking sequence of plies (e.g. 0°/45°/-45°/90°) "
            "that gives the laminate its directional stiffness.\n\n"
            "**Morphology** — how the wrinkle is shaped through the "
            "thickness: aligned (stack), bulging (convex), pinched "
            "(concave), uniform, or fading (graded).\n\n"
            "**FE solve** — a 3-D finite-element calculation of the actual "
            "stresses inside the wrinkled region. Slower than the analytical "
            "estimate but gives stress fields and per-ply failure.\n\n"
            "**Knockdown** — the fraction of pristine strength the part "
            "still has. `0.62` means the wrinkle reduces strength to 62 % "
            "of the pristine value.\n\n"
            "**Damage index D** — a 0-to-1 score; `0` means no damage from "
            "the wrinkle, `1` means complete loss of load-carrying capacity."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def run_analysis_cached(cfg_payload: tuple) -> dict:
    """Cached analysis run. cfg_payload is a hashable tuple of config items."""
    cfg_dict = dict(cfg_payload)
    angles = list(cfg_dict.pop("angles_tuple"))
    material_dict = dict(cfg_dict.pop("material_tuple"))
    material = OrthotropicMaterial.from_dict(material_dict)
    cfg = AnalysisConfig(
        amplitude=cfg_dict["amplitude"],
        wavelength=cfg_dict["wavelength"],
        width=cfg_dict["width"],
        morphology=cfg_dict["morphology"],
        decay_floor=cfg_dict["decay_floor"],
        loading=cfg_dict["loading"],
        material=material,
        angles=angles,
        ply_thickness=cfg_dict["ply_thickness"],
        applied_strain=cfg_dict["applied_strain"],
        nx=cfg_dict.get("nx", 12),
        ny=cfg_dict.get("ny", 6),
        nz_per_ply=cfg_dict.get("nz_per_ply", 1),
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
_demo_pending = st.session_state.pop("_demo_pending", False)
if run_clicked or _demo_pending:
    if _demo_pending:
        # Hardcoded analytical-only demo so a first-time visitor can land on
        # the Results tab in ~2 s without touching the sidebar.
        _demo_seed_name = (
            "IM7_8552" if "IM7_8552" in MATERIAL_NAMES else MATERIAL_NAMES[0]
        )
        _demo_material_dict = LIB.get(_demo_seed_name).to_dict()
        _demo_layup = (
            0, 45, -45, 90, 0, 45, -45, 90, 0, 45, -45, 90,
            90, -45, 45, 0, 90, -45, 45, 0, 90, -45, 45, 0,
        )
        cfg_payload = tuple(sorted({
            "amplitude": 0.366,
            "wavelength": 16.0,
            "width": 12.0,
            "morphology": "stack",
            "decay_floor": 0.0,
            "loading": "compression",
            "ply_thickness": 0.183,
            "angles_tuple": _demo_layup,
            "applied_strain": -0.01,
            "material_tuple": tuple(sorted(_demo_material_dict.items())),
            "analytical_only": True,
        }.items()))
    else:
        try:
            layup = parse_layup(layup_str)
        except ValueError as e:
            st.error(f"Could not parse layup: {e}")
            st.stop()

        try:
            # Validate the custom material up front so the cache isn't keyed
            # on an invalid OrthotropicMaterial that will only blow up inside
            # AnalysisConfig.
            OrthotropicMaterial.from_dict(material_dict)
        except ValueError as e:
            st.error(f"Invalid custom material: {e}")
            st.stop()

        cfg_items: dict = {
            "amplitude": amplitude,
            "wavelength": wavelength,
            "width": width,
            "morphology": morphology,
            "decay_floor": decay_floor,
            "loading": loading,
            "ply_thickness": ply_thickness,
            "angles_tuple": tuple(layup),
            "applied_strain": applied_strain_pct / 100.0,
            "material_tuple": tuple(sorted(material_dict.items())),
            "analytical_only": bool(analytical_only),
        }
        # Mesh keys only matter for the FE path; omitting them in
        # analytical-only mode means the cache key doesn't churn when the
        # user tweaks nx/ny/nz.
        if not analytical_only:
            cfg_items["nx"] = int(nx)
            cfg_items["ny"] = int(ny)
            cfg_items["nz_per_ply"] = int(nz_per_ply)
        cfg_payload = tuple(sorted(cfg_items.items()))

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
    if _demo_pending:
        st.success(
            "✓ Demo analysis complete with sensible defaults "
            "(IM7/8552, quasi-isotropic 24-ply, 1 % compression). "
            "Open the **Results** tab below to see the numbers — "
            "then tweak any sidebar parameter and click *Run analysis* "
            "to compare against your own configuration."
        )


tab_overview, tab_configure, tab_results, tab_export = st.tabs(
    ["Overview", "Configure", "Results", "Export"]
)

with tab_overview:
    st.markdown(_HERO_INTRO_MD)
    st.image(_hero_schematic(), width="stretch")

    _demo_cols = st.columns([2, 1, 2])
    if _demo_cols[1].button(
        "▶ Try a demo analysis",
        type="primary",
        width="stretch",
        help=(
            "One-click analytical run with IM7/8552 and a quasi-isotropic "
            "[0/45/-45/90]_3s layup. Lands on the Results tab in ~2 s."
        ),
    ):
        st.session_state["_demo_pending"] = True
        st.rerun()

    st.markdown("---")
    st.markdown(
        "**What to do next**\n\n"
        "1. *Try the demo above* for an instant analytical run with sensible "
        "defaults — you'll land back here with results on the **Results** tab.\n"
        "2. *Configure your own laminate* using the sidebar (material, layup, "
        "wrinkle amplitude and wavelength, loading mode and strain).\n"
        "3. *Click* **▶ Run analysis** at the bottom of the sidebar.\n"
        "4. *Review* the Results tab — the **Before / after** card up top "
        "shows strength and stiffness loss in plain language.\n"
        "5. *Export* the run as JSON or CSV from the **Export** tab.\n\n"
        "Switch on **Expert mode** at the top of the sidebar to expose the "
        "morphology selector, custom-material editor, decay floor, mesh "
        "density, and the full FE solve."
    )

with tab_configure:
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
        st.markdown("### No results yet")
        st.markdown(
            "Click **▶ Run analysis** in the sidebar. Once the analysis "
            "completes this tab will show:\n\n"
            "- The peak fibre misalignment angle in the wrinkle\n"
            "- The analytical and FE-predicted strength knockdown\n"
            "- A breakdown by failure criterion "
            "(Tsai-Wu, Hashin, Puck, LaRC)\n"
            "- A 3-D view of the stress field through the laminate\n"
            "- Per-ply failure indices and the controlling ply"
        )
        st.caption(
            "Tip: tick **Analytical only** inside the *Advanced — mesh & "
            "solver* expander for a fast estimate that skips the FE solve. "
            "Open the **What do these terms mean?** expander at the bottom "
            "of the sidebar if any term is unfamiliar."
        )
    else:
        r = st.session_state["results"]

        # ------------------------------------------------------------------
        # Before / after comparison card (#98 item 6)
        # ------------------------------------------------------------------
        _knockdown = float(r.get("analytical_knockdown", 1.0))
        _wrinkled_strength = float(r.get("analytical_strength_MPa", 0.0))
        _pristine_strength = (
            _wrinkled_strength / _knockdown if _knockdown > 1e-6 else None
        )
        _strength_delta_pct = (1.0 - _knockdown) * 100.0

        _cfg_runtime = dict(st.session_state.get("cfg_payload", ()))
        _mat_runtime = dict(_cfg_runtime.get("material_tuple", ()))
        _E1_pristine_GPa: float | None = None
        if _mat_runtime.get("E1"):
            _E1_pristine_GPa = float(_mat_runtime["E1"]) / 1000.0

        _fe_runtime = r.get("fe")
        _E1_wrinkled_GPa: float | None = None
        _stiffness_delta_pct: float | None = None
        if _fe_runtime is not None and _E1_pristine_GPa is not None:
            _modulus_ret = float(_fe_runtime.get("modulus_retention", 1.0))
            _E1_wrinkled_GPa = _E1_pristine_GPa * _modulus_ret
            _stiffness_delta_pct = (1.0 - _modulus_ret) * 100.0

        st.subheader("Before / after this wrinkle")
        ba_cols = st.columns(2)
        with ba_cols[0]:
            st.markdown("**Pristine baseline**")
            if _pristine_strength is not None:
                st.markdown(f"Strength · **{_pristine_strength:,.0f} MPa**")
            if _E1_pristine_GPa is not None:
                st.markdown(f"Stiffness (E₁) · **{_E1_pristine_GPa:.1f} GPa**")
        with ba_cols[1]:
            st.markdown("**Your wrinkled laminate**")
            _strength_arrow = "▼" if _strength_delta_pct >= 0 else "▲"
            st.markdown(
                f"Strength · **{_wrinkled_strength:,.0f} MPa**  "
                f"({_strength_arrow} {abs(_strength_delta_pct):.0f} %)"
            )
            if _E1_wrinkled_GPa is not None and _stiffness_delta_pct is not None:
                _stiff_arrow = "▼" if _stiffness_delta_pct >= 0 else "▲"
                st.markdown(
                    f"Stiffness (E₁) · **{_E1_wrinkled_GPa:.1f} GPa**  "
                    f"({_stiff_arrow} {abs(_stiffness_delta_pct):.1f} %)"
                )
            else:
                st.markdown(
                    "Stiffness (E₁) · _run with FE on to compute_"
                )

        if _stiffness_delta_pct is not None:
            st.caption(
                f"This wrinkle costs you **{abs(_strength_delta_pct):.0f} %** "
                f"of strength and **{abs(_stiffness_delta_pct):.1f} %** of "
                "stiffness. Wrinkles almost always hurt strength far more "
                "than stiffness."
            )
        else:
            st.caption(
                f"This wrinkle costs you **{abs(_strength_delta_pct):.0f} %** "
                "of strength. Untick *Analytical only* in the sidebar's "
                "*Advanced* expander to also see the stiffness drop."
            )

        st.subheader("Analytical predictions")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Max fibre misalignment", f"{r['max_angle_deg']:.2f}°",
            help=(
                "How far the worst fibres have rotated away from straight. "
                "Below ~3° is mild; above ~10° starts to dominate failure."
            ),
        )
        c2.metric(
            "Analytical knockdown", f"{r['analytical_knockdown']:.3f}",
            help=(
                "Fraction of pristine strength remaining. `1.0` = no loss; "
                "`0.5` = half the strength lost; lower is worse."
            ),
        )
        c3.metric(
            "Predicted strength", f"{r['analytical_strength_MPa']:.1f} MPa",
            help=(
                "Closed-form failure stress estimate for the wrinkled "
                "laminate. Compare against the allowable stress your "
                "design needs to carry."
            ),
        )
        c4.metric(
            "Damage index D", f"{r['damage_index']:.3f}",
            help=(
                "0-to-1 severity score. `0` = no damage; `1` = total loss "
                "of load-carrying capacity. Above 0.5 is severe."
            ),
        )

        d1, d2, d3 = st.columns(3)
        d1.metric(
            "Morphology factor M_f", f"{r['morphology_factor']:.3f}",
            help=(
                "How damaging this wrinkle shape is vs the stack baseline "
                "(`1.0`). `>1` = more damaging; `<1` = less damaging."
            ),
        )
        d2.metric(
            "Effective fibre angle θ_eff", f"{r['effective_angle_deg']:.2f}°",
            help=(
                "Single-angle representation of the wrinkle that drives "
                "the analytical knockdown. Roughly the peak misalignment "
                "scaled by the wrinkle envelope."
            ),
        )
        d3.metric(
            "Yield strain γ_Y eff", f"{r['gamma_Y_eff']:.4f}",
            help=(
                "Matrix shear-yield strain used by the Budiansky-Fleck "
                "kink-band model. Internal parameter — useful for "
                "advanced calibration, less so for first-time interpretation."
            ),
        )

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
            f1.metric(
                "Modulus retention", f"{fe['modulus_retention']:.3f}",
                help=(
                    "Fraction of pristine stiffness remaining. Usually "
                    "stays above `0.9` even for severe wrinkles — wrinkles "
                    "hurt strength far more than they hurt stiffness."
                ),
            )
            worst = (
                min(fe["retention_factors"].values())
                if fe["retention_factors"] else None
            )
            f2.metric(
                "Strength retention (worst criterion)",
                f"{worst:.3f}" if worst is not None else "—",
                help=(
                    "Fraction of strength remaining under the *worst* of "
                    "all failure criteria evaluated. `1.0` = no loss; the "
                    "controlling criterion is shown below."
                ),
            )
            f3.metric(
                "Max displacement", f"{fe['max_displacement_mm']:.4f} mm",
                help=(
                    "Peak nodal displacement magnitude in the FE solve. "
                    "Sanity-check that this scales with the applied strain; "
                    "extremely small values suggest constraint issues."
                ),
            )

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
                y_centers = fe["element_centers"][:, 1]
                y_unique = np.unique(y_centers)

                def _y_station_slider() -> float | None:
                    """Render the shared keyed y-station slider.

                    Uses a stable ``key`` so the chosen station is
                    preserved across view-mode switches. Returns
                    ``None`` if the mesh has no y-variation to scrub.
                    """
                    if y_unique.size <= 1:
                        return None
                    return st.select_slider(
                        "y-station [mm]",
                        options=[float(y) for y in y_unique],
                        value=float(y_unique[len(y_unique) // 2]),
                        key="viz_y_station",
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
                    st.plotly_chart(fig_3d, width="stretch")

                    st.markdown("**y-slice scrubber**")
                    y_station = _y_station_slider()
                    if y_station is not None:
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
                            st.plotly_chart(fig_slice, width="stretch")
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
                    st.plotly_chart(fig_3d, width="stretch")
                    # No y-slice for the deformed-mesh view: a stress
                    # component picker would be confusing here and a
                    # displacement-magnitude slice helper does not
                    # exist. See issue #77.
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
                        st.plotly_chart(fig_3d, width="stretch")

                        st.markdown("**y-slice scrubber**")
                        y_station = _y_station_slider()
                        if y_station is not None:
                            fig_slice = streamlit_viz.fi_y_slice_figure(
                                fe["element_centers"], fe["elements"],
                                fe["nodes"], fi_dict[crit_for_3d], y_station,
                                criterion=crit_for_3d,
                            )
                            if fig_slice is not None:
                                st.plotly_chart(fig_slice, width="stretch")

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
        payload["config"]["material"] = dict(payload["config"].pop("material_tuple"))

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

        # ------------------------------------------------------------------
        # Analysis validation summary — NCR attachment
        # ------------------------------------------------------------------
        st.divider()
        st.subheader("Analysis validation summary (NCR attachment)")
        st.caption(
            "A concise engineering validation of this wrinkle, intended to "
            "be attached to a Nonconformance Report. It carries the "
            "geometry, laminate, WrinkleFE analysis, cited criteria, and a "
            "**non-binding** recommended disposition — no part/serial or "
            "MRB sign-off (that lives on the NCR itself). It does not issue "
            "a final disposition; a qualified Material Review Board reviews, "
            "may modify, and approves the outcome."
        )

        _cfg = dict(st.session_state["cfg_payload"])
        _mat = dict(_cfg.get("material_tuple", ()))
        _angles = list(_cfg.get("angles_tuple", ()))
        _res = st.session_state["results"]

        with st.form("summary_form"):
            summary_reference = st.text_input(
                "Reference (optional)",
                placeholder="e.g. NCR no. or part reference",
            )
            summary_prepared_by = st.text_input("Prepared by (optional)")
            summary_notes = st.text_area(
                "Engineering notes (optional)", height=80
            )
            summary_submit = st.form_submit_button(
                "Generate analysis summary"
            )

        if summary_submit:
            _fe = _res.get("fe")
            summary = build_analysis_summary(
                defect={
                    "amplitude_mm": _cfg.get("amplitude"),
                    "wavelength_mm": _cfg.get("wavelength"),
                    "width_mm": _cfg.get("width"),
                    "morphology": _cfg.get("morphology"),
                    "loading": _cfg.get("loading"),
                    "ply_thickness_mm": _cfg.get("ply_thickness"),
                    "n_plies": len(_angles),
                    "layup": _angles,
                    "material_name": _mat.get("name"),
                },
                engineering={
                    "analytical_knockdown": _res.get("analytical_knockdown"),
                    "analytical_strength_MPa": _res.get(
                        "analytical_strength_MPa"
                    ),
                    "damage_index": _res.get("damage_index"),
                    "max_angle_deg": _res.get("max_angle_deg"),
                    "effective_angle_deg": _res.get("effective_angle_deg"),
                    "morphology_factor": _res.get("morphology_factor"),
                    "fe": (
                        {
                            "modulus_retention": _fe.get("modulus_retention"),
                            "retention_factors": _fe.get(
                                "retention_factors"
                            ),
                            "critical_criterion": _fe.get(
                                "critical_criterion"
                            ),
                            "critical_mode": _fe.get("critical_mode"),
                            "critical_ply": _fe.get("critical_ply"),
                        }
                        if _fe
                        else None
                    ),
                },
                reference=summary_reference,
                prepared_by=summary_prepared_by,
                notes=summary_notes,
                tool_version=_wrinklefe_version,
            )

            _dr = summary["disposition_recommendation"]
            st.success(
                f"Severity: **{_dr['severity']}** · Recommended path "
                f"(non-binding): {_dr['recommended_path']}"
            )
            summary_md = render_summary_markdown(summary)
            with st.expander("Preview summary", expanded=True):
                st.markdown(summary_md)

            _fn_base = (
                (summary_reference or "wrinkle_analysis_summary")
                .strip()
                .replace(" ", "_")
                or "wrinkle_analysis_summary"
            )
            dl1, dl2, dl3 = st.columns(3)
            with dl1:
                st.download_button(
                    "Download summary (PDF)",
                    data=render_summary_pdf(summary),
                    file_name=f"{_fn_base}.pdf",
                    mime="application/pdf",
                )
            with dl2:
                st.download_button(
                    "Download summary (Markdown)",
                    data=summary_md.encode(),
                    file_name=f"{_fn_base}.md",
                    mime="text/markdown",
                )
            with dl3:
                st.download_button(
                    "Download summary (JSON)",
                    data=json.dumps(summary, indent=2).encode(),
                    file_name=f"{_fn_base}.json",
                    mime="application/json",
                )
