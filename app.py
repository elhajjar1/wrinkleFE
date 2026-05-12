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
import re
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
from wrinklefe.core.material import MaterialLibrary, OrthotropicMaterial
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


# ---------------------------------------------------------------------------
# Sidebar helpers
# ---------------------------------------------------------------------------

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
# Sidebar inputs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("**Material & layup**")
    default_idx = (
        MATERIAL_OPTIONS.index("IM7_8552") if "IM7_8552" in MATERIAL_OPTIONS else 0
    )
    material_choice = st.selectbox(
        "Material", MATERIAL_OPTIONS, index=default_idx,
        help=(
            "Pick a built-in carbon/epoxy or glass/epoxy system, or choose "
            f"**{CUSTOM_MATERIAL_LABEL}** to enter your own ply properties."
        ),
    )

    if material_choice == CUSTOM_MATERIAL_LABEL:
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
                "Material name", value="custom", max_chars=64,
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
        "Layup",
        value=DEFAULT_LAYUP, height=80,
        help=(
            "Accepts contracted notation like `[0/45/-45/90]_3s` "
            "or an explicit comma-separated list of angles in degrees."
        ),
    )
    with st.popover("Layup notation help", use_container_width=True):
        st.markdown(
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
        )

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
    morphology = st.selectbox(
        "Morphology", MORPHOLOGIES, index=0,
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
    decay_floor = 0.0
    if morphology == "graded":
        decay_floor = st.slider(
            "Decay floor", 0.0, 1.0, 0.0, 0.05,
            help="Minimum amplitude fraction at the outer surfaces.",
        )

    st.image(
        _morphology_schematic(morphology),
        caption=f"{morphology.capitalize()} morphology",
        use_container_width=True,
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

    with st.expander("Advanced — mesh & solver", expanded=False):
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
            disabled=analytical_only,
            help=(
                "Hex elements along the wrinkle (x) direction across the "
                "domain length. More elements resolve the curvature but "
                "scale solve time roughly linearly."
            ),
        )
        ny = st.number_input(
            "Mesh divisions in y", 4, 32, 6, 2,
            disabled=analytical_only,
            help=(
                "Hex elements across the laminate width (y). Wrinkle is "
                "uniform in y, so a coarse mesh is usually adequate."
            ),
        )
        nz_per_ply = st.number_input(
            "Mesh divisions per ply (z)", 1, 4, 1,
            disabled=analytical_only,
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

    run_clicked = st.button("Run analysis", type="primary", use_container_width=True)

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

_PLY_TOKEN_RE = re.compile(
    r"""\s*
        (?P<sign>±)?\s*
        (?P<angle>[+-]?\d+(?:\.\d+)?)\s*
        (?:_(?P<rep>\d+))?\s*
    """,
    re.VERBOSE,
)


def _expand_ply_token(token: str) -> List[float]:
    """Expand a single ply entry like ``0``, ``45_2``, ``±45``, or ``±30_2``."""
    token = token.strip()
    if not token:
        return []
    m = _PLY_TOKEN_RE.fullmatch(token)
    if not m:
        raise ValueError(f"Could not parse ply token: {token!r}")
    angle = float(m.group("angle"))
    rep = int(m.group("rep")) if m.group("rep") else 1
    if m.group("sign") == "±":
        return [angle, -angle] * rep
    return [angle] * rep


def _parse_contracted_layup(s: str) -> List[float]:
    """Parse contracted notation like ``[0/45/-45/90]_3s`` or ``[0/±45/90]s``."""
    m = re.fullmatch(
        r"\s*\[\s*(?P<inner>[^\[\]]*)\]\s*_?\s*(?P<rep>\d+)?\s*(?P<sym>[sS])?\s*",
        s,
    )
    if not m:
        raise ValueError(
            f"Could not parse contracted layup {s!r}. "
            "Expected a form like '[0/45/-45/90]_3s'."
        )
    plies: List[float] = []
    for tok in m.group("inner").split("/"):
        plies.extend(_expand_ply_token(tok))
    if not plies:
        raise ValueError("Contracted layup contains no plies.")
    repeat = int(m.group("rep")) if m.group("rep") else 1
    plies = plies * repeat
    if m.group("sym"):
        plies = plies + plies[::-1]
    return plies


def parse_layup(s: str) -> List[float]:
    """Parse a layup string into a flat list of ply angles (degrees).

    Two notations are accepted:

    * **Contracted** — ``[0/45/-45/90]_3s`` (sublaminate in brackets, optional
      repeat count, trailing ``s`` for symmetry). ``±`` and ``_n`` ply-level
      modifiers are also supported, e.g. ``[0/±45/90_2]s``.
    * **Explicit list** — comma-, semicolon-, or newline-separated angles,
      e.g. ``0, 45, -45, 90, ...``.
    """
    s = s.strip()
    if not s:
        raise ValueError("Layup is empty.")
    if "[" in s or "]" in s:
        return _parse_contracted_layup(s)
    out: List[float] = []
    for tok in s.replace(";", ",").replace("\n", ",").split(","):
        out.extend(_expand_ply_token(tok))
    if not out:
        raise ValueError("Layup is empty.")
    return out


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
if run_clicked:
    try:
        layup = parse_layup(layup_str)
    except ValueError as e:
        st.error(f"Could not parse layup: {e}")
        st.stop()

    try:
        # Validate the custom material up front so the cache isn't keyed on
        # an invalid OrthotropicMaterial that will only blow up inside
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
    # Mesh keys only matter for the FE path; omitting them in analytical-only
    # mode means the cache key doesn't churn when the user tweaks nx/ny/nz.
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
