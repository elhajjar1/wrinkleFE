#!/usr/bin/env python3
"""Visualize the Li E/F progressive-damage FE: mesh + load-displacement.

Builds a representative Dataset E (Li 2024 6.3-s-4) and Dataset F (Li 2025
S-M-2) wrinkled coupon exactly as the validation driver does, renders the
x-z mesh slice (coloured by fibre-direction modulus E1, so the graded
resin pocket is visible, and by local fibre-misalignment angle), runs the
progressive-damage solver capturing the per-increment carried-stress
history, and plots the nominal stress-strain (load-displacement) response
with the first-failure and ultimate points annotated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from scipy.spatial import ConvexHull

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from wrinklefe.core.laminate import Laminate  # noqa: E402
from wrinklefe.core.material import MaterialLibrary  # noqa: E402
from wrinklefe.core.mesh import WrinkleMesh  # noqa: E402
from wrinklefe.core.morphology import WrinkleConfiguration  # noqa: E402
from wrinklefe.core.resin_pocket import (  # noqa: E402
    ResinPocketSpec,
    compute_resin_blend,
)
from wrinklefe.core.wrinkle import GaussianSinusoidal  # noqa: E402
from wrinklefe.solver.progressive_damage import (  # noqa: E402
    ProgressiveDamageSolver,
)

OUT = Path(__file__).resolve().parent
ML = MaterialLibrary()

# (label, material, n_plies, t_ply, A_half, L, z_frac, Xc)
CASES = {
    "E (Li 2024) 6.3-s-4": dict(
        material="AC318_S6C10", n_plies=15, t_ply=0.42,
        amplitude=0.708 / 2.0, wavelength=7.4, z_frac=0.5,
    ),
    "F (Li 2025) S-M-2": dict(
        material="AC318_S6C10_vacbag", n_plies=14, t_ply=0.44,
        amplitude=1.5 / 2.0, wavelength=12.9, z_frac=0.5,
    ),
}


def build(material, n_plies, t_ply, amplitude, wavelength, z_frac,
          nx=16, ny=2, nz=2):
    mat = ML.get(material)
    lam = Laminate.from_angles([0.0] * n_plies, mat, ply_thickness=t_ply)
    Lx = max(3.0 * wavelength, 10.0)
    cx = Lx / 2.0
    profile = GaussianSinusoidal(
        amplitude=amplitude, wavelength=wavelength,
        width=wavelength / 2.0, center=cx,
    )
    wc = WrinkleConfiguration.from_morphology_name(
        "graded", profile,
        interface1=n_plies // 2 - 1, interface2=n_plies // 2,
        decay_floor=0.0,
    )
    wc.wrinkle_z_position = z_frac
    mesh = WrinkleMesh(
        laminate=lam, wrinkle_config=wc, Lx=Lx, Ly=10.0,
        nx=nx, ny=ny, nz_per_ply=nz,
    ).generate()
    resin = ML.get("EPOXY_S6C10")
    mesh.resin_material = resin
    z_lo = float(mesh.nodes[:, 2].min())
    z_hi = float(mesh.nodes[:, 2].max())
    w = compute_resin_blend(
        mesh, ResinPocketSpec.from_wrinkle(
            amplitude=amplitude, wavelength=wavelength,
            center_x=cx, z_center=z_lo + z_frac * (z_hi - z_lo),
        )
    )
    mesh.resin_blend = w
    host = lam.plies[0].material
    mesh.resin_blend_materials = {
        int(e): host.blend(resin, float(w[e]))
        for e in np.flatnonzero(w > 0.0)
    }
    return mesh, lam, mat


def _slice_polys(mesh):
    """x-z polygons (convex hull per element) for the first y-layer."""
    cents = mesh.nodes[mesh.elements].mean(axis=1)
    y0 = cents[:, 1].min()
    band = mesh.domain_size[1] / mesh.ny
    sel = np.flatnonzero(cents[:, 1] < y0 + 0.5 * band)
    polys, elems = [], []
    for e in sel:
        xz = mesh.nodes[mesh.elements[e]][:, [0, 2]]
        try:
            hull = ConvexHull(xz)
            polys.append(xz[hull.vertices])
            elems.append(int(e))
        except Exception:
            continue
    return polys, np.array(elems)


def plot_mesh(mesh, mat, label, path):
    polys, elems = _slice_polys(mesh)
    # Per-element E1 (resin pocket softens it) and fibre angle (deg).
    E1 = np.full(mesh.n_elements, mat.E1)
    for e, m in (mesh.resin_blend_materials or {}).items():
        E1[e] = m.E1
    ang = np.degrees(mesh.element_fiber_angles_array())
    if mesh.resin_blend is not None:
        ang = ang * (1.0 - mesh.resin_blend)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, vals, cmap, title, unit in [
        (axes[0], E1[elems] / 1000.0, "viridis",
         "Fibre-direction modulus E1 (graded resin pocket = blue lens)",
         "E1 (GPa)"),
        (axes[1], ang[elems], "inferno",
         "Local fibre-misalignment angle (drives kink-band failure)",
         "angle (deg)"),
    ]:
        pc = PolyCollection(polys, array=vals, cmap=cmap,
                            edgecolors="k", linewidths=0.15)
        ax.add_collection(pc)
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_ylabel("z (mm)")
        ax.set_title(title, fontsize=9)
        fig.colorbar(pc, ax=ax, label=unit, fraction=0.025, pad=0.01)
    axes[1].set_xlabel("x (mm)")
    fig.suptitle(f"{label}  — wrinkled hex8 mesh (x-z slice, "
                 f"{mesh.n_elements} elements)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}  ({mesh.n_elements} elems)")


def run_history(mesh, lam, mat):
    res = ProgressiveDamageSolver(
        mesh, lam, applied_strain=-1.8 * mat.Xc / mat.E1,
        n_increments=15, residual_factor=0.1,
    ).solve()
    eps = np.array([abs(e) for e, _s in res.history]) * 100.0  # % strain
    sig = np.array([s for _e, s in res.history])
    return eps, sig, res, mat.Xc


def main():
    curves = {}
    for label, kw in CASES.items():
        mesh, lam, mat = build(**kw)
        tag = label.split()[0]
        plot_mesh(mesh, mat, label, OUT / f"li_mesh_{tag}.png")
        eps, sig, res, Xc = run_history(mesh, lam, mat)
        curves[label] = (eps, sig, res, Xc)
        print(f"{label}: peak={res.peak_stress:.1f} MPa  "
              f"KD={res.peak_stress / Xc:.3f}  "
              f"n_failed={res.n_failed_elements}  "
              f"first-fail@inc {res.failed_at_increment}")

    # Load-displacement (nominal stress-strain) curves.
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"E": "#1f77b4", "F": "#ff7f0e"}
    for label, (eps, sig, res, Xc) in curves.items():
        tag = label.split()[0]
        c = colors[tag]
        ax.plot(eps, sig, "-o", color=c, ms=4, label=f"{label}")
        ax.axhline(res.peak_stress, color=c, ls=":", lw=0.8)
        ax.annotate(f"ultimate {res.peak_stress:.0f} MPa "
                    f"(KD {res.peak_stress / Xc:.2f})",
                    xy=(eps[-1], res.peak_stress), fontsize=8, color=c,
                    ha="right", va="bottom")
        ax.axhline(Xc, color=c, ls="--", lw=0.6, alpha=0.5)
    ax.set_xlabel("applied compressive strain |ε| (%)")
    ax.set_ylabel("nominal carried stress  σ = reaction / area  (MPa)")
    ax.set_title("Progressive-damage load-displacement (nominal stress-strain)\n"
                 "dashed = pristine Xc baseline; markers = load increments")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "li_load_displacement.png", dpi=200)
    plt.close(fig)
    print(f"wrote {OUT / 'li_load_displacement.png'}")


if __name__ == "__main__":
    main()
