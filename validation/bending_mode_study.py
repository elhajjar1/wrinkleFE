#!/usr/bin/env python3
"""Issue #439: does a bending-dominated failure mode explain the worst misses?

The largest unsafe misses of the angle-based kink-band model are all high
amplitude-to-thickness wrinkles (Shi multidirectional, Thor quasi-isotropic
wave 1, the most severe Elhajjar case). Thor et al. (2021) attribute their
wave-1 failure to bending from the wave geometry; Shi et al. (2025) report
buckling participation. This study asks two things, without touching the
shipped models:

1. **Transition metrics.** For every compression case, which geometric
   metric separates the cases the analytical model gets badly wrong?
   Candidates: peak angle, amplitude/thickness, amplitude/wavelength, and
   the *load-path eccentricity* e/t measured on the case's own wrinkle
   geometry.
2. **A prototype bending knockdown.** Treat the wrinkle as an eccentric
   column section: the stiffness-weighted centroid of the laminate at the
   wrinkle is offset by e from the far-field load line, so the outer fibres
   carry ``sigma * (1 + 6 e / t_c)`` over a section of thickness t_c. The
   knockdown at which they reach the pristine strength is::

       KD_bend = (t_c / t_0) / (1 + 6 e / t_c)

   and the prototype predicts ``min(KD_kink, KD_bend)``.

The geometry comes from the same mesh the FE solves (generated, not
solved): the recipe each dataset already uses, with Shi's FE recipe
(one-sided ``tool_flat`` trough) for Dataset H.

Run::

    python validation/bending_mode_study.py
"""
from __future__ import annotations

import csv
import dataclasses
import json
import math
import statistics
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "validation"))

import numpy as np  # noqa: E402

LEDGER = REPO / "tests" / "test_validation" / "ledger.json"
OUT_CSV = REPO / "validation" / "bending_mode_cases.csv"

#: Failure modes the source papers report, by dataset label.
REPORTED_MODE = {
    "A Elhajjar comp": "kinking (early kink-band at the severe end)",
    "C Mukhopadhyay comp": "fibre compression below 8-9 deg, delamination above",
    "D Wang conc/conv": "not reported",
    "E Li2024 UD comp": "not reported",
    "F Li2025 UD comp": "not reported",
    "H Shi2025 UD comp": "kinking; buckling participation at t/T = 30%",
    "H Shi2025 MD comp": "kinking; buckling participation at t/T = 30%",
    "J Thor2021 QI comp": "wave 1 bending -> delamination; wave 2 kinking",
    "J Thor2021 UD comp": "wave 1 bending -> delamination; wave 2 kinking",
    "K Pilato2022 UD comp": "delamination (paper's FE), fibre breakage post-test",
}


def _axial_modulus(material, angle_deg: float) -> float:
    """In-plane axial modulus Ex of a ply at ``angle_deg`` (CLT)."""
    c = math.cos(math.radians(angle_deg))
    s = math.sin(math.radians(angle_deg))
    inv = (c**4 / material.E1
           + (1.0 / material.G12 - 2.0 * material.nu12 / material.E1)
           * s**2 * c**2
           + s**4 / material.E2)
    return 1.0 / inv


def section_geometry(cfg) -> dict:
    """Eccentricity and thickness of the wrinkled section, from the mesh.

    Generates (does not solve) the FE mesh of ``cfg``. For each x column
    on the first y row, the ply boundaries give each ply's mid-height and
    thickness; the axial-stiffness-weighted centroid ``z_c`` follows. The
    eccentricity ``e`` is the largest offset of ``z_c`` from its far-field
    value (the first column), and ``t_c`` is the section thickness there.
    """
    from wrinklefe.analysis import WrinkleAnalysis
    from wrinklefe.core.mesh import WrinkleMesh

    kw = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
    # Surface resin pockets (Shi's FE recipe) are a material zone, not
    # geometry, and they are FE-only, so they are switched off here.
    kw.update(nx=96, ny=1, nz_per_ply=1, analytical_only=True, verbose=False,
              enable_surface_resin_pockets=False)
    cfg = type(cfg)(**kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = WrinkleAnalysis(cfg).run(analytical_only=True)
        mesh = WrinkleMesh(
            laminate=res.laminate, wrinkle_config=res.wrinkle_config,
            Lx=cfg.domain_length, Ly=cfg.domain_width,
            nx=cfg.nx, ny=cfg.ny, nz_per_ply=1,
        ).generate()
    nodes = np.asarray(mesh.nodes)
    row = nodes[np.isclose(nodes[:, 1], nodes[:, 1].min())]
    mods = np.array([_axial_modulus(cfg.material, a) for a in cfg.angles])
    xs = np.unique(np.round(row[:, 0], 9))
    zc, thick = [], []
    for x in xs:
        z = np.sort(row[np.isclose(row[:, 0], x), 2])
        mid, t = 0.5 * (z[1:] + z[:-1]), np.diff(z)
        zc.append(float(np.sum(mods * t * mid) / np.sum(mods * t)))
        thick.append(float(z[-1] - z[0]))
    zc, thick = np.asarray(zc), np.asarray(thick)
    offset = np.abs(zc - zc[0])
    i = int(np.argmax(offset))
    return {"e": float(offset[i]), "t_c": float(thick[i]),
            "t_0": float(thick[0])}


def kd_bend(geom: dict) -> float:
    e, t_c, t_0 = geom["e"], geom["t_c"], geom["t_0"]
    return (t_c / t_0) / (1.0 + 6.0 * e / t_c)


def compression_cases() -> list[dict]:
    """Every compression case with its config, measurement and errors."""
    import strength_error_summary as ses
    from validate import case_config

    from wrinklefe.core.layup import parse_layup

    h2h = {(r["dataset"], r["case"]): r for r in ses.head_to_head()}
    out = []

    def add(label, case, cfg, thickness, amplitude, wavelength):
        r = h2h[(label, case)]
        deployed = ("gate (fitted)" if label[0] in "EF" else "analytical")
        out.append({
            "dataset": label, "case": case, "cfg": cfg,
            "theta_deg": float(r["theta_deg"]),
            "a_over_t": amplitude / thickness,
            "a_over_l": amplitude / wavelength,
            "measured": float(r["measured"]),
            "deployed": float(r[deployed]),
            "analytical": float(r["analytical"]),
            "fe": float(r["FE LaRC05"]),
        })

    for label, cases in ses._md_cases().items():
        if "tens" in label or "onset" in label:
            continue
        for case, _theta, _kd, cfg in cases:
            t = len(cfg.angles) * cfg.ply_thickness
            add(label, case, cfg, t, cfg.amplitude, cfg.wavelength)
    for case, _th, _dt, _kd, cfg in ses._li2024_configs():
        add("E Li2024 UD comp", case, cfg,
            len(cfg.angles) * cfg.ply_thickness, cfg.amplitude, cfg.wavelength)

    ledger = json.loads(LEDGER.read_text())
    shi_fe = dict(morphology="tool_flat", surface_pocket_side="bottom",
                  surface_transition_plies=10, enable_surface_resin_pockets=True)
    for prefix, label, overrides in (
        ("li_2025", "F Li2025 UD comp", {}),
        ("shi_2025_ud", "H Shi2025 UD comp", shi_fe),
        ("shi_2025_md", "H Shi2025 MD comp", shi_fe),
        ("thor_2021_qi", "J Thor2021 QI comp", {}),
        ("thor_2021_ud", "J Thor2021 UD comp", {}),
        ("pilato_2022", "K Pilato2022 UD comp", {}),
    ):
        ds = next(d for d in ledger["datasets"] if d["name"].startswith(prefix))
        t = len(parse_layup(ds["layup"])) * float(ds["ply_thickness_mm"])
        for case in ds["cases"]:
            cfg = case_config(ds, case)
            if overrides:
                kw = {f.name: getattr(cfg, f.name)
                      for f in dataclasses.fields(cfg)}
                kw.update(overrides)
                cfg = type(cfg)(**kw)
            add(label, case["case"], cfg, t, cfg.amplitude, cfg.wavelength)
    return out


def _err(pred: float, meas: float) -> float:
    return (pred - meas) / meas * 100.0


def best_threshold(cases, key, unsafe):
    """Threshold on ``key`` that best separates ``unsafe`` cases."""
    vals = sorted({c[key] for c in cases})
    best = None
    for v in vals:
        wrong = sum((c[key] >= v) != unsafe(c) for c in cases)
        if best is None or wrong < best[1]:
            best = (v, wrong)
    return best


def main() -> None:
    cases = compression_cases()
    for c in cases:
        geom = section_geometry(c["cfg"])
        c.update(geom)
        c["e_over_t"] = geom["e"] / geom["t_0"]
        c["kd_bend"] = kd_bend(geom)
        c["prototype"] = min(c["deployed"], c["kd_bend"])

    print("1. Compression cases: geometry metrics and errors (signed %)")
    print(f"   {'dataset':22s} {'case':16s} {'theta':>6s} {'A/t':>6s} "
          f"{'e/t':>6s} {'meas':>6s} {'deploy%':>8s} {'FE%':>7s} "
          f"{'KDbend':>7s} {'proto%':>7s}")
    for c in cases:
        print(f"   {c['dataset']:22s} {c['case']:16s} {c['theta_deg']:6.1f} "
              f"{c['a_over_t']:6.3f} {c['e_over_t']:6.3f} {c['measured']:6.3f} "
              f"{_err(c['deployed'], c['measured']):+8.1f} "
              f"{_err(c['fe'], c['measured']):+7.1f} {c['kd_bend']:7.3f} "
              f"{_err(c['prototype'], c['measured']):+7.1f}")

    def unsafe(c):
        return _err(c["deployed"], c["measured"]) > 15.0

    print("\n2. Which metric separates the deployed model's unsafe misses "
          "(> +15 %)?")
    n_bad = sum(unsafe(c) for c in cases)
    print(f"   {n_bad} of {len(cases)} cases are unsafe by more than 15 %")
    for key, name in (("theta_deg", "peak angle (deg)"),
                      ("a_over_t", "amplitude / thickness"),
                      ("a_over_l", "amplitude / wavelength"),
                      ("e_over_t", "eccentricity / thickness")):
        v, wrong = best_threshold(cases, key, unsafe)
        print(f"   {name:26s} best split at >= {v:.3f}: "
              f"{wrong} of {len(cases)} misclassified")

    print("\n3. Prototype min(deployed, KD_bend) vs the deployed model")
    for name, key in (("deployed", "deployed"), ("prototype", "prototype"),
                      ("FE LaRC05", "fe")):
        errs = [_err(c[key], c["measured"]) for c in cases]
        print(f"   {name:10s} mean {statistics.mean(errs):+6.1f}  "
              f"MAE {statistics.mean(abs(e) for e in errs):5.1f}  "
              f"unsafe {sum(e > 0.5 for e in errs):2d}/{len(errs)}  "
              f"> +15 %: {sum(e > 15 for e in errs)}  worst {max(errs):+6.1f}")

    fields = ["dataset", "case", "theta_deg", "a_over_t", "a_over_l",
              "e_over_t", "e", "t_c", "t_0", "measured", "deployed",
              "analytical", "fe", "kd_bend", "prototype"]
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields + ["reported_mode"],
                           extrasaction="ignore")
        w.writeheader()
        for c in cases:
            row = {k: (f"{c[k]:.4f}" if isinstance(c[k], float) else c[k])
                   for k in fields}
            row["reported_mode"] = REPORTED_MODE.get(c["dataset"], "")
            w.writerow(row)
    print(f"\n   per-case table -> {OUT_CSV.relative_to(REPO)}")


if __name__ == "__main__":
    main()
