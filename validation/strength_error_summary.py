#!/usr/bin/env python3
"""How far each strength prediction can be trusted: signed errors vs test.

Regenerates every number behind the FE-strength caveat the app and the NCR
summary carry (``wrinklefe.io.export.FE_STRENGTH_CAVEAT``) and the tables
in ``docs/interpreting_results.md`` ("How far to trust each number").

Signed, not absolute, errors are the point. A prediction above the
measured knockdown says the wrinkled part keeps more strength than it
does, which is the unsafe direction for a disposition; a mean absolute
error hides which way a model misses. Error = (predicted - measured) /
measured.

Three sections:

1. **Analytical paths** on every single-wrinkle dataset, each predicted by
   the model that physically applies to it (the same cases and models as
   ``plot_all_validation.py``).
2. **FE LaRC05 strength** on the Li (2025) UD cases: wrinkled over
   pristine strength at first failure, ``(E_eff,w / E_eff,p) * (FI_p /
   FI_w)``, capped at 1 (stress at failure ~ E_eff * eps / FI). Possible
   since the kinking fix gave LaRC05 its Xc-calibrated intrinsic
   misalignment: before it, a flat UD pristine coupon could not fail
   (max FI ~1e-10) and the FE strengths had to be normalised to the
   near-pristine *wrinkle* S-M-5 instead.
3. **Progressive damage (crack band)**: the values pinned in the
   validation ledger, not re-run here (minutes per case; the slow test
   lane re-checks them).
4. **Head-to-head matrix** (issue #433): every applicable model scored on
   every strength dataset, on the same cases, so model choice can be
   compared rather than inferred. Per-case predictions, with the wrinkle
   severity, go to ``validation/head_to_head_cases.csv``.

Run::

    python validation/strength_error_summary.py
"""
from __future__ import annotations

import csv
import dataclasses
import functools
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

#: A prediction more than this far above the measurement counts as
#: non-conservative; below it, rounding noise.
_NON_CONSERVATIVE_PCT = 0.5


def _err(pred: float, meas: float) -> float:
    return (pred - meas) / meas * 100.0


def _fe_retention_kd(cfg, **overrides) -> float:
    """First-ply FE LaRC05 strength retention, wrinkled over pristine.

    ``(E_eff,w / E_eff,p) * (FI_p / FI_w)``, capped at 1 (stress at
    failure ~ E_eff * eps / FI). *overrides* replace config fields (the
    mesh, an FE-only morphology) before the solve.
    """
    from wrinklefe.analysis import WrinkleAnalysis

    kw = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
    kw.update(analytical_only=False, verbose=False, **overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = WrinkleAnalysis(type(cfg)(**kw)).run()
    fi = np.asarray(r.failure_indices["larc05"]).mean(axis=-1)
    fi_w = float(fi[np.isfinite(fi)].max())
    fi_p = float(r.baseline_fi["larc05"])
    return min(float(r.modulus_retention_global) * fi_p / fi_w, 1.0)


def analytical_section() -> None:
    import plot_all_validation as pav

    print("1. Analytical paths (model that applies to each dataset)")
    print(f"   {'dataset':24s} {'method':24s} {'n':>3s} {'MAE%':>6s} "
          f"{'min%':>7s} {'max%':>7s} {'non-cons.':>9s}")
    total = outside = 0
    for label, (fn, _c, _m, method) in pav.DATASETS.items():
        errs = [_err(p, m) for m, p in fn()]
        total += len(errs)
        outside += sum(abs(e) > 20.0 for e in errs)
        nonc = sum(e > _NON_CONSERVATIVE_PCT for e in errs)
        print(f"   {label:24s} {method:24s} {len(errs):3d} "
              f"{statistics.mean(abs(e) for e in errs):6.1f} "
              f"{min(errs):+7.1f} {max(errs):+7.1f} {nonc:5d}/{len(errs)}")
    print(f"   -> {total - outside} of {total} cases within +/-20 %\n")


#: The Li (2025) FE mesh (section 2), reused for Li (2024) in section 4.
_LI_FE_MESH = {"nx": 16, "ny": 4, "nz_per_ply": 2}


@functools.cache
def fe_larc05_errors() -> dict[str, tuple[float, float, float]]:
    """``{case: (FE knockdown, measured, error %)}`` for Li (2025) UD.

    Also used by ``tests/test_fe_strength_caveat.py`` to keep the caveat's
    claims true.
    """
    from validate import case_config

    ledger = json.loads(LEDGER.read_text())
    ds = next(d for d in ledger["datasets"] if d["name"].startswith("li_2025"))
    out = {}
    for case in ds["cases"]:
        kd = _fe_retention_kd(case_config(ds, case), **_LI_FE_MESH)
        meas = float(case["measured_kd"])
        out[case["case"]] = (kd, meas, _err(kd, meas))
    return out


def fe_larc05_section() -> None:
    rows = fe_larc05_errors()
    print("2. FE LaRC05 strength, Li (2025) UD glass/epoxy "
          "(wrinkled / pristine)")
    print(f"   {'case':7s} {'FE KD':>7s} {'meas.':>7s} {'error%':>8s}")
    for name, (kd, meas, e) in rows.items():
        print(f"   {name:7s} {kd:7.3f} {meas:7.3f} {e:+8.1f}")
    errs = [e for _, _, e in rows.values()]
    print(f"   -> MAE {statistics.mean(abs(e) for e in errs):.1f} %, "
          f"non-conservative on "
          f"{sum(e > _NON_CONSERVATIVE_PCT for e in errs)} of {len(errs)}\n")


@functools.cache
def fe_larc05_errors_shi() -> dict[str, tuple[float, float, float]]:
    """``{case: (FE knockdown, measured, error %)}`` for Shi (2025), Dataset H.

    Covers BOTH halves (H-UD-* and H-MD-*). The multidirectional half is
    the first measured-strength check of the FE retention path on a
    multidirectional laminate; ``tests/test_fe_strength_caveat.py`` keeps
    the caveat's claim about it true.

    FE recipe: the ledger's analytical recipe is a plain graded profile
    (the analytical path reads only the peak angle); the FE mesh instead
    uses the one-sided geometry the specimens actually have —
    ``morphology='tool_flat'`` with the flat tool face on the bottom and
    a 10-ply ramp (see the ledger's ``geometry_note``; for these
    symmetric layups the flat-bottom crest is the z-mirror of the
    paper's flat-bottom dip, which is a symmetry of the specimen).
    nx = 48 is mesh-checked: at nx = 64 every error moves by < 2.5
    percentage points (H-MD: -5.8/-5.0/-3.0 %).
    """
    from validate import case_config

    ledger = json.loads(LEDGER.read_text())
    out = {}
    for ds in ledger["datasets"]:
        if not ds["name"].startswith("shi_2025"):
            continue
        for case in ds["cases"]:
            kd = _fe_retention_kd(
                case_config(ds, case), nx=48, ny=4, nz_per_ply=1,
                morphology="tool_flat", surface_pocket_side="bottom",
                surface_transition_plies=10,
                enable_surface_resin_pockets=True,
            )
            meas = float(case["measured_kd"])
            out[case["case"]] = (kd, meas, _err(kd, meas))
    return out


def fe_larc05_shi_section() -> None:
    rows = fe_larc05_errors_shi()
    print("2b. FE LaRC05 strength, Shi (2025) CFRP, Dataset H "
          "(wrinkled / pristine; tool_flat recipe, nx=48)")
    print(f"   {'case':8s} {'FE KD':>7s} {'meas.':>7s} {'error%':>8s}")
    for name, (kd, meas, e) in rows.items():
        print(f"   {name:8s} {kd:7.3f} {meas:7.3f} {e:+8.1f}")
    md = [e for n, (_k, _m, e) in rows.items() if n.startswith("H-MD")]
    ud = [e for n, (_k, _m, e) in rows.items() if n.startswith("H-UD")]
    print(f"   -> multidirectional (first measured check): "
          f"{min(md):+.1f} % to {max(md):+.1f} %; UD all conservative "
          f"({min(ud):+.1f} % to {max(ud):+.1f} %)\n")


@functools.cache
def fe_larc05_errors_calvo(nx: int = 200) -> dict[str, tuple[float, float, float]]:
    """``{case: (FE knockdown, measured, error %)}`` for Calvo (2023), Dataset I.

    Tension, multidirectional, seven distributed waves: the ledger recipe
    (multi-wave placements, graded decay with floor 0.5) run through the
    FE. First-ply retention, so it reads the onset of off-axis matrix
    cracking against a fibre-governed measured ultimate: expected to be
    conservative. nx = 200 (1 mm columns, ~13 per bump) is mesh-checked
    against nx = 300.
    """
    from validate import case_config

    ledger = json.loads(LEDGER.read_text())
    ds = next(d for d in ledger["datasets"] if d["name"].startswith("calvo_2023"))
    out = {}
    for case in ds["cases"]:
        kd = _fe_retention_kd(case_config(ds, case), nx=nx, ny=2,
                              nz_per_ply=1)
        meas = float(case["measured_kd"])
        out[case["case"]] = (kd, meas, _err(kd, meas))
    return out


#: FE mesh for Datasets J and K (whole-thickness waves, ``uniform``).
#: Mesh-checked on the first case of each: nx = 72, nz_per_ply = 2 or
#: ny = 4 move the retention by at most 0.03.
_JK_FE_MESH = {"nx": 48, "ny": 2, "nz_per_ply": 1}


@functools.cache
def fe_larc05_errors_ledger(prefix: str) -> dict[str, tuple[float, float, float]]:
    """``{case: (FE knockdown, measured, error %)}`` for one ledger dataset.

    Used for Datasets J (Thor 2021) and K (Pilato 2022), whose ledger
    recipes are whole-thickness waves (``morphology='uniform'``): the FE
    mesh carries the coupon's own waviness, so its retention includes the
    bending that the angle-based model does not see (issue #439).
    """
    from validate import case_config

    ledger = json.loads(LEDGER.read_text())
    ds = next(d for d in ledger["datasets"] if d["name"].startswith(prefix))
    out = {}
    for case in ds["cases"]:
        kd = _fe_retention_kd(case_config(ds, case), **_JK_FE_MESH)
        meas = float(case["measured_kd"])
        out[case["case"]] = (kd, meas, _err(kd, meas))
    return out


def fe_larc05_calvo_section() -> None:
    rows = fe_larc05_errors_calvo()
    print("2c. FE LaRC05 strength, Calvo (2023) CFRP tension, Dataset I "
          "(7 distributed waves, nx=200)")
    print(f"   {'case':8s} {'FE KD':>7s} {'meas.':>7s} {'error%':>8s}")
    for name, (kd, meas, e) in rows.items():
        print(f"   {name:8s} {kd:7.3f} {meas:7.3f} {e:+8.1f}")
    print("   -> first-ply retention vs a fibre-governed tension ultimate "
          "(conservative by construction)\n")


def progressive_section() -> None:
    pd = json.loads(LEDGER.read_text())["progressive_damage"]
    print("3. Progressive damage (crack band), pinned ledger values")
    print(f"   {'recipe':16s} {'case':7s} {'pred.':>7s} {'meas.':>7s} "
          f"{'error%':>8s}")
    for recipe in pd["recipes"]:
        for c in recipe["cases"]:
            pred, meas = c["expected_progressive_kd"], c["measured_kd"]
            print(f"   {recipe['name']:16s} {c['case']:7s} {pred:7.3f} "
                  f"{meas:7.3f} {_err(pred, meas):+8.1f}")


# ----------------------------------------------------------------------
# 4. Head-to-head matrix (issue #433)
# ----------------------------------------------------------------------

#: FE mesh for the A-D head-to-head. Mesh-checked on one case per dataset
#: (A, B, C-comp, D): nx = 72, nz_per_ply = 2 or ny = 4 each move the
#: retention by at most 4 points, and refining makes it LESS conservative.
_MD_FE_MESH = {"nx": 48, "ny": 2, "nz_per_ply": 1}

#: The model columns, in print order.
MODELS = ("analytical", "gate (fitted)", "gate (blind)", "FE LaRC05")

#: Cells left empty on purpose, with the physical reason.
_GATE_MD = "the gate is UD-scoped (no multidirectional form)"
NOT_APPLICABLE = {
    ("A Elhajjar comp", "gate (fitted)"): _GATE_MD,
    ("B Elhajjar tens", "gate (fitted)"): _GATE_MD,
    ("C Mukhopadhyay comp", "gate (fitted)"): _GATE_MD,
    ("C Mukhopadhyay tens", "gate (fitted)"): _GATE_MD,
    ("C Mukhopadhyay onset", "gate (fitted)"): _GATE_MD,
    ("D Wang conc/conv", "gate (fitted)"): _GATE_MD,
    ("H Shi2025 MD comp", "gate (fitted)"): _GATE_MD,
    ("I Calvo2023 MD tens", "gate (fitted)"): _GATE_MD,
    ("J Thor2021 QI comp", "gate (fitted)"): _GATE_MD,
    ("E Li2024 UD comp", "gate (blind)"):
        "in-sample: the moulded preset was fitted to these cases",
    ("F Li2025 UD comp", "gate (blind)"):
        "in-sample: the vacuum-bag preset was fitted to these cases",
    ("H Shi2025 UD comp", "gate (fitted)"):
        "no preset is calibrated for this carbon (see gate (blind))",
    ("J Thor2021 UD comp", "gate (fitted)"):
        "no preset is calibrated for IM7/8552 (see gate (blind))",
    ("K Pilato2022 UD comp", "gate (fitted)"):
        "no preset is calibrated for this material (see gate (blind))",
}


def _theta_deg(A: float, lam: float) -> float:
    return math.degrees(math.atan(2.0 * math.pi * A / lam))


def _li2024_configs() -> list[tuple[str, float, float, float, object]]:
    """Li (2024), Dataset E, as ledger-style configs.

    Dataset E is a (theta, D/T) grid with no ledger entry. The recipe is
    Dataset F's ledger recipe (same prepreg, same [0]_14 x 0.44 mm
    panel) with the moulded card: ``A = (D/T) * T`` is the half-amplitude
    and ``lambda = 2 pi A / tan(theta)``, the conventions the gate was
    calibrated on (``penetration_gate.predict_from_geometry``).
    """
    import copy

    import plot_all_validation as pav
    from validate import case_config

    ledger = json.loads(LEDGER.read_text())
    ds = copy.deepcopy(
        next(d for d in ledger["datasets"] if d["name"].startswith("li_2025"))
    )
    ds["material"] = "AC318_S6C10"
    thickness = 14 * float(ds["ply_thickness_mm"])
    out = []
    for i, (theta, dt, kd) in enumerate(pav.E_GRID, start=1):
        A = dt * thickness
        lam = 2.0 * math.pi * A / math.tan(math.radians(theta))
        case = {"case": f"E-{i}", "amplitude_p2p_mm": 2.0 * A,
                "wavelength_mm": lam}
        out.append((f"E-{i}", theta, dt, kd, case_config(ds, case)))
    return out


def _md_cases() -> dict[str, list[tuple[str, float, float, object]]]:
    """Datasets A-D: ``{label: [(case, theta_deg, KD_exp, config)]}``.

    The configs are exactly the ones the analytical parity series runs
    (``plot_all_validation.case_config``): the FE sees the same inputs.
    """
    import plot_all_validation as pav

    def elhajjar(A, loading):
        lam = pav.elhajjar_wavelength(A)
        return _theta_deg(A, lam), pav.case_config(
            A, lam, material="T700_2510", angles=pav.ELHAJJAR, t_ply=0.152,
            loading=loading)

    def mukho(A, loading):
        lam = pav.mukhopadhyay_wavelength(A)
        return _theta_deg(A, lam), pav.case_config(
            A, lam, material="IM7_8552", angles=pav.MUKHO, t_ply=0.125,
            loading=loading, morphology="graded")

    out: dict[str, list] = {}
    for label, rows, build, loading in (
        ("A Elhajjar comp", pav.A_ROWS, elhajjar, "compression"),
        ("B Elhajjar tens", pav.B_ROWS, elhajjar, "tension"),
        ("C Mukhopadhyay comp", pav.C_COMP_ROWS, mukho, "compression"),
        ("C Mukhopadhyay tens", pav.C_TENS_ROWS, mukho, "tension"),
        ("C Mukhopadhyay onset", pav.C_ONSET_ROWS, mukho, "tension"),
    ):
        for A, kd in rows:
            theta, cfg = build(A, loading)
            out.setdefault(label, []).append((f"A={A}", theta, kd, cfg))
    for A, morph, kd in pav.D_ROWS:
        cfg = pav.case_config(A, 24.0, material="T800S_M21",
                              angles=pav.WANG, t_ply=0.19,
                              loading="compression", morphology=morph)
        out.setdefault("D Wang conc/conv", []).append(
            (f"{morph} A={A}", _theta_deg(A, 24.0), kd, cfg))
    return out


@functools.cache
def head_to_head() -> list[dict]:
    """Every dataset x model on the same cases (issue #433).

    One row per case: ``dataset``, ``case``, ``theta_deg``, ``dt`` (D/T,
    UD only), ``measured`` and one prediction per entry of
    :data:`MODELS` (``None`` where the cell is in
    :data:`NOT_APPLICABLE`).

    * **analytical** -- the angle-based model that applies: kink-band in
      compression, three-mechanism in tension (its onset variant for the
      Mukhopadhyay onset series).
    * **gate (fitted)** -- the shipped preset on the data it was fitted
      to (E, F): a fit quality, not a prediction.
    * **gate (blind)** -- the shipped presets, unchanged, on UD carbon
      neither was fitted to (Shi 2025, Thor 2021, Pilato 2022). Their
      ``gamma_Y`` was fitted to glass, so this is a transfer probe; the
      column holds the moulded-preset value and the CSV also carries the
      vacuum-bag one.
    * **FE LaRC05** -- first-ply retention (:func:`_fe_retention_kd`).
      A-E use the analytical recipe's own inputs; F, H, I reuse sections
      2, 2b and 2c; J and K run their ledger recipes
      (:func:`fe_larc05_errors_ledger`). In tension it is a first-ply quantity scored against
      the measured ultimate (B, C-tens, I), except the onset series.
    """
    import plot_all_validation as pav
    from validate import case_config

    from wrinklefe.analysis import WrinkleAnalysis
    from wrinklefe.core.penetration_gate import (
        GATE_LI2024_MOULDED,
        GATE_LI2025_VACBAG,
        penetration_gate_kd,
    )

    def row(dataset, case, theta, measured, dt=None, **preds):
        r = {"dataset": dataset, "case": case, "theta_deg": theta,
             "dt": dt, "measured": measured}
        r.update({m: None for m in MODELS})
        r.update(preds)
        return r

    rows: list[dict] = []
    for label, cases in _md_cases().items():
        onset = label.endswith("onset")
        for case, theta, kd, cfg in cases:
            res = WrinkleAnalysis(cfg).run(analytical_only=True)
            an = (res.analytical_onset_knockdown if onset
                  else res.analytical_knockdown)
            rows.append(row(label, case, theta, kd, analytical=float(an),
                            **{"FE LaRC05": _fe_retention_kd(
                                cfg, **_MD_FE_MESH)}))

    for case, theta, dt, kd, cfg in _li2024_configs():
        an = WrinkleAnalysis(cfg).run(analytical_only=True)
        rows.append(row(
            "E Li2024 UD comp", case, theta, kd, dt=dt,
            analytical=float(an.analytical_knockdown),
            **{"gate (fitted)": penetration_gate_kd(
                theta, dt, GATE_LI2024_MOULDED, z_position=0.5),
               "FE LaRC05": _fe_retention_kd(cfg, **_LI_FE_MESH)}))

    ledger = json.loads(LEDGER.read_text())

    def ledger_ds(prefix):
        return next(d for d in ledger["datasets"]
                    if d["name"].startswith(prefix))

    ds = ledger_ds("li_2025")
    fe = fe_larc05_errors()
    for (theta, dt, z, _kd), case in zip(pav.F_GRID, ds["cases"],
                                         strict=True):
        an = WrinkleAnalysis(case_config(ds, case)).run(analytical_only=True)
        rows.append(row(
            "F Li2025 UD comp", case["case"], theta,
            float(case["measured_kd"]), dt=dt,
            analytical=float(an.analytical_knockdown),
            **{"gate (fitted)": penetration_gate_kd(
                theta, dt, GATE_LI2025_VACBAG, z_position=z),
               "FE LaRC05": fe[case["case"]][0]}))

    fe = fe_larc05_errors_shi()
    for half, label in (("ud", "H Shi2025 UD comp"),
                        ("md", "H Shi2025 MD comp")):
        ds = ledger_ds(f"shi_2025_{half}")
        n_plies = 20
        for case in ds["cases"]:
            an = WrinkleAnalysis(case_config(ds, case)).run(
                analytical_only=True)
            theta = float(case["peak_angle_deg"])
            dt = (float(case["amplitude_p2p_mm"]) / 2.0
                  / (n_plies * float(ds["ply_thickness_mm"])))
            preds = {"analytical": float(an.analytical_knockdown),
                     "FE LaRC05": fe[case["case"]][0]}
            r = row(label, case["case"], theta, float(case["measured_kd"]),
                    dt=dt if half == "ud" else None, **preds)
            if half == "ud":
                r["gate (blind)"] = penetration_gate_kd(
                    theta, dt, GATE_LI2024_MOULDED)
                r["gate (blind, vacuum-bag preset)"] = penetration_gate_kd(
                    theta, dt, GATE_LI2025_VACBAG)
            rows.append(r)

    ds = ledger_ds("calvo_2023")
    fe = fe_larc05_errors_calvo()
    for case in ds["cases"]:
        an = WrinkleAnalysis(case_config(ds, case)).run(analytical_only=True)
        rows.append(row(
            "I Calvo2023 MD tens", case["case"],
            float(case["peak_angle_deg"]), float(case["measured_kd"]),
            analytical=float(an.analytical_knockdown),
            **{"FE LaRC05": fe[case["case"]][0]}))

    from wrinklefe.core.layup import parse_layup

    for prefix, label in (("thor_2021_qi", "J Thor2021 QI comp"),
                          ("thor_2021_ud", "J Thor2021 UD comp"),
                          ("pilato_2022", "K Pilato2022 UD comp")):
        ds = ledger_ds(prefix)
        fe = fe_larc05_errors_ledger(prefix)
        thickness = (len(parse_layup(ds["layup"]))
                     * float(ds["ply_thickness_mm"]))
        unidirectional = "UD" in label
        for case in ds["cases"]:
            an = WrinkleAnalysis(case_config(ds, case)).run(
                analytical_only=True)
            theta = float(case["peak_angle_deg"])
            dt = float(case["amplitude_p2p_mm"]) / 2.0 / thickness
            r = row(label, case["case"], theta, float(case["measured_kd"]),
                    dt=dt if unidirectional else None,
                    analytical=float(an.analytical_knockdown),
                    **{"FE LaRC05": fe[case["case"]][0]})
            if unidirectional:
                r["gate (blind)"] = penetration_gate_kd(
                    theta, dt, GATE_LI2024_MOULDED)
                r["gate (blind, vacuum-bag preset)"] = penetration_gate_kd(
                    theta, dt, GATE_LI2025_VACBAG)
            rows.append(r)

    return rows


def _write_head_to_head_csv(rows: list[dict], path: Path) -> None:
    fields = ["dataset", "case", "theta_deg", "dt", "measured", *MODELS,
              "gate (blind, vacuum-bag preset)"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else
                            ("" if v is None else v))
                        for k, v in ((f, r.get(f)) for f in fields)})


def head_to_head_section() -> None:
    rows = head_to_head()
    print("4. Head-to-head: every applicable model on every strength dataset")
    print("   (signed error %; worst = most non-conservative case; "
          "nc = non-conservative count)")
    print(f"   {'dataset':22s} {'model':14s} {'n':>3s} {'mean%':>7s} "
          f"{'MAE%':>6s} {'worst%':>7s} {'nc':>5s}")
    datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    for d in datasets:
        drows = [r for r in rows if r["dataset"] == d]
        for m in MODELS:
            if (d, m) in NOT_APPLICABLE:
                continue
            errs = [_err(r[m], r["measured"]) for r in drows
                    if r[m] is not None]
            if not errs:
                continue
            nc = sum(e > _NON_CONSERVATIVE_PCT for e in errs)
            print(f"   {d:22s} {m:14s} {len(errs):3d} "
                  f"{statistics.mean(errs):+7.1f} "
                  f"{statistics.mean(abs(e) for e in errs):6.1f} "
                  f"{max(errs):+7.1f} {nc:2d}/{len(errs)}")
    vac = [_err(r["gate (blind, vacuum-bag preset)"], r["measured"])
           for r in rows if "gate (blind, vacuum-bag preset)" in r]
    print(f"   (gate (blind) is the moulded preset; the vacuum-bag preset on "
          f"the same UD carbon cases: {min(vac):+.1f} % to {max(vac):+.1f} %)")
    print("   Not applicable:")
    for (d, m), why in NOT_APPLICABLE.items():
        if m != "gate (fitted)" or why != _GATE_MD:
            print(f"     {d} / {m}: {why}")
    print(f"     every multidirectional dataset / gate: {_GATE_MD}")
    out = REPO / "validation" / "head_to_head_cases.csv"
    _write_head_to_head_csv(rows, out)
    print(f"   per-case predictions with severity -> "
          f"{out.relative_to(REPO)}\n")


def main() -> None:
    analytical_section()
    fe_larc05_section()
    fe_larc05_shi_section()
    fe_larc05_calvo_section()
    progressive_section()
    print()
    head_to_head_section()


if __name__ == "__main__":
    main()
