#!/usr/bin/env python3
"""Predicted-vs-experimental knockdown validation across Datasets A-F.

Runs every single-wrinkle experimental case from the consolidated
validation database (VALIDATION_DATA, Jun 2026 revision) through the
WrinkleFE pipeline twice:

* **analytical** -- the closed-form knockdown models
  (``AnalysisResults.analytical_knockdown``; for the Mukhopadhyay
  delamination-onset series, ``analytical_onset_knockdown``), and
* **FE** -- the 3-D finite-element solve with the LaRC05 retention
  factor (``AnalysisResults.retention_factors['larc05']``), the same
  pristine-baseline FI ratio used by the project validation ledger.

and plots predicted KD against experimental KD with the +/-20 % pass
corridor (``|pred - exp| / exp <= 0.20``), one figure per loading mode:

* ``validation/fig_predicted_vs_experimental_compression.png``
  (Datasets A, C-compression, D, E single-wrinkle, F single-wrinkle)
* ``validation/fig_predicted_vs_experimental_tension.png``
  (Datasets B, C tension-ultimate, C tension-onset)

Raw records go to ``validation/predicted_vs_experimental.csv`` and a
per-dataset summary table (N, MAE, PASS counts per model) is printed
to stdout.

Multi-wrinkle cases (Dataset E double-wrinkle 6.3-d-*, Dataset F
D-*/T-*) are excluded per VALIDATION_DATA section 6.4. The Mukhopadhyay
onset series has no FE counterpart (delamination onset is an analytical
curved-beam prediction; the linear FE retention factor measures ultimate
first-ply failure).

Usage::

    python validation/plot_predicted_vs_experimental.py            # full run (FE: ~30-45 min)
    python validation/plot_predicted_vs_experimental.py --no-fe    # analytical only (~seconds)
    python validation/plot_predicted_vs_experimental.py --plot-only  # re-plot from existing CSV
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from wrinklefe.analysis import (  # noqa: E402
    AnalysisConfig,
    WrinkleAnalysis,
    estimate_wavelength_from_amplitude,
)
from wrinklefe.core.layup import parse_layup  # noqa: E402
from wrinklefe.core.material import MaterialLibrary  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
CSV_PATH = OUT_DIR / "predicted_vs_experimental.csv"

TOLERANCE = 0.20  # pass criterion |pred - exp| / exp <= 0.20
APPLIED_STRAIN = 0.01  # nominal strain magnitude for the FE solves

# ----------------------------------------------------------------------
# Layups
# ----------------------------------------------------------------------
ELHAJJAR_LAYUP = parse_layup("[0/45/90/-45/0/45/-45/0]_s")          # 16 plies
MUKHOPADHYAY_LAYUP = parse_layup("[45_2/90_2/-45_2/0_2]_3s")        # 48 plies
WANG_LAYUP = parse_layup("[45/0/-45/90/45/0/-45/0/45/0]_s")         # 20 plies


def _elhajjar_case(case: str, loading: str, A: float, kd: float) -> dict:
    """Datasets A & B: T700/2510, 16-ply dispersed layup, wavelength from
    the amplitude fallback rule (K_lambda = 19.9, lambda_min = 8.2)."""
    lam = estimate_wavelength_from_amplitude(A)
    return dict(
        dataset="A" if loading == "compression" else "B",
        case=case, loading=loading, series="ultimate", kd_exp=kd,
        material="T700_2510", angles=ELHAJJAR_LAYUP, t_ply=0.152,
        amplitude=A, wavelength=lam, width=0.75 * lam,
        morphology="uniform", z_pos=0.5, nz_per_ply=3, fe=True,
    )


def _mukhopadhyay_case(case: str, loading: str, series: str,
                       A: float, kd: float) -> dict:
    """Dataset C: IM7/8552, 48-ply blocked layup, lambda = max(22 A, 10),
    embedded wrinkle with per-ply linear decay -> graded morphology.
    nz_per_ply = 1 keeps the 48-ply FE solve tractable (48 z-layers)."""
    lam = max(22.0 * A, 10.0)
    return dict(
        dataset="C", case=case, loading=loading, series=series, kd_exp=kd,
        material="IM7_8552", angles=MUKHOPADHYAY_LAYUP, t_ply=0.125,
        amplitude=A, wavelength=lam, width=0.75 * lam,
        morphology="graded", z_pos=0.5, nz_per_ply=1,
        fe=(series != "onset"),
    )


def _wang_case(case: str, A: float, morph: str, kd: float) -> dict:
    """Dataset D: T800/epoxy (library card T800S_M21 -- the closest
    built-in T800 system; KD is dimensionless so the allowables only
    enter through the FE FI ratio), 20 plies, lambda = 24 mm gauge,
    explicit concave/convex morphology."""
    return dict(
        dataset="D", case=case, loading="compression", series=morph,
        kd_exp=kd, material="T800S_M21", angles=WANG_LAYUP, t_ply=0.19,
        amplitude=A, wavelength=24.0, width=18.0,
        morphology=morph, z_pos=0.5, nz_per_ply=3, fe=True,
    )


def _li2024_case(case: str, n_plies: int, A1: float, L: float,
                 kd: float) -> dict:
    """Dataset E (Li 2024 single wrinkle): UD glass/epoxy, measured crest
    amplitude A1 (half-amplitude A = A1/2 per the dataset convention),
    wavelength = L, width = L/2, graded morphology."""
    return dict(
        dataset="E", case=case, loading="compression", series="ultimate",
        kd_exp=kd, material="AC318_S6C10", angles=[0.0] * n_plies,
        t_ply=0.42, amplitude=A1 / 2.0, wavelength=L, width=L / 2.0,
        morphology="graded", z_pos=0.5, nz_per_ply=3, fe=True,
    )


def _li2025_case(case: str, A_pp: float, L: float, z_pos: float,
                 kd: float) -> dict:
    """Dataset F (Li 2025 single wrinkle): UD glass/epoxy [0]_14,
    half-amplitude A = A_pp/2, wavelength = L, width = L/2, graded.
    z_pos places the wrinkle (Middle = 0.5; Above = plies 10-11 of 14)."""
    return dict(
        dataset="F", case=case, loading="compression", series="ultimate",
        kd_exp=kd, material="AC318_S6C10", angles=[0.0] * 14,
        t_ply=0.44, amplitude=A_pp / 2.0, wavelength=L, width=L / 2.0,
        morphology="graded", z_pos=z_pos, nz_per_ply=3, fe=True,
    )


CASES: list[dict] = [
    # --- Dataset A: Elhajjar (2025) compression, Fig. 5b -------------
    *[_elhajjar_case(c, "compression", A, kd) for c, A, kd in [
        ("E-C01", 0.0073, 1.02), ("E-C02", 0.0122, 1.00),
        ("E-C03", 0.0194, 0.95), ("E-C04", 0.0243, 0.90),
        ("E-C05", 0.0486, 0.80), ("E-C06", 0.0729, 0.72),
        ("E-C07", 0.1215, 0.62), ("E-C08", 0.1944, 0.52),
        ("E-C09", 0.2430, 0.47), ("E-C10", 0.3645, 0.40),
        ("E-C11", 0.4860, 0.37), ("E-C12", 0.6075, 0.35),
        ("E-C13", 0.7290, 0.32),
    ]],
    # --- Dataset B: Elhajjar (2025) tension, Fig. 5b -----------------
    *[_elhajjar_case(c, "tension", A, kd) for c, A, kd in [
        ("E-T01", 0.0073, 1.00), ("E-T02", 0.0122, 0.95),
        ("E-T03", 0.0243, 0.90), ("E-T04", 0.1215, 0.77),
        ("E-T05", 0.2430, 0.65), ("E-T06", 0.4860, 0.55),
        ("E-T07", 0.7290, 0.47),
    ]],
    # --- Dataset C: Mukhopadhyay (2015) -------------------------------
    *[_mukhopadhyay_case(c, "compression", "ultimate", A, kd)
      for c, A, kd in [("M-C1", 0.168, 0.82), ("M-C2", 0.372, 0.68),
                       ("M-C3", 0.492, 0.67)]],
    *[_mukhopadhyay_case(c, "tension", "ultimate", A, kd)
      for c, A, kd in [("M-Tu1", 0.168, 0.94), ("M-Tu2", 0.372, 0.83),
                       ("M-Tu3", 0.492, 0.77)]],
    *[_mukhopadhyay_case(c, "tension", "onset", A, kd)
      for c, A, kd in [("M-To1", 0.372, 0.70), ("M-To2", 0.492, 0.67),
                       ("M-To3", 0.570, 0.51)]],
    # --- Dataset D: Wang (2021) compression ---------------------------
    _wang_case("W-1", 0.38, "convex", 0.729),
    _wang_case("W-2", 0.76, "convex", 0.677),
    _wang_case("W-3", 0.38, "concave", 0.635),
    _wang_case("W-4", 0.76, "concave", 0.419),
    # --- Dataset E: Li (2024) single-wrinkle UD compression -----------
    # (case, n_plies, measured A1 [mm], L [mm], KD_exp = X_Test / 830)
    _li2024_case("6.3-s-1", 15, 0.314, 11.4, 0.907),
    _li2024_case("6.3-s-2", 15, 0.332, 5.6, 0.823),
    _li2024_case("6.3-s-3", 15, 0.328, 3.6, 0.758),
    _li2024_case("6.3-s-4", 15, 0.708, 7.4, 0.612),
    _li2024_case("6.3-s-5", 15, 0.992, 11.0, 0.523),
    _li2024_case("4.2-s-4", 10, 0.696, 7.4, 0.545),
    _li2024_case("4.2-s-5", 10, 0.886, 11.0, 0.506),
    _li2024_case("8.4-s-4", 20, 0.702, 7.4, 0.657),
    _li2024_case("8.4-s-5", 20, 0.997, 11.0, 0.558),
    # --- Dataset F: Li (2025) single-wrinkle UD compression -----------
    # (case, A_pp [mm], L [mm], z position fraction, KD_exp / 335.5)
    _li2025_case("S-M-1", 1.5, 26.0, 0.5, 0.891),
    _li2025_case("S-M-2", 1.5, 12.9, 0.5, 0.629),
    _li2025_case("S-M-3", 1.5, 8.1, 0.5, 0.472),
    _li2025_case("S-M-4", 1.0, 8.6, 0.5, 0.943),
    _li2025_case("S-M-5", 0.5, 4.3, 0.5, 1.000),
    _li2025_case("S-A-2", 1.5, 12.9, 10.0 / 14.0, 0.981),
]


def build_config(case: dict, analytical_only: bool) -> AnalysisConfig:
    lam = case["wavelength"]
    return AnalysisConfig(
        amplitude=case["amplitude"],
        wavelength=lam,
        width=case["width"],
        morphology=case["morphology"],
        loading=case["loading"],
        material=MaterialLibrary().get(case["material"]),
        angles=list(case["angles"]),
        ply_thickness=case["t_ply"],
        wrinkle_z_position=case["z_pos"],
        nx=20, ny=6, nz_per_ply=case["nz_per_ply"],
        domain_length=max(3.0 * lam, 10.0),
        domain_width=10.0,
        applied_strain=(+APPLIED_STRAIN if case["loading"] == "tension"
                        else -APPLIED_STRAIN),
        analytical_only=analytical_only,
    )


def _run_fe_knockdown(case: dict) -> float | None:
    """Full FE solve -> LaRC05-based FE knockdown.

    Multidirectional laminates use the ledger's retention factor
    (max_FI_pristine / max_FI_wrinkled, both linear in load, so the
    ratio is the predicted strength knockdown).

    For pure-UD compression the pristine LaRC05 FI is ~0 (with zero
    initial misalignment the kinking criterion never activates), so the
    FI ratio degenerates to 0. Following the strength basis documented
    in docs/internal/VALIDATION.md for the Li 2025 assessment, the load
    is scaled to max FI = 1 and the predicted wrinkled strength
    sigma_w = (E_eff * |strain|) / max_FI is referenced to the material
    allowable (the FE's own pristine strength), capped at KD = 1.

    Large-amplitude wrinkles can invert hex elements at nz_per_ply = 3
    (the through-thickness displacement gradient exceeds the element
    height, e.g. Wang W-2/W-4 with A = 4 t_ply); retry once with
    nz_per_ply = 1 as the mesh generator's own diagnostic suggests
    before giving up on the case.
    """
    for nz in dict.fromkeys([case["nz_per_ply"], 1]):
        try:
            res = WrinkleAnalysis(
                build_config({**case, "nz_per_ply": nz}, False)
            ).run()
        except Exception as exc:  # MeshValidationError, solver failures
            print(f"    FE failed at nz_per_ply={nz}: "
                  f"{type(exc).__name__}: {str(exc)[:120]}", flush=True)
            continue
        if not res.retention_factors:
            return None
        retention = res.retention_factors.get("larc05")
        baseline = (res.baseline_fi or {}).get("larc05")
        if baseline is None or baseline > 1e-6:
            return retention
        # Pristine FI ~ 0 (UD compression): strength-basis fallback.
        fi = res.failure_indices["larc05"].mean(axis=-1)
        fi = fi[np.isfinite(fi)]
        max_fi = float(fi.max()) if fi.size else 0.0
        if max_fi <= 0.0:
            return 1.0
        material = MaterialLibrary().get(case["material"])
        allowable = (material.Xt if case["loading"] == "tension"
                     else material.Xc)
        sigma_fail = (res.modulus_retention * material.E1
                      * APPLIED_STRAIN) / max_fi
        return min(1.0, sigma_fail / allowable)
    return None


def run_cases(with_fe: bool, done: dict | None = None) -> list[dict]:
    records = []
    done = done or {}
    fieldnames = ["dataset", "case", "loading", "series", "morphology",
                  "amplitude_mm", "wavelength_mm", "kd_exp",
                  "kd_analytical", "kd_fe"]
    # Incremental CSV so an interrupted run can --resume.
    csv_file = CSV_PATH.open("w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    n = len(CASES)
    for i, case in enumerate(CASES, 1):
        key = (case["dataset"], case["case"], case["loading"])
        if key in done:
            rec = done[key]
            records.append(rec)
            writer.writerow(rec)
            csv_file.flush()
            print(f"[{i:2d}/{n}] {case['dataset']} {case['case']:<8} "
                  f"(resumed from CSV)", flush=True)
            continue

        t0 = time.time()
        res_a = WrinkleAnalysis(build_config(case, True)).run()
        if case["series"] == "onset":
            kd_analytical = res_a.analytical_onset_knockdown
        else:
            kd_analytical = res_a.analytical_knockdown

        kd_fe = None
        if with_fe and case["fe"]:
            kd_fe = _run_fe_knockdown(case)

        rec = dict(
            dataset=case["dataset"], case=case["case"],
            loading=case["loading"], series=case["series"],
            morphology=case["morphology"],
            amplitude_mm=round(case["amplitude"], 4),
            wavelength_mm=round(case["wavelength"], 3),
            kd_exp=case["kd_exp"],
            kd_analytical=(None if kd_analytical is None
                           else round(float(kd_analytical), 4)),
            kd_fe=(None if kd_fe is None else round(float(kd_fe), 4)),
        )
        records.append(rec)
        writer.writerow(rec)
        csv_file.flush()
        print(
            f"[{i:2d}/{n}] {case['dataset']} {case['case']:<8} "
            f"{case['loading']:<11} exp={case['kd_exp']:.3f} "
            f"ana={rec['kd_analytical']} fe={rec['kd_fe']} "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )
    csv_file.close()
    print(f"wrote {CSV_PATH}")
    return records


def read_csv() -> list[dict]:
    records = []
    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            row["kd_exp"] = float(row["kd_exp"])
            for k in ("kd_analytical", "kd_fe"):
                row[k] = float(row[k]) if row[k] not in ("", "None") else None
            records.append(row)
    return records


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
DATASET_STYLE = {
    # dataset label -> (colour, marker, legend text)
    "A": ("#1f77b4", "o", "A - Elhajjar 2025 (UNC)"),
    "B": ("#1f77b4", "o", "B - Elhajjar 2025 (UNT)"),
    "C": ("#d62728", "s", "C - Mukhopadhyay 2015"),
    "C-onset": ("#ff9896", "s", "C - Mukhopadhyay 2015 (onset)"),
    "D": ("#2ca02c", "^", "D - Wang 2021 (conc/conv)"),
    "E": ("#9467bd", "D", "E - Li 2024 (UD glass)"),
    "F": ("#ff7f0e", "*", "F - Li 2025 (UD glass)"),
}


def _stats(pairs: list[tuple[float, float]]) -> tuple[float, int]:
    """Return (MAE %, n_pass) for (exp, pred) pairs under the 20 % rule."""
    errs = [abs(p - e) / e for e, p in pairs]
    n_pass = sum(1 for err in errs if err <= TOLERANCE)
    return 100.0 * float(np.mean(errs)), n_pass


def plot_panel(records: list[dict], loading: str, path: Path) -> None:
    rows = [r for r in records if r["loading"] == loading]
    fig, ax = plt.subplots(figsize=(7.5, 7))

    # 1:1 line and the +/-20 % corridor |pred - exp| / exp <= 0.20
    x = np.linspace(0.0, 1.15, 50)
    ax.fill_between(x, 0.8 * x, 1.2 * x, color="0.85", zorder=0,
                    label="±20 % error band")
    ax.plot(x, x, "k--", lw=1.0, zorder=1, label="1:1 (perfect prediction)")
    ax.plot(x, 0.8 * x, "k:", lw=0.7, zorder=1)
    ax.plot(x, 1.2 * x, "k:", lw=0.7, zorder=1)

    seen = set()
    ana_pairs, fe_pairs = [], []
    for r in rows:
        key = r["dataset"]
        if r["series"] == "onset":
            key = "C-onset"
        colour, marker, label = DATASET_STYLE[key]
        ms = 130 if marker == "*" else 60
        if r["kd_analytical"] is not None:
            ana_pairs.append((r["kd_exp"], r["kd_analytical"]))
            ax.scatter(r["kd_exp"], r["kd_analytical"], s=ms, c=colour,
                       marker=marker, edgecolors="k", linewidths=0.6,
                       zorder=3,
                       label=(label if key not in seen else None))
            seen.add(key)
        if r["kd_fe"] is not None:
            fe_pairs.append((r["kd_exp"], r["kd_fe"]))
            ax.scatter(r["kd_exp"], r["kd_fe"], s=ms, facecolors="none",
                       edgecolors=colour, marker=marker, linewidths=1.4,
                       zorder=3)

    # Filled-vs-open proxy legend entries for the two models
    ax.scatter([], [], s=60, c="0.4", marker="o", edgecolors="k",
               label="Analytical model (filled)")
    ax.scatter([], [], s=60, facecolors="none", edgecolors="0.4",
               marker="o", linewidths=1.4, label="FE / LaRC05 (open)")

    ana_mae, ana_pass = _stats(ana_pairs)
    note = (f"Analytical: MAE = {ana_mae:.1f} %, "
            f"{ana_pass}/{len(ana_pairs)} within ±20 %")
    if fe_pairs:
        fe_mae, fe_pass = _stats(fe_pairs)
        note += (f"\nFE: MAE = {fe_mae:.1f} %, "
                 f"{fe_pass}/{len(fe_pairs)} within ±20 %")
    ax.text(0.03, 0.97, note, transform=ax.transAxes, va="top",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="0.7"))

    ax.set_xlabel("Experimental knockdown  KD = σ_wrinkled / σ_pristine")
    ax.set_ylabel("Predicted knockdown")
    ax.set_title(f"WrinkleFE predicted vs experimental knockdown — "
                 f"{loading}")
    ax.set_xlim(0, 1.15)
    ax.set_ylim(0, 1.15)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"wrote {path}")


def print_summary(records: list[dict]) -> None:
    print("\nPer-dataset summary (pass = |pred - exp| / exp <= 20 %):")
    header = (f"{'Dataset':<10} {'Loading':<12} {'N':>3} "
              f"{'MAE ana %':>10} {'PASS ana':>9} "
              f"{'MAE FE %':>9} {'PASS FE':>8}")
    print(header)
    print("-" * len(header))
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        key = (r["dataset"] + ("-onset" if r["series"] == "onset" else ""),
               r["loading"])
        groups.setdefault(key, []).append(r)
    for (ds, loading), rows in sorted(groups.items()):
        ana = [(r["kd_exp"], r["kd_analytical"]) for r in rows
               if r["kd_analytical"] is not None]
        fe = [(r["kd_exp"], r["kd_fe"]) for r in rows
              if r["kd_fe"] is not None]
        ana_mae, ana_pass = _stats(ana)
        fe_str = ("-", "-")
        if fe:
            fe_mae, fe_pass = _stats(fe)
            fe_str = (f"{fe_mae:.1f}", f"{fe_pass}/{len(fe)}")
        print(f"{ds:<10} {loading:<12} {len(rows):>3} "
              f"{ana_mae:>10.1f} {f'{ana_pass}/{len(ana)}':>9} "
              f"{fe_str[0]:>9} {fe_str[1]:>8}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-fe", action="store_true",
                        help="skip the FE solves (analytical only)")
    parser.add_argument("--plot-only", action="store_true",
                        help="re-plot from the existing CSV without re-running")
    parser.add_argument("--resume", action="store_true",
                        help="skip cases already present in the CSV")
    args = parser.parse_args(argv)

    if args.plot_only:
        records = read_csv()
    else:
        done = {}
        if args.resume and CSV_PATH.exists():
            done = {(r["dataset"], r["case"], r["loading"]): r
                    for r in read_csv()}
        records = run_cases(with_fe=not args.no_fe, done=done)

    plot_panel(records, "compression",
               OUT_DIR / "fig_predicted_vs_experimental_compression.png")
    plot_panel(records, "tension",
               OUT_DIR / "fig_predicted_vs_experimental_tension.png")
    print_summary(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
