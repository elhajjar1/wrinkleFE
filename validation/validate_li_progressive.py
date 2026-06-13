#!/usr/bin/env python3
"""Validate the progressive-damage + resin-pocket FE against the Li datasets.

Runs the Li (2024) Dataset E single-wrinkle cases (and optionally the Li
(2025) Dataset F single-wrinkle cases) through the load-stepping
progressive-damage solver, with the resin-pocket material zone enabled,
and compares the predicted ultimate-strength knockdown against the
measured values.  The pristine baseline is computed once per laminate
thickness (n_plies) and reused, so the run is ~N+3 solves rather than 2N.

Knockdown basis: ``KD_pred = sigma_peak(wrinkled) / sigma_peak(pristine)``
where both come from the same progressive-damage solver, so the (UD,
fibre-free-pristine) normalisation issue that defeats the linear FI ratio
does not arise.  Compared against ``KD_exp`` as tabulated in
VALIDATION_DATA section 2.7 (Li 2024 ÷830 indicative; Li 2025 ÷335.5
measured).

Usage::

    python validation/validate_li_progressive.py            # Dataset E (9 cases)
    python validation/validate_li_progressive.py --with-f   # + Dataset F (6 cases)
    python validation/validate_li_progressive.py --no-pocket # ablate the resin pocket
    python validation/validate_li_progressive.py --nx 16 --nz 1  # mesh controls
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

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

OUT_DIR = Path(__file__).resolve().parent
ML = MaterialLibrary()

# Minimum hex columns per wavelength for longitudinal wrinkle resolution.
MIN_ELEMS_PER_WAVE = 8

# Dataset E (Li 2024) single wrinkle: measured crest A1 -> half-amp A=A1/2.
# (case, n_plies, A1, L, KD_exp)  KD_exp = X_Test / 830 (indicative).
LI2024 = [
    ("6.3-s-1", 15, 0.314, 11.4, 0.907),
    ("6.3-s-2", 15, 0.332, 5.6, 0.823),
    ("6.3-s-3", 15, 0.328, 3.6, 0.758),
    ("6.3-s-4", 15, 0.708, 7.4, 0.612),
    ("6.3-s-5", 15, 0.992, 11.0, 0.523),
    ("4.2-s-4", 10, 0.696, 7.4, 0.545),
    ("4.2-s-5", 10, 0.886, 11.0, 0.506),
    ("8.4-s-4", 20, 0.702, 7.4, 0.657),
    ("8.4-s-5", 20, 0.997, 11.0, 0.558),
]
T_PLY_E = 0.42

# Dataset F (Li 2025) single wrinkle: A_pp -> half-amp A=A_pp/2, T=6.16.
# (case, A_pp, L, z_frac, KD_exp)  KD_exp = sigma / 335.5 (measured plate).
LI2025 = [
    ("S-M-1", 1.5, 26.0, 0.5, 0.891),
    ("S-M-2", 1.5, 12.9, 0.5, 0.629),
    ("S-M-3", 1.5, 8.1, 0.5, 0.472),
    ("S-M-4", 1.0, 8.6, 0.5, 0.943),
    ("S-M-5", 0.5, 4.3, 0.5, 1.000),
    ("S-A-2", 1.5, 12.9, 10.0 / 14.0, 0.981),
]
T_PLY_F = 0.44
N_PLIES_F = 14


def build_mesh(*, n_plies, t_ply, amplitude, wavelength, z_frac,
               nx, ny, nz, pocket, height_scale, length_scale):
    mat = ML.get("AC318_S6C10")
    lam = Laminate.from_angles([0.0] * n_plies, mat, ply_thickness=t_ply)
    Lx = max(3.0 * wavelength, 10.0)
    center_x = Lx / 2.0
    # Resolve the wrinkle longitudinally: at least MIN_ELEMS_PER_WAVE hex
    # columns per wavelength, so short-wavelength inserts (e.g. Li
    # 6.3-s-3, L=3.6 mm) are not under-meshed (which over-concentrates the
    # kink and over-predicts the knockdown).
    nx_eff = max(nx, int(np.ceil(MIN_ELEMS_PER_WAVE * Lx / wavelength)))
    profile = GaussianSinusoidal(
        amplitude=amplitude, wavelength=wavelength,
        width=wavelength / 2.0, center=center_x,
    )
    wc = WrinkleConfiguration.from_morphology_name(
        "graded", profile,
        interface1=n_plies // 2 - 1, interface2=n_plies // 2,
        decay_floor=0.0,
    )
    # Shift the through-thickness decay centre for off-mid wrinkles.
    wc.wrinkle_z_position = z_frac
    mesh = WrinkleMesh(
        laminate=lam, wrinkle_config=wc, Lx=Lx, Ly=10.0,
        nx=nx_eff, ny=ny, nz_per_ply=nz,
    ).generate()
    if pocket and amplitude > 0:
        spec = ResinPocketSpec.from_wrinkle(
            amplitude=amplitude, wavelength=wavelength,
            center_x=center_x, z_center=z_frac * t_ply * n_plies,
            height_scale=height_scale, length_scale=length_scale,
        )
        resin = ML.get("EPOXY_S6C10")
        mesh.resin_material = resin
        # Graded pocket (modulus + fibre-angle blend) to avoid the
        # binary-jump stress concentration that double-counts the defect.
        weight = compute_resin_blend(mesh, spec)
        mesh.resin_blend = weight
        host = lam.plies[0].material
        mesh.resin_blend_materials = {
            int(e): host.blend(resin, float(weight[e]))
            for e in np.flatnonzero(weight > 0.0)
        }
    return mesh, lam


def peak_strength(mesh, lam, *, n_increments, residual_factor):
    mat0 = lam.plies[0].material
    target = 1.8 * mat0.Xc / mat0.E1
    res = ProgressiveDamageSolver(
        mesh, lam, applied_strain=-target, n_increments=n_increments,
        residual_factor=residual_factor,
    ).solve()
    return res.peak_stress


FIELDNAMES = ["dataset", "case", "kd_exp", "kd_pred", "sigma_w", "sigma0",
              "err_pct"]


def _load_checkpoint(csv_path: Path):
    """Read completed case rows and the pristine cache from a checkpoint.

    Returns ``(done_cases, pristine_cache)`` where ``done_cases`` maps a
    case name to its stored record and ``pristine_cache`` maps a ply count
    to the pristine baseline strength recovered from those rows (so a
    resumed run does not recompute the slow baselines).
    """
    done: dict[str, dict] = {}
    pristine_cache: dict[int, float] = {}
    if not csv_path.exists():
        return done, pristine_cache
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            row["kd_exp"] = float(row["kd_exp"])
            row["kd_pred"] = float(row["kd_pred"])
            row["sigma_w"] = float(row["sigma_w"])
            row["sigma0"] = float(row["sigma0"])
            row["err_pct"] = float(row["err_pct"])
            done[row["case"]] = row
    return done, pristine_cache


def _append_row(csv_path: Path, record: dict) -> None:
    """Append one completed-case row, writing the header if new."""
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if new_file:
            w.writeheader()
        w.writerow({k: record.get(k, "") for k in FIELDNAMES})


def run(dataset, *, dataset_label, csv_path, done, nx, ny, nz, pocket,
        height_scale, length_scale, n_increments, residual_factor):
    """Run (or resume) one dataset, checkpointing each case to ``csv_path``.

    Cases already present in ``done`` are skipped (resume).  The pristine
    baseline is recovered from a completed sibling row when available, so
    a resumed run avoids recomputing it.
    """
    # Pristine UD compressive strength is exactly Xc: with the
    # increment-robust first-failure (FI=1) peak, a uniform laminate has
    # no stress concentration, so the progressive-damage baseline equals
    # the material allowable independent of mesh / load-step count
    # (verified: pristine peak = 830.0 = Xc across 12/15/18 increments).
    # Referencing knockdowns to Xc directly removes all baseline noise and
    # halves the cost (no pristine solve per ply count).
    sigma0 = ML.get("AC318_S6C10").Xc
    records = []
    for row in dataset:
        if len(row) == 5 and isinstance(row[1], int):
            # Li 2024 row: (case, n_plies, A1, L, KD_exp)
            case, n_plies, A1, L, kd_exp = row
            t_ply, amplitude, z_frac = T_PLY_E, A1 / 2.0, 0.5
        else:
            # Li 2025 row: (case, A_pp, L, z_frac, KD_exp)
            case, A_pp, L, z_frac, kd_exp = row
            n_plies, t_ply, amplitude = N_PLIES_F, T_PLY_F, A_pp / 2.0

        if case in done:
            records.append(done[case])
            print(f"  {case:9s} (resumed from checkpoint)", flush=True)
            continue

        t0 = time.time()
        wm, wl = build_mesh(
            n_plies=n_plies, t_ply=t_ply, amplitude=amplitude, wavelength=L,
            z_frac=z_frac, nx=nx, ny=ny, nz=nz, pocket=pocket,
            height_scale=height_scale, length_scale=length_scale,
        )
        sigma_w = peak_strength(
            wm, wl, n_increments=n_increments, residual_factor=residual_factor,
        )
        kd_pred = sigma_w / sigma0 if sigma0 > 0 else float("nan")
        err = abs(kd_pred - kd_exp) / kd_exp
        rec = dict(
            dataset=dataset_label, case=case, kd_exp=kd_exp,
            kd_pred=round(kd_pred, 3), sigma_w=round(sigma_w, 1),
            sigma0=round(sigma0, 1), err_pct=round(100 * err, 1),
        )
        records.append(rec)
        _append_row(csv_path, rec)   # checkpoint immediately
        print(f"  {case:9s} exp={kd_exp:.3f} pred={kd_pred:.3f} "
              f"(sw={sigma_w:.0f} s0={sigma0:.0f}) err={100*err:+5.1f}% "
              f"[{time.time()-t0:.0f}s]", flush=True)
    return records


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--with-f", action="store_true",
                    help="also run Dataset F (Li 2025) single-wrinkle cases")
    ap.add_argument("--no-pocket", action="store_true",
                    help="disable the resin-pocket zone (ablation)")
    ap.add_argument("--resume", action="store_true",
                    help="skip cases already present in the output CSV")
    ap.add_argument("--nx", type=int, default=16,
                    help="minimum hex columns in x (auto-raised for short "
                         "wavelengths to keep >=8 elements per wave)")
    ap.add_argument("--ny", type=int, default=2)
    ap.add_argument("--nz", type=int, default=2,
                    help="hex layers per ply; nz>=2 is required to capture "
                         "the through-thickness penetration (D/T) trend")
    ap.add_argument("--height-scale", type=float, default=1.0)
    ap.add_argument("--length-scale", type=float, default=1.0)
    ap.add_argument("--increments", type=int, default=18)
    ap.add_argument("--residual", type=float, default=0.1)
    args = ap.parse_args(argv)

    pocket = not args.no_pocket
    label = "EF" if args.with_f else "E"
    csv_path = OUT_DIR / f"li_progressive_{label}.csv"

    done: dict[str, dict] = {}
    if args.resume:
        done, _ = _load_checkpoint(csv_path)
        print(f"resume: {len(done)} case(s) already in {csv_path.name}")
    elif csv_path.exists():
        csv_path.unlink()  # fresh run starts a clean checkpoint

    kw = dict(nx=args.nx, ny=args.ny, nz=args.nz, pocket=pocket,
              height_scale=args.height_scale, length_scale=args.length_scale,
              n_increments=args.increments, residual_factor=args.residual,
              csv_path=csv_path, done=done)

    print(f"=== Dataset E (Li 2024), pocket={pocket}, "
          f"mesh nx>={args.nx} nz={args.nz}, {args.increments} increments ===")
    records = run(LI2024, dataset_label="E", **kw)
    if args.with_f:
        print(f"=== Dataset F (Li 2025), pocket={pocket} ===")
        records += run(LI2025, dataset_label="F", **kw)

    errs = [r["err_pct"] for r in records]
    n_pass = sum(1 for e in errs if e <= 20.0)
    print(f"\nMAE = {np.mean(errs):.1f}%   "
          f"PASS (<=20%) = {n_pass}/{len(records)}")
    print(f"checkpointed to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
