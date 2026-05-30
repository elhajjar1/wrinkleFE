#!/usr/bin/env python3
"""Validate WrinkleFE analytical predictions against the experimental dataset.

This driver encodes the experimental reference tables from
``VALIDATION_DATA.md`` (Datasets A-F, experimental-only) and runs every
*single-wrinkle* case through the real :class:`WrinkleAnalysis` pipeline,
comparing the predicted normalised knockdown ``analytical_knockdown``
(and, for delamination onset, ``analytical_onset_knockdown``) against the
experimentally measured value.

Multi-wrinkle cases (Li 2024 double-wrinkle, Li 2025 D-*/T-*) are listed
but skipped: ``AnalysisConfig`` accepts a multi-wrinkle override but the
analytical multi-wrinkle model is uncalibrated, so they are out of scope
for a like-for-like single-wrinkle validation.

Pass criterion (every dataset): ``|pred - ref| / ref <= 0.20``.

Run::

    python validation/validate_dataset.py

Outputs:
- A per-case table and per-dataset scorecard (N, MAE, PASS/N) to stdout.
- ``validation/dataset_predictions.csv`` with the raw records.

Amplitude / wavelength conventions
----------------------------------
WrinkleFE's amplitude is the *half* peak-to-peak displacement (the
coefficient of the cosine carrier), so the peak fibre angle is
``theta_max = arctan(2*pi*A / lambda)``.  The mapping from each paper's
reported geometry is chosen to reproduce that paper's stated peak fibre
misalignment angle:

- Elhajjar (A/B): the tabulated ``A`` already *is* the WrinkleFE
  half-amplitude (``arctan(2*pi*0.61/6.6) = 30 deg`` matches the paper's
  representative specimen).  Wavelength from the amplitude rule
  ``lambda = max(19.9*A, 8.2)`` (VALIDATION_DATA.md Section 1.4).
- Mukhopadhyay (C): tabulated ``A`` is the WrinkleFE half-amplitude;
  ``lambda = max(22*A, 10)``.
- Wang (D): tabulated ``A`` is the half-amplitude; ``lambda = 24`` mm
  (gauge length) measured directly.
- Li 2024 (E): the *measured* peak-to-peak amplitude ``A1`` (Table 2)
  maps to a WrinkleFE half-amplitude ``A1 / 2`` with ``lambda = L``;
  this reproduces the paper's resin-insert ``alpha_max`` (e.g.
  ``arctan(2*pi*(0.314/2)/11.4) = 4.9 deg`` vs the stated 5 deg).  The
  *nominal* resin amplitude over-states the fibre angle threefold and is
  not used.
- Li 2025 (F): the cosine ``y = (A/2) cos(2 pi x / L)`` makes ``A`` the
  peak-to-peak height, so the WrinkleFE half-amplitude is ``A / 2`` with
  ``lambda = L`` (reproduces ``alpha_max`` to two decimals).
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from typing import Callable, List, Optional

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary

TOL = 0.20  # +/-20 % pass corridor (VALIDATION_DATA.md Section 1)

_LIB = MaterialLibrary()


# ----------------------------------------------------------------------
# Layups (VALIDATION_DATA.md Section 2)
# ----------------------------------------------------------------------
def _sym(half: List[float]) -> List[float]:
    """Mirror a half-stack into a symmetric laminate."""
    return half + list(reversed(half))


# Elhajjar [0/45/90/-45/0/45/-45/0]_s  (16 plies)
ELHAJJAR_LAYUP = _sym([0, 45, 90, -45, 0, 45, -45, 0])
# Mukhopadhyay [+45_2/90_2/-45_2/0_2]_3s  (48 plies, blocked)
MUKHOPADHYAY_LAYUP = _sym([45, 45, 90, 90, -45, -45, 0, 0] * 3)
# Wang [45/0/-45/90/45/0/-45/0/45/0]_s  (20 plies)
WANG_LAYUP = _sym([45, 0, -45, 90, 45, 0, -45, 0, 45, 0])


# ----------------------------------------------------------------------
# Case record
# ----------------------------------------------------------------------
@dataclass
class Case:
    dataset: str
    case_id: str
    loading: str            # "compression" / "tension"
    kd_exp: float
    cfg: AnalysisConfig
    onset: bool = False     # compare analytical_onset_knockdown instead


def _cfg(
    *,
    amplitude: float,
    wavelength: float,
    width: float,
    morphology: str,
    loading: str,
    material: str,
    angles: List[float],
    ply_thickness: float,
    wrinkle_z_position: float = 0.5,
) -> AnalysisConfig:
    return AnalysisConfig(
        amplitude=amplitude,
        wavelength=wavelength,
        width=width,
        morphology=morphology,
        loading=loading,
        material=_LIB.get(material),
        angles=list(angles),
        ply_thickness=ply_thickness,
        wrinkle_z_position=wrinkle_z_position,
        applied_strain=(+0.01 if loading == "tension" else -0.01),
        nx=20,
        ny=6,
        nz_per_ply=3,
        domain_length=max(3.0 * wavelength, 10.0),
        domain_width=10.0,
        analytical_only=True,
    )


# ----------------------------------------------------------------------
# Dataset builders
# ----------------------------------------------------------------------
def _elhajjar_wl(A: float) -> float:
    return max(19.9 * A, 8.2)


def build_cases() -> List[Case]:
    cases: List[Case] = []

    # --- Dataset A: Elhajjar compression (T700/2510, dispersed) -------
    # (D/T, A, KD_exp)
    A_rows = [
        (0.003, 0.0073, 1.02), (0.005, 0.0122, 1.00), (0.008, 0.0194, 0.95),
        (0.010, 0.0243, 0.90), (0.020, 0.0486, 0.80), (0.030, 0.0729, 0.72),
        (0.050, 0.1215, 0.62), (0.080, 0.1944, 0.52), (0.100, 0.2430, 0.47),
        (0.150, 0.3645, 0.40), (0.200, 0.4860, 0.37), (0.250, 0.6075, 0.35),
        (0.300, 0.7290, 0.32),
    ]
    for dt, A, kd in A_rows:
        wl = _elhajjar_wl(A)
        cases.append(Case(
            "A Elhajjar comp", f"E-C{dt:.3f}", "compression", kd,
            _cfg(amplitude=A, wavelength=wl, width=0.75 * wl,
                 morphology="uniform", loading="compression",
                 material="T700_2510", angles=ELHAJJAR_LAYUP,
                 ply_thickness=0.152)))

    # --- Dataset B: Elhajjar tension ----------------------------------
    B_rows = [
        (0.003, 0.0073, 1.00), (0.005, 0.0122, 0.95), (0.010, 0.0243, 0.90),
        (0.050, 0.1215, 0.77), (0.100, 0.2430, 0.65), (0.200, 0.4860, 0.55),
        (0.300, 0.7290, 0.47),
    ]
    for dt, A, kd in B_rows:
        wl = _elhajjar_wl(A)
        cases.append(Case(
            "B Elhajjar tens", f"E-T{dt:.3f}", "tension", kd,
            _cfg(amplitude=A, wavelength=wl, width=0.75 * wl,
                 morphology="uniform", loading="tension",
                 material="T700_2510", angles=ELHAJJAR_LAYUP,
                 ply_thickness=0.152)))

    # --- Dataset C: Mukhopadhyay (IM7/8552, blocked, embedded) --------
    def muk_wl(A: float) -> float:
        return max(22.0 * A, 10.0)

    # Compression (catastrophic)
    for cid, A, kd in [("M-C1", 0.168, 0.82), ("M-C2", 0.372, 0.68),
                       ("M-C3", 0.492, 0.67)]:
        wl = muk_wl(A)
        cases.append(Case(
            "C Mukhop comp", cid, "compression", kd,
            _cfg(amplitude=A, wavelength=wl, width=0.75 * wl,
                 morphology="graded", loading="compression",
                 material="IM7_8552", angles=MUKHOPADHYAY_LAYUP,
                 ply_thickness=0.125)))

    # Tension ultimate (fibre fracture)
    for cid, A, kd in [("M-Tu1", 0.168, 0.94), ("M-Tu2", 0.372, 0.83),
                       ("M-Tu3", 0.492, 0.77)]:
        wl = muk_wl(A)
        cases.append(Case(
            "C Mukhop tens-ult", cid, "tension", kd,
            _cfg(amplitude=A, wavelength=wl, width=0.75 * wl,
                 morphology="graded", loading="tension",
                 material="IM7_8552", angles=MUKHOPADHYAY_LAYUP,
                 ply_thickness=0.125)))

    # Tension onset (delamination first load-drop) -> onset KD
    for cid, A, kd in [("M-To1", 0.372, 0.70), ("M-To2", 0.492, 0.67),
                       ("M-To3", 0.570, 0.51)]:
        wl = muk_wl(A)
        cases.append(Case(
            "C Mukhop tens-onset", cid, "tension", kd,
            _cfg(amplitude=A, wavelength=wl, width=0.75 * wl,
                 morphology="graded", loading="tension",
                 material="IM7_8552", angles=MUKHOPADHYAY_LAYUP,
                 ply_thickness=0.125),
            onset=True))

    # --- Dataset D: Wang (T800/epoxy, concave/convex) -----------------
    # Material alias T800_Epoxy_Wang2021 is undefined; nearest built-in
    # is T800S_M21 (Hexcel T800S / M21 toughened epoxy).
    for cid, A, morph, kd in [("W-1", 0.38, "convex", 0.729),
                              ("W-2", 0.76, "convex", 0.677),
                              ("W-3", 0.38, "concave", 0.635),
                              ("W-4", 0.76, "concave", 0.419)]:
        cases.append(Case(
            f"D Wang {morph}", cid, "compression", kd,
            _cfg(amplitude=A, wavelength=24.0, width=18.0,
                 morphology=morph, loading="compression",
                 material="T800S_M21", angles=WANG_LAYUP,
                 ply_thickness=0.19)))

    # --- Dataset E: Li 2024 single-wrinkle (AC318, UD, graded) --------
    # WrinkleFE half-amplitude = measured A1 (peak-to-peak) / 2; wl = L.
    # (case_id, n_plies, A1_measured, L, KD_exp)
    E_rows = [
        ("6.3-s-1", 15, 0.314, 11.4, 0.907), ("6.3-s-2", 15, 0.332, 5.6, 0.823),
        ("6.3-s-3", 15, 0.328, 3.6, 0.758), ("6.3-s-4", 15, 0.708, 7.4, 0.612),
        ("6.3-s-5", 15, 0.992, 11.0, 0.523), ("4.2-s-4", 10, 0.696, 7.4, 0.545),
        ("4.2-s-5", 10, 0.886, 11.0, 0.506), ("8.4-s-4", 20, 0.702, 7.4, 0.657),
        ("8.4-s-5", 20, 0.997, 11.0, 0.558),
    ]
    for cid, n, A1, L, kd in E_rows:
        amp = A1 / 2.0
        t_ply = 6.3 / 15.0  # 0.42 mm nominal ply (constant across thicknesses)
        cases.append(Case(
            "E Li2024 comp", cid, "compression", kd,
            _cfg(amplitude=amp, wavelength=L, width=L / 2.0,
                 morphology="graded", loading="compression",
                 material="AC318_S6C10", angles=[0] * n,
                 ply_thickness=t_ply)))

    # --- Dataset F: Li 2025 single-wrinkle (AC318, UD [0]_14) ---------
    # Cosine y=(A/2)cos -> WrinkleFE half-amplitude = A/2; wl = L.
    # Position: Middle -> z=0.5 (plies 7-8 of 14); Above -> z=0.75 (10-11).
    t_ply_F = 7.1 / 14.0
    F_rows = [
        ("S-M-1", 1.5, 26.0, 0.5, 0.891), ("S-M-2", 1.5, 12.9, 0.5, 0.629),
        ("S-M-3", 1.5, 8.1, 0.5, 0.472), ("S-M-4", 1.0, 8.6, 0.5, 0.943),
        ("S-M-5", 0.5, 4.3, 0.5, 1.000), ("S-A-2", 1.5, 12.9, 0.75, 0.981),
    ]
    for cid, A, L, zpos, kd in F_rows:
        amp = A / 2.0
        cases.append(Case(
            "F Li2025 comp", cid, "compression", kd,
            _cfg(amplitude=amp, wavelength=L, width=L / 2.0,
                 morphology="graded", loading="compression",
                 material="AC318_S6C10", angles=[0] * 14,
                 ply_thickness=t_ply_F, wrinkle_z_position=zpos)))

    return cases


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------
@dataclass
class Record:
    dataset: str
    case_id: str
    loading: str
    kd_exp: float
    kd_pred: float
    err: float
    passed: bool


def run() -> List[Record]:
    records: List[Record] = []
    for c in build_cases():
        res = WrinkleAnalysis(c.cfg).run(analytical_only=True)
        if c.onset:
            kd_pred = res.analytical_onset_knockdown
            if kd_pred is None:
                kd_pred = float("nan")
        else:
            kd_pred = res.analytical_knockdown
        err = abs(kd_pred - c.kd_exp) / c.kd_exp if c.kd_exp else float("nan")
        records.append(Record(
            c.dataset, c.case_id, c.loading, c.kd_exp, float(kd_pred),
            float(err), bool(err <= TOL)))
    return records


def report(records: List[Record]) -> None:
    # Per-case table
    print(f"\n{'Dataset':<22}{'Case':<11}{'load':<6}"
          f"{'KD_exp':>8}{'KD_pred':>9}{'err%':>8}  P/F")
    print("-" * 72)
    # group preserves insertion order
    groups: dict = {}
    for r in records:
        groups.setdefault(r.dataset, []).append(r)
    for ds, rs in groups.items():
        for r in rs:
            flag = "PASS" if r.passed else "FAIL"
            print(f"{r.dataset:<22}{r.case_id:<11}{r.loading[:4]:<6}"
                  f"{r.kd_exp:>8.3f}{r.kd_pred:>9.3f}{100*r.err:>7.1f}%  {flag}")

    # Per-dataset scorecard
    print(f"\n{'Dataset':<24}{'N':>4}{'MAE%':>8}{'PASS/N':>9}")
    print("-" * 45)
    tot_err = 0.0
    tot_n = 0
    tot_pass = 0
    for ds, rs in groups.items():
        n = len(rs)
        mae = sum(r.err for r in rs) / n * 100.0
        npass = sum(1 for r in rs if r.passed)
        print(f"{ds:<24}{n:>4}{mae:>8.1f}{npass:>6}/{n}")
        tot_err += sum(r.err for r in rs)
        tot_n += n
        tot_pass += npass
    print("-" * 45)
    print(f"{'TOTAL':<24}{tot_n:>4}{tot_err/tot_n*100:>8.1f}"
          f"{tot_pass:>6}/{tot_n}")


def write_csv(records: List[Record], path: str) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "case_id", "loading", "kd_exp",
                    "kd_pred", "rel_err", "passed"])
        for r in records:
            w.writerow([r.dataset, r.case_id, r.loading,
                        f"{r.kd_exp:.4f}", f"{r.kd_pred:.4f}",
                        f"{r.err:.4f}", int(r.passed)])


def main() -> None:
    records = run()
    report(records)
    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(here, "dataset_predictions.csv")
    write_csv(records, csv_path)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
