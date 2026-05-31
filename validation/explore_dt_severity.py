#!/usr/bin/env python3
"""Experimental: add a D/T (through-thickness penetration) severity gate.

Tests the hypothesis (from the Li 2024/2025 controlled grids) that wrinkle
knockdown needs a SECOND axis beyond the peak misalignment angle: the
through-thickness penetration ratio ``D/T = A / T``.  The angle-only
graded model is correct on the angle axis but blind to absolute size at
fixed angle (Li 2025 types 2/4/5: identical angle, KD 1.00 -> 0.63).

Model
-----
Gate the graded knockdown by how much of the section the wrinkle
penetrates::

    KD_dt = 1 - (1 - KD_graded) * S(D/T)
    S(D/T) = min(1, (D/T / dt0) ** p)

A shallow wrinkle (D/T -> 0) realises none of its angle-driven knockdown
(S -> 0, KD -> 1); a deep one (D/T >= dt0) realises all of it (S -> 1).
``dt0`` and ``p`` are fit to the Li 2025 constant-angle triple
(S-M-2/4/5, all alpha=20 deg) -- the cleanest pure-D/T signal, on a
consistent measured-pristine normalisation.

Scope
-----
The gate is applied ONLY to UD layups.  In multidirectional / blocked
laminates (Mukhopadhyay) the low-D/T knockdown comes from a different
mechanism (delamination of the 0-blocks), which is present even for
shallow wrinkles, so a UD-kinking size gate must not touch it.  This is
the key transferability caveat the run demonstrates.

Findings (calibrated on F only; E held out)
-------------------------------------------
* **F (Li 2025): 2/6 -> 4/6, MAE 27% -> 14%.** The gate fixes exactly the
  two pure-D/T cases S-M-4 and S-M-5 (constant 20 deg angle, varying
  amplitude) -- the cases F was designed to expose. Residual fails are
  S-M-1 (angle axis) and S-A-2 (through-thickness position) -- separate
  axes, untouched.
* **Mukhopadhyay: unchanged 3/3** -- UD-scope correctly excludes the
  blocked carbon laminate (low-D/T knockdown there is delamination, a
  different mechanism).
* **A / B / D: structurally unchanged** -- they use the uniform /
  convex / concave paths, not graded.
* **E (Li 2024): breaks (8/9 -> 1/9) -- a DATA issue, not a model one.**
  E shows large knockdowns at low D/T (6.3-s-5: D/T=0.079, KD=0.523) that
  directly contradict F (S-M-5: D/T=0.035, KD=1.000) for the *same*
  material and path. Root cause: E has no measured pristine and reports
  KD = sigma / 830 (datasheet Xc, flagged "indicative"), while E's
  wrinkled strengths (420-753 MPa) *exceed* F's measured pristine
  (335.5 MPa) -- E and F are different material realisations (different
  cure/Vf) on incompatible absolute KD scales. A gate calibrated on F's
  measured-pristine data therefore cannot also fit E; E cannot
  co-calibrate the D/T axis.

Conclusion: D/T is a real, independent severity axis; gating the graded
UD knockdown by it recovers the constant-angle Li 2025 cases without
disturbing the carbon/multidirectional or non-graded datasets. It is a
UD-scoped term, not a universal replacement, and the residual misfires
(S-M-1, and short-wavelength low-D/T cases) confirm the full model is a
2-D KD(theta, D/T) surface -- the closed-form analogue of the CZM
strength-vs-toughness competition.

Run::

    python validation/explore_dt_severity.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary

TOL = 0.20
_LIB = MaterialLibrary()

# Gate parameters, fit to Li 2025 S-M-2 (D/T=0.106 -> S=1) and
# S-M-4 (D/T=0.070 -> S=0.137):  (0.070/0.106)**p = 0.137 -> p = 4.78.
DT0 = 0.106
DT_P = 4.78


def _sym(h: List[float]) -> List[float]:
    return h + list(reversed(h))


MUK = _sym([45, 45, 90, 90, -45, -45, 0, 0] * 3)


def dt_gate(dt: float) -> float:
    """Through-thickness penetration gate S(D/T) in [0, 1]."""
    return min(1.0, (dt / DT0) ** DT_P)


@dataclass
class GCase:
    ds: str
    cid: str
    amp: float          # WrinkleFE half-amplitude (mm)
    lam: float
    t_ply: float
    angles: List[float]
    mat: str
    kd_exp: float
    zpos: float = 0.5

    @property
    def T(self) -> float:
        return self.t_ply * len(self.angles)

    @property
    def dt(self) -> float:
        return self.amp / self.T

    @property
    def is_ud(self) -> bool:
        return all(abs(a) < 5.0 for a in self.angles)


def graded_cases() -> List[GCase]:
    cs: List[GCase] = []
    # C Mukhopadhyay (graded, multidirectional carbon) -- A is half-amp
    for cid, A, kd in [("M-C1", 0.168, 0.82), ("M-C2", 0.372, 0.68),
                       ("M-C3", 0.492, 0.67)]:
        cs.append(GCase("C Mukhop comp", cid, A, max(22 * A, 10.0),
                        0.125, MUK, "IM7_8552", kd))
    # E Li 2024 (graded UD glass) -- half-amp = measured A1 / 2, lam = L
    E = [("6.3-s-1", 15, 0.314, 11.4, 0.907), ("6.3-s-2", 15, 0.332, 5.6, 0.823),
         ("6.3-s-3", 15, 0.328, 3.6, 0.758), ("6.3-s-4", 15, 0.708, 7.4, 0.612),
         ("6.3-s-5", 15, 0.992, 11.0, 0.523), ("4.2-s-4", 10, 0.696, 7.4, 0.545),
         ("4.2-s-5", 10, 0.886, 11.0, 0.506), ("8.4-s-4", 20, 0.702, 7.4, 0.657),
         ("8.4-s-5", 20, 0.997, 11.0, 0.558)]
    for cid, n, A1, L, kd in E:
        cs.append(GCase("E Li2024 comp", cid, A1 / 2.0, L, 6.3 / 15.0,
                        [0] * n, "AC318_S6C10", kd))
    # F Li 2025 (graded UD glass) -- half-amp = A_pp / 2, lam = L
    F = [("S-M-1", 1.5, 26.0, 0.5, 0.891), ("S-M-2", 1.5, 12.9, 0.5, 0.629),
         ("S-M-3", 1.5, 8.1, 0.5, 0.472), ("S-M-4", 1.0, 8.6, 0.5, 0.943),
         ("S-M-5", 0.5, 4.3, 0.5, 1.000), ("S-A-2", 1.5, 12.9, 0.75, 0.981)]
    for cid, App, L, z, kd in F:
        cs.append(GCase("F Li2025 comp", cid, App / 2.0, L, 7.1 / 14.0,
                        [0] * 14, "AC318_S6C10", kd, zpos=z))
    return cs


def base_kd(c: GCase) -> float:
    cfg = AnalysisConfig(
        amplitude=c.amp, wavelength=c.lam, width=c.lam / 2.0,
        morphology="graded", loading="compression",
        material=_LIB.get(c.mat), angles=list(c.angles),
        ply_thickness=c.t_ply, wrinkle_z_position=c.zpos,
        applied_strain=-0.01, nx=20, ny=6, nz_per_ply=3,
        domain_length=max(3 * c.lam, 10.0), domain_width=10.0,
        analytical_only=True)
    return WrinkleAnalysis(cfg).run().analytical_knockdown


def main() -> None:
    cases = graded_cases()
    print(f"{'dataset':<16}{'case':<9}{'D/T':>6}{'UD':>4}{'S(D/T)':>7}"
          f"{'KDexp':>7}{'base':>7}{'+D/T':>7}{'e0%':>6}{'e1%':>6}")
    print("-" * 75)
    agg = {}
    for c in cases:
        b = base_kd(c)
        if c.is_ud:
            s = dt_gate(c.dt)
            g = 1.0 - (1.0 - b) * s
        else:
            s = float("nan")
            g = b  # gate not applied to multidirectional layups
        e0 = abs(b - c.kd_exp) / c.kd_exp
        e1 = abs(g - c.kd_exp) / c.kd_exp
        sflag = "-" if c.is_ud is False else f"{s:.2f}"
        print(f"{c.ds:<16}{c.cid:<9}{c.dt:>6.3f}{('Y' if c.is_ud else 'n'):>4}"
              f"{sflag:>7}{c.kd_exp:>7.3f}{b:>7.3f}{g:>7.3f}"
              f"{100*e0:>6.1f}{100*e1:>6.1f}")
        a = agg.setdefault(c.ds, [0, 0.0, 0.0, 0, 0])
        a[0] += 1
        a[1] += e0
        a[2] += e1
        a[3] += int(e0 <= TOL)
        a[4] += int(e1 <= TOL)
    print("\n" + f"{'dataset':<16}{'N':>3}{'MAE0%':>7}{'MAE1%':>7}"
          f"{'pass0':>7}{'pass1':>7}")
    print("-" * 47)
    for ds, (n, s0, s1, p0, p1) in agg.items():
        print(f"{ds:<16}{n:>3}{100*s0/n:>7.1f}{100*s1/n:>7.1f}"
              f"{p0:>5}/{n}{p1:>5}/{n}")
    print("\nbase = angle-only graded (current model);  +D/T = with the "
          "UD-scoped penetration gate.")
    print("A/B (Elhajjar, uniform) and D (Wang, convex/concave) use "
          "non-graded paths -> structurally unchanged.")


if __name__ == "__main__":
    main()
