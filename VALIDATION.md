# WrinkleFE Validation Ledger

This file tracks every external paper assessed for inclusion in the
WrinkleFE knockdown validation database, with the decision and rationale.
The active validation dataset (data points, pass counts, MAE) lives in the
**Validation** section of [README.md](README.md); this file is the
evaluation history behind it.

## Inclusion methodology

A paper qualifies as a knockdown validation point when it reports, for a
**flat coupon containing fiber waviness**:

- a material with tensile/compressive strength allowables (Xt / Xc),
- the stacking sequence and ply thickness,
- the wrinkle geometry (amplitude A, wavelength λ, or peak misalignment θ),
- the loading mode (tension or compression),
- the measured strength (or knockdown) of the wrinkled vs pristine coupon.

Each case is run through `WrinkleAnalysis`; predicted vs measured knockdown
gives the per-point error, aggregated into the README table.

## Reproducible harness

The machine-readable ledger lives at `tests/test_validation/ledger.json`
(per-case analysis recipe, measured knockdown, and a pinned analytical
baseline computed by the current code). One command recomputes every
case and fails on drift from the pinned baselines:

```bash
python scripts/validate.py            # compare; exit 1 on drift
python scripts/validate.py --update   # re-pin after a deliberate model change
```

`tests/test_validation/test_ledger.py` runs the same comparisons in CI.
The pinned values are regression baselines, not claims of experimental
agreement — the measured-vs-predicted error per dataset is printed
alongside. This is the harness whose absence forced the revert of the
graded-decay fix (issue #254, commit `00584b4`); datasets whose raw
case data are not yet in the repository (Elhajjar 2025, Mukhopadhyay
2015, Li 2026) should be added to the ledger as their points land.

## Included datasets

See the table in [README.md](README.md). Current sources:

- Elhajjar (2025), *Scientific Reports* 15:25977 — compression + tension.
- Mukhopadhyay, Jones & Hallett (2015), *Compos. A* 73:132–142
  (compression) and 77:219–228 (tension).
- Li, Y. et al. (2026), *Composites Part A* 205:109719 — S-glass/epoxy,
  compression (material `AC318_S6C10`).

## Evaluated — not included

### Varkonyi, Belnoue, Kratz & Hallett (2020)

*Predicting consolidation-induced wrinkles and their effects on composites
structural performance.* Int. J. Material Forming 13:907–921.
DOI 10.1007/s12289-019-01514-2.

- **Evaluated**: 2026-05-17
- **Decision**: Not added to the knockdown validation database.
- **Rationale**:
  1. *Failure-mechanism mismatch* — strength is governed by skin–stringer
     interface delamination at a stepped ply run-out under tension-induced
     bending, not the fiber kink-band (compression) or fiber/matrix/OOP
     (tension) mechanisms WrinkleFE's analytical knockdown represents.
     Matrix cracking and fiber failure were explicitly not modelled.
  2. *Geometry mismatch* — the specimen is a stepped stringer-foot run-out
     (skin `[−45/0/45/90]₆S`, upper block `[−45/0/45/90]₃S`), not a flat
     coupon with a single embedded wrinkle representable by `AnalysisConfig`.
  3. *No strength allowables* — only elastic constants (Table 3) and
     cohesive/delamination properties (Table 4: σᴵ=60, σᴵᴵ=90 MPa,
     GᵢC=0.26, GᵢᵢC=1.002 N/mm) are reported; no Xt/Xc. Material is
     IM7/8552, already present in the library as `IM7_8552`.
  4. *Figure 3* is experimental Load–Strain curves for the α=90° co-cured
     and co-bonded tensile tests (delamination load-drop and progressive
     stiffness degradation) — not a digitisable (A, λ, KD) data point.
- **Usable cross-reference only**: Table 2 wrinkle geometry (co-cured 90°
  h≈0.42 mm, λ≈5.91 mm, θ₊≈12.7°; co-bonded 90° h≈0.46 mm, λ≈4.07 mm,
  θ₊≈20°) and an apparent joint knockdown 27.3–27.6 / 34 ≈ 0.80. These are
  delamination-controlled and must not enter the kink-band/tension MAE.
- **Follow-up**: the proper flat-coupon embedded-wrinkle data are this
  paper's refs [3]/[4] — Mukhopadhyay, Jones & Hallett (2015) — already an
  active dataset above.

### Sun, Zhou, Zheng & Wang (2025)

*Inversion analysis of constitutive relations of blade spar laminates with
wrinkle defects.* AIP Advances 15, 065118. DOI 10.1063/5.0276297.

- **Evaluated**: 2026-05-17
- **Decision**: Not added to the knockdown validation database.
- **Rationale**:
  1. *No measured strength or knockdown* — the paper is an inverse
     parameter-identification study (BRBP neural network + PSO) that
     back-fits elastic constants (E₁, E₂, G₁₂, ν₂₁) from tensile
     load–displacement curves. Outputs are load–displacement curves and
     inverted moduli focused on *initial-damage onset* (stress relaxation),
     not ultimate failure. No failure-stress vs A/λ data is reported.
  2. *No pristine baseline* — all four groups (A–D) are wrinkled, so no
     defect-free reference exists to normalise a knockdown against.
  3. *Strengths are reference values, not measurements* — Table III
     (XT=2100, XC=−1346, YT=44.46, SL=60.28 MPa, etc.) are taken from
     GB/T 3961-2005, the manufacturer, and wind-farm literature as FE
     inputs; explicitly not measured here, so they cannot anchor a
     "measured" material entry.
  4. Paper validates an ML inversion method (errors <5%, FE-vs-test <8%),
     not a wrinkle knockdown model.
- **Usable cross-reference only**: flat UD `[0]₆` glass/epoxy coupons
  (A12/HRC1, single-ply 0.734 mm, 150×25×5 mm, ASTM D3039) in tension with
  clean wrinkle geometry (A/λ = 0.01 / 0.1 / 0.18 / 0.25; amplitudes from
  0.1 / 1.1 / 1.8 / 2.5 mm copper-wire molds, λ ≈ 10 mm). Geometrically in
  our model's domain, but the strength response is absent. Digitising the
  Fig. 4 / Fig. 14 load–displacement curves against the near-flat Group A
  would yield only a figure-derived, non-pristine apparent KD biased to the
  initial-damage region — below the database data-quality bar.

## Evaluated — deferred (pending model capability)

### Li, Li, Ge & Liang (2025)

*Experimental Investigation on the Effect of Multi-Wrinkle Fiber Defects
on the Compressive Testing of Unidirectional Composites.* Polymer
Composites 46(16):15176–15187. DOI 10.1002/pc.30121. **Distinct** from
Li et al. (2026), *Compos. A* 205:109719 already in the database
(different journal, year, and specimens; same material system).

- **Evaluated**: 2026-05-18
- **Decision**: *Qualifies* methodologically (meets every inclusion
  criterion: material with Xc `AC318_S6C10`, UD `[0]₁₄` ply 0.44 mm,
  wrinkle geometry, compression, measured strength with a pristine
  baseline). **Inclusion deferred** pending a model capability tracked by
  [issue #161](https://github.com/ranipdx-glitch/wrinkleFE/issues/161).
- **Dataset** (pristine plate 335.52 MPa; A = peak-to-peak amplitude;
  θ_max = arctan(2π·(A/2)/λ)):

  | Case | A (mm) | λ (mm) | α_max | σ (MPa) | KD meas | KD analytical | FE max FI |
  |------|--------|--------|-------|---------|---------|---------------|-----------|
  | S-M-1 | 1.5 | 26.0 | 10° | 298.81 | 0.891 | 0.839 | 0.396 |
  | S-M-2 | 1.5 | 12.9 | 20° | 211.10 | 0.629 | 0.782 | 0.767 |
  | S-M-3 | 1.5 |  8.1 | 30° | 158.41 | 0.472 | 0.744 | 1.058 |
  | S-M-4 | 1.0 |  8.6 | 20° | 316.34 | 0.943 | 0.855 | 0.765 |
  | S-M-5 | 0.5 |  4.3 | 20° | 335.63 | 1.000 | 0.923 | 0.777 |
  | S-A-2 | 1.5 | 12.9 | 20° | 329.29 | 0.981 | 0.782 | 0.767 |

  (S-A-2 is S-M-2 geometry at a near-surface interface.)
- **Finding (model-wide structural limitation)**: the diagnostic trio
  S-M-2 / S-M-4 / S-M-5 has an *identical* peak angle (20°) with amplitude
  1.5 / 1.0 / 0.5 mm and a measured strength span of 0.63 → 1.00 (~60%).
  On a proper **strength** basis — LaRC05 max-FI scales ~linearly with
  load (k ≈ 0.82–0.91), load scaled to FI = 1, σ_f = E_eff·ε_f, referenced
  to the near-pristine wrinkle S-M-5 — the FE strength knockdown for the
  trio is 1.000 / 1.000 / 1.000 (the FE does not reach first-ply failure
  for ≤20° wrinkles at all, regardless of amplitude). The
  amplitude-at-constant-angle effect is captured by **neither** the
  analytical Budiansky–Fleck path **nor** the FE LaRC05 strength path —
  both reduce the wrinkle to its peak misalignment angle, and the FE
  strength response is in fact flatter than the analytical one (S-M-2 FE
  strength error ≈ 59%). (Underlying FE LaRC05 max-FI at ε = −0.01:
  0.767 / 0.765 / 0.777, ≤1.5% spread; modulus retention 0.945 for all
  three — both confirm the same flat behaviour.) Capturing the effect
  needs geometrically nonlinear progressive-damage FE (issue #161). S-A-2
  additionally exposes a missing through-thickness wrinkle-position
  parameter (out of scope; excluded on future inclusion).
- **Related**: the `graded` compression through-thickness decay has a
  separate real bug (`decay_floor` inert in compression, decay scale
  hard-wired to amplitude A), tracked by issue #254. A "mirror the
  tension path" fix was prototyped and **reverted** (commit `00584b4`)
  pending a reproducible harness for *all* existing datasets; it
  corrects the angle response but does not address the amplitude gap
  above. The harness now exists (see *Reproducible harness* above, with
  this dataset's cases pinned in `tests/test_validation/ledger.json`,
  and a strict-xfail test in `tests/test_validation/test_ledger.py`
  documenting the compression `decay_floor` inertness), unblocking the
  re-land.

## Identified — pending evaluation

- _None currently._
