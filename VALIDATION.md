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

## Identified — pending evaluation

- Li, Y., Li, X., Ge, J. & Liang, J. (2025). *Experimental Investigation on
  the Effect of Multi-Wrinkle Fiber Defects on the Compressive Testing of
  Unidirectional Composites.* Polymer Composites (Wiley). **Distinct** from
  Li et al. (2026), *Compos. A* 205:109719 already in the database
  (different journal, year, and specimens). Not yet evaluated.
