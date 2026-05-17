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

## Identified — pending evaluation

- Li, Y., Li, X., Ge, J. & Liang, J. (2025). *Experimental Investigation on
  the Effect of Multi-Wrinkle Fiber Defects on the Compressive Testing of
  Unidirectional Composites.* Polymer Composites (Wiley). **Distinct** from
  Li et al. (2026), *Compos. A* 205:109719 already in the database
  (different journal, year, and specimens). Not yet evaluated.
