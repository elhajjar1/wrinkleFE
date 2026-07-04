# WrinkleFE Validation Ledger

This file tracks every external paper assessed for inclusion in the
WrinkleFE knockdown validation database, with the decision and rationale.
The active validation dataset (data points, pass counts, MAE) lives in the
**Validation** section of [README.md](../../README.md); this file is the
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
graded-decay fix (issue #254, commit `00584b4`). The ledger currently
holds the Li (2025) single-wrinkle compression cases; the
multidirectional case data (Elhajjar 2025, Mukhopadhyay 2015, Li 2026)
are not yet in the repository and should be added to the ledger as their
points land.

## Included datasets

See the table in [README.md](../../README.md). Current sources:

- Elhajjar (2025), *Scientific Reports* 15:25977 — compression + tension.
- Mukhopadhyay, Jones & Hallett (2015), *Compos. A* 73:132–142
  (compression) and 77:219–228 (tension).
- Li, Y. et al. (2026), *Composites Part A* 205:109719 — S-glass/epoxy,
  compression (material `AC318_S6C10`).
- **Dataset E — Li, X. et al. (2024)**, *Composites Science and
  Technology* 256:110762 — UD glass/epoxy single-wrinkle compression,
  9 cases, **moulded** material realization (material card
  `AC318_S6C10`). Normalised ÷830 MPa (indicative — the paper reports
  no measured pristine).
- **Dataset F — Li, Li, Ge & Liang (2025)**, *Polymer Composites*
  46:15176–15187 — UD glass/epoxy single-wrinkle compression, 6 cases
  (incl. the near-surface position case S-A-2), **vacuum-bag**
  realization with a measured pristine of 335.5 MPa (material card
  `AC318_S6C10_vacbag`, measured Xc = 335.5, E1 = 50.8 GPa).
- **Dataset G — Hsiao & Daniel (1996)**, *Composites Science and
  Technology* 56(5):581–593 — UD **carbon**/epoxy (IM6G/3501-6, material
  card `IM6G_3501_6`) periodic waviness under compression, with measured
  defect-free baselines. Two specimens: a *uniform*-waviness coupon
  (θ ≈ 15°, modulus knockdown 0.571) and a *graded*-waviness coupon
  (θ ≈ 7.2°, modulus knockdown 0.941, strength knockdown 0.660). The
  first carbon system and the first dataset carrying a measured
  **stiffness** knockdown — see *Stiffness / modulus validation* below.

### Stiffness / modulus validation

Everything above scores **strength** knockdown. WrinkleFE also predicts a
**stiffness** (axial Young's-modulus) knockdown, exercised by
`validation/validate_modulus.py` against every UD dataset that reports a
measured modulus:

- **FE** — the linear static solve's `modulus_retention` (mean
  fibre-direction stress / applied strain, wrinkled vs pristine).
- **Analytical** — the analytical path's `analytical_modulus_knockdown`
  (UD-scoped): a closed-form CLT series-average of the off-axis lamina
  modulus over the wrinkle profile, the same off-axis-compliance
  integration as Hsiao & Daniel (1996), at zero FE cost. (The driver
  also recomputes this estimate standalone for datasets/configs outside
  the single-`WrinkleAnalysis` path.)

| Dataset | Material | analytical MAE | FE MAE |
|---|---|---|---|
| F — Li (2025) | S-glass/epoxy | 3.9 % | 6.9 % |
| G — Hsiao & Daniel (1996) | carbon/epoxy | 1.2 % | 5.1 % |
| E — Li (2024), indicative | S-glass/epoxy | 7.7 % | 10.3 % |

Both predictors confirm the datasets' headline: stiffness is far more
wrinkle-tolerant than strength — modulus knockdown stays 0.81–0.98 for
the Li S-glass cases and drops only to ≈0.52–0.57 for the carbon
uniform-waviness case at θ = 15° (whose *strength* would fall far
further). The analytical CLT tracks the angle, amplitude/penetration and
through-thickness-position axes; the homogenised-continuum FE is flatter
on the amplitude/position axes, consistent with the strength findings.
The modulus validation is **UD-only** — no multidirectional dataset in
the ledger reports a measured modulus knockdown.

`validation/plot_modulus_validation.py` renders the consolidated
predicted-vs-experimental view (`validation/fig_modulus_validation.png`):
modulus knockdown vs misalignment angle, and a parity plot, for all three
datasets — the stiffness counterpart of `plot_all_validation.py`.

### UD predictor: the two-parameter (θ, D/T, z) penetration gate

The adopted predictor for the UD single-wrinkle datasets (E, F) is the
penetration gate, `wrinklefe.core.penetration_gate`
(`AnalysisConfig.penetration_gate`), documented in
[WRINKLE_MODELING_FINDINGS.md](WRINKLE_MODELING_FINDINGS.md) (items D.3 /
D.5). It exists because angle-only Budiansky–Fleck knockdown is
**scale-invariant**: at a *fixed* peak misalignment angle, the Li grids
show knockdown still varying strongly with through-thickness penetration
(Li 2025 S-M-2/4/5: identical θ ≈ 20°, KD 0.63 → 1.00 as `D/T` falls).
The gate adds that second axis, and a third for wrinkle position:

```
KD = 1 − (1 − KD_angle(θ))·S(D/T)·P(z)
KD_angle(θ) = 1 / (1 + θ_rad / γ_Y)        # Budiansky–Fleck angle floor
S(D/T)      = min(1, (D/T / dt0)^p)         # penetration gate
P(z)        = (2·min(z, 1−z))^position_q    # through-thickness position
```

Two calibrated presets ship, one per material realization:
`GATE_LI2024_MOULDED` (Dataset E, moulded) and `GATE_LI2025_VACBAG`
(Dataset F, vacuum-bag; carries the position exponent for the
near-surface S-A-2 case). The gate is UD-scoped, has zero FE cost, and is
the package's best UD knockdown predictor.

### E vs F: two material realizations, no shared normalization

E and F are the **same prepreg system** (AC318 S-glass / S6C10-800) cured
two different ways: E **moulded** at 2.9 MPa, F **vacuum-bag** at ~1 bar.
The lower consolidation gives F roughly half the compressive strength
(F's measured pristine Xc = 335.5 MPa vs the moulded ÷830 MPa
normalization for E) — a ~2× difference. Because every E wrinkled
strength exceeds F's pristine, the two **cannot share an absolute KD
normalization**: pooling them is meaningless. This is why there are two
material cards (`AC318_S6C10` vs `AC318_S6C10_vacbag`) and two gate
presets, and why the cross-dataset comparison below uses a parity plot
of `(KD_exp, KD_pred)` pairs rather than an absolute-strength axis.

### Consolidated parity chart

`validation/plot_all_validation.py` renders every single-wrinkle case
(Datasets A–F) on one predicted-vs-experimental parity plot with a ±20 %
corridor, writing `validation/fig_all_validation_parity.png`. Each
dataset is predicted by the model that physically applies to it:
multidirectional A–D via the analytical Budiansky–Fleck / three-mechanism
models run through `WrinkleAnalysis`, UD E/F via the penetration gate.
This is the single cross-dataset predicted-vs-experimental view. Running
the script prints the per-dataset scorecard; the UD entries are
**E: 2.8 % MAE, 9/9** and **F: 5.0 % MAE, 6/6** (within ±20 %).

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

## Evaluated — initially deferred, since included (issue #161 closed)

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
  > **Update**: the single-wrinkle cases (S-M-1…5 and the near-surface
  > S-A-2) are now an **included** dataset — Dataset F above — predicted by
  > the two-parameter (θ, D/T, z) penetration gate (`AC318_S6C10_vacbag`),
  > MAE 5.0 %, 6/6. The gate is a *separate fitted predictor*, not the
  > analytical/FE pipeline; the analysis below explains why that BF /
  > FE-strength pipeline could not capture the amplitude-at-constant-angle
  > effect.
  >
  > **Update (issue #161 closed)**: the gate path is now wired into the
  > reproducible harness — `scripts/validate.py` scores it per case
  > against `expected_gate_kd` baselines pinned in
  > `tests/test_validation/ledger.json` (the S-A-2 through-thickness
  > position rides the new per-case `z_frac` field through the gate's
  > `P(z)` factor), and the #161 acceptance criteria are permanent
  > regression tests in `tests/test_validation/test_ledger.py`: the
  > S-M-2/4/5 trio lands at **+2.2 % / −0.6 % / −0.3 %** (band ±15 %),
  > the amplitude and angle orderings are asserted monotonic, and all
  > six cases sit inside the ±20 % parity band. The gate is the issue's
  > candidate direction 3 (a zone-based knockdown layered on the BF
  > core). The multi-wrinkle D-/T- cases remain out of the ledger (the
  > gate is calibrated on single wrinkles; gate × multi-wrinkle wiring
  > is issue #342).
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
  three — both confirm the same flat behaviour.) In the *analytical*
  family the effect is now captured by the penetration gate (see the
  closure note above). On the *FE* side, the crack-band
  progressive-damage solver (`ProgressiveDamageSolver`,
  `crack_band=True`) does develop a genuine amplitude trend at constant
  angle — at Gf = 3.0, nx = 16 the trio predicts 0.805 / 0.912 / 0.957
  (measured 0.629 / 0.943 / 1.000): S-M-4/5 within ~4 %, S-M-2 still
  +28 % (see `validation/validate_li_progressive.py` and the
  `li_progressive_*.csv` runs). **Caveat (measured 2026-07-04)**: the
  crack band does *not* make the predicted strength mesh-objective in
  this setting. Refining to a wavelength-proportional 12 elements per
  wavelength (nx = 36) at the same Gf = 3.0 collapses every S-M case to
  0.32–0.57 (errors −33 % to −62 %, MAE 46 % vs 14 % at nx = 16): the
  finer mesh resolves a steeper local misalignment, so FI-driven
  initiation fires earlier and the h-scaled softening slope does not
  compensate. The Gf calibration is therefore only valid at the mesh
  density it was fitted at (nx = 16, nz_per_ply = 2) — do not "improve"
  the mesh without recalibrating. The FE remains an open research
  direction, no longer tracked by a blocking issue. S-A-2's
  through-thickness position is no longer out of scope: the
  `wrinkle_z_position` parameter and the gate's `P(z)` factor reproduce
  it (pinned via `z_frac` in the ledger).
- **Related (resolved)**: the `graded` compression through-thickness
  decay had a separate real bug, tracked and closed by issue #254:
  `decay_floor` was inert in compression (honored in tension), and the
  decay scale was originally hard-wired to the amplitude A. An early
  "mirror the tension path" fix was **reverted** (commit `00584b4`)
  pending a reproducible harness. Both halves are now fixed: the decay
  scale became the explicit `through_thickness_decay_scale` parameter
  (auto default `max(λ/2, A)`), and `decay_floor` is applied with the
  same floor semantics on both loading paths
  (`Φ = floor + (1 − floor)·gaussian`), with defaults preserving prior
  results bit-for-bit — confirmed by zero drift across this dataset's
  pinned baselines in `tests/test_validation/ledger.json`. Neither
  change addresses the amplitude-at-constant-angle gap above, which
  was closed separately by the penetration-gate path (issue #161).

## Identified — pending evaluation

- _None currently._
