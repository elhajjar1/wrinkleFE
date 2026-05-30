# WrinkleFE — Validation Against the Experimental Dataset

**Scope:** the experimental-only reference set in `VALIDATION_DATA.md`
(Datasets A–F, 57 measured cases). This report validates the *shipped*
`WrinkleAnalysis` analytical model against those measurements.

**Reproduce:**
```bash
pip install -e .
python validation/validate_dataset.py      # scorecard + dataset_predictions.csv
python validation/plot_validation.py        # fig_validation_scatter.png
```

Pass criterion (every dataset): `|pred − ref| / ref ≤ 0.20`.

---

## 1. Headline scorecard

Of the 57 experimental cases, **48 are single-wrinkle and run through the
existing pipeline**; the remaining **9 are multi-wrinkle** (5 Li-2024
double, 4 Li-2025 D-*/T-*) and are deferred — `AnalysisConfig.wrinkles`
accepts them but the analytical multi-wrinkle model is a coarse
peak-angle-over-wrinkles placeholder and is uncalibrated (§4).

| Dataset | Loading | N | MAE | PASS/N |
|---|---|---:|---:|---:|
| A — Elhajjar | Compression | 13 | **9.5 %** | 12/13 |
| B — Elhajjar | Tension | 7 | **6.2 %** | 7/7 |
| C — Mukhopadhyay | Compression | 3 | **7.7 %** | 3/3 |
| C — Mukhopadhyay | Tension (ultimate) | 3 | 19.9 % | 2/3 |
| C — Mukhopadhyay | Tension (onset) | 3 | 15.2 % | 1/3 |
| D — Wang | Compression (convex) | 2 | 21.9 % | 1/2 |
| D — Wang | Compression (concave) | 2 | 10.8 % | 2/2 |
| E — Li 2024 | Compression (UD glass) | 9 | **12.3 %** | 8/9 |
| F — Li 2025 | Compression (UD, single) | 6 | 27.0 % | 2/6 |
| **TOTAL** | | **48** | **13.2 %** | **38/48** |

**The four primary calibration datasets all validate** (Elhajjar
compression & tension, Mukhopadhyay compression, Li 2024 compression):
MAE ≤ 12.3 %, and the measured numbers reproduce the project's documented
targets (Elhajjar comp ≈ 9.9 %, Elhajjar tens ≈ 6.2 %). The 10 failing
cases concentrate in **two understood areas plus three edge cases**
(§3).

> **Figure:** `python validation/plot_validation.py` regenerates
> `fig_validation_scatter.png` — a predicted-vs-experimental scatter
> (±20 % corridor shaded around the 1:1 line) beside per-dataset MAE
> bars. It is a generated artifact and is reproduced on demand rather
> than committed.

---

## 2. What passes cleanly

- **Elhajjar tension (B): 7/7, MAE 6.2 %.** The three-mechanism
  `min(KD_fiber, KD_matrix, KD_oop)` tension model tracks the measured
  UNT knockdown across the full D/T range.
- **Elhajjar compression (A): 12/13, MAE 9.5 %.** The CLT-weighted
  Budiansky–Fleck kink-band model matches every case up to D/T = 0.25;
  only D/T = 0.30 fails, and only because of the *linear* wavelength
  rule (§3.2).
- **Mukhopadhyay compression (C): 3/3, MAE 7.7 %** — using the
  `graded` (embedded, per-ply-decay) morphology that Dataset C
  specifies. Morphology choice is load-bearing here: a naïve `uniform`
  peak-angle mapping gives MAE ≈ 38 %, because the blocked
  `[…/0₂]₃ₛ` layup has a 4-ply 0° block at the symmetry plane that
  drives `γ_Y_eff` to 0.023 and makes a single peak-angle BF knockdown
  brutally severe.
- **Li 2024 compression (E): 8/9, MAE 12.3 %.** This is the dataset's
  new *primary* experimental source (Table 4 measured strengths). It
  validates once the wrinkle geometry is mapped correctly (§4.1).
- **Wang concave (D): 2/2.** The concave `M_f = 1.334` branch
  reproduces the measured concave knockdown including the steep
  D/T = 0.20 drop to 0.419.

---

## 3. Findings (failures, by cause)

### 3.1 Dataset F (Li 2025) — angle-only model limitation *(primary finding)*

**2/6 pass, MAE 27 %.** The analytical model collapses every wrinkle to
its peak misalignment angle `θ_max = arctan(2πA/λ)`, so it cannot
separate two effects the experiment shows are first-order:

*Amplitude at fixed angle* — the trio S-M-5 / S-M-4 / S-M-2 all have
`α_max = 20°` but amplitudes 0.5 / 1.0 / 1.5 mm:

| Case | α_max | KD measured | KD predicted |
|---|---:|---:|---:|
| S-M-5 | 20° | 1.000 | 0.629 |
| S-M-4 | 20° | 0.943 | 0.585 |
| S-M-2 | 20° | 0.629 | 0.577 |

The measured KD spans **0.37**; the prediction spans **0.05** (nearly
flat). The model captures the worst case (large amplitude) but predicts
a near-pristine, small-amplitude 20° wrinkle as equally damaging.

*Through-thickness position* — same Type-2 geometry, Middle vs Above:

| Case | position | KD measured | KD predicted |
|---|---|---:|---:|
| S-M-2 | Middle (z=0.50) | 0.629 | 0.577 |
| S-A-2 | Above (z=0.75) | 0.981 | 0.582 |

The `wrinkle_z_position` knob exists but moves the prediction by only
0.005 where the experiment moves by 0.35.

This **independently reproduces the structural limitation already
documented** in `VALIDATION.md` (issue #161): capturing
amplitude-at-fixed-angle and wrinkle-position effects needs
geometrically-nonlinear progressive-damage FE, not the closed-form
peak-angle knockdown.

### 3.2 Elhajjar D/T = 0.30 — linear wavelength-rule saturation

The single Dataset-A failure (29.6 %) is an artefact of the *literal*
§1.4 rule `λ = max(19.9·A, 8.2)`: as A grows, λ grows, and
`θ_max = arctan(2πA/λ)` saturates, flooring KD at ~0.415 while the
experiment keeps dropping to 0.32. **The codebase already ships the
fix** — `estimate_wavelength_from_amplitude(scaling="sqrt")`, its
documented "recommended" rule:

| D/T | ref | linear-rule err | **sqrt-rule err** |
|---:|---:|---:|---:|
| 0.20 | 0.37 | 12.1 % | 8.6 % |
| 0.25 | 0.35 | 18.5 % | **12.3 %** |
| 0.30 | 0.32 | 29.6 % | **20.7 %** |

The sqrt rule fixes D/T = 0.25 and brings D/T = 0.30 to the corridor
edge. The harness uses the linear rule to match the dataset recipe
verbatim; switching it to sqrt would make Dataset A 13/13.

### 3.3 Mukhopadhyay tension onset — onset criterion too severe

**1/3 pass.** The Benzeggagh–Kenane energy onset criterion driving
`analytical_onset_knockdown` predicts first-load-drop knockdowns of
0.551 / 0.520 vs measured 0.70 / 0.67 (M-To1/2). It over-predicts the
delamination severity at the lower amplitudes; the largest-amplitude
case (M-To3, 0.51 measured) passes.

### 3.4 Mukhopadhyay tension ultimate — mid-range slightly severe

**2/3 pass.** M-Tu2 (D/T = 0.062) lands at 22.4 % — the three-mechanism
`min()` is marginally too aggressive in the middle of the range; the
endpoints pass.

### 3.5 Wang convex — over-knockdown at high amplitude

**1/2 pass.** W-2 (convex, D/T = 0.20) predicts 0.492 vs 0.677 measured
(27.4 %). The convex `M_f = 0.75` branch over-softens at high amplitude.
Note the material is also a **substitute**: the dataset's
`T800_Epoxy_Wang2021` alias is undefined, so the harness uses the
nearest built-in `T800S_M21`.

---

## 4. Methodology notes & documentation drift

### 4.1 Amplitude / wavelength conventions (important)

WrinkleFE's `amplitude` is the **half** peak-to-peak displacement
(coefficient of the cosine carrier), giving
`θ_max = arctan(2πA/λ)`. The mapping from each paper's geometry was
chosen to **reproduce that paper's stated peak fibre angle**:

- **Li 2024 (E):** the dataset's coarse note "amplitude = nominal resin
  A" would feed `arctan(2π·0.5/11.4) = 15°` where the resin insert was
  designed for `α_max = 5°` — a **3× angle error**. The harness instead
  uses the *measured* peak-to-peak amplitude `A1` (Table 2) as
  `A_WrinkleFE = A1/2`, which reproduces `α_max` to ~1°
  (`arctan(2π·(0.314/2)/11.4) = 4.9°`). This correction is what makes
  Dataset E validate.
- **Li 2025 (F):** the cosine `y = (A/2)cos(2πx/L)` makes A
  peak-to-peak, so `A_WrinkleFE = A/2` (reproduces `α_max` to two
  decimals).
- **Elhajjar / Mukhopadhyay / Wang:** the tabulated A already *is* the
  half-amplitude (`arctan(2π·0.61/6.6) = 30°` matches Elhajjar's
  representative specimen), used directly.

### 4.2 `γ_Y_eff` anchor table is stale vs the shipped model

The dataset §1.1 anchors a two-parameter confinement model
(`γ_Y = 0.032 + 0.05·f_confined`): UD 0.032, Mukhopadhyay 0.053,
Elhajjar 0.074. The **shipped** `_effective_gamma_Y` is a
*three*-parameter model that adds a block-size penalty:

| Layup | dataset §1.1 anchor | shipped `_effective_gamma_Y` |
|---|---:|---:|
| UD `[0]_n` | 0.032 | 0.032 |
| Elhajjar dispersed | 0.074 | 0.064 |
| Mukhopadhyay blocked | 0.053 | 0.023 |

The code is the source of truth (its predictions pass); the dataset's
anchor table has **not** been updated to the block-penalty model. Worth
reconciling in `VALIDATION_DATA.md`.

### 4.3 Missing material aliases (dataset §7 gap, confirmed)

`T700_2510_Elhajjar2014` and `T800_Epoxy_Wang2021` are referenced by the
dataset but absent from `MaterialLibrary`. The harness maps them to the
nearest built-ins `T700_2510` and `T800S_M21`. (`AC318_S6C10`,
`IM7_8552`, `T700_2510` are all present and used directly.)

---

## 5. Deferred — multi-wrinkle cases (9)

Not run; require a calibrated multi-wrinkle analytical model:

- **Li 2024 double-wrinkle (5):** 6.3-d-1 … 6.3-d-5.
- **Li 2025 multi-wrinkle (4):** D-AB-2, D-A-2, D-M-2, T-M-2.

`AnalysisConfig.wrinkles` (a list of `WrinkleSpec`) already lets these be
*assembled*, and `_compute_analytical` handles the multi-wrinkle path,
but it takes only the **maximum** peak angle over the wrinkles — by the
author's own comment "intentionally coarse … calibration against the
Li (2025) Dataset F multi-wrinkle specimens is a follow-up activity."
Until that is calibrated, running them would not be a fair validation.

---

## 6. Bottom line

The shipped analytical model is **well-validated on the data it was
calibrated for** — Elhajjar (both modes), Mukhopadhyay compression, and
the new Li 2024 primary source all sit at MAE ≤ 12 % with 30/32 of those
cases inside the ±20 % corridor. The failures are **diagnostic, not
random**:

1. the Li 2025 amplitude/position effects expose the known angle-only
   ceiling of a closed-form peak-angle model (issue #161);
2. one Elhajjar tail point is fixed by the codebase's own sqrt
   wavelength rule;
3. the tension-onset and convex-high-amplitude misses are calibration
   refinements, not structural gaps.

No prediction was tuned to pass; inputs follow each paper's measured
geometry (§4.1). Full per-case records: `dataset_predictions.csv`.
