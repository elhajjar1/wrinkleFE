# Wrinkle compressive-knockdown modelling: findings (items D.1–D.5)

This note records the outcome of the five-item modelling programme (list
"D") aimed at closing the residual gaps in WrinkleFE's prediction of
compressive strength knockdown for unidirectional (UD) laminates with
fibre waviness, validated against the Li (2024 Dataset E / 2025 Dataset
F) controlled `(θ, D/T)` grids.

## Starting point

The FE progressive-damage + graded-resin-pocket path (crest-located)
reached **combined E+F MAE 17.4 %, 8/15 within ±20 %**.  The residual
clustered into two symptoms:

- **Short-wavelength / sharp wrinkles over-knock** (E 6.3-s-2/-3).
- **Deep, large-amplitude wrinkles under-knock** (F S-M-2/-3); the model
  could not span the measured KD range (≈1.0 for tiny wrinkles to ≈0.47
  for steep deep ones).

## D.1 — Crack-band regularization (Bažant–Oh) — **adopted**

Energy-regularized linear softening of the fibre-compression mode, driven
by the combined MaxStress+LaRC05 failure index, with the softening end
point scaled by the element size so the dissipated energy per crack area
equals the fibre-kink fracture energy `Gf` regardless of mesh.

- **Mesh sensitivity ~halved** (E 6.3-s-4 spread ~15 % → ~7 % over
  nx = 12/16/24); calibration reduced to one physical parameter (`Gf`).
- **Fixed the short-wavelength over-knock**: E 6.3-s-2 +26 %→+1 %,
  6.3-s-3 +31 %→+0.5 %.
- **E: MAE 14.4 %→10.3 %, 6/9→8/9** (Gf = 12 N/mm) — now beats the
  analytical graded model (12.3 %) via full FE.
- **F: 21.8 %→20.1 %, 2/6→3/6** (Gf = 3 N/mm).  Crack-band raises KD
  (energy-regularized softening carries more load), which *helps* the
  over-knock cases but *worsens* F's deep under-knock cases — a softening
  regularizer cannot fix a damage-*initiation* deficit.

## D.3 — Two-parameter (θ, D/T) penetration gate — **adopted (best predictor)**

`KD = 1 − (1 − KD_angle(θ))·min(1, (D/T / dt0)^p)`, the VALIDATION_DATA
§2.7 penetration-gate form.  Captures both axes of the Li grids — the
angle response *and* the steep penetration drop (KD 1.00→0.94→0.63 over
D/T 0.041→0.122) — that defeated the angle-only and FE models.

- **E: MAE 2.8 %, 9/9** (`GATE_LI2024_MOULDED`: γ_Y = 0.2577,
  dt0 = 0.0938, p = 0.59).
- **F: MAE 6.0 %, 5/5** on the five mid-plane cases used to fit the gate
  (`GATE_LI2025_VACBAG`: γ_Y = 0.6215, dt0 = 0.122, p = 4.31; matches
  §2.7's predicted dt0 ≈ 0.12, p ≈ 4.8). Adding the near-surface S-A-2
  case via the D.5 position factor gives the full six-case F result
  **5.0 %, 6/6** (the value reported in the consolidated chart and the
  "Net picture" table below).
- Material-realization specific (E moulded vs F vacuum-bag give
  contradictory KD at the same (θ, D/T) — the §2.7 normalization issue);
  UD-scoped.
- Shipped as `wrinklefe.core.penetration_gate` and wired into the
  analytical pipeline (`AnalysisConfig.penetration_gate`).

This is the **best knockdown predictor in the package** for UD wrinkles,
at zero FE cost.

## D.4 — Continuum geometric nonlinearity — **negative finding (not adopted)**

Added the geometric stiffness `K_geo` to the hex8 element (verified
against Euler column theory, σ_cr ∝ 1/L²) and a linearized-buckling
solver.  The microbuckling knockdown it produces is badly wrong (Li F
0.30/0.06/0.38 vs measured 0.89/0.63/0.47), for two physical reasons:

1. **Imperfection sensitivity (Koiter)** — the bifurcation load of the
   wrinkled structure sits far below its limit load, so the linear
   eigenvalue over-predicts the knockdown.
2. **Wrong scale** — buckling of the homogenised ply-mesh is local
   *structural* buckling of the soft wrinkle region, not the sub-ply
   *fibre kinking* that governs (fibre buckling on the matrix foundation).

`K_geo` remains correct, tested, reusable infrastructure for genuine
structural-buckling analyses, but continuum geometric nonlinearity does
not help the wrinkle-knockdown problem.

## D.2 — Pimenta–Pinho kink micromechanics — **assessed, cannot anchor the gate**

The plan was to replace the gate's *fitted* γ_Y with a first-principles
kink-band value.  Tested on F's constant-penetration triple
(`validation/d2_kink_micromechanics.py`): the Argon / Budiansky–Fleck
form with the **physical** matrix-shear-yield strain
(γ_y = S12/G12 ≈ 0.011) gives MAE **87 %** (KD 0.14/0.08/0.05 vs measured
0.89/0.63/0.47) — a ~10× over-prediction.  The fitted gate γ_Y (0.62) is
**57×** the physical value.  Pimenta–Pinho, being a foundation-based kink
model, shares this leading-order over-prediction, so it cannot anchor the
gate.

The missing physics the fitted γ_Y absorbs is the fibre **bending**
stiffness / couple-stress length scale (Fleck & Shu) — the same physics
D.4 showed the homogenised continuum cannot resolve.  Compressive kinking
strength is, in the state of the art, a *calibrated* quantity; the gate
is the appropriate modelling level.

## D.5 — Through-thickness wrinkle position — **gate factor adopted; FE cannot capture it**

The Li 2025 S-A-2 case is S-M-2's geometry (θ = 20°, D/T = 0.122) with the
wrinkle near the surface instead of mid-plane; measured KD jumps from 0.63
(Middle) to 0.98 (Above).  Two deliverables:

- **FE movable interface (capability added).** The graded through-thickness
  decay now centres at `wrinkle_z_position` (asymmetric taper to each
  surface), so an off-mid wrinkle is genuinely placed off-mid.  *But the
  continuum FE does not reproduce the position knockdown*: crack-band
  gives S-A-2 (above) 0.76 vs S-M-2 (mid) 0.81 — slightly *more*
  knockdown, opposite to measured.  The near-surface mildness is a
  free-surface / local load-shedding mechanism the homogenised continuum
  cannot represent (consistent with D.4).
- **Gate position factor (adopted).** `P(z) = (2·min(z, 1−z))^position_q`
  scales the gate's knockdown deficit (1 at mid-plane, 0 at a surface),
  calibrated on the S-M-2/S-A-2 pair (`position_q = 5.26` for F; one
  point, indicative).  With it the **full F dataset incl. S-A-2 is MAE
  5.0 %, 6/6**.

## Net picture / recommended use

| Need | Model | Accuracy (E / F) |
|---|---|---|
| UD knockdown prediction (incl. position) | **D.3+D.5 (θ, D/T, z) gate** | 2.8 % / **5.0 %, 6/6** |
| Mesh-objective FE field, wavelength axis | D.1 crack-band FE | 10.3 % / 20.1 % |
| Multidirectional (A–D) | existing BF / three-mechanism | (unchanged) |
| Continuum / structural buckling | D.4 `K_geo` (correct, but not for kinking) | — |

The throughline: the governing physics is **fibre-scale** (kinking with
fibre bending), which closed-form models capture (the gate) but the
homogenised continuum FE cannot.  The gate (D.3) is the production UD
predictor; the FE (D.1) gives mesh-objective field detail.

**Consolidated parity chart.** `validation/plot_all_validation.py`
renders every single-wrinkle case (Datasets A–F) on one
predicted-vs-experimental parity plot with a ±20 % corridor, each dataset
predicted by the model that physically applies to it (multidirectional
A–D via the analytical BF / three-mechanism path; UD E/F via this gate).
Running it prints the per-dataset scorecard — E 2.8 % (9/9), F 5.0 %
(6/6) — and writes `validation/fig_all_validation_parity.png`.

## Open items

- **E/F unified calibration**: blocked by E's indicative ÷830
  normalization; needs E's measured moulded pristine from the authors.
  Until then the gate parameters are material-realization specific.
- **Position factor robustness**: `position_q` is fit to a single point
  (S-A-2); more position data would firm it up.
- **Multi-wrinkle** (Li D-/T- cases): separate capability (the gate is
  single-wrinkle; the FE multi-wrinkle path exists but is uncalibrated).
