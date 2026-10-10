# Bending-dominated failure at high amplitude-to-thickness (issue #439)

**Finding.** The angle-based kink-band model's worst unsafe misses are
not a kinking problem. They occur when the wrinkle moves the laminate's
**load path**, i.e. when the stiffness-weighted centroid of the section
at the wrinkle is offset from the far-field load line. Split by that
offset (the eccentricity e, over the far-field thickness t), the
deployed model is never more than 15% unsafe below e/t = 0.10 (39
cases). Above e/t = 0.15 it is unsafe on 7 of 8 cases. There, first-ply
FE run on the specimen's own geometry is never unsafe (−40% to −1%).

Regenerate everything here with `python validation/bending_mode_study.py`
(per-case table: `validation/bending_mode_cases.csv`).

## Method

Every compression strength case in the validation set (50 cases,
Datasets A, C, D, E, F, H, J, K) was scored with the model a user would
deploy (kink-band, or the fitted gate on E/F) and with first-ply FE
LaRC05, as in the head-to-head (#433). For each case the FE mesh of its
recipe was generated, not solved, and measured:

- **e**: the largest offset, along the length, of the section's
  axial-stiffness-weighted centroid from its far-field value (each ply
  weighted by its CLT axial modulus);
- **t_c**: the section thickness at that point.

Shi (Dataset H) uses its FE recipe (one-sided `tool_flat` trough); the
others use their ledger or head-to-head recipes.

Eccentricity separates geometries that amplitude alone does not:

| Profile | Examples | e |
|---|---|---|
| Whole-thickness wave (both surfaces wavy) | Thor (J), Pilato (K), Elhajjar under its `uniform` recipe (A) | ≈ A |
| One-sided surface trough, flat tool face | Shi (H) | ≈ half the trough depth |
| Embedded wrinkle, flat outer surfaces | Li (E, F), Mukhopadhyay (C), Wang (D) | ≈ 0 (UD) to a few % of t |

An embedded wrinkle misaligns fibres but leaves the load path where it
was; a wave that moves the surfaces bends the section.

## Transition metrics

Six of the 50 cases are more than 15% unsafe under the deployed model:
Elhajjar A = 0.61 and 0.73 mm, Shi UD t/T = 30%, Shi multidirectional
t/T = 20% and 30%, and Thor quasi-isotropic wave 1. The best single
threshold on each metric:

| Metric | Best split | Misclassified (of 50) |
|---|---|---|
| Amplitude / thickness | ≥ 0.227 | 2 |
| Eccentricity / thickness | ≥ 0.117 | 3 |
| Peak angle | ≥ 33.1° | 4 |
| Amplitude / wavelength | ≥ 0.104 | 4 |

Peak angle, the variable the kink-band model uses, is the weakest
separator. Amplitude/thickness splits the six best, but it flags
Wang's wrinkles (A/t = 0.20), where the deployed model is fine and FE is
unsafe; eccentricity does not. The model-choice table therefore uses
eccentricity:

| e/t | Cases | Deployed: unsafe (> +15%) | FE LaRC05: unsafe (> +15%) | FE range |
|---|---|---|---|---|
| < 0.05 | 31 | 6 (0) | 24 (17) | −24% … +92% |
| 0.05–0.10 | 8 | 1 (0) | 5 (2) | −26% … +24% |
| 0.10–0.15 | 3 | 3 (1) | 1 (1) | −37% … +20% |
| ≥ 0.15 | 8 | **7 (5)** | **0 (0)** | −40% … −1% |

**Regime selector.** Using the deployed model below e/t = 0.10 and FE
above:
- cases more than 15% unsafe drop from 6 to 1, the borderline Elhajjar
  A = 0.24 mm at e/t = 0.102, with FE at +20%;
- the worst miss drops from +115% to +20%;
- the MAE drops from 20.9% to 16.8%.

## Reported failure modes

| Dataset | What the paper reports | e/t of its unsafe cases |
|---|---|---|
| J Thor (2021) | wave 1 (A/t 0.23): bending from the wave geometry, delamination, then fibre breakage; wave 2: fibre kinking | 0.25 |
| H Shi (2025) | kinking, with buckling participation at t/T = 30% | 0.12–0.18 |
| A Elhajjar (2025) | early kink-band failure at the severe end (about 0.4 normalised strength) | 0.25–0.31 |
| C Mukhopadhyay (2015) | fibre compression below 8–9°, delamination above | 0.01 (deployed model conservative) |
| K Pilato (2022) | delamination in the paper's FE; delamination and fibre breakage post-test | 0.05 (deployed model conservative) |

A delamination-mode label alone does not predict an unsafe miss:
Mukhopadhyay's delamination cases (13–16°) are predicted conservatively.
What the unsafe cases share is the load-path offset.

## A closed-form bending knockdown: partial

Treating the wrinkled section as an eccentric column, the outer fibres
reach the pristine strength at

    KD_bend = (t_c / t_0) / (1 + 6 e / t_c)

and the prototype predicts min(deployed, KD_bend). Results:

| Case | e/t | Measured | Deployed | Prototype |
|---|---|---|---|---|
| Thor quasi-isotropic, wave 1 | 0.25 | 0.347 | +52.1% | **+15.1%** |
| Elhajjar A = 0.73 mm | 0.31 | 0.320 | +29.6% | **+10.4%** |
| Elhajjar A = 0.61 mm | 0.25 | 0.350 | +18.5% | **+13.1%** |
| Shi multidirectional, t/T = 30% | 0.18 | 0.326 | +115.2% | +115.2% (KD_bend 0.72) |
| Shi multidirectional, t/T = 20% | 0.12 | 0.475 | +55.6% | +55.6% (KD_bend 0.76) |

It fixes the whole-thickness waves, where the eccentricity acts over the
whole coupon. It does not explain Shi's one-sided trough, where the
measured knockdown is far below what global eccentricity alone gives.
That points to a local effect, a thinned, curved ligament under the
trough, which the FE mesh captures and a beam formula does not. Over
all 50 cases the prototype cuts the >15% misses from 6 to 4.

## Recommendation

**Adopt, in stages:**

1. **Report the regime.** Compute e/t from the wrinkle geometry in the
   analysis and report it beside the knockdown, with a flag when
   e/t ≥ 0.10. The flag says the kink-band knockdown is outside its
   validated regime and names FE on the measured profile as the
   recommended path. This changes no number, only what is reported.
2. **Prefer FE above the threshold in the guidance** in
   `docs/interpreting_results.md`, for the specimen's own geometry. Do
   not prefer it below the threshold, where FE is unsafe on 29 of 39
   cases.
3. **Do not ship KD_bend as a knockdown yet.** It helps whole-thickness
   waves but misses the one-sided trough. It could serve as a
   conservative screening bound for whole-thickness waves, after more
   data.

**Needs data:**
- **Threshold.** The 0.10–0.15 transition holds 3 cases from 2 studies.
- **Regime size.** The ≥ 0.15 regime holds 8 cases from 3 studies.
- **Elhajjar profile.** Its `uniform` recipe assumes a whole-thickness
  wave; confirming the measured profile would settle where A sits.

## Caveats

- Eccentricity is only as good as the recipe's geometry.
  - **J and K** have measured whole-thickness waves.
  - **H** uses the one-sided FE recipe.
  - **A** assumes `uniform`.
  - **C and D** use the head-to-head recipes, which are not
    specimen-tuned.
- The deployed model on UD carbon (K, J UD) is badly conservative
  (−57% to −79%) for a different reason, the angle-only law on UD, which
  is issue #435. The regime selector does not address it.
- Tension is excluded. The bending argument is about compression.
