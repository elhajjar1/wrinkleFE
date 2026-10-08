# Interpreting results

A WrinkleFE run reports several numbers that all look like "the
knockdown". They are not interchangeable: they come from different
models, answer different questions, and carry different caveats. This
page says what each one means, what it does **not** mean, and which one
governs when they disagree.

For the units and sign conventions behind every value below, see
[Units & conventions](units_conventions.md).

## The headline numbers

### `analytical_knockdown` — closed-form residual strength fraction

`AnalysisResults.analytical_knockdown` is the **combined analytical
strength knockdown**: the predicted failure stress of the wrinkled
laminate as a fraction of the pristine one, so `1.0` means no strength
loss. In compression it is the CLT-weighted Budiansky–Fleck kink-band
knockdown (with the layup-dependent confinement `gamma_Y_eff` and the
optional Argon–Fleck quadratic coefficient); in tension it is the
*ultimate* fibre-failure knockdown taken as the minimum of the
three-mechanism model. `analytical_strength_MPa` is the same result
expressed as an absolute stress (against `Xc` for compression, `Xt` for
tension).

It does **not** mean:

- a margin of safety — it is a ratio against the pristine laminate, not
  against a design allowable;
- the delamination-onset load in tension — that is
  `analytical_onset_knockdown`, which is always strictly below
  `analytical_knockdown` and is `None` for compression or when the
  material card lacks both `GIc` and `GIIc`;
- a UD compression prediction that is scale-aware. The angle-based
  models are scale-invariant: at a fixed misalignment angle they cannot
  reproduce the dependence on through-thickness penetration. For
  unidirectional laminates set `AnalysisConfig.penetration_gate` (or
  `wrinklefe analyze --gate …`) so the knockdown comes from the
  (θ, D/T, z) penetration-gate model instead. The gate is **UD-scoped** —
  do not apply it to multidirectional or blocked laminates.

### `damage_index` — a reporting-only severity scalar

`AnalysisResults.damage_index` (the interlaminar damage index *D*, 0 =
pristine, 1 = full loss of load-carrying capacity) is computed from the
amplitude, the peak misalignment angle and the morphology factor. It is
**for reporting only and is not used in the knockdown computation** — no
strength number is derived from it. It matters downstream because the
severity banding below reads it as a second, independent metric.

### Stiffness: `analytical_modulus_knockdown`, `modulus_retention`, `modulus_retention_global`

Three different stiffness numbers, in increasing order of fidelity to a
measured coupon modulus:

| Attribute | Source | What it is |
|-----------|--------|------------|
| `analytical_modulus_knockdown` | closed form | CLT membrane series-average of the off-axis lamina modulus over the wrinkle profile, `E_x / E_x0`. Loading-independent. Available on the analytical path (no mesh needed). |
| `modulus_retention` | FE, local proxy | `E_eff = <σ₁₁> / ε_applied` from the mean *element-frame* fibre-direction stress, wrinkled vs pristine. |
| `modulus_retention_global` | FE, global reaction | `E_eff = σ_nominal / ε_applied` with `σ_nominal = R / A` — the total axial reaction on the loaded face over the cross-section — wrinkled vs pristine. |

`modulus_retention` averages the *local* fibre stress rather than the
coupon's global axial response, so it **over-predicts** the retained
modulus (it is flatter on the amplitude, penetration and position axes
than a measured `E_x / E_x0`). It is kept for backward compatibility.
**Prefer `modulus_retention_global`** for a coupon-level stiffness
knockdown: it captures load redistribution around the wrinkle and is
correspondingly lower.

A retention of exactly `1.0` is ambiguous on its own — it can be a
genuine no-knockdown result or a failed computation that fell back.
Check the companion flags `modulus_retention_failed` and
`modulus_retention_global_failed` (both also log a WARNING when they
fire) before quoting a `1.0`.

Stiffness knockdown is not strength knockdown. A wrinkle can retain most
of its axial modulus while losing far more of its compressive strength;
do not substitute one for the other in a disposition.

### `retention_factors` — FE first-ply-failure strength retention

`AnalysisResults.retention_factors` is a dict keyed by failure criterion
(LaRC05, Hashin, Puck). Each entry is `pristine_max_FI / wrinkled_max_FI`,
clipped at `1.0` — how much of the pristine *first-ply-failure* strength
survives under that criterion. `baseline_fi` carries the pristine maxima
the ratio was formed against.

It does **not** mean ultimate strength. It is a ratio of failure indices
from a single linear solve, so it reports the onset of the first ply
failure, not the load the coupon finally carries — which is why the
progressive-damage path exists. It is also a ratio of indices at one load,
and the LaRC05 kinking index is mildly nonlinear in load, so it is a
first-order strength ratio. (Before the LaRC05 kinking fix, a pristine UD
ply could not kink at all and this ratio was undefined for UD layups; see
`retention_degenerate`.)

**Treat FE strength as indicative, never as an allowable.** Where the FE
strength prediction has been checked against measured strength it missed
on the unsafe side exactly where it matters most: on the six Li (2025) UD
specimens it over-predicted retained strength on the two most severe
wrinkles, by up to +39%, while under-predicting the four milder ones. On
the one multidirectional laminate with measured strengths (three CFRP
wrinkles, Shi et al. 2025, Dataset H) it ran conservative, −8% to −1%
— a first check, not a validation (see
[How far to trust each number](#how-far-to-trust-each-number)). The app
and the NCR summary print this warning next to the number.

### `progressive_knockdown` — ultimate strength from load-stepping FE

With `AnalysisConfig.enable_progressive_damage = True` (or
`wrinklefe analyze --progressive`) the FE path load-steps a ply-discount
solve on both the wrinkled coupon and a pristine baseline and reports:

- `progressive_strength_MPa` — peak carried nominal stress over the
  wrinkled load history (the ultimate strength);
- `progressive_pristine_strength_MPa` — the same for the flat baseline;
- `progressive_knockdown` — their ratio;
- `progressive_history` — the `(applied_strain, nominal_stress)` samples.

This is the only FE route that carries UD compression *past* first-ply
failure, and therefore the only FE strength knockdown that is meaningful
for pristine UD. Read `progressive_knockdown` against
`retention_factors`: FPF is the onset, `progressive_knockdown` the
ultimate. They are different loads and the gap between them is real.

Its fracture-energy calibration (crack band, `Gf = 3.0`) is only valid at
the mesh it was fitted at, `nx = 16`, `nz_per_ply = 2`: refining the mesh
changes the answer, and on a 12-elements-per-wavelength mesh it inverts the
amplitude ordering. Against measured strength it **over-predicts the most
severe wrinkle** (+24% calibrated, +42% refined) while under-predicting the
milder ones. It is a research output, not an allowable.

The peak is only as trustworthy as the ramp that brackets it: if
`progressive_max_strain` is too small the history never reaches the peak
and the "ultimate" is just the last increment. Increase
`progressive_n_increments` (default 15) if the history looks coarse
around the maximum.

### CZM outcomes — delamination, not a strength knockdown

A cohesive-zone run (`enable_czm=True`) does not produce a knockdown
factor. It produces a delamination picture:

- `czm_damage` — the cohesive damage variable per interface element and
  Gauss point (0 = intact, 1 = fully separated);
- `czm_crack_length_per_interface` — crack length in mm per ply
  interface, computed as the in-plane area of elements with
  `damage > 0.99` divided by the mesh width;
- `czm_energy_dissipated` / `czm_energy_per_interface` — dissipated
  cohesive energy (N·mm);
- `czm_load_displacement` — the `(λ, ‖u‖)` increment samples.

**Always check `czm_converged` first.** When it is `False`, the damage
and energy fields are the state of a solve that did not complete and
must not be quoted. `czm_failure_diagnostics` records the first
non-converged increment and `czm_failure_hint` names the knob to reach
for (`czm_n_load_increments`, `czm_newton_tol`, the applied strain).

### The acceptance limit — a safe-side inverse answer

`find_critical_value` (CLI: `wrinklefe critical`) inverts the forward
model: *given this allowable, what is the largest wrinkle we can
accept?* The acceptable set is always `{x : objective(x) >= target}`, so
a larger objective is safer. For a decreasing objective the answer is
the **largest** acceptable value.

The semantics that matter for a disposition:

- `critical_value` is **verified by a real forward evaluation**, not by
  the root tolerance. The engine backs the raw root off to the safe
  side and re-evaluates, so the returned value satisfies the criterion
  when you re-run it.
- `critical_value_root` is the raw `brentq` root. It is diagnostic only —
  never quote it as the limit.
- A refusal is a returned outcome, not an exception. Check
  `result.status` (`"converged"` or otherwise) and read `result.message`,
  which names the measurement behind the refusal and the knob that has
  to move. A non-monotonic or flat objective means there is no single
  crossing to report.

The NCR summary states this basis verbatim: *"Largest value still
satisfying the target under a real forward evaluation: the root-find is
backed off to the safe side and re-verified, so this is not an
interpolation of the scan curve."*

## Which number governs

- **Multidirectional laminates** — the angle-based analytical models are
  the intended path. Use `analytical_knockdown` for screening and the FE
  numbers to check the mechanism and the load redistribution.
- **Unidirectional laminates in compression** — the angle-only knockdown
  is scale-invariant and under-predicts the penetration effect. Use the
  penetration gate. `progressive_knockdown` is the only FE number that
  reaches past first-ply failure here, but it is mesh-calibrated and
  over-predicts severe wrinkles, so use it to study the mechanism, not
  to set a knockdown. `retention_factors` will not help here.
- **Tension with a delamination concern** — `analytical_knockdown` is
  the ultimate; `analytical_onset_knockdown` is the first load drop. If
  the drawing requirement is written against onset, the onset number
  governs.
- **Stiffness-critical checks** — `modulus_retention_global`, not
  `modulus_retention`.
- **When the analytical and FE numbers disagree** — they are answering
  different questions before they are disagreeing. Confirm you are
  comparing like with like (FPF vs ultimate, local vs global stiffness,
  angle-only vs gated) before treating the difference as model error.
  Where they genuinely bracket the answer, the lower (more conservative)
  number is the one to carry into a disposition.
- **Never let an FE strength number raise a knockdown.** The FE strength
  paths have only ever erred on the unsafe side in validation, so an FE
  result *above* the analytical knockdown is not evidence that the part
  is stronger.

## How far to trust each number

Signed errors against measured strength, `(predicted − measured) /
measured`. A positive error means the prediction says the wrinkled part
keeps *more* strength than it did: the unsafe direction. Every figure here
is regenerated by `python validation/strength_error_summary.py`.

**Analytical paths** — each dataset predicted by the model that applies to
it; 46 of the 54 analytical cases fall within ±20% (the parity figure,
which also plots Dataset H's FE series, shows 49 of 60 overall):

| Dataset | Model | n | Mean abs. error | Range | Non-conservative |
|---|---|---|---|---|---|
| A Elhajjar (2025), compression | Budiansky–Fleck | 13 | 9.5% | −9.1% … **+29.6%** | 5 / 13 |
| B Elhajjar (2025), tension | three-mechanism | 7 | 6.2% | −15.7% … +9.7% | 3 / 7 |
| C Mukhopadhyay (2015), compression | Budiansky–Fleck | 3 | 7.7% | −10.8% … −5.2% | 0 / 3 |
| C Mukhopadhyay (2015), tension | three-mechanism | 3 | 19.9% | −22.4% … −18.2% | 0 / 3 |
| C Mukhopadhyay (2015), delam. onset | three-mechanism onset | 3 | 15.2% | −22.5% … +1.9% | 1 / 3 |
| D Wang (2021), concave/convex | Budiansky–Fleck + morphology | 4 | 16.3% | −27.4% … −1.6% | 0 / 4 |
| E Li (2024), UD | penetration gate | 9 | 2.8% | −6.6% … +8.5% | 4 / 9 |
| F Li (2025), UD | penetration gate + position | 6 | 5.0% | −12.9% … +14.6% | 2 / 6 |
| H Shi (2025), UD CFRP | Budiansky–Fleck (no gate preset) | 3 | 12.6% | +0.2% … **+25.9%** | 3 / 3 |
| H Shi (2025), multidirectional CFRP | Budiansky–Fleck | 3 | 58.8% | +5.8% … **+115.2%** | 3 / 3 |

Of the eight analytical cases outside ±20%, four are conservative. The
four that are not: the most severe wrinkle in dataset A (+29.6%,
A = 0.73 mm, measured knockdown 0.32, predicted 0.42), and Dataset H's
severe cases — +25.9% on the UD half, and +55.6% / +115.2% on the
multidirectional half, where the measured severities exceed the UD
half's (buckling participation) and an angle-only model cannot follow.
For multidirectional strength, read the FE retention table below
(−8% to −1% on those same cases) rather than the analytical column.

**FE strength (LaRC05)** — Li (2025) UD glass/epoxy, the FE path's
first measured-strength check (Dataset H below is the second). Wrinkled
over pristine strength at first failure:

| Case | Wrinkle | FE | Measured | Error |
|---|---|---|---|---|
| S-M-1 | 1.5 mm, 10° | 0.847 | 0.891 | −4.9% |
| S-M-2 | 1.5 mm, 20° | 0.756 | 0.629 | **+20.1%** |
| S-M-3 | 1.5 mm, 30° | 0.655 | 0.472 | **+38.7%** |
| S-M-4 | 1.0 mm, 20° | 0.767 | 0.943 | −18.7% |
| S-M-5 | 0.5 mm, 20° | 0.759 | 1.000 | −24.1% |
| S-A-2 | 1.5 mm, 20°, near-surface | 0.756 | 0.981 | −23.0% |

The four 20° cases all predict about 0.76 whatever the amplitude: the FE
first-ply path sees the wrinkle's angle, not its size, so it over-predicts
the severe 20° wrinkle and under-predicts the mild ones. (These figures
follow the LaRC05 kinking fix. Before it, a pristine UD ply could not kink,
the FE strengths had to be normalised to the S-M-5 wrinkle, and the FE
over-predicted every non-reference case, by up to +58%.)

**Dataset H — Shi et al. (2025) CFRP** (first-ply FE, `tool_flat`
recipe, nx = 48; `validation/strength_error_summary.py` section 2b). The
multidirectional half is the first measured-strength check of the FE
retention path outside UD:

| Case | Layup | t/T | FE retention | Measured | Error |
|---|---|---|---|---|---|
| H-MD-10 | [45/0/−45/90/45/0/−45/0/45/0]s | 10% | 0.702 | 0.760 | −7.7% |
| H-MD-20 | 〃 | 20% | 0.440 | 0.475 | −7.4% |
| H-MD-30 | 〃 | 30% | 0.323 | 0.326 | −0.9% |
| H-UD-10 | [0]₂₀ | 10% | 0.471 | 0.638 | −26.2% |
| H-UD-20 | 〃 | 20% | 0.313 | 0.494 | −36.7% |
| H-UD-30 | 〃 | 30% | 0.240 | 0.400 | −39.9% |

Conservative across the board: −8% to −1% on the multidirectional
laminate, and strongly conservative on the UD half, where the FE sees
the full local stress concentration of this short (5.5 mm span)
one-sided wrinkle while the specimens carry on past first-ply failure.
One dataset, three cases per layup, with buckling participation in the
30% failures — a first check, not a validation. (Mesh-checked: at
nx = 64 every error moves by less than 2.5 percentage points, MD
−5.8/−5.0/−3.0%.)

**Progressive damage (crack band)** — the predictions pinned in the
validation ledger:

| Mesh | Case | Predicted | Measured | Error |
|---|---|---|---|---|
| nx = 16 (calibrated) | S-M-2 | 0.822 | 0.629 | **+30.7%** |
| nx = 16 (calibrated) | S-M-4 | 0.924 | 0.943 | −2.0% |
| nx = 16 (calibrated) | S-M-5 | 1.003 | 1.000 | +0.3% |
| nx = 36 (refined) | S-M-2 | 0.882 | 0.629 | **+40.3%** |

The pattern across all three: the unsafe misses concentrate on the most
severe wrinkles, which is where a disposition can least afford one. (The
milder two now sit within a few per cent, but the fracture-energy
calibration predates the fibre-angle sign fix, so treat that agreement
as fortuitous until the calibration is redone.) The
analytical knockdown (the penetration gate for UD) is the validated
strength path; carry the usual design margin on it, and more for severe
wrinkles.

## Severity bands

The values below are transcribed from `_SEVERITY_BANDS` in
{mod}`wrinklefe.io.export`, which remains the authoritative source — the
NCR summary produced by `build_analysis_summary` / `recommend_disposition`
is generated from that table, not from this page.

A wrinkle is scored on two metrics: the residual-strength fraction (the
analytical knockdown) and the damage index *D*. **The worst (lowest) tier
from either metric governs.** `recommend_disposition` reports which one
did, in `governed_by`.

| Severity | Residual strength (knockdown) ≥ | Damage index D < | Recommended path | Required approvals |
|----------|--------------------------------|------------------|------------------|--------------------|
| Negligible | 0.97 | 0.05 | Candidate for USE-AS-IS, contingent on confirming residual strength ≥ design allowable for the affected location. | Originating/Design Engineering |
| Minor | 0.90 | 0.20 | USE-AS-IS with documented stress justification, or a cosmetic blend/local rework if the wrinkle is surface-accessible. | Design Engineering; Quality |
| Moderate | 0.75 | 0.40 | Engineering disposition required: REPAIR per an approved scheme, or USE-AS-IS only if a positive margin of safety is demonstrated by substantiating analysis or test. | Design Engineering; Stress; Quality |
| Major | 0.50 | 0.65 | REPAIR or REWORK per a qualified procedure with full MRB substantiation. Customer/DER concurrence is likely required. | Design Engineering; Stress; Quality; Customer/DER |
| Severe | 0.0 | 1.01 | REJECT — SCRAP, or major REPAIR only under an engineering-approved, fully substantiated scheme. Mandatory customer/DER review. | Design Engineering; Stress; Quality; Customer/DER; Program Management |

`recommend_disposition(knockdown, damage_index, loading=…)` returns the
band label, the recommended path, the required approvals, a rationale
naming the governing metric, and `is_final: False`. Passing `loading`
annotates the rationale only — compression-dominated wrinkles are called
out as the less tolerant case, tension as the more tolerant one — it
does not move the band.

## Scope and authority

The severity bands are generic engineering guidance, not a disposition.
The scope note carried in `wrinklefe.io.export` states it directly:

> Scope/authority note: the recommendation produced here is *decision
> support only*. It does not constitute a final disposition. Severity
> thresholds below are generic engineering guidance and MUST be
> superseded by the program-specific allowables, drawing requirements,
> and process specifications that the Material Review Board (MRB)
> applies. The qualified MRB reviews, may modify, and approves the final
> disposition.

Every NCR validation summary produced by `build_analysis_summary` carries
the same language on its face:

> This validation summary was prepared with WrinkleFE decision-support
> tooling and is intended as an attachment to a Nonconformance Report.
> The analysis and recommendation are advisory and do not constitute a
> final material disposition. A qualified Material Review Board must
> review, may modify, and formally approve the disposition.

and on the disposition block itself:

> Decision support only. The Material Review Board reviews, may modify,
> and approves the final disposition against the controlling drawing and
> program allowables.

An acceptance limit attached to a summary carries its own note:

> Decision support only. This limit is advisory and is superseded by the
> program-specific allowables, drawing requirements, and process
> specifications applied by the MRB.

## Where these numbers end up

- **The NCR attachment.** `wrinklefe.io.build_analysis_summary` assembles
  the wrinkle geometry, the laminate, the engineering results, the cited
  criteria and the recommended (non-binding) disposition into a
  structured summary; `export_summary` writes it as Markdown, JSON or
  PDF. It deliberately carries no QMS/admin fields (NCR number,
  part/serial, work order, MRB sign-off) — that paperwork lives on the
  NCR itself.
- **The acceptance limit.** `wrinklefe critical` (or `find_critical_value`)
  produces the limit; it is attached to a summary through the
  `critical_limit` argument, and only when it was derived for the *same*
  configuration the results came from.
- **A worked end-to-end run** — measure, configure, run, read, invert,
  export — is on the [tutorial](tutorial.md) page.
