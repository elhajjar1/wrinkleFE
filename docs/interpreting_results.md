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
  do not apply it to multidirectional or blocked laminates. It is also
  **material-scoped**: the shipped presets were fitted to one S-glass
  prepreg and are unsafe on new UD carbon (7 of 10 cases, up to +97%).
  Whether re-fitting its `gamma_Y` to one wrinkled test of a new
  material is enough is **unresolved**: it predicted two carbon sets'
  other cases within 2–4%, missed a third by up to −29%, and failed on a
  quarantined wet lay-up set (up to +71%). See [Validation](validation.md)
  ("Gate transfer to new materials").

### `damage_index` — a reporting-only severity scalar

`AnalysisResults.damage_index` (the interlaminar damage index *D*, 0 =
pristine, 1 = full loss of load-carrying capacity) is computed from the
amplitude, the peak misalignment angle and the morphology factor. It is
**for reporting only and is not used in the knockdown computation** — no
strength number is derived from it. It matters downstream because the
severity banding below reads it as a second, independent metric.

### `load_path_eccentricity_ratio` and `bending_dominated` — is the knockdown in its regime?

`AnalysisResults.load_path_eccentricity_ratio` is e/t: how far the wrinkle
moves the laminate's load path, as a fraction of the laminate thickness.
- **e** is the largest offset of the section's axial-stiffness-weighted
  centroid from the straight line joining its values at the ends of the
  domain.
- It is measured on the ply boundaries the FE mesh is built on, in about
  a millisecond and without a solve, so analytical-only runs report it too.
- `load_path_eccentricity_mm` is the same offset in millimetres.

Typical values:

| Wrinkle | e |
|---|---|
| Whole-thickness wave (both surfaces wavy) | about its amplitude |
| One-sided surface trough over a flat tool face | about half the trough depth |
| Embedded wrinkle with flat outer surfaces | close to zero |

`bending_dominated` is `True` for a **compression** analysis with
e/t ≥ 0.10. At that point the section bends as well as kinks, and the
kink-band `analytical_knockdown` does not represent that.

On measured cases (issue #439):
- **Below 0.10** (39 cases), the deployed analytical model was never more
  than 15% unsafe.
- **Above 0.15** (8 cases), it was unsafe on 7, by up to +115%. First-ply
  FE on the specimen's own wrinkle geometry was never unsafe there.

When the flag is set, prefer FE on the measured profile. The text
summary, the results JSON, the NCR summary and the app all carry the
same warning. The flag changes no number. Tension analyses report the
ratio but are not flagged, because the study covered compression only.

### `manufacturing_process` — is the laminate in the validated scope?

Set `AnalysisConfig.manufacturing_process` (CLI `--process`, app
**Manufacturing process**) to how the laminate was made:
`prepreg_autoclave`, `prepreg_press`, `prepreg_vacuum_bag`, `rtm`,
`infusion`, `wet_layup`, `other`, or `unspecified` (the default).

The knockdown models have been validated only on laminates made by the
routes a validation dataset documents: prepreg cured in an autoclave
(Elhajjar, Thor), in a hot press or closed mould (Elhajjar, Li 2024) and
under a vacuum bag (Li 2025). The other datasets do not state their
route, so they add nothing to the scope. Laminate quality — fibre volume fraction, porosity, how
straight the fibres are in the undamaged laminate — differs between
routes, and no model represents it.

- **Out of scope** (`rtm`, `infusion`, `wet_layup`, `other`):
  `process_out_of_scope` is set and `process_scope_caveat` carries the
  warning. For wet lay-up it adds what happened on the one wet lay-up
  dataset (Dataset L, quarantined): the kink-band was about 56%
  conservative on average while first-ply FE over-predicted retained
  strength by up to about 100%.
- **A penetration-gate preset outside its fit:** the presets record the
  material system (`AC318_S6C10` cards) and the route they were fitted
  to. Applied to another material, or to a laminate whose stated route
  differs, `gate_out_of_scope` is set and `gate_scope_caveat` says so. A
  gate you calibrated yourself is not checked.

Wet lay-up, infusion and RTM stay out of scope by decision: a literature
search found one usable infusion dataset (tension only) and none for the
other two. A process comes into scope once two wrinkle datasets from two
labs, each with an in-standard pristine baseline, are in the validation
ledger (see [Validation](validation.md), "Manufacturing-process scope").

The text summary, the results JSON (`scope` block, schema 1.4), the NCR
summary and the app carry the same warnings. **Nothing changes a
number**: the scope is derived from
`wrinklefe.core.process.VALIDATION_EVIDENCE`, so a wrinkle dataset made
by a new route brings that route into scope when it is added (issue
#445).

### `validation_margin` — how far to discount the knockdown, by regime

The results JSON (`validation_margin` block, schema 1.5) and the NCR
summary (all three formats) report, next to `analytical_knockdown`, the
**validation margin** that applies to it, the regime it comes from and
the number of validation cases behind it (issue #455). Reporting only:
no knockdown changes, and the severity banding still reads the
unmargined value.

**Definition.** The margin *m* is the factor to multiply the predicted
knockdown by so that 95% of the validation cases in the same regime
would have been conservative:

> measured knockdown ≥ *m* × predicted knockdown, for 95% of the regime's cases.

So *m* is the 5% quantile of measured / predicted, capped at 1 (a
margin never raises a knockdown). The report gives the margined
knockdown *m* × `analytical_knockdown` alongside.

**Regimes.** Four things decide how far a prediction can be trusted, so
each cell of their product is a regime:

- **Model path.** Kink-band (analytical, compression), three-mechanism
  (analytical, tension) or the penetration gate, whichever produced
  `analytical_knockdown`. The table also scores the FE strength paths.
- **Load.** Compression or tension.
- **Laminate type.** UD-dominated when at least 75% of the plies are
  within 5° of the load axis, otherwise multidirectional. The validation
  layups sit well either side: 89–100% for the UD sets, 25–40% for the
  multidirectional ones.
- **Bending regime.** `bending_dominated` (compression only; see above).
  For each validation case it is measured on the specimen's own wrinkle
  geometry. For Dataset H's one-sided trough that is the ledger's
  `eccentricity_recipe`, because its knockdown recipe is an embedded
  profile.

**Fitting method, and why not a normal.** The residuals are fat-tailed
and often multi-modal. On the kink-band UD cases measured / predicted
runs from 1.0 to 4.7. On the multidirectional tension cases it falls in
two tight clusters. A normal fitted to such a sample puts the 5%
quantile in the wrong place. It is too high when a minority of cases
sits far on the unsafe side, because it averages them into a symmetric
bell. It is too low when two tight clusters are read as one wide one.

The margin therefore comes from a **two-component Gaussian mixture**,
the model Elhajjar (2025) fitted to the fat-tailed, multi-modal strength
distributions that wrinkles produce. It follows either shape, and it
reduces to a normal when the residuals are unimodal. The details:

- It is fitted by expectation-maximisation to ln(measured / predicted),
  because the errors are multiplicative and the ratio stays positive.
- Every split of the sorted sample is tried as a start, so the fit is
  deterministic.
- Each component's standard deviation is floored at 0.05, about the test
  scatter of one measured knockdown. For scale, Dataset M's replicate
  coupons, with identical geometry and so identical predictions, differ
  in measured knockdown by up to about 20% (M-MD-8 0.699, M-MD-9 0.585).
  The floor also stops a component collapsing onto a single case.

**Minimum sample: 10 cases.** That is two per free parameter of the
mixture (two means, two standard deviations, one weight). A regime with
fewer cases reports **insufficient evidence** and its *n*, not a margin.

**The current table** (`python scripts/fit_margins.py` prints it;
non-quarantined ledger cases only):

| Regime | n | Margin | Mixture 5% quantile | Normal 5% quantile (comparison) | Smallest ratio |
|---|---|---|---|---|---|
| kink-band · compression · UD-dominated · not bending-dominated | 14 | **1.00** | 1.219 | 1.212 | 0.998 |
| three-mechanism · tension · multidirectional | 10 | **0.771** | 0.771 | 0.747 | 0.771 |
| kink-band · compression · UD-dominated · bending-dominated | 3 | insufficient evidence | — | — | 0.794 |
| kink-band · compression · multidirectional · not bending-dominated | 2 | insufficient evidence | — | — | 0.945 |
| kink-band · compression · multidirectional · bending-dominated | 3 | insufficient evidence | — | — | 0.465 |
| penetration gate · compression · UD-dominated | 6 | insufficient evidence | — | — | 0.866 |
| three-mechanism · tension · UD-dominated | 4 | insufficient evidence | — | — | 0.719 |
| progressive damage (crack band) · compression · UD-dominated | 3 | insufficient evidence | — | — | 0.765 |
| FE first-ply retention (LaRC05), any regime | 0 | insufficient evidence | — | — | — |

Reading it:

- **Multidirectional compression has no margin yet**, in either bending
  regime. That is the most common case in practice. The ledger holds
  only five such cases (Shi, Thor). Datasets A, C and D are scored by
  `validation/strength_error_summary.py` but are not in the ledger.
- **The kink-band UD margin is 1.00** because the angle-only law is
  strongly conservative on most UD sets: measured / predicted is 1.9 to
  4.7 on Li (2025), Thor and Pilato. One case, Shi H-UD-10, is level at
  0.998.
- **The tension multidirectional margin, 0.771,** sits above the normal's
  0.747 because the residuals are two tight clusters, not one wide bell.
  Both mixture components sit at the 0.05 floor, so the floor sets this
  value.
- **The gate's six cases are in-sample:** the preset was fitted to them.
  Even with more cases its margin would measure fit quality, not
  prediction error.
- **FE first-ply retention has no pinned predictions in the ledger.** Its
  scores live in the head-to-head matrix (below), which is not pinned.
  It keeps the caveat it carries everywhere.

**Not a basis value.** The margin is a point estimate from a few
published studies. It is not a CMH-17 A- or B-basis value, which would
need a statistical tolerance bound on test data at a stated confidence.
Read it as how far this regime's predictions have missed on the unsafe
side so far, not as an allowable.

**Keeping it current.** The ledger is not installed with the package,
so the table ships as package data
(`src/wrinklefe/core/validation_margins.json`), generated by
`scripts/fit_margins.py`. After a model change,
`python scripts/validate.py --update` re-pins the ledger and regenerates
the table in one step. `python scripts/validate.py` and
`tests/test_validation/test_margins_ledger.py` fail while the table is stale.

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
progressive-damage path exists. Even as an onset it has one measured
check in tension (Dataset M), where it was unsafe on 9 of 12 coupons, by
up to +154%. It is also a ratio of indices at one load,
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
- **Any compression result with `bending_dominated` set** (e/t ≥ 0.10) —
  the analytical knockdown is outside its validated regime. Run FE on the
  measured wrinkle profile and carry that number, with margin.
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
  paths have erred on the unsafe side in validation, by up to +92% on a
  multidirectional laminate, so an FE result *above* the analytical
  knockdown is not evidence that the part is stronger.

## How far to trust each number

Signed errors against measured strength, `(predicted − measured) /
measured`. A positive error means the prediction says the wrinkled part
keeps *more* strength than it did: the unsafe direction. Every figure here
is regenerated by `python validation/strength_error_summary.py`.

**Analytical paths** — each dataset predicted by the model that applies to
it; 55 of the 77 analytical cases fall within ±20% (the parity figure,
which also plots the FE series of Datasets H to K and Dataset M's FE
onset, shows 70 of 105 overall). Dataset L is **quarantined** (issue #445): a low-quality wet
lay-up laminate, listed for reference and left out of every count here.

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
| I Calvo (2023), multidirectional CFRP, tension, 7 distributed waves | three-mechanism | 1 | 8.4% | −8.4% | 0 / 1 |
| M Elhajjar & Shams (2016), multidirectional T700GC/2510, tension, steep rod-made wrinkles (23–36°) | three-mechanism | 9 | 13.4% | −7.6% … **+29.6%** | 8 / 9 |
| M Elhajjar & Shams (2016), UD T700GC/2510, tension (22–27°) | three-mechanism | 4 | 19.5% | −1.0% … **+39.2%** | 2 / 4 |
| J Thor (2021), quasi-isotropic IM7/8552, whole-thickness waves | Budiansky–Fleck | 2 | 30.0% | −7.9% … **+52.1%** | 1 / 2 |
| J Thor (2021), UD IM7/8552 | Budiansky–Fleck (no gate preset) | 2 | 57.9% | −58.9% … −56.8% | 0 / 2 |
| K Pilato (2022), near-UD industrial wrinkles | Budiansky–Fleck (no gate preset) | 5 | 77.5% | −78.7% … −76.8% | 0 / 5 |
| *L Krishnappa & Gururaja (2024), wet lay-up UD carbon, 18 strip-induced wrinkles (quarantined)* | Budiansky–Fleck (no gate preset) | 18 | 55.8% | −78.7% … −6.9% | 0 / 18 |

Of the 22 analytical cases outside ±20%, eleven are conservative; seven
of those are the kink-band on UD carbon (J, K), where the angle-only law
is far too pessimistic. The quarantined Dataset L adds fifteen more
conservative ones. Six of the eleven unsafe ones are Dataset M's steep
tension wrinkles (+20% to +39%, below). The five in compression are:
- the most severe wrinkle in dataset A (+29.6%, A = 0.73 mm, measured
  knockdown 0.32, predicted 0.42);
- Dataset H's severe cases, +25.9% on the UD half and +55.6% / +115.2%
  on the multidirectional half, where buckling takes part;
- Thor's severe quasi-isotropic wave (+52.1%), which fails by bending of
  the wavy coupon: delamination, then fibre breakage.

All five sit at high amplitude-to-thickness ratios, where the failure
mode is not the kinking the model represents (issue #439). On the H and
J cases the FE retention, run on the specimen's own geometry, is within
−9% to −1% (tables below). On the other multidirectional compression
datasets, with generic recipes, the FE misses on the unsafe side: see
[the head-to-head](#head-to-head-which-model-when-provisional).

**FE strength (LaRC05)** — Li (2025) UD glass/epoxy, the FE path's
first measured-strength check (Datasets H and I below are the second and
third). Wrinkled over pristine strength at first failure:

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

**FE onset in tension (LaRC05 first ply)** — Dataset M, Elhajjar & Shams
(2016): T700GC/2510 tension coupons with a measured damage onset, the
DIC limit point (interlaminar damage, no load drop). No pristine onset is
reported, so the FE is compared in absolute stress: the wrinkled coupon's
nominal stress at first ply failure.

| Case | Layup, wrinkle | FE onset (MPa) | Measured (MPa) | Error |
|---|---|---|---|---|
| M-MD-1, -2 | multidirectional, A = 0.8 mm, 30° | 192 | 82, 111 | **+134%, +74%** |
| M-MD-3 to -5 | multidirectional, A = 0.6 mm, 23° | 277 | 190–214 | **+30% to +46%** |
| M-MD-6, -7 | multidirectional, A = 0.3 mm, ~36° | 180 | 181, 230 | −0.4%, −21.7% |
| M-MD-8, -9 | multidirectional, A = 0.5 mm, ~34° | 184 | 222, 170 | −17.3%, **+8.1%** |
| M-UD-10, -11 | UD, A = 0.28 mm, ~27° | 534 | 435, 419 | **+23%, +28%** |
| M-UD-13 | UD, A = 0.35 mm, ~22° | 639 | 252 | **+154%** |

Unsafe on 9 of 12 (median +29%), worst where the dip is deepest. The
limit point marks interlaminar damage, which an intralaminar first-ply
criterion does not represent. Specimen M-UD-12 is left out (preloaded
before its DIC test). Against the *ultimate* the same retention is
conservative on all 13 (−27% to −58%), as on Dataset I: that is the
"conservative by construction" reading, and it does not carry over to
onset. Angles marked ~ are estimated, not measured; a ramp over 8 plies
instead of 15 lowers every FE onset by 14–21%, and nx = 64 moves it by
about 1%.

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

### Head-to-head: which model, when (provisional)

The tables above mostly score each dataset with the one model that
applies to it. Section 4 of `validation/strength_error_summary.py`
(issue #433) instead runs **every applicable model on every strength
dataset**, on the same cases. The FE on Datasets A–E is given exactly
the inputs the analytical model gets: same amplitude, wavelength,
envelope, layup and material, at the default interfaces and
through-thickness decay. Per-case values, with the wrinkle angle, are
in `validation/head_to_head_cases.csv`.

| Dataset | Analytical | Gate | FE LaRC05 (first ply) |
|---|---|---|---|
| A Elhajjar, MD compression (13) | **+0.9%** mean, 5 unsafe, worst +29.6% | — | +4.1%, 7 unsafe, worst +23.8% |
| B Elhajjar, MD tension (7) | **−0.7%**, 3 unsafe, worst +9.7% | — | −16.4%, 3 unsafe, worst +9.2% |
| C Mukhopadhyay, MD compression (3) | **−7.7%**, 0 unsafe | — | +38.7%, 3 unsafe, worst +48.2% |
| C Mukhopadhyay, MD tension (3) | −19.9%, 0 unsafe | — | **−7.5%**, 1 unsafe, worst +5.5% |
| C Mukhopadhyay, onset (3) | −13.9%, 1 unsafe | — | +6.8%, 2 unsafe, worst +19.4% |
| D Wang, MD compression (4) | **−16.3%**, 0 unsafe | — | +46.4%, 4 unsafe, worst +92.3% |
| E Li 2024, UD glass (9) | −37.4%, 0 unsafe | +0.1% *(fitted)* | +30.4%, 9 unsafe, worst +61.2% |
| F Li 2025, UD glass (6) | −58.2%, 0 unsafe | +0.4% *(fitted)* | −2.0%, 2 unsafe, worst +38.7% |
| H Shi, UD carbon (3) | +12.6%, 2 unsafe, worst +25.9% | −19.6% *(blind, moulded preset)* | −34.3%, 0 unsafe |
| H Shi, MD compression (3) | +58.8%, 3 unsafe, worst +115.2% | — | **−5.3%**, 0 unsafe |
| I Calvo, MD tension (1) | **−8.4%** | — | −25.7% |
| M Elhajjar & Shams, MD tension (9) | +11.7%, 8 unsafe, worst +29.6% | — | −49.4% (vs the ultimate); onset: +32.0%, 6 unsafe, worst +134% |
| M Elhajjar & Shams, UD tension (4) | +18.8%, 2 unsafe, worst +39.2% | — | −41.0% (vs the ultimate); onset (3): +68.1%, 3 unsafe, worst +154% |
| J Thor, quasi-isotropic compression (2) | +22.1%, 1 unsafe, worst +52.1% | — | **−6.2%**, 0 unsafe |
| J Thor, UD carbon (2) | −57.9%, 0 unsafe | +75.0% *(blind)*, 2 unsafe, worst +96.7% | **−1.4%**, 1 unsafe, worst +2.2% |
| K Pilato, near-UD carbon (5) | −77.5%, 0 unsafe | +29.7% *(blind)*, 5 unsafe, worst +36.6% | **+4.2%**, 4 unsafe, worst +9.3% |
| *L Krishnappa, wet lay-up UD carbon (18), quarantined* | −55.8%, 0 unsafe, −79% … −7% | +15.5% *(blind)*, 14 unsafe, worst +45.1% | +15.0%, 10 unsafe, worst +99.7% (embedded mid-plane wrinkles) |

Mean signed error; "unsafe" counts cases predicted above the
measurement. "—" means the gate is UD-scoped and has no
multidirectional form. Dataset L is quarantined (issue #445) and does
not feed the guidance below.

Split by severity, multidirectional compression (A, C, D, H, J) reads:

| Peak angle | n | Analytical | FE LaRC05 |
|---|---|---|---|
| < 5° | 6 | −7.8% mean, 0 unsafe | +8.5%, 4 unsafe |
| 5–10° | 6 | −10.7%, 0 unsafe | +23.0%, 5 unsafe |
| ≥ 10° | 13 | +19.6%, **9 unsafe**, worst +115% | +10.6%, **5 unsafe**, worst +92% |

**Provisional guidance.** Confidence reflects the number of independent
sources behind each row, not the size of the errors.

Every row rests on prepreg laminates. No row covers wet lay-up, infusion or RTM: an analysis that states one of those processes carries the out-of-scope warning (issue #445; see "`manufacturing_process`" above).

| Case | Use | Evidence | Process route (as stated) | Confidence |
|---|---|---|---|---|
| Multidirectional compression, peak angle < 10° | Analytical kink-band. FE is unsafe here (9 of 12). | 12 cases, 4 labs | Prepreg: autoclave or hot press (A, J); C, D not stated | **Moderate**  |
| Multidirectional compression, peak angle ≥ 10° | The analytical model is unsafe at high amplitude-to-thickness (Shi up to +115%, Thor +52%). FE is accurate when its mesh reproduces the specimen's wrinkle geometry (Shi −8% to −1%, Thor −9% to −4%) and unsafe when it does not (Wang, Mukhopadhyay, up to +92%). Use FE on the measured profile; otherwise take the lower of the two and add margin. | 13 cases, 5 labs | Prepreg: autoclave or hot press (A, J); C, D, H not stated | **Low**: issues #434, #439  |
| Any compression wrinkle that moves the load path (eccentricity e/t ≥ 0.15: whole-thickness waves, one-sided surface troughs) | FE on the measured profile: never unsafe there (−40% to −1%), while the deployed analytical model is unsafe on 7 of 8. Below e/t = 0.10 the reverse holds. See [the bending-mode investigation](bending_mode_investigation.md). | 8 cases, 3 studies above 0.15; 39 below 0.10 | Prepreg: autoclave or hot press (A, J); H not stated | **Low–moderate**: issue #439  |
| Multidirectional tension, ultimate | Analytical three-mechanism up to about 20°. On steep wrinkles (23–36°, Dataset M) it was unsafe on 8 of 9, by up to +30%: add margin there. | 20 cases, 3 labs; 9 above 20°, from one lab | Prepreg: autoclave or hot press (B, M); C, I not stated | **Moderate** below 20°, **Low** above |
| UD tension, ultimate | Analytical three-mechanism, with margin: within 1% on the smaller wrinkles, +37% to +39% unsafe on the larger (angles estimated). | 4 cases, 1 lab | Prepreg: hot press (M) | **Low** |
| Tension, damage onset | No reliable predictor. First-ply FE is an onset estimate only: unsafe on 9 of 12 Dataset M coupons (−22% to +154%, worst on the deepest wrinkles) and 2 of 3 Mukhopadhyay. | 15 cases, 2 labs | Prepreg: hot press (M); C not stated | **Low** |
| UD glass compression | Penetration gate, on the material it was fitted to. Kink-band is far too conservative (−37% to −58%); FE ignores wrinkle depth. | 15 cases, 1 group, in-sample | Prepreg: hot press (E), vacuum bag (F) | **Fit only**  |
| UD carbon compression (not wet lay-up) | FE on the specimen's measured geometry: −5% to +9% on Thor and Pilato, −26% to −40% (conservative) on Shi. The kink-band is conservative except on Shi's one-sided troughs (+0.2% to +26%). Never the gate presets as shipped: unsafe on 7 of 10, up to +97%. Re-fitting the moulded preset's `gamma_Y` to one wrinkled test is unproven: within 2–4% on Shi and Pilato, −29% / +15% on Thor's two cases. For an embedded wrinkle with flat surfaces, FE has had unsafe misses on every dataset with one (C, D, E, F). Wet lay-up is outside this row (issue #445): on the quarantined Dataset L, FE was up to +100% unsafe and the one-test gate up to +71%. | 10 cases, 3 labs | Prepreg: autoclave (J); H, K not stated | **Low**: issues #435, #445  |

Caveats on the FE column:
- The A–E FE recipe uses the analytical inputs at the default interfaces
  and decay. It is not tuned per specimen the way Dataset H's one-sided
  `tool_flat` recipe or Dataset J's whole-thickness wave is, and those
  two are the multidirectional datasets where the FE ran conservative.
- On Dataset D the FE gives the same retention for the convex and
  concave pairs, so it does not reproduce the morphology difference that
  Wang measured (0.677 vs 0.419 at A = 0.76 mm).
- Mesh-checked on one case each of A, B, C, D, J and K: nx = 72, two
  elements per ply or ny = 4 move the retention by at most 4 points, and
  on A–D refinement makes it *less* conservative.
- In tension the FE is a first-ply quantity. Against a measured ultimate
  (B, C tension, I, M) its conservatism is partly by construction;
  against a measured damage onset (Dataset M) it was unsafe on 9 of 12
  coupons, −22% to +154%.

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
