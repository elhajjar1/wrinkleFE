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
optional Argon–Fleck quadratic coefficient). A UD-dominated laminate (at
least 75% 0° plies) on any morphology except `graded` takes its strain
budget from the material card instead, `gamma_Y_eff = S12 / Xc`
(issue #451; see "UD laminates: which kink-band budget" below). In
tension it is the
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
  material card lacks both `GIc` and `GIIc`. It is **unvalidated**
  (issue #463): against Dataset M's measured onset it was above the
  measurement on all nine multidirectional coupons, by up to about
  240%, and 95% to 99% below it on the UD ones. The results JSON
  flags it (`analytical_onset_validated: false` and
  `analytical_onset_caveat`, schema 1.7);
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

### UD laminates: which kink-band budget

Without a gate, the compression knockdown of a UD-dominated laminate (at
least 75% 0° plies) depends on which kink-band strain budget its
morphology selects (issue #451):

| Morphology | Budget (`gamma_Y_eff`) | On validation data |
|---|---|---|
| any except `graded` (peak-angle cross-section) | `S12 / Xc` from the material card | Thor and Pilato whole-thickness waves: −10% to −23%, never unsafe; Shi's one-sided troughs as `tool_flat`: −59% to −71% |
| `graded` (averaged over the wrinkle profile) | the layup model's 0.032 for UD | Hsiao & Daniel −22%; Krishnappa embedded −7% to −34% (quarantined); Shi's one-sided troughs entered as `graded` (the wrong shape, the ledger recipe before issue #461): +0.2% to +26% |

- **The card's `S12` and `Xc` now set the UD answer** on the peak-angle
  path. Use measured values for the material: a card whose `Xc` is a
  vendor value well above the coupon's pristine strength gives too small
  a budget, which errs conservative.
- **Pick the morphology that matches the wrinkle.** `uniform` for a wave
  through the whole thickness; `graded` for an embedded wrinkle with
  flat surfaces; `tool_flat` for a one-sided surface trough (see
  "One-sided surface troughs" below). `graded` uses a different budget
  and a different angle measure (profile-averaged against peak), so on
  UD it can differ substantially from the other two.
- `AnalysisResults.gamma_Y_eff` reports the budget the run used.
  Multidirectional laminates (at most 40% 0° plies in the validation
  set) keep the layup confinement model on every path.

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
| One-sided surface trough over a flat tool face (`tool_flat`) | about half the trough depth |
| Embedded wrinkle with flat outer surfaces | close to zero (exactly zero on UD) |

The offset is measured on the modelled geometry, so a one-sided trough
shows up only if it is modelled with one surface free: as `tool_flat`,
not `graded` (see "One-sided surface troughs" below).

`bending_dominated` is `True` for a **compression** analysis with
e/t ≥ 0.10. At that point the section bends as well as kinks, and the
angle-based kink-band knockdown does not represent that.

On measured cases (issue #439):
- **Below 0.10** (39 cases), the deployed analytical model was never more
  than 15% unsafe.
- **Above 0.15** (8 cases), it was unsafe on 7, by up to +115%, with
  Shi's one-sided troughs scored as `graded`. On their `tool_flat`
  geometry (issue #461) the kink-band is unsafe on 6 of the 8, by up to
  +52% (Thor's wave 1; +15% with the bending bound below). First-ply FE
  on the specimen's own wrinkle geometry was never unsafe there.

### `governing_bound` — the bending bound on whole-thickness waves

What happens next depends on whether the wave runs through the whole
thickness. `whole_thickness_wave` is `True` when **both** outer surfaces
follow the load path: at the section where the offset e is largest, each
surface has moved, the same way, by at least half of e. It is measured
on the same generated ply boundaries as e, so it works for every
morphology, decay setting and multi-wrinkle layout:

| Wrinkle | Surfaces at the offset | `whole_thickness_wave` |
|---|---|---|
| `uniform`, or `graded` with a `decay_floor` that keeps the surfaces wavy | both move by about e | `True` |
| `tool_flat` (a face pinned flat by the tool: a one-sided trough) | one moves about 2e, the pinned one 0 | `False` |
| `graded` (floor 0), `stack`, `convex`, `concave`, `tool_flat` pinned on both faces, multi-wrinkle specs | both flat | `False` |

**A bending-dominated whole-thickness wave** (`bending_bound_applied`)
is treated as an eccentric column (issue #452). The far-field load runs
a distance e from the centroid of the section at the wrinkle, whose
thickness is t_c, so the outer fibres reach the pristine strength at

```text
bending_knockdown = (t_c / t_0) / (1 + 6 e / t_c)
```

and `analytical_knockdown` becomes the lower of `kink_band_knockdown`
(the angle-based value, or the penetration gate's when one is
configured) and `bending_knockdown`. `governing_bound` says which one
won: `"kink_band"` or `"bending"`. On the measured whole-thickness waves
in this regime:

| Case | e/t | Measured | Kink-band | With the bound |
|---|---|---|---|---|
| Thor quasi-isotropic, wave 1 (J-QI-1) | 0.25 | 0.347 | +52.1% | **+15.1%** (bending governs) |
| Elhajjar A = 0.73 mm | 0.31 | 0.320 | +29.6% | **+10.4%** (bending governs) |
| Elhajjar A = 0.61 mm | 0.25 | 0.350 | +18.5% | **+13.1%** (bending governs) |
| Elhajjar A = 0.24–0.49 mm (3 cases) | 0.10–0.20 | 0.37–0.47 | +1.0% … +12.1% | unchanged (kink-band governs) |
| Thor UD, wave 1 (J-UD-1) | 0.25 | 0.252 | −10.0% (UD budget, issue #451) | unchanged (kink-band governs) |

The bound is a screening bound, not a validated allowable. It is
homogeneous-section beam theory with no column-stability (P-delta)
amplification, and it still leaves Thor's wave 1 15% unsafe. Check it
against FE on the measured wrinkle profile.

**A bending-dominated wrinkle that is not a whole-thickness wave** (a
one-sided trough, like Shi's Dataset H, modelled as `tool_flat`) gets no bound:
the eccentric-column formula did not explain its measured knockdown,
which points to a local effect under the trough. The kink-band knockdown
stands, `governing_bound` is `"kink_band"`, and the advice is to prefer
FE on the measured profile. The trough has to be modelled with its free
surface for any of this to happen: see the next section.

Nothing changes below e/t = 0.10: the bound is not consulted there. It
switches on at the threshold, so a whole-thickness wave whose kink-band
knockdown is above 1/1.6 = 0.625 at e/t = 0.10 (a long wave with a small peak
angle) steps down as it crosses it; no validation case sits
near that step. In tension `governing_bound`,
`kink_band_knockdown` and `bending_knockdown` are `None` and nothing
changes: an axial pull straightens the wave rather than bending it
further, and the evidence is compression only. Tension analyses still
report e/t and `whole_thickness_wave`.

The text summary (a `Governing bound:` line and the bending warning),
the results JSON (`analytical` block, schema 1.5), the NCR summary and
the app carry the same information. The warning text differs by case:
`BENDING_BOUND_CAVEAT` when the bound was applied,
`BENDING_REGIME_CAVEAT` (prefer FE) when it was not
(`wrinklefe.core.eccentricity`).

### One-sided surface troughs: model them as `tool_flat`

`bending_dominated` is measured on the geometry you modelled, not on the
part (issue #461). Choose the morphology by what the outer surfaces do:

| Wrinkle | Outer surfaces | Morphology |
|---|---|---|
| Embedded | both flat | `graded` (decay floor 0), or `stack` / `convex` / `concave` |
| A wave through the whole thickness | both wavy | `uniform` |
| A one-sided surface trough (or hump) over a flat tool face | one follows the wrinkle, the tool face stays flat | `tool_flat`, `surface_pocket_side` set to the tool face |

**Do not model a one-sided trough as `graded`.** `graded` with
`decay_floor=0` keeps both outer surfaces flat, which is what an
embedded wrinkle is. On a UD laminate the centroid of a section with
flat surfaces sits exactly midway between them, so e/t is 0 at any
amplitude. On a multidirectional laminate only the shift of stiff plies
against compliant ones moves it. Shi's troughs (Dataset H) show the gap.
The validation ledger itself scored them as `graded` until issue #461
corrected its recipe to `tool_flat`, after these errors were known,
because the specimens are one-sided (see
[Validation](validation.md), "Dataset H"). Errors are against the
measured knockdown; positive is unsafe:

| Shi case (depth / thickness) | `graded` (the ledger recipe before issue #461) | `tool_flat` (flat bottom, 10-ply ramp; the ledger recipe now) |
|---|---|---|
| H-UD-10 (0.10) | e/t 0.000, not flagged, +0.2% | e/t 0.050, not flagged, −58.9% |
| H-UD-20 (0.20) | e/t 0.000, not flagged, +11.8% | e/t 0.100, **flagged**, −68.4% |
| H-UD-30 (0.30) | e/t 0.000, not flagged, +25.9% | e/t 0.150, **flagged**, −71.0% |
| H-MD-10 (0.10) | e/t 0.012, not flagged, +5.8% | e/t 0.058, not flagged, −42.1% |
| H-MD-20 (0.20) | e/t 0.022, not flagged, +55.6% | e/t 0.117, **flagged**, −22.4% |
| H-MD-30 (0.30) | e/t 0.030, not flagged, +115.2% | e/t 0.175, **flagged**, +5.0% |

The 10% troughs are not flagged on either morphology. On `graded` the
analytical model was within +6% on them; on `tool_flat` it is 42% to
59% conservative.

**Why the kink-band errs so far on the conservative side here.** On
one-sided troughs it is usually excessively conservative, for two
reasons:
- It applies the trough's peak misalignment angle to the whole cross-section
  (on the 20% UD trough the effective angle equals the 23.5° peak),
  although the misalignment fades to nothing toward the flat tool face.
- It does not credit the straighter plies near that face with the load
  the wavy plies shed: on a UD section every ply is knocked down alike.

First-ply FE on the measured profile resolves both effects and is
closer: −8% to −1% on the multidirectional half, −26% to −40% on the UD
half. The kink-band is not always conservative here: the deepest
multidirectional trough is 5% unsafe. The warning a flagged trough
gets says all of this, and still advises FE on the measured profile.

Other choices for the same troughs:
- `uniform` flags all six (e/t 0.10 to 0.30), but as whole-thickness
  waves, so it applies the eccentric-column bound, which was not
  validated on troughs. The kink-band governs there, with the same
  errors as `tool_flat`.
- `graded` with `wrinkle_z_position` at exactly 1.0 (or 0.0) centres the
  decay on a surface and leaves that surface wavy. That flags the 20% and
  30% troughs (e/t 0.10 to 0.16), but the knockdown stays the `graded`
  one (+13% to +117%). Any position strictly between 0 and 1 keeps both
  surfaces flat and flags nothing.
- `tool_flat` with `surface_pocket_side="both"` pins both faces flat, so
  it reads like `graded`. So do `stack`, `convex` and `concave`.

**Why the trough is not detected on `graded`.** A trough entered as
`graded` has the inputs of an embedded wrinkle, and measured embedded
wrinkles cover the same ranges:
- amplitude / thickness: Krishnappa's mid-plane L-WM-4-3, 0.207; Shi's
  20% trough, 0.200;
- amplitude / wavelength and peak angle: Li's S-M-3, 0.093 and 30°;
  Shi's 20% trough, 0.069 and 23.5°.

Any threshold that warned on Shi's 20% troughs would also warn on
embedded wrinkles, which are not bending-dominated. The one thing that
separates the two is whether an outer surface follows the wrinkle, and
the morphology is where you state that. The app's **Morphology** help
and the CLI's `--morphology` help say so
(`wrinklefe.core.eccentricity.ONE_SIDED_TROUGH_ADVICE`).

**Setting up `tool_flat` for a trough:**
- `amplitude` is the trough depth.
- `surface_pocket_side` is the face that stays flat on the tool.
- `surface_transition_plies` is about the number of plies the trough
  fades out over (10 of 20 for Shi). It also bounds the amplitude:
  `amplitude ≤ 0.8 · surface_transition_plies · ply_thickness / nz_per_ply`.
  The default of 2 rejects a trough deeper than 1.6 plies, and the error
  message names the ply count needed.
- On a UD laminate e/t is depth / (2 · thickness), so the flag comes on
  at a depth of 20% of the thickness. Shi's 20% UD trough sits on that
  line (e/t 0.10001), so a slightly shallower one is not flagged. On
  Shi's multidirectional layup the flag comes on a little earlier.

What you then get is the `BENDING_REGIME_CAVEAT` warning (no bound,
prefer FE) and the peak-angle kink-band knockdown, which `tool_flat`
shares with `uniform`. On Shi's troughs that knockdown is conservative
on five of six cases (−22% to −71%) but +5.0% on H-MD-30; these are the
errors the validation now scores. First-ply FE on this geometry was −40%
to −1%. Follow the warning and run FE on the measured profile.

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
  dataset (Dataset L, quarantined): the kink-band was about 32%
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

The results JSON (`validation_margin` block, schema 1.6) and the NCR
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
  geometry, which is the case's knockdown recipe. For Dataset H's
  one-sided troughs that is `tool_flat` since issue #461 (before it, a
  separate `eccentricity_recipe` supplied that geometry, because the
  knockdown recipe was an embedded profile).

**Fitting method, and why not a normal.** The residuals are fat-tailed
and often multi-modal. On the kink-band UD cases measured / predicted
runs from 1.1 to 3.4. On the multidirectional tension cases it falls in
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
| kink-band · compression · UD-dominated · not bending-dominated | 14 | **1.00** | 1.188 | 0.961 | 1.200 |
| kink-band · compression · multidirectional · not bending-dominated | 17 | **1.00** | 1.001 | 0.917 | 1.017 |
| three-mechanism · tension · multidirectional | 20 | **0.896** | 0.896 | 0.885 | 0.876 |
| kink-band · compression · UD-dominated · bending-dominated | 3 | insufficient evidence | — | — | 1.111 |
| kink-band · compression · multidirectional · bending-dominated | 8 | insufficient evidence | — | — | 0.869 |
| penetration gate · compression · UD-dominated | 6 | insufficient evidence | — | — | 0.866 |
| three-mechanism · tension · UD-dominated | 4 | insufficient evidence | — | — | 0.850 |
| progressive damage (crack band) · compression · UD-dominated | 3 | insufficient evidence | — | — | 0.765 |
| FE first-ply retention (LaRC05), any regime | 0 | insufficient evidence | — | — | — |

Reading it:

- **Multidirectional compression that is not bending-dominated, the
  most common case in practice, has a margin of 1.00** (issue #462
  brought Datasets A, C and D into the ledger: 17 cases from five labs,
  2 before). The kink-band was conservative on every one: measured /
  predicted runs from 1.017 (Wang's deeper concave wrinkle) to 1.73
  (Shi's 10% trough), and 13 of the 17 sit between 1.017 and 1.12. The
  mixture's 5% quantile is 1.001, just above 1, so the cap binds. A
  normal fit would put it at 0.917, because the few far-safe cases
  (Wang's other three and Shi's trough, 1.20 to 1.73) widen one bell.
  A margin of 1.00 says no case in this regime has been unsafe yet, not
  that the prediction is exact; the cases sit close to parity, so one
  unsafe case could pull it below 1. Most of them rest on assumed
  wavelengths (Elhajjar's and Mukhopadhyay's amplitude-scaled rules,
  Wang's fixed 24 mm).
- **Bending-dominated multidirectional compression still has no
  margin, and is unsafe on 7 of its 8 cases.** Elhajjar's five largest
  whole-thickness waves joined Thor's wave 1 and Shi's two deeper
  troughs. Measured / predicted is 0.869 to 0.990 on all but H-MD-20
  (1.29). That is what the bending warning says: run FE on the measured
  profile and carry a margin of your own.
- **The kink-band UD margin is 1.00** because the kink-band stays
  conservative on the UD sets: measured / predicted is 1.2 to 1.3 on
  Thor and Pilato with the UD strain budget of issue #451 (1.9 to 4.7
  before it), 1.9 to 3.2 on Li (2025) and 2.4 on Shi's 10% trough
  (H-UD-10, level at 0.998 on the `graded` recipe it had before issue
  #461). No case is below 1.2, yet a normal fit would put the 5%
  quantile below 1 (0.961), because the wide spread of ratios reads as
  one wide bell. The mixture puts it at 1.188, so the margin is capped
  at 1.
- **The tension multidirectional margin is 0.896** (0.881 before issue
  #462 added Elhajjar's tension cases and Mukhopadhyay's ultimates: 20
  cases from three labs, 10 before). Six of the 20 are unsafe (ratio
  0.876 to 0.969), and M-MD-2 (0.876) is the one still below the margin.
  It sits above the normal's 0.885 because the residuals form two
  clusters, near parity and about 17% on the safe side, not one wide
  bell.
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
multidirectional laminates in compression it has erred both ways: −9% to
−1% where the mesh reproduces the specimen's own wrinkle (Shi et al.
2025 and Thor et al. 2021, Datasets H and J), but over-predicting
retained strength by up to +92% on Datasets A, C and D (see
[How far to trust each number](#how-far-to-trust-each-number) and
[the head-to-head](#head-to-head-which-model-when-provisional)). For an
embedded wrinkle under flat surfaces no FE strength path is offered
(issue #453). The app and the NCR summary print this warning next to the
number.

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
- `czm_load_displacement` — the `(λ, ‖u‖)` increment samples;
- `czm_damage_history` — one record per converged load increment
  (issue #453): the carried nominal stress (reaction over the loaded
  face's section, the progressive-damage solver's measure), maximum and
  mean damage, the damaged (`d > 0`) and fully separated (`d = 1`)
  interface area, and the *initiation index* (largest
  `delta_eff / delta_0` over the interface, ≥ 1 once a point has reached
  its mixed-mode interlaminar strength). Two onset readings:
  `czm_damage_history.initiation()` is the load at which the first
  interface point reaches that strength — exact and independent of the
  increment size without a thermal load, because the model is linear
  until then (one elastic increment is enough; a ramp that never reaches
  it is extrapolated and flagged `extrapolated`);
  `czm_damage_history.separation()` is the first increment with a
  traction-free point, a crack, resolved to one increment and dependent
  on how finely the mesh resolves the cohesive zone.

The CZM path rebuilds the mesh with duplicated interface nodes. It
carries over the resin-zone tags the linear path computed (the crest
resin lens, a `tool_flat` laminate's surface pockets), so those zones
are in the cohesive solve and `tool_flat` combines with `enable_czm`.

The delamination-onset load is **not a strength prediction** for an
embedded wrinkle: see the scope note under
[How far to trust each number](#how-far-to-trust-each-number).

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
  the kink-band knockdown is outside its validated regime. On a
  whole-thickness wave `analytical_knockdown` already carries the
  eccentric-column bending bound (`governing_bound`), which brought the
  measured cases to within about +15%; treat it as a screening number.
  On a one-sided trough there is no bound. Either way, run FE on the
  measured wrinkle profile and carry the lower number, with margin. A
  one-sided trough is flagged only when it is modelled as `tool_flat`;
  as `graded` it reads e/t ≈ 0 (see "One-sided surface troughs" above).
- **Unidirectional laminates in compression** — the angle-only knockdown
  is scale-invariant and under-predicts the penetration effect. Use the
  penetration gate. `progressive_knockdown` is the only FE number that
  reaches past first-ply failure here, but it is mesh-calibrated and
  over-predicts severe wrinkles, so use it to study the mechanism, not
  to set a knockdown. `retention_factors` will not help here.
- **Embedded wrinkles under flat surfaces** — no FE strength number
  governs, in either loading: the first-ply FE and both size-aware
  candidates (progressive peak, CZM delamination onset) were unsafe on
  every such dataset (issue #453). Carry the analytical knockdown.
- **Tension with a delamination concern** — `analytical_knockdown` is
  the ultimate. No model predicts the first load drop reliably, so a
  requirement written against onset has no validated number to carry.
  Do not use `analytical_onset_knockdown` (or the critical-value
  search's `onset_knockdown` objective) to set it: it is unvalidated,
  and unsafe or wildly low on Dataset M (issue #463). The closest
  estimate is the FE cohesive-zone onset of issue #453, conservative
  on average on Dataset M (−23%) but up to +52% unsafe on the
  deepest wrinkle; a real onset model is issue #465.
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
it; 61 of the 74 analytical strength cases fall within ±20%, the
headline parity figure (`validation/fig_all_validation_parity.png`).
Mukhopadhyay's three analytical onset estimates are listed below but
scored with the first-ply FE series in the second figure
(`validation/fig_fe_onset_parity.png`), because onset is a different
measured quantity. Dataset L is **quarantined** (issue #445): a low-quality wet
lay-up laminate, listed for reference and left out of every count here.

| Dataset | Model | n | Mean abs. error | Range | Non-conservative |
|---|---|---|---|---|---|
| A Elhajjar (2025), compression | Budiansky–Fleck, bending bound | 13 | 7.6% | −9.1% … +13.1% | 5 / 13 |
| B Elhajjar (2025), tension | three-mechanism | 7 | 6.2% | −15.7% … +9.7% | 3 / 7 |
| C Mukhopadhyay (2015), compression | Budiansky–Fleck | 3 | 7.7% | −10.8% … −5.2% | 0 / 3 |
| C Mukhopadhyay (2015), tension | three-mechanism | 3 | 12.2% | −15.8% … −7.1% | 0 / 3 |
| C Mukhopadhyay (2015), delam. onset | three-mechanism onset | 3 | 15.2% | −22.5% … +1.9% | 1 / 3 |
| D Wang (2021), concave/convex | Budiansky–Fleck + morphology | 4 | 16.3% | −27.4% … −1.6% | 0 / 4 |
| E Li (2024), UD | penetration gate | 9 | 2.8% | −6.6% … +8.5% | 4 / 9 |
| F Li (2025), UD | penetration gate + position | 6 | 5.0% | −12.9% … +14.6% | 2 / 6 |
| H Shi (2025), UD CFRP, one-sided troughs (`tool_flat`) | Budiansky–Fleck, UD budget S12/Xc (no gate preset) | 3 | 66.1% | −71.0% … −58.9% | 0 / 3 |
| H Shi (2025), multidirectional CFRP, one-sided troughs (`tool_flat`) | Budiansky–Fleck | 3 | 23.2% | −42.1% … +5.0% | 1 / 3 |
| I Calvo (2023), multidirectional CFRP, tension, 7 distributed waves | three-mechanism | 1 | 8.4% | −8.4% | 0 / 1 |
| M Elhajjar & Shams (2016), multidirectional T700GC/2510, tension, steep rod-made wrinkles (23–36°) | three-mechanism | 9 | 10.8% | −21.9% … +14.2% | 3 / 9 |
| M Elhajjar & Shams (2016), UD T700GC/2510, tension (22–27°) | three-mechanism | 4 | 19.7% | −22.7% … +17.6% | 2 / 4 |
| J Thor (2021), quasi-isotropic IM7/8552, whole-thickness waves | Budiansky–Fleck, bending bound | 2 | 11.5% | −7.9% … +15.1% | 1 / 2 |
| J Thor (2021), UD IM7/8552 | Budiansky–Fleck, UD budget S12/Xc (no gate preset) | 2 | 16.6% | −23.2% … −10.0% | 0 / 2 |
| K Pilato (2022), near-UD industrial wrinkles | Budiansky–Fleck, UD budget S12/Xc (no gate preset) | 5 | 20.3% | −22.9% … −16.7% | 0 / 5 |
| *L Krishnappa & Gururaja (2024), wet lay-up UD carbon, 18 strip-induced wrinkles (quarantined)* | Budiansky–Fleck (no gate preset; UD budget S12/Xc on the 12 WL/WR cases) | 18 | 31.7% | −48.0% … −6.9% | 0 / 18 |

All 13 analytical strength cases outside ±20% are conservative. Four
are the kink-band on UD carbon (J, K), just outside the band (−22% to
−23%), three are Dataset M tension coupons (−22% to −23%), five are
Shi's one-sided troughs (H, −22% to −71%) and one is Wang's (D, −27%).
Two of Mukhopadhyay's onset estimates are also outside, at −22% and
−21%. Before the UD strain budget was
anchored to the material card (issue #451) the kink-band was 57–79%
conservative on all seven J UD and K cases. The quarantined Dataset L
adds fifteen more conservative ones.

None is more than 20% unsafe. The last three were Dataset H's severe
cases, +25.9% on the UD half and +55.6% / +115.2% on the
multidirectional half, where
buckling takes part. Those errors came from modelling the specimens'
one-sided troughs as an embedded `graded` wrinkle. Issue #461 corrected
the recipe to `tool_flat`, after those errors were known, because that
is the trough's geometry (see "One-sided surface troughs" above). On it
the analytical path is conservative except H-MD-30 (+5.0%), and the 20%
and 30% troughs are flagged bending-dominated, with the advice to run
FE; the bending bound does not cover a one-sided trough. (Dataset M's
steep tension wrinkles were six more unsafe cases, +20% to +39%, until
issue #454: the compression floor, not the tension model, was setting
those predictions; see the guidance table below.)

The whole-thickness waves that used to be on this list now carry the
eccentric-column bending bound (issue #452; see "`governing_bound`"
above): the most severe Dataset A wrinkle went from +29.6% to +10.4%
(A = 0.73 mm, measured knockdown 0.32, predicted 0.35), and Thor's severe
quasi-isotropic wave, which fails by bending of the wavy coupon, from
+52.1% to +15.1%. All of these sit at high amplitude-to-thickness ratios,
where the failure mode is not the kinking the kink-band model represents
(issue #439). On Thor's waves and Shi's multidirectional troughs the FE
retention, run on the specimen's own geometry, is within −9% to −1%
(tables below). On the other multidirectional compression
datasets, with generic recipes, the FE misses on the unsafe side: see
[the head-to-head](#head-to-head-which-model-when-provisional).

**FE strength (LaRC05)** — Li (2025) UD glass/epoxy, the FE path's
first measured-strength check (Dataset H below was the second; the
head-to-head further down scores the others). Wrinkled over pristine
strength at first failure:

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

### Size-aware FE candidates, embedded flat-surface wrinkles (issue #453)

The first-ply FE reads a wrinkle's angle but not its size, and ignores
the delamination that starts at the wrinkle; that is why it over-predicts
retained strength on embedded wrinkles under flat surfaces. Two
candidates that do see size were scored on the same cases and meshes as
the head-to-head (`validation/size_aware_fe_study.py`; the numbers are
pinned as `size_aware_fe` in the validation ledger):

- **(a) progressive peak** — the peak carried stress of the ply-discount
  progressive-damage solve exactly as `enable_progressive_damage` runs it
  (ramp to 1.8 × `Xc / E1`, 15 increments, residual stiffness 0.1), over
  the same solver's pristine coupon;
- **(b) CZM delamination onset** — the load at which the first point of
  any ply interface reaches its mixed-mode interlaminar strength
  (`czm_damage_history.initiation()`, cohesive layers at every
  interface), over the pristine progressive peak and capped at 1.
  Threshold: the bilinear law's own initiation condition, which adds no
  free parameter and is exact and increment-independent because the
  model is linear until then. A later threshold (a fully separated
  point) could not be reached: the Newton solve stops converging one to
  three increments past initiation on these meshes.

Signed error against the measurement; "unsafe" = above +0.5%:

| Dataset (n) | First-ply FE | (a) progressive peak | (b) CZM onset |
|---|---|---|---|
| C Mukhopadhyay, MD compression (3) | +38.7% mean, worst +48.2%, 3 unsafe | +38.8%, worst **+49.0%**, 3 unsafe | +39.4%, worst **+49.3%**, 3 unsafe |
| D Wang, MD compression (4) | +46.4%, worst +92.3%, 4 unsafe | +57.9%, worst **+109.3%**, 4 unsafe | +70.3%, worst **+138.7%**, 4 unsafe |
| E Li 2024, UD glass (9) | +30.4%, worst +61.2%, 9 unsafe | −12.9%, worst **+6.3%**, 2 unsafe | +4.9%, worst **+25.3%**, 6 unsafe |
| *L-WM Krishnappa, wet lay-up UD, embedded (6), quarantined* | +61.1%, worst +99.7%, 6 unsafe | +42.0%, worst **+64.2%**, 6 unsafe | +53.6%, worst **+74.1%**, 6 unsafe |
| M Elhajjar & Shams, tension damage onset, MPa (12) | +41.0%, worst +154%, 9 unsafe | +79.1%, worst **+201.6%**, 12 unsafe | −23.3%, worst **+52.2%**, 2 unsafe |

Neither measure reaches a worst case of +15% on these sets, and
neither tracks wrinkle size: on Dataset E's six fixed-angle cases
(15.8–16.7°, depth ratio 0.026–0.083) the measured knockdown falls from
0.76 to 0.52 while every FE measure moves by less than 0.04 — the peak
load slightly *rises* with depth. Its one within-bar score (E) comes from
a conservative offset (−39% on the mildest 16° case), not from a size
response. On the multidirectional coupons (C, D) the CZM onset sees the
coupon, not the wrinkle: the interface reaches its strength at the free
edges of a near-flat coupon at 459 MPa (C) and 1486 MPa (D), above the
pristine coupon's own peak, so every knockdown caps at 1. On Dataset M
the wrinkle does govern (a near-flat coupon initiates at 312 MPa, above
every wrinkled value) and the onset is conservative on average, but
still +52% on the deepest 30° dip, and a peak load is not an onset at
all (+202%). Mesh: the CZM onset moves by at most 4% when `nx` or
`nz_per_ply` is doubled on C, D, L-WM and M; E was scored at `nx = 48`
(E-7: 656/549/526/521/517 MPa at `nx` 16/32/48/64/96). The
multidirectional progressive runs (C, D, M and their pristine coupons)
report `converged = False`: the equilibrium loop hit its iteration cap
on post-peak increments.

**Scope exclusion.** For an embedded wrinkle under flat surfaces
WrinkleFE offers no FE strength path: the analytical knockdown is the
only strength number to carry, with margin. `FE_STRENGTH_CAVEAT` says
so. The per-increment CZM damage record (`czm_damage_history`) stays
available as a delamination diagnostic, not as a strength prediction.

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
| A Elhajjar, MD compression (13) | **−1.0%** mean, 5 unsafe, worst +13.1% | — | +4.1%, 7 unsafe, worst +23.8% |
| B Elhajjar, MD tension (7) | **−0.7%**, 3 unsafe, worst +9.7% | — | −16.4%, 3 unsafe, worst +9.2% |
| C Mukhopadhyay, MD compression (3) | **−7.7%**, 0 unsafe | — | +38.7%, 3 unsafe, worst +48.2% |
| C Mukhopadhyay, MD tension (3) | −12.2%, 0 unsafe | — | **−7.5%**, 1 unsafe, worst +5.5% |
| C Mukhopadhyay, onset (3) | −13.9%, 1 unsafe | — | +6.8%, 2 unsafe, worst +19.4% |
| D Wang, MD compression (4) | **−16.3%**, 0 unsafe | — | +46.4%, 4 unsafe, worst +92.3% |
| E Li 2024, UD glass (9) | −37.4%, 0 unsafe | +0.1% *(fitted)* | +30.4%, 9 unsafe, worst +61.2% |
| F Li 2025, UD glass (6) | −58.2%, 0 unsafe | +0.4% *(fitted)* | −2.0%, 2 unsafe, worst +38.7% |
| H Shi, UD carbon (3) | −66.1%, 0 unsafe | −19.6% *(blind, moulded preset)* | −34.3%, 0 unsafe |
| H Shi, MD compression (3) | −19.8%, 1 unsafe, worst +5.0% | — | **−5.3%**, 0 unsafe |
| I Calvo, MD tension (1) | **−8.4%** | — | −25.7% |
| M Elhajjar & Shams, MD tension (9) | **−4.9%**, 3 unsafe, worst +14.2% | — | −49.4% (vs the ultimate); onset: +32.0%, 6 unsafe, worst +134% |
| M Elhajjar & Shams, UD tension (4) | **−2.8%**, 2 unsafe, worst +17.6% | — | −41.0% (vs the ultimate); onset (3): +68.1%, 3 unsafe, worst +154% |
| J Thor, quasi-isotropic compression (2) | +3.6%, 1 unsafe, worst +15.1% | — | **−6.2%**, 0 unsafe |
| J Thor, UD carbon (2) | −16.6%, 0 unsafe | +75.0% *(blind)*, 2 unsafe, worst +96.7% | **−1.4%**, 1 unsafe, worst +2.2% |
| K Pilato, near-UD carbon (5) | −20.3%, 0 unsafe | +29.7% *(blind)*, 5 unsafe, worst +36.6% | **+4.2%**, 4 unsafe, worst +9.3% |
| *L Krishnappa, wet lay-up UD carbon (18), quarantined* | −31.7%, 0 unsafe, −48% … −7% | +15.5% *(blind)*, 14 unsafe, worst +45.1% | +15.0%, 10 unsafe, worst +99.7% (embedded mid-plane wrinkles) |

Mean signed error; "unsafe" counts cases predicted above the
measurement. "—" means the gate is UD-scoped and has no
multidirectional form. Dataset L is quarantined (issue #445) and does
not feed the guidance below.

Split by severity, multidirectional compression (A, C, D, H, J) reads:

| Peak angle | n | Analytical | FE LaRC05 |
|---|---|---|---|
| < 5° | 6 | −7.8% mean, 0 unsafe | +8.5%, 4 unsafe |
| 5–10° | 6 | −10.7%, 0 unsafe | +23.0%, 5 unsafe |
| ≥ 10° | 13 | −3.3%, **7 unsafe**, worst +15% | +10.6%, **5 unsafe**, worst +92% |

**Provisional guidance.** Confidence reflects the number of independent
sources behind each row, not the size of the errors.

Every row rests on prepreg laminates. No row covers wet lay-up, infusion or RTM: an analysis that states one of those processes carries the out-of-scope warning (issue #445; see "`manufacturing_process`" above).

| Case | Use | Evidence | Process route (as stated) | Confidence |
|---|---|---|---|---|
| Multidirectional compression, peak angle < 10° | Analytical kink-band. FE is unsafe here (9 of 12). | 12 cases, 4 labs | Prepreg: autoclave or hot press (A, D, J); C not stated | **Moderate**  |
| Multidirectional compression, peak angle ≥ 10° | The analytical model is unsafe at high amplitude-to-thickness: on whole-thickness waves the bending bound (issue #452) holds Thor to +15% and Elhajjar to +13% (+52% and +30% without it). On Shi's one-sided troughs, modelled as `tool_flat` (issue #461), it is −42% to +5% and flags the 20% and 30% troughs as bending-dominated; modelled as `graded` (an embedded wrinkle, the wrong shape) it was up to +115% unsafe with no warning. FE is accurate when its mesh reproduces the specimen's wrinkle geometry (Shi −8% to −1%, Thor −9% to −4%) and unsafe when it does not (Wang, Mukhopadhyay, up to +92%). Use FE on the measured profile; otherwise take the lower of the two and add margin. | 13 cases, 5 labs | Prepreg: autoclave or hot press (A, D, J); C, H not stated | **Low**: issues #434, #439  |
| Any compression wrinkle that moves the load path (eccentricity e/t ≥ 0.15: whole-thickness waves, one-sided surface troughs) | FE on the measured profile: never unsafe there (−40% to −1%), while the deployed analytical model is unsafe on 6 of 8. With the eccentric-column bending bound on the whole-thickness waves (issue #452), 1 of the 8 stays more than 15% unsafe (3 without it): Thor's wave 1 at +15.1%. Shi's two one-sided troughs here, which the bound does not cover, are −71% and +5% on their `tool_flat` geometry (+26% and +115% scored as `graded`, before issue #461). Below e/t = 0.10 the analytical model is the better choice. e/t is measured on the modelled geometry: enter a one-sided trough as `tool_flat`, with the tool face pinned. As `graded`, the recipe Shi's knockdowns were scored on before issue #461, it reads e/t ≈ 0 and gets no warning (see "One-sided surface troughs" above). See [the bending-mode investigation](bending_mode_investigation.md). | 8 cases, 3 studies above 0.15; 39 below 0.10 | Prepreg: autoclave or hot press (A, J); H not stated | **Low–moderate**: issues #439, #452, #461  |
| Multidirectional tension, ultimate | Analytical three-mechanism. On steep wrinkles (23–36°, Dataset M) it is −22% to +14% (mean −4.9%, unsafe on 3 of 9); before issue #454 the compression floor set those predictions and was unsafe on 8 of 9, by up to +30%. The #454 corrections carry no fitted constant but were diagnosed on these coupons, so above 20° the agreement is in-sample: keep a margin there. | 20 cases, 3 labs; 9 above 20°, from one lab | Prepreg: autoclave or hot press (B, M); C, I not stated | **Moderate** below 20°, **Low** above |
| UD tension, ultimate | Analytical three-mechanism, with margin: −23% (conservative) on the smaller wrinkles, +16% to +18% unsafe on the larger. The angles are estimated, and the larger wrinkles knock down more at the lower estimated angle, which no angle-driven model follows. | 4 cases, 1 lab, in-sample (issue #454) | Prepreg: hot press (M) | **Low** |
| Tension, damage onset | No reliable predictor. First-ply FE is an onset estimate only: unsafe on 9 of 12 Dataset M coupons (−22% to +154%, worst on the deepest wrinkles) and 2 of 3 Mukhopadhyay. The analytical onset estimate is unvalidated and flagged in the output: on Dataset M it is above the measurement on all nine multidirectional coupons (up to about +240%) and 95–99% below it on the UD ones (issue #463). | 15 cases, 2 labs | Prepreg: hot press (M); C not stated | **Low** |
| UD glass compression | Penetration gate, on the material it was fitted to. Kink-band is far too conservative (−37% to −58%); FE ignores wrinkle depth. | 15 cases, 1 group, in-sample | Prepreg: hot press (E), vacuum bag (F) | **Fit only**  |
| UD carbon compression (not wet lay-up) | FE on the specimen's measured geometry: −5% to +9% on Thor and Pilato, −26% to −40% (conservative) on Shi. The kink-band, with its UD strain budget read from the material card (S12/Xc, issue #451), is a conservative screen on Thor's and Pilato's whole-thickness waves (−10% to −23%, never unsafe). On Shi's one-sided troughs, modelled as `tool_flat` (issue #461), it is strongly conservative (−59% to −71%) and flags the 20% and 30% troughs as bending-dominated; entered as `graded`, which keeps the layup budget, it over-predicted by +0.2% to +26% with no warning. Never the gate presets as shipped: unsafe on 7 of 10, up to +97%. Re-fitting the moulded preset's `gamma_Y` to one wrinkled test is unproven: within 2–4% on Shi and Pilato, −29% / +15% on Thor's two cases. For an embedded wrinkle with flat surfaces, FE has had unsafe misses on every dataset with one (C, D, E, F), and the two size-aware FE candidates of issue #453 did not change that (see [above](#size-aware-fe-candidates-embedded-flat-surface-wrinkles-issue-453)). Wet lay-up is outside this row (issue #445): on the quarantined Dataset L, FE was up to +100% unsafe and the one-test gate up to +71%. | 10 cases, 3 labs | Prepreg: autoclave (J); H, K not stated | **Low**: issues #435, #445  |

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
