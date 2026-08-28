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
failure, not the load the coupon finally carries. For pristine UD in
compression the linear LaRC05 index never activates at all, so this path
yields no useful knockdown — which is exactly why the progressive-damage
path exists.

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
  penetration gate for the closed-form answer, and
  `progressive_knockdown` for the FE answer. `retention_factors` will
  not help here.
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
