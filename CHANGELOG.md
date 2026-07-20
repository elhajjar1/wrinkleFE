# Changelog

All notable changes to WrinkleFE are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

In addition to the standard categories, a **Numerical results** category
calls out any change that shifts predictions (failure-criterion fixes,
CZM-law changes, default-parameter changes, field-composition changes).
Those are the entries to scan when upgrading an engineering analysis
tool — and a results JSON's `provenance` block lets you detect which
version produced a given file.

## [Unreleased]

### Added
- CLI — **config-first sweeps/compares, transverse + stochastic exposure,
  and ergonomics** (issue #375, CLI slice). `sweep` and `compare` gain
  `--config PATH`: the file supplies the base `AnalysisConfig` (laminate,
  material, mesh, penetration gate) so a **UD amplitude sweep through the
  penetration gate** — previously impossible from the CLI — is reachable
  with `wrinklefe sweep --config ud_gate.json --parameter amplitude ...`;
  explicitly-passed geometry flags override the file via the #259
  SUPPRESS-default precedence. `analyze` exposes the through-width
  transverse surface (`--transverse-mode {uniform,gaussian_decay,
  sinusoidal_y,elliptical}`, `--transverse-span`, `--transverse-width`; a
  non-uniform mode forces the FE path) and the previously config-only
  `--nz-per-ply`, `--ply-thickness` (sets the gate D/T) and `--output-csv`.
  A new **`wrinklefe stochastic`** subcommand wraps
  `stochastic.probabilistic_analysis`: a `--config` base plus repeatable
  `--distribution FIELD:DIST:P1:P2` specs (`normal`/`uniform`/`lognormal`),
  `--n-samples`/`--seed`/`--method`, printing percentile knockdowns and
  writing JSON/CSV. `wrinklefe --version` now reads the installed package
  metadata instead of a hardcoded string. (App config upload/download and
  the app transverse controls are a separate follow-up; this slice does not
  complete #375.)
- App / CLI — **`tool_flat` surface-pocket controls live in the morphology
  definition** (issue #371, Part B — *Fixes #371*, completing the issue).
  The Streamlit Morphology selectbox gains **`tool_flat`** (Expert mode)
  with its own schematic cartoon (flat pinned face, uniform-amplitude core,
  amber resin wedges at the troughs). Selecting it renders the pinned-side
  and **surface-transition-plies** controls *directly under* the morphology
  controls — no longer in the FE expert section — alongside a live
  inversion-bound caption (max safe amplitude = `0.8 · S · t / nz`) and a
  pre-run warning naming both remedies, so the config `ValueError` is never
  the user's first feedback. Surface pockets **auto-enable** for `tool_flat`
  (no checkbox). The Analyze-tab cross-section renders the *thick* pockets
  via the actual `tool_flat` decay (pinned plies flat, uniform core); a
  fidelity test binds the analytic preview gap area to the multi-layer FE
  tagged volume (~10 %). The CLI accepts `--morphology tool-flat` (alias for
  `tool_flat`) and `--surface-transition-plies N`, with help noting the
  pockets auto-enable and the amplitude bound.
- Analysis — **`tool_flat` morphology with significant surface resin
  pockets** (issue #371, Part A). A new through-thickness decay mode /
  morphology (`morphology="tool_flat"`) models a wrinkle cured against
  rigid tooling: a uniform-amplitude core, a short linear transition over
  `surface_transition_plies` plies (new config field, default 2), and an
  **exactly-flat pinned surface** on `surface_pocket_side`
  (`"top"`/`"bottom"`/`"both"`). This fixes the root cause of the reported
  bug that surface pockets were invisible and mechanically negligible: the
  linear-decay morphologies (`stack`/`convex`/`concave`, `graded` with
  `decay_floor=0`) spread the wave across the whole thickness, so at the
  24-ply defaults (t=0.183 mm, A=0.5 mm) the outermost undulating ply moved
  only ~0.045 mm — a trough gap of **~0.25 of one ply thickness**. Under
  `tool_flat` the mismatch collects at the flat surface, so the trough
  pocket is ≈ the full amplitude (**~2.7 ply thicknesses** at defaults).
  Surface pockets **auto-enable** for `tool_flat` (they are its defining
  physics; skipped only for `analytical_only`), and the analytical path
  equals `uniform` (M_f = 1.0; the pocket effect is FE-only). Toggling the
  pockets now moves `modulus_retention_global` by a clearly significant
  margin (measured **~3.1–3.4 %** at A=0.5–0.55 mm, `surface_transition_plies=4`,
  side `both`) versus a negligible **~0.3 %** for the legacy `stack`
  morphology — a ~10× larger effect, the original complaint resolved.
  `compute_surface_resin_blend` tags **all** stretched layers in the
  multi-ply transition zone (volume-conserving), and its height metric is
  now the tilt-invariant vertical stretch (top-face minus bottom-face
  mean-z) so a realistically-localized wrinkle's in-plane slope no longer
  corrupts the gap. A verified element-inversion bound (`amplitude ≤ 0.8 ·
  surface_transition_plies · ply_thickness / nz_per_ply`) is enforced at
  construction with a message naming both remedies; multi-wrinkle / CZM /
  transverse combinations raise `NotImplementedError`. App/UX exposure of
  the new morphology is a deliberate follow-up (Part B).
- Analysis — **through-width (transverse) wrinkle surfaces reachable from
  `AnalysisConfig`** (issue #300). The already-implemented, already-tested
  `WrinkleSurface3D` transverse modes are now selectable through three new
  config fields: `transverse_mode`
  (`"uniform"`/`"gaussian_decay"`/`"sinusoidal_y"`/`"elliptical"`, default
  `"uniform"`), `transverse_span` (→ `span_y`, `None` tracks
  `domain_width`), and `transverse_width` (→ `width_y`, `None` resolves to
  `span_y / 4` — a localized mid-width patch). With the default
  `"uniform"` the pipeline still builds the bare x-only
  `GaussianSinusoidal`, so results are bit-identical (regression-safe). A
  non-uniform mode wraps the profile in a `WrinkleSurface3D` on the FE
  single-wrinkle path so the crest amplitude varies across the specimen
  width; at the same crest amplitude a localized wrinkle predicts a milder
  knockdown than the uniform baseline (real manufacturing wrinkles are
  localized, and the uniform assumption overstates the defect volume).
  FE-only and single-wrinkle for now: analytical-only, multi-wrinkle
  (`wrinkles`), and `enable_czm` combinations are rejected at construction
  with actionable messages. New `examples/transverse_wrinkle_knockdown.py`
  demonstrates localized-vs-uniform knockdown; CLI/app exposure is a
  deliberate follow-up.
- CLI — **wrinkle-defect capabilities on `analyze`** (issue #346). The
  new defect models shipped for the scripting API are now reachable from
  the command line: `--wrinkle-z-position Z` (off-mid-plane wrinkle,
  validated to `[0, 1]`), `--gate {li2024-moulded,li2025-vacbag}` (the
  two-parameter (θ, D/T) penetration gate, selecting a calibrated
  `GateParameters` preset), `--resin-pocket` (crest resin lens),
  `--surface-resin-pockets` / `--surface-pocket-side {top,bottom,both}`
  (tool-flat surface pockets), and `--progressive` / `--increments N`
  (load-stepping ultimate strength). The flags inherit the #259
  config-file precedence, so any flag left off keeps the `--config`
  value and any flag passed overrides it (and is written by
  `--save-config`); the FE-only features force the FE path with the same
  precedence as `--enable-czm` rather than silently no-op'ing under
  `--analytical-only`. The result summary now prints the progressive
  knockdown when a progressive run happened. `sweep --parameter
  wrinkle_z_position` works end-to-end over a `--config` base. The
  surface-resin-pocket flags (issue #361, newer than #346) are included
  as part of the same CLI-reachability story. Buckling was **deferred**:
  the linearized microbuckling solver is standalone diagnostic
  infrastructure with no `AnalysisConfig` knob and is documented as not a
  usable knockdown predictor, so there is no clean field to surface.
- Streamlit app — **surface resin pockets in the through-thickness
  cross-section** (issue #361, Part 4 follow-up). The Analyze-tab
  cross-section now shades the neat-resin pockets that fill the wrinkle
  troughs under a tool-flat surface, and an expert-mode sidebar toggle
  (*Surface resin pockets*, with a top/bottom/both side selector) drives
  both the preview and the FE run so a solve models exactly what the
  picture shows. The zone is rendered *analytically* — the amber fill
  between the flat tool line and the deformed outermost undulating ply —
  rather than by deforming an FE mesh at render time, so the preview
  stays responsive; a fidelity test cross-checks that rendered gap area
  against the resin volume `compute_surface_resin_blend` tags on a coarse
  mesh (agree within ~4%, well inside #361's 10% conservation tolerance).
  Shown only for tool-flat morphologies (`stack`/`convex`/`concave`, or
  `graded` with `decay_floor=0`); an incompatible morphology (`uniform`,
  or `graded` with a non-zero floor) is withheld from the config and
  flagged with a sidebar note, so a run can never build an invalid
  config. Opt-in and off by default (feature-off preview unchanged).
- Tool-flat surfaces with surface resin pockets (issue #361).
  Parts cured against rigid tooling / a caul sheet keep perfectly flat
  outer surfaces while the fibres undulate internally; the wrinkle
  troughs fill with neat resin just under the flat surface. New
  `SurfacePocketSpec` / `compute_surface_resin_blend` in
  `wrinklefe.core.resin_pocket` tag, per column, the transition element
  that stretches to span the gap between the flat surface and the
  outermost undulating ply, weighting it by the excess-stretch fraction
  `max(0, (h - h0) / h)` — exactly volume-conserving (equal to the
  integrated kinematic gap `-w(x)·decay_last`). Enabled via
  `AnalysisConfig.enable_surface_resin_pockets`, `surface_pocket_side`
  (`top`/`bottom`/`both`) and `surface_pocket_min_gap`, reusing
  `resin_pocket_material` / `resin_pocket_graded`. FE-only effect
  (`modulus_retention_global` and first-ply failure in the isotropic
  resin zone); it composes with the crest lens (per-element maximum) and
  needs no solver changes. Requires a tool-flat morphology whose decay
  reaches 0 at the chosen surface (`stack`/`convex`/`concave`, or
  `graded` with `decay_floor=0`); `uniform` and `graded` with a non-zero
  floor are rejected with a message naming the fix. Disabled by default
  (bit-identical results when off).
- Save / load an `AnalysisConfig` (issue #259).
  `AnalysisConfig.to_dict()` / `from_dict()` provide a round-trippable,
  `config_version`-stamped serialisation; `save_json` / `load_json`
  (and extension-dispatching `save` / `load`, plus optional YAML when
  PyYAML is installed) read and write config files. Library materials
  serialise by preset name, custom materials inline, and penetration-gate
  presets by their registry name (new
  `wrinklefe.core.penetration_gate.GATE_PRESETS`). Loading rejects
  unknown keys and version mismatches loudly. The `wrinklefe analyze`
  CLI gains `--config PATH` (load a config, with explicitly-passed flags
  overriding the file) and `--save-config PATH` (write the effective
  config). A follow-up will surface the same config download/upload in
  the Streamlit app.
- Streamlit app — **Through-thickness cross-section** on the Configure
  tab: a new panel that draws the deformed ply stack in the (x, z) plane
  so users can see how the wrinkle manifests through the laminate
  thickness. It reuses the real `WrinkleConfiguration.apply_to_nodes`
  field the FE mesh uses, so the picture faithfully tracks the active
  morphology, its through-thickness amplitude decay, the dual-wrinkle
  phase offset, and any in-plane amplitude profile. Each ply is a band
  coloured by fibre angle (the same hue map as the layup visualizer),
  with the wrinkle-interface plies outlined for the dual morphologies.
- Monte-Carlo / Latin-hypercube uncertainty propagation (issue #301):
  `wrinklefe.stochastic.probabilistic_analysis(base_config,
  distributions, n_samples, seed, method="lhs"|"mc")` samples
  `AnalysisConfig` fields from user distributions (`("normal", m, s)`,
  `("uniform", lo, hi)`, `("lognormal", mu, sigma)`, or any frozen
  `scipy.stats` distribution), runs the analytical path per sample, and
  returns a `ProbabilisticResults` with percentile
  knockdowns/strengths, mean ± std, the input samples for sensitivity
  scatter, an optional histogram+scatter `plot()`, and a `summary()`
  that explicitly labels the output as model-input-propagation
  statistics — **not** CMH-17 A-/B-basis allowables. Fixed seeds are
  fully reproducible; degenerate (zero-variance) distributions
  reproduce the deterministic result exactly; invalid draws fail loudly
  instead of being clipped; `n_workers` reuses the #260 process pool
  for FE-path sampling. 1000 analytical samples run in ~0.7 s for
  UD/gate configs.
- Vectorized `_laminate_modulus_knockdown` (issue #301 enabler): the
  multidirectional analytical modulus knockdown ran a per-(ply,
  x-station) Python loop (12,000 6×6 rotations/condensations), making
  every multidirectional analytical run take ~1.2 s. The wrinkle-tilt
  rotation and plane-stress condensation are now batched over the whole
  (ply, x) grid (the in-plane rotation is computed once per ply, and
  `T_sigma(θ)^-1 = T_sigma(−θ)` replaces the batched solve) — 1.18 s →
  22 ms (~52×) per analytical run, results identical to the loop
  (regression-tested against it; ledger baselines zero-drift).
- Mesh-resolution warning (issue #306): `WrinkleMesh.generate` warns
  when the hex mesh samples the wrinkle wavelength with fewer than 4
  elements (element `dx` vs `lambda`), naming the offending spacing and
  the `nx` needed — under-sampled wrinkles previously produced silent
  aliasing noise.
- CZM-capable glass/aramid presets (issue #268): `S2_GLASS_EPOXY` and
  `KEVLAR49_EPOXY` now carry representative interlaminar toughness
  (`GIc`/`GIIc` in the published glass/aramid ranges), so
  `enable_czm=True` runs for every built-in material instead of raising
  for those two. Configurations that previously errored now produce
  cohesive-damage results.
- Coordinate-aware maxima (issue #297):
  `FieldResults.max_displacement_location()` and
  `max_stress_location()` return the physical `(x, y, z)` of the
  governing node / element centroid alongside the value.
- Failure-mode breakdown plot (issue #269, part 1):
  `wrinklefe.viz.plot_failure_mode_breakdown` with a stable
  per-failure-mode colour map (`MODE_COLORS`).
- Process-parallel parametric sweeps (issue #260):
  `WrinkleAnalysis.parametric_sweep(..., n_workers=N)`,
  `wrinklefe.sweep.run_sweep(..., n_workers=N)`, and
  `wrinklefe sweep --parallel N` fan the independent per-point solves
  out over a `ProcessPoolExecutor` (`N=0` uses all CPU cores; the
  default `N=1` keeps the exact sequential path). Results are identical
  to and ordered like the sequential run — measured 3.6× at 4 workers
  on an 8-value full-FE amplitude sweep. `run_sweep` progress becomes
  completion-based in parallel mode; `KeyboardInterrupt` (or a worker
  failure) cancels the queued futures instead of draining the pool.
  Peak memory scales with workers × per-solve footprint — size `N` by
  available RAM for fine meshes.
- Vectorized `FieldResults.max_principal_stress` (issue #295): the
  per-Gauss-point Python double loop (one `np.linalg.eigvalsh` call per
  point) is replaced by a single batched eigen-solve on the
  `(n_elem, n_gp, 3, 3)` tensor stack — ~6× on a 50k-element × 8-GP
  field (4.2 s → 0.7 s on first access; results identical to 1e-10,
  regression-tested against the old loop kept as the test oracle).
  Element centroids are now computed once per `FieldResults` (lazy
  `element_centers` property) instead of rebuilt on every
  `stress_through_thickness` call (372 ms → 5 ms per query on the same
  mesh).
- Vectorized `evaluate_field` for LaRC05, Puck, and Budiansky–Fleck
  (issue #299) — the three most expensive criteria were the last ones
  running the base class's per-Gauss-point Python loop. The
  fracture-plane / action-plane searches broadcast over a
  `(N, n_theta)` grid processed in cache-sized row blocks; measured on
  an 80,000-point field: LaRC05 12.0 s → 0.51 s (23×), Puck 13.8 s →
  0.67 s (21×), kink-band 93 ms → 7 ms (13×). Failure post-processing
  no longer dwarfs the linear solve when these criteria are enabled
  (the progressive-damage crack-band loop, which evaluates
  MaxStress+LaRC05 every equilibrium iteration, inherits the speedup).
  Outputs are **bit-identical** to per-point `evaluate()` — enforced by
  a per-criterion equivalence suite using exact array equality across
  randomized samples covering every branch regime. The base-class loop
  fallback now logs at DEBUG so future criteria authors notice.
- Penetration-gate validation harness (issue #161): the calibrated UD
  gate — the only strength path sensitive to wrinkle amplitude and
  through-thickness position independently of the peak angle — is now
  pinned in the reproducible ledger. `scripts/validate.py` scores a
  per-case gate column (with drift detection and `--update` re-pinning)
  for any dataset naming a `penetration_gate` preset; the Li 2025
  dataset carries `expected_gate_kd` baselines and a per-case `z_frac`
  through-thickness position (S-A-2 rides the gate's `P(z)` factor to
  its measured near-surface KD). The issue's acceptance criteria are
  permanent regression tests: the S-M-2/4/5 amplitude trio (identical
  20° angle, measured KD 0.629/0.943/1.000) lands at +2.2 %/−0.6 %/
  −0.3 % (±15 % band), orderings asserted monotonic, all six cases
  within the ±20 % parity band. README and VALIDATION.md updated to
  document the in-repo reproducible UD validation.
- Cohesive-zone delamination in multi-wrinkle FE (issue #283):
  `enable_czm=True` now runs with an `AnalysisConfig.wrinkles` list
  instead of raising `NotImplementedError`. Cohesive layers are inserted
  along the **full length** of every nominated interface, and
  `czm_interfaces="near_crest"` nominates the interface nearest *each*
  wrinkle (deduplicated) — wrinkles sharing an interface index get one
  continuous cohesive surface, so a delamination initiating at one crest
  can propagate toward its neighbour (crest-to-crest link-up, the Li 2025
  multi-wrinkle failure pattern; see
  `examples/08_multi_wrinkle_czm_linkup.py`). Regression anchors: a
  one-entry `wrinkles` list reproduces the scalar-config CZM solution
  bit-tight; far-separated wrinkles match independent single-wrinkle
  solves within a few percent with an intact interface between them;
  scalar (named-morphology) CZM interface resolution is unchanged.
- Analytical stiffness (axial-modulus) knockdown on the analytical path:
  `AnalysisResults.analytical_modulus_knockdown`, a closed-form CLT
  series-average of the off-axis lamina modulus over the wrinkle profile
  (`analysis._profile_modulus_knockdown`). Populated for unidirectional
  layups (loading-independent, zero FE cost) — the closed-form companion
  to the FE `modulus_retention`, which previously was the only stiffness
  knockdown (the analytical path reported none). Surfaced in
  `AnalysisResults.summary()`, the `analyze --output-json` payload,
  `results_to_dict`, and the Streamlit app; validated by
  `validation/validate_modulus.py` (analytical MAE 3.9 % / 1.2 % on the
  Li 2025 / Hsiao & Daniel UD datasets).
- Resin-pocket material zone (`wrinklefe.core.resin_pocket`:
  `ResinPocketSpec`, `compute_resin_mask`, `compute_resin_blend`):
  a graded neat-epoxy lens at the wrinkle crest, tagged into the FE mesh
  via `AnalysisConfig.enable_resin_pocket` /
  `resin_pocket_graded` / `resin_pocket_material` /
  `resin_pocket_height_scale` / `resin_pocket_length_scale`. The modulus
  and fibre-misalignment angle blend together so the wrinkle defect is
  counted once. Adds `OrthotropicMaterial.isotropic()` / `.blend()` and
  an isotropic neat-epoxy card `EPOXY_S6C10`.
- Progressive-damage FE solver
  (`wrinklefe.solver.progressive_damage`: `ProgressiveDamageSolver`,
  `ProgressiveDamageResult`), enabled via
  `AnalysisConfig.enable_progressive_damage` with
  `progressive_n_increments` / `progressive_residual_factor` /
  `progressive_max_strain`. Load-steps to ultimate load with optional
  crack-band (Bažant–Oh) regularization — the first FE route to a real
  UD compression knockdown.
- Two-parameter (θ, D/T, z) penetration gate
  (`wrinklefe.core.penetration_gate`: `GateParameters`,
  `penetration_gate_kd`, `angle_floor`, `position_factor`,
  `predict_from_geometry`, `calibrate_gate`, plus presets
  `GATE_LI2024_MOULDED` and `GATE_LI2025_VACBAG`), wired through
  `AnalysisConfig.penetration_gate`. A closed-form UD predictor
  `KD = 1 − (1 − KD_angle(θ))·S(D/T)·P(z)` at zero FE cost.
- Linear buckling / geometric stiffness
  (`wrinklefe.solver.buckling`: `LinearBucklingSolver`,
  `BucklingResult`, `microbuckling_knockdown`), backed by
  `Hex8Element.geometric_stiffness_matrix` and
  `assemble_geometric_stiffness`.
- Movable wrinkle through-thickness position
  (`AnalysisConfig.wrinkle_z_position`, 0.5 = mid-plane); the graded
  decay centres there.
- `AC318_S6C10_vacbag` material card — the Li 2025 vacuum-bag
  realization of the AC318 / S6C10-800 S-glass/epoxy prepreg (measured
  Xc = 335.5 MPa, E1 = 50.8 GPa).
- `IM6G_3501_6` material card — Hercules IM6G / 3501-6 carbon/epoxy
  (Vf 0.66) from Hsiao & Daniel (1996), the material behind validation
  **Dataset G** (UD carbon, measured stiffness *and* strength knockdown).
  This brings the built-in `MaterialLibrary` to 12 cards (11
  fibre-reinforced systems + the `EPOXY_S6C10` neat-epoxy card).
- Stiffness-only validation driver (`validation/validate_modulus.py`):
  compares WrinkleFE's axial Young's-modulus knockdown — the FE
  `modulus_retention` and a closed-form CLT series-average estimate of
  the off-axis lamina modulus over the wrinkle profile — against the
  measured modulus knockdown in the UD datasets E (Li 2024), F (Li 2025),
  and G (Hsiao & Daniel 1996). It is the first stiffness (as opposed to
  strength) validation in the repository; analytical MAE 3.9 % (F) /
  1.2 % (G), FE MAE 6.9 % (F) / 5.1 % (G).
- Stiffness validation chart (`validation/plot_modulus_validation.py` →
  `validation/fig_modulus_validation.png`): the modulus counterpart of
  the strength parity chart — predicted-vs-experimental modulus knockdown
  (analytical and FE) vs misalignment angle, plus a parity panel, across
  the UD modulus datasets E/F/G.
- Combined validation parity chart
  (`validation/plot_all_validation.py` →
  `validation/fig_all_validation_parity.png`): a predicted-vs-experimental
  parity plot of all single-wrinkle cases (Datasets A–F) inside a ±20%
  band, each predicted with the model that applies to it
  (Budiansky–Fleck / three-mechanism for multidirectional A–D, the
  penetration gate for UD E/F).
- `wrinklefe sweep` and `wrinklefe compare` gained `--output-json` and
  `--output-csv`: machine-readable batch results (a JSON array of
  per-run objects matching `analyze --output-json`, and a tidy
  one-row-per-run CSV with full float precision). The stdout tables are
  unchanged.
- `examples/` directory of runnable workflow scripts (basic knockdown,
  parametric sweep, morphology comparison, CZM delamination, export
  round-trip, custom material, mesh convergence), executed in CI.
- Sphinx documentation site (`docs/`) with an autogenerated API
  reference, published configuration for Read the Docs.
- **Theory: physics & mechanics** documentation page (`docs/theory.md`):
  a consolidated, code-accurate reference for the wrinkle kinematics,
  the CLT-weighted Budiansky–Fleck kink-band (incl. the confinement /
  block-penalty yield-strain model), the tension three-mechanism
  minimum, the unidirectional penetration gate, the resin-pocket and
  progressive-damage / crack-band routes, and the cohesive-zone law.
- `mesh_convergence_study()` helper and a `wrinklefe converge` CLI
  command for refinement studies.
- Structured `logging` across the analysis pipeline; `wrinklefe ... -v`
  attaches a DEBUG stderr handler.
- Multi-wrinkle finite-element solve (`AnalysisConfig.wrinkles`),
  including overlapping/interacting wrinkles.
- Contracted layup notation in the NCR validation summary
  (`to_contracted_layup`).
- Committed validation-ledger harness (`scripts/validate.py`,
  `tests/test_validation/ledger.json`).
- `provenance` block on JSON exports and the NCR summary recording the
  installed version, numerics stack, platform, and timestamp.
- GitHub issue forms, pull-request template, and this changelog.
- Docstring examples are now executed in CI (issue #296). A dedicated
  `doctests` job runs `pytest --doctest-modules src/wrinklefe`, and
  `doctest_optionflags = "NORMALIZE_WHITESPACE ELLIPSIS"` is set. The
  runnable `>>>` examples were made exact (real expected output,
  NumPy-version-stable reprs) so they act as regression guards; examples
  that need a generated mesh or a full FE solve are marked
  `# doctest: +SKIP`. Kept out of the default `addopts` so a doc example
  cannot block the core suite. Docstring-only change — no numeric
  results shift.

### Fixed
- Phase 0 correctness batch (issue #374, *Fixes #374*) — six small hazards
  from a full-codebase scan; none change numerical results for valid runs
  (ledger zero-drift):
  - **Silent modulus-retention fallback.** The local (σ₁₁ proxy) and global
    (reaction-based) FE modulus-retention blocks swallowed every exception
    and set the no-knockdown value `1.0` (the local block logged nothing),
    so a bug in the FE stiffness path read as a clean bill of health. Both
    now log a `WARNING` with `exc_info` and set a companion boolean flag
    (`AnalysisResults.modulus_retention_failed` /
    `modulus_retention_global_failed`) so a fallback `1.0` is
    distinguishable from a genuinely computed `1.0`. The value stays a
    `float` (its many consumers call `float()`/format it); the flag is
    surfaced in `summary()` and serialised only when set.
  - **Stale app results.** `reset_inputs()` now also drops the run-derived
    `results` / `cfg_payload`, and the Analyze tab renders an "inputs have
    changed since this run" banner when the live sidebar no longer matches
    the payload the shown results were computed from. The Reset button moved
    to an `on_click` callback, fixing a latent `StreamlitAPIException` from
    writing widget-keyed state after instantiation.
  - **CI mypy blind spot.** The lint job now installs the `streamlit` extra
    so mypy type-checks the app against real streamlit/plotly types (was
    `Any`), and the documented local gate passes on `main` (the `app.py`
    `reset_inputs` loop annotation is fixed). `check_untyped_defs` is now
    enabled repo-wide and clean; the `parametric_sweep.py` "str → float |
    None" suspect was a typing false-positive (a legitimately heterogeneous
    params dict inferred too narrowly), fixed by annotating `DEFAULTS`.
  - **`.gitignore` traps.** The wholesale `examples/` and `validation/*`
    ignores hid new scripts from `git status`; replaced by per-directory
    `.gitignore`s that ignore only generated outputs, so drivers stay
    visible. Added the previously phantom
    `examples/08_multi_wrinkle_czm_linkup.py` (crest-to-crest CZM link-up)
    that the README already listed.
  - **Dead scipy guard.** Removed the `except ImportError: pass` around
    `scipy.stats.gaussian_kde` in `viz/plots_2d.py` (scipy is a hard
    dependency; the guard would have silently skipped the KDE).
- Results-export schema drift (issue #345): the structured JSON export
  (`wrinklefe.io.results.results_to_dict` / `export_results_json`) and the
  NCR validation summary (`wrinklefe.io.export.build_analysis_summary`)
  silently dropped several `AnalysisResults` fields. `results_to_dict`
  now serialises `modulus_retention_global` and `analytical_onset_knockdown`,
  and — for progressive-damage runs — a gated `progressive` block
  (`strength_MPa`, `pristine_strength_MPa`, `knockdown`, `n_increments`,
  and the `(strain, stress)` load history). The NCR markdown/PDF renderers
  now surface the global coupon modulus retention and a progressive-damage
  section, wired end-to-end from the Streamlit Export tab. A new
  dataclass-walking drift-guard test asserts every `AnalysisResults` field
  is either exported or on an explicit allowlist, so a future field cannot
  silently go unexported. Analytical-only runs are unchanged (no
  `progressive` block, no empty NCR rows).
- Dual-wrinkle amplitude contract in the FE mesh (issue #305): the
  `stack`/`convex`/`concave` morphologies build the mesh by summing two
  through-thickness–decayed displacement fields, and each constituent
  previously carried the full amplitude `A`, so the in-phase `stack` mesh
  peaked at `2A` — double the intended geometry, meshed fibre angle and FE
  knockdown, and inconsistent with the analytical
  `theta_max = arctan(2*pi*A/lambda)`. Each constituent is now generated at
  half amplitude `A/2` (`morphology._profile_at_half_amplitude`), so the
  `stack` mesh composes to exactly `A` and its fibre angle matches the
  analytical profile. Explicit multi-wrinkle `WrinkleSpec` configurations
  are unaffected (each listed wrinkle keeps its specified amplitude). See
  Numerical results.
- Linear-buckling eigensolve correctness/robustness
  (`solver/buckling.py`): for a wrinkled (non-uniform) pre-stress the
  geometric "mass" matrix `M = -K_geo` is **indefinite**, which violated
  the SPD assumption of the previous `eigsh` shift-invert. It returned
  spurious, run-to-run-varying eigenvalues — and on macOS arm64 sometimes
  no surviving positive mode at all, so `critical_load_factor` came back
  `inf` and the buckling-knockdown test flaked in CI. The solve is now the
  symmetric-definite pencil `M φ = μ K φ` (the material stiffness `K` is
  SPD), with `λ = 1/μ` and a deterministic ARPACK start vector — finite,
  reproducible, and matching a dense reference. This is infrastructure
  only; `microbuckling_knockdown` is still **not** the production UD
  predictor (see Numerical results).
- Documentation accuracy (physics audit): the README "How It Works" and
  `ARCHITECTURE.md` confinement section now show the full three-parameter
  effective yield strain `gamma_Y_eff = max(0.032 + 0.050·f_conf −
  0.010·max(n_block−1,0), 0.016)` (the block-penalty and floor terms were
  missing), document `theta_eff = M_f·theta_max` and the optional
  Argon–Fleck quadratic term, and correct the graded through-thickness
  decay scale to `max(λ/2, A)`. The `MaterialLibrary` docstring (and its
  doctest) now lists all 12 registered cards (11 fibre-reinforced systems
  + the `EPOXY_S6C10` neat-epoxy card) instead of a stale list of nine.
- API-reference docstrings (physics audit, follow-up): the LaRC05 module
  docstring described "iterative φ_c computation" and a Ramberg-Osgood
  nonlinear-shear amplification that the code does not apply — corrected
  to the linear closed-form load-induced φ_c that is actually used (the
  `max_phi_c_iter` / `phi_c_tol` parameters are documented as reserved /
  unused). The linear-buckling module docstring now carries the item-D.4
  negative-finding note (the eigenvalue over-predicts the UD wrinkle
  knockdown; use the penetration gate instead), matching the README and
  the new theory page.
- `wrinklefe sweep` now validates its inputs: an unknown `--parameter`,
  `--min >= --max`, or `--steps < 2` print a one-line error and exit
  non-zero (code 2) before any solve, instead of a raw traceback or a
  silently degenerate sweep. The `--parameter` help no longer implies
  only amplitude/wavelength/width are accepted.
- Graded-morphology compression knockdown now honours `decay_floor`
  (previously inert in compression while honoured in tension).
- JSON export stamps the real installed version instead of a hardcoded
  `0.1.0` literal.
- Latent crashes surfaced by static typing: `np.trapz` removal under
  numpy 2.0 in the stress-resultants path; `WrinkleSurface3D` attribute
  access in `max_angle` / `fiber_angles_at_nodes`.

### Changed
- App — **surface-pocket controls relocated into the morphology
  definition** (issue #371, Part B). The standalone *Surface resin pockets*
  expander in the Expert FE section is gone; its controls now live directly
  under the Morphology selector. For `tool_flat` the pockets are implicit
  (auto-enabled, shown as a caption) with the pinned-side and
  transition-ply controls inline; the legacy tool-flat morphologies
  (`stack`/`convex`/`concave`, `graded` with `decay_floor=0`) keep an
  *advanced* opt-in whose help explains their pockets are inherently small
  (~0.25 ply thickness — use `tool_flat` for significant pockets). The
  `sb_surface_transition_plies` widget key joins `DEFAULTS` so *Reset to
  defaults* round-trips it. Rationale: under the linear-decay morphologies
  the pockets were mechanically negligible by construction (measured
  ~0.25 ply-thickness trough gap), so a tooling-dominated wrinkle belongs
  in the morphology definition, not as an add-on toggle.
- Packaging — **PyVista/VTK moved to an optional `vtk` extra** (issue
  #302). Plain `pip install wrinklefe` no longer pulls in VTK (~150 MB
  lighter) and stays headless-safe; the 3D cohesive-zone plots
  (`plot_interface_damage_3d` / `plot_crack_front_3d`) now require
  `pip install "wrinklefe[vtk]"` (also included in `[all]`). PyVista was
  already imported lazily, so `import wrinklefe`, the CLI, the Streamlit
  app, and the docs build are unaffected when it is absent; the
  `_require_pyvista` error message now names the `wrinklefe[vtk]` extra.
- Streamlit app — acknowledgment gate and intro reworded to a
  professional-tool framing (issue #333): the gate leads with what
  WrinkleFE computes rather than "free academic software", and the
  supporting copy, email placeholder, and acknowledgment checkbox use
  neutral, work-agnostic wording. Gate mechanics, the
  `WRINKLEFE_DISABLE_GATE` off-switch, usage logging, and the
  MIT/attribution facts are unchanged.
- Streamlit app — the **Configure** and **Results** tabs are merged into
  a single default **Analyze** tab (issue #334), leaving three tabs
  `["Analyze", "Export", "Help"]`. The wrinkle/laminate preview lives in
  an expander that is open before the first run and auto-collapses once
  results exist, so results lead the view after a run. Export and Help
  are unchanged. (Supersedes the #358 tab-order entry below.)
- Results-export `SCHEMA_VERSION` bumped `1.0` → `1.1` (issue #345): the
  additive `modulus_retention_global`, `analytical_onset_knockdown`, and
  gated `progressive` fields in the structured JSON export. Additive only;
  existing consumers of 1.0 fields are unaffected.
- `AnalysisConfig` now validates ply angles at construction (issue #344)
  — **breaking only for previously-accepted invalid inputs**: a config
  with `|angle| > 90` (e.g. `angles=[900.0, 0.0, 452.0]`) now raises
  `ValueError` naming the offending index and value instead of silently
  flowing a non-canonical angle into CLT trig, where the tension-mechanism
  heuristic mis-classified it (a 900° ply read as a 90° ply). The check
  reuses the shared `validate_ply_angle` rule from `core.layup` (issue
  #343), so the parser and the config validator can never drift. Valid
  layups (decimals, `±90`, long stacks) are unaffected. Follow-up:
  `Laminate.from_angles` is intentionally left unvalidated for now.
- `parse_layup` input strictness (issue #308) — **breaking for inputs
  that previously parsed**: ply-angle tokens with `|angle| > 90` (e.g.
  the repeat-count-like `[02/902]s`, which silently parsed as 2° and
  902° plies) and leading-zero tokens now raise `ValueError` instead of
  building a wrong laminate; ASCII `+-45`/`-+45` are now accepted as
  the ± shorthand. Scripts feeding the newly-rejected forms must switch
  to explicit repeat syntax (e.g. `[0_2/90_2]s`).
- Mesh aspect-ratio warning re-baselined (issue #303): the warning now
  compares each element against the mesh's own median aspect ratio and
  flags only outliers, instead of a fixed 10:1 threshold that flagged
  71–100 % of elements on typical thin-ply meshes (default runs are now
  quiet; genuinely anomalous elements still warn).
- CI enforces the full Ruff ruleset and `mypy` over the whole tree.
- Streamlit app — the **Cohesive Zone Modeling** sidebar controls now
  render only in **Expert mode**. CZM requires the full nonlinear FE
  solve, which is itself expert-only (novice mode forces
  `analytical_only=True`), so the control now sits with the other expert
  FE settings instead of the simplified novice sidebar. The analytical
  path and CZM behaviour are unchanged.
- Streamlit app — tab layout: **Configure** is now the first (default)
  tab, so the app opens on the laminate/geometry view instead of the
  intro. The old **Overview** tab moved to the end and is renamed
  **Help**. Within the Configure tab the **Wrinkle mid-surface profile**
  plot moved to the bottom, so the through-thickness cross-section leads.

### Removed
- The dead `export` optional-dependency extra (`meshio` was never
  imported; native `.inp`/VTK writers need no extra).

### Numerical results
- **Penetration gate × multi-wrinkle (issue #342)**: with a
  `penetration_gate` preset and the geometry supplied via
  `AnalysisConfig.wrinkles`, the gate previously took its angle from the
  wrinkle specs but its penetration `D/T` from the leftover scalar
  `cfg.amplitude` (typically the unused 0.366 default) — a silently
  plausible wrong knockdown (0.98 instead of 0.64 on the issue's repro).
  The gate now evaluates per spec — `theta_i = arctan(2πA_i/λ_i)`,
  `D_i/T = A_i/T`, and `z_i = (ply_interface+1)/n_plies` through the
  position factor `P(z)` (`cfg.wrinkle_z_position` is a scalar-path
  parameter and is ignored when specs are present) — and returns the
  weakest-link (minimum) knockdown over the wrinkles. Scalar-config gate
  results are unchanged (pinned ledger baselines show zero drift); any
  gate × `wrinkles` configuration returns different (correct) values.
- **LaRC05 / Puck last-bit normalization (issue #299)**: the scalar
  `evaluate()` paths now square via an explicit product (`x * x`)
  instead of `x ** 2` — scalar `np.float64` pow routes through libm and
  could land 1 ULP away from the exact product the vectorized field
  path computes. Failure indices from these two criteria may therefore
  shift by at most one floating-point ULP (≈ 1e-16 relative) toward the
  exactly-rounded value; no physical or tolerance-visible change.
- **Linear-buckling microbuckling knockdown**: with the eigensolve
  corrected (indefinite `-K_geo` handled via the symmetric-definite
  pencil), the bifurcation load of the homogenised ply-mesh *rises* with
  the wrinkle (tilted fibres carry less destabilising axial pre-stress;
  e.g. the Li 20 mm coupon goes pristine λ ≈ 8.30 → amplitude-0.6 wrinkle
  λ ≈ 8.65), so `microbuckling_knockdown` returns ≈ 1.0 (no knockdown)
  rather than the spurious sub-1.0 values the old indefinite-`M` solve
  produced. This sharpens the documented negative finding (item D.4): the
  linear eigenvalue gets the wrinkle-knockdown *sign* wrong, which is why
  the UD wrinkle knockdown is taken from the penetration gate. No
  production prediction path consumes `microbuckling_knockdown`, so no
  user-facing knockdown changes.
- **Penetration gate**: when `AnalysisConfig.penetration_gate` is set to
  a `GateParameters` preset, `analytical_knockdown` (and
  `analytical_strength_MPa`) are computed from the two-parameter
  (θ, D/T, z) gate instead of Budiansky–Fleck, so UD configurations
  return different (calibrated) knockdowns. The default
  `penetration_gate=None` preserves previous results bit-for-bit.
- **Graded compression `decay_floor`**: configurations that set
  `decay_floor` under compression now produce different (correct)
  knockdowns. The default `decay_floor=0.0` preserves previous results
  bit-for-bit.
- **Multi-wrinkle fibre angles**: fibre-misalignment fields now derive
  from the slope of the composed displacement field ("compose then
  differentiate"). FE results shift for dual-wrinkle morphologies
  (`stack`/`convex`/`concave`) wherever through-thickness decay < 1 or
  wrinkles overlap; single-wrinkle results are unchanged. Analytical
  predictions are unaffected.
- **Dual-wrinkle mesh amplitude (issue #305)**: the `stack`/`convex`/
  `concave` FE meshes previously peaked at up to `2A` because each of the
  two summed constituents carried the full amplitude `A`. Each constituent
  is now half amplitude, so the in-phase `stack` mesh peaks at exactly `A`
  and its meshed fibre angle drops to the analytical
  `arctan(2*pi*A/lambda)` (roughly halved). FE-derived quantities (fibre
  angles, FE knockdown, stiffness retention) shift for these three
  morphologies; single-wrinkle (`uniform`/`graded`) and explicit
  `WrinkleSpec` multi-wrinkle configurations are unchanged, and analytical
  predictions (which already used the configured `A`) are unaffected.

## [1.0.0]

Initial public release: analytical Budiansky–Fleck knockdown plus a 3-D
finite-element pipeline with LaRC05/Hashin/Puck ply failure and
cohesive-zone delamination; five wrinkle morphologies; the material
library; JSON/CSV/Abaqus/VTK export; a command-line interface; and the
Streamlit web application.

[Unreleased]: https://github.com/elhajjar1/wrinkleFE/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/elhajjar1/wrinkleFE/releases/tag/v1.0.0
