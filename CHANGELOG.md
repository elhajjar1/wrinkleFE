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

### Fixed
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
- CI enforces the full Ruff ruleset and `mypy` over the whole tree.

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
