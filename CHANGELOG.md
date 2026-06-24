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

## [1.0.0]

Initial public release: analytical Budiansky–Fleck knockdown plus a 3-D
finite-element pipeline with LaRC05/Hashin/Puck ply failure and
cohesive-zone delamination; five wrinkle morphologies; the material
library; JSON/CSV/Abaqus/VTK export; a command-line interface; and the
Streamlit web application.

[Unreleased]: https://github.com/elhajjar1/wrinkleFE/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/elhajjar1/wrinkleFE/releases/tag/v1.0.0
