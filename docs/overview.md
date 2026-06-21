# Overview

WrinkleFE predicts the strength and stiffness knockdown of composite
laminates that contain fibre-waviness (wrinkle) defects. It pairs fast
closed-form models with full 3-D finite-element analysis behind a single
{class}`~wrinklefe.analysis.AnalysisConfig`, so the same wrinkle geometry
can be screened analytically in milliseconds and then, when needed,
solved on a structured hexahedral mesh.

## Analytical knockdown

For multidirectional laminates the analytical path reduces a wrinkle to
its peak fibre-misalignment angle and applies angle-based knockdown
models: a CLT-weighted Budiansky–Fleck kink-band law with
layup-dependent confinement in compression, and a three-mechanism model
in tension (fibre `cos^2(theta)`, Hashin matrix, and a curved-beam
through-thickness `sigma_33` delamination term with a thick-ply in-situ
correction). These run from `amplitude`, `wavelength`, and `width`
without meshing.

## 3-D finite element

The FE path deforms a structured hexahedral mesh to the wrinkle profile
and evaluates ply failure with LaRC05, Hashin, and Puck criteria, with
optional cohesive-zone modelling (CZM) for delamination. Five
morphologies are supported — `stack`, `convex`, and `concave` (dual
wrinkles distinguished by phase) and `uniform` and `graded` (single
wrinkles distinguished by their through-thickness amplitude profile);
see the wrinkle-geometry table in the README/landing page for the full
parameter reference.

## Unidirectional wrinkle-defect capabilities

Angle-based models are scale-invariant: at a *fixed* misalignment angle
they cannot reproduce the strong dependence of UD compressive strength
on how deep the wrinkle penetrates through the thickness. WrinkleFE adds
three capabilities aimed at unidirectional laminates:

- **Penetration gate (θ, D/T, z).** A closed-form UD knockdown predictor
  in `wrinklefe.core.penetration_gate` that is both scale-aware (it
  depends on the through-thickness penetration `D/T = A/T`, not just the
  angle) and position-aware (it accounts for the wrinkle's
  through-thickness location). Calibrated presets `GATE_LI2024_MOULDED`
  and `GATE_LI2025_VACBAG` (Li 2024/2025 UD glass/epoxy) ship with the
  package, and `AnalysisConfig.penetration_gate` routes the analytical
  knockdown through it.
- **Resin-pocket material zone.** `wrinklefe.core.resin_pocket` tags the
  hex elements inside a resin lens at the wrinkle crest and assigns them
  an isotropic epoxy card (the built-in `EPOXY_S6C10`), capturing the
  soft, fibre-free inclusion a machined wrinkle leaves at the crest.
  Enabled and shaped via `AnalysisConfig.enable_resin_pocket`,
  `resin_pocket_graded`, `resin_pocket_material`,
  `resin_pocket_height_scale`, and `resin_pocket_length_scale`.
- **Progressive-damage FE.** `wrinklefe.solver.progressive_damage`
  (`ProgressiveDamageSolver`) load-steps a ply-discount FE solve to
  ultimate load — the first FE route to a real UD compression knockdown,
  since a linear first-ply-failure index never activates for pristine
  UD. Controlled by `AnalysisConfig.enable_progressive_damage`,
  `progressive_n_increments`, `progressive_residual_factor`, and
  `progressive_max_strain`.

A movable wrinkle is supported through
`AnalysisConfig.wrinkle_z_position`, the through-thickness location of
the wrinkle as a fraction of the laminate thickness (`0.5` = mid-plane),
which also feeds the penetration gate's position factor.

In short: **use the angle-based analytical models for multidirectional
laminates, and the penetration gate for unidirectional laminates**,
because at a fixed angle the UD knockdown still varies with
through-thickness penetration.

See the [API reference](api/index.md) for the full public surface and
the [validation page](validation.md) for how these models compare against
experiment.
