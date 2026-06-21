# WrinkleFE Architecture

## Module Dependency Diagram

```
core ──→ elements ──→ solver ──→ failure ──→ analysis
  │                                            ↑
  └──────────────────── viz                    │
                                     sweep ────┘
```

`core` has no internal dependencies. Each layer depends only on layers to its left.

## Data Flow

```
AnalysisConfig          User-specified parameters (amplitude, wavelength,
       │                morphology, loading, layup, material, mesh density)
       ▼
WrinkleAnalysis         Orchestrates the full pipeline:
       │                  1. Build Laminate from material + stacking sequence
       │                  2. Create GaussianSinusoidal wrinkle profile
       │                  3. Configure WrinkleConfiguration (dual-wrinkle morphology)
       │                  4. Generate WrinkleMesh → MeshData (structured hex)
       │                  5. Assemble & solve (StaticSolver)
       │                  6. Evaluate failure (FailureEvaluator)
       │                  7. Compute analytical knockdown (BF kink-band / tension model)
       ▼
AnalysisResults         Knockdown factors, failure stresses, damage fields,
                        morphology factors, mesh data, field results
```

## Public API

```python
from wrinklefe.analysis import WrinkleAnalysis, AnalysisConfig, AnalysisResults

config = AnalysisConfig(
    amplitude=0.366, wavelength=16.0, width=12.0,
    morphology="concave", loading="compression",
)
result = WrinkleAnalysis(config).run()
print(result.summary())
```

For lower-level access:

```python
from wrinklefe.core.material import OrthotropicMaterial, MaterialLibrary
from wrinklefe.core.laminate import Laminate, LoadState
from wrinklefe.core.wrinkle import GaussianSinusoidal
from wrinklefe.core.morphology import WrinkleConfiguration, MORPHOLOGY_PHASES
from wrinklefe.core.mesh import WrinkleMesh, MeshData
from wrinklefe.solver.static import StaticSolver
from wrinklefe.failure.evaluator import FailureEvaluator
```

## Core Modules

| Module | Description |
|--------|-------------|
| `core/material.py` | `OrthotropicMaterial` dataclass + `MaterialLibrary` (9 built-in presets) |
| `core/laminate.py` | `Laminate`, `Ply`, `LoadState`; Classical Lamination Theory ABD matrices |
| `core/wrinkle.py` | `GaussianSinusoidal` wrinkle profile: z(x) = A exp(-x^2/w^2) cos(2pi x/lam) |
| `core/morphology.py` | `WrinkleConfiguration`, `MorphologyFactor`, `MORPHOLOGY_PHASES`; phase-to-Mf mapping |
| `core/mesh.py` | `WrinkleMesh` generates structured hex mesh; returns `MeshData` (nodes, elements, fiber angles) |
| `core/transforms.py` | Stress/strain coordinate transforms between material and global frames |
| `core/penetration_gate.py` | `GateParameters`, `penetration_gate_kd` -- closed-form UD (theta, D/T, z) penetration-gate knockdown; calibration presets `GATE_LI2024_MOULDED`, `GATE_LI2025_VACBAG` |
| `core/resin_pocket.py` | `ResinPocketSpec`, `compute_resin_mask`, `compute_resin_blend` -- graded neat-epoxy lens at the wrinkle crest |
| `elements/hex8.py` | Standard 8-node hexahedral element stiffness, B-matrix, and `geometric_stiffness_matrix` (initial-stress K_geo) |
| `elements/hex8i.py` | Incompatible-modes hex element (improved bending) |
| `elements/gauss.py` | Gauss quadrature points and weights (1D/3D) |
| `solver/static.py` | `StaticSolver` -- direct or iterative linear solve |
| `solver/assembler.py` | `GlobalAssembler` -- global stiffness assembly from element contributions; also `assemble_geometric_stiffness` (K_geo) for buckling |
| `solver/boundary.py` | `BoundaryCondition`, `BoundaryHandler` -- apply Dirichlet/Neumann BCs |
| `solver/results.py` | `FieldResults` -- displacement, stress, strain fields over the mesh |
| `solver/nonlinear.py` | `NewtonRaphsonSolver` -- displacement-controlled nonlinear static solve |
| `solver/arclength.py` | `ArcLengthSolver` -- Crisfield cylindrical arc-length continuation |
| `solver/progressive_damage.py` | `ProgressiveDamageSolver`, `ProgressiveDamageResult` -- load-stepping ply-discount damage to ultimate load (optional crack-band fibre-mode regularization) |
| `solver/buckling.py` | `LinearBucklingSolver`, `BucklingResult`, `microbuckling_knockdown` -- linearized (eigenvalue) buckling from the geometric stiffness |
| `failure/base.py` | `FailureCriterion` ABC and `FailureResult` dataclass |
| `failure/larc05.py` | LaRC05 composite failure criterion implementation |
| `failure/evaluator.py` | `FailureEvaluator` applies criteria across all elements; `LaminateFailureReport` |
| `viz/plots_2d.py` | 2D matplotlib plots (wrinkle profiles, knockdown curves, distributions) |
| `viz/plots_3d.py` | 3D PyVista plots (mesh, damage contours) |
| `viz/style.py` | Publication styling constants and helpers |
| `sweep/parametric_sweep.py` | Parametric sweep over amplitude/wavelength/morphology; CSV output |
| `analysis.py` | Top-level orchestrator: `WrinkleAnalysis`, `AnalysisConfig`, `compare_morphologies`, `parametric_sweep` |
| `cli.py` | Entry point referenced by `[project.scripts]` (the `wrinklefe` command) |
| `io/export.py` | Native JSON, Abaqus `.inp`, and legacy VTK export (no extra dependencies) |

## Wrinkle-defect modelling

A group of modules adds defect-aware physics on top of the baseline
wrinkled-mesh FE path. Each is opt-in and composes with the others through
`MeshData`'s per-element material override.

- **Resin-pocket material zone** (`core/resin_pocket.py`). The machined cosine
  insert that creates the wrinkle in the Li UD glass/epoxy datasets is co-cured
  bulk epoxy, so the lens it leaves at the crest is fibre-free, soft, isotropic
  matrix rather than homogenised composite. `ResinPocketSpec` describes the
  lens geometry; `compute_resin_mask` flags the hex elements whose centroids
  fall inside it and `compute_resin_blend` produces a graded `(1 - w) * host +
  w * resin` weight. The defect is counted once: the modulus is blended in and
  the fibre-misalignment angle is scaled by `(1 - w)` (`MeshData.resin_angle_scale`)
  so the resin and kink-band paths do not double-count. The neat-epoxy card is
  `EPOXY_S6C10`, built via `OrthotropicMaterial.isotropic`.

- **Progressive-damage FE** (`solver/progressive_damage.py`). The linear path
  reports only a first-ply failure index, so for UD compression (where the
  pristine LaRC05 index never activates) it returns no knockdown. The
  `ProgressiveDamageSolver` ramps the applied strain in increments, re-solving
  with the `StaticSolver` and degrading newly-failed elements (ply-discount, by
  failure-mode family) until each increment settles; the peak carried stress
  over the history is the ultimate strength. An optional crack-band (Bažant-Oh)
  regularization makes the dominant fibre-compression mode mesh-objective. This
  is the first FE route to a real UD compression knockdown.

- **Penetration gate** (`core/penetration_gate.py`). The angle-only
  Budiansky-Fleck knockdown is scale-invariant and cannot reproduce the strong
  dependence of compressive strength on through-thickness penetration `D/T`.
  The closed-form UD predictor is `KD = 1 - (1 - KD_angle(theta)) * S(D/T) * P(z)`,
  with `GateParameters` carrying the calibrated `(gamma_Y, dt0, p)` (and an
  optional position exponent). `penetration_gate_kd` / `predict_from_geometry`
  evaluate it, `calibrate_gate` least-squares-fits it, and the presets
  `GATE_LI2024_MOULDED` / `GATE_LI2025_VACBAG` cover the two AC318/S6C10
  realizations. It is wired into the analysis via `AnalysisConfig.penetration_gate`.

- **Linear buckling / geometric stiffness** (`solver/buckling.py`). Compressive
  failure of a wrinkled UD laminate is at root a geometric instability that a
  fixed-geometry static solve cannot capture. `elements/hex8.py` supplies the
  per-element `geometric_stiffness_matrix` and `solver/assembler.py` the
  `assemble_geometric_stiffness` (K_geo); `LinearBucklingSolver` then solves the
  generalized eigenproblem `K phi = -lambda K_geo phi`, and
  `microbuckling_knockdown` returns the ratio of wrinkled to pristine critical
  load factors. Kept as structural-buckling infrastructure.

- **Movable through-thickness position** (`AnalysisConfig.wrinkle_z_position`).
  The wrinkle centre's through-thickness position is a fraction of the laminate
  thickness (`0.5` = mid-plane); the graded per-ply decay and the resin lens
  centre there, and the penetration gate's position factor `P(z)` makes a
  near-surface wrinkle far milder than a mid-plane one.

## Logging

Library modules report progress through standard `logging` with one module-level logger each (`logging.getLogger(__name__)`), all under the `wrinklefe` hierarchy (e.g. `wrinklefe.analysis`, `wrinklefe.solver.static`, `wrinklefe.failure.evaluator`). Conventions:

- **DEBUG** — per-element/per-iteration detail (assembly progress, Newton residuals, preconditioner setup).
- **INFO** — one-line milestones (mesh built, solve timings, failure-criteria peaks, analysis complete with knockdown).
- **WARNING** — degraded-but-continuing situations (non-converged increments, preconditioner fallback, mesh decay-ratio feasibility).

The library never calls `logging.basicConfig` or attaches handlers; that is left to the application. The CLI's `-v`/`--verbose` flag attaches a stderr handler and sets the `wrinklefe` logger to DEBUG; without it the default output is unchanged. The legacy `verbose=` parameters on solver APIs are deprecated and ignored.

## Confinement Model

The matrix yield strain `gamma_Y` is not a fixed constant -- it depends on how much lateral support neighboring off-axis plies provide to the 0-degree plies that fail by kink-banding. Each 0-degree ply is scored by whether its neighbors are off-axis (confined) or also 0-degree (unconfined). The effective yield strain is then `gamma_Y_eff = 0.032 + 0.050 * f_confined`, where `f_confined` is the weighted confinement fraction. This means dispersed layups like `[0/45/90/-45]s` are significantly more resistant to wrinkle-induced knockdown than blocked layups like `[0_4/90_4]s`.
