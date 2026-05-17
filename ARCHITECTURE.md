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
| `core/material.py` | `OrthotropicMaterial` dataclass + `MaterialLibrary` (4 built-in presets) |
| `core/laminate.py` | `Laminate`, `Ply`, `LoadState`; Classical Lamination Theory ABD matrices |
| `core/wrinkle.py` | `GaussianSinusoidal` wrinkle profile: z(x) = A exp(-x^2/w^2) cos(2pi x/lam) |
| `core/morphology.py` | `WrinkleConfiguration`, `MorphologyFactor`, `MORPHOLOGY_PHASES`; phase-to-Mf mapping |
| `core/mesh.py` | `WrinkleMesh` generates structured hex mesh; returns `MeshData` (nodes, elements, fiber angles) |
| `core/transforms.py` | Stress/strain coordinate transforms between material and global frames |
| `elements/hex8.py` | Standard 8-node hexahedral element stiffness and B-matrix |
| `elements/hex8i.py` | Incompatible-modes hex element (improved bending) |
| `elements/gauss.py` | Gauss quadrature points and weights (1D/3D) |
| `solver/static.py` | `StaticSolver` -- direct or iterative linear solve |
| `solver/assembler.py` | Global stiffness matrix assembly from element contributions |
| `solver/boundary.py` | `BoundaryCondition`, `BoundaryHandler` -- apply Dirichlet/Neumann BCs |
| `solver/results.py` | `FieldResults` -- displacement, stress, strain fields over the mesh |
| `failure/base.py` | `FailureCriterion` ABC and `FailureResult` dataclass |
| `failure/larc05.py` | LaRC05 composite failure criterion implementation |
| `failure/evaluator.py` | `FailureEvaluator` applies criteria across all elements; `LaminateFailureReport` |
| `viz/plots_2d.py` | 2D matplotlib plots (wrinkle profiles, knockdown curves, distributions) |
| `viz/plots_3d.py` | 3D PyVista plots (mesh, damage contours) |
| `viz/style.py` | Publication styling constants and helpers |
| `sweep/parametric_sweep.py` | Parametric sweep over amplitude/wavelength/morphology; CSV output |

## Confinement Model

The matrix yield strain `gamma_Y` is not a fixed constant -- it depends on how much lateral support neighboring off-axis plies provide to the 0-degree plies that fail by kink-banding. Each 0-degree ply is scored by whether its neighbors are off-axis (confined) or also 0-degree (unconfined). The effective yield strain is then `gamma_Y_eff = 0.032 + 0.050 * f_confined`, where `f_confined` is the weighted confinement fraction. This means dispersed layups like `[0/45/90/-45]s` are significantly more resistant to wrinkle-induced knockdown than blocked layups like `[0_4/90_4]s`.

## Morphology Axes: `stack` vs `uniform`

The `MORPHOLOGIES` selectbox in `app.py` flattens **two orthogonal model axes** into one list. `stack`/`convex`/`concave` set the dual-wrinkle inter-ply **phase** φ (`MORPHOLOGY_PHASES` in `core/morphology.py`); `uniform`/`graded` (`SINGLE_WRINKLE_MODES`) set the single-wrinkle through-thickness **amplitude decay** mode. `stack` means "two aligned wrinkles, φ=0"; `uniform` means "one wrinkle, full amplitude at every ply (no decay)". They are not points on the same scale.

In `analysis.py`, `run()` resolves the name via `WrinkleConfiguration.from_morphology_name`: `stack` builds a 2-wrinkle config with φ=0, `uniform` builds a 1-wrinkle config with `decay_mode="uniform"`. `_compute_analytical` then uses `aggregate_morphology_factor` (φ=0 → exp(0) = 1.0 for stack; N=1 → 1.0 for uniform) and the geometry-only angle `theta_max = arctan(2πA/λ)`. **Neither analytical input depends on the through-thickness decay mode**, so all analytical scalars are bit-identical for the two names. Measured (quarter-isotropic layup, compression):

| Config (A / λ / w, mm) | metric | `stack` | `uniform` | Δ |
|---|---|---|---|---|
| 0.366 / 16 / 12 | M_f | 1.000000 | 1.000000 | 0 |
| | effective_angle_deg | 8.178987 | 8.178987 | 0 |
| | analytical_knockdown | 0.605573 | 0.605573 | 0 |
| | damage_index | 0.478676 | 0.478676 | 0 |
| 0.6 / 20 / 10 | analytical_knockdown | 0.568790 | 0.568790 | 0 |
| 0.2 / 12 / 8 | analytical_knockdown | 0.652277 | 0.652277 | 0 |

So on the **analytical axis the two coincide exactly** (Δ = 0 across all reported metrics and configs). They diverge only in the **FE/mesh path**, where `decay_mode` changes `apply_to_nodes`/`fiber_angles_at_nodes`: for config A (`nx=12`) the FE `mesh_max_angle_deg` is ≈10.36 for `stack` (two interface plies at full amplitude, RSS-combined, linear decay to the surfaces) vs ≈7.33 for `uniform` (single wrinkle, no decay). For contrast, `stack` is non-trivial on its own (phase) axis: `convex` gives M_f≈0.750, `concave` M_f≈1.334 versus `stack` M_f=1.0.

**Recommendation:** the UI should expose the two axes separately -- a phase/inter-ply morphology control (`stack`/`convex`/`concave`, plus the existing explicit-`phase` override) and an independent through-thickness amplitude-profile control (`default`/`uniform`/`graded` + `decay_floor`). The current single flattened selectbox makes `stack` and `uniform` look like alternatives on one scale when they are independent; it also hides valid combinations (e.g. a convex dual-wrinkle with uniform decay). This is a UI-only change; the core API (`AnalysisConfig.morphology` + `decay_floor`/`phase`) and analytical/FE behavior are correct as-is and must not change.
