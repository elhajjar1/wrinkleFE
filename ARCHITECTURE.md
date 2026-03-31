# WrinkleFE Architecture

## Module Dependency Diagram

```
core ──→ elements ──→ solver ──→ failure ──→ analysis ──→ gui
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
| `gui/` | PyQt6 desktop application (see below) |

## GUI Architecture

```
WrinkleFEMainWindow (QMainWindow)
  ├── MaterialPanel       ─┐
  ├── WrinklePanel         │  input panels (left dock)
  ├── MeshPanel            │  each emits Qt signals on change
  ├── AnalysisPanel       ─┘
  ├── SweepPanel              parametric sweep tab
  └── results area            plots + summary (central widget)

Signals flow:  panel.configChanged → MainWindow.on_config_changed → validate
               MainWindow.runAnalysis → AnalysisWorker(QThread) → result signal
               MainWindow.runSweep   → SweepWorker(QThread)    → result signal
```

- **Panels** expose input widgets and emit `configChanged` signals.
- **MainWindow** collects panel state into an `AnalysisConfig` and dispatches to workers.
- **AnalysisWorker / SweepWorker** (QThread subclasses) run the computation off the UI thread and emit result signals on completion.
- **Dialogs** (`gui/dialogs/`) handle material editing and export options.

## Confinement Model

The matrix yield strain `gamma_Y` is not a fixed constant -- it depends on how much lateral support neighboring off-axis plies provide to the 0-degree plies that fail by kink-banding. Each 0-degree ply is scored by whether its neighbors are off-axis (confined) or also 0-degree (unconfined). The effective yield strain is then `gamma_Y_eff = 0.032 + 0.050 * f_confined`, where `f_confined` is the weighted confinement fraction. This means dispersed layups like `[0/45/90/-45]s` are significantly more resistant to wrinkle-induced knockdown than blocked layups like `[0_4/90_4]s`.
