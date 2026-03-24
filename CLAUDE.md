# CLAUDE.md

Project instructions for Claude Code working in the WrinkleFE repository.

## Overview

WrinkleFE is a Python package for predicting strength retention in composite laminates with fiber waviness (wrinkle) defects. It combines analytical knockdown models with a 3D finite element pipeline and a PyQt6 GUI.

## Repository Layout

```
wrinklefe/
  src/wrinklefe/
    analysis.py          # High-level analysis pipeline (WrinkleAnalysis, AnalysisConfig)
    cli.py               # Command-line entry point
    core/
      material.py        # OrthotropicMaterial dataclass + MaterialLibrary (4 built-in materials)
      laminate.py         # Laminate, Ply, LoadState
      wrinkle.py          # GaussianSinusoidal wrinkle profile
      morphology.py       # WrinkleConfiguration, MorphologyFactor, MORPHOLOGY_PHASES
      mesh.py             # WrinkleMesh → MeshData (structured hex mesh)
      transforms.py       # Stress/strain coordinate transforms
    elements/
      hex8.py             # 8-node hexahedral element
      hex8i.py            # Incompatible-modes hex element
      gauss.py            # Gauss quadrature
    solver/
      static.py           # StaticSolver (direct/iterative)
      assembler.py        # Global stiffness assembly
      boundary.py         # BoundaryCondition, BoundaryHandler
      results.py          # FieldResults (displacement, stress, strain)
    failure/
      base.py             # FailureCriterion ABC, FailureResult
      larc05.py           # LaRC05 implementation
      evaluator.py        # FailureEvaluator, LaminateFailureReport
    gui/
      main_window.py      # PyQt6 GUI (wrinklefe-gui entry point)
      panels/             # GUI panel widgets
      dialogs/            # GUI dialog widgets
    viz/
      plots_2d.py         # 2D matplotlib plots
      plots_3d.py         # 3D PyVista plots
      style.py            # Publication styling
    io/
      export.py           # Mesh/results export (meshio)
  validation/
    validate_elhajjar2025.py   # Compression + tension validation (13+7 cases)
    validate_jin2026.py        # Jin et al. damage/morphology validation
    validate_tension_analytical.py  # Tension model unit tests
    reference_data.json        # Experimental reference data
  tests/                 # pytest test suite
  joss/                  # JOSS paper (paper.tex, paper.bib)
  pyproject.toml         # Package metadata, dependencies, tool config
```

## Analytical Strength Models

### Compression: Budiansky-Fleck Kink-Band

The compression knockdown uses kink-band theory only (no separate damage multiplier):

```
KD = 1 / (1 + theta_eff / gamma_Y_eff)
```

where `theta_eff = theta_max * M_f(phi, loading)` and `gamma_Y_eff` is the layup-dependent effective matrix yield strain (see below).

Implementation: `analysis.py`, lines ~644-648.

### Confinement Model for gamma_Y

The matrix yield strain `gamma_Y` is not a fixed material constant. It is computed at runtime from the laminate stacking sequence:

```
gamma_Y_eff = 0.032 + 0.050 * f_confined
```

where `f_confined` is the weighted confinement fraction of 0-degree plies. Each 0-degree ply scores 1.0 if both neighbors are off-axis, 0.5 if one neighbor is off-axis, and 0.0 if both neighbors are 0-degree (block interior). This partial-confinement model handles both dispersed and blocked layups.

The compression knockdown uses CLT weighting to separate confinement from load redistribution:

```
KD_lam = f_0 * 1/(1 + theta_eff/gamma_Y_eff) + (1 - f_0)
```

where `f_0` is the axial stiffness fraction carried by 0-degree plies.

Calibrated (with CLT weighting) against Elhajjar (2025) and Mukhopadhyay et al. (2015):
- UD `[0]_n`: f_confined = 0.0, gamma_Y = 0.032
- Mukhopadhyay `[+45_2/90_2/-45_2/0_2]_3s`: f_confined = 0.417, gamma_Y = 0.053
- Elhajjar `[0/45/90/-45/0/45/-45/0]_s`: f_confined = 0.833, gamma_Y = 0.074

Constants: `_GAMMA_Y_UD = 0.032`, `_ALPHA_CONF = 0.050` in `analysis.py`.

Functions: `_confined_fraction(angles)` and `_effective_gamma_Y(angles)` in `analysis.py`.

References: Elhajjar (2025) Scientific Reports 15:25977.

### Tension: Three-Mechanism Model

Tension knockdown uses three competing failure mechanisms for the 0-degree plies:

1. **Fiber tension (LaRC04 criterion #3):** `KD_fiber = cos^2(theta)` (Pinho Eq. 82)
2. **Matrix tension (LaRC04 criterion #1):** Hashin sigma_22/tau_12 interaction with in-situ strengths `Yt_is = 1.12*sqrt(2)*Yt`, `S12_is = sqrt(2)*S12` (Pinho Eqs. 40, 47, 57)
3. **Out-of-plane delamination (curved-beam):** sigma_33 at crest (mode I) and tau_13 at inflection (mode II). Effective thickness = max consecutive 0-degree plies times ply thickness.

The 0-degree ply knockdown is the minimum of all three:

```
KD_0 = min(KD_fiber, KD_matrix, KD_oop)
```

Laminate-level knockdown uses CLT axial stiffness weighting:

```
KD_lam = f_0 * KD_0 + (1 - f_0) * 1.0
```

where `f_0` is the axial stiffness fraction carried by 0-degree plies.

Implementation: `WrinkleAnalysis._tension_knockdown_analytical()` in `analysis.py`.

### GIc/GIIc Not Used in GUI

Fracture toughness values (GIc, GIIc) cancel in the wrinkled/pristine retention ratio and do not affect knockdown predictions. They have been removed from the GUI material inputs. The fields remain in `OrthotropicMaterial` as `Optional[float]` for advanced use but are set to `None` in the GUI pipeline.

## Material Library

Four built-in materials in `MaterialLibrary` (`core/material.py`):

1. `AS4_3501_6`
2. `IM7_8552` (default)
3. `T300_914`
4. `T700_2510`

## Validation

**20/20 PASS across Elhajjar (2025) dataset (overall MAE = 4.3%):**

| Dataset | Loading | Cases | Pass | MAE |
|---------|---------|-------|------|-----|
| Elhajjar (2025) | Compression | 13 | 13/13 | ~3.5% |
| Elhajjar (2025) | Tension | 7 | 7/7 | ~5.5% |

Validation scripts in `validation/`. Run with:

```bash
python validation/validate_elhajjar2025.py
```

## Running

```bash
# Install
pip install -e ".[all]"

# CLI
wrinklefe --help

# GUI
wrinklefe-gui

# Tests
pytest
```

## Key References

- Elhajjar, R. (2025). Scientific Reports, 15:25977.
- Jin, L. et al. (2026). Thin-Walled Structures, 219:114237.
- Budiansky, B. & Fleck, N.A. (1993). J. Mech. Phys. Solids, 41(1), 183-211.
- Pinho, S.T. et al. (2005). NASA-TM-2005-213530 (LaRC04/05).
- Timoshenko, S.P. & Gere, J.M. (1961). Theory of Elastic Stability.
