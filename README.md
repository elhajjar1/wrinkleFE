# WrinkleFE

Finite element analysis of wrinkled composite laminates with advanced failure theories.

## Features

- **Analytical knockdown models** for compression (Budiansky-Fleck kink-band) and tension (three-mechanism: fiber, matrix, out-of-plane delamination)
- **Layup-dependent confinement model** computes effective matrix yield strain from stacking sequence rather than using a fixed material constant
- **3D structured hex mesh** with wrinkle geometry from Jin et al. (2026) Gaussian-sinusoidal parameterization
- **Dual-wrinkle morphology** (stack, convex, concave) with physics-based morphology factor
- **LaRC05 failure criteria** with in-situ strengths
- **PyQt6 GUI** for interactive analysis
- **6 built-in carbon/epoxy material systems**

## Installation

```bash
pip install -e ".[all]"
```

Dependencies: numpy, scipy, matplotlib, pyvista, pyvistaqt, PyQt6.

## Usage

### GUI

```bash
wrinklefe-gui
```

### CLI

```bash
wrinklefe --help
```

### Python API

```python
from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis

config = AnalysisConfig(
    amplitude=0.366, wavelength=16.0, width=12.0,
    morphology="concave", loading="compression",
)
analysis = WrinkleAnalysis(config)
result = analysis.run()
print(result.summary())
```

## Analytical Models

### Compression

Budiansky-Fleck kink-band knockdown:

    KD = 1 / (1 + theta_eff / gamma_Y_eff)

The effective yield strain `gamma_Y_eff` is computed from the layup at runtime:

    gamma_Y_eff = 0.02 + 0.226 * f_confined

where `f_confined` is the fraction of 0-degree plies sandwiched between off-axis neighbors. Calibrated against Elhajjar (2025) and Mukhopadhyay (2015).

### Tension

Three competing mechanisms for 0-degree plies, CLT-weighted to laminate level:

1. Fiber tension: cos^2(theta) (LaRC04)
2. Matrix tension: Hashin sigma_22/tau_12 interaction with in-situ strengths
3. Out-of-plane: curved-beam sigma_33 delamination (mode I at crest, mode II at inflection)

    KD_lam = f_0 * min(KD_fiber, KD_matrix, KD_oop) + (1 - f_0)

## Validation

20/20 PASS across Elhajjar (2025) dataset (overall MAE = 4.3%):

- Elhajjar (2025): 13/13 compression, 7/7 tension

## References

- Elhajjar, R. (2025). Scientific Reports, 15:25977.
- Jin, L. et al. (2026). Thin-Walled Structures, 219:114237.
- Budiansky, B. & Fleck, N.A. (1993). J. Mech. Phys. Solids, 41(1), 183-211.
- Pinho, S.T. et al. (2005). NASA-TM-2005-213530.

## License

MIT
