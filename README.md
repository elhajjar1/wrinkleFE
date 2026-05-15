# WrinkleFE

An open-source Python finite element package for predicting strength and stiffness knockdown in composite laminates containing fiber waviness defects.

## Try it in your browser

The fastest way to use WrinkleFE is the hosted Streamlit app — no install required:

### [Launch the WrinkleFE Streamlit app](https://wrinklefe.streamlit.app/)

Pick a material, set the wrinkle amplitude / wavelength / morphology, and the
app returns the analytical knockdown, plots, and (optionally) a full FE solve.
Public link, no account needed.

## Features

- **Compression model:** CLT-weighted Budiansky-Fleck kink-band with layup-dependent confinement
- **Tension model:** Three-mechanism (fiber cos^2 theta, Hashin matrix, curved-beam sigma_33 delamination) with thick-ply in-situ correction
- **3D finite element:** Structured hexahedral mesh with LaRC04/05 failure criteria
- **Five morphologies:** Stack, convex, concave, uniform, graded (with configurable decay floor)
- **Graded averaging:** Through-thickness ply-averaged knockdown for graded wrinkles
- **9 built-in materials:** AS4/3501-6, IM7/8552, T300/914, T700/2510, AC318/S6C10 (S-glass/epoxy), T800S/M21, IM10/8552, S-2 glass/epoxy, Kevlar-49/epoxy
- **Comprehensive test suite** covering all modules (run `pytest` to see the current count)

## Developer / library install

If you want to script against the package or contribute to development:

```bash
pip install wrinklefe
```

Or install the latest source:

```bash
git clone https://github.com/elhajjar1/wrinklefe.git
cd wrinklefe
pip install -e ".[all]"
```

Verify the install:

```bash
python -c "import wrinklefe; print('WrinkleFE installed successfully')"
```

Run the test suite:

```bash
pytest
```

## Quick Start (Python API)

```python
from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis

config = AnalysisConfig(
    amplitude=0.366, wavelength=16.0, width=12.0,
    morphology="stack", loading="compression",
)
result = WrinkleAnalysis(config).run()
print(result.summary())
```

`amplitude` (`A`) is the peak displacement of the wrinkled mid-surface
from the flat reference (crest height, **not** peak-to-peak). For a
measured wrinkle (e.g. from a cross-section micrograph or CT slice),
`A = (z_max − z_min) / 2`. The peak fibre misalignment angle scales as
`θ_max ≈ arctan(2πA/λ)`, which drives the Budiansky-Fleck compressive
knockdown.

### Tension analysis

```python
config = AnalysisConfig(
    amplitude=0.366, wavelength=16.0, width=12.0,
    morphology="stack", loading="tension",
    angles=[0, 45, 90, -45, 0, 45, -45, 0, 0, -45, 45, 0, -45, 90, 45, 0],
    ply_thickness=0.152,
)
result = WrinkleAnalysis(config).run()
print(result.summary())
```

### Graded morphology (embedded wrinkle)

```python
config = AnalysisConfig(
    amplitude=0.5, wavelength=15.0, width=11.0,
    morphology="graded", decay_floor=0.0,
    loading="compression",
)
result = WrinkleAnalysis(config).run()
print(result.analytical_knockdown)
```

### Command line

```bash
wrinklefe --help
```

## Validation

Validated against 31 experimental data points from three independent datasets:

| Dataset | Loading | Cases | Pass | MAE |
|---------|---------|-------|------|-----|
| Elhajjar (2025) | Compression | 13 | 11/13 | 9.9% |
| Elhajjar (2025) | Tension | 7 | 7/7 | 6.2% |
| Mukhopadhyay (2015) | Compression | 3 | 3/3 | 17.4% |
| Mukhopadhyay (2015) | Tension | 3 | 3/3 | 12.1% |
| Li et al. (2026) | Compression | 5 | 4/5 | 8.6% |
| **Total** | | **31** | **28/31** | **9.5%** |

## How It Works

### Compression

CLT-weighted Budiansky-Fleck:

```
KD_lam = f_0 / (1 + theta / gamma_Y_eff) + (1 - f_0)
gamma_Y_eff = 0.032 + 0.050 * f_confined
```

### Tension

Three-mechanism minimum, CLT-weighted:

```
KD_lam = f_0 * min(cos^2(theta), KD_matrix, KD_oop) + (1 - f_0)
```

For graded morphology, the BF knockdown is averaged over the wrinkle profile in both the longitudinal (x) and through-thickness (z) directions, with Gaussian decay (scale = amplitude A) confining the effect to the wrinkle zone.

## References

- Elhajjar, R. (2025). Scientific Reports, 15:25977.
- Li, Y. et al. (2026). Composites Part A, 205:109719.
- Budiansky, B. & Fleck, N.A. (1993). J. Mech. Phys. Solids, 41(1), 183-211.
- Pinho, S.T. et al. (2005). NASA-TM-2005-213530.
- Camanho, P.P. et al. (2006). Composites Part A, 37(2), 165-176.
- Jin, L. et al. (2026). Thin-Walled Structures, 219:114237.

## License

MIT - see [LICENSE](LICENSE)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## Citation

If you use WrinkleFE in your research, please cite:

> Elhajjar, R. (2025). WrinkleFE: An open-source finite element package for strength prediction of wrinkled composite laminates (Version 1.0.0) [Computer software]. University of Wisconsin-Milwaukee. https://github.com/elhajjar1/WrinkleFE

```bibtex
@software{elhajjar2025wrinklefe,
  author = {Elhajjar, Rani},
  title = {{WrinkleFE}: An Open-Source Finite Element Package for Strength
           Prediction of Wrinkled Composite Laminates},
  year = {2025},
  version = {1.0.0},
  publisher = {GitHub},
  url = {https://github.com/elhajjar1/WrinkleFE},
  note = {University of Wisconsin-Milwaukee}
}
```

Please also cite the underlying experimental validation data:

> Elhajjar, R. (2025). Fat-tailed failure strength distributions and manufacturing defects in advanced composites. *Scientific Reports*, 15, 25977. https://doi.org/10.1038/s41598-025-25977-3
