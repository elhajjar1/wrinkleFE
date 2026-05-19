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

### Wrinkle geometry parameters

All wrinkle length parameters use a single, consistent unit:
**millimetres (mm)** — the same unit as `ply_thickness` and
`domain_length` (the default `amplitude=0.366` mm is exactly two ply
thicknesses of 0.183 mm). Lengths are **not** normalized by thickness.
The longitudinal coordinate `x` runs along the laminate in the fibre
direction; out-of-plane displacement `z(x)` is measured from the flat
(undeformed) mid-surface. Angles are in **radians**.

| Parameter | Symbol | Definition | Units |
|-----------|--------|------------|-------|
| `amplitude` | A | Peak crest height from the flat mid-surface to the wrinkle crest (peak-to-midplane, **not** peak-to-peak; `A = (z_max − z_min)/2` for a measured wrinkle). Must be ≥ 0. | mm |
| `wavelength` | λ | Period of the `cos(2πx/λ)` carrier along the longitudinal x-direction (crest-to-crest distance). Must be > 0. Wavenumber `k = 2π/λ`. | mm |
| `width` | w | Longitudinal envelope decay length about the centre. Exact meaning is profile-dependent: Gaussian length scale `exp(−(x−x₀)²/w²)`, tapered flat-top extent (`\|x−x₀\| < w/2`), or triangular half-base (`\|x−x₀\| < w`). Must be > 0. | mm |
| `center` | x₀ | Longitudinal position of the wrinkle crest / envelope peak, in the global x coordinate. Default 0.0. | mm |
| `phase_offset` | φ | Phase of one wrinkle relative to a reference, mapping to a geometric offset `Δx = φλ/(2π)`. Stack φ=0, convex φ=+π/2, concave φ=−π/2. | rad |
| `decay_floor` | — | Fraction of amplitude retained at the surface plies in `graded` mode (0 = full decay to zero, 1 = no decay). Clamped to [0, 1]. | dimensionless |

Peak fibre misalignment: `θ_max ≈ arctan(2πA/λ)` (exact for a pure
cosine; dimensionless because A and λ share the mm length unit). See the
`WrinkleProfile` class docstring in `src/wrinklefe/core/wrinkle.py` for
the full per-profile geometric definitions.

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

### What the in-repo tests check

The integration tests under `tests/test_integration/` exercise the full
`WrinkleAnalysis` pipeline and assert physical-sanity properties of the
analytical knockdown rather than reproducing absolute experimental
strengths:

- `test_elhajjar_validation.py` (10 tests): zero-amplitude returns
  knockdown ≈ 1, knockdown decreases monotonically with amplitude,
  morphology ordering convex > stack > concave in compression, knockdown
  stays in `(0, 1]`, and strength equals `Xc * knockdown`.
- `test_tension_validation.py` (13 tests): tension pipeline completes,
  uses `Xt` (not `Xc`), three-mechanism (`kd_fiber`, `kd_matrix`,
  `kd_oop`) decomposition is populated with the controlling mode, and
  tension knockdown is no more severe than compression for the same
  defect.

These act as regression guards on the analytical model. They are not a
quantitative validation against experimental data: this repository does
not currently ship the Elhajjar (2025), Mukhopadhyay (2015), or Li et
al. (2026) experimental data points, nor a script that regenerates
case-by-case error statistics. (Tracking issue: #22.)

### Quantitative validation against experiment

Comparison of the analytical predictions against published experimental
data is documented in the accompanying paper:

- Elhajjar, R. (2025). *Fat-tailed failure strength distributions and
  manufacturing defects in advanced composites.* Scientific Reports,
  15:25977. https://doi.org/10.1038/s41598-025-25977-3

Additional datasets referenced by the model calibration (Mukhopadhyay
et al., 2015; Li et al., 2026) are cited in [References](#references)
below. Reproducing case-level pass/fail tables from this repository
alone is not currently possible — the raw data and regeneration script
are not included.

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
