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

## Quick Start

### Streamlit web app

The fastest path is the hosted Streamlit instance — no install, no Python
required: **<https://wrinklefe.streamlit.app/>**. Pick a material from the
sidebar (or **Custom…** to enter your own elastic constants and strength
allowables), enter a layup in contracted notation (e.g. `[0/45/-45/90]_3s`),
set the wrinkle geometry, and click **Run analysis**. The app ships with
per-morphology schematic cartoons, a live wrinkle preview, and the same
analytical + FE pipeline as the Python API. See
[`DEPLOYMENT_STREAMLIT.md`](DEPLOYMENT_STREAMLIT.md) for the full feature
tour and instructions for self-hosting.

To run the app locally:

```bash
pip install -r requirements.txt
pip install -e .
streamlit run app.py
```

### Python API

```python
from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis

config = AnalysisConfig(
    amplitude=0.366, wavelength=16.0, width=12.0,
    morphology="stack", loading="compression",
)
result = WrinkleAnalysis(config).run()
print(result.summary())
```

`amplitude` (`A`) is the **half-amplitude** [mm]: the peak displacement
of the wrinkled mid-surface from the flat (unwrinkled) reference plane,
so `z(x) = A·cos(2πx/λ)` (modulated by the envelope) and the
peak-to-trough height is `2A`. For a measured wrinkle (e.g. from a
cross-section micrograph or CT slice), `A = (z_max − z_min) / 2`. The
peak fibre misalignment angle scales as `θ_max ≈ arctan(2πA/λ)`, which
drives the Budiansky-Fleck compressive knockdown.

### Wrinkle geometry parameters

All wrinkle length parameters use a single, consistent unit:
**millimetres (mm)** — the same unit as `ply_thickness` and
`domain_length` (the default `amplitude=0.366` mm is exactly two ply
thicknesses of 0.183 mm). Lengths are **not** normalized by thickness.
The longitudinal coordinate `x` runs along the laminate in the fibre
direction; out-of-plane displacement `z(x)` is measured from the flat
(undeformed) mid-surface. Angles are in **radians**.

This table is the canonical reference for every wrinkle-geometry
parameter exposed by `AnalysisConfig`, the CLI, and the Streamlit UI.
The `AnalysisConfig` docstrings, CLI `--help`, and Streamlit `help=`
tooltips all mirror these definitions; the
`tests/test_param_docs_match.py` regression test pins the defaults so
the docs and the dataclass cannot drift.

| Parameter | Units | Default | Definition | Constraint |
|-----------|-------|---------|------------|------------|
| `amplitude` (A) | mm | `0.366` | Half-amplitude: peak displacement of the wrinkled mid-surface from the flat reference, with `z(x) = A·cos(2π(x − x₀)/λ)` (modulated by the envelope) and peak-to-trough height `2A`. For a measured wrinkle, `A = (z_max − z_min)/2`. | ≥ 0 (`0` = flat / no wrinkle) |
| `wavelength` (λ) | mm | `16.0` | Spatial period of the `cos(2π(x − x₀)/λ)` carrier along the longitudinal x-direction (crest-to-crest distance). Wavenumber `k = 2π/λ`. | > 0 |
| `width` (w) | mm | `12.0` | Longitudinal envelope decay length about the centre `x₀`. Exact meaning is profile-dependent: Gaussian 1/e length scale in `exp(−(x−x₀)²/w²)`, tapered flat-top extent (`\|x−x₀\| < w/2`), or triangular half-base (`\|x−x₀\| < w`). Also used as the transverse (y-direction) extent of the wrinkle in 3-D dual-wrinkle / graded mesh deformation. | > 0 |
| `phase` (φ) | rad | `None` | Explicit dual-wrinkle phase offset between the two wrinkle centrelines. `None` derives φ from `morphology` via `MORPHOLOGY_PHASES` (stack φ=0, convex φ=+π/2, concave φ=−π/2). A float overrides the named-morphology phase so arbitrary offsets can be swept (e.g. `0` to `π`). Ignored for single-wrinkle morphologies (`uniform`, `graded`). | finite when set |
| `decay_floor` | dimensionless | `0.0` | Graded morphology only: minimum fraction of the wrinkle amplitude retained at the laminate outer surfaces. `0.0` = full decay to zero amplitude at the surfaces (pure graded); `1.0` = no decay (equivalent to `uniform`). | in `[0, 1]` |

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

### Batch parametric sweeps

For exploring how the knockdown varies across a parameter range, use
`WrinkleAnalysis.parametric_sweep` to sweep a single
`AnalysisConfig` field (any numeric field — `amplitude`, `wavelength`,
`width`, `phase`, `applied_strain`, ...):

```python
from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis

base = AnalysisConfig(
    amplitude=0.366, wavelength=16.0, width=12.0,
    morphology="stack", loading="compression",
)
results = WrinkleAnalysis.parametric_sweep(
    base, parameter="amplitude", values=[0.1, 0.2, 0.3, 0.4],
    analytical_only=True,
)
for r in results:
    print(f"A={r.config.amplitude:.3f}  KD={r.analytical_knockdown:.4f}")
```

For multi-parameter cross-product sweeps with JSON output and plots,
use `wrinklefe.sweep.run_sweep`:

```python
import numpy as np
from wrinklefe.sweep import run_sweep, save_sweep_results, plot_sweep_results

sweep = run_sweep({
    "amplitude":  np.linspace(0.183, 0.549, 3),
    "wavelength": np.linspace(8.0, 24.0, 3),
})
save_sweep_results(sweep, "./sweep_output/")
plot_sweep_results(sweep, "./sweep_output/")
```

### Command line

```bash
wrinklefe --help

# Single-parameter sweep (analytical-only is the default, fast)
wrinklefe sweep --parameter amplitude --min 0.1 --max 0.5 --steps 5
```

### Exporting results to CSV / JSON

Numeric outputs (load factor, per-ply failure index, knockdown factors,
stress-field summary) can be written to a schema-versioned JSON or a
Pandas-friendly per-ply CSV for downstream comparison and plotting in
Excel, Pandas, or shared Jupyter notebooks:

```python
from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.io.results import export_results_csv, export_results_json

result = WrinkleAnalysis(AnalysisConfig()).run()

export_results_json(result, "results.json")  # schema-versioned JSON
export_results_csv(result, "per_ply.csv")    # per-ply tabular CSV
```

The JSON output is deterministic (`sort_keys=True`), pins a top-level
`schema_version` field, and reduces large numpy arrays (e.g.
per-Gauss-point stress fields) to summary statistics so the file stays
compact. The CSV is one row per ply with columns `ply_index,
angle_deg, max_FI, min_RF, critical_mode, critical_criterion`, suitable
for `pandas.read_csv` or `csv.DictReader`.

The Streamlit web app exposes the same exports as **Download results as
JSON** and **Download per-ply results as CSV** buttons on the Export
tab.

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

## Supported morphologies

WrinkleFE ships five wrinkle morphologies (defined in
`src/wrinklefe/core/morphology.py`). The first three are *dual-wrinkle*
modes distinguished by the phase offset φ between two adjacent wrinkle
centrelines; the last two are *single-wrinkle* modes that vary the
through-thickness amplitude.

- **`stack`** (φ = 0) — peaks aligned. Morphology factor `M_f = 1.0`,
  used as the baseline.
- **`convex`** (φ = +π/2) — interface bulges outward. `M_f < 1`; the
  *least* damaging mode in compression.
- **`concave`** (φ = −π/2) — interface pinches inward. `M_f > 1`; the
  *most* damaging mode in compression.
- **`uniform`** — single wrinkle at full amplitude on every ply through
  the thickness.
- **`graded`** — single wrinkle whose amplitude decays from the wrinkle
  core toward the surfaces, controlled by `decay_floor` (0 = full
  decay to zero at the surface, 1 = uniform).

## Supported failure criteria

The criteria below live in `src/wrinklefe/failure/` and can be selected
through `FailureEvaluator` or used independently:

- **LaRC04/05** (`larc05.py`) — Pinho/Camanho 3-D criterion with
  fibre-kinking under compression, in-situ matrix strengths, and a
  fracture-plane search. Default for the FE solve.
- **Tsai-Wu** (`tsai_wu.py`) — 3-D tensor-polynomial criterion with a
  configurable interaction coefficient.
- **Tsai-Hill** (`tsai_hill.py`) — 3-D extension of the classical
  quadratic Tsai-Hill index.
- **Hashin** (`hashin.py`) — 3-D Hashin criterion with separate
  fibre-tension/-compression and matrix-tension/-compression modes.
- **Puck** (`puck.py`) — action-plane (Mode A/B/C) inter-fibre-failure
  criterion with simplified fibre failure.
- **Maximum Stress** (`max_stress.py`) and **Maximum Strain**
  (`max_strain.py`) — non-interactive checks against the principal
  material-frame allowables.
- **Budiansky-Fleck kink-band** (`kinkband.py`) — analytical compression
  knockdown with an optional interlaminar damage coupling
  (`InterlaminarDamage`); this is the model exposed in the
  `analytical_knockdown` field of `AnalysisResults`.
- **Progressive damage** (`progressive.py`) — `PlyDiscount` and
  `ContinuumDamage` post-failure stiffness reduction models that wrap
  any of the criteria above.

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
