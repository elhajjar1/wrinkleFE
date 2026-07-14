# Getting started

## Install

```bash
pip install wrinklefe            # core: analytical + FE pipeline
pip install "wrinklefe[streamlit]"  # + the Streamlit web app
```

From a clone, `pip install -e ".[all]"` installs everything including
the dev tooling.

## A five-minute analysis

```python
from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis

config = AnalysisConfig(
    amplitude=0.366, wavelength=16.0, width=12.0,
    morphology="stack", loading="compression",
)
result = WrinkleAnalysis(config).run()
print(result.summary())
```

`amplitude` is the **half-amplitude** in mm (peak displacement of the
wrinkled mid-surface from the flat reference); `wavelength` is the
crest-to-crest period; `width` is the longitudinal envelope decay
length. All lengths are millimetres. The full parameter reference is
the wrinkle-geometry table in the [overview](overview.md) and the
{class}`~wrinklefe.analysis.AnalysisConfig` API page.

## Unidirectional wrinkle knockdown (penetration gate)

For *unidirectional* laminates the angle-only models under-predict the
scale effect: at a fixed misalignment angle the compressive knockdown
still drops as the wrinkle penetrates deeper through the thickness. The
penetration-gate model captures this from two inputs — the peak angle
`theta_deg` and the through-thickness penetration `D/T = A/T` — using a
calibrated preset:

```python
from wrinklefe.core.penetration_gate import (
    penetration_gate_kd,
    predict_from_geometry,
    GATE_LI2025_VACBAG,
)

# From the two model inputs directly:
kd = penetration_gate_kd(theta_deg=10.0, dt=0.12, params=GATE_LI2025_VACBAG)
print(f"knockdown = {kd:.3f}")

# Or straight from the wrinkle geometry (A, lambda, layup):
kd_geom = predict_from_geometry(
    amplitude=0.5, wavelength=15.0,
    n_plies=14, ply_thickness=0.183,
    params=GATE_LI2025_VACBAG,
)
print(f"knockdown = {kd_geom:.3f}")
```

`GATE_LI2025_VACBAG` (and `GATE_LI2024_MOULDED`) are calibrated to the
Li 2024/2025 UD glass/epoxy grids; `penetration_gate_kd` also takes an
optional `z_position` (0.5 = mid-plane) for the through-thickness
position factor. To route a full `AnalysisConfig` run through the gate,
set `AnalysisConfig.penetration_gate = GATE_LI2025_VACBAG`. The gate is
**UD-scoped** — do not apply it to multidirectional / blocked laminates.

For the FE side, `AnalysisConfig.enable_resin_pocket=True` tags a soft
epoxy lens at the wrinkle crest, and
`AnalysisConfig.enable_progressive_damage=True` load-steps the solve to
ultimate load for a real UD compression knockdown; both are documented
on the {class}`~wrinklefe.analysis.AnalysisConfig` API page.

The same pipeline is available from the command line:

```bash
wrinklefe analyze --amplitude 0.366 --wavelength 16 --morphology stack
wrinklefe sweep --parameter amplitude --min 0.1 --max 0.5 --steps 8 \
    --no-analytical-only --parallel 4   # full-FE sweep on 4 processes
wrinklefe converge --tolerance 0.01   # mesh-convergence study
wrinklefe materials                   # list the material library
```

An `analyze` run can be saved to a config file and reloaded later.
`--save-config PATH` writes the effective configuration; `--config PATH`
reloads it, with any flag on the same command line overriding the file
value:

```bash
wrinklefe analyze --amplitude 0.4 --morphology concave --save-config case.json
wrinklefe analyze --config case.json                  # reuse verbatim
wrinklefe analyze --config case.json --amplitude 0.9  # override one value
```

The same round-trip is available on {class}`~wrinklefe.analysis.AnalysisConfig`
via `to_dict` / `from_dict` and `save_json` / `load_json`.

## Runnable examples

The repository's `examples/` directory contains scripts for the common
workflows — parametric sweeps, morphology comparison, CZM delamination,
multi-wrinkle crest-to-crest delamination link-up, export round-trips,
custom materials, mesh convergence — each with its
expected runtime and output in the header. CI executes them all on
every push.
