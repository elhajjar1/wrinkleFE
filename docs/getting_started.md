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

The same pipeline is available from the command line:

```bash
wrinklefe analyze --amplitude 0.366 --wavelength 16 --morphology stack
wrinklefe converge --tolerance 0.01   # mesh-convergence study
wrinklefe materials                   # list the material library
```

## Runnable examples

The repository's `examples/` directory contains scripts for the common
workflows — parametric sweeps, morphology comparison, CZM delamination,
export round-trips, custom materials, mesh convergence — each with its
expected runtime and output in the header. CI executes them all on
every push.
