# Tutorial: from micrograph to NCR attachment

A worked, end-to-end walkthrough of the workflow a stress engineer
actually runs: **measure** a wrinkle on a micrograph, **configure** the
analysis, **run** it, **read** the numbers, **derive the acceptance
limit**, and **export** the validation summary that gets attached to a
Nonconformance Report.

Every command below is real and every output block is trimmed but
unedited, produced with WrinkleFE 1.0.0 on a minimal install
(`pip install -e .`, no extras). Your own run will differ in the date
and provenance lines the summary records.

The case: a 24-ply quasi-isotropic IM7/8552 panel, `[0/45/-45/90]_3s`,
with a single out-of-plane wrinkle found at the mid-thickness during
ultrasonic inspection and confirmed on a sectioned micrograph.

## 1. Measure the wrinkle

Three numbers come off the micrograph, all in millimetres (see
[Units & conventions](units_conventions.md)):

| Symbol | Config field | How to read it off the section |
|--------|--------------|--------------------------------|
| *A* | `amplitude` | The **half**-amplitude: `A = (z_max − z_min) / 2` of the wrinkled mid-surface against the flat reference. The peak-to-trough height you measure directly is `2A`. |
| *λ* | `wavelength` | Crest-to-crest distance of the underlying cosine carrier along the load direction. |
| *w* | `width` | Longitudinal envelope decay length about the wrinkle centre — the Gaussian 1/e length over which the wrinkle dies away. |

For this panel the section gives a peak-to-trough height of 0.48 mm
(so `A = 0.24 mm`), a crest-to-crest spacing of 18 mm, and a wrinkle
that fades out over roughly 10 mm either side.

The fourth input is the **morphology** — how the wrinkle sits through
the thickness. `stack` (two aligned wrinkles, the dual-wrinkle baseline)
is the right default for an interior wrinkle spanning a ply interface;
`concave` is the most damaging in compression and `convex` the least.
The [overview](overview.md) has the full set.

## 2. Configure

Two equivalent paths.

### CLI

Pass the measurements as flags and write the effective configuration to
a file with `--save-config`, so the run is reproducible and the exact
inputs can be attached to the NCR:

```bash
wrinklefe analyze \
    --amplitude 0.24 --wavelength 18.0 --width 10.0 \
    --morphology stack --loading compression \
    --layup "[0/45/-45/90]_3s" --material IM7_8552 \
    --save-config wrinkle_NCR.json
```

`--save-config` writes every resolved field, not just the ones you
typed:

```text
{
  "amplitude": 0.24,
  "amplitude_profile": "constant",
  "amplitude_profile_axis": "x",
  "amplitude_profile_decay_length": null,
  "analytical_only": false,
  "angles": [
    0.0,
    45.0,
    ...
```

Re-running from the file reproduces the case exactly, and any flag on
the same command line overrides the stored value:

```bash
wrinklefe analyze --config wrinkle_NCR.json                  # reuse verbatim
wrinklefe analyze --config wrinkle_NCR.json --amplitude 0.30 # one override
```

### Streamlit app

```bash
pip install "wrinklefe[streamlit]"
streamlit run app.py
```

The sidebar carries the same fields — *Amplitude A [mm]*, *Wavelength λ
[mm]*, *Envelope width w [mm]*, the morphology selector, the layup and
material pickers. One difference to watch: the app's strain input is
*Applied strain magnitude [%]* and takes the magnitude only (the sign
comes from the loading mode), while the library and CLI take a signed
fraction — `1.0 %` in the app is `applied_strain = -0.01` in compression.
See [Units & conventions](units_conventions.md#percent-vs-fraction-the-one-place-they-differ).

### Python

```python
from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis

config = AnalysisConfig(
    amplitude=0.24, wavelength=18.0, width=10.0,
    morphology="stack", loading="compression",
)
result = WrinkleAnalysis(config).run()
```

`AnalysisConfig.save_json` / `load_json` are the API equivalents of
`--save-config` / `--config`.

## 3. Run

The command above already ran the analysis while saving the config
(about 11 s on this laptop — the analytical path alone is milliseconds;
the FE solve is the rest):

```text
=================================================================
  WrinkleFE Analysis Results
=================================================================

  Configuration:
    Morphology:      stack
    Amplitude:       0.240 mm
    Wavelength:      18.0 mm
    Width:           10.0 mm
    Amplitude profile: constant (d=None, axis=x)
    Loading:         compression
    Applied strain:  -0.0100

  Analytical Predictions:
    Morphology factor M_f:  1.0000
    Max angle theta_max:    4.79 deg (0.0836 rad)
    Effective angle:        4.79 deg (0.0836 rad)
    Damage index D:         0.2253
    Combined knockdown:     0.6865
    Modulus knockdown:      0.9811
    Predicted strength:     823.8 MPa

  Mesh:
    Nodes:    2275
    Elements: 1728
    DOFs:     6825

  FE Results:
    Max displacement: 5.455599e-01 mm
    Modulus retention (local σ₁₁):  0.9950
    Modulus retention (global E_x): 0.9954
=================================================================
```

Add `--analytical-only` to skip the FE solve when you only need the
closed-form screen, or `--no-fe` for the same effect.

## 4. Read the numbers

[Interpreting results](interpreting_results.md) is the full reference;
here is what this run says.

- **Combined knockdown 0.6865.** The predicted compressive strength of
  the wrinkled laminate is 68.65 % of the pristine one — a 31 % strength
  loss — from the Budiansky–Fleck kink-band path at a peak fibre
  misalignment of 4.79°. It is a ratio against the *pristine* laminate,
  not a margin against a design allowable.
- **Predicted strength 823.8 MPa.** The same result against the
  material's `Xc`.
- **Damage index D = 0.2253.** Reporting only — nothing in the knockdown
  is derived from it — but the severity banding reads it as a second,
  independent metric.
- **Modulus knockdown 0.9811 (closed form) and modulus retention
  0.9954 (FE, global).** The wrinkle costs about 31 % of the strength
  but under 2 % of the axial stiffness. That asymmetry is the point:
  **stiffness retention is not strength retention**, and a
  stiffness-based screen would have passed this wrinkle. Of the two FE
  numbers, quote `modulus_retention_global` (the coupon-level reaction
  response), not the local σ₁₁ proxy.

Severity, from the bands in
[Interpreting results](interpreting_results.md#severity-bands): a
knockdown of 0.6865 lands in **Major** (≥ 0.50), while D = 0.2253 lands
in *Moderate* (< 0.40). The worst tier governs, so the recommendation is
Major — governed by residual strength.

## 5. Derive the acceptance limit

The forward question is "what does this wrinkle cost?". The question a
disposition is written around is the inverse: **"how large a wrinkle
could we have accepted?"** `wrinklefe critical` root-finds it:

```bash
wrinklefe critical --config wrinkle_NCR.json \
    --target-knockdown 0.85 --save-config wrinkle_NCR_limit.json
```

```text
============================================================
  WrinkleFE Critical-Value Search: amplitude
  Morphology: stack | Loading: compression | Analytical path
============================================================

  Objective: knockdown >= 0.8500   (direction: decreasing)
  Range:     [0.000000, 4.392000]  (lower: zero, upper: laminate_thickness)
  Scan:      9 points, log-spaced, 1 sign change

       amplitude    Knockdown   Strength (MPa)   phase
------------------------------------------------------------
          0.0000       1.0000           1200.0   scan
          0.0044       0.9886           1186.3   scan
          0.0118       0.9703           1164.4   scan
          0.0316       0.9264           1111.6   scan
          0.0848       0.8353           1002.4   scan
          0.2275       0.6948            833.8   scan
          0.6103       0.5535            664.2   scan
          1.6372       0.4637            556.4   scan
          4.3920       0.4264            511.7   scan
->        0.0748       0.8500           1020.0   verify
------------------------------------------------------------

  Largest acceptable amplitude: 0.074824
  Achieved knockdown:  0.850010  (target 0.850000, rtol 1.0e-03)
  Criterion satisfied: knockdown 0.850010 >= 0.850000
  Forward-model evaluations: 15   (0.25 s)
============================================================
Critical configuration written to: wrinkle_NCR_limit.json
```

Read the `verify` row, not the scan: the answer is **backed off to the
safe side and re-evaluated**, so `0.074824 mm` satisfies
`knockdown >= 0.85` under a real forward run rather than by
interpolation. The measured 0.24 mm is more than three times that limit,
which is why this wrinkle is a Major nonconformance and not a paperwork
exercise.

A refusal is a returned outcome, not an exception — always check
`result.status` and read `result.message` before quoting a limit.

## 6. Export the NCR validation summary

`build_analysis_summary` assembles the geometry, laminate, engineering
results, cited criteria and the *non-binding* recommended disposition;
`export_summary` writes it as Markdown, JSON or PDF. The acceptance
limit rides along only when it was derived for this same configuration.

```python
import numpy as np

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.goalseek import find_critical_value
from wrinklefe.io import build_analysis_summary, export_summary

config = AnalysisConfig.load_json("wrinkle_NCR.json")
result = WrinkleAnalysis(config).run()
limit = find_critical_value(config, target_knockdown=0.85)

summary = build_analysis_summary(
    defect={
        "amplitude_mm": config.amplitude,
        "wavelength_mm": config.wavelength,
        "width_mm": config.width,
        "morphology": config.morphology,
        "loading": config.loading,
        "ply_thickness_mm": config.ply_thickness,
        "n_plies": len(config.angles),
        "layup": config.angles,
        "material_name": config.material.name,
    },
    engineering={
        "analytical_knockdown": result.analytical_knockdown,
        "analytical_strength_MPa": result.analytical_strength_MPa,
        "damage_index": result.damage_index,
        "max_angle_deg": np.degrees(result.max_angle_rad),
        "effective_angle_deg": np.degrees(result.effective_angle_rad),
        "morphology_factor": result.morphology_factor,
        "fe": {
            "modulus_retention": result.modulus_retention,
            "modulus_retention_global": result.modulus_retention_global,
            "retention_factors": result.retention_factors,
        },
    },
    critical_limit={
        "parameter": "amplitude",
        "parameter_units": "mm",
        "objective": "knockdown",
        "target": 0.85,
        "critical_value": limit.critical_value,
        "achieved_knockdown": limit.achieved_knockdown,
        "method": "analytical",
        "n_evaluations": limit.n_evaluations,
        "rtol": limit.rtol,
    },
    reference="NCR-2026-0417",
    prepared_by="R. Stress",
)

export_summary(summary, "NCR-2026-0417-wrinkle.md", fmt="md")
```

The rendered attachment (sections 3 and 5 shown; the file also carries
the date, tool version, provenance, geometry, laminate and cited
criteria):

```markdown
## 3. Engineering analysis (WrinkleFE)

- Analytical knockdown: **0.6865** (68.65% residual strength)
- Predicted strength: 823.8 MPa
- Damage index D: **0.2253**
- Max fibre misalignment: 4.789°
- Effective fibre angle: 4.789°
- Morphology factor: 1

**Finite-element evaluation**

- Modulus retention (local σ₁₁ proxy): 0.995
- Modulus retention (global coupon E_x/E_x0): 0.9954
- Min strength retention: 0.8951

**Acceptance limit (goal-seek)**

- Largest acceptable amplitude: **0.07482** mm
- Criterion: knockdown ≥ 0.85
- Achieved at that value: knockdown 0.85
- Method: analytical path, 15 forward evaluations, rtol 0.001
- Basis: Largest value still satisfying the target under a real forward
  evaluation: the root-find is backed off to the safe side and
  re-verified, so this is not an interpolation of the scan curve.

## 5. Recommended disposition (NON-BINDING)

- **Severity:** Major
- **Recommended path:** REPAIR or REWORK per a qualified procedure with
  full MRB substantiation. Customer/DER concurrence is likely required.
- **Rationale:** Predicted residual strength is 68.7% of pristine
  (≈31.3% strength loss); damage index D = 0.225. Severity is governed
  by residual strength (analytical knockdown). Loading is
  compression-dominated: fibre wrinkles are least tolerant under
  compression (kink-band / micro-buckling driven), so treat the
  recommendation conservatively.
- **Required approvals:** [Design Engineering, Stress, Quality, Customer/DER]
```

The attachment closes with the language every WrinkleFE summary carries:

> This validation summary was prepared with WrinkleFE decision-support
> tooling and is intended as an attachment to a Nonconformance Report.
> The analysis and recommendation are advisory and do not constitute a
> final material disposition. A qualified Material Review Board must
> review, may modify, and formally approve the disposition.

The summary deliberately carries no NCR number, part/serial, work order
or MRB sign-off: that paperwork lives on the NCR this is attached to.
`fmt="json"` writes the structured form and `fmt="pdf"` a paginated PDF;
the Streamlit app exposes the same generator under its summary tab.

## Where to go next

- The wrinkle here is multidirectional. For a **unidirectional**
  laminate in compression the angle-only knockdown is scale-invariant
  and under-predicts the penetration effect — use the penetration gate
  (`--gate li2025-vacbag`) and the progressive-damage FE path
  (`--progressive`). See [getting started](getting_started.md) and
  `examples/11_penetration_gate.py` / `examples/12_progressive_damage.py`.
- Measurement uncertainty on *A* and *λ* propagates to the knockdown:
  `wrinklefe stochastic` (and `examples/14_stochastic_knockdown.py`)
  turn input distributions into percentile knockdowns.
- The `examples/` directory holds runnable scripts for every workflow,
  each with its expected runtime in the header.
