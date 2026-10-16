---
title: 'WrinkleFE: Scope-aware strength knockdown prediction for composite laminates with out-of-plane fibre waviness'
tags:
  - Python
  - composite materials
  - fibre waviness
  - wrinkle defects
  - finite element analysis
  - failure analysis
authors:
  - name: Rani F. Elhajjar
    orcid: 0000-0003-4778-5186
    affiliation: 1
affiliations:
  - name: University of Wisconsin-Milwaukee, Milwaukee, Wisconsin, United States
    index: 1
    ror: 031q21x57
date: 16 October 2026
bibliography: paper.bib
---

# Summary

Out-of-plane fibre waviness, or wrinkling, is one of the most common manufacturing defects in fibre-reinforced polymer laminates [@luong2025outofplane]. A wrinkle misaligns the load-bearing fibres and can remove half or more of a laminate's compressive strength, so a part found to contain one must be repaired, scrapped or shown by analysis to remain fit for service. WrinkleFE is an open-source Python package that predicts the strength and stiffness knockdown of a laminate containing a wrinkle. It combines closed-form models, which run in milliseconds, with a three-dimensional finite element (FE) solver, and it reports how far each prediction can be trusted for the case at hand, including, where the case's regime has enough validation cases, a margin derived from them. It is used as a library, from the command line, and through a hosted web application that needs no installation.

# Statement of need

Engineers dispositioning a wrinkled part, and researchers studying wrinkle mechanics, currently choose between coupon testing, which is slow and costly, and bespoke FE models, which take expertise and time to build and are rarely checked against measured strengths. Published closed-form models, such as the kink-band theory of @budiansky1993compressive, apply only under assumptions that are hard to check for a given part. No open tool known to the author brings these models together, checks them against measured strengths published by several laboratories, and tells the user when a case falls outside the evidence.

WrinkleFE addresses that need in three ways. First, it implements the models a wrinkle analysis requires behind one configuration object: closed-form compression and tension knockdowns, a penetration-depth model for unidirectional laminates, and an FE path with ply-level failure criteria, cohesive-zone delamination and progressive damage. Second, every prediction path is scored against published measured strengths in a validation record kept under version control and checked by the test suite. Third, the package reports the limits of that evidence alongside each number, rather than leaving them in the literature. Analyses bound a wrinkle that moves the laminate's load path, where the kink-band model alone was unsafe in validation, and flag one-sided surface troughs, which that bound does not cover, manufacturing processes outside the validated scope, and calibrated models used outside the material they were fitted to. A non-conformance summary (JSON, Markdown or PDF) packages the analysis, the criteria cited and these caveats for engineering review.

The target users are composites researchers, quality and materials-review engineers in aerospace and wind energy, and educators.

# State of the field

General-purpose FE codes, commercial ones such as Abaqus or open-source frameworks such as FEniCS [@alnaes2015fenics], can model a wrinkled laminate, but the user must build the wavy ply geometry, assign ply-wise material orientations through the wrinkle, implement composite failure criteria such as LaRC04/05 [@pinho2005failure; @pinho2012material], Hashin [@hashin1980failure] or Puck [@puck1998failure], and validate the result. Most wrinkle studies have done exactly this, case by case [@mukhopadhyay2015compressive; @thor2021numerical; @wang2021experimental]. The models in those studies are not distributed as reusable software, and their validity outside the original specimens is rarely tested.

We considered building on a general FE framework instead of writing a solver. We chose a self-contained package because the value lies in the domain layer: wrinkle morphologies, laminate mechanics, failure criteria, calibrated closed-form models and the validation record. A purpose-built structured hexahedral solver keeps that layer installable with `pip` and fast enough for a web application. It also keeps the analytical and FE paths on the same geometry, so they can be compared directly.

# Software design

The package is organised around `AnalysisConfig`, a serialisable dataclass describing the laminate, the material, the wrinkle and the loading, and `WrinkleAnalysis`, which runs the requested paths and returns an `AnalysisResults` object.

- **Geometry** (`wrinklefe.core`): Gaussian-enveloped sinusoidal wrinkles with several through-thickness morphologies. These include a one-sided "tool-flat" profile with resin-filled surface pockets, multi-wrinkle layouts and through-width envelopes.
- **Closed-form models**: a laminate-weighted Budiansky-Fleck kink-band model in compression, whose strain budget comes from a layup confinement model for multidirectional laminates and, on the default peak-angle path, from the material's own shear and compressive strengths for unidirectional ones; an eccentric-column bending bound for whole-thickness waves that move the load path; a three-mechanism model in tension; and a two-parameter penetration-depth model with calibrated presets for unidirectional laminates.
- **Finite elements** (`wrinklefe.elements`, `wrinklefe.solver`, `wrinklefe.failure`): hexahedral elements on a mesh that follows the wrinkled plies; LaRC05, Hashin, Puck and other failure criteria; bilinear cohesive interfaces with a per-increment damage record and an exact delamination-onset load; linear buckling; and progressive damage with crack-band regularisation [@bazant1983crack].
- **Scope reporting**: the load-path eccentricity of the wrinkled section and which bound governed; the manufacturing-process scope the validation data support; the fitted scope of calibrated presets; and a validation margin for the case's regime (model path, load, laminate type and bending regime). The margin is the factor that would have kept 95% of that regime's validation cases conservative, from a two-component Gaussian-mixture fit to the residuals [@elhajjar2025fat]; a regime with fewer than ten cases reports insufficient evidence instead, and three of the eight regimes currently have enough. None of this changes a knockdown; all of it reaches the schema-versioned results JSON, the non-conformance summary and the web application.
- **Interfaces**: a Python API, a command-line interface with sweeps and an inverse "acceptable limit" search, and a Streamlit application.

The validation ledger (`tests/test_validation/ledger.json`) records the source, recipe, manufacturing route, laminate quality and measured values of 12 of the 13 datasets and pins each expected prediction, so any change in a model shows up as test drift; the thirteenth, a dataset the penetration-depth presets were fitted to, is held in the validation scripts. Datasets of doubtful quality can be quarantined: kept as regression guards but excluded from pooled statistics. The test suite, run in continuous integration on Linux and macOS for Python 3.10 to 3.13, contains about 3,000 fast tests plus about 130 slower FE checks.

# Research impact statement

WrinkleFE is scored against measured strengths from 13 datasets, labelled A to M [@elhajjar2025fat; @mukhopadhyay2015compressive; @mukhopadhyay2015tensile; @wang2021experimental; @li2024effects; @li2025experimental; @hsiao1996effect; @shi2025wrinkled; @calvo2023distributed; @thor2021numerical; @pilato2022experimental; @krishnappa2024compressive; @elhajjar2016limit; @elhajjar2011gaussian], published by 10 research groups, one of them the author's own (Datasets A, B and M, 33 of the 74 pooled cases). They cover compression and tension, unidirectional and multidirectional laminates, and glass and carbon fibres. A head-to-head matrix runs every applicable model on every dataset, and the documentation turns the results into model-choice guidance with a stated confidence for each regime. \autoref{fig:parity} shows each analytical strength prediction against its measurement. The comparison has already changed the models, not only their warnings. Reading the unidirectional kink-band strain budget from the material card moved two UD carbon datasets from 58% and 78% conservative on average to 17% and 20%, with no case turning unsafe. Bounding a wrinkle that moves the load path with an eccentric-column model took the worst whole-thickness cases from 30% and 52% unsafe to 10% and 15% unsafe. The tension model no longer borrows the compression knockdown as a floor; that floor had set every prediction on one steep-wrinkle dataset and was unsafe on 10 of its 13 cases; the corrected model's errors there run from 23% conservative to 18% unsafe. Two size-aware FE measures, the progressive-damage peak and the cohesive delamination onset, did not respond to wrinkle size and were up to 139% unsafe on embedded wrinkles, so users are directed to the analytical knockdown there. Scoring one dataset's surface troughs as the one-sided troughs they are, rather than as embedded wrinkles (a correction made after the earlier errors were known), turned predictions up to 115% unsafe into conservative ones except a 5% over-prediction on the deepest trough, which is flagged as bending-dominated; the software now steers users to that geometry. Calibrated penetration-depth presets do not transfer between materials, and the analytical estimate of tension damage onset proved unreliable, so it is reported as unvalidated. First-ply FE, conservative against the ultimate tensile strength, is up to about 150% unsafe against measured damage onset. With these changes 61 of the 74 non-quarantined cases in \autoref{fig:parity} sit within 20% of the measured ultimate strength, up from 54, and none is more than 20% unsafe, down from 11. These results are in-sample: the corrections were developed with these data in view, and 15 of the 61 are the two glass datasets the presets were fitted to; with the default kink-band model on those, 46 of the 74 are within 20%, and still none is more than 20% unsafe. First-ply FE and onset estimates are plotted in the documentation.

![Predicted against measured strength knockdown for the analytical strength cases of Datasets A to F and H to M (Dataset G, a single strength, is not plotted). Each case is predicted by the analytical model that applies to it: the kink-band or tension model by default and, for the two unidirectional glass datasets (E and F), the penetration-depth presets fitted to them. The shaded band is $\pm$20%. Hollow markers are the quarantined wet lay-up dataset, which is left out of the pooled count. 61 of the 74 pooled cases fall inside the band, and every case outside it is conservative.\label{fig:parity}](parity.png)

The package was first released in March 2026 (version 1.0.0), and releases from version 1.2.0 onward are archived on Zenodo [@wrinklefe_zenodo]. Students in a composite design course at the University of Wisconsin-Milwaukee have used it, and its public web application lets engineers without FE experience run an analysis in a browser.

# AI usage disclosure

Generative AI was used in developing this software and in drafting this paper. The author conceived, designed and directed WrinkleFE and is responsible for its scientific content. The author defined its scope and modelling approach, chose the models it implements, supplied three validation datasets from the author's own experiments (A, B and M), selected the published ones, and made the scientific judgements that shaped the results: which data to trust, how each specimen geometry is modelled, which model corrections to adopt, and where the software must warn rather than predict.

The author used Claude Code (Anthropic), an agentic coding assistant running Anthropic's Claude models, as an implementation tool under that direction: for writing code, tests and documentation, searching the literature, transcribing published validation data, and drafting text for this manuscript. Most of the code and documentation was written this way. Assisted commits carry a `Co-Authored-By` trailer, which names the model from September 2026 onward (Claude Opus 5 and 5.5, Fable 5 and 5.1); earlier ones name only "Claude". Every change was merged through a pull request after the full test suite passed and a review window for the author. Transcribed values were checked against their sources and are pinned by tests.

# References
