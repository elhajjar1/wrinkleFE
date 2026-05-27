# Cohesive Zone Modeling — Implementation Plan

Adding intrinsic CZM (zero-thickness interface elements with a bilinear
traction-separation law) to wrinkleFE to predict interlaminar delamination
at composite wrinkles.

Reference for the constitutive math:
<https://bleyerj.github.io/comet-fenicsx/tours/interfaces/intrinsic_czm/intrinsic_czm.html>

---

## 1. Why — how CZM links to wrinkleFE's objectives

wrinkleFE today predicts **first-failure knockdown** via two analytical
routes (Budiansky-Fleck kink-band for compression; three-mechanism for
tension, one of which is a curved-beam σ₃₃ delamination check), backed by a
3D **linear** FE solve plus ply-level criteria (LaRC05, Hashin, Puck, etc.).
The σ₃₃ delamination check in `failure/larc05.py` is a **point-in-time
trigger**: it tells you whether the interlaminar stress at the wrinkle crest
exceeds an allowable, not whether a crack would actually propagate, arrest,
or coalesce.

CZM replaces that trigger with a physical interface constitutive law:
zero-thickness elements between plies carry a traction-separation response
with an elastic branch, a peak strength, and a softening branch whose
enclosed area equals the mode-specific fracture toughness `Gc`. A damage
variable `d ∈ [0, 1]` accumulates monotonically per integration point.

### How this improves existing predictions

| Current capability | After CZM |
|---|---|
| Tension knockdown uses an analytical σ₃₃ check at the crest — yes/no | Resolve actual interlaminar crack initiation **and propagation** along the wrinkle; predict residual stiffness after onset |
| Compression knockdown via BF kink-band only | Capture secondary delamination that follows kink-band formation and accelerates collapse |
| First-failure load only | Full load-displacement curve to ultimate failure; distinguish stable vs unstable crack growth |
| Morphology effects through `MorphologyFactor` heuristic | Mechanistic explanation of why concave/stack/graded differ — interface mixed-mode angle changes with morphology |
| No size effect | R-curve and size effect emerge naturally |

### New features CZM unlocks

1. Mixed-mode delamination maps at wrinkle crests (mode-I at the peak,
   mode-II on the flanks) via Benzeggagh-Kenane.
2. Crack-front tracking along the wrinkle as a new field in
   `AnalysisResults`.
3. Energy-dissipation budget per interface — identifies the weakest ply
   boundary.
4. Layup-design lever: stacking sequence affects which interfaces see
   mode-I vs mode-II exposure.
5. Pathway to fatigue and R-curve extensions later (Paris-law on `d`,
   fiber-bridging traction tail).

---

## 2. Implementation phases

### Phase 0 — Nonlinear solver (prerequisite)

Today `solver/static.py` is a single linear solve via `spsolve` or CG.
Intrinsic CZM is nonlinear from the first load increment (the T-δ curve has
a peak then softens), so we need:

- New `solver/nonlinear.py` with `NewtonRaphsonSolver`:
  outer load-increment loop, inner Newton with simple backtracking line
  search, residual `R = F_ext − F_int`, convergence on `‖R‖` and `‖Δu‖`.
- Existing linear path stays the default; nonlinear is opt-in.
- `assembler.py` gains `assemble_internal_force()` alongside the existing
  `assemble_stiffness()`; existing behavior unchanged.
- Arc-length continuation is out of scope for v1.

### Phase 1 — Cohesive element

New `elements/cohesive8.py`:

- 8-node zero-thickness interface element (4 bottom + 4 top, coincident in
  the reference configuration), 2×2 Gauss integration on the mid-surface.
- Intrinsic bilinear law:
  - Effective opening
    `δ_eff = √(⟨δ_n⟩₊² + β² (δ_s² + δ_t²))`, default β = 1.
  - Damage `d` monotonic, clamped to [0, 1].
  - Traction `T_i = (1 − d) K δ_i`; for δ_n < 0 use penalty `K δ_n` (no
    damage accumulation from compression).
- Mode mixity via Benzeggagh-Kenane,
  `Gc(ψ) = GIc + (GIIc − GIc) (G_II / (G_I + G_II))^η`, default η = 1.45.
- Per-Gauss-point state: `(d, δ_max_eff)`.

### Phase 2 — Mesh & assembler hooks

- `core/mesh.py`: `WrinkleMesh.insert_interface_elements(interfaces)` that
  duplicates nodes on the chosen ply-boundary z-planes and emits a parallel
  `interface_elements` array.
- `MeshData`: add `interface_elements` and `interface_node_pairs`.
- `solver/assembler.py`: element-type dispatch so bulk hex8 and cohesive8
  elements coexist; identical global-DOF scatter logic.

### Phase 3 — Config & orchestration

Additive, all optional, on `AnalysisConfig`:

```python
enable_czm: bool = False
czm_interfaces: list[int] | str = "near_crest"   # or "all"
czm_law: str = "bilinear"                         # v1
czm_GIc: float = 0.28        # N/mm
czm_GIIc: float = 0.79
czm_sigma_max: float = 60.0  # MPa
czm_tau_max: float  = 90.0   # MPa
czm_penalty: float  = 1e6    # N/mm³
czm_BK_eta: float   = 1.45
n_load_increments: int = 10
newton_tol: float = 1e-4
```

`MaterialLibrary` presets gain default `(GIc, GIIc, σ_max, τ_max)` from
literature; sources documented in `CITATION.cff` style.
`WrinkleAnalysis.run()` branches on `enable_czm` — if true, builds interface
elements and switches to `NewtonRaphsonSolver`.

### Phase 4 — Reporting

`FieldResults` / `AnalysisResults` gain (only populated when CZM is on):

- `czm_damage` — `(n_iface_elems, n_gauss)`
- `czm_separation` — `(n_iface_elems, n_gauss, 3)` in the local frame
- `czm_traction` — `(n_iface_elems, n_gauss, 3)`
- `energy_dissipated: float` and `energy_per_interface: dict[int, float]`
- `crack_length_per_interface: dict[int, float]` (span where `d > 0.99`)
- `load_displacement_curve` — `(n_inc, 2)`

`failure/delamination.py` is a thin reporter consuming `czm_damage` and
emitting a uniform `LaminateFailureReport`. The existing curved-beam σ₃₃
check stays alive as a side-by-side calibration reference.

### Phase 5 — Visualization

- `viz/plots_3d.py`: PyVista — color interface elements by `d`, crack-front
  contour at `d = 0.5`.
- `viz/plots_2d.py`: traction-separation curve at a probe Gauss point,
  load-displacement curve with the linear-elastic prediction overlaid,
  energy-dissipation bar chart per interface.

### Phase 6 — Tests

Mirror existing `tests/` layout:

1. `tests/elements/test_cohesive8_law.py` — single-element mode-I area
   equals GIc within 1%; irreversibility on unload-reload.
2. `tests/elements/test_cohesive8_mixed_mode.py` — Benzeggagh-Kenane
   envelope honored across mixed-mode angles.
3. `tests/elements/test_cohesive8_compression.py` — δ_n < 0 produces
   penalty traction, no damage accumulates.
4. `tests/solver/test_newton.py` — softening 1-DOF spring converges, line
   search prevents divergence.
5. `tests/integration/test_dcb.py` — Double-Cantilever Beam; load-disp
   within 5% of Mi/Crisfield analytical.
6. `tests/integration/test_enf.py` — End-Notched Flexure (mode-II).
7. `tests/integration/test_wrinkle_tension_czm.py` — concave wrinkle under
   tension; delamination initiates at the crest and propagates outward;
   ultimate load compared against the analytical tension prediction.

### Phase 7 — Validation & docs

- Reproduce one published wrinkle-delamination experiment (e.g. Mukhopadhyay
  et al., Wilhelmsson et al.) in `validation/`.
- Update `ARCHITECTURE.md` module table.
- Update `VALIDATION.md` with DCB/ENF and wrinkle-delamination
  comparisons.
- Update `README.md` Features list.

---

## 3. Risks and explicit non-goals

- **Ill-conditioning** if `czm_penalty` is too high relative to bulk
  stiffness — defaults tuned per material, trade-off documented.
- **Mesh-size dependence**: the cohesive zone must be resolved by ≥3–5
  elements. Emit a runtime warning when `λ_cz / element_size < 3` using
  Hillerborg's estimate `λ_cz ≈ E·G_c / σ_max²`.
- **Snap-back**: v1 ships Newton + line search only; arc-length is a v2
  item, with a clear error message when Newton fails near peak load.
- **Out of scope for v1**: extrinsic CZM (XFEM-style), fiber-bridging
  traction tail, fatigue degradation, exponential or PPR laws, rate
  dependence.

---

## 4. Agentic execution architecture

Two structural features drive the architecture: a **serial numerical
backbone** (Newton + cohesive element + assembler — bugs here cascade)
and a **fan-out tail** (tests, viz, docs, validation — independent files,
low coupling).

```
                  ┌─ Plan agent (architectural Qs within a phase)
                  │
Main thread ──────┼─ Implementer (general-purpose, in worktree)  ┐
(orchestrator)    │                                                │ commit
                  ├─ Explore (read-only audits)                    │   ↓
                  │                                                │ merge
                  ├─ /code-review at each gate                     │   ↓
                  │                                                │ gate
                  └─ /verify after benchmark phases               ┘
```

### Roles

| Role | Agent type | When | Scope |
|---|---|---|---|
| Orchestrator | this thread | always | Owns branch state, gates, user clarifications. Does NOT implement the numerics itself. |
| Numerics implementer | `general-purpose` in a `worktree` | Phases 0+1, then Phase 2 | One agent per phase; tight scope; must land single-element T-δ test before declaring done. |
| Architecture consultant | `Plan` | one-off inside a phase | E.g. "consistent tangent vs perturbation tangent for cohesive8 — pick and justify." No code written. |
| Auditor | `Explore` | after each implementer commit | Read-only pass: "does `cohesive8.py` enforce d monotonicity? where? show line." |
| Reviewer | `/code-review` skill | at phase gates | Higher signal than auditor; posts findings; orchestrator triages. |
| Verifier | `/verify` skill or focused subagent | after Phase 2 and Phase 7 | Runs DCB + ENF benchmarks against analytical — the truth oracle. |
| Fan-out implementers | `general-purpose`, multiple in parallel, each in a `worktree` | Phases 5/6/7 only | Viz, remaining tests, docs — independent files. Merge sequentially. |

### Sequencing and gates

1. **Phase 0 + 1** (serial, one implementer) — gate: single-element
   pure-mode-I test passes, area-under-curve equals GIc within 1%,
   irreversibility holds. `/code-review` here.
2. **Phase 2** (serial, one implementer) — gate: DCB benchmark within 5%
   of Mi/Crisfield. `/verify` here — most important gate.
3. **Phase 3 + 4** (serial, one implementer) — gate: end-to-end
   `WrinkleAnalysis(enable_czm=True)` runs on a concave tension wrinkle
   without crashing.
4. **Phases 5, 6, 7** (parallel, three implementers in worktrees) — merge
   sequentially, run full test suite between merges.

### Guardrails

- **Worktree isolation** for every implementer agent. Prevents cross-agent
  edit collisions on shared files like `assembler.py` and `analysis.py`.
- **No implementer judges its own correctness.** DCB/ENF benchmarks are the
  oracle. Orchestrator runs `/verify` independently.
- **Truth-table tests live in the prompt**, not the agent's judgment. The
  implementer is told the exact test names that must exist and pass.
- **PR subscription** (`subscribe_pr_activity`) once the PR is open;
  orchestrator handles CI failures and review comments incrementally.
- **`/ultrareview`** worth running once at the end (user-triggered) —
  multi-agent cloud review is well-matched to numerics code where subtle
  sign errors hide.

### What we explicitly will NOT do

- Parallel agents on Phase 0/1/2 — those files are tightly coupled; parallel
  edits become a merge nightmare.
- A single mega-agent for the whole plan — context exhaustion, drift, and
  the failure mode where 80% works and 20% is silently wrong.
- Skip the DCB/ENF gate — without it, we have no signal that the math is
  right.

---

## 5. Suggested sequencing

1. Phase 0 + Phase 1 single-element tests — get the cohesive element +
   Newton solver working on a 1-element problem. Highest risk, do first.
2. Phase 2 mesh/assembler plumbing — DCB benchmark passes.
3. Phase 3 config/orchestration + Phase 4 reporting — `enable_czm=True`
   end-to-end on a concave tension wrinkle.
4. Phase 5 viz + Phase 6 remaining tests + Phase 7 docs/validation — in
   parallel.

Each phase is independently shippable behind the `enable_czm` flag; the
default linear path is untouched throughout.

---

## 6. Known limitations of the v1 CZM implementation

Documented honestly here so users understand what the feature does and does
not do. None block the package's primary tension-wrinkle delamination use
case; each is a bounded, intentional design choice that could be lifted
later if a real use case demands it.

### 6.1 ~~Mode-II damage is suppressed under normal compression~~ — REMOVED

**Previously documented limitation, now fixed.** The original v1 cohesive
law followed the Abaqus default and suppressed all damage evolution when
`δ_n < 0`, which caused the cohesive zone to collapse to one element wide
in closed-mode-II loading (e.g. 3-pt-bend ENF). After the Phase 7 NASA TM
ENF experimental validation exposed the issue empirically, the early-return
compression branch in `Cohesive8Element._law_local` was removed and the
standard Macaulay-bracketed damage evolution now applies.

Under the new (correct) behaviour:
- **Pure compression** (no shear): `δ_eff = 0`, no damage growth — the
  conservative "no damage from pure compression" property is preserved.
- **Compression + shear**: `δ_eff = β · ‖δ_shear‖` and `mode_ratio = 1.0`,
  so damage accumulates on the pure mode-II envelope.
- **Normal traction in compression** remains the penalty `K · δ_n` (no
  `(1-d)` reduction — closed surfaces transmit normal load fully); shear
  always uses the damaged secant `(1-d) · K · δ_{s,t}`.

No friction is modelled between damaged surfaces in closed contact — the
assumption is that frictional sliding work is small compared to the
fracture energy `GIIc`. This is the standard simplification in v1
cohesive zone implementations (Abaqus, LS-DYNA tie-break).

Tests that previously xfailed on this limitation:
`tests/integration/test_enf_monotonic.py` and the new
`tests/integration/test_enf_experimental_validation.py` (Phase 7).

### 6.2 Arc-length continuation does not yet support snap-back on DCB-style problems

`ArcLengthSolver` (`src/wrinklefe/solver/arclength.py`) uses cylindrical
arc-length (`‖Δu‖² = Δs²`). On a 1-DOF snap-through problem it correctly
traverses the limit point (`tests/test_solver/test_arclength_snapback.py`).
On a real DCB system the cylindrical norm is dominated by penalty BC
contributions and the constraint degenerates to ordinary displacement
control — the solver stagnates once the first cohesive element fully
damages.

Fix path: indirect displacement control, parametrising the arc by a
single interior DOF (typically the crack-mouth opening displacement, CMOD)
instead of `‖Δu‖`. Roughly 300 LOC of new solver work. Not on the roadmap
— wrinkle knockdown is captured by peak load under displacement control,
and snap-back is a post-peak numerical phenomenon (real DCB labs use
displacement-controlled rigs and don't observe snap-back anyway).

### 6.3 No fiber-bridging or R-curve effects

The v1 bilinear traction-separation law has constant `Gc`. Real CFRP
composites show R-curve behavior — apparent toughness rises with crack
extension due to fiber bridging behind the crack tip. Modeled as a
trailing traction tail it would add ~50-100 N/mm of dissipated energy on
moderate crack growth. The v1 implementation ignores this; predictions
are correspondingly conservative for the steady-state plateau.

Fix path: add a fiber-bridging traction term to `_law_local` and a new
`bridging_law` field on `CohesiveProperties`. Out of scope for v1.

### 6.4 No fatigue, no rate dependence, no thermal effects

All neglected by the static intrinsic CZM formulation. Each is a major
extension; none are planned.

### 6.5 Streamlit / CLI integration is not wired

`enable_czm=True` works from the Python API. The Streamlit app and CLI
do not yet surface the toggle. Users currently need to script against
the package to invoke CZM. A reasonable follow-up.
