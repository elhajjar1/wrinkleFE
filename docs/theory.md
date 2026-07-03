# Theory: physics & mechanics

This page is the mechanics reference behind WrinkleFE: how a wrinkle's
geometry is turned into a strength and stiffness knockdown, what each
model assumes, and where each one applies. It mirrors the implementation
in `src/wrinklefe/` (with the controlling functions named inline) so the
documented physics and the code cannot drift.

WrinkleFE separates the problem into a **fast analytical screen** and an
optional **3-D finite-element solve**, both driven by one
{class}`~wrinklefe.analysis.AnalysisConfig`. The guiding split:

- **Multidirectional laminates** are governed by *fibre-scale kinking
  (compression)* or *competing fibre / matrix / delamination mechanisms
  (tension)*, which depend mainly on the peak fibre-misalignment angle.
  Use the angle-based analytical models.
- **Unidirectional laminates** show a strong dependence on how deep the
  wrinkle penetrates the thickness *at a fixed angle* — an effect the
  angle-only models cannot capture (they are scale-invariant). Use the
  penetration gate (and the progressive-damage FE route).

## 1. Wrinkle kinematics

A wrinkle is an out-of-plane ripple of the plies. The canonical profile
(`core/wrinkle.py`, {class}`~wrinklefe.core.wrinkle.GaussianSinusoidal`)
is a sinusoidal carrier under a Gaussian envelope:

$$
z(x) = A \, \exp\!\left(-\frac{(x-x_0)^2}{w^2}\right)\, \cos\!\left(\frac{2\pi (x-x_0)}{\lambda}\right)
$$

where $A$ is the **half-amplitude** (peak displacement of the wrinkled
mid-surface from the flat reference; peak-to-trough height is $2A$),
$\lambda$ the wavelength (crest-to-crest period), $w$ the longitudinal
envelope decay length, and $x_0$ the crest position. Other profiles
(rectangular/tapered, triangular, pure-sinusoidal, Gaussian bump) share
the same $(A, \lambda, w)$ contract; see the
{class}`~wrinklefe.core.wrinkle.WrinkleProfile` docstrings.

The local fibre-misalignment angle is the profile slope,
$\theta(x) = \arctan\!\big(\mathrm{d}z/\mathrm{d}x\big)$. The analytical
models reduce the wrinkle to a single **peak misalignment angle** using
the closed form for a pure cosine:

$$
\theta_{\max} = \arctan\!\left(\frac{2\pi A}{\lambda}\right)
$$

This deliberately uses the *un-attenuated* amplitude $A$ (not the
envelope-reduced numerical maximum) because the calibration datasets
report knockdown against the full wrinkle amplitude.

For unidirectional laminates a second geometric measure matters — the
**through-thickness penetration**

$$
D/T = \frac{A}{T}, \qquad T = n_\text{plies}\, t_\text{ply}
$$

i.e. how far the wrinkle reaches through the laminate thickness $T$.

## 2. Compression — Budiansky–Fleck kink-band

Under axial compression a wavy 0° ply fails by **fibre kink-banding**:
the misaligned fibres rotate, the matrix shears, and a band of buckled
fibres forms. The Budiansky–Fleck model gives the kink-band knockdown as
a function of the misalignment angle and the matrix shear-yield strain.
WrinkleFE wraps it in a *Classical-Lamination-Theory (CLT) stiffness
weighting* so it applies to real layups (`analysis.py`,
`WrinkleAnalysis._compute_analytical`):

$$
\mathrm{KD}_\text{lam} = f_0\, \mathrm{KD}_\text{BF} + (1 - f_0),
\qquad
\mathrm{KD}_\text{BF} = \frac{1}{1 + r + c_\text{AF}\, r^2},
\quad r = \frac{\theta_\text{eff}}{\gamma_{Y,\text{eff}}}
$$

with $\theta_\text{eff} = M_f\, \theta_{\max}$ (morphology factor folded
in, §2.3). The denominator is the Argon–Fleck quadratic extension; its
coefficient $c_\text{AF}$ (`kink_band_quadratic_coeff`) defaults to **0**,
recovering the classical *linear* Budiansky–Fleck floor
$\mathrm{KD}_\text{BF} = 1/(1 + \theta_\text{eff}/\gamma_{Y,\text{eff}})$.

### 2.1 CLT stiffness weighting $f_0$

Only the 0° plies kink; off-axis plies carry through at full strength and
redistribute load. $f_0$ is the **axial-stiffness fraction** carried by
the 0° plies, not a simple ply count:

$$
f_0 = \frac{n_0\,Q^{0}_{11}}{n_0\,Q^{0}_{11} + n_{45}\,Q^{45}_{11} + n_{90}\,Q^{90}_{11}}
$$

with $Q^{0}_{11}=E_{11}$, $Q^{45}_{11}=E_{11}/4 + E_{22}/4 + G_{12}/2$,
$Q^{90}_{11}=E_{22}$. The $(1-f_0)$ term in $\mathrm{KD}_\text{lam}$ is the
off-axis plies riding through at $\mathrm{KD}=1$.

### 2.2 Layup-dependent yield strain (confinement)

The matrix shear-yield strain is not a constant — it depends on how much
lateral support the 0° plies receive (`_effective_gamma_Y`,
`_confined_fraction`):

$$
\gamma_{Y,\text{eff}} = \max\!\Big(
\underbrace{0.032}_{\gamma_{Y,\text{UD}}}
+ \underbrace{0.050}_{\alpha_\text{conf}} f_\text{conf}
- \underbrace{0.010}_{\beta_\text{block}} \max(n_\text{block}-1,\,0),
\;\; \underbrace{0.016}_{\text{floor}} \Big)
$$

- $f_\text{conf}\in[0,1]$ is the **confinement fraction**: each 0° ply
  scores $1.0$ if *both* neighbours are off-axis (or a free surface),
  $0.5$ if one is, $0.0$ if it is buried inside a 0° block. Dispersion
  raises $\gamma_{Y,\text{eff}}$ and so reduces the knockdown.
- $n_\text{block}$ is the **longest run of consecutive 0° plies**. The
  block penalty captures that the inner 0° faces of a thick block are
  bracketed by another 0° ply that does *not* restrain kink-band lateral
  expansion, so blocked 0-plies kink at a lower applied shear strain than
  the neighbour score alone implies.
- The result is floored at $0.016$ (half the UD value) so a long block
  cannot drive a degenerate knockdown, and the block penalty is
  suppressed for pure UD `[0]_n` so UD stays at the $0.032$ calibration
  point.

Net effect: a dispersed `[0/45/90/-45]s` is far more wrinkle-tolerant
than a blocked `[0_4/90_4]s`.

### 2.3 Morphology factor $M_f$

Real wrinkles come in pairs at adjacent interfaces. The morphology factor
(`core/morphology.py`) scales the effective angle by the phase offset
$\varphi$ between two wrinkle centrelines:

$$
M_f(\varphi) = \exp\!\big(-\alpha_\text{asym}\sin\varphi - \alpha_\text{offset}(1 - |\cos\varphi|)\big)
$$

| Morphology | Phase $\varphi$ | $M_f$ (compression) | Meaning |
|---|---|---|---|
| `stack` | $0$ | $1.0$ (baseline) | aligned crests |
| `convex` | $+\pi/2$ | $<1$ | interface bulges outward (least damaging) |
| `concave` | $-\pi/2$ | $>1$ | interface pinches inward (most damaging) |

with $\alpha_\text{asym}=0.288$, $\alpha_\text{offset}=0$ in compression
(in tension $0.033$ and $0.183$). For $N$ wrinkles the factors combine as
a geometric mean; single-wrinkle modes (`uniform`, `graded`) have
$M_f=1$.

**Amplitude contract in the FE mesh.** The morphology factor above scales
the *analytical* angle. The FE mesh builds the same paired-wrinkle geometry
by *summing* the two constituents' through-thickness–decayed displacement
fields (`apply_to_nodes`). To keep the mesh consistent with the definition
of $A$ as the peak deflection of the wrinkle (§1, peak-to-trough $=2A$),
each of the two constituents of a `stack`/`convex`/`concave` morphology is
generated at **half amplitude** $A/2$, so the in-phase (`stack`) sum peaks
at exactly $A$ rather than $2A$. This makes the meshed fibre angle equal
the analytical $\theta_{\max}=\arctan(2\pi A/\lambda)$ for the `stack`
morphology, and the phase-offset morphologies (`convex`/`concave`) partly
cancel below $A$. (Explicit multi-wrinkle configurations built through the
`WrinkleSpec` API bypass this convention: each listed wrinkle carries the
amplitude the caller specifies.) See issue #305.

### 2.4 Graded morphology averaging

For the `graded` morphology the Budiansky–Fleck knockdown is averaged
over the wrinkle profile rather than evaluated at a single angle
(`_profile_proportional_kd`). The longitudinal average sweeps 500 points
along $x$ using the *analytical* slope of $z(x)$; the through-thickness
average weights each ply $p$ by a Gaussian envelope:

$$
\Phi_p = \text{decay\_floor} + (1 - \text{decay\_floor})
\exp\!\left(-\frac{(z_p - z_c)^2}{2\sigma^2}\right),
\quad \theta_{xz} = \theta_x\,\Phi_p
$$

centred at $z_c = \texttt{wrinkle\_z\_position}\cdot T$ (mid-plane by
default) with decay scale $\sigma = \texttt{through\_thickness\_decay\_scale}$
(auto default $\max(\lambda/2, A)$). `decay_floor` $=0$ is a fully
embedded wrinkle that fades to flat at the surfaces; `decay_floor` $=1$
collapses to `uniform`. The **tension** graded path uses an analogous
*linear* through-thickness taper instead of the Gaussian.

## 3. Tension — three competing mechanisms

In tension a wrinkled 0° ply can fail three ways, and the *most severe*
governs (`_tension_knockdown_analytical`):

$$
\mathrm{KD}_0 = \min\!\big(\mathrm{KD}_\text{fiber},\; \mathrm{KD}_\text{matrix},\; \mathrm{KD}_\text{oop}\big),
\qquad
\mathrm{KD}_\text{lam} = f_0\, \mathrm{KD}_0 + (1 - f_0)
$$

with the same CLT weighting $f_0$ as compression.

**1. Fibre load-rotation** — the misaligned fibre carries only the
axial-projected load:

$$
\mathrm{KD}_\text{fiber} = \cos^2\theta
$$

**2. In-situ matrix cracking** — a Hashin/LaRC $\sigma_{22}$–$\tau_{12}$
interaction at the misalignment angle, using *in-situ* (thin/thick-ply
constrained) strengths:

$$
\sigma_\text{fail} = \left[\left(\frac{\sin^2\theta}{Y_t^\text{is}}\right)^2
+ \left(\frac{\sin\theta\cos\theta}{S_{12}^\text{is}}\right)^2\right]^{-1/2},
\qquad
\mathrm{KD}_\text{matrix} = \min\!\left(\frac{\sigma_\text{fail}}{X_t},\,1\right)
$$

The transverse in-situ strength is $Y_t^\text{is} = 1.12\sqrt{2}\,Y_t$;
the shear in-situ strength uses the Camanho thick-ply correction
$S_{12}^\text{is} = \sqrt{8\,G_{IIc}/(\pi\, t_\text{eff}\, \Lambda_{22})}$
over the effective 0° block thickness $t_\text{eff}$, with
$\Lambda_{22}=2(1/E_{22}-\nu_{12}^2/E_{11})$.

**3. Curved-beam out-of-plane delamination** — the wrinkle curvature
generates interlaminar stresses: a mode-I $\sigma_{33}$ at the crest
(peak curvature) and a mode-II $\tau_{13}$ at the inflection (peak
curvature gradient):

$$
\sigma_{33} = X_t\, h_\text{eff}\, \kappa_{\max}, \quad
\tau_{13} = X_t\, h_\text{eff}\, \kappa'_{\max}, \quad
\kappa_{\max} = \left(\tfrac{2\pi}{\lambda}\right)^2 A
$$

with $\mathrm{KD}_\text{oop} = 1/\sqrt{1 + \max\big((\sigma_{33}/Y_t)^2,\,(\tau_{13}/S_{13})^2\big)}$.
A separate Benzeggagh–Kenane mixed-mode delamination-*onset* knockdown is
reported alongside the ultimate value.

Finally the tension knockdown is **floored by the compression value** for
the same defect — tension is never predicted to be worse than
compression.

## 4. The unidirectional scale effect

The angle-based models above are **scale-invariant**: at a *fixed* peak
misalignment angle they return the same knockdown regardless of how deep
the wrinkle runs through the thickness. The Li UD glass/epoxy grids show
this is wrong for unidirectional material — at $\theta\approx20°$ the
measured knockdown spans $0.63 \to 1.00$ as $D/T$ falls. WrinkleFE adds
three UD-specific capabilities.

### 4.1 Penetration gate $(\theta, D/T, z)$

A closed-form UD predictor (`core/penetration_gate.py`) that is both
*scale-aware* (depends on $D/T$) and *position-aware* (depends on the
wrinkle's through-thickness location):

$$
\mathrm{KD} = 1 - \big(1 - \mathrm{KD}_\text{angle}(\theta)\big)\, S(D/T)\, P(z)
$$

$$
\mathrm{KD}_\text{angle}(\theta) = \frac{1}{1 + \theta_\text{rad}/\gamma_Y},
\quad
S(D/T) = \min\!\Big(1, (D/T \,/\, dt_0)^{p}\Big),
\quad
P(z) = \big(2\min(z, 1-z)\big)^{q}
$$

$\mathrm{KD}_\text{angle}$ is the Budiansky–Fleck angle floor, $S$ is the
penetration gate (a steep power law that drops the knockdown as the
wrinkle penetrates deeper), and $P$ is the through-thickness position
factor (1 at mid-plane, $\to 0$ at a free surface, so a near-surface
wrinkle is far milder). Two calibrated presets ship, **one per material
realization** (they cannot share a normalization — see the validation
ledger):

| Preset | $\gamma_Y$ | $dt_0$ | $p$ | $q$ (position) | Dataset |
|---|---|---|---|---|---|
| `GATE_LI2024_MOULDED` | 0.2577 | 0.0938 | 0.59 | — (none) | E (moulded) |
| `GATE_LI2025_VACBAG` | 0.6215 | 0.1220 | 4.31 | 5.26 | F (vacuum-bag) |

`predict_from_geometry` maps $(A,\lambda, n_\text{plies}, t_\text{ply})$
to $(\theta_{\max}, D/T)$; `AnalysisConfig.penetration_gate` routes
`analytical_knockdown` through the gate. It is **UD-scoped** (do not apply
to multidirectional laminates) and costs no FE solve. The fitted
$\gamma_Y$ absorbs the fibre *bending* stiffness / couple-stress length
scale that a homogenised continuum cannot resolve — compressive kinking
strength is, in the state of the art, a calibrated quantity.

### 4.2 Resin-pocket material zone

A machined cosine insert leaves a fibre-free **neat-epoxy lens** at the
wrinkle crest (`core/resin_pocket.py`). `ResinPocketSpec` describes a
raised-cosine lens, `compute_resin_mask` flags the hex elements inside it,
and `compute_resin_blend` returns a graded weight $w\in[0,1]$ (1 at the
lens centre). The FE path then blends the modulus,
$(1-w)\,\text{host} + w\,\text{resin}$ (the isotropic `EPOXY_S6C10` card),
**and** scales the fibre-misalignment angle by $(1-w)$ so the soft
inclusion and the kink-band are counted *once*, not twice.

### 4.3 Progressive-damage FE (the first FE route to a UD knockdown)

A linear first-ply-failure index never activates for pristine UD
compression, so there is no FE knockdown from a single linear solve. The
`ProgressiveDamageSolver` (`solver/progressive_damage.py`) instead ramps
the applied strain in increments, re-solving and degrading newly-failed
elements (ply-discount by failure-mode family, residual factor 0.1) using
a combined **MaxStress + LaRC05** criterion. The ultimate strength is the
maximum of the first-failure load (interpolated to global $\mathrm{FI}=1$,
which pins the pristine baseline to $X_c$) and the peak redistributed
stress over the load history.

An optional **crack-band (Bažant–Oh) regularization** (`crack_band=True`,
off by default) makes the dominant fibre-compression mode mesh-objective:
the linear-softening end point is scaled by the element size $h$,

$$
r_f = \frac{2\,G_{c,\text{fiber}}\, E_1}{X_c^2\, h},
\qquad
d = 1 - \frac{r_f - r}{r\,(r_f - 1)}
$$

so the dissipated energy per crack area equals the fibre-kink fracture
energy $G_{c,\text{fiber}}$ regardless of mesh, degrading $E_1$,
$\nu_{12}$, $\nu_{13}$.

### 4.4 Linear buckling — a documented negative finding

Compressive failure of a wavy UD laminate is at root a geometric
instability, so a linearized-buckling route was tried
(`solver/buckling.py`): assemble the geometric stiffness $K_\text{geo}$
from a pre-stressed solve and solve the eigenproblem
$K\,\phi = \lambda\, M\,\phi$ with $M = -K_\text{geo}$, taking the
smallest positive $\lambda$, and
`microbuckling_knockdown` $= \lambda_\text{wrinkled}/\lambda_\text{pristine}$.
For a non-uniform (wrinkled) pre-stress $M$ is **indefinite**, which is
solved correctly as the symmetric-definite pencil $M\,\phi = \mu\,K\,\phi$
($K$ is SPD), $\lambda = 1/\mu$.
**This does not work for the wrinkle knockdown — it gets the sign
wrong**: with the eigenproblem solved correctly the bifurcation load
*rises* with the wrinkle (tilting the fibres out of the load path reduces
the destabilising axial pre-stress), so `microbuckling_knockdown`
$\to 1$ (no knockdown) instead of the measured drop, because (1) the
wrinkled structure is imperfection-sensitive (Koiter — bifurcation sits
far below the limit load) and (2) buckling of the homogenised ply-mesh is
*structural* buckling of the soft region, not the sub-ply *fibre kinking*
that governs. $K_\text{geo}$ is retained as correct, tested
infrastructure for genuine structural-buckling analyses, but it is
**not** used as the UD wrinkle predictor. See
[Wrinkle modelling findings](wrinkle_modeling_findings.md) (item D.4).

## 5. Finite-element mechanics

The FE path (`core/mesh.py`, `elements/`, `solver/`) deforms a structured
hexahedral mesh to the wrinkle profile and assigns each element its local
fibre-misalignment angle, then solves the linear (or, for damage,
load-stepped) system and evaluates ply failure.

- **Elements.** 8-node hexahedra (`elements/hex8.py`) with an
  incompatible-modes variant (`elements/hex8i.py`) for improved bending
  response, $2\times2\times2$ Gauss integration (`elements/gauss.py`).
  `hex8.py` also supplies the per-element geometric stiffness
  $K_\text{geo}$ used by the buckling route.
- **Ply failure — LaRC05.** The default FE criterion
  (`failure/larc05.py`) is the Pinho/Camanho 3-D criterion: fibre tension,
  fibre **kinking** under compression (stress rotated into the
  misalignment frame, with a Mohr–Coulomb compressive matrix interaction),
  in-situ matrix strengths, and a fracture-plane search for matrix
  failure. Because LaRC05 kinking does *not* trigger for a pristine UD
  ply at zero initial misalignment, the progressive-damage UD path pairs
  it with a plain MaxStress $|\sigma_{11}|\ge X_c$ check (§4.3). Other
  criteria available through `FailureEvaluator`: Hashin, Puck, Tsai-Wu,
  Tsai-Hill, maximum stress, and maximum strain.
- **Effective laminate stiffness** comes from Classical Lamination Theory
  (`core/laminate.py`): the ABD matrices $A=\sum \bar{Q}_k t_k$,
  $B=\tfrac12\sum \bar{Q}_k(z_k^2-z_{k-1}^2)$,
  $D=\tfrac13\sum \bar{Q}_k(z_k^3-z_{k-1}^3)$, an FSDT transverse-shear
  $H$ matrix with shear-correction factor $5/6$, thermal resultants, and
  effective membrane moduli $E_x, E_y, G_{xy}, \nu_{xy}$ from $A^{-1}$.
- **Stiffness (modulus) knockdown.** The analysis reports two estimates
  of the axial Young's-modulus knockdown $E_x/E_{x,0}$. The FE
  `modulus_retention` is the ratio of the wrinkled to pristine mean
  fibre-direction stress at a fixed applied strain (any layup). For
  **unidirectional** layups the analytical path additionally returns a
  closed-form `analytical_modulus_knockdown`
  (`analysis._profile_modulus_knockdown`): a Classical-Lamination-Theory
  series-average of the off-axis lamina modulus
  $1/E_x(\theta)=\cos^4\theta/E_1+(1/G_{12}-2\nu_{12}/E_1)\sin^2\theta\cos^2\theta+\sin^4\theta/E_2$
  over the wrinkle profile (the same off-axis-compliance integration as
  Hsiao & Daniel 1996), evaluated with no FE solve. It is loading-
  independent and UD-scoped (it stays $1.0$ for multidirectional layups,
  where `modulus_retention` is the predictor). Both are validated against
  the UD modulus datasets (Li 2025, Hsiao & Daniel 1996) by
  `validation/validate_modulus.py` — see the
  [validation page](validation.md). Stiffness is consistently far more
  wrinkle-tolerant than strength.

### 5.1 Cohesive-zone delamination (CZM)

For interlaminar cracking at the wrinkle, zero-thickness 8-node cohesive
elements (`elements/cohesive8.py`) carry a **bilinear
traction–separation** law. The effective opening uses a Macaulay bracket
on the normal component so compression does not open damage:

$$
\delta_\text{eff} = \sqrt{\langle\delta_n\rangle_+^2 + \beta^2(\delta_s^2 + \delta_t^2)}
$$

A scalar damage $d$ accumulates **monotonically** in $[0,1]$; tractions
are the damaged secant $T_i = (1-d)\,K\,\delta_i$, except that a *closed*
interface ($\delta_n<0$) transmits the full penalty $K\,\delta_n$ in the
normal direction (no $(1-d)$ reduction — closed surfaces carry normal load
fully) while shear still uses the damaged secant. Mixed-mode fracture
follows **Benzeggagh–Kenane**:

$$
G_c(\psi) = G_{Ic} + (G_{IIc} - G_{Ic})\left(\frac{G_{II}}{G_I + G_{II}}\right)^{\eta},
\quad \eta = 1.45
$$

with the mode ratio frozen at initiation (path-independent envelope). No
friction is modelled between closed damaged surfaces. The full nonlinear
solve uses Newton–Raphson with backtracking line search
(`solver/nonlinear.py`); an arc-length continuation exists
(`solver/arclength.py`) but does not yet handle DCB-style snap-back.

The CZM is validated against the NASA/TM-2020-220498 IM7/8552 coupon
data — DCB (mode I), ENF and 4-point-bend (mode II), and MMB (mixed
mode) — under `tests/integration/`. The mode-II and mixed-mode peaks fall
within ~15 %; the **DCB peak load over-predicts by ~35 %** (the bilinear
law with no fibre-bridging R-curve), so that experimental test is marked
`xfail` — an honest, documented partial match rather than a clean pass.

## 6. Scope, calibration, and honest limitations

- **Use the angle-based analytical models for multidirectional laminates,
  the penetration gate for unidirectional laminates.** At a fixed angle
  the UD knockdown still varies with $D/T$, which only the gate captures.
- **The UD gate is material-realization specific.** The moulded (E) and
  vacuum-bag (F) realizations of the same prepreg give contradictory
  knockdown at the same $(\theta, D/T)$ — roughly a 2× consolidation
  difference — so they ship as two cards and two presets and cannot share
  an absolute normalization.
- **Compressive kinking strength is calibrated, not first-principles.**
  The physical matrix-shear-yield strain over-predicts the kink knockdown
  ~10×; the fitted $\gamma_Y$ absorbs fibre-bending physics the
  homogenised continuum cannot resolve (findings D.2/D.4).
- **The in-repo integration tests assert physical-sanity properties**
  (monotonicity, morphology ordering, mode bookkeeping), not absolute
  experimental agreement; the quantitative comparison lives in the
  [validation ledger](validation.md) and the consolidated parity chart.
- **Through-thickness position** ($P(z)$) is fit to a single near-surface
  data point and is indicative; multi-wrinkle interaction and a unified
  E/F calibration remain open.

For the modelling programme that produced these choices (items D.1–D.5),
see [Wrinkle modelling findings](wrinkle_modeling_findings.md); for the
predicted-vs-experimental scorecard see the
[validation page](validation.md); for the public surface that exposes
every model, see the [API reference](api/index.md).

## Symbols

| Symbol | Meaning |
|---|---|
| $A$ | wrinkle half-amplitude (mm); peak-to-trough $=2A$ |
| $\lambda$ | wrinkle wavelength (mm) |
| $w$ | longitudinal envelope decay length (mm) |
| $T$, $t_\text{ply}$ | laminate thickness, single-ply thickness (mm) |
| $\theta_{\max}$, $\theta_\text{eff}$ | peak / morphology-scaled misalignment angle |
| $D/T$ | through-thickness penetration $A/T$ |
| $f_0$ | axial-stiffness fraction of the 0° plies |
| $f_\text{conf}$, $n_\text{block}$ | confinement fraction, longest 0° block |
| $\gamma_{Y,\text{eff}}$ | effective matrix shear-yield strain |
| $M_f$ | morphology factor (phase-dependent) |
| $\mathrm{KD}$ | knockdown = wrinkled strength / pristine strength |
| $d$, $G_c$ | cohesive damage variable, fracture toughness |
