"""Fitting a measured trace: micrograph CSV -> profile -> knockdown.

Every other example here starts from an amplitude and a wavelength that
someone already knows. On a real non-conformance they are not known: the
wrinkle arrives as a ply-boundary trace digitized off a polished-section
micrograph, in pixels, on an image that was not quite level. Reading the
amplitude off that by eye puts an unquantified manual step in front of an
otherwise traceable pipeline, and leaves the choice of morphology to
judgement (issue #270).

``wrinklefe.core.fit`` closes that gap. This script walks the whole path:

1. a synthetic "digitized" CSV in pixel coordinates, with noise and a
   stage tilt, standing in for the file a metallurgical lab would send;
2. ``load_trace`` to scale pixels to millimetres;
3. ``rank_families`` to answer "which morphology?" with a number;
4. ``fit_profile`` for the chosen family, with one-sigma parameter
   estimates and a residual that says how representative the
   idealization actually is;
5. ``to_config_kwargs`` straight into ``AnalysisConfig`` for the
   knockdown.

Two points the script measures rather than asserts. The tilt is fitted
jointly as a nuisance parameter, not subtracted beforehand, and turning
that off biases the amplitude badly -- step 6 shows by how much. And the
families are ranked by BIC rather than by raw residual, because three of
the five carry four shape parameters and two carry three: a
four-parameter family can always match a three-parameter one, so a raw
residual ranking is decided by noise. Step 3 prints both columns.

Expected runtime: ~2 s (analytical path).
Expected output:  the recovered geometry within a few percent of the
                  truth used to build the CSV, ``gaussian_sinusoidal``
                  ranked first, and a knockdown well below 1.
"""

import tempfile
from pathlib import Path

import numpy as np

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.fit import fit_profile, load_trace, rank_families
from wrinklefe.core.wrinkle import GaussianSinusoidal

# --------------------------------------------------------------------
# 1. Stand in for the lab's file.
#
# The "truth" here is only used to build the CSV and to score the fit at
# the end; nothing downstream of load_trace ever sees it.
# --------------------------------------------------------------------
TRUE_AMPLITUDE = 0.42      # mm
TRUE_WAVELENGTH = 14.0     # mm
TRUE_WIDTH = 9.0           # mm
PX_PER_MM = 100.0          # micrograph calibration
STAGE_TILT = 0.03          # mm/mm, the section was not mounted level

truth = GaussianSinusoidal(
    amplitude=TRUE_AMPLITUDE, wavelength=TRUE_WAVELENGTH,
    width=TRUE_WIDTH, center=0.0,
)
rng = np.random.default_rng(20250924)
x_mm = np.linspace(-22.0, 22.0, 140)
z_mm = np.asarray(truth.displacement(x_mm), dtype=float)
z_mm = z_mm + STAGE_TILT * x_mm + 0.8           # tilt and an arbitrary datum
z_mm = z_mm + rng.normal(0.0, 0.01, x_mm.size)  # digitization scatter

tmpdir = Path(tempfile.mkdtemp())
csv_path = tmpdir / "ply_boundary_trace.csv"
csv_path.write_text(
    "x_px,z_px\n"
    + "".join(
        f"{xi * PX_PER_MM:.1f},{zi * PX_PER_MM:.1f}\n"
        for xi, zi in zip(x_mm, z_mm)
    )
)
print(f"digitized trace: {csv_path}  ({x_mm.size} points, pixels)")

# --------------------------------------------------------------------
# 2. Pixels -> millimetres.
# --------------------------------------------------------------------
x, z = load_trace(csv_path, scale=1.0 / PX_PER_MM, skip_header=1)
print(f"loaded {x.size} points spanning {x[0]:.1f} to {x[-1]:.1f} mm")

# --------------------------------------------------------------------
# 3. Which morphology? Ranked, not chosen by eye.
# --------------------------------------------------------------------
ranked = rank_families(x, z)
by_rms = {r.family: i for i, r in enumerate(rank_families(x, z, criterion="rms"))}

print("\nfamily ranking (BIC; raw-residual rank shown for contrast)")
print(f"  {'family':<24} {'k':>2} {'RMS (mm)':>10} {'BIC':>10} {'rank by RMS':>12}")
for result in ranked:
    print(
        f"  {result.family:<24} {len(result.fitted_parameters):2d} "
        f"{result.rms_residual:10.5f} {result.bic:10.1f} "
        f"{by_rms[result.family] + 1:12d}"
    )

best = ranked[0]
print(f"\nbest family: {best.family}")

# --------------------------------------------------------------------
# 4. The fit, as a report.
# --------------------------------------------------------------------
fit = fit_profile(x, z, best.family)
print("\n" + fit.summary())
print(
    f"  recovered vs truth: "
    f"A {fit.amplitude:.4f} / {TRUE_AMPLITUDE} mm "
    f"({abs(fit.amplitude - TRUE_AMPLITUDE) / TRUE_AMPLITUDE:.1%}), "
    f"lambda {fit.wavelength:.3f} / {TRUE_WAVELENGTH} mm "
    f"({abs(fit.wavelength - TRUE_WAVELENGTH) / TRUE_WAVELENGTH:.1%}), "
    f"w {fit.width:.3f} / {TRUE_WIDTH} mm "
    f"({abs(fit.width - TRUE_WIDTH) / TRUE_WIDTH:.1%})"
)
if fit.trend_coeffs is not None:
    slope, offset = fit.trend_coeffs
    print(
        f"  stage tilt recovered as a nuisance parameter: "
        f"{slope:+.4f} mm/mm (applied {STAGE_TILT:+.4f}), "
        f"datum {offset:+.3f} mm"
    )
print(
    f"  residual {fit.rms_residual:.5f} mm is "
    f"{fit.rms_residual / fit.amplitude:.1%} of the fitted amplitude -- "
    "this is the number that says whether an idealized family is a fair\n"
    "  description of the defect, or whether it needs meshing directly."
)

# --------------------------------------------------------------------
# 5. Measured trace -> knockdown, in one hop.
# --------------------------------------------------------------------
cfg = AnalysisConfig(
    **fit.to_config_kwargs(),
    angles=[0.0, 45.0, -45.0, 90.0] * 2,
    loading="compression",
)
result = WrinkleAnalysis(cfg).run(analytical_only=True)
print(
    f"\nknockdown from the measured trace: "
    f"{result.analytical_knockdown:.4f} "
    f"({result.analytical_strength_MPa:.1f} MPa, "
    f"theta_max {np.degrees(result.max_angle_rad):.2f} deg)"
)

# --------------------------------------------------------------------
# 6. What the tilt would have cost.
# --------------------------------------------------------------------
raw = fit_profile(x, z, best.family, detrend=False)
raw_cfg = AnalysisConfig(
    **raw.to_config_kwargs(),
    angles=[0.0, 45.0, -45.0, 90.0] * 2,
    loading="compression",
)
raw_result = WrinkleAnalysis(raw_cfg).run(analytical_only=True)
direction = (
    "NON-CONSERVATIVE -- it reports the part as healthier than it is"
    if raw_result.analytical_knockdown > result.analytical_knockdown
    else "conservative"
)
print(
    f"\nwithout detrending, the same trace fits the tilt itself as one huge "
    f"wrinkle:\n"
    f"  A = {raw.amplitude:.3f} mm, lambda = {raw.wavelength:.0f} mm, "
    f"w = {raw.width:.0f} mm, centre {raw.center:.0f} mm "
    f"(outside the {x[0]:.0f}..{x[-1]:.0f} mm trace)\n"
    f"  residual {raw.rms_residual:.4f} mm, "
    f"{raw.rms_residual / fit.rms_residual:.0f}x the detrended fit"
)
print(
    "  the fit announces its own unreliability: one-sigma on the amplitude is "
    f"{raw.sigmas['amplitude']:.2g} mm against a fitted "
    f"{raw.amplitude:.2g} mm --\n"
    "  a parameter known only to within itself. Large sigmas next to a poor\n"
    "  residual are the signature of a trend that was never accounted for."
)
print(
    f"  knockdown {raw_result.analytical_knockdown:.4f} vs "
    f"{result.analytical_knockdown:.4f}: a shift of "
    f"{abs(raw_result.analytical_knockdown - result.analytical_knockdown):.4f}, "
    f"and it is\n  {direction},\n"
    "  from nothing but a section that was not mounted level."
)
