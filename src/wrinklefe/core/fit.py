"""Fit an idealized :class:`~wrinklefe.core.wrinkle.WrinkleProfile` to measured data.

The wrinkle a user actually needs to disposition is a *measured* one — a
ply-boundary trace digitized from a polished-section micrograph, or an
out-of-plane surface profile from C-scan or profilometry.  Without this
module the amplitude and wavelength that drive every downstream number
are read off an image by eye, which puts an unquantified manual step in
front of an otherwise traceable pipeline and leaves the choice of
morphology to judgement (issue #270).

What this module adds:

- :func:`fit_profile` — nonlinear least squares of one profile family to
  ``(x, z)`` points, with initial guesses derived from the data itself
  (peak-to-peak amplitude, FFT dominant period, envelope span).
- :func:`rank_families` — the same fit across every family, ranked, so
  "which morphology?" is answered by a number instead of judgement.
- :class:`FitResult` — the fitted profile plus one-sigma parameter
  estimates, RMS residual and :math:`R^2`, so a report can state how
  representative the idealization actually is.
- :func:`load_trace` — a 2-column CSV reader with a pixel-to-millimetre
  scale factor, since raw digitized traces rarely arrive in mm.

**Scope.** This fits the *idealized families* to data.  Meshing an
arbitrary measured profile directly (spline-based, non-parametric) is a
much larger feature and deliberately out of scope; the fit residual is
what tells a user when that bigger hammer is actually needed.

Example
-------
>>> import numpy as np
>>> from wrinklefe.core.wrinkle import GaussianSinusoidal
>>> truth = GaussianSinusoidal(amplitude=0.5, wavelength=12.0,
...                            width=8.0, center=0.0)
>>> x = np.linspace(-20.0, 20.0, 80)
>>> fit = fit_profile(x, truth.displacement(x), "gaussian_sinusoidal")
>>> bool(abs(fit.amplitude - 0.5) < 0.01)
True
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from wrinklefe.core.wrinkle import (
    GaussianBump,
    GaussianSinusoidal,
    PureSinusoidal,
    RectangularSinusoidal,
    TriangularSinusoidal,
    WrinkleProfile,
)

__all__ = [
    "FAMILIES",
    "FitResult",
    "fit_profile",
    "load_trace",
    "rank_families",
]

# Family name -> (class, the parameters that are actually identifiable).
#
# Two families are degenerate in one of the four shared constructor
# arguments and MUST hold it fixed: ``PureSinusoidal.displacement``
# never reads ``width`` and ``GaussianBump.displacement`` never reads
# ``wavelength``.  Letting the optimiser vary a parameter the model
# ignores leaves that Jacobian column identically zero — the covariance
# is then singular and the reported one-sigma is meaningless.
FAMILIES: dict[str, tuple[type[WrinkleProfile], tuple[str, ...]]] = {
    "gaussian_sinusoidal": (
        GaussianSinusoidal, ("amplitude", "wavelength", "width", "center"),
    ),
    "rectangular_sinusoidal": (
        RectangularSinusoidal, ("amplitude", "wavelength", "width", "center"),
    ),
    "triangular_sinusoidal": (
        TriangularSinusoidal, ("amplitude", "wavelength", "width", "center"),
    ),
    "pure_sinusoidal": (
        PureSinusoidal, ("amplitude", "wavelength", "center"),
    ),
    "gaussian_bump": (
        GaussianBump, ("amplitude", "width", "center"),
    ),
}


@dataclass(frozen=True)
class FitResult:
    """One profile family fitted to a measured trace.

    Attributes
    ----------
    profile : WrinkleProfile
        The fitted profile, ready to hand to ``AnalysisConfig`` or to
        :class:`~wrinklefe.core.morphology.WrinkleConfiguration`.
    family : str
        Which family this is, as a key of :data:`FAMILIES`.
    amplitude : float
        Fitted amplitude [mm].  For a degenerate family the parameter
        the model ignores carries a placeholder, not a fitted value —
        ``fitted_parameters`` says which are real.
    wavelength : float
        Fitted wavelength [mm], or a placeholder for ``gaussian_bump``.
    width : float
        Fitted envelope half-width [mm], or a placeholder for
        ``pure_sinusoidal``.
    center : float
        Fitted centre position [mm].
    fitted_parameters : tuple[str, ...]
        The shape parameters that were actually varied.
    sigmas : dict[str, float]
        One-sigma estimates from the covariance, for the fitted shape
        parameters.  A value is ``inf`` when the fit is too
        ill-conditioned to support an estimate — information, not a
        failure.
    rms_residual : float
        Root-mean-square of ``z_measured - z_model`` (mm).  This is the
        number that says how representative the idealization is.
    r_squared : float
        Coefficient of determination.  Can be negative for a family that
        fits worse than a horizontal line.
    aic, bic : float
        Information criteria, ``n ln(RSS/n) + 2k`` and
        ``n ln(RSS/n) + k ln n``.  :func:`rank_families` sorts on BIC,
        **not** on the raw residual — see there for why that matters.
    n_points : int
        How many points the fit used.
    trend_coeffs : tuple[float, float] or None
        The fitted ``(slope, offset)`` of the measurement tilt and
        datum, or ``None`` when trend fitting was off.  Reported so a
        user can see how much tilt their digitization carried.
    """

    profile: WrinkleProfile
    family: str
    amplitude: float
    wavelength: float
    width: float
    center: float
    fitted_parameters: tuple[str, ...]
    sigmas: dict[str, float] = field(default_factory=dict)
    rms_residual: float = 0.0
    r_squared: float = 0.0
    aic: float = math.inf
    bic: float = math.inf
    n_points: int = 0
    trend_coeffs: tuple[float, float] | None = None

    def to_config_kwargs(self) -> dict[str, float]:
        """Geometry keyword arguments for :class:`AnalysisConfig`.

        One call from a measured trace to a runnable analysis::

            fit = rank_families(x, z)[0]
            cfg = AnalysisConfig(**fit.to_config_kwargs(), angles=[0.0] * 8)
        """
        return {
            "amplitude": self.amplitude,
            "wavelength": self.wavelength,
            "width": self.width,
        }

    def summary(self) -> str:
        """One-line human summary, for a report or a ranking table."""
        parts = []
        for name in self.fitted_parameters:
            value = getattr(self, name)
            s = self.sigmas.get(name, math.inf)
            parts.append(
                f"{name}={value:.4g}"
                + (f"+/-{s:.2g}" if math.isfinite(s) else "+/-?")
            )
        return (
            f"{self.family}: " + ", ".join(parts)
            + f" | RMS={self.rms_residual:.4g} mm, R^2={self.r_squared:.4f}"
        )


def _first_data_line(path: str | Path, *, skip_header: int = 0) -> str:
    """The first non-blank, non-comment line after the header, or ``""``."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for _ in range(skip_header):
                if not handle.readline():
                    return ""
            while True:
                line = handle.readline()
                if not line:
                    return ""
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    return stripped
    except OSError:
        return ""


def _sniff_delimiter(first_line: str) -> str | None:
    """Guess the column separator from the first data line.

    ``numpy.genfromtxt`` treats ``delimiter=None`` as "split on runs of
    whitespace", which silently turns a comma-separated file into a
    single column of ``nan`` rather than raising.  Sniffing keeps the
    advertised "CSV or whitespace" behaviour honest.

    Returns ``None`` (numpy's whitespace splitting) when no delimiter
    character is present.
    """
    for candidate in (",", ";", "\t"):
        if candidate in first_line:
            return candidate
    return None


def load_trace(
    path: str | Path,
    *,
    x_column: int = 0,
    z_column: int = 1,
    scale: float = 1.0,
    z_scale: float | None = None,
    delimiter: str | None = None,
    skip_header: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a 2-column ``(x, z)`` trace from a CSV/text file.

    Parameters
    ----------
    path : str or Path
        File to read.  Anything ``numpy.genfromtxt`` accepts.
    x_column, z_column : int
        Zero-based column indices.
    scale : float
        Multiplies **both** columns — the pixel-to-millimetre factor for
        a trace digitized off a micrograph at a known magnification.
    z_scale : float or None
        Extra factor on ``z`` alone, for the case where the two axes
        were calibrated separately (a C-scan with different in-plane and
        depth scales).  Applied after ``scale``.
    delimiter : str or None
        ``None`` (default) sniffs the file: comma, semicolon or tab if
        the first data line contains one, otherwise whitespace.  Pass an
        explicit delimiter to override.  This cannot be left to numpy —
        ``genfromtxt(delimiter=None)`` splits on whitespace *only*, so a
        comma-separated file parses as one column of ``nan``.
    skip_header : int
        Header lines to skip.

    Returns
    -------
    x, z : np.ndarray
        Two 1-D arrays, sorted by ``x``, with non-finite rows dropped.

    Raises
    ------
    ValueError
        If the file has fewer columns than requested, or fewer than four
        usable rows (no family here has fewer than three parameters).
    """
    first_line = _first_data_line(path, skip_header=skip_header)
    if delimiter is None:
        delimiter = _sniff_delimiter(first_line)
    data = np.genfromtxt(
        path, delimiter=delimiter, skip_header=skip_header, dtype=float,
    )
    if data.ndim == 1:
        # genfromtxt collapses to 1-D for a file with one row *or* one
        # column, and the two need opposite reshapes.  ``np.atleast_2d``
        # always assumes one row, which turns a single-column file into
        # a wide single row and lets it slip past the column check with
        # x and z read from the same data.  Disambiguate on the file.
        n_fields = len(
            first_line.split(delimiter) if delimiter else first_line.split()
        )
        data = data.reshape(1, -1) if n_fields > 1 else data.reshape(-1, 1)
    if data.ndim != 2 or data.shape[1] <= max(x_column, z_column):
        raise ValueError(
            f"{path}: expected at least {max(x_column, z_column) + 1} "
            f"columns, got shape {data.shape}."
        )
    x = data[:, x_column] * float(scale)
    z = data[:, z_column] * float(scale)
    if z_scale is not None:
        z = z * float(z_scale)

    finite = np.isfinite(x) & np.isfinite(z)
    x, z = x[finite], z[finite]
    if x.size < 4:
        raise ValueError(
            f"{path}: only {x.size} usable rows after dropping non-finite "
            "values; a profile fit needs at least 4."
        )
    order = np.argsort(x)
    return x[order], z[order]


def _dominant_period(x: np.ndarray, z: np.ndarray) -> float:
    """Wavelength guess from the dominant FFT component.

    Falls back to the full span when the trace carries no clear period
    (a single bump, say), which is the right starting point for the
    families that have no wavelength anyway.
    """
    span = float(x[-1] - x[0])
    n = int(x.size)
    if n < 8 or span <= 0.0:
        return max(span, 1.0)
    # Resample onto a uniform grid; digitized traces are rarely uniform.
    xi = np.linspace(x[0], x[-1], n)
    zi = np.interp(xi, x, z)
    zi = zi - zi.mean()
    spectrum = np.abs(np.fft.rfft(zi))
    if spectrum.size < 2:
        return max(span, 1.0)
    k = int(np.argmax(spectrum[1:]) + 1)          # skip the DC bin
    freqs = np.fft.rfftfreq(n, d=span / (n - 1))
    if freqs[k] <= 0.0:
        return max(span, 1.0)
    return float(1.0 / freqs[k])


def _initial_guess(
    x: np.ndarray, z: np.ndarray, *, detrend: bool = False,
) -> dict[str, float]:
    """Starting parameters derived from the data, not from defaults.

    With ``detrend`` the guess is taken from a line-subtracted copy of
    the trace.  That is a different question from how the *fit* handles
    tilt: subtracting a line biases the estimate (see
    :func:`fit_profile`), but a seed only has to land in the right
    basin, and on a tilted trace every one of these statistics is
    otherwise swamped.  Measured on a 0.5 mm wrinkle under a 0.05 mm/mm
    tilt, the raw guess gives amplitude 1.0 (twice the truth, since the
    tilt owns the peak-to-peak), wavelength 40.8 (the tilt owns the
    dominant FFT bin) and a centre pinned to the trace edge; the
    optimiser then converges to a local minimum 55 % low in amplitude.
    Seeding from the detrended copy recovers the amplitude exactly at
    every tilt, because the trend the seed ignores is still fitted.
    """
    if detrend and x.size >= 2 and float(x[-1] - x[0]) > 0.0:
        slope, offset = np.polyfit(x, z, 1)
        z = z - (slope * x + offset)
    span = float(x[-1] - x[0])
    amplitude = 0.5 * float(np.ptp(z))
    if amplitude <= 0.0:
        amplitude = 1.0e-6
    center = float(x[int(np.argmax(np.abs(z - np.median(z))))])
    wavelength = _dominant_period(x, z)

    # Envelope span: the extent over which |z| stays above a tenth of
    # the peak.  ``width`` enters as exp(-(dx/w)^2), so the half-extent
    # is a good scale for it.
    peak = float(np.max(np.abs(z)))
    if peak > 0.0:
        inside = x[np.abs(z) >= 0.1 * peak]
        width = (
            0.5 * float(inside[-1] - inside[0])
            if inside.size >= 2
            else span / 4.0
        )
    else:
        width = span / 4.0
    width = max(width, span / 100.0)

    return {
        "amplitude": amplitude,
        "wavelength": max(wavelength, span / 100.0),
        "width": width,
        "center": center,
    }


def fit_profile(
    x: np.ndarray,
    z: np.ndarray,
    family: str = "gaussian_sinusoidal",
    *,
    detrend: bool = True,
    max_nfev: int = 20_000,
) -> FitResult:
    """Fit one profile family to a measured trace by least squares.

    Parameters
    ----------
    x, z : array_like
        The measured trace.  ``x`` along the laminate, ``z`` out of
        plane, both in mm (use :func:`load_trace` to scale from pixels).
    family : str
        A key of :data:`FAMILIES`, or ``"auto"`` to fit every family and
        return the best (equivalent to ``rank_families(x, z)[0]`` — see
        there for what "best" means and why it is not the raw residual).
    detrend : bool
        Fit a measurement tilt and datum offset (``+ a x + b``) jointly
        with the shape.  A digitized micrograph trace is almost never
        level, and an uncorrected tilt inflates the peak-to-peak and so
        biases the amplitude badly — measured, a 0.05 mm/mm tilt puts
        the recovered amplitude 74 % high.

        These are **nuisance parameters estimated jointly**, not removed
        in a pre-step, and that is deliberate.  Subtracting a pre-fitted
        line is biased whenever the feature is not symmetric in the
        window, because the line absorbs part of the wrinkle; and a
        full-trace detrend subtracts the *mean*, which pushes the far
        field negative — somewhere a family that decays to zero cannot
        follow.  Measured on a Gaussian bump, pre-detrending gave RMS
        0.0856 where joint fitting gives 0.0043.  The two extra
        parameters cost every family the same, so the BIC ordering in
        :func:`rank_families` is unaffected.
    max_nfev : int
        Iteration cap handed to the optimiser.

    Returns
    -------
    FitResult

    Raises
    ------
    ValueError
        For an unknown family, mismatched array lengths, or fewer points
        than the model has parameters.
    RuntimeError
        If the optimiser fails to converge.
    """
    from scipy.optimize import curve_fit

    if family == "auto":
        ranked = rank_families(x, z, detrend=detrend, max_nfev=max_nfev)
        if not ranked:
            raise RuntimeError(
                "no profile family could be fitted to this trace; check "
                "the units and that x and z are the columns you expect"
            )
        return ranked[0]
    if family not in FAMILIES:
        raise ValueError(
            f"unknown profile family {family!r}; valid families are "
            f"{sorted(FAMILIES)} (or 'auto' to pick one)"
        )
    x = np.asarray(x, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    if x.shape != z.shape:
        raise ValueError(
            f"x and z must have the same length, got {x.shape} and {z.shape}"
        )
    finite = np.isfinite(x) & np.isfinite(z)
    x, z = x[finite], z[finite]
    order = np.argsort(x)
    x, z = x[order], z[order]

    cls, free = FAMILIES[family]
    trend = bool(detrend)
    n_shape = len(free)
    n_params = n_shape + (2 if trend else 0)
    if x.size < n_params:
        raise ValueError(
            f"fitting {family!r} needs at least {n_params} points "
            f"({n_shape} shape"
            + (" + 2 trend" if trend else "")
            + f"), got {x.size}"
        )

    guess = _initial_guess(x, z, detrend=trend)
    # Placeholders for the parameter a degenerate family ignores.  They
    # must still be valid for the constructor (positive); they are
    # simply never read by that family's ``displacement``.
    fixed = dict(guess)

    def model(xq: np.ndarray, *values: float) -> np.ndarray:
        params = dict(fixed)
        params.update(dict(zip(free, values[:n_shape], strict=True)))
        try:
            profile = cls(
                amplitude=abs(params["amplitude"]),
                wavelength=abs(params["wavelength"]),
                width=abs(params["width"]),
                center=params["center"],
            )
        except ValueError:
            # The optimiser probed an invalid corner; steer it back with
            # a large but finite residual rather than raising out of it.
            return np.full_like(xq, 1.0e6)
        out = np.asarray(profile.displacement(xq), dtype=float)
        if trend:
            out = out + values[n_shape] * xq + values[n_shape + 1]
        return out

    span = float(x[-1] - x[0]) or 1.0
    p0 = [guess[name] for name in free]
    lower = [
        0.0 if name in ("amplitude", "wavelength", "width") else x[0] - span
        for name in free
    ]
    upper = [np.inf if name != "center" else x[-1] + span for name in free]
    if trend:
        slope0, offset0 = np.polyfit(x, z, 1)
        p0 += [float(slope0), float(offset0)]
        lower += [-np.inf, -np.inf]
        upper += [np.inf, np.inf]

    try:
        popt, pcov = curve_fit(
            model, x, z, p0=p0, bounds=(lower, upper), max_nfev=max_nfev,
        )
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"least-squares fit of {family!r} did not converge: {exc}"
        ) from exc

    values = dict(fixed)
    values.update(
        dict(zip(free, (float(v) for v in popt[:n_shape]), strict=True))
    )
    for key in ("amplitude", "wavelength", "width"):
        values[key] = abs(values[key])
    trend_coeffs: tuple[float, float] | None = (
        (float(popt[n_shape]), float(popt[n_shape + 1])) if trend else None
    )

    with np.errstate(invalid="ignore"):
        diag = (
            np.diag(pcov) if pcov is not None
            else np.full(n_params, np.inf)
        )
    sigmas = {
        name: (float(np.sqrt(d)) if np.isfinite(d) and d >= 0.0 else math.inf)
        for name, d in zip(free, diag[:n_shape], strict=False)
    }

    profile = cls(
        amplitude=values["amplitude"],
        wavelength=values["wavelength"],
        width=values["width"],
        center=values["center"],
    )
    fitted = np.asarray(profile.displacement(x), dtype=float)
    if trend_coeffs is not None:
        fitted = fitted + trend_coeffs[0] * x + trend_coeffs[1]
    residual = z - fitted
    rms = float(np.sqrt(np.mean(residual ** 2)))
    ss_tot = float(np.sum((z - z.mean()) ** 2))
    rss = float(np.sum(residual ** 2))
    r2 = 1.0 - rss / ss_tot if ss_tot > 0.0 else 0.0

    # Information criteria, so families with different parameter counts
    # can be compared.  The Gaussian-error log-likelihood reduces to
    # n ln(RSS/n) up to an additive constant identical across families,
    # which therefore cannot affect the ordering.
    n_pts = int(x.size)
    if rss > 0.0 and n_pts > 0:
        ll_term = n_pts * math.log(rss / n_pts)
        aic = ll_term + 2.0 * n_params
        bic = ll_term + n_params * math.log(n_pts)
    else:                               # an exact fit: nothing to penalise
        aic = bic = -math.inf

    return FitResult(
        profile=profile,
        family=family,
        amplitude=values["amplitude"],
        wavelength=values["wavelength"],
        width=values["width"],
        center=values["center"],
        fitted_parameters=tuple(free),
        sigmas=sigmas,
        rms_residual=rms,
        r_squared=float(r2),
        aic=float(aic),
        bic=float(bic),
        n_points=n_pts,
        trend_coeffs=trend_coeffs,
    )


def rank_families(
    x: np.ndarray,
    z: np.ndarray,
    *,
    families: list[str] | None = None,
    detrend: bool = True,
    criterion: str = "bic",
    **kwargs: Any,
) -> list[FitResult]:
    """Fit every family and return them best-first.

    This is the "which morphology?" question answered with a number
    rather than a judgement call: ``rank_families(x, z)[0]`` is the
    family that best describes the measured trace.

    **Ranked by BIC, not by raw residual, and that matters.**  Issue
    #270 asks for a ranking by residual, but a raw-residual ranking
    cannot satisfy its own acceptance criterion: three of these families
    have four shape parameters and two have three, and a
    four-parameter family can always match a three-parameter one at
    least as well.  Measured on data generated by ``pure_sinusoidal``,
    the residuals come out a statistical tie (``triangular`` 0.016670
    against the true ``pure`` 0.016721 — a 0.3 % gap that is pure noise)
    and the raw ranking puts the wrong family first.  BIC breaks that
    tie correctly by charging for the extra parameters, and recovers the
    generating family for all five.

    Pass ``criterion="rms"`` for the literal residual ordering when you
    want it — comparing two families of equal parameter count, say,
    where the penalty is constant and cancels.

    Parameters
    ----------
    families : list[str] or None
        Which families to try.  Default: all of :data:`FAMILIES`.
    detrend : bool
        Passed to :func:`fit_profile`.  Applied identically to every
        family, so the comparison stays like-for-like.
    criterion : {"bic", "aic", "rms"}
        What to sort on.  ``"bic"`` (default) penalises parameter count
        most strongly and is the right default for picking a morphology;
        ``"aic"`` penalises more weakly; ``"rms"`` not at all.

    Returns
    -------
    list[FitResult]
        Sorted best-first.  Families whose fit fails to converge are
        omitted rather than ranked last — a family that cannot describe
        the data at all should not sit at the bottom of the table as
        though it merely fit poorly.  Empty only if every family failed,
        which is itself worth reporting.
    """
    if criterion not in ("bic", "aic", "rms"):
        raise ValueError(
            f"criterion must be 'bic', 'aic' or 'rms', got {criterion!r}"
        )
    names = list(families) if families is not None else list(FAMILIES)
    results: list[FitResult] = []
    for name in names:
        try:
            results.append(fit_profile(x, z, name, detrend=detrend, **kwargs))
        except (RuntimeError, ValueError):
            continue
    key = {
        "bic": lambda r: r.bic,
        "aic": lambda r: r.aic,
        "rms": lambda r: r.rms_residual,
    }[criterion]
    results.sort(key=key)
    return results
