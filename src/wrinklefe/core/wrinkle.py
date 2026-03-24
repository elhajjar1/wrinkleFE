"""Parametric wrinkle profiles and 3D wrinkle surfaces for composite fiber waviness modeling.

Implements the Jin et al. (2026) Gaussian-enveloped sinusoidal wrinkle geometry
and several alternative envelope shapes. Each profile class provides analytical
displacement, slope, and curvature, plus convenience methods for fiber
misalignment angle computation.

References
----------
Jin, L. et al. (2026). Interlaminar damage analysis of dual-wrinkled composite
laminates under multidirectional static loading. Thin-Walled Structures,
219, 114237.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class WrinkleProfile(ABC):
    """Base class for 1-D wrinkle profiles z(x).

    Parameters
    ----------
    amplitude : float
        Peak out-of-plane displacement *A* (mm).
    wavelength : float
        Sinusoidal wavelength *lambda* (mm).
    width : float
        Envelope width parameter *w* (mm).  Interpretation depends on the
        concrete subclass (Gaussian sigma, half-extent, etc.).
    center : float, optional
        Longitudinal center position *x0* (mm).  Default is 0.0.
    """

    def __init__(
        self,
        amplitude: float,
        wavelength: float,
        width: float,
        center: float = 0.0,
    ) -> None:
        if amplitude < 0:
            raise ValueError("amplitude must be non-negative")
        if wavelength <= 0:
            raise ValueError("wavelength must be positive")
        if width <= 0:
            raise ValueError("width must be positive")
        self.amplitude = amplitude
        self.wavelength = wavelength
        self.width = width
        self.center = center

    # ---- abstract interface ------------------------------------------------

    @abstractmethod
    def displacement(self, x: np.ndarray) -> np.ndarray:
        """Out-of-plane displacement z(x) (mm)."""

    @abstractmethod
    def slope(self, x: np.ndarray) -> np.ndarray:
        """First derivative dz/dx (dimensionless)."""

    @abstractmethod
    def curvature(self, x: np.ndarray) -> np.ndarray:
        """Second derivative d^2z/dx^2 (1/mm)."""

    # ---- concrete helpers --------------------------------------------------

    def domain(self) -> Tuple[float, float]:
        """Effective longitudinal extent covering 99.7 % of the Gaussian envelope.

        Returns ``(center - 3*width, center + 3*width)``.
        """
        return (self.center - 3.0 * self.width, self.center + 3.0 * self.width)

    def fiber_angle(self, x: np.ndarray) -> np.ndarray:
        """Local fiber misalignment angle theta(x) = arctan(dz/dx) (radians).

        Parameters
        ----------
        x : array_like
            Longitudinal coordinate(s) (mm).

        Returns
        -------
        numpy.ndarray
            Fiber angle at each *x* location (radians).
        """
        return np.arctan(self.slope(np.asarray(x, dtype=float)))

    def max_angle(self) -> float:
        """Numerical maximum fiber misalignment angle (radians).

        Uses :func:`scipy.optimize.minimize_scalar` to find max |dz/dx| over
        the profile domain, then returns ``arctan(max|dz/dx|)``.
        """
        xlo, xhi = self.domain()

        def neg_abs_slope(xv: float) -> float:
            return -float(np.abs(self.slope(np.atleast_1d(xv))[0]))

        result = minimize_scalar(neg_abs_slope, bounds=(xlo, xhi), method="bounded")
        return float(np.arctan(np.abs(self.slope(np.atleast_1d(result.x))[0])))

    def max_angle_approx(self) -> float:
        """Closed-form approximation: theta ~ arctan(2*pi*A / lambda) (radians)."""
        return float(np.arctan(2.0 * np.pi * self.amplitude / self.wavelength))


# ---------------------------------------------------------------------------
# Concrete profile classes
# ---------------------------------------------------------------------------

class GaussianSinusoidal(WrinkleProfile):
    r"""Jin et al. Gaussian-enveloped sinusoidal wrinkle.

    .. math::
        z(x) = A \exp\!\bigl(-(x-x_0)^2 / w^2\bigr)\,
               \cos\!\bigl(2\pi (x-x_0)/\lambda\bigr)

    Parameters
    ----------
    amplitude, wavelength, width, center
        See :class:`WrinkleProfile`.
    """

    def displacement(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        dx = x - self.center
        A = self.amplitude
        w = self.width
        lam = self.wavelength
        gauss = np.exp(-(dx ** 2) / (w ** 2))
        cos_term = np.cos(2.0 * np.pi * dx / lam)
        return A * gauss * cos_term

    def slope(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        dx = x - self.center
        A = self.amplitude
        w = self.width
        lam = self.wavelength
        k = 2.0 * np.pi / lam
        gauss = np.exp(-(dx ** 2) / (w ** 2))
        cos_term = np.cos(k * dx)
        sin_term = np.sin(k * dx)
        dgauss = -2.0 * dx / (w ** 2)
        return A * gauss * (dgauss * cos_term - k * sin_term)

    def curvature(self, x: np.ndarray) -> np.ndarray:
        r"""Full analytical second derivative.

        .. math::
            \frac{d^2z}{dx^2} = A\,e^{-(x-x_0)^2/w^2}
            \Bigl[\bigl(\tfrac{4(x-x_0)^2}{w^4} - \tfrac{2}{w^2}\bigr)
            \cos(k\,\xi)
            + \tfrac{4(x-x_0)}{w^2}\,k\,\sin(k\,\xi)
            - k^2\cos(k\,\xi)\Bigr]

        where :math:`\xi = x - x_0` and :math:`k = 2\pi/\lambda`.
        """
        x = np.asarray(x, dtype=float)
        dx = x - self.center
        A = self.amplitude
        w = self.width
        lam = self.wavelength
        k = 2.0 * np.pi / lam
        gauss = np.exp(-(dx ** 2) / (w ** 2))
        cos_term = np.cos(k * dx)
        sin_term = np.sin(k * dx)
        d2gauss = (4.0 * dx ** 2 / w ** 4) - (2.0 / w ** 2)
        cross = 4.0 * dx * k / (w ** 2)
        return A * gauss * (d2gauss * cos_term + cross * sin_term - k ** 2 * cos_term)


class RectangularSinusoidal(WrinkleProfile):
    r"""Uniform-amplitude sinusoidal wrinkle with smooth tanh tapered ends.

    The envelope is effectively 1 for :math:`|x - x_0| < w/2` and tapers
    smoothly to 0 outside using a ``tanh`` transition with
    ``taper_width = w / 20``.

    .. math::
        \text{env}(x) = \tfrac{1}{2}\bigl[
            \tanh\!\bigl((w/2 - |x - x_0|)/t_w\bigr) + 1\bigr]

    Parameters
    ----------
    amplitude, wavelength, width, center
        See :class:`WrinkleProfile`.
    """

    def _envelope(self, x: np.ndarray) -> np.ndarray:
        dx = np.abs(x - self.center)
        tw = self.width / 20.0
        return 0.5 * (np.tanh((self.width / 2.0 - dx) / tw) + 1.0)

    def _d_envelope(self, x: np.ndarray) -> np.ndarray:
        """First derivative of the envelope with respect to *x*."""
        dx = x - self.center
        adx = np.abs(dx)
        tw = self.width / 20.0
        arg = (self.width / 2.0 - adx) / tw
        sech2 = 1.0 / np.cosh(arg) ** 2
        # d|dx|/dx = sign(dx)
        sign = np.sign(dx)
        return 0.5 * sech2 * (-sign / tw)

    def _d2_envelope(self, x: np.ndarray) -> np.ndarray:
        """Second derivative of the envelope with respect to *x*."""
        dx = x - self.center
        adx = np.abs(dx)
        tw = self.width / 20.0
        arg = (self.width / 2.0 - adx) / tw
        sech2 = 1.0 / np.cosh(arg) ** 2
        tanh_val = np.tanh(arg)
        sign = np.sign(dx)
        # d2(env)/dx2 = 0.5 * 2 * sech^2(arg) * tanh(arg) * (sign/tw)^2 / tw
        # via chain rule: d/dx[sech^2(arg)*(-sign/tw)] = ...
        # Simplified: second deriv = sech^2(arg)*tanh(arg) / tw^2
        return sech2 * tanh_val / (tw ** 2)

    def displacement(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        dx = x - self.center
        return self.amplitude * self._envelope(x) * np.cos(2.0 * np.pi * dx / self.wavelength)

    def slope(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        dx = x - self.center
        k = 2.0 * np.pi / self.wavelength
        env = self._envelope(x)
        denv = self._d_envelope(x)
        cos_term = np.cos(k * dx)
        sin_term = np.sin(k * dx)
        return self.amplitude * (denv * cos_term - env * k * sin_term)

    def curvature(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        dx = x - self.center
        k = 2.0 * np.pi / self.wavelength
        env = self._envelope(x)
        denv = self._d_envelope(x)
        d2env = self._d2_envelope(x)
        cos_term = np.cos(k * dx)
        sin_term = np.sin(k * dx)
        return self.amplitude * (
            d2env * cos_term
            - 2.0 * denv * k * sin_term
            - env * k ** 2 * cos_term
        )

    def domain(self) -> Tuple[float, float]:
        half = self.width / 2.0 + self.width / 4.0  # extend slightly beyond taper
        return (self.center - half, self.center + half)


class TriangularSinusoidal(WrinkleProfile):
    r"""Triangular (linear-decay) envelope sinusoidal wrinkle.

    .. math::
        \text{env}(x) = \max\!\bigl(0,\; 1 - |x - x_0|/w\bigr)

    The envelope derivative is discontinuous at :math:`x = x_0` and at the
    envelope edges.  The slope and curvature are computed analytically within
    the non-zero region and set to zero outside.

    Parameters
    ----------
    amplitude, wavelength, width, center
        See :class:`WrinkleProfile`.
    """

    def _envelope(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, 1.0 - np.abs(x - self.center) / self.width)

    def _d_envelope(self, x: np.ndarray) -> np.ndarray:
        dx = x - self.center
        adx = np.abs(dx)
        inside = adx < self.width
        denv = np.where(inside, -np.sign(dx) / self.width, 0.0)
        return denv

    def displacement(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        dx = x - self.center
        return self.amplitude * self._envelope(x) * np.cos(2.0 * np.pi * dx / self.wavelength)

    def slope(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        dx = x - self.center
        k = 2.0 * np.pi / self.wavelength
        env = self._envelope(x)
        denv = self._d_envelope(x)
        cos_term = np.cos(k * dx)
        sin_term = np.sin(k * dx)
        return self.amplitude * (denv * cos_term - env * k * sin_term)

    def curvature(self, x: np.ndarray) -> np.ndarray:
        r"""Curvature within the non-zero envelope region.

        The triangular envelope has a discontinuous first derivative, so the
        second derivative of the envelope is zero almost everywhere (it is a
        Dirac delta at the kinks, which we ignore for practical purposes).
        Therefore:

        .. math::
            \frac{d^2 z}{dx^2} \approx
                2\,\text{denv}\,(-k\sin(k\xi))
                + \text{env}\,(-k^2\cos(k\xi))
        """
        x = np.asarray(x, dtype=float)
        dx = x - self.center
        k = 2.0 * np.pi / self.wavelength
        env = self._envelope(x)
        denv = self._d_envelope(x)
        cos_term = np.cos(k * dx)
        sin_term = np.sin(k * dx)
        return self.amplitude * (
            -2.0 * denv * k * sin_term - env * k ** 2 * cos_term
        )


class PureSinusoidal(WrinkleProfile):
    r"""Infinite sinusoidal wrinkle (no envelope).

    .. math::
        z(x) = A\cos\!\bigl(2\pi(x - x_0)/\lambda\bigr)

    The *width* parameter is not used for the profile shape but is retained
    for interface compatibility.  The default domain spans
    :math:`[x_0 - 3\lambda,\; x_0 + 3\lambda]`.

    Parameters
    ----------
    amplitude, wavelength, width, center
        See :class:`WrinkleProfile`.
    """

    def displacement(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        dx = x - self.center
        return self.amplitude * np.cos(2.0 * np.pi * dx / self.wavelength)

    def slope(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        dx = x - self.center
        k = 2.0 * np.pi / self.wavelength
        return -self.amplitude * k * np.sin(k * dx)

    def curvature(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        dx = x - self.center
        k = 2.0 * np.pi / self.wavelength
        return -self.amplitude * k ** 2 * np.cos(k * dx)

    def domain(self) -> Tuple[float, float]:
        """Default domain: 3 wavelengths on each side of center."""
        return (self.center - 3.0 * self.wavelength, self.center + 3.0 * self.wavelength)

    def max_angle(self) -> float:
        """Exact result for a pure cosine: arctan(2*pi*A/lambda)."""
        return self.max_angle_approx()


class GaussianBump(WrinkleProfile):
    r"""Single Gaussian out-of-plane bump (no oscillation).

    .. math::
        z(x) = A\,\exp\!\bigl(-(x - x_0)^2 / w^2\bigr)

    Parameters
    ----------
    amplitude, wavelength, width, center
        See :class:`WrinkleProfile`.  The *wavelength* parameter is not used
        for the shape but is retained for interface compatibility.
    """

    def displacement(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        dx = x - self.center
        return self.amplitude * np.exp(-(dx ** 2) / (self.width ** 2))

    def slope(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        dx = x - self.center
        w = self.width
        return -2.0 * self.amplitude * dx / (w ** 2) * np.exp(-(dx ** 2) / (w ** 2))

    def curvature(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        dx = x - self.center
        w = self.width
        gauss = np.exp(-(dx ** 2) / (w ** 2))
        return self.amplitude * (4.0 * dx ** 2 - 2.0 * w ** 2) / (w ** 4) * gauss

    def max_angle(self) -> float:
        """Analytical maximum slope occurs at x - x0 = +/- w/sqrt(2).

        max|dz/dx| = 2A / (w * sqrt(2*e))
        """
        max_slope = 2.0 * self.amplitude / (self.width * np.sqrt(2.0 * np.e))
        return float(np.arctan(max_slope))


# ---------------------------------------------------------------------------
# 3-D wrinkle surface
# ---------------------------------------------------------------------------

class WrinkleSurface3D:
    """Extends a 1-D :class:`WrinkleProfile` into a 3-D surface z(x, y).

    The surface is constructed as the product of the longitudinal profile and
    a transverse modulation function:

    .. math::
        z(x, y) = \\text{profile}(x)\\;f(y)

    Parameters
    ----------
    profile : WrinkleProfile
        Underlying 1-D wrinkle profile.
    transverse_mode : str
        One of ``"uniform"``, ``"gaussian_decay"``, ``"sinusoidal_y"``,
        or ``"elliptical"``.
    width_y : float
        Transverse width parameter (mm).  Used by ``"gaussian_decay"``
        and ``"elliptical"`` modes.
    span_y : float
        Total specimen width in the *y*-direction (mm).  Used by
        ``"sinusoidal_y"`` mode and to define the transverse centerline
        for ``"gaussian_decay"`` and ``"elliptical"`` modes.
    """

    _VALID_MODES = {"uniform", "gaussian_decay", "sinusoidal_y", "elliptical"}

    def __init__(
        self,
        profile: WrinkleProfile,
        transverse_mode: str = "uniform",
        width_y: float = 10.0,
        span_y: float = 20.0,
    ) -> None:
        if transverse_mode not in self._VALID_MODES:
            raise ValueError(
                f"transverse_mode must be one of {self._VALID_MODES}, "
                f"got '{transverse_mode}'"
            )
        if width_y <= 0:
            raise ValueError("width_y must be positive")
        if span_y <= 0:
            raise ValueError("span_y must be positive")
        self.profile = profile
        self.transverse_mode = transverse_mode
        self.width_y = width_y
        self.span_y = span_y

    # ---- transverse functions ----------------------------------------------

    def _f_y(self, y: np.ndarray) -> np.ndarray:
        """Transverse modulation f(y)."""
        y = np.asarray(y, dtype=float)
        if self.transverse_mode == "uniform":
            return np.ones_like(y)
        elif self.transverse_mode == "gaussian_decay":
            y_centered = y - self.span_y / 2.0
            return np.exp(-(y_centered ** 2) / (self.width_y ** 2))
        elif self.transverse_mode == "sinusoidal_y":
            return np.cos(np.pi * y / self.span_y)
        elif self.transverse_mode == "elliptical":
            y_centered = y - self.span_y / 2.0
            ratio = y_centered / self.width_y
            return np.sqrt(np.maximum(0.0, 1.0 - ratio ** 2))
        # Should not reach here due to __init__ validation
        raise ValueError(f"Unknown transverse mode: {self.transverse_mode}")

    def _df_dy(self, y: np.ndarray) -> np.ndarray:
        """First derivative of the transverse modulation df/dy."""
        y = np.asarray(y, dtype=float)
        if self.transverse_mode == "uniform":
            return np.zeros_like(y)
        elif self.transverse_mode == "gaussian_decay":
            y_centered = y - self.span_y / 2.0
            return -2.0 * y_centered / (self.width_y ** 2) * np.exp(
                -(y_centered ** 2) / (self.width_y ** 2)
            )
        elif self.transverse_mode == "sinusoidal_y":
            return -(np.pi / self.span_y) * np.sin(np.pi * y / self.span_y)
        elif self.transverse_mode == "elliptical":
            y_centered = y - self.span_y / 2.0
            ratio = y_centered / self.width_y
            inner = np.maximum(0.0, 1.0 - ratio ** 2)
            safe = np.where(inner > 0.0, inner, 1.0)  # avoid division by zero
            deriv = np.where(
                inner > 0.0,
                -y_centered / (self.width_y ** 2 * np.sqrt(safe)),
                0.0,
            )
            return deriv
        raise ValueError(f"Unknown transverse mode: {self.transverse_mode}")

    def _d2f_dy2(self, y: np.ndarray) -> np.ndarray:
        """Second derivative of the transverse modulation d^2f/dy^2."""
        y = np.asarray(y, dtype=float)
        if self.transverse_mode == "uniform":
            return np.zeros_like(y)
        elif self.transverse_mode == "gaussian_decay":
            y_centered = y - self.span_y / 2.0
            wy = self.width_y
            gauss = np.exp(-(y_centered ** 2) / (wy ** 2))
            return (4.0 * y_centered ** 2 / wy ** 4 - 2.0 / wy ** 2) * gauss
        elif self.transverse_mode == "sinusoidal_y":
            return -(np.pi / self.span_y) ** 2 * np.cos(np.pi * y / self.span_y)
        elif self.transverse_mode == "elliptical":
            y_centered = y - self.span_y / 2.0
            wy = self.width_y
            ratio = y_centered / wy
            inner = np.maximum(0.0, 1.0 - ratio ** 2)
            safe = np.where(inner > 0.0, inner, 1.0)
            deriv = np.where(
                inner > 0.0,
                -1.0 / (wy ** 2 * safe ** 1.5),
                0.0,
            )
            return deriv
        raise ValueError(f"Unknown transverse mode: {self.transverse_mode}")

    # ---- public interface --------------------------------------------------

    def displacement(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Surface displacement z(x, y) = profile(x) * f(y).

        Parameters
        ----------
        x, y : array_like
            Coordinate arrays (mm).  Must be broadcastable to a common shape.

        Returns
        -------
        numpy.ndarray
            Out-of-plane displacement (mm).
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return self.profile.displacement(x) * self._f_y(y)

    def gradient(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Surface gradient (dz/dx, dz/dy).

        Parameters
        ----------
        x, y : array_like
            Coordinate arrays (mm).  Must be broadcastable.

        Returns
        -------
        dz_dx : numpy.ndarray
            Partial derivative with respect to *x*.
        dz_dy : numpy.ndarray
            Partial derivative with respect to *y*.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        zx = self.profile.slope(x) * self._f_y(y)
        zy = self.profile.displacement(x) * self._df_dy(y)
        return zx, zy

    def curvature_tensor(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""Second-derivative (curvature) tensor of the surface.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(..., 2, 2)`` containing:

            .. math::
                \begin{bmatrix}
                \partial^2 z/\partial x^2 & \partial^2 z/\partial x\,\partial y \\
                \partial^2 z/\partial x\,\partial y & \partial^2 z/\partial y^2
                \end{bmatrix}
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        d2z_dx2 = self.profile.curvature(x) * self._f_y(y)
        d2z_dxdy = self.profile.slope(x) * self._df_dy(y)
        d2z_dy2 = self.profile.displacement(x) * self._d2f_dy2(y)

        # Stack into (..., 2, 2) tensor
        shape = np.broadcast_shapes(np.shape(d2z_dx2), np.shape(d2z_dxdy), np.shape(d2z_dy2))
        tensor = np.empty(shape + (2, 2), dtype=float)
        tensor[..., 0, 0] = d2z_dx2
        tensor[..., 0, 1] = d2z_dxdy
        tensor[..., 1, 0] = d2z_dxdy
        tensor[..., 1, 1] = d2z_dy2
        return tensor

    def fiber_angle(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fiber misalignment angle arctan(dz/dx) assuming fibers run in *x*.

        Parameters
        ----------
        x, y : array_like
            Coordinate arrays (mm).

        Returns
        -------
        numpy.ndarray
            Misalignment angle (radians).
        """
        dz_dx, _ = self.gradient(x, y)
        return np.arctan(dz_dx)
