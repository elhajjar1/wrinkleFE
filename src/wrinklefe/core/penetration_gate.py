"""Two-parameter (theta, D/T) penetration-gate knockdown model (D.3).

The angle-only Budiansky-Fleck kink-band knockdown is *scale-invariant*:
it reduces a wrinkle to its peak misalignment angle ``theta_max`` and so
cannot reproduce the strong dependence of compressive strength on the
through-thickness penetration ``D/T = A/T`` that the Li (2024/2025)
controlled grids expose (VALIDATION_DATA section 2.7): at *fixed* angle,
knockdown still varies from ~1.0 to ~0.6 as the wrinkle penetrates deeper.

This implements the **penetration-gate** form (section 2.7 candidate 1):

    KD(theta, D/T) = 1 - (1 - KD_angle(theta)) * S(D/T),
    KD_angle(theta) = 1 / (1 + theta_rad / gamma_Y),     # Budiansky-Fleck
    S(D/T)          = min(1, (D/T / dt0) ** p).

Limits:

* ``D/T -> 0`` -> ``S -> 0`` -> ``KD -> 1``: a shallow wrinkle realises
  none of its angle-driven knockdown.
* ``S -> 1`` -> ``KD -> KD_angle(theta)``: a deep wrinkle realises all of
  it.

The steep power ``p`` (~4-5) gives the sharp penetration transition the
Li constant-angle triple shows (KD 1.00 -> 0.94 -> 0.63 over
D/T = 0.041 -> 0.122); the gradual Bazant ``1/sqrt(1 + D/T/dt0)``
size-effect form cannot.

Three parameters — the matrix yield strain ``gamma_Y`` (angle response),
the transitional penetration ``dt0`` and the gate exponent ``p`` — are
**material-realization specific** (the Li 2024 moulded and Li 2025
vacuum-bag AC318/S6C10 cannot share a calibration; see section 2.7).
Calibrated presets for both are provided.  **UD-scoped**: do NOT apply to
multidirectional / blocked laminates, whose low-D/T knockdown is
delamination-driven rather than penetration-gated.

References
----------
- VALIDATION_DATA section 2.7 (two-parameter (theta, D/T) model inputs).
- Budiansky, B. & Fleck, N.A. (1993). J. Mech. Phys. Solids 41(1), 183.
- Li, X. et al. (2024) CST 256:110762; Li, Y. et al. (2025) Polym.
  Compos. 46:15176.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GateParameters:
    """Calibrated penetration-gate constants for one material realization.

    Parameters
    ----------
    gamma_Y : float
        Effective matrix yield strain in the Budiansky-Fleck angle floor
        ``KD_angle = 1 / (1 + theta_rad / gamma_Y)``.  Must be > 0.
    dt0 : float
        Transitional through-thickness penetration ``(D/T)0``.  Must be > 0.
    p : float
        Gate exponent (steepness of the penetration transition).  Must
        be > 0.
    name : str
        Identifier (e.g. the material realization).
    """

    gamma_Y: float
    dt0: float
    p: float
    name: str = "gate"
    position_q: float | None = None

    def __post_init__(self) -> None:
        for attr in ("gamma_Y", "dt0", "p"):
            v = getattr(self, attr)
            if not (v > 0 and math.isfinite(v)):
                raise ValueError(
                    f"GateParameters.{attr} must be a positive finite "
                    f"float, got {v}"
                )
        if self.position_q is not None and not (
            self.position_q > 0 and math.isfinite(self.position_q)
        ):
            raise ValueError(
                f"GateParameters.position_q must be a positive finite "
                f"float or None, got {self.position_q}"
            )


# Calibrated presets (least-squares fit to the VALIDATION_DATA section 2.7
# single-wrinkle grids; S-A-2 excluded — near-surface position axis).
#   Li 2024 moulded (Dataset E): MAE 2.8 %, 9/9 within +/-20 %.
#   Li 2025 vacuum-bag (Dataset F): MAE 6.0 %, 5/5 within +/-20 %.
GATE_LI2024_MOULDED = GateParameters(
    gamma_Y=0.2577, dt0=0.0938, p=0.59, name="AC318_S6C10_molded",
)
GATE_LI2025_VACBAG = GateParameters(
    gamma_Y=0.6215, dt0=0.1220, p=4.31, name="AC318_S6C10_vacbag",
    # Through-thickness position factor (item D.5) calibrated to the
    # S-M-2 (Middle, KD 0.629) vs S-A-2 (Above, KD 0.981) pair: a wrinkle
    # at z=10/14 reproduces the measured near-surface mildness.  Steep
    # (q~5.3) -- a single calibration point, so indicative.
    position_q=5.26,
)


def angle_floor(theta_deg, gamma_Y: float):
    """Budiansky-Fleck angle-only knockdown ``1 / (1 + theta_rad/gamma_Y)``.

    Accepts a scalar or array ``theta_deg`` (degrees).
    """
    theta_rad = np.radians(np.asarray(theta_deg, dtype=float))
    return 1.0 / (1.0 + theta_rad / gamma_Y)


def position_factor(z_position, params: GateParameters):
    """Through-thickness position factor (item D.5), in ``[0, 1]``.

    A wrinkle is most damaging at the mid-plane and far milder near a free
    surface (Li 2025 S-A-2 "Above": same theta, D/T as the Middle S-M-2,
    yet KD 0.98 vs 0.63 — a near-surface wrinkle's plies shed load locally
    without failing the bulk).  The factor scales the gate's knockdown
    *deficit*:

        P(z) = (2 * min(z, 1 - z)) ** position_q

    so ``P = 1`` at the mid-plane (``z = 0.5``) and ``P -> 0`` at either
    surface.  Returns 1.0 when ``position_q`` is unset (no position model).
    """
    if params.position_q is None:
        return np.ones_like(np.asarray(z_position, dtype=float))
    z = np.asarray(z_position, dtype=float)
    d = 2.0 * np.minimum(z, 1.0 - z)        # 1 at mid, 0 at a surface
    return np.clip(d, 0.0, 1.0) ** params.position_q


def penetration_gate_kd(theta_deg, dt, params: GateParameters,
                        z_position=0.5):
    """Two-parameter (theta, D/T) penetration-gate knockdown.

    Parameters
    ----------
    theta_deg : float or array
        Peak fibre-misalignment angle ``theta_max`` in degrees.
    dt : float or array
        Through-thickness penetration ratio ``D/T = A/T``.
    params : GateParameters
        Calibrated gate constants for the material realization.
    z_position : float or array, optional
        Through-thickness position of the wrinkle centre as a fraction of
        the laminate thickness (0.5 = mid-plane).  Only consulted when
        ``params.position_q`` is set (item D.5).  Default 0.5.

    Returns
    -------
    float or np.ndarray
        Predicted normalised residual strength (knockdown) in ``(0, 1]``.
    """
    dt_arr = np.asarray(dt, dtype=float)
    s = np.minimum(1.0, (dt_arr / params.dt0) ** params.p)
    ka = angle_floor(theta_deg, params.gamma_Y)
    deficit = (1.0 - ka) * s * position_factor(z_position, params)
    kd = 1.0 - deficit
    # Scalar-in -> scalar-out.
    if (np.isscalar(theta_deg) and np.isscalar(dt)
            and np.isscalar(z_position)):
        return float(kd)
    return kd


def predict_from_geometry(
    amplitude: float,
    wavelength: float,
    n_plies: int,
    ply_thickness: float,
    params: GateParameters,
) -> float:
    """Penetration-gate knockdown from wrinkle geometry.

    Derives the two model inputs from the geometry on the conventions the
    gate was calibrated with (VALIDATION_DATA section 2.7): the peak
    misalignment ``theta_max = arctan(2*pi*A/lambda)`` and the penetration
    ``D/T = A / (n_plies * ply_thickness)``, with ``A`` the wrinkle
    half-amplitude.

    Parameters
    ----------
    amplitude : float
        Wrinkle half-amplitude *A* (mm).
    wavelength : float
        Wrinkle wavelength *lambda* (mm).
    n_plies : int
        Number of plies in the laminate.
    ply_thickness : float
        Ply thickness (mm).
    params : GateParameters
        Calibrated gate constants for the material realization.

    Returns
    -------
    float
        Predicted knockdown.
    """
    if wavelength <= 0 or ply_thickness <= 0 or n_plies <= 0:
        raise ValueError(
            "wavelength, ply_thickness, n_plies must all be > 0"
        )
    theta_deg = math.degrees(math.atan(2.0 * math.pi * amplitude / wavelength))
    dt = amplitude / (n_plies * ply_thickness)
    return float(penetration_gate_kd(theta_deg, dt, params))


def calibrate_gate(theta_deg, dt, kd, *, name: str = "gate") -> GateParameters:
    """Least-squares-fit the gate parameters to measured (theta, D/T, KD).

    Parameters
    ----------
    theta_deg, dt, kd : array-like
        Peak angle (deg), penetration ratio, and measured knockdown for a
        set of single-wrinkle cases.
    name : str
        Identifier for the resulting :class:`GateParameters`.

    Returns
    -------
    GateParameters
        Fitted ``(gamma_Y, dt0, p)``.

    Raises
    ------
    ImportError
        If SciPy is unavailable.
    """
    from scipy.optimize import least_squares

    th = np.asarray(theta_deg, dtype=float)
    d = np.asarray(dt, dtype=float)
    k = np.asarray(kd, dtype=float)

    def resid(par):
        return penetration_gate_kd(th, d, GateParameters(*par, name=name)) - k

    sol = least_squares(
        resid, x0=[0.3, 0.1, 3.0],
        bounds=([1e-3, 1e-3, 0.2], [3.0, 1.0, 12.0]),
    )
    g, t, p = sol.x
    return GateParameters(gamma_Y=float(g), dt0=float(t), p=float(p), name=name)
