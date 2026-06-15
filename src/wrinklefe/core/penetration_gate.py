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

    def __post_init__(self) -> None:
        for attr in ("gamma_Y", "dt0", "p"):
            v = getattr(self, attr)
            if not (v > 0 and math.isfinite(v)):
                raise ValueError(
                    f"GateParameters.{attr} must be a positive finite "
                    f"float, got {v}"
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
)


def angle_floor(theta_deg, gamma_Y: float):
    """Budiansky-Fleck angle-only knockdown ``1 / (1 + theta_rad/gamma_Y)``.

    Accepts a scalar or array ``theta_deg`` (degrees).
    """
    theta_rad = np.radians(np.asarray(theta_deg, dtype=float))
    return 1.0 / (1.0 + theta_rad / gamma_Y)


def penetration_gate_kd(theta_deg, dt, params: GateParameters):
    """Two-parameter (theta, D/T) penetration-gate knockdown.

    Parameters
    ----------
    theta_deg : float or array
        Peak fibre-misalignment angle ``theta_max`` in degrees.
    dt : float or array
        Through-thickness penetration ratio ``D/T = A/T``.
    params : GateParameters
        Calibrated gate constants for the material realization.

    Returns
    -------
    float or np.ndarray
        Predicted normalised residual strength (knockdown) in ``(0, 1]``.
    """
    dt_arr = np.asarray(dt, dtype=float)
    s = np.minimum(1.0, (dt_arr / params.dt0) ** params.p)
    ka = angle_floor(theta_deg, params.gamma_Y)
    kd = 1.0 - (1.0 - ka) * s
    # Scalar-in -> scalar-out.
    if np.isscalar(theta_deg) and np.isscalar(dt):
        return float(kd)
    return kd


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
