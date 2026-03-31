"""Budiansky-Fleck kink-band model and interlaminar damage model.

This module provides two classes from the existing dual-wrinkle analytical
model:

1. **BudianskyFleckKinkBand** -- Kink-band compression failure combined
   with interlaminar damage.  The key concave knockdown function::

       KD = 1 / (1 + theta_eff / gamma_Y)

   creates fat-tailed failure distributions via Jensen's inequality when
   the fibre misalignment theta_eff is a random variable.

2. **InterlaminarDamage** -- Analytical damage index model calibrated
   against Jin et al. FE simulations::

       D = D0 * (A/A1)^1.5 * (1 + beta * max(theta - theta_c, 0)) * M_f

   and the damage-to-strength conversion::

       sigma/sigma_0 = (1 - D)^1.5

Combined Knockdown
------------------
The master equation for compression strength with both mechanisms::

    sigma / sigma_0 = (1 - D)^1.5 / (1 + theta_eff / gamma_Y)
                    = KD_delamination * KD_kinkband

References
----------
- Budiansky, B. & Fleck, N. A. (1993). J. Mech. Phys. Solids, 41(1), 183-211.
- Elhajjar, R. (2025). Scientific Reports, 15:25977 (fat-tail statistics).
- Jin, L. et al. (2026). Thin-Walled Structures, 219:114237 (wrinkle geometry).
"""

from __future__ import annotations

import numpy as np

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.failure.base import FailureCriterion, FailureResult


class BudianskyFleckKinkBand(FailureCriterion):
    """Kink-band compression failure model with interlaminar damage.

    The failure index is evaluated as::

        FI = |sigma_11| / (sigma_c * combined_knockdown)

    where ``sigma_c`` is the pristine compressive strength (``material.Xc``)
    and the combined knockdown factor accounts for both fibre misalignment
    (kink-band theory) and interlaminar damage.

    Parameters
    ----------
    theta_eff : float
        Effective fibre misalignment angle in radians.  This is the
        wrinkle-induced angle modified by the morphology factor::

            theta_eff = theta_max * M_f(phi, loading)

    damage_index : float
        Interlaminar damage index D in [0, 1).  Computed from
        :class:`InterlaminarDamage` or set directly.

    Attributes
    ----------
    name : str
        ``"budiansky_fleck"``

    Notes
    -----
    The kink-band knockdown ``1 / (1 + theta / gamma_Y)`` is a **concave**
    function of the misalignment angle.  By Jensen's inequality, for any
    random variable theta::

        E[f(theta)] <= f(E[theta])

    meaning that computing the knockdown at the mean angle **overestimates**
    the mean knockdown (and hence the mean strength).  Typical Jensen gaps
    are 5-15 % for realistic wrinkle distributions.
    """

    name = "budiansky_fleck"

    def __init__(
        self,
        theta_eff: float = 0.0,
        damage_index: float = 0.0,
    ) -> None:
        if damage_index < 0.0 or damage_index >= 1.0:
            raise ValueError(
                f"damage_index must be in [0, 1), got {damage_index}"
            )
        if theta_eff < 0.0:
            raise ValueError(
                f"theta_eff must be non-negative, got {theta_eff}"
            )
        self.theta_eff = theta_eff
        self.damage_index = damage_index

    # ------------------------------------------------------------------
    # Knockdown factors
    # ------------------------------------------------------------------

    def knockdown(self, gamma_Y: float | None = None, material: OrthotropicMaterial | None = None) -> float:
        """Kink-band knockdown factor.

        .. math::

            KD_{\\text{kink}} = \\frac{1}{1 + \\theta_{\\text{eff}} / \\gamma_Y}

        Parameters
        ----------
        gamma_Y : float, optional
            Matrix yield shear strain.  If *None*, taken from *material*.
        material : OrthotropicMaterial, optional
            Material providing ``gamma_Y`` if not given directly.

        Returns
        -------
        float
            Knockdown factor in (0, 1].
        """
        if gamma_Y is None:
            if material is None:
                raise ValueError("Provide gamma_Y or material")
            gamma_Y = material.gamma_Y
        return 1.0 / (1.0 + self.theta_eff / gamma_Y)

    def delamination_knockdown(self) -> float:
        """Damage-based knockdown factor.

        .. math::

            KD_{\\text{delam}} = (1 - D)^{1.5}

        Returns
        -------
        float
            Knockdown factor in (0, 1].
        """
        return (1.0 - self.damage_index) ** 1.5

    def combined_knockdown(self, gamma_Y: float | None = None, material: OrthotropicMaterial | None = None) -> float:
        """Combined kink-band and delamination knockdown.

        .. math::

            KD = \\frac{(1 - D)^{1.5}}{1 + \\theta_{\\text{eff}} / \\gamma_Y}

        Parameters
        ----------
        gamma_Y : float, optional
            Matrix yield shear strain.
        material : OrthotropicMaterial, optional
            Material providing ``gamma_Y`` if not given directly.

        Returns
        -------
        float
            Combined knockdown factor in (0, 1].
        """
        return self.knockdown(gamma_Y, material) * self.delamination_knockdown()

    # ------------------------------------------------------------------
    # FailureCriterion interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        stress_local: np.ndarray,
        material: OrthotropicMaterial,
        context=None,
    ) -> FailureResult:
        """Evaluate kink-band criterion at a single material point.

        The failure index is::

            FI = |sigma_11| / (Xc * combined_knockdown)

        Only the fibre-direction normal stress is used.  The combined
        knockdown incorporates the effective misalignment and damage
        stored in this instance.

        Parameters
        ----------
        stress_local : np.ndarray
            Shape ``(6,)`` stress vector in local material coordinates.
        material : OrthotropicMaterial
            Material with ``Xc`` (compressive strength) and ``gamma_Y``.

        Returns
        -------
        FailureResult
            Failure index, mode (``"kink_band"``), reserve factor, and
            criterion name.
        """
        stress_local = np.asarray(stress_local, dtype=np.float64)
        s1 = stress_local[0]

        kd = self.combined_knockdown(material=material)
        allowable = material.Xc * kd

        fi = abs(s1) / allowable if allowable > 0 else float("inf")

        mode = "kink_band"
        rf = 1.0 / fi if fi > 0 else float("inf")

        return FailureResult(
            index=float(fi),
            mode=mode,
            reserve_factor=rf,
            criterion_name=self.name,
        )


class InterlaminarDamage:
    """Analytical interlaminar damage model from the dual-wrinkle repository.

    Computes a damage index D that captures the combined effect of wrinkle
    amplitude, fibre misalignment angle, and morphology on interlaminar
    integrity.

    Damage Index
    ------------
    ::

        D = D0 * (A / A1)^1.5 * (1 + beta * max(theta - theta_c, 0)) * M_f

    where:
    - ``D0``   = base damage coefficient (default 0.15)
    - ``A``    = wrinkle amplitude [mm]
    - ``A1``   = reference amplitude (1 ply thickness, default 0.183 mm)
    - ``beta`` = angle sensitivity parameter (default 3.0)
    - ``theta``= maximum fibre misalignment angle [rad]
    - ``theta_c`` = critical angle below which no angle-driven damage (default 0.1 rad)
    - ``M_f``  = morphology factor (from phase offset and loading)

    Damage-to-Strength Conversion
    ------------------------------
    ::

        sigma / sigma_0 = (1 - D)^1.5

    Parameters
    ----------
    D0 : float
        Base damage coefficient (default 0.15).
    beta_angle : float
        Angle sensitivity parameter (default 3.0).
    theta_crit : float
        Critical misalignment angle in radians (default 0.1).

    Examples
    --------
    >>> dmg = InterlaminarDamage()
    >>> D = dmg.damage_index(amplitude=0.366, theta=0.15, morphology_factor=1.0)
    >>> dmg.damage_to_strength(D)  # returns (1-D)^1.5
    """

    def __init__(
        self,
        D0: float = 0.15,
        beta_angle: float = 3.0,
        theta_crit: float = 0.1,
    ) -> None:
        self.D0 = D0
        self.beta_angle = beta_angle
        self.theta_crit = theta_crit

    def damage_index(
        self,
        amplitude: float,
        theta: float,
        morphology_factor: float,
        A1: float = 0.183,
    ) -> float:
        """Compute the interlaminar damage index D.

        Parameters
        ----------
        amplitude : float
            Wrinkle amplitude A [mm].
        theta : float
            Maximum fibre misalignment angle [rad].
        morphology_factor : float
            Morphology factor M_f from phase offset and loading mode.
        A1 : float
            Reference amplitude (1 ply thickness) [mm].  Default 0.183.

        Returns
        -------
        float
            Damage index D in [0, 1).  Clamped to [0, 0.999] for
            numerical safety.
        """
        amplitude_term = (amplitude / A1) ** 1.5
        angle_excess = max(theta - self.theta_crit, 0.0)
        angle_term = 1.0 + self.beta_angle * angle_excess

        D = self.D0 * amplitude_term * angle_term * morphology_factor
        return float(np.clip(D, 0.0, 0.999))

    def damage_to_strength(self, D: float) -> float:
        """Convert damage index to strength retention ratio.

        Parameters
        ----------
        D : float
            Damage index in [0, 1).

        Returns
        -------
        float
            Strength ratio ``(1 - D)^1.5`` in (0, 1].
        """
        return float((1.0 - D) ** 1.5)
