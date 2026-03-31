"""Puck action-plane failure criterion for orthotropic composite laminates.

Implements the Puck criterion with separate fibre failure (FF) and
inter-fibre failure (IFF) evaluations.  The IFF check searches over
fracture plane angles to find the critical orientation.

Fibre Failure (FF) -- Simplified
--------------------------------
Tension (sigma_1 >= 0)::

    FI_FF = sigma_1 / Xt

Compression (sigma_1 < 0)::

    FI_FF = -sigma_1 / Xc

(The full Puck FF includes stress magnification factors from transverse
stresses; this implementation uses the simplified form.)

Inter-Fibre Failure (IFF) -- Action-Plane Search
-------------------------------------------------
The fracture plane is defined by the angle theta measured from the
2-direction.  Stresses resolved on the fracture plane::

    sigma_n(theta)  = s2*cos^2(theta) + s3*sin^2(theta) + 2*t23*sin(theta)*cos(theta)
    tau_nt(theta)   = (s3-s2)*sin(theta)*cos(theta) + t23*(cos^2(theta) - sin^2(theta))
    tau_n1(theta)   = t12*cos(theta) + t13*sin(theta)

Three IFF modes are evaluated depending on the sign and magnitude of
sigma_n:

**Mode A** (sigma_n >= 0)::

    FI_A = sqrt( (tau_nt / (S23 + p_perp_psi_t * sigma_n))^2
               + (tau_n1 / (S12 + p_perp_par_t * sigma_n))^2 )
         + p_perp_psi_t * sigma_n / S23

**Mode B** (sigma_n < 0, |tau_nt/sigma_n| >= threshold)::

    FI_B = sqrt( tau_nt^2 + (tau_n1 + p_perp_par_c * sigma_n)^2 ) / S12
         + p_perp_psi_c * sigma_n / S23

**Mode C** (sigma_n < 0, |tau_nt/sigma_n| < threshold)::

    FI_C = [ (tau_nt / (2*(1 + p_perp_psi_c)*S23))^2 + (tau_n1/S12)^2 ]
           * (-Yc / sigma_n)

The search sweeps theta from -90 deg to +90 deg in 1 deg increments and
returns the maximum IFF failure index found.

Default Puck Inclination Parameters
------------------------------------
- p_perp_par_t  = 0.30  (p_perp_parallel, tension)
- p_perp_par_c  = 0.25  (p_perp_parallel, compression)
- p_perp_psi_t  = 0.20  (p_perp_psi, tension)
- p_perp_psi_c  = 0.25  (p_perp_psi, compression)

References
----------
- Puck, A. & Schurmann, H. (1998). Composites Science and Technology, 58.
- Puck, A. & Schurmann, H. (2002). Composites Science and Technology, 62.
"""

from __future__ import annotations

import numpy as np

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.failure.base import FailureCriterion, FailureResult


class PuckCriterion(FailureCriterion):
    """Puck action-plane failure criterion for 3-D orthotropic composites.

    Evaluates both fibre failure (FF) and inter-fibre failure (IFF).
    The overall failure index is the maximum of the two.

    Parameters
    ----------
    p_perp_par_t : float
        Inclination parameter p_perp_parallel for tension (default 0.30).
    p_perp_par_c : float
        Inclination parameter p_perp_parallel for compression (default 0.25).
    p_perp_psi_t : float
        Inclination parameter p_perp_psi for tension (default 0.20).
    p_perp_psi_c : float
        Inclination parameter p_perp_psi for compression (default 0.25).
    n_theta : int
        Number of fracture-plane angles to search over [-90, 90] degrees
        (default 181, i.e. 1-degree increments).

    Attributes
    ----------
    name : str
        ``"puck"``
    """

    name = "puck"

    def __init__(
        self,
        p_perp_par_t: float = 0.30,
        p_perp_par_c: float = 0.25,
        p_perp_psi_t: float = 0.20,
        p_perp_psi_c: float = 0.25,
        n_theta: int = 181,
    ) -> None:
        self.p_perp_par_t = p_perp_par_t
        self.p_perp_par_c = p_perp_par_c
        self.p_perp_psi_t = p_perp_psi_t
        self.p_perp_psi_c = p_perp_psi_c
        self.n_theta = n_theta

    # ------------------------------------------------------------------
    # Fibre failure
    # ------------------------------------------------------------------

    @staticmethod
    def _fibre_failure(
        s1: float, Xt: float, Xc: float
    ) -> tuple[float, str]:
        """Evaluate simplified fibre failure.

        Parameters
        ----------
        s1 : float
            Fibre-direction stress sigma_11.
        Xt, Xc : float
            Tensile and compressive fibre strengths.

        Returns
        -------
        fi : float
            Fibre failure index.
        mode : str
            ``"fiber_tension"`` or ``"fiber_compression"``.
        """
        if s1 >= 0:
            return s1 / Xt, "fiber_tension"
        else:
            return -s1 / Xc, "fiber_compression"

    # ------------------------------------------------------------------
    # Inter-fibre failure (action-plane search)
    # ------------------------------------------------------------------

    def _inter_fibre_failure(
        self,
        stress_local: np.ndarray,
        material: OrthotropicMaterial,
    ) -> tuple[float, str, float]:
        """Search over fracture plane angles for maximum IFF index.

        Parameters
        ----------
        stress_local : np.ndarray
            Shape ``(6,)`` stress vector.
        material : OrthotropicMaterial
            Material strength properties.

        Returns
        -------
        fi_max : float
            Maximum inter-fibre failure index over all angles.
        mode : str
            IFF sub-mode at the critical angle (``"iff_mode_a"``,
            ``"iff_mode_b"``, or ``"iff_mode_c"``).
        theta_crit : float
            Critical fracture plane angle in radians.
        """
        s1, s2, s3, t23, t13, t12 = stress_local

        S12 = material.S12
        S23 = material.S23
        Yc = material.Yc

        p_ppt = self.p_perp_par_t
        p_ppc = self.p_perp_par_c
        p_pst = self.p_perp_psi_t
        p_psc = self.p_perp_psi_c

        thetas = np.linspace(-np.pi / 2, np.pi / 2, self.n_theta)
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)

        # Stresses on the fracture plane (vectorised over theta)
        sigma_n = s2 * cos_t**2 + s3 * sin_t**2 + 2.0 * t23 * sin_t * cos_t
        tau_nt = (s3 - s2) * sin_t * cos_t + t23 * (cos_t**2 - sin_t**2)
        tau_n1 = t12 * cos_t + t13 * sin_t

        fi_arr = np.zeros_like(thetas)
        mode_arr = np.empty(len(thetas), dtype=object)

        for i in range(len(thetas)):
            sn = sigma_n[i]
            tnt = tau_nt[i]
            tn1 = tau_n1[i]

            if sn >= 0:
                # Mode A (tension on fracture plane)
                denom_nt = S23 + p_pst * sn
                denom_n1 = S12 + p_ppt * sn
                # Guard against denominator approaching zero
                if denom_nt <= 0 or denom_n1 <= 0:
                    fi_arr[i] = float("inf")
                else:
                    fi_arr[i] = (
                        np.sqrt((tnt / denom_nt) ** 2 + (tn1 / denom_n1) ** 2)
                        + p_pst * sn / S23
                    )
                mode_arr[i] = "iff_mode_a"
            else:
                # sigma_n < 0: distinguish Mode B vs Mode C
                # Threshold: ratio |tau_nt / sigma_n|
                ratio = abs(tnt / sn) if abs(sn) > 1e-12 else float("inf")
                threshold = S23 / abs(sn) if abs(sn) > 1e-12 else 0.0

                if ratio >= threshold:
                    # Mode B
                    inner = tnt**2 + (tn1 + p_ppc * sn) ** 2
                    fi_arr[i] = np.sqrt(inner) / S12 + p_psc * sn / S23
                    mode_arr[i] = "iff_mode_b"
                else:
                    # Mode C
                    denom_c = 2.0 * (1.0 + p_psc) * S23
                    bracket = (tnt / denom_c) ** 2 + (tn1 / S12) ** 2
                    fi_arr[i] = bracket * (-Yc / sn)
                    mode_arr[i] = "iff_mode_c"

        idx_max = int(np.argmax(fi_arr))
        return float(fi_arr[idx_max]), str(mode_arr[idx_max]), float(thetas[idx_max])

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        stress_local: np.ndarray,
        material: OrthotropicMaterial,
        context=None,
    ) -> FailureResult:
        """Evaluate the Puck criterion at a single material point.

        The overall failure index is the maximum of fibre failure (FF) and
        inter-fibre failure (IFF).

        Parameters
        ----------
        stress_local : np.ndarray
            Shape ``(6,)`` stress vector in local material coordinates.
        material : OrthotropicMaterial
            Material with strength allowables and inclination parameters.

        Returns
        -------
        FailureResult
            Contains the governing failure index, mode, reserve factor,
            and criterion name.
        """
        stress_local = np.asarray(stress_local, dtype=np.float64)

        fi_ff, mode_ff = self._fibre_failure(
            stress_local[0], material.Xt, material.Xc
        )
        fi_iff, mode_iff, _ = self._inter_fibre_failure(stress_local, material)

        if fi_ff >= fi_iff:
            fi, mode = fi_ff, mode_ff
        else:
            fi, mode = fi_iff, mode_iff

        rf = 1.0 / fi if fi > 0 else float("inf")

        return FailureResult(
            index=fi,
            mode=mode,
            reserve_factor=rf,
            criterion_name=self.name,
        )
