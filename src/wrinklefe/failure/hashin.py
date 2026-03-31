"""Hashin 3D failure criterion for orthotropic composite plies.

Implements the four-mode Hashin criterion extended to 3D stress states.
Each mode has a physically motivated quadratic failure surface that
accounts for stress interactions within that failure mechanism.

Failure Modes
-------------

**1. Fibre Tension** (sigma_11 >= 0):

.. math::

    \\text{FI}_{ft}^2 = \\left(\\frac{\\sigma_{11}}{X_t}\\right)^2
                       + \\frac{\\tau_{12}^2 + \\tau_{13}^2}{S_{12}^2}

**2. Fibre Compression** (sigma_11 < 0):

.. math::

    \\text{FI}_{fc} = \\frac{|\\sigma_{11}|}{X_c}

**3. Matrix Tension** (sigma_22 + sigma_33 >= 0):

.. math::

    \\text{FI}_{mt}^2 = \\left(\\frac{\\sigma_{22} + \\sigma_{33}}{Y_t}\\right)^2
                       + \\frac{\\tau_{23}^2 - \\sigma_{22}\\,\\sigma_{33}}{S_{23}^2}
                       + \\frac{\\tau_{12}^2 + \\tau_{13}^2}{S_{12}^2}

**4. Matrix Compression** (sigma_22 + sigma_33 < 0):

.. math::

    \\text{FI}_{mc}^2 = \\left[\\left(\\frac{Y_c}{2\\,S_{23}}\\right)^2 - 1\\right]
                        \\frac{\\sigma_{22} + \\sigma_{33}}{Y_c}
                       + \\frac{(\\sigma_{22} + \\sigma_{33})^2}{4\\,S_{23}^2}
                       + \\frac{\\tau_{23}^2 - \\sigma_{22}\\,\\sigma_{33}}{S_{23}^2}
                       + \\frac{\\tau_{12}^2 + \\tau_{13}^2}{S_{12}^2}

The overall failure index is the maximum across all active modes.
Failure occurs when FI >= 1.0.

References
----------
- Hashin, Z. (1980). Failure criteria for unidirectional fiber composites.
  *Journal of Applied Mechanics*, 47(2), 329-334.
- Hashin, Z. & Rotem, A. (1973). A fatigue failure criterion for fiber
  reinforced materials. *J. Composite Materials*, 7(4), 448-464.
"""

from __future__ import annotations

import numpy as np

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.failure.base import FailureCriterion, FailureResult


class HashinCriterion(FailureCriterion):
    """Hashin 3D four-mode failure criterion for unidirectional composites.

    The criterion distinguishes fibre and matrix failure in both tension
    and compression, providing physically meaningful failure mode
    identification.  The 3D formulation accounts for through-thickness
    stresses (sigma_33, tau_23, tau_13), making it suitable for thick
    laminates and wrinkle-induced interlaminar stress states.

    Examples
    --------
    >>> from wrinklefe.core.material import OrthotropicMaterial
    >>> mat = OrthotropicMaterial()
    >>> criterion = HashinCriterion()
    >>> import numpy as np
    >>> # Pure fibre compression at 50% of Xc
    >>> stress = np.array([-750.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    >>> result = criterion.evaluate(stress, mat)
    >>> result.mode
    'fiber_compression'
    >>> abs(result.index - 0.5) < 1e-10
    True
    """

    name: str = "hashin"

    def evaluate(
        self,
        stress_local: np.ndarray,
        material: OrthotropicMaterial,
        context=None,
    ) -> FailureResult:
        """Evaluate the Hashin 3D criterion at a single material point.

        Parameters
        ----------
        stress_local : np.ndarray
            Shape ``(6,)`` stress vector in local coordinates
            ``[sigma_11, sigma_22, sigma_33, tau_23, tau_13, tau_12]``.
        material : OrthotropicMaterial
            Material with strength allowables Xt, Xc, Yt, Yc, S12, S23.

        Returns
        -------
        FailureResult
            Failure index (linear scale), dominant mode, reserve factor,
            and criterion name.
        """
        s = stress_local  # [sigma_11, sigma_22, sigma_33, tau_23, tau_13, tau_12]

        # ---------------------------------------------------------------
        # 1. Fibre Tension  (sigma_11 >= 0)
        #    FI_ft^2 = (sigma_11 / Xt)^2 + (tau_12^2 + tau_13^2) / S12^2
        # ---------------------------------------------------------------
        if s[0] >= 0:
            fi_ft = np.sqrt(
                (s[0] / material.Xt) ** 2
                + (s[5] ** 2 + s[4] ** 2) / material.S12 ** 2
            )
        else:
            fi_ft = 0.0

        # ---------------------------------------------------------------
        # 2. Fibre Compression  (sigma_11 < 0)
        #    FI_fc = |sigma_11| / Xc
        # ---------------------------------------------------------------
        if s[0] < 0:
            fi_fc = abs(s[0]) / material.Xc
        else:
            fi_fc = 0.0

        # ---------------------------------------------------------------
        # 3. Matrix Tension  (sigma_22 + sigma_33 >= 0)
        #    FI_mt^2 = ((s22+s33)/Yt)^2
        #            + (tau_23^2 - s22*s33) / S23^2
        #            + (tau_12^2 + tau_13^2) / S12^2
        # ---------------------------------------------------------------
        sig_t = s[1] + s[2]  # sigma_22 + sigma_33
        if sig_t >= 0:
            fi_mt_sq = (
                (sig_t / material.Yt) ** 2
                + (s[3] ** 2 - s[1] * s[2]) / material.S23 ** 2
                + (s[5] ** 2 + s[4] ** 2) / material.S12 ** 2
            )
            fi_mt = np.sqrt(max(fi_mt_sq, 0.0))
        else:
            fi_mt = 0.0

        # ---------------------------------------------------------------
        # 4. Matrix Compression  (sigma_22 + sigma_33 < 0)
        #    FI_mc^2 = [(Yc/(2*S23))^2 - 1] * (s22+s33)/Yc
        #            + (s22+s33)^2 / (4*S23^2)
        #            + (tau_23^2 - s22*s33) / S23^2
        #            + (tau_12^2 + tau_13^2) / S12^2
        # ---------------------------------------------------------------
        if sig_t < 0:
            fi_mc_sq = (
                ((material.Yc / (2 * material.S23)) ** 2 - 1)
                * sig_t / material.Yc
                + sig_t ** 2 / (4 * material.S23 ** 2)
                + (s[3] ** 2 - s[1] * s[2]) / material.S23 ** 2
                + (s[5] ** 2 + s[4] ** 2) / material.S12 ** 2
            )
            fi_mc = np.sqrt(max(fi_mc_sq, 0.0))
        else:
            fi_mc = 0.0

        # ---------------------------------------------------------------
        # Determine dominant failure mode
        # ---------------------------------------------------------------
        modes = {
            "fiber_tension": fi_ft,
            "fiber_compression": fi_fc,
            "matrix_tension": fi_mt,
            "matrix_compression": fi_mc,
        }
        dominant = max(modes, key=modes.get)
        fi_max = modes[dominant]

        return FailureResult(
            index=fi_max,
            mode=dominant,
            reserve_factor=1.0 / fi_max if fi_max > 0 else float("inf"),
            criterion_name=self.name,
        )
