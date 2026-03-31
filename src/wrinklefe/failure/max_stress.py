"""Maximum Stress failure criterion for orthotropic composite plies.

The Maximum Stress criterion is the simplest interactive failure criterion.
Each stress component is independently compared to the corresponding
material allowable, and the component with the highest ratio determines
the failure index and the dominant failure mode.

Failure Indices
---------------
For each stress component the failure index is:

.. math::

    \\text{FI}_{11} =
    \\begin{cases}
        \\sigma_{11} / X_t & \\text{if } \\sigma_{11} \\geq 0 \\\\
        |\\sigma_{11}| / X_c & \\text{if } \\sigma_{11} < 0
    \\end{cases}

    \\text{FI}_{22} =
    \\begin{cases}
        \\sigma_{22} / Y_t & \\text{if } \\sigma_{22} \\geq 0 \\\\
        |\\sigma_{22}| / Y_c & \\text{if } \\sigma_{22} < 0
    \\end{cases}

    \\text{FI}_{33} =
    \\begin{cases}
        \\sigma_{33} / Z_t & \\text{if } \\sigma_{33} \\geq 0 \\\\
        |\\sigma_{33}| / Z_c & \\text{if } \\sigma_{33} < 0
    \\end{cases}

    \\text{FI}_{23} = |\\tau_{23}| / S_{23}

    \\text{FI}_{13} = |\\tau_{13}| / S_{13}

    \\text{FI}_{12} = |\\tau_{12}| / S_{12}

The overall failure index is:

.. math::

    \\text{FI} = \\max(\\text{FI}_{11}, \\text{FI}_{22}, \\text{FI}_{33},
                       \\text{FI}_{23}, \\text{FI}_{13}, \\text{FI}_{12})

Failure occurs when FI >= 1.0.

References
----------
- Jones, R.M. (1999). *Mechanics of Composite Materials*, 2nd ed.
  Taylor & Francis, Chapter 2.
"""

from __future__ import annotations

import numpy as np

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.failure.base import FailureCriterion, FailureResult


class MaxStressCriterion(FailureCriterion):
    """Maximum Stress failure criterion.

    Compares each local stress component independently against the
    corresponding material strength allowable.  The component with the
    highest stress-to-strength ratio determines the failure index and
    the dominant failure mode.

    This criterion does **not** account for stress interaction effects
    (coupling between fibre and matrix stresses), but it clearly identifies
    which mode is most critical.

    Examples
    --------
    >>> from wrinklefe.core.material import OrthotropicMaterial
    >>> mat = OrthotropicMaterial()
    >>> criterion = MaxStressCriterion()
    >>> import numpy as np
    >>> stress = np.array([-800.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    >>> result = criterion.evaluate(stress, mat)
    >>> result.mode
    'fiber_compression'
    >>> result.index  # 800 / 1500
    0.533...
    """

    name: str = "max_stress"

    def evaluate(
        self,
        stress_local: np.ndarray,
        material: OrthotropicMaterial,
        context=None,
    ) -> FailureResult:
        """Evaluate the Maximum Stress criterion at a single material point.

        Parameters
        ----------
        stress_local : np.ndarray
            Shape ``(6,)`` stress vector in local coordinates
            ``[sigma_11, sigma_22, sigma_33, tau_23, tau_13, tau_12]``.
        material : OrthotropicMaterial
            Material with strength allowables Xt, Xc, Yt, Yc, Zt, Zc,
            S12, S13, S23.

        Returns
        -------
        FailureResult
            Failure index, dominant mode, reserve factor, and criterion name.
        """
        s = stress_local

        # --- Fibre direction (1-direction) ---
        if s[0] >= 0:
            fi_11 = s[0] / material.Xt
            mode_11 = "fiber_tension"
        else:
            fi_11 = abs(s[0]) / material.Xc
            mode_11 = "fiber_compression"

        # --- In-plane transverse (2-direction) ---
        if s[1] >= 0:
            fi_22 = s[1] / material.Yt
            mode_22 = "matrix_tension"
        else:
            fi_22 = abs(s[1]) / material.Yc
            mode_22 = "matrix_compression"

        # --- Through-thickness (3-direction) ---
        if s[2] >= 0:
            fi_33 = s[2] / material.Zt
            mode_33 = "through_thickness_tension"
        else:
            fi_33 = abs(s[2]) / material.Zc
            mode_33 = "through_thickness_compression"

        # --- Shear components ---
        fi_23 = abs(s[3]) / material.S23
        fi_13 = abs(s[4]) / material.S13
        fi_12 = abs(s[5]) / material.S12

        # --- Determine dominant mode ---
        all_fi = [fi_11, fi_22, fi_33, fi_23, fi_13, fi_12]
        all_modes = [
            mode_11, mode_22, mode_33,
            "shear_23", "shear_13", "shear_12",
        ]

        idx = int(np.argmax(all_fi))
        fi_max = all_fi[idx]
        mode = all_modes[idx]

        return FailureResult(
            index=fi_max,
            mode=mode,
            reserve_factor=1.0 / fi_max if fi_max > 0 else float("inf"),
            criterion_name=self.name,
        )
