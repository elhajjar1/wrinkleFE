"""Tsai-Wu 3D tensor polynomial failure criterion for orthotropic composites.

The Tsai-Wu criterion is a single smooth failure surface in 6D stress space
defined by a tensor polynomial with linear and quadratic terms.  It accounts
for the full interaction between all stress components and naturally
captures the difference between tensile and compressive strengths.

Failure Surface
---------------
The failure surface is defined by:

.. math::

    f = F_i\\,\\sigma_i + F_{ij}\\,\\sigma_i\\,\\sigma_j = 1

Expanded in Voigt notation (i = 1..6):

.. math::

    f = F_1\\,\\sigma_{11} + F_2\\,\\sigma_{22} + F_3\\,\\sigma_{33}
      + F_{11}\\,\\sigma_{11}^2 + F_{22}\\,\\sigma_{22}^2 + F_{33}\\,\\sigma_{33}^2
      + 2\\,F_{12}\\,\\sigma_{11}\\,\\sigma_{22}
      + 2\\,F_{13}\\,\\sigma_{11}\\,\\sigma_{33}
      + 2\\,F_{23}\\,\\sigma_{22}\\,\\sigma_{33}
      + F_{44}\\,\\tau_{23}^2 + F_{55}\\,\\tau_{13}^2 + F_{66}\\,\\tau_{12}^2

where the strength tensors are:

**Linear terms** (capture tension/compression asymmetry):

.. math::

    F_1 = \\frac{1}{X_t} - \\frac{1}{X_c}, \\quad
    F_2 = \\frac{1}{Y_t} - \\frac{1}{Y_c}, \\quad
    F_3 = \\frac{1}{Z_t} - \\frac{1}{Z_c}

**Quadratic diagonal terms:**

.. math::

    F_{11} = \\frac{1}{X_t\\,X_c}, \\quad
    F_{22} = \\frac{1}{Y_t\\,Y_c}, \\quad
    F_{33} = \\frac{1}{Z_t\\,Z_c}

    F_{44} = \\frac{1}{S_{23}^2}, \\quad
    F_{55} = \\frac{1}{S_{13}^2}, \\quad
    F_{66} = \\frac{1}{S_{12}^2}

**Interaction terms** (parameterised by f_{12}^*):

.. math::

    F_{12} = f_{12}^* \\sqrt{F_{11}\\,F_{22}}, \\quad
    F_{13} = f_{12}^* \\sqrt{F_{11}\\,F_{33}}, \\quad
    F_{23} = f_{12}^* \\sqrt{F_{22}\\,F_{33}}

The dimensionless interaction coefficient f_{12}^* is typically in the
range [-0.5, 0] and must satisfy the stability condition
f_{12}^{*2} < 1 to ensure a closed failure surface.

Failure occurs when f >= 1.0.

Note
----
The Tsai-Wu criterion does not naturally decompose into physical failure
modes.  The mode identification provided here is approximate, based on the
relative magnitude of fibre, matrix, and shear contributions to the
polynomial.

References
----------
- Tsai, S.W. & Wu, E.M. (1971). A general theory of strength for
  anisotropic materials. *J. Composite Materials*, 5(1), 58-80.
- Tsai, S.W. (1992). *Theory of Composites Design*. Think Composites.
"""

from __future__ import annotations

import numpy as np

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.failure.base import FailureCriterion, FailureResult


class TsaiWuCriterion(FailureCriterion):
    """Tsai-Wu 3D tensor polynomial failure criterion.

    The criterion evaluates a single scalar failure index that captures
    the full interaction between all stress components.  It is the most
    general quadratic criterion for orthotropic materials.

    Parameters
    ----------
    f12_star : float, optional
        Dimensionless interaction coefficient, typically in [-0.5, 0].
        Default is -0.5.  The same value is used for all normal-stress
        interaction pairs (F12, F13, F23).  Must satisfy
        ``f12_star**2 < 1`` for the failure surface to be closed
        (ellipsoidal).

    Examples
    --------
    >>> from wrinklefe.core.material import OrthotropicMaterial
    >>> mat = OrthotropicMaterial()
    >>> criterion = TsaiWuCriterion(f12_star=-0.5)
    >>> import numpy as np
    >>> # Pure fibre compression at Xc should give FI = 1.0
    >>> stress = np.array([-1500.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    >>> result = criterion.evaluate(stress, mat)
    >>> abs(result.index - 1.0) < 1e-10
    True
    """

    name: str = "tsai_wu"

    def __init__(self, f12_star: float = -0.5) -> None:
        """Initialise with the interaction coefficient.

        Parameters
        ----------
        f12_star : float
            Interaction coefficient, typically in the range [-0.5, 0].
            ``F_{12} = f12_star * sqrt(F_{11} * F_{22})``, and similarly
            for F13 and F23.
        """
        if f12_star ** 2 >= 1.0:
            raise ValueError(
                f"f12_star**2 must be < 1 for a closed failure surface, "
                f"got f12_star={f12_star} (f12_star**2={f12_star**2})"
            )
        self.f12_star = f12_star

    def evaluate(
        self,
        stress_local: np.ndarray,
        material: OrthotropicMaterial,
        context=None,
    ) -> FailureResult:
        """Evaluate the Tsai-Wu 3D criterion at a single material point.

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
            Failure index (f >= 1.0 means failure), approximate dominant
            mode, reserve factor, and criterion name.
        """
        s = stress_local

        # --- Linear strength tensor components ---
        F1 = 1.0 / material.Xt - 1.0 / material.Xc
        F2 = 1.0 / material.Yt - 1.0 / material.Yc
        F3 = 1.0 / material.Zt - 1.0 / material.Zc

        # --- Quadratic diagonal components ---
        F11 = 1.0 / (material.Xt * material.Xc)
        F22 = 1.0 / (material.Yt * material.Yc)
        F33 = 1.0 / (material.Zt * material.Zc)
        F44 = 1.0 / material.S23 ** 2
        F55 = 1.0 / material.S13 ** 2
        F66 = 1.0 / material.S12 ** 2

        # --- Interaction terms ---
        F12 = self.f12_star * np.sqrt(F11 * F22)
        F13 = self.f12_star * np.sqrt(F11 * F33)
        F23 = self.f12_star * np.sqrt(F22 * F33)

        # --- Tsai-Wu polynomial ---
        # f = F_i * s_i + F_ij * s_i * s_j
        linear = F1 * s[0] + F2 * s[1] + F3 * s[2]

        quadratic = (
            F11 * s[0] ** 2
            + F22 * s[1] ** 2
            + F33 * s[2] ** 2
            + 2 * F12 * s[0] * s[1]
            + 2 * F13 * s[0] * s[2]
            + 2 * F23 * s[1] * s[2]
            + F44 * s[3] ** 2
            + F55 * s[4] ** 2
            + F66 * s[5] ** 2
        )

        fi = linear + quadratic

        # --- Approximate mode identification ---
        # The Tsai-Wu polynomial does not decompose into distinct modes.
        # We estimate the dominant contributor from the relative magnitude
        # of fibre-direction, matrix-direction, and shear terms.
        fiber_contrib = abs(F1 * s[0]) + F11 * s[0] ** 2
        matrix_contrib = abs(F2 * s[1]) + F22 * s[1] ** 2
        shear_contrib = F66 * s[5] ** 2

        if fiber_contrib >= matrix_contrib and fiber_contrib >= shear_contrib:
            mode = "fiber_tension" if s[0] >= 0 else "fiber_compression"
        elif matrix_contrib >= shear_contrib:
            mode = "matrix_tension" if s[1] >= 0 else "matrix_compression"
        else:
            mode = "shear"

        return FailureResult(
            index=fi,
            mode=mode,
            reserve_factor=1.0 / fi if fi > 0 else float("inf"),
            criterion_name=self.name,
        )
