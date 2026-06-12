"""Tsai-Hill failure criterion extended to 3-D orthotropic composites.

The Tsai-Hill criterion is a quadratic interaction criterion that accounts
for coupling between normal stress components.  It generalises the Hill
yield criterion to composites with distinct tensile and compressive
strengths by selecting the appropriate allowable based on the sign of each
normal stress.

Mathematical Formulation (3-D)
------------------------------
The failure index is::

    FI = (s1/X)^2 - s1*s2/X^2 + (s2/Y)^2 + (s3/Z)^2
         - s2*s3/Y^2 - s1*s3/X^2
         + (t12/S12)^2 + (t13/S13)^2 + (t23/S23)^2

where:
- ``X = Xt`` if ``s1 >= 0``, else ``X = Xc``
- ``Y = Yt`` if ``s2 >= 0``, else ``Y = Yc``
- ``Z = Zt`` if ``s3 >= 0``, else ``Z = Zc``

FI >= 1.0 indicates failure.

The dominant mode is identified by the term contributing the largest
magnitude to the failure index.

References
----------
- Hill, R. (1950). *The Mathematical Theory of Plasticity*.
- Tsai, S. W. (1968). NASA CR-71.
"""

from __future__ import annotations

import numpy as np

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.failure.base import FailureCriterion, FailureResult


class TsaiHillCriterion(FailureCriterion):
    """Tsai-Hill failure criterion for 3-D orthotropic composites.

    Uses sign-dependent strengths (tension vs compression) for each
    normal stress component.

    Attributes
    ----------
    name : str
        ``"tsai_hill"``
    """

    name = "tsai_hill"

    def evaluate(
        self,
        stress_local: np.ndarray,
        material: OrthotropicMaterial,
        context=None,
    ) -> FailureResult:
        """Evaluate 3-D Tsai-Hill criterion at a single material point.

        Parameters
        ----------
        stress_local : np.ndarray
            Shape ``(6,)`` stress vector::

                [sigma_11, sigma_22, sigma_33, tau_23, tau_13, tau_12]

        material : OrthotropicMaterial
            Material with strength allowables.

        Returns
        -------
        FailureResult
            Failure index, dominant mode, reserve factor, and criterion name.
        """
        stress_local = np.asarray(stress_local, dtype=np.float64)
        s1, s2, s3, t23, t13, t12 = stress_local

        # Sign-dependent strengths
        X = material.Xt if s1 >= 0 else material.Xc
        Y = material.Yt if s2 >= 0 else material.Yc
        Z = material.Zt if s3 >= 0 else material.Zc

        # Individual terms for mode identification
        # Positive terms (always contribute to failure)
        term_11 = (s1 / X) ** 2
        term_22 = (s2 / Y) ** 2
        term_33 = (s3 / Z) ** 2
        term_s12 = (t12 / material.S12) ** 2
        term_s13 = (t13 / material.S13) ** 2
        term_s23 = (t23 / material.S23) ** 2

        # Interaction terms (can be negative, reducing FI)
        term_12 = -s1 * s2 / X**2
        term_23 = -s2 * s3 / Y**2
        term_13 = -s1 * s3 / X**2

        fi = (term_11 + term_22 + term_33
              + term_12 + term_23 + term_13
              + term_s12 + term_s13 + term_s23)

        # Identify dominant mode from the positive (non-interaction) terms
        mode_terms = {
            "fiber_tension" if s1 >= 0 else "fiber_compression": term_11,
            "matrix_transverse_tension" if s2 >= 0 else "matrix_transverse_compression": term_22,
            "matrix_thickness_tension" if s3 >= 0 else "matrix_thickness_compression": term_33,
            "shear_12": term_s12,
            "shear_13": term_s13,
            "shear_23": term_s23,
        }
        mode = max(mode_terms, key=lambda m: mode_terms[m])

        # ``fi`` is the quadratic Tsai-Hill invariant (FI=1 at failure, paper
        # convention).  Every contributing term is purely quadratic in stress,
        # so under proportional load scaling R the index scales as R^2 * FI(1).
        # The linear-in-load strength ratio is therefore 1/sqrt(FI), not 1/FI.
        rf = 1.0 / np.sqrt(fi) if fi > 0 else float("inf")

        return FailureResult(
            index=float(fi),
            mode=mode,
            reserve_factor=rf,
            criterion_name=self.name,
        )

    # ------------------------------------------------------------------
    # Vectorised field evaluation
    # ------------------------------------------------------------------

    def evaluate_field(
        self,
        stress_field: np.ndarray,
        material: OrthotropicMaterial,
        contexts=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised Tsai-Hill evaluation across an array of stress states.

        Same maths as :meth:`evaluate` on ``N`` points at once via NumPy
        broadcasting; sign-dependent strengths are selected with
        :func:`numpy.where`.

        Parameters
        ----------
        stress_field : np.ndarray
            Shape ``(N, 6)`` stress array.
        material : OrthotropicMaterial
            Material with strength allowables (shared across all N).
        contexts : list, optional
            Ignored — Tsai-Hill has no context dependence.

        Returns
        -------
        indices, modes, reserve_factors : np.ndarray, np.ndarray, np.ndarray
            Each of shape ``(N,)``.  ``reserve_factors = 1/sqrt(FI)`` for the
            purely quadratic Tsai-Hill index.
        """
        s = np.asarray(stress_field, dtype=np.float64)
        if s.ndim != 2 or s.shape[1] != 6:
            raise ValueError(
                f"stress_field must have shape (N, 6), got {s.shape}"
            )

        s1 = s[:, 0]
        s2 = s[:, 1]
        s3 = s[:, 2]
        t23 = s[:, 3]
        t13 = s[:, 4]
        t12 = s[:, 5]

        X = np.where(s1 >= 0, material.Xt, material.Xc)
        Y = np.where(s2 >= 0, material.Yt, material.Yc)
        Z = np.where(s3 >= 0, material.Zt, material.Zc)

        term_11 = (s1 / X) ** 2
        term_22 = (s2 / Y) ** 2
        term_33 = (s3 / Z) ** 2
        term_s12 = (t12 / material.S12) ** 2
        term_s13 = (t13 / material.S13) ** 2
        term_s23 = (t23 / material.S23) ** 2

        term_12 = -s1 * s2 / X ** 2
        term_23 = -s2 * s3 / Y ** 2
        term_13 = -s1 * s3 / X ** 2

        fi = (term_11 + term_22 + term_33
              + term_12 + term_23 + term_13
              + term_s12 + term_s13 + term_s23)

        # --- Dominant mode by largest positive (non-interaction) term ---
        positive_terms = np.vstack([
            term_11, term_22, term_33, term_s12, term_s13, term_s23
        ])  # (6, N)
        idx = np.argmax(positive_terms, axis=0)

        # Build mode array.  Mode strings for normal terms depend on stress
        # sign; shear modes are unconditional.
        modes = np.empty(s.shape[0], dtype="U32")
        is_11 = idx == 0
        modes[is_11 & (s1 >= 0)] = "fiber_tension"
        modes[is_11 & (s1 < 0)] = "fiber_compression"
        is_22 = idx == 1
        modes[is_22 & (s2 >= 0)] = "matrix_transverse_tension"
        modes[is_22 & (s2 < 0)] = "matrix_transverse_compression"
        is_33 = idx == 2
        modes[is_33 & (s3 >= 0)] = "matrix_thickness_tension"
        modes[is_33 & (s3 < 0)] = "matrix_thickness_compression"
        modes[idx == 3] = "shear_12"
        modes[idx == 4] = "shear_13"
        modes[idx == 5] = "shear_23"

        # --- Reserve factor = 1/sqrt(FI) ---
        rf = np.where(fi > 0.0, 1.0 / np.sqrt(np.where(fi > 0.0, fi, 1.0)), np.inf)

        return fi.astype(np.float64), modes, rf
