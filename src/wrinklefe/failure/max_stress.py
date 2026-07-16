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
    >>> float(result.index)  # 800 / 1200 (Xc = 1200 MPa)
    0.6666666666666666
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

    # ------------------------------------------------------------------
    # Vectorised field evaluation
    # ------------------------------------------------------------------

    _MODE_LABELS_FIELD = np.array(
        [
            "fiber_tension",         # idx 0: sigma_11 >= 0
            "fiber_compression",     # idx 1: sigma_11 <  0
            "matrix_tension",        # idx 2: sigma_22 >= 0
            "matrix_compression",    # idx 3: sigma_22 <  0
            "through_thickness_tension",      # idx 4: sigma_33 >= 0
            "through_thickness_compression",  # idx 5: sigma_33 <  0
            "shear_23",                       # idx 6
            "shear_13",                       # idx 7
            "shear_12",                       # idx 8
        ],
        dtype="U32",
    )

    def evaluate_field(
        self,
        stress_field: np.ndarray,
        material: OrthotropicMaterial,
        contexts=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised Max-Stress evaluation across an array of stress states.

        Computes the same failure index and dominant mode as :meth:`evaluate`
        but for ``N`` stress vectors at once using NumPy broadcasting.

        Parameters
        ----------
        stress_field : np.ndarray
            Shape ``(N, 6)`` array of local stress vectors.
        material : OrthotropicMaterial
            Material with strength allowables.  All N points share the
            same material (callers group by ply id beforehand).
        contexts : list, optional
            Ignored — Max-Stress has no context dependence.

        Returns
        -------
        indices : np.ndarray
            Shape ``(N,)`` failure indices.
        modes : np.ndarray
            Shape ``(N,)`` dominant-mode string labels.
        reserve_factors : np.ndarray
            Shape ``(N,)`` reserve factors (``1/FI``; ``inf`` where FI == 0).
        """
        s = np.asarray(stress_field, dtype=np.float64)
        if s.ndim != 2 or s.shape[1] != 6:
            raise ValueError(
                f"stress_field must have shape (N, 6), got {s.shape}"
            )

        s11, s22, s33 = s[:, 0], s[:, 1], s[:, 2]
        t23, t13, t12 = s[:, 3], s[:, 4], s[:, 5]

        # Stack the nine non-negative component ratios.  For sign-dependent
        # normals, the inactive branch is left at zero so it never wins the
        # argmax.
        ratios = np.zeros((9, s.shape[0]), dtype=np.float64)
        ratios[0] = np.where(s11 >= 0, s11 / material.Xt, 0.0)
        ratios[1] = np.where(s11 < 0, -s11 / material.Xc, 0.0)
        ratios[2] = np.where(s22 >= 0, s22 / material.Yt, 0.0)
        ratios[3] = np.where(s22 < 0, -s22 / material.Yc, 0.0)
        ratios[4] = np.where(s33 >= 0, s33 / material.Zt, 0.0)
        ratios[5] = np.where(s33 < 0, -s33 / material.Zc, 0.0)
        ratios[6] = np.abs(t23) / material.S23
        ratios[7] = np.abs(t13) / material.S13
        ratios[8] = np.abs(t12) / material.S12

        idx = np.argmax(ratios, axis=0)
        fi = ratios[idx, np.arange(s.shape[0])]
        modes = self._MODE_LABELS_FIELD[idx]

        with np.errstate(divide="ignore"):
            rf = np.where(fi > 0, 1.0 / np.where(fi > 0, fi, 1.0), np.inf)

        return fi, modes, rf
