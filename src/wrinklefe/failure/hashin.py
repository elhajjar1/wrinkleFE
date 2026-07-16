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
    >>> # Pure fibre compression at 62.5% of Xc (Xc = 1200 MPa)
    >>> stress = np.array([-750.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    >>> result = criterion.evaluate(stress, mat)
    >>> result.mode
    'fiber_compression'
    >>> bool(abs(result.index - 0.625) < 1e-10)
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
        #    FI_mc^2 = [(Yc/(2*S23))^2 - 1] * (s22+s33)/Yc        (linear in stress)
        #            + (s22+s33)^2 / (4*S23^2)                    (quadratic)
        #            + (tau_23^2 - s22*s33) / S23^2               (quadratic)
        #            + (tau_12^2 + tau_13^2) / S12^2              (quadratic)
        #
        # Note: the first term is *linear* in the stress, while the others
        # are *quadratic*.  Under proportional load scaling by R, the
        # polynomial FI_mc^2(R) = A*R^2 + B*R mixes orders, so
        # RF = 1/FI_mc would underestimate the true strength ratio.
        # Instead, we solve A*R_f^2 + B*R_f = 1 for the positive root.
        # ---------------------------------------------------------------
        if sig_t < 0:
            # Quadratic coefficient A: pure stress^2 terms
            A_mc = (
                sig_t ** 2 / (4 * material.S23 ** 2)
                + (s[3] ** 2 - s[1] * s[2]) / material.S23 ** 2
                + (s[5] ** 2 + s[4] ** 2) / material.S12 ** 2
            )
            # Linear coefficient B: linear-in-stress term
            B_mc = ((material.Yc / (2 * material.S23)) ** 2 - 1) * sig_t / material.Yc

            fi_mc_sq = A_mc + B_mc
            fi_mc = np.sqrt(max(fi_mc_sq, 0.0))

            # Closed-form reserve factor: positive root of A*R^2 + B*R - 1 = 0
            rf_mc = _quadratic_reserve_factor(A_mc, B_mc)
        else:
            fi_mc = 0.0
            rf_mc = float("inf")

        # ---------------------------------------------------------------
        # Determine dominant failure mode (by FI value, as before).
        # Per-mode reserve factors:
        #   - fiber_tension, fiber_compression, matrix_tension are pure
        #     quadratics in the stress, so RF = 1/sqrt(FI^2) = 1/FI.
        #   - matrix_compression has a mixed linear+quadratic FI^2, so RF
        #     comes from the quadratic root above.
        # ---------------------------------------------------------------
        modes = {
            "fiber_tension": fi_ft,
            "fiber_compression": fi_fc,
            "matrix_tension": fi_mt,
            "matrix_compression": fi_mc,
        }
        dominant = max(modes, key=lambda m: modes[m])
        fi_max = modes[dominant]

        if fi_max <= 0:
            reserve_factor = float("inf")
        elif dominant == "matrix_compression":
            reserve_factor = rf_mc
        else:
            reserve_factor = 1.0 / fi_max

        return FailureResult(
            index=fi_max,
            mode=dominant,
            reserve_factor=reserve_factor,
            criterion_name=self.name,
        )

    # ------------------------------------------------------------------
    # Vectorised field evaluation
    # ------------------------------------------------------------------

    _HASHIN_MODE_LABELS = np.array(
        ["fiber_tension", "fiber_compression",
         "matrix_tension", "matrix_compression"],
        dtype="U32",
    )

    def evaluate_field(
        self,
        stress_field: np.ndarray,
        material: OrthotropicMaterial,
        contexts=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised Hashin 3D evaluation across an array of stress states.

        Identical maths to :meth:`evaluate` but computed on ``N`` points at
        once with NumPy broadcasting.  See the module docstring for the
        per-mode failure surfaces.

        Parameters
        ----------
        stress_field : np.ndarray
            Shape ``(N, 6)`` array of local stress vectors.
        material : OrthotropicMaterial
            Material with strength allowables (shared by all N points).
        contexts : list, optional
            Ignored — Hashin has no context dependence.

        Returns
        -------
        indices, modes, reserve_factors : np.ndarray, np.ndarray, np.ndarray
            Each of shape ``(N,)``.  ``reserve_factors`` uses the closed-form
            quadratic root for the matrix-compression branch (see
            :func:`_quadratic_reserve_factor`).
        """
        s = np.asarray(stress_field, dtype=np.float64)
        if s.ndim != 2 or s.shape[1] != 6:
            raise ValueError(
                f"stress_field must have shape (N, 6), got {s.shape}"
            )

        s11, s22, s33 = s[:, 0], s[:, 1], s[:, 2]
        t23, t13, t12 = s[:, 3], s[:, 4], s[:, 5]

        Xt, Xc = material.Xt, material.Xc
        Yt, Yc = material.Yt, material.Yc
        S12, S23 = material.S12, material.S23

        # --- Fibre tension (active when s11 >= 0) ---
        ft_sq = (s11 / Xt) ** 2 + (t12 ** 2 + t13 ** 2) / S12 ** 2
        fi_ft = np.where(s11 >= 0, np.sqrt(np.maximum(ft_sq, 0.0)), 0.0)

        # --- Fibre compression (active when s11 < 0) ---
        fi_fc = np.where(s11 < 0, -s11 / Xc, 0.0)

        # --- Matrix branches (depend on sign of s22 + s33) ---
        sig_t = s22 + s33
        shear_term_23 = (t23 ** 2 - s22 * s33) / S23 ** 2
        shear_term_12 = (t12 ** 2 + t13 ** 2) / S12 ** 2

        # Matrix tension (sig_t >= 0)
        mt_sq = (sig_t / Yt) ** 2 + shear_term_23 + shear_term_12
        fi_mt = np.where(sig_t >= 0, np.sqrt(np.maximum(mt_sq, 0.0)), 0.0)

        # Matrix compression (sig_t < 0)
        A_mc = (
            sig_t ** 2 / (4.0 * S23 ** 2)
            + shear_term_23
            + shear_term_12
        )
        B_mc = ((Yc / (2.0 * S23)) ** 2 - 1.0) * sig_t / Yc
        mc_sq = A_mc + B_mc
        fi_mc = np.where(sig_t < 0, np.sqrt(np.maximum(mc_sq, 0.0)), 0.0)

        # --- Stack and pick dominant mode by FI value ---
        all_fi = np.vstack([fi_ft, fi_fc, fi_mt, fi_mc])  # (4, N)
        idx = np.argmax(all_fi, axis=0)
        fi_max = all_fi[idx, np.arange(s.shape[0])]
        modes = self._HASHIN_MODE_LABELS[idx]

        # --- Reserve factor ---
        # For ft/fc/mt branches the FI is a pure quadratic in load, so
        # RF = 1/FI.  For mc, solve A*R^2 + B*R = 1 elementwise.
        rf = np.full_like(fi_max, np.inf)

        # mc branch via vectorised quadratic roots, then mask in
        rf_mc = _quadratic_reserve_factor_vec(A_mc, B_mc)

        # Default: 1/FI; override for the mc-dominant points.
        nonzero = fi_max > 0
        rf[nonzero] = 1.0 / fi_max[nonzero]
        mc_dominant = (idx == 3) & nonzero
        rf[mc_dominant] = rf_mc[mc_dominant]

        return fi_max, modes, rf


def _quadratic_reserve_factor_vec(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Vectorised version of :func:`_quadratic_reserve_factor`.

    Smallest positive root of ``A*R^2 + B*R - 1 = 0`` for each elementwise
    pair ``(A[i], B[i])``.  Returns ``inf`` where no positive root exists or
    where ``A == 0`` and ``B <= 0`` (degenerate).

    Parameters
    ----------
    A, B : np.ndarray
        Same-shape arrays of polynomial coefficients.

    Returns
    -------
    np.ndarray
        Same shape as inputs; smallest positive root of ``A*R^2 + B*R = 1``.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    rf = np.full(A.shape, np.inf, dtype=np.float64)

    # Quadratic branch: A > 0.  Smallest positive root of A*R^2 + B*R - 1 = 0
    # is r_plus = (-B + sqrt(B^2 + 4A)) / (2A) (which is always >= 0 because
    # sqrt(B^2 + 4A) >= |B|).
    quad = A > 0.0
    if np.any(quad):
        Aq = A[quad]
        Bq = B[quad]
        disc = Bq * Bq + 4.0 * Aq
        # disc >= B^2 >= 0 for A > 0, so sqrt is safe.
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        r_plus = (-Bq + sqrt_disc) / (2.0 * Aq)
        rf_quad = np.where(r_plus > 0.0, r_plus, np.inf)
        rf[quad] = rf_quad

    # Degenerate branch: A == 0, B > 0  =>  R = 1/B
    lin = (A == 0.0) & (B > 0.0)
    if np.any(lin):
        rf[lin] = 1.0 / B[lin]

    # All other cases (A < 0 or A == 0 & B <= 0): leave as inf.
    return rf


def _quadratic_reserve_factor(A: float, B: float) -> float:
    """Smallest positive root of ``A*R^2 + B*R - 1 = 0``.

    Used by the Hashin matrix-compression branch, where the failure
    polynomial mixes linear and quadratic terms in the applied load.

    Parameters
    ----------
    A : float
        Coefficient of ``R^2`` (sum of squared-stress / squared-strength
        terms).  Always non-negative for the matrix-compression branch.
    B : float
        Coefficient of ``R`` (the linear ``((Yc/(2 S23))^2 - 1) * sig_t / Yc``
        term).  Can have either sign depending on stress and material.

    Returns
    -------
    float
        Smallest positive ``R`` satisfying ``A*R^2 + B*R = 1``.
        Returns ``float('inf')`` if no positive root exists (a fully
        zero/degenerate state).
    """
    # Degenerate: no quadratic content (pure linear term).  A*R^2 + B*R = 1
    # becomes B*R = 1 => R = 1/B (only positive when B > 0).
    if A <= 0.0:
        if B > 0.0:
            return 1.0 / B
        return float("inf")

    disc = B * B + 4.0 * A
    if disc < 0.0:
        # Should not happen with A > 0, but guard against fp noise.
        return float("inf")
    sqrt_disc = np.sqrt(disc)

    # Roots of A*R^2 + B*R - 1 = 0
    r_plus = (-B + sqrt_disc) / (2.0 * A)
    r_minus = (-B - sqrt_disc) / (2.0 * A)

    # Pick smallest positive root; else +inf
    candidates = [r for r in (r_plus, r_minus) if r > 0.0]
    if not candidates:
        return float("inf")
    return float(min(candidates))
