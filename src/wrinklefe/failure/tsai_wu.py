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
    >>> # Pure fibre compression at Xc (=1200 MPa) should give FI = 1.0
    >>> stress = np.array([-1200.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    >>> result = criterion.evaluate(stress, mat)
    >>> bool(abs(result.index - 1.0) < 1e-10)
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
        # Track linear (L) and quadratic (Q) contributions separately so the
        # reserve factor (strength ratio) can be computed from the proper
        # quadratic root, not from 1/f (which is only correct for criteria
        # linear in load scale).
        L = F1 * s[0] + F2 * s[1] + F3 * s[2]

        Q = (
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

        fi = L + Q

        # --- Reserve factor (strength ratio R) ---
        # The applied load is scaled by R; failure occurs when
        #     R^2 * Q + R * L = 1.
        # Solve for the positive root.  Sibling criteria (e.g. Hashin) return
        # +inf when there is no load contribution; we follow the same
        # convention for degenerate / unbounded cases.
        if Q > 0.0:
            disc = L * L + 4.0 * Q
            if disc < 0.0:
                # Cannot occur for Q > 0, but guard for floating-point safety.
                reserve_factor = float("inf")
            else:
                reserve_factor = (-L + np.sqrt(disc)) / (2.0 * Q)
                if reserve_factor <= 0.0:
                    reserve_factor = float("inf")
        elif Q == 0.0:
            if L > 0.0:
                reserve_factor = 1.0 / L
            else:
                # L <= 0 (including L == 0): no failure under increasing load.
                reserve_factor = float("inf")
        else:
            # Q < 0: unusual (open failure surface in this direction).
            disc = L * L + 4.0 * Q
            if disc < 0.0:
                # No real positive root reachable; treat as unbounded.
                reserve_factor = float("inf")
            else:
                # Two real roots; take the smallest positive one.
                sqrt_disc = np.sqrt(disc)
                r1 = (-L + sqrt_disc) / (2.0 * Q)
                r2 = (-L - sqrt_disc) / (2.0 * Q)
                positive_roots = [r for r in (r1, r2) if r > 0.0]
                reserve_factor = min(positive_roots) if positive_roots else float("inf")

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
            reserve_factor=reserve_factor,
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
        """Vectorised Tsai-Wu evaluation across an array of stress states.

        Identical maths to :meth:`evaluate` but computed on ``N`` points at
        once with NumPy broadcasting.

        Parameters
        ----------
        stress_field : np.ndarray
            Shape ``(N, 6)`` array of local stress vectors.
        material : OrthotropicMaterial
            Material with strength allowables (shared by all N points).
        contexts : list, optional
            Ignored — Tsai-Wu has no context dependence.

        Returns
        -------
        indices, modes, reserve_factors : np.ndarray, np.ndarray, np.ndarray
            Each of shape ``(N,)``.  ``reserve_factors`` is the positive root
            of ``R^2 Q + R L = 1`` (the strength ratio).
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
        Zt, Zc = material.Zt, material.Zc
        S12, S13, S23 = material.S12, material.S13, material.S23

        F1 = 1.0 / Xt - 1.0 / Xc
        F2 = 1.0 / Yt - 1.0 / Yc
        F3 = 1.0 / Zt - 1.0 / Zc
        F11 = 1.0 / (Xt * Xc)
        F22 = 1.0 / (Yt * Yc)
        F33 = 1.0 / (Zt * Zc)
        F44 = 1.0 / S23 ** 2
        F55 = 1.0 / S13 ** 2
        F66 = 1.0 / S12 ** 2
        F12 = self.f12_star * np.sqrt(F11 * F22)
        F13 = self.f12_star * np.sqrt(F11 * F33)
        F23 = self.f12_star * np.sqrt(F22 * F33)

        L = F1 * s11 + F2 * s22 + F3 * s33
        Q = (
            F11 * s11 ** 2
            + F22 * s22 ** 2
            + F33 * s33 ** 2
            + 2.0 * F12 * s11 * s22
            + 2.0 * F13 * s11 * s33
            + 2.0 * F23 * s22 * s33
            + F44 * t23 ** 2
            + F55 * t13 ** 2
            + F66 * t12 ** 2
        )
        fi = L + Q

        # --- Reserve factor: smallest positive R with R^2 Q + R L = 1 ---
        # Three branches, matching the scalar evaluate path.
        rf = np.full_like(fi, np.inf)

        # Q > 0: closed surface, take r_plus = (-L + sqrt(L^2 + 4Q)) / (2Q).
        qp = Q > 0.0
        if np.any(qp):
            Lp = L[qp]
            Qp = Q[qp]
            disc = Lp * Lp + 4.0 * Qp
            sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
            r_plus = (-Lp + sqrt_disc) / (2.0 * Qp)
            rf_qp = np.where(r_plus > 0.0, r_plus, np.inf)
            rf[qp] = rf_qp

        # Q == 0, L > 0  =>  R = 1/L
        qz = (Q == 0.0) & (L > 0.0)
        if np.any(qz):
            rf[qz] = 1.0 / L[qz]

        # Q < 0: unusual open surface; take smallest positive root if any.
        qn = Q < 0.0
        if np.any(qn):
            Ln = L[qn]
            Qn = Q[qn]
            disc = Ln * Ln + 4.0 * Qn
            valid = disc >= 0.0
            if np.any(valid):
                Lv = Ln[valid]
                Qv = Qn[valid]
                sqrt_disc = np.sqrt(disc[valid])
                r1 = (-Lv + sqrt_disc) / (2.0 * Qv)
                r2 = (-Lv - sqrt_disc) / (2.0 * Qv)
                pos1 = np.where(r1 > 0.0, r1, np.inf)
                pos2 = np.where(r2 > 0.0, r2, np.inf)
                rf_qn = np.minimum(pos1, pos2)
                # rf_qn already +inf where no positive root.
                # Write back to rf via composed mask.
                qn_idx = np.where(qn)[0][valid]
                rf[qn_idx] = rf_qn

        # --- Approximate mode identification ---
        fiber_contrib = np.abs(F1 * s11) + F11 * s11 ** 2
        matrix_contrib = np.abs(F2 * s22) + F22 * s22 ** 2
        shear_contrib = F66 * t12 ** 2

        modes = np.empty(s.shape[0], dtype="U32")
        is_fiber = (fiber_contrib >= matrix_contrib) & (fiber_contrib >= shear_contrib)
        is_matrix = (~is_fiber) & (matrix_contrib >= shear_contrib)
        is_shear = ~(is_fiber | is_matrix)

        modes[is_fiber & (s11 >= 0)] = "fiber_tension"
        modes[is_fiber & (s11 < 0)] = "fiber_compression"
        modes[is_matrix & (s22 >= 0)] = "matrix_tension"
        modes[is_matrix & (s22 < 0)] = "matrix_compression"
        modes[is_shear] = "shear"

        return fi, modes, rf
