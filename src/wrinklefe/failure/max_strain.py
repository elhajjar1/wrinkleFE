"""Maximum Strain failure criterion for orthotropic composite laminates.

This criterion computes strains from the applied stress state via the
compliance matrix and compares each strain component independently against
its ultimate allowable value.  The failure index is the maximum ratio across
all six strain components.

Mathematical Formulation
------------------------
Given stress vector ``sigma`` in Voigt notation (11, 22, 33, 23, 13, 12),
the working strains are the *uncoupled engineering strains* obtained by
dividing each stress component by its corresponding direct modulus::

    epsilon = [sigma_11/E1, sigma_22/E2, sigma_33/E3,
               tau_23/G23,  tau_13/G13,  tau_12/G12]

This deliberately omits the Poisson-coupling off-diagonal terms of the
full compliance matrix ``[S]``.  It is the standard engineering
"maximum strain" convention (Jones, *Mechanics of Composite Materials*;
Daniel & Ishai; Tsai), and it is the *only* basis on which the working
strains are consistent with the uniaxial-test allowables below: each
uniaxial strength then maps exactly to a failure index of 1.0.

Using the coupled compliance path (``epsilon = [S] @ sigma``) instead
breaks this self-calibration -- e.g. pure transverse compression
``sigma_22 = -Yc`` induces a Poisson strain ``eps_33 = nu23*Yc/E2`` that
is compared against the unrelated allowable ``Zt/E3``, spuriously firing
a through-thickness tension failure (FI ~ 1.38 instead of 1.0).

Ultimate strains are derived from uniaxial strengths::

    eps_1t = Xt / E1      (fibre tension)
    eps_1c = Xc / E1      (fibre compression)
    eps_2t = Yt / E2      (transverse tension)
    eps_2c = Yc / E2      (transverse compression)
    eps_3t = Zt / E3      (through-thickness tension)
    eps_3c = Zc / E3      (through-thickness compression)
    gamma_23_ult = S23 / G23
    gamma_13_ult = S13 / G13
    gamma_12_ult = S12 / G12

For each normal strain component, the appropriate sign-dependent allowable
is selected (tensile if strain >= 0, compressive otherwise).  For shear
strains, the absolute value is compared against the ultimate shear strain.

The failure index is::

    FI = max(|eps_i| / eps_i_ult)   over all six components

and the dominant failure mode corresponds to the component with the
highest ratio.
"""

from __future__ import annotations

import numpy as np

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.failure.base import FailureCriterion, FailureResult


class MaxStrainCriterion(FailureCriterion):
    """Maximum Strain failure criterion for 3-D orthotropic composites.

    Each strain component is independently checked against its ultimate
    allowable.  The overall failure index is the largest component ratio.

    Attributes
    ----------
    name : str
        ``"max_strain"``
    """

    name = "max_strain"

    # Labels for reporting the dominant failure mode
    _MODE_LABELS = (
        "fiber_tension", "fiber_compression",
        "matrix_transverse_tension", "matrix_transverse_compression",
        "matrix_thickness_tension", "matrix_thickness_compression",
        "shear_23", "shear_13", "shear_12",
    )

    def evaluate(
        self,
        stress_local: np.ndarray,
        material: OrthotropicMaterial,
        context=None,
    ) -> FailureResult:
        """Evaluate Maximum Strain criterion at a single material point.

        Parameters
        ----------
        stress_local : np.ndarray
            Shape ``(6,)`` stress vector in local material coordinates::

                [sigma_11, sigma_22, sigma_33, tau_23, tau_13, tau_12]

        material : OrthotropicMaterial
            Material with elastic constants and strength allowables.

        Returns
        -------
        FailureResult
            Failure index, dominant mode, reserve factor, and criterion name.
        """
        stress_local = np.asarray(stress_local, dtype=np.float64)

        # Uncoupled engineering strains (no Poisson off-diagonal coupling).
        # This is the basis on which the uniaxial-test allowables below are
        # defined, so each uniaxial strength maps exactly to FI = 1.0.
        # Using the full compliance matrix would introduce Poisson-coupled
        # strains that are inconsistent with the uniaxial allowables.
        strain = np.array([
            stress_local[0] / material.E1,
            stress_local[1] / material.E2,
            stress_local[2] / material.E3,
            stress_local[3] / material.G23,
            stress_local[4] / material.G13,
            stress_local[5] / material.G12,
        ])

        # Ultimate strains (all positive magnitudes)
        eps1t = material.Xt / material.E1
        eps1c = material.Xc / material.E1
        eps2t = material.Yt / material.E2
        eps2c = material.Yc / material.E2
        eps3t = material.Zt / material.E3
        eps3c = material.Zc / material.E3
        gamma23_ult = material.S23 / material.G23
        gamma13_ult = material.S13 / material.G13
        gamma12_ult = material.S12 / material.G12

        # Failure ratios for each component
        ratios = np.zeros(9)

        # Fibre direction (1)
        if strain[0] >= 0:
            ratios[0] = strain[0] / eps1t          # fiber_tension
        else:
            ratios[1] = abs(strain[0]) / eps1c      # fiber_compression

        # Transverse direction (2)
        if strain[1] >= 0:
            ratios[2] = strain[1] / eps2t           # matrix_transverse_tension
        else:
            ratios[3] = abs(strain[1]) / eps2c      # matrix_transverse_compression

        # Through-thickness direction (3)
        if strain[2] >= 0:
            ratios[4] = strain[2] / eps3t           # matrix_thickness_tension
        else:
            ratios[5] = abs(strain[2]) / eps3c      # matrix_thickness_compression

        # Shear components
        ratios[6] = abs(strain[3]) / gamma23_ult
        ratios[7] = abs(strain[4]) / gamma13_ult
        ratios[8] = abs(strain[5]) / gamma12_ult

        # Overall failure index and dominant mode
        idx_max = int(np.argmax(ratios))
        fi = float(ratios[idx_max])
        mode = self._MODE_LABELS[idx_max]
        rf = 1.0 / fi if fi > 0 else float("inf")

        return FailureResult(
            index=fi,
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
        """Vectorised Max-Strain evaluation across an array of stress states.

        Computes the same failure index and dominant mode as :meth:`evaluate`
        for ``N`` stress vectors at once using NumPy broadcasting.

        Parameters
        ----------
        stress_field : np.ndarray
            Shape ``(N, 6)`` array of local stress vectors.
        material : OrthotropicMaterial
            Material with elastic and strength properties (shared by all N).
        contexts : list, optional
            Ignored — Max-Strain has no context dependence.

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

        n = s.shape[0]

        # Uncoupled engineering strains (matches scalar evaluate exactly).
        eps11 = s[:, 0] / material.E1
        eps22 = s[:, 1] / material.E2
        eps33 = s[:, 2] / material.E3
        g23 = s[:, 3] / material.G23
        g13 = s[:, 4] / material.G13
        g12 = s[:, 5] / material.G12

        # Ultimate strain magnitudes (scalars)
        eps1t = material.Xt / material.E1
        eps1c = material.Xc / material.E1
        eps2t = material.Yt / material.E2
        eps2c = material.Yc / material.E2
        eps3t = material.Zt / material.E3
        eps3c = material.Zc / material.E3
        gamma23_ult = material.S23 / material.G23
        gamma13_ult = material.S13 / material.G13
        gamma12_ult = material.S12 / material.G12

        ratios = np.zeros((9, n), dtype=np.float64)
        ratios[0] = np.where(eps11 >= 0,  eps11 / eps1t,  0.0)   # fiber_tension
        ratios[1] = np.where(eps11 <  0, -eps11 / eps1c,  0.0)   # fiber_compression
        ratios[2] = np.where(eps22 >= 0,  eps22 / eps2t,  0.0)   # matrix_transverse_tension
        ratios[3] = np.where(eps22 <  0, -eps22 / eps2c,  0.0)   # matrix_transverse_compression
        ratios[4] = np.where(eps33 >= 0,  eps33 / eps3t,  0.0)   # matrix_thickness_tension
        ratios[5] = np.where(eps33 <  0, -eps33 / eps3c,  0.0)   # matrix_thickness_compression
        ratios[6] = np.abs(g23) / gamma23_ult                    # shear_23
        ratios[7] = np.abs(g13) / gamma13_ult                    # shear_13
        ratios[8] = np.abs(g12) / gamma12_ult                    # shear_12

        idx = np.argmax(ratios, axis=0)
        fi = ratios[idx, np.arange(n)]
        labels = np.asarray(self._MODE_LABELS, dtype="U32")
        modes = labels[idx]

        rf = np.where(fi > 0, 1.0 / np.where(fi > 0, fi, 1.0), np.inf)
        return fi, modes, rf
