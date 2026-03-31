"""Maximum Strain failure criterion for orthotropic composite laminates.

This criterion computes strains from the applied stress state via the
compliance matrix and compares each strain component independently against
its ultimate allowable value.  The failure index is the maximum ratio across
all six strain components.

Mathematical Formulation
------------------------
Given stress vector ``sigma`` in Voigt notation (11, 22, 33, 23, 13, 12),
compute the strain vector::

    epsilon = [S] @ sigma

where ``[S]`` is the 6x6 compliance matrix.

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
        S = material.compliance_matrix
        strain = S @ stress_local

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
