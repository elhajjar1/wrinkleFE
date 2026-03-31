"""Progressive damage models for composite failure analysis.

This module provides two material degradation strategies that can be applied
after a failure criterion detects damage at a material point:

- :class:`PlyDiscount` : Instantaneous stiffness reduction (ply discount method).
- :class:`ContinuumDamage` : Gradual degradation using the
  Matzenmiller--Lubliner--Taylor (MLT) continuum damage mechanics framework.

Both models implement the :class:`ProgressiveDamageModel` abstract interface,
returning a new :class:`~wrinklefe.core.material.OrthotropicMaterial` with
degraded properties based on the detected failure mode.

Degradation Rules
-----------------
Properties degraded depend on the failure mode:

- **Fibre failure** (tension or compression): E1, nu12, nu13
- **Matrix failure** (tension or compression): E2, G12, G23, nu23
- **Shear failure**: G12, G13, or G23 (depending on plane)
- **Through-thickness failure**: E3, G13, G23

References
----------
- Matzenmiller, A., Lubliner, J., & Taylor, R.L. (1995).
  A constitutive model for anisotropic damage in fiber-composites.
  *Mechanics of Materials*, 20(2), 125--152.
- Camanho, P.P. & Matthews, F.L. (1999). A progressive damage model for
  mechanically fastened joints in composite laminates.
  *J. Composite Materials*, 33(24), 2248--2280.
- Lapczyk, I. & Hurtado, J.A. (2007). Progressive damage modeling in
  fiber-reinforced materials. *Composites Part A*, 38(11), 2333--2341.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.failure.base import FailureResult


class ProgressiveDamageModel(ABC):
    """Abstract interface for material degradation after failure detection.

    Concrete implementations take a pristine (or previously degraded)
    :class:`OrthotropicMaterial` and a :class:`FailureResult` and return
    a **new** material instance with reduced stiffness in the directions
    associated with the detected failure mode.

    The original material is never modified in place.
    """

    @abstractmethod
    def degrade(
        self,
        material: OrthotropicMaterial,
        failure: FailureResult,
    ) -> OrthotropicMaterial:
        """Return a new OrthotropicMaterial with degraded properties.

        The degradation strategy depends on the failure mode reported in
        *failure*:

        - **Fibre failure** (``fiber_tension``, ``fiber_compression``):
          Degrade E1, nu12, nu13.
        - **Matrix failure** (``matrix_tension``, ``matrix_compression``):
          Degrade E2, E3, G12, G23, nu23.
        - **Delamination / through-thickness** (``through_thickness_tension``,
          ``through_thickness_compression``): Degrade E3, G13, G23.
        - **Shear** (``shear_12``, ``shear_13``, ``shear_23``):
          Degrade the corresponding shear modulus.

        Parameters
        ----------
        material : OrthotropicMaterial
            Current material state (may already be partially degraded).
        failure : FailureResult
            Failure result indicating the active failure mode and index.

        Returns
        -------
        OrthotropicMaterial
            New material with degraded properties.
        """


# ======================================================================
# Ply Discount Method
# ======================================================================

class PlyDiscount(ProgressiveDamageModel):
    """Simple ply discount (sudden degradation) method.

    After failure detection, the stiffness components associated with the
    failed mode are reduced to a small residual fraction of their original
    values.  Using a residual factor slightly above zero (default 1 %)
    prevents numerical singularity in the stiffness matrix while
    effectively removing the failed ply's contribution to that load path.

    Parameters
    ----------
    residual_factor : float, optional
        Fraction of original stiffness retained after failure.
        Must satisfy 0 < residual_factor < 1.  Default is 0.01 (1 %).

    Examples
    --------
    >>> model = PlyDiscount(residual_factor=0.01)
    >>> degraded = model.degrade(material, failure_result)
    >>> degraded.E1  # E1 reduced if fibre failure
    1610.0
    """

    def __init__(self, residual_factor: float = 0.01):
        if not (0.0 < residual_factor < 1.0):
            raise ValueError(
                f"residual_factor must be in (0, 1), got {residual_factor}"
            )
        self.residual_factor = residual_factor

    def degrade(
        self,
        material: OrthotropicMaterial,
        failure: FailureResult,
    ) -> OrthotropicMaterial:
        """Apply ply-discount degradation based on failure mode.

        Degradation rules by failure mode:

        - ``fiber_tension`` / ``fiber_compression``:
          E1, nu12, nu13 are multiplied by *residual_factor*.
        - ``matrix_tension`` / ``matrix_compression``:
          E2, G12, G23, nu23 are multiplied by *residual_factor*.
        - ``shear`` / ``shear_12``:
          G12 is multiplied by *residual_factor*.
        - ``shear_13``:
          G13 is multiplied by *residual_factor*.
        - ``shear_23``:
          G23 is multiplied by *residual_factor*.
        - ``through_thickness_tension`` / ``through_thickness_compression``:
          E3, G13, G23 are multiplied by *residual_factor*.

        If the failure index is below 1.0 (no failure), the original
        material is returned unchanged.

        Parameters
        ----------
        material : OrthotropicMaterial
            Current material state.
        failure : FailureResult
            Failure evaluation result.

        Returns
        -------
        OrthotropicMaterial
            New material with degraded properties.  The ``name`` attribute
            is updated to indicate degradation.
        """
        if failure.index < 1.0:
            return material

        r = self.residual_factor
        mode = failure.mode.lower()

        # Start with current properties
        props = material.to_dict()

        if mode in ("fiber_tension", "fiber_compression"):
            props["E1"] = material.E1 * r
            props["nu12"] = material.nu12 * r
            props["nu13"] = material.nu13 * r

        elif mode in ("matrix_tension", "matrix_compression"):
            props["E2"] = material.E2 * r
            props["G12"] = material.G12 * r
            props["G23"] = material.G23 * r
            props["nu23"] = material.nu23 * r

        elif mode in ("shear", "shear_12"):
            props["G12"] = material.G12 * r

        elif mode == "shear_13":
            props["G13"] = material.G13 * r

        elif mode == "shear_23":
            props["G23"] = material.G23 * r

        elif mode in ("through_thickness_tension",
                       "through_thickness_compression"):
            props["E3"] = material.E3 * r
            props["G13"] = material.G13 * r
            props["G23"] = material.G23 * r

        else:
            # Unknown mode: degrade matrix properties as a conservative default
            props["E2"] = material.E2 * r
            props["G12"] = material.G12 * r
            props["G23"] = material.G23 * r
            props["nu23"] = material.nu23 * r

        props["name"] = f"{material.name}_degraded"

        return OrthotropicMaterial.from_dict(props)


# ======================================================================
# Continuum Damage Mechanics (MLT model)
# ======================================================================

class ContinuumDamage(ProgressiveDamageModel):
    """Matzenmiller--Lubliner--Taylor continuum damage mechanics model.

    Uses three scalar damage variables:

    - *d_fiber* : fibre damage (affects E1)
    - *d_matrix* : matrix damage (affects E2, E3)
    - *d_shear* : shear damage (affects G12, G13, G23)

    Each variable evolves in ``[0, 1)`` and is capped at 0.99 for
    numerical stability.  Damage only increases (no healing): once a
    region is damaged it stays damaged or gets worse.

    Degraded stiffnesses:

    .. math::

        E_1^d = (1 - d_f) \\, E_1

        E_2^d = (1 - d_m) \\, E_2

        G_{12}^d = (1 - d_s) \\, G_{12}

    Poisson's ratios are scaled by the corresponding damage variable to
    maintain thermodynamic consistency.

    Parameters
    ----------
    d_fiber : float, optional
        Initial fibre damage variable.  Default is 0.0 (undamaged).
    d_matrix : float, optional
        Initial matrix damage variable.  Default is 0.0.
    d_shear : float, optional
        Initial shear damage variable.  Default is 0.0.

    Examples
    --------
    >>> cdm = ContinuumDamage()
    >>> cdm.update_damage(failure_result)
    >>> degraded_mat = cdm.degrade(material, failure_result)
    """

    _MAX_DAMAGE: float = 0.99

    def __init__(
        self,
        d_fiber: float = 0.0,
        d_matrix: float = 0.0,
        d_shear: float = 0.0,
    ):
        self.d_fiber = np.clip(d_fiber, 0.0, self._MAX_DAMAGE)
        self.d_matrix = np.clip(d_matrix, 0.0, self._MAX_DAMAGE)
        self.d_shear = np.clip(d_shear, 0.0, self._MAX_DAMAGE)

    # ------------------------------------------------------------------
    # Damage evolution
    # ------------------------------------------------------------------

    def update_damage(self, failure: FailureResult) -> None:
        """Update internal damage variables from a failure result.

        The evolution law is:

        .. math::

            d_{\\text{new}} = \\max\\bigl(d_{\\text{old}},\\;
            \\min(1 - 1/\\text{FI},\\; 0.99)\\bigr)

        This ensures damage only increases and is capped below 1.

        For FI < 1 (no failure), the damage variable is unchanged.

        Parameters
        ----------
        failure : FailureResult
            Current failure evaluation.  The ``index`` (FI) and ``mode``
            fields drive the update.
        """
        if failure.index <= 1.0:
            return

        # Compute incremental damage from failure index
        d_increment = min(1.0 - 1.0 / failure.index, self._MAX_DAMAGE)

        mode = failure.mode.lower()

        if mode in ("fiber_tension", "fiber_compression"):
            # Fibre failure is typically catastrophic
            self.d_fiber = max(self.d_fiber, self._MAX_DAMAGE)
        elif mode in ("matrix_tension", "matrix_compression"):
            self.d_matrix = max(self.d_matrix, d_increment)
        elif mode in ("shear", "shear_12", "shear_13", "shear_23"):
            self.d_shear = max(self.d_shear, d_increment)
        elif mode in ("through_thickness_tension",
                       "through_thickness_compression"):
            # Through-thickness failure damages both matrix and shear
            self.d_matrix = max(self.d_matrix, d_increment)
            self.d_shear = max(self.d_shear, d_increment)
        else:
            # Default: treat as matrix damage
            self.d_matrix = max(self.d_matrix, d_increment)

    # ------------------------------------------------------------------
    # Material degradation
    # ------------------------------------------------------------------

    def degrade(
        self,
        material: OrthotropicMaterial,
        failure: FailureResult,
    ) -> OrthotropicMaterial:
        """Apply current damage state to produce a degraded material.

        First calls :meth:`update_damage` to incorporate the latest failure
        result, then applies the three damage variables to the elastic
        constants.

        Parameters
        ----------
        material : OrthotropicMaterial
            Undamaged (or previously degraded) material.
        failure : FailureResult
            Current failure evaluation result.

        Returns
        -------
        OrthotropicMaterial
            New material with stiffness reduced according to the damage
            variables.
        """
        self.update_damage(failure)

        # Effective stiffness with damage
        df = self.d_fiber
        dm = self.d_matrix
        ds = self.d_shear

        props = material.to_dict()
        props["E1"] = material.E1 * (1.0 - df)
        props["E2"] = material.E2 * (1.0 - dm)
        props["E3"] = material.E3 * (1.0 - dm)
        props["G12"] = material.G12 * (1.0 - ds)
        props["G13"] = material.G13 * (1.0 - ds)
        props["G23"] = material.G23 * (1.0 - ds)

        # Scale Poisson's ratios to maintain thermodynamic consistency
        # nu_ij / E_i = nu_ji / E_j must hold for the degraded state.
        # Scaling nu12 and nu13 by sqrt((1-df)*(1-dm)) is a common approach;
        # here we use the simpler proportional scaling:
        props["nu12"] = material.nu12 * (1.0 - df)
        props["nu13"] = material.nu13 * (1.0 - df)
        props["nu23"] = material.nu23 * (1.0 - dm)

        props["name"] = f"{material.name}_damaged"

        return OrthotropicMaterial.from_dict(props)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def damage_vector(self) -> np.ndarray:
        """Current damage state as a 3-element array [d_fiber, d_matrix, d_shear].

        Returns
        -------
        np.ndarray
            Shape ``(3,)`` damage variables.
        """
        return np.array([self.d_fiber, self.d_matrix, self.d_shear])

    @property
    def is_damaged(self) -> bool:
        """Whether any damage variable is non-zero."""
        return self.d_fiber > 0.0 or self.d_matrix > 0.0 or self.d_shear > 0.0

    def reset(self) -> None:
        """Reset all damage variables to zero (undamaged state)."""
        self.d_fiber = 0.0
        self.d_matrix = 0.0
        self.d_shear = 0.0

    def __repr__(self) -> str:
        return (
            f"ContinuumDamage(d_fiber={self.d_fiber:.4f}, "
            f"d_matrix={self.d_matrix:.4f}, d_shear={self.d_shear:.4f})"
        )
