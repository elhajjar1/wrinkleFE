"""Base classes for composite failure criteria.

This module defines the abstract interface that all failure criteria must
implement, along with the :class:`FailureResult` data container returned by
every evaluation.

Stress Convention
-----------------
All criteria operate on a 6-component stress vector in **local material
coordinates** using Voigt notation::

    stress_local = [sigma_11, sigma_22, sigma_33, tau_23, tau_13, tau_12]

where:
- sigma_11 : fibre-direction normal stress
- sigma_22 : in-plane transverse normal stress
- sigma_33 : through-thickness normal stress
- tau_23   : transverse shear stress (2-3 plane)
- tau_13   : interlaminar shear stress (1-3 plane)
- tau_12   : in-plane shear stress (1-2 plane)

Failure Index Convention
------------------------
A **failure index** (FI) >= 1.0 indicates failure.  The **reserve factor**
is defined as RF = 1 / FI and represents the factor by which the applied
load can be scaled before failure occurs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from wrinklefe.core.material import OrthotropicMaterial


@dataclass
class FailureResult:
    """Result from evaluating a failure criterion at a single integration point.

    Attributes
    ----------
    index : float
        Failure index (FI).  FI >= 1.0 means the material has failed
        according to the criterion.
    mode : str
        Dominant failure mode label, e.g. ``"fiber_tension"``,
        ``"matrix_compression"``, ``"shear_12"``.
    reserve_factor : float
        Reserve factor RF = 1 / FI.  Values > 1.0 indicate a positive
        margin of safety; values <= 1.0 indicate failure.  Set to
        ``float('inf')`` when FI == 0.
    criterion_name : str
        Name of the criterion that produced this result (e.g.
        ``"max_stress"``, ``"hashin"``, ``"tsai_wu"``).
    """

    index: float
    mode: str
    reserve_factor: float
    criterion_name: str
    detail: dict = field(default_factory=dict)


class FailureCriterion(ABC):
    """Abstract base class for composite failure criteria.

    All concrete criteria must implement :meth:`evaluate`, which takes a
    single stress state in local material coordinates and returns a
    :class:`FailureResult`.

    The convenience method :meth:`evaluate_field` loops over an array of
    stress states and can be overridden for vectorised implementations.

    Subclasses should set the class attribute :attr:`name` to a short
    identifier string (e.g. ``"max_stress"``).
    """

    name: str = "base"

    @abstractmethod
    def evaluate(
        self,
        stress_local: np.ndarray,
        material: OrthotropicMaterial,
        context: Optional[Dict[str, Any]] = None,
    ) -> FailureResult:
        """Evaluate the failure criterion at a single material point.

        Parameters
        ----------
        stress_local : np.ndarray
            Shape ``(6,)`` stress vector in local material coordinates::

                [sigma_11, sigma_22, sigma_33, tau_23, tau_13, tau_12]

            Units must be consistent with the material strength values
            (typically MPa).
        material : OrthotropicMaterial
            Orthotropic material providing strength allowables
            (Xt, Xc, Yt, Yc, Zt, Zc, S12, S13, S23).
        context : dict, optional
            Element-level geometric data for physics-based criteria.
            Supported keys:

            - ``'misalignment_angle'`` (float): local fibre misalignment
              angle in radians (used by LaRC04/05 for the kinking frame).
            - ``'ply_thickness'`` (float): ply thickness in mm (used for
              in-situ strength corrections).

            Criteria that do not need context simply ignore this parameter.

        Returns
        -------
        FailureResult
            Contains the failure index, dominant failure mode, reserve
            factor, and criterion name.
        """

    def evaluate_field(
        self,
        stress_field: np.ndarray,
        material: OrthotropicMaterial,
        contexts: Optional[list] = None,
    ) -> list[FailureResult]:
        """Evaluate the failure criterion over an array of stress states.

        This default implementation loops over rows of *stress_field*.
        Subclasses may override with a vectorised version for performance.

        Parameters
        ----------
        stress_field : np.ndarray
            Shape ``(N, 6)`` array where each row is a local stress state.
        material : OrthotropicMaterial
            Orthotropic material with strength properties.
        contexts : list of dict, optional
            Per-point context dicts (same length as stress_field).
            If ``None``, each point gets ``None`` context.

        Returns
        -------
        list[FailureResult]
            One :class:`FailureResult` per row of *stress_field*.
        """
        results: list[FailureResult] = []
        for i in range(stress_field.shape[0]):
            ctx = contexts[i] if contexts is not None else None
            results.append(self.evaluate(stress_field[i], material, ctx))
        return results
