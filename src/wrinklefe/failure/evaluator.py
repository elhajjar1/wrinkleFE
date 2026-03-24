"""Multi-criteria failure evaluator for composite laminates.

This module provides :class:`FailureEvaluator`, which runs multiple failure
criteria on the same stress state and produces comprehensive failure reports
including first-ply failure (FPF), last-ply failure (LPF), per-ply failure
indices, and failure envelopes in 2D load space.

The companion :class:`LaminateFailureReport` dataclass collects all results
into a structured container with a human-readable summary.

Typical Usage
-------------
>>> from wrinklefe.failure.evaluator import FailureEvaluator
>>> evaluator = FailureEvaluator.default_criteria()
>>> report = evaluator.evaluate_laminate(laminate, load)
>>> print(report.summary())

References
----------
- Jones, R.M. (1999). Mechanics of Composite Materials, 2nd ed. Taylor & Francis.
- Reddy, J.N. (2004). Mechanics of Laminated Composite Plates and Shells.
- MIL-HDBK-17-3F (2002). Composite Materials Handbook, Vol. 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wrinklefe.failure.base import FailureCriterion, FailureResult
from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.core.laminate import Laminate, LoadState


@dataclass
class LaminateFailureReport:
    """Complete failure analysis report for a laminate.

    Attributes
    ----------
    fpf : dict[str, dict]
        First-ply failure results for each criterion. Each entry is a dict
        with keys ``"fi"`` (failure index), ``"ply"`` (ply index),
        ``"mode"`` (failure mode string), and ``"load_factor"`` (scaling
        factor to reach FI = 1).
    lpf : dict[str, dict]
        Last-ply failure results for each criterion. Same structure as
        *fpf* but reports the ply with the maximum failure index.
    ply_failure_indices : dict[str, np.ndarray]
        Per-ply failure indices for each criterion. Each array has shape
        ``(n_plies,)`` and contains the maximum FI at that ply.
    critical_ply : int
        Index of the ply that fails first (across all criteria).
    critical_mode : str
        Failure mode of the critical ply.
    critical_criterion : str
        Name of the criterion that predicts the lowest load factor.
    """

    fpf: dict[str, dict] = field(default_factory=dict)
    lpf: dict[str, dict] = field(default_factory=dict)
    ply_failure_indices: dict[str, np.ndarray] = field(default_factory=dict)
    critical_ply: int = 0
    critical_mode: str = ""
    critical_criterion: str = ""

    def summary(self) -> str:
        """Formatted text summary of the failure analysis.

        Returns
        -------
        str
            Multi-line summary including FPF/LPF results for each criterion,
            the critical ply, mode, and load factor.
        """
        lines = [
            "=" * 65,
            "  Laminate Failure Analysis Report",
            "=" * 65,
        ]

        # Critical result
        lines.append(f"  Critical criterion : {self.critical_criterion}")
        lines.append(f"  Critical ply       : {self.critical_ply}")
        lines.append(f"  Critical mode      : {self.critical_mode}")
        if self.critical_criterion in self.fpf:
            lf = self.fpf[self.critical_criterion].get("load_factor", float("nan"))
            lines.append(f"  FPF load factor    : {lf:.4f}")
        lines.append("")

        # Per-criterion FPF
        lines.append("  First-Ply Failure (FPF):")
        lines.append(f"  {'Criterion':<18s} {'FI':>8s} {'Ply':>5s} {'Mode':<25s} {'LF':>8s}")
        lines.append("  " + "-" * 66)
        for crit_name, data in sorted(self.fpf.items()):
            fi = data.get("fi", 0.0)
            ply = data.get("ply", -1)
            mode = data.get("mode", "")
            lf = data.get("load_factor", float("inf"))
            lines.append(
                f"  {crit_name:<18s} {fi:8.4f} {ply:5d} {mode:<25s} {lf:8.4f}"
            )

        lines.append("")

        # Per-criterion LPF
        lines.append("  Last-Ply Failure (LPF):")
        lines.append(f"  {'Criterion':<18s} {'FI':>8s} {'Ply':>5s} {'Mode':<25s} {'LF':>8s}")
        lines.append("  " + "-" * 66)
        for crit_name, data in sorted(self.lpf.items()):
            fi = data.get("fi", 0.0)
            ply = data.get("ply", -1)
            mode = data.get("mode", "")
            lf = data.get("load_factor", float("inf"))
            lines.append(
                f"  {crit_name:<18s} {fi:8.4f} {ply:5d} {mode:<25s} {lf:8.4f}"
            )

        lines.append("=" * 65)
        return "\n".join(lines)


class FailureEvaluator:
    """Evaluates multiple failure criteria on the same stress state.

    Provides comparison between criteria and identification of the critical
    failure mode, ply, and load factor.

    Parameters
    ----------
    criteria : list[FailureCriterion]
        List of failure criteria to evaluate.  Each criterion must implement
        the :class:`~wrinklefe.failure.base.FailureCriterion` interface.

    Examples
    --------
    >>> from wrinklefe.failure.max_stress import MaxStressCriterion
    >>> from wrinklefe.failure.hashin import HashinCriterion
    >>> evaluator = FailureEvaluator([MaxStressCriterion(), HashinCriterion()])
    >>> results = evaluator.evaluate_point(stress, material)
    """

    def __init__(self, criteria: list[FailureCriterion]):
        if not criteria:
            raise ValueError("At least one failure criterion must be provided.")
        self.criteria = list(criteria)

    # ------------------------------------------------------------------
    # Point-level evaluation
    # ------------------------------------------------------------------

    def evaluate_point(
        self,
        stress_local: np.ndarray,
        material: OrthotropicMaterial,
    ) -> dict[str, FailureResult]:
        """Evaluate all criteria at a single material point.

        Parameters
        ----------
        stress_local : np.ndarray
            Shape ``(6,)`` stress vector in local material coordinates
            ``[sigma_11, sigma_22, sigma_33, tau_23, tau_13, tau_12]``.
        material : OrthotropicMaterial
            Material with strength allowables.

        Returns
        -------
        dict[str, FailureResult]
            Mapping from criterion name to its :class:`FailureResult`.
        """
        stress_local = np.asarray(stress_local, dtype=np.float64).ravel()
        if stress_local.shape != (6,):
            raise ValueError(
                f"stress_local must have shape (6,), got {stress_local.shape}"
            )

        results: dict[str, FailureResult] = {}
        for criterion in self.criteria:
            result = criterion.evaluate(stress_local, material, None)
            results[criterion.name] = result
        return results

    # ------------------------------------------------------------------
    # Laminate-level evaluation
    # ------------------------------------------------------------------

    def evaluate_laminate(
        self,
        laminate: Laminate,
        load: LoadState,
    ) -> LaminateFailureReport:
        """Evaluate failure for each ply in the laminate under given load.

        For each ply, this method:

        1. Computes the local stress state using CLT via
           :meth:`~wrinklefe.core.laminate.Laminate.ply_stresses_local`.
        2. Evaluates all criteria at that stress state.
        3. Tracks the first-ply failure (ply with the minimum load factor
           that would cause FI >= 1) and last-ply failure (ply with the
           maximum failure index).

        The load factor is computed assuming linear scaling: if the current
        failure index is *f*, then the load factor to reach FI = 1 is
        ``1 / f``.  A load factor < 1 means the laminate has already failed
        at the current load level.

        Parameters
        ----------
        laminate : Laminate
            Laminate with ply definitions and ABD matrices.
        load : LoadState
            Applied load state (force/moment resultants and environmental
            loads).

        Returns
        -------
        LaminateFailureReport
            Complete failure report for the laminate.
        """
        n_plies = laminate.n_plies

        # Storage: per-criterion, per-ply
        ply_fi: dict[str, np.ndarray] = {
            c.name: np.zeros(n_plies) for c in self.criteria
        }
        ply_modes: dict[str, list[str]] = {
            c.name: [""] * n_plies for c in self.criteria
        }

        # Evaluate each ply
        for k in range(n_plies):
            # CLT returns [sigma_1, sigma_2, tau_12] (3-component plane stress)
            stress_2d = laminate.ply_stresses_local(load, k, position="mid")

            # Expand to full 6-component vector for the failure criteria.
            # sigma_33 = tau_23 = tau_13 = 0  (plane stress assumption)
            stress_6 = np.zeros(6, dtype=np.float64)
            stress_6[0] = stress_2d[0]  # sigma_11
            stress_6[1] = stress_2d[1]  # sigma_22
            stress_6[5] = stress_2d[2]  # tau_12

            material = laminate.plies[k].material

            for criterion in self.criteria:
                result = criterion.evaluate(stress_6, material)
                ply_fi[criterion.name][k] = result.index
                ply_modes[criterion.name][k] = result.mode

        # Build FPF and LPF for each criterion
        fpf: dict[str, dict] = {}
        lpf: dict[str, dict] = {}

        for criterion in self.criteria:
            name = criterion.name
            fi_array = ply_fi[name]
            modes = ply_modes[name]

            # FPF: ply with the maximum FI (first to fail under load scaling)
            fpf_ply = int(np.argmax(fi_array))
            fpf_fi = float(fi_array[fpf_ply])
            fpf_lf = 1.0 / fpf_fi if fpf_fi > 0.0 else float("inf")

            fpf[name] = {
                "fi": fpf_fi,
                "ply": fpf_ply,
                "mode": modes[fpf_ply],
                "load_factor": fpf_lf,
            }

            # LPF: simplified as the ply with the maximum FI
            # (In a full progressive analysis, this would require iterative
            # degradation.  Here we report the same ply as the most critical.)
            lpf_ply = int(np.argmax(fi_array))
            lpf_fi = float(fi_array[lpf_ply])
            lpf_lf = 1.0 / lpf_fi if lpf_fi > 0.0 else float("inf")

            lpf[name] = {
                "fi": lpf_fi,
                "ply": lpf_ply,
                "mode": modes[lpf_ply],
                "load_factor": lpf_lf,
            }

        # Determine the globally critical criterion and ply
        best_crit = ""
        best_lf = float("inf")
        best_ply = 0
        best_mode = ""

        for crit_name, data in fpf.items():
            lf = data["load_factor"]
            if lf < best_lf:
                best_lf = lf
                best_crit = crit_name
                best_ply = data["ply"]
                best_mode = data["mode"]

        return LaminateFailureReport(
            fpf=fpf,
            lpf=lpf,
            ply_failure_indices=ply_fi,
            critical_ply=best_ply,
            critical_mode=best_mode,
            critical_criterion=best_crit,
        )

    # ------------------------------------------------------------------
    # Field-level evaluation (FE post-processing)
    # ------------------------------------------------------------------

    def evaluate_field(
        self,
        stress_local_field: np.ndarray,
        materials: list[OrthotropicMaterial],
        ply_ids: np.ndarray,
        fiber_angles: np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Evaluate criteria over an FE stress field.

        Loops over every element and Gauss point, selecting the material
        from *materials* using the element's ply ID from *ply_ids*.

        Parameters
        ----------
        stress_local_field : np.ndarray
            Shape ``(n_elements, n_gauss, 6)`` local stresses at each
            Gauss point.
        materials : list[OrthotropicMaterial]
            List of materials, one per unique ply.  Indexed by *ply_ids*.
        ply_ids : np.ndarray
            Shape ``(n_elements,)`` integer array giving the ply index for
            each element.
        fiber_angles : np.ndarray or None, optional
            Shape ``(n_elements,)`` per-element fibre misalignment angles
            in radians from wrinkle geometry.  Passed to criteria (e.g.
            LaRC05) via the ``context`` dict.

        Returns
        -------
        fi_fields : dict[str, np.ndarray]
            Mapping from criterion name to a ``(n_elements, n_gauss)``
            array of failure index values.
        mode_fields : dict[str, np.ndarray]
            Mapping from criterion name to a ``(n_elements, n_gauss)``
            array of failure mode strings.
        """
        stress_local_field = np.asarray(stress_local_field, dtype=np.float64)
        ply_ids = np.asarray(ply_ids, dtype=int)

        if stress_local_field.ndim != 3 or stress_local_field.shape[2] != 6:
            raise ValueError(
                "stress_local_field must have shape (n_elements, n_gauss, 6), "
                f"got {stress_local_field.shape}"
            )

        n_elements, n_gauss, _ = stress_local_field.shape

        if ply_ids.shape != (n_elements,):
            raise ValueError(
                f"ply_ids must have shape ({n_elements},), got {ply_ids.shape}"
            )

        # Pre-allocate output arrays
        fi_fields: dict[str, np.ndarray] = {
            c.name: np.zeros((n_elements, n_gauss), dtype=np.float64)
            for c in self.criteria
        }
        mode_fields: dict[str, np.ndarray] = {
            c.name: np.empty((n_elements, n_gauss), dtype="U20")
            for c in self.criteria
        }

        for e in range(n_elements):
            mat = materials[ply_ids[e]]
            # Build context for physics-based criteria (LaRC05)
            ctx = None
            if fiber_angles is not None:
                ctx = {"misalignment_angle": float(fiber_angles[e])}
            for g in range(n_gauss):
                stress = stress_local_field[e, g, :]
                for criterion in self.criteria:
                    result = criterion.evaluate(stress, mat, ctx)
                    fi_fields[criterion.name][e, g] = result.index
                    mode_fields[criterion.name][e, g] = result.mode

        return fi_fields, mode_fields

    # ------------------------------------------------------------------
    # Failure envelope
    # ------------------------------------------------------------------

    def strength_ratio_envelope(
        self,
        laminate: Laminate,
        load_type: str = "Nx-Ny",
        n_points: int = 360,
    ) -> dict[str, np.ndarray]:
        """Compute failure envelope in 2D load space.

        Sweeps the angle theta from 0 to 2*pi in *n_points* steps. At each
        angle, a unit load vector is constructed in the plane defined by
        *load_type*, and the laminate is evaluated to find the strength ratio
        (minimum reserve factor across all plies).  The failure envelope
        point is the unit load scaled by that ratio.

        Parameters
        ----------
        laminate : Laminate
            Laminate definition.
        load_type : str, optional
            Load plane for the envelope sweep.  Supported values:

            - ``"Nx-Ny"`` : in-plane biaxial (force resultants)
            - ``"Nx-Nxy"`` : axial + in-plane shear
            - ``"Mx-My"`` : biaxial bending moments
            - ``"Nx-Mx"`` : axial force + bending moment

            Default is ``"Nx-Ny"``.
        n_points : int, optional
            Number of points around the envelope.  Default is 360.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from criterion name to a ``(n_points, 2)`` array of
            ``[load1, load2]`` coordinates on the failure surface.

        Raises
        ------
        ValueError
            If *load_type* is not one of the supported options.
        """
        valid_types = ("Nx-Ny", "Nx-Nxy", "Mx-My", "Nx-Mx")
        if load_type not in valid_types:
            raise ValueError(
                f"load_type must be one of {valid_types}, got {load_type!r}"
            )

        angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)

        # Map load_type to LoadState field indices
        # LoadState.to_vector() returns [Nx, Ny, Nxy, Mx, My, Mxy]
        index_map = {
            "Nx-Ny": (0, 1),
            "Nx-Nxy": (0, 2),
            "Mx-My": (3, 4),
            "Nx-Mx": (0, 3),
        }
        idx1, idx2 = index_map[load_type]

        envelopes: dict[str, np.ndarray] = {
            c.name: np.zeros((n_points, 2), dtype=np.float64)
            for c in self.criteria
        }

        for i, theta in enumerate(angles):
            # Construct unit load in the chosen plane
            load_vec = np.zeros(6, dtype=np.float64)
            load_vec[idx1] = np.cos(theta)
            load_vec[idx2] = np.sin(theta)
            load = LoadState.from_vector(load_vec)

            # Evaluate laminate at unit load
            report = self.evaluate_laminate(laminate, load)

            for criterion in self.criteria:
                name = criterion.name
                fi_max = float(report.ply_failure_indices[name].max())
                # Strength ratio = 1 / FI  (scale at which failure occurs)
                sr = 1.0 / fi_max if fi_max > 0.0 else 1.0e12

                envelopes[name][i, 0] = sr * np.cos(theta)
                envelopes[name][i, 1] = sr * np.sin(theta)

        return envelopes

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def default_criteria(cls) -> FailureEvaluator:
        """Create an evaluator with the LaRC04/05 criterion.

        Returns an evaluator containing
        :class:`~wrinklefe.failure.larc05.LaRC05Criterion`, which is the
        primary failure criterion for wrinkle analysis as it accounts for
        fibre kinking via misalignment-frame stress rotation.

        Returns
        -------
        FailureEvaluator
            Evaluator with LaRC05 criterion.
        """
        from wrinklefe.failure.larc05 import LaRC05Criterion

        return cls([LaRC05Criterion()])
