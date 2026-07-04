"""Puck action-plane failure criterion for orthotropic composite laminates.

Implements the Puck criterion with separate fibre failure (FF) and
inter-fibre failure (IFF) evaluations.  The IFF check searches over
fracture plane angles to find the critical orientation.

Fibre Failure (FF) -- Simplified
--------------------------------
Tension (sigma_1 >= 0)::

    FI_FF = sigma_1 / Xt

Compression (sigma_1 < 0)::

    FI_FF = -sigma_1 / Xc

(The full Puck FF includes stress magnification factors from transverse
stresses; this implementation uses the simplified form.)

Inter-Fibre Failure (IFF) -- Action-Plane Search
-------------------------------------------------
The fracture plane is defined by the angle theta measured from the
2-direction.  Stresses resolved on the fracture plane::

    sigma_n(theta)  = s2*cos^2(theta) + s3*sin^2(theta) + 2*t23*sin(theta)*cos(theta)
    tau_nt(theta)   = (s3-s2)*sin(theta)*cos(theta) + t23*(cos^2(theta) - sin^2(theta))
    tau_n1(theta)   = t12*cos(theta) + t13*sin(theta)

Three IFF modes are evaluated depending on the sign and magnitude of
sigma_n:

**Mode A** (sigma_n >= 0)::

    FI_A = sqrt( (tau_nt / (S23 + p_perp_psi_t * sigma_n))^2
               + (tau_n1 / (S12 + p_perp_par_t * sigma_n))^2 )
         + p_perp_psi_t * sigma_n / S23

**Mode B** (sigma_n < 0, |tau_nt/sigma_n| >= threshold)::

    FI_B = sqrt( tau_nt^2 + (tau_n1 + p_perp_par_c * sigma_n)^2 ) / S12
         + p_perp_psi_c * sigma_n / S23

**Mode C** (sigma_n < 0, |tau_nt/sigma_n| < threshold)::

    FI_C = [ (tau_nt / (2*(1 + p_perp_psi_c)*S23))^2 + (tau_n1/S12)^2 ]
           * (-Yc / sigma_n)

The search sweeps theta from -90 deg to +90 deg in 1 deg increments and
returns the maximum IFF failure index found.

Default Puck Inclination Parameters
------------------------------------
- p_perp_par_t  = 0.30  (p_perp_parallel, tension)
- p_perp_par_c  = 0.25  (p_perp_parallel, compression)
- p_perp_psi_t  = 0.20  (p_perp_psi, tension)
- p_perp_psi_c  = 0.25  (p_perp_psi, compression)

References
----------
- Puck, A. & Schurmann, H. (1998). Composites Science and Technology, 58.
- Puck, A. & Schurmann, H. (2002). Composites Science and Technology, 62.
"""

from __future__ import annotations

import numpy as np

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.failure.base import FailureCriterion, FailureResult


class PuckCriterion(FailureCriterion):
    """Puck action-plane failure criterion for 3-D orthotropic composites.

    Evaluates both fibre failure (FF) and inter-fibre failure (IFF).
    The overall failure index is the maximum of the two.

    Parameters
    ----------
    p_perp_par_t : float
        Inclination parameter p_perp_parallel for tension (default 0.30).
    p_perp_par_c : float
        Inclination parameter p_perp_parallel for compression (default 0.25).
    p_perp_psi_t : float
        Inclination parameter p_perp_psi for tension (default 0.20).
    p_perp_psi_c : float
        Inclination parameter p_perp_psi for compression (default 0.25).
    n_theta : int
        Number of fracture-plane angles to search over [-90, 90] degrees
        (default 181, i.e. 1-degree increments).

    Attributes
    ----------
    name : str
        ``"puck"``
    """

    name = "puck"

    def __init__(
        self,
        p_perp_par_t: float = 0.30,
        p_perp_par_c: float = 0.25,
        p_perp_psi_t: float = 0.20,
        p_perp_psi_c: float = 0.25,
        n_theta: int = 181,
    ) -> None:
        self.p_perp_par_t = p_perp_par_t
        self.p_perp_par_c = p_perp_par_c
        self.p_perp_psi_t = p_perp_psi_t
        self.p_perp_psi_c = p_perp_psi_c
        self.n_theta = n_theta

    # ------------------------------------------------------------------
    # Fibre failure
    # ------------------------------------------------------------------

    @staticmethod
    def _fibre_failure(
        s1: float, Xt: float, Xc: float
    ) -> tuple[float, str]:
        """Evaluate simplified fibre failure.

        Parameters
        ----------
        s1 : float
            Fibre-direction stress sigma_11.
        Xt, Xc : float
            Tensile and compressive fibre strengths.

        Returns
        -------
        fi : float
            Fibre failure index.
        mode : str
            ``"fiber_tension"`` or ``"fiber_compression"``.
        """
        if s1 >= 0:
            return s1 / Xt, "fiber_tension"
        else:
            return -s1 / Xc, "fiber_compression"

    # ------------------------------------------------------------------
    # Inter-fibre failure (action-plane search)
    # ------------------------------------------------------------------

    def _inter_fibre_failure(
        self,
        stress_local: np.ndarray,
        material: OrthotropicMaterial,
    ) -> tuple[float, str, float]:
        """Search over fracture plane angles for maximum IFF index.

        Parameters
        ----------
        stress_local : np.ndarray
            Shape ``(6,)`` stress vector.
        material : OrthotropicMaterial
            Material strength properties.

        Returns
        -------
        fi_max : float
            Maximum inter-fibre failure index over all angles.
        mode : str
            IFF sub-mode at the critical angle (``"iff_mode_a"``,
            ``"iff_mode_b"``, or ``"iff_mode_c"``).
        theta_crit : float
            Critical fracture plane angle in radians.
        """
        s1, s2, s3, t23, t13, t12 = stress_local

        S12 = material.S12
        S23 = material.S23
        Yc = material.Yc

        p_ppt = self.p_perp_par_t
        p_ppc = self.p_perp_par_c
        p_pst = self.p_perp_psi_t
        p_psc = self.p_perp_psi_c

        thetas = np.linspace(-np.pi / 2, np.pi / 2, self.n_theta)
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)

        # Stresses on the fracture plane (vectorised over theta)
        sigma_n = s2 * cos_t**2 + s3 * sin_t**2 + 2.0 * t23 * sin_t * cos_t
        tau_nt = (s3 - s2) * sin_t * cos_t + t23 * (cos_t**2 - sin_t**2)
        tau_n1 = t12 * cos_t + t13 * sin_t

        # Pre-compute the Mode B / Mode C corner slope. Mode B and C live on
        # the compressive (sigma_n < 0) half of the action-plane failure
        # envelope, and the corner between them is at a constant geometric
        # ratio |tau_nt / sigma_n| = sqrt(1 + 2 p_psc). Above that ratio
        # (shear-dominated) Mode C governs; below it (compression-dominated)
        # Mode B governs. See issue #85 — previously the threshold simplified
        # to a constant on |tau_nt| alone, with the labels physically swapped.
        bc_corner_slope = self._mode_bc_corner_slope(p_psc)

        fi_arr = np.zeros_like(thetas)
        mode_arr = np.empty(len(thetas), dtype=object)

        for i in range(len(thetas)):
            sn = sigma_n[i]
            tnt = tau_nt[i]
            tn1 = tau_n1[i]

            # Squares are written ``x * x`` rather than ``x ** 2``:
            # ``np.float64.__pow__`` routes through libm ``pow`` and can
            # land 1 ULP away from the exact product, whereas the
            # vectorised field path's array ``** 2`` is an exact multiply
            # — the explicit product keeps the two paths bit-identical
            # (issue #299).
            if sn >= 0:
                # Mode A (tension on fracture plane)
                denom_nt = S23 + p_pst * sn
                denom_n1 = S12 + p_ppt * sn
                # Guard against denominator approaching zero
                if denom_nt <= 0 or denom_n1 <= 0:
                    fi_arr[i] = float("inf")
                else:
                    r_nt = tnt / denom_nt
                    r_n1 = tn1 / denom_n1
                    fi_arr[i] = (
                        np.sqrt(r_nt * r_nt + r_n1 * r_n1)
                        + p_pst * sn / S23
                    )
                mode_arr[i] = "iff_mode_a"
            else:
                # sigma_n < 0: distinguish Mode B vs Mode C using the
                # geometric (stress-magnitude-independent) corner slope.
                if abs(tnt) > bc_corner_slope * abs(sn):
                    # Mode C: shear-dominated, mild compression.
                    denom_c = 2.0 * (1.0 + p_psc) * S23
                    r_nt = tnt / denom_c
                    r_n1 = tn1 / S12
                    bracket = r_nt * r_nt + r_n1 * r_n1
                    fi_arr[i] = bracket * (-Yc / sn)
                    mode_arr[i] = "iff_mode_c"
                else:
                    # Mode B: compression-dominated, mild shear.
                    b1 = tn1 + p_ppc * sn
                    inner = tnt * tnt + b1 * b1
                    fi_arr[i] = np.sqrt(inner) / S12 + p_psc * sn / S23
                    mode_arr[i] = "iff_mode_b"

        idx_max = int(np.argmax(fi_arr))
        return float(fi_arr[idx_max]), str(mode_arr[idx_max]), float(thetas[idx_max])

    @staticmethod
    def _mode_bc_corner_slope(p_psc: float) -> float:
        """Slope ``|tau_nt / sigma_n|`` at the Mode B / Mode C corner.

        Following Puck & Schurmann (2002), the corner sits at a constant
        ratio derived from the action-plane strength ``RA_perp_perp`` and
        the shear stress ``tau_nt_c`` at the corner — both proportional
        to ``S23 / (1 + p_perp_psi_c)`` — leaving a stress-magnitude-
        independent slope of ``sqrt(1 + 2 p_perp_psi_c)``.

        Exposed as a static method so the Mode B/C classification can be
        unit-tested without driving a full action-plane search (the Mode C
        formula has a 1/|sigma_n| factor that dominates the argmax-over-
        theta result and would mask classification regressions). See #85.
        """
        return float(np.sqrt(1.0 + 2.0 * float(p_psc)))

    @staticmethod
    def _classify_iff_mode_bc(sn: float, tnt: float, p_psc: float) -> str:
        """Return ``"iff_mode_b"`` or ``"iff_mode_c"`` for a single plane
        with ``sigma_n < 0``.

        Mode C governs when the shear-to-compression ratio exceeds the
        Mode B/C corner slope; Mode B governs below it. Pure helper for
        tests; production code uses the inline check in
        :meth:`_inter_fibre_failure` for vectorisation. See #85.
        """
        slope = PuckCriterion._mode_bc_corner_slope(p_psc)
        return "iff_mode_c" if abs(tnt) > slope * abs(sn) else "iff_mode_b"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        stress_local: np.ndarray,
        material: OrthotropicMaterial,
        context=None,
    ) -> FailureResult:
        """Evaluate the Puck criterion at a single material point.

        The overall failure index is the maximum of fibre failure (FF) and
        inter-fibre failure (IFF).

        Parameters
        ----------
        stress_local : np.ndarray
            Shape ``(6,)`` stress vector in local material coordinates.
        material : OrthotropicMaterial
            Material with strength allowables and inclination parameters.

        Returns
        -------
        FailureResult
            Contains the governing failure index, mode, reserve factor,
            and criterion name.
        """
        stress_local = np.asarray(stress_local, dtype=np.float64)

        fi_ff, mode_ff = self._fibre_failure(
            stress_local[0], material.Xt, material.Xc
        )
        fi_iff, mode_iff, _ = self._inter_fibre_failure(stress_local, material)

        if fi_ff >= fi_iff:
            fi, mode = fi_ff, mode_ff
        else:
            fi, mode = fi_iff, mode_iff

        rf = 1.0 / fi if fi > 0 else float("inf")

        return FailureResult(
            index=fi,
            mode=mode,
            reserve_factor=rf,
            criterion_name=self.name,
        )

    # ------------------------------------------------------------------
    # Vectorised field evaluation (issue #299)
    # ------------------------------------------------------------------

    _IFF_MODE_LABELS = np.array(
        ["iff_mode_a", "iff_mode_b", "iff_mode_c"], dtype="U32"
    )

    # Rows per block for the (N, n_theta) action-plane grid. Small enough
    # that a block's ~10 temporaries (block x n_theta float64) stay cache
    # resident — measured ~3x faster than materialising the full grid on
    # an 80k-point field.
    _FIELD_CHUNK = 2048

    def evaluate_field(
        self,
        stress_field: np.ndarray,
        material: OrthotropicMaterial,
        contexts=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised Puck evaluation across an array of stress states.

        Broadcasts the action-plane search over a ``(N, n_theta)`` grid —
        the fracture-plane resolution of stresses is pure elementwise
        math — and takes the per-point argmax over the angle axis,
        reproducing per-point :meth:`evaluate` exactly (same expressions,
        same first-maximum tie-breaking). Rows are processed in blocks of
        :attr:`_FIELD_CHUNK` so the angle-grid temporaries stay cache
        resident; chunking does not change any per-element arithmetic.

        Parameters
        ----------
        stress_field : np.ndarray
            Shape ``(N, 6)`` array of local stress vectors.
        material : OrthotropicMaterial
            Material with strength allowables; shared by all N points.
        contexts : list, optional
            Ignored — Puck has no context dependence.

        Returns
        -------
        indices, modes, reserve_factors : np.ndarray
            Shape ``(N,)`` arrays matching :meth:`evaluate` per point.
        """
        s = np.asarray(stress_field, dtype=np.float64)
        if s.ndim != 2 or s.shape[1] != 6:
            raise ValueError(
                f"stress_field must have shape (N, 6), got {s.shape}"
            )
        n = s.shape[0]
        indices = np.empty(n, dtype=np.float64)
        modes = np.empty(n, dtype="U32")
        reserve_factors = np.empty(n, dtype=np.float64)
        for start in range(0, n, self._FIELD_CHUNK):
            block = slice(start, min(start + self._FIELD_CHUNK, n))
            fi_b, mode_b, rf_b = self._evaluate_field_block(s[block], material)
            indices[block] = fi_b
            modes[block] = mode_b
            reserve_factors[block] = rf_b
        return indices, modes, reserve_factors

    def _evaluate_field_block(
        self,
        s: np.ndarray,
        material: OrthotropicMaterial,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One cache-sized block of the vectorised evaluation."""
        n = s.shape[0]
        s1 = s[:, 0]
        s2 = s[:, 1][:, None]
        s3 = s[:, 2][:, None]
        t23 = s[:, 3][:, None]
        t13 = s[:, 4][:, None]
        t12 = s[:, 5][:, None]

        S12 = material.S12
        S23 = material.S23
        Yc = material.Yc
        p_ppt = self.p_perp_par_t
        p_ppc = self.p_perp_par_c
        p_pst = self.p_perp_psi_t
        p_psc = self.p_perp_psi_c

        thetas = np.linspace(-np.pi / 2, np.pi / 2, self.n_theta)
        cos_t = np.cos(thetas)[None, :]
        sin_t = np.sin(thetas)[None, :]

        # Fracture-plane stresses, (N, n_theta). Same expressions as the
        # scalar path so results agree bitwise.
        sigma_n = s2 * cos_t**2 + s3 * sin_t**2 + 2.0 * t23 * sin_t * cos_t
        tau_nt = (s3 - s2) * sin_t * cos_t + t23 * (cos_t**2 - sin_t**2)
        tau_n1 = t12 * cos_t + t13 * sin_t

        bc_corner_slope = self._mode_bc_corner_slope(p_psc)

        tension_plane = sigma_n >= 0
        mode_c_mask = ~tension_plane & (
            np.abs(tau_nt) > bc_corner_slope * np.abs(sigma_n)
        )
        mode_b_mask = ~tension_plane & ~mode_c_mask

        fi_grid = np.zeros_like(sigma_n)
        with np.errstate(divide="ignore", invalid="ignore"):
            # Mode A (tension on the fracture plane)
            denom_nt = S23 + p_pst * sigma_n
            denom_n1 = S12 + p_ppt * sigma_n
            fi_a = (
                np.sqrt((tau_nt / denom_nt) ** 2 + (tau_n1 / denom_n1) ** 2)
                + p_pst * sigma_n / S23
            )
            bad_a = tension_plane & ((denom_nt <= 0) | (denom_n1 <= 0))
            np.copyto(fi_grid, fi_a, where=tension_plane)
            fi_grid[bad_a] = np.inf

            # Mode C (shear-dominated, mild compression)
            denom_c = 2.0 * (1.0 + p_psc) * S23
            bracket = (tau_nt / denom_c) ** 2 + (tau_n1 / S12) ** 2
            np.copyto(fi_grid, bracket * (-Yc / sigma_n), where=mode_c_mask)

            # Mode B (compression-dominated, mild shear)
            inner = tau_nt**2 + (tau_n1 + p_ppc * sigma_n) ** 2
            np.copyto(
                fi_grid,
                np.sqrt(inner) / S12 + p_psc * sigma_n / S23,
                where=mode_b_mask,
            )

        rows = np.arange(n)
        idx_max = np.argmax(fi_grid, axis=1)
        fi_iff = fi_grid[rows, idx_max]
        mode_code = np.where(
            tension_plane[rows, idx_max],
            0,
            np.where(mode_c_mask[rows, idx_max], 2, 1),
        )
        modes_iff = self._IFF_MODE_LABELS[mode_code]

        # Fibre failure (simplified): tension vs compression on sigma_11.
        fi_ff = np.where(s1 >= 0, s1 / material.Xt, -s1 / material.Xc)
        modes_ff = np.where(
            s1 >= 0, "fiber_tension", "fiber_compression"
        ).astype("U32")

        ff_governs = fi_ff >= fi_iff
        indices = np.where(ff_governs, fi_ff, fi_iff)
        modes = np.where(ff_governs, modes_ff, modes_iff).astype("U32")

        with np.errstate(divide="ignore"):
            reserve_factors = np.where(
                indices > 0,
                1.0 / np.where(indices > 0, indices, 1.0),
                np.inf,
            )
        return indices, modes, reserve_factors
