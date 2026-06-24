"""LaRC04/05 failure criterion for orthotropic composite laminates.

Implements the NASA Langley Research Center failure criteria (Dávila,
Camanho, Pinho 2005) with:

- **Fibre kinking** under compression via misalignment-frame rotation
  with a closed-form load-induced φ_c rotation
- **Fibre tension** with fibre-matrix shear interaction
- **Matrix failure** via fracture-plane search (Mohr-Coulomb)
- **In-situ strength** corrections (fracture-toughness-based when GIc/GIIc
  are available, simplified 1.12√2 fallback otherwise)

The fibre kinking sub-criterion is the key connection to wrinkle modelling:
it reads the per-element fibre misalignment angle from ``context['misalignment_angle']``
and rotates the stress state into the kink-band frame.

Friction Coefficients
---------------------
Derived from the fracture plane angle α₀ (typically 53° for CFRP)::

    μ_L = -S_L · cos(2α₀) / (Y_c · cos²(α₀))
    μ_T = -1 / tan(2α₀)

Load-induced φ_c rotation
-------------------------
Under compression the misaligned fibres rotate further before kinking.
The additional rotation is taken from the Dávila & Camanho linear
closed form (NOT an iterative scheme)::

    φ_c = (|σ₁| - σ₂) · φ₀ / (G₁₂ + |σ₁| - σ₂)      (clamped to [0, φ₀])

For wrinkle problems φ₀ is the dominant angle read from the geometry, so
this small load-induced correction is evaluated once, with no iteration
instability. A Ramberg-Osgood nonlinear-shear refinement
(``γ_12 = τ_12 / G_12 + β · τ_12³``) is provided as a helper but is **not**
applied to the kinking failure index in the current implementation, which
uses the linear closed-form φ_c above.

References
----------
- Pinho, S. T., Dávila, C. G., Camanho, P. P., Iannucci, L., & Robinson, P.
  (2005). NASA/TM-2005-213530. "Failure models and criteria for FRP under
  in-plane or three-dimensional stress states including shear non-linearity."
- Dávila, C. G., Camanho, P. P., & Rose, C. A. (2005). J. Composite
  Materials, 39(4). "Failure criteria for FRP laminates."
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.failure.base import FailureCriterion, FailureResult


class LaRC05Criterion(FailureCriterion):
    """LaRC04/05 failure criterion for 3-D orthotropic composites.

    Parameters
    ----------
    ply_thickness : float
        Ply thickness in mm for in-situ correction (default 0.183).
    t_ref : float
        Reference ply thickness in mm (default 0.183).  Plies thinner
        than ``2 * t_ref`` use thin-ply in-situ strengths.
    n_theta : int
        Number of fracture-plane angles to search (default 181).
    max_phi_c_iter : int
        Reserved for API compatibility (default 20). The current φ_c is a
        single linear closed-form evaluation, so this iteration cap is not
        used.
    phi_c_tol : float
        Reserved for API compatibility (radians, default 1e-6). Unused
        while φ_c is computed in closed form (see ``_compute_phi_c``).
    """

    name = "larc05"

    def __init__(
        self,
        ply_thickness: float = 0.183,
        t_ref: float = 0.183,
        n_theta: int = 181,
        max_phi_c_iter: int = 20,
        phi_c_tol: float = 1e-6,
    ) -> None:
        self.ply_thickness = ply_thickness
        self.t_ref = t_ref
        self.n_theta = n_theta
        self.max_phi_c_iter = max_phi_c_iter
        self.phi_c_tol = phi_c_tol

    # ------------------------------------------------------------------
    # Friction coefficients from α₀
    # ------------------------------------------------------------------

    @staticmethod
    def _friction_coefficients(material: OrthotropicMaterial) -> tuple[float, float]:
        """Compute Mohr-Coulomb friction coefficients from fracture angle.

        Parameters
        ----------
        material : OrthotropicMaterial
            Must have ``alpha_0`` (degrees), ``S12``, ``Yc``, ``S23``.

        Returns
        -------
        mu_L : float
            Longitudinal friction coefficient.
        mu_T : float
            Transverse friction coefficient.
        """
        alpha_0_rad = np.radians(material.alpha_0)
        cos_2a = np.cos(2.0 * alpha_0_rad)
        cos_a2 = np.cos(alpha_0_rad) ** 2
        tan_2a = np.tan(2.0 * alpha_0_rad)

        # Friction coefficients (Pinho et al. 2005, Eq. 12-13)
        mu_T = -1.0 / tan_2a
        mu_L = -material.S12 * cos_2a / (material.Yc * cos_a2)

        # Clamp to physically reasonable range
        mu_L = max(0.0, min(mu_L, 1.0))
        mu_T = max(0.0, min(mu_T, 1.0))

        return mu_L, mu_T

    # ------------------------------------------------------------------
    # In-situ strengths
    # ------------------------------------------------------------------

    def _in_situ_strengths(
        self,
        material: OrthotropicMaterial,
        ply_thickness: float | None = None,
    ) -> tuple[float, float]:
        """Compute in-situ transverse tensile and shear strengths.

        Uses fracture-toughness-based corrections when GIc/GIIc are
        available; falls back to simplified 1.12√2 factors otherwise.

        Parameters
        ----------
        material : OrthotropicMaterial
            Material with elastic and fracture-toughness properties.
        ply_thickness : float, optional
            Effective ply thickness (mm) to use for the in-situ
            correction.  If ``None`` (the default), falls back to the
            instance attribute ``self.ply_thickness``.  Passing this
            explicitly lets a per-element override be threaded through
            ``evaluate`` without mutating instance state (see #192).

        Returns
        -------
        Yt_is : float
            In-situ transverse tensile strength (MPa).
        S12_is : float
            In-situ in-plane shear strength (MPa).
        """
        t = self.ply_thickness if ply_thickness is None else ply_thickness
        is_thin = t < 2.0 * self.t_ref

        if material.GIc is not None and material.GIIc is not None and is_thin:
            # Fracture-toughness-based (Camanho et al. 2006)
            # Lambda_22 = 2 * (1/E2 - nu21^2/E1)
            nu21 = material.nu12 * material.E2 / material.E1
            Lambda_22 = 2.0 * (1.0 / material.E2 - nu21 ** 2 / material.E1)
            Lambda_44 = 1.0 / material.G12

            Yt_is = np.sqrt(8.0 * material.GIc / (np.pi * t * Lambda_22))
            S12_is = np.sqrt(8.0 * material.GIIc / (np.pi * t * Lambda_44))

            # Ensure in-situ >= unconstrained
            Yt_is = max(Yt_is, material.Yt)
            S12_is = max(S12_is, material.S12)
        elif is_thin:
            # Simplified thin-ply corrections
            Yt_is = 1.12 * np.sqrt(2.0) * material.Yt
            S12_is = np.sqrt(2.0) * material.S12
        else:
            # Thick ply — no correction
            Yt_is = material.Yt
            S12_is = material.S12

        return Yt_is, S12_is

    # ------------------------------------------------------------------
    # Nonlinear shear
    # ------------------------------------------------------------------

    @staticmethod
    def _nonlinear_shear_strain(tau: float, G12: float, beta: float) -> float:
        """Ramberg-Osgood nonlinear shear strain.

        γ₁₂ = τ₁₂/G₁₂ + β·τ₁₂³
        """
        return tau / G12 + beta * tau ** 3

    @staticmethod
    def _effective_shear_stress(gamma: float, G12: float) -> float:
        """Effective linear-equivalent shear stress from nonlinear strain.

        τ_eff = G₁₂ · γ₁₂  (secant modulus approach)
        """
        return G12 * gamma

    # ------------------------------------------------------------------
    # Fibre tension with shear interaction
    # ------------------------------------------------------------------

    @staticmethod
    def _fibre_tension(stress: np.ndarray, material: OrthotropicMaterial) -> float:
        """Fibre tensile failure index with fibre-matrix shear interaction.

        FI = (σ₁/Xt)² + (τ₁₂/S₁₂)² + (τ₁₃/S₁₃)²

        The quadratic shear interaction accounts for shear-driven fibre
        splitting under combined tension + shear loading.
        """
        s1, s2, s3, t23, t13, t12 = stress
        fi = (s1 / material.Xt) ** 2 + (t12 / material.S12) ** 2
        if material.S13 > 0:
            fi += (t13 / material.S13) ** 2
        return float(np.sqrt(max(fi, 0.0)))

    # ------------------------------------------------------------------
    # Fibre kinking with misalignment frame + nonlinear shear
    # ------------------------------------------------------------------

    def _compute_phi_c(
        self,
        s1: float, s2: float, t12: float,
        phi_0: float,
        G12: float, beta: float,
    ) -> float:
        """Compute load-induced additional misalignment φ_c.

        Uses the linear closed-form approximation from Dávila & Camanho::

            φ_c = (|σ₁| - σ₂) · φ₀ / (G₁₂ + |σ₁| - σ₂)

        For wrinkle problems where φ₀ is the dominant angle from the
        geometry, this small correction captures the load-induced
        amplification without iteration instability.

        Returns
        -------
        phi_c : float
            Additional rotation (radians), clamped to [0, φ₀].
        """
        if phi_0 == 0.0 or s1 >= 0:
            return 0.0

        delta_s = abs(s1) - s2
        if delta_s <= 0:
            return 0.0

        phi_c = delta_s * phi_0 / (G12 + delta_s)

        # Clamp: φ_c should not exceed φ₀ (physically, the load-induced
        # rotation cannot be larger than the initial imperfection for
        # stable kinking)
        return min(phi_c, phi_0)

    def _fibre_kinking(
        self,
        stress: np.ndarray,
        material: OrthotropicMaterial,
        Yt_is: float,
        S12_is: float,
        phi_0: float,
    ) -> float:
        """Evaluate fibre kinking criterion under compression.

        Rotates the 3D stress state by the total misalignment angle
        (φ₀ + φ_c) and evaluates matrix failure in the kink band.

        Parameters
        ----------
        stress : np.ndarray
            Shape ``(6,)`` stress vector.
        material : OrthotropicMaterial
            Material with elastic and strength properties.
        Yt_is, S12_is : float
            In-situ strengths.
        phi_0 : float
            Initial fibre misalignment from wrinkle geometry (radians).

        Returns
        -------
        float
            Fibre kinking failure index.
        """
        s1, s2, s3, t23, t13, t12 = stress

        # Compute load-induced rotation
        phi_c = self._compute_phi_c(
            s1, s2, t12, phi_0, material.G12, material.beta_shear
        )
        phi = phi_0 + phi_c

        cos_p = np.cos(phi)
        sin_p = np.sin(phi)

        # 3D stress rotation into misalignment (kink-band) frame
        # In-plane rotation by φ about the 3-axis
        sigma_22m = s2 * cos_p ** 2 + s1 * sin_p ** 2 - 2.0 * t12 * sin_p * cos_p
        tau_12m = (s1 - s2) * sin_p * cos_p + t12 * (cos_p ** 2 - sin_p ** 2)

        # Out-of-plane shear components in kink frame
        tau_23m = t23 * cos_p - t13 * sin_p

        # NOTE: The nonlinear shear (Ramberg-Osgood) is used only in the
        # phi_c computation above. The matrix failure criterion in the kink
        # frame uses the actual rotated stress values, not amplified ones.
        # This follows Pinho et al. 2005: the nonlinearity affects the
        # equilibrium rotation, but FI is evaluated with the stress tensor.

        # Friction coefficients
        mu_L, mu_T = self._friction_coefficients(material)

        if sigma_22m >= 0:
            # Tensile transverse stress in kink band
            fi = (
                (tau_12m / S12_is) ** 2
                + (sigma_22m / Yt_is) ** 2
                + (tau_23m / material.S23) ** 2
            )
        else:
            # Compressive — Mohr-Coulomb friction model
            denom_l = S12_is + mu_L * abs(sigma_22m)
            denom_t = material.S23 + mu_T * abs(sigma_22m)
            if denom_l <= 0 or denom_t <= 0:
                return float("inf")
            fi = (tau_12m / denom_l) ** 2 + (tau_23m / denom_t) ** 2

        # Return linear-in-load FI (sqrt of the quadratic form) so it can
        # be compared directly against the other sub-criteria — see #79.
        return float(np.sqrt(max(fi, 0.0)))

    # ------------------------------------------------------------------
    # Matrix failure (fracture-plane search)
    # ------------------------------------------------------------------

    def _matrix_failure(
        self,
        stress: np.ndarray,
        material: OrthotropicMaterial,
        Yt_is: float,
        S12_is: float,
    ) -> tuple[float, str]:
        """Search fracture plane angles for maximum matrix failure index.

        Uses vectorised evaluation over all candidate fracture planes.

        Returns
        -------
        fi_max : float
            Maximum matrix failure index.
        mode : str
            ``"matrix_tension"`` or ``"matrix_compression"``.
        """
        s1, s2, s3, t23, t13, t12 = stress

        mu_L, mu_T = self._friction_coefficients(material)

        # Transverse shear strength on fracture plane
        alpha_0_rad = np.radians(material.alpha_0)
        tan_2a = np.tan(2.0 * alpha_0_rad)
        S_T = material.Yc * np.cos(alpha_0_rad) * (
            np.sin(alpha_0_rad) + np.cos(alpha_0_rad) / tan_2a
        )

        S_L = S12_is

        thetas = np.linspace(-np.pi / 2, np.pi / 2, self.n_theta)
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)

        # Stresses on candidate fracture planes
        sigma_n = s2 * cos_t ** 2 + s3 * sin_t ** 2 + 2.0 * t23 * sin_t * cos_t
        tau_nt = (s3 - s2) * sin_t * cos_t + t23 * (cos_t ** 2 - sin_t ** 2)
        tau_n1 = t12 * cos_t + t13 * sin_t

        fi_arr = np.zeros(len(thetas))

        for i in range(len(thetas)):
            sn = sigma_n[i]
            tnt = tau_nt[i]
            tn1 = tau_n1[i]

            if sn >= 0:
                # Tensile fracture plane
                fi_arr[i] = (
                    (tnt / S_T) ** 2
                    + (tn1 / S_L) ** 2
                    + (sn / Yt_is) ** 2
                )
            else:
                # Compressive fracture plane with friction
                denom_t = S_T + mu_T * abs(sn)
                denom_l = S_L + mu_L * abs(sn)
                if denom_t <= 0 or denom_l <= 0:
                    fi_arr[i] = float("inf")
                else:
                    fi_arr[i] = (tnt / denom_t) ** 2 + (tn1 / denom_l) ** 2

        idx_max = int(np.argmax(fi_arr))
        # Return linear-in-load FI (sqrt of the quadratic form) so it can
        # be compared directly against the other sub-criteria — see #79.
        fi_max = float(np.sqrt(max(fi_arr[idx_max], 0.0)))
        mode = "matrix_tension" if sigma_n[idx_max] >= 0 else "matrix_compression"
        return fi_max, mode

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        stress_local: np.ndarray,
        material: OrthotropicMaterial,
        context: dict[str, Any] | None = None,
    ) -> FailureResult:
        """Evaluate the LaRC04/05 criterion at a single material point.

        Parameters
        ----------
        stress_local : np.ndarray
            Shape ``(6,)`` stress vector in local material coordinates.
        material : OrthotropicMaterial
            Material with strength, elastic, and LaRC properties.
        context : dict, optional
            Element-level data.  Recognised keys:

            - ``'misalignment_angle'`` (float): fibre misalignment φ₀
              in radians from the wrinkle geometry model.
            - ``'ply_thickness'`` (float): overrides instance ply_thickness.

        Returns
        -------
        FailureResult
        """
        stress_local = np.asarray(stress_local, dtype=np.float64)
        s1 = stress_local[0]

        # Extract context.  Treat the ply-thickness override as a local
        # value so concurrent / interleaved evaluate() calls do not
        # corrupt the instance state (see #192).
        phi_0 = 0.0
        t_eff = self.ply_thickness
        if context is not None:
            phi_0 = context.get("misalignment_angle", 0.0)
            t_override = context.get("ply_thickness", None)
            if t_override is not None:
                t_eff = t_override

        Yt_is, S12_is = self._in_situ_strengths(material, t_eff)

        # --- Fibre failure ---
        if s1 >= 0:
            fi_fiber = self._fibre_tension(stress_local, material)
            mode_fiber = "fiber_tension"
        else:
            fi_fiber = self._fibre_kinking(
                stress_local, material, Yt_is, S12_is, phi_0
            )
            mode_fiber = "fiber_kinking"

        # --- Matrix failure ---
        fi_matrix, mode_matrix = self._matrix_failure(
            stress_local, material, Yt_is, S12_is
        )

        # --- Governing criterion ---
        if fi_fiber >= fi_matrix:
            fi, mode = fi_fiber, mode_fiber
        else:
            fi, mode = fi_matrix, mode_matrix

        # Reserve factor. Every sub-FI is now on the linear-in-load scale
        # (see #79), so rf = 1 / fi holds uniformly.
        rf = 1.0 / fi if fi > 0 else float("inf")

        # Sub-criterion detail for diagnostics
        phi_c = self._compute_phi_c(
            s1, stress_local[1], stress_local[5],
            phi_0, material.G12, material.beta_shear
        ) if s1 < 0 else 0.0

        detail = {
            "fi_fiber": float(fi_fiber),
            "fi_matrix": float(fi_matrix),
            "mode_fiber": mode_fiber,
            "mode_matrix": mode_matrix,
            "phi_0": phi_0,
            "phi_c": phi_c,
        }

        return FailureResult(
            index=float(fi),
            mode=mode,
            reserve_factor=float(rf),
            criterion_name=self.name,
            detail=detail,
        )
