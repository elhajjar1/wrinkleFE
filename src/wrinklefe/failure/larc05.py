"""LaRC04/05 failure criterion for orthotropic composite laminates.

Implements the NASA Langley Research Center failure criteria (Dávila,
Camanho, Pinho 2005) with:

- **Fibre kinking** under compression: LaRC04/05 kink-band model with the
  Xc-calibrated intrinsic misalignment and a kink-plane search
- **Fibre tension** with fibre-matrix shear interaction
- **Matrix failure** via fracture-plane search (Mohr-Coulomb)
- **In-situ strength** corrections (fracture-toughness-based when GIc/GIIc
  are available, simplified 1.12√2 fallback otherwise)

The fibre kinking sub-criterion is the key connection to wrinkle modelling.
A wrinkle reaches it through the stress state itself: the FE evaluates
failure on stresses already rotated into the local (wrinkled) fibre frame,
where the out-of-plane misalignment appears as ``τ₁₃``. The kink-plane
search below finds that plane. ``context['misalignment_angle']`` is an
*additional* initial imperfection, for callers whose stresses are not in
the misaligned fibre frame (CLT ply stresses with a known misalignment).

Friction Coefficients
---------------------
Derived from the fracture plane angle α₀ (typically 53° for CFRP)::

    μ_L = -S_L · cos(2α₀) / (Y_c · cos²(α₀))
    μ_T = -1 / tan(2α₀)

Fibre kinking (LaRC04/05)
-------------------------
Follows Dávila, Camanho & Rose (2005, LaRC03) and Pinho et al. (2005,
LaRC04; 2012, LaRC05):

1. **Intrinsic misalignment** ``φ_C``, the angle that makes a pristine ply
   kink exactly at ``σ₁₁ = −X_c`` (no fitting parameter: it follows from
   ``X_c``, ``S_L`` and ``η_L``)::

       φ_C = arctan[ (1 − √(1 − 4 (S_L/X_c + η_L) S_L/X_c)) / (2 (S_L/X_c + η_L)) ]

2. **Kink-plane search** over ``ψ`` (rotation about the fibre axis), plus
   the plane of maximum fibre-direction shear ``ψ* = atan2(τ₁₃, τ₁₂)``,
   which is always included exactly. A ψ = 90° plane is how an
   out-of-plane wrinkle's ``τ₁₃`` drives kinking.
3. **Misalignment under load** in each plane (LaRC03 closed form)::

       φ = (|τ₁₂ψ| + (G₁₂ − X_c) φ_C + G₁₂ φ_extra) / (G₁₂ + σ₁₁ − σ₂ψ)

   with ``φ_extra`` the optional ``misalignment_angle``. A non-positive
   denominator is the shear instability itself, and ``|φ| ≥ 45°`` is
   outside the closed form's small-angle range; both mean the band has
   already kinked. There the index is reported as ``1 / R``, with ``R``
   the exact load multiple at which kinking first occurs (``R ≤ 1``), so
   it is finite and at least 1.
4. **Matrix failure in the misalignment frame**, the existing
   Mohr-Coulomb form, maximised over ψ.

At ``σ₁₁ = −X_c`` with no other stress, step 3 gives ``φ = φ_C`` and
step 4 gives ``FI = 1`` exactly, for any material. The previous model had
no ``φ_C``: a pristine ply could never kink (``FI = 0`` at ``−X_c``), and
the FE path rotated already-rotated stresses by the wrinkle angle a second
time, in the in-plane (1-2) plane. If ``φ_C`` has no real solution (a
material whose shear strength is high relative to ``X_c``, e.g. neat
resin), kinking cannot be calibrated to ``X_c`` and the fibre-compression
index falls back to ``|σ₁₁| / X_c``.

The kinking index is not exactly linear in load (``φ`` depends on the
stress), as in the published criterion, so ``1 / FI`` is a first-order
reserve factor for this mode.

References
----------
- Pinho, S. T., Dávila, C. G., Camanho, P. P., Iannucci, L., & Robinson, P.
  (2005). NASA/TM-2005-213530. "Failure models and criteria for FRP under
  in-plane or three-dimensional stress states including shear non-linearity."
- Dávila, C. G., Camanho, P. P., & Rose, C. A. (2005). J. Composite
  Materials, 39(4). "Failure criteria for FRP laminates."
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.failure.base import FailureCriterion, FailureResult

#: Misalignment beyond which the kink band is taken to have formed (the
#: index is then ``1 / reserve factor``; see the module docstring). The
#: LaRC03 closed form is a small-angle expression; past 45 degrees its
#: rotated stresses stop increasing with load.
_PHI_KINKED = np.pi / 4.0


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
        Reserved for API compatibility (default 20); unused. The kinking
        misalignment is a closed-form evaluation.
    phi_c_tol : float
        Reserved for API compatibility (radians, default 1e-6); unused.
    n_psi : int
        Kink-plane angles searched over ``[0, π)`` (default 36, i.e. 5°
        steps). The plane of maximum fibre-direction shear is always
        evaluated exactly in addition, so the dominant plane is never
        missed between grid points.
    """

    name = "larc05"

    def __init__(
        self,
        ply_thickness: float = 0.183,
        t_ref: float = 0.183,
        n_theta: int = 181,
        max_phi_c_iter: int = 20,
        phi_c_tol: float = 1e-6,
        n_psi: int = 36,
    ) -> None:
        self.ply_thickness = ply_thickness
        self.t_ref = t_ref
        self.n_theta = n_theta
        self.max_phi_c_iter = max_phi_c_iter
        self.phi_c_tol = phi_c_tol
        self.n_psi = n_psi

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
        # Squares as ``x * x`` (not ``x ** 2``): scalar np.float64 pow
        # routes through libm and can be 1 ULP off the exact product the
        # vectorised field path computes — the explicit multiply keeps
        # evaluate() and evaluate_field() bit-identical (issue #299).
        r1 = s1 / material.Xt
        r12 = t12 / material.S12
        fi = r1 * r1 + r12 * r12
        if material.S13 > 0:
            r13 = t13 / material.S13
            fi += r13 * r13
        return float(np.sqrt(max(fi, 0.0)))

    # ------------------------------------------------------------------
    # Fibre kinking (LaRC04/05 kink-band model)
    # ------------------------------------------------------------------

    def intrinsic_misalignment(
        self,
        material: OrthotropicMaterial,
        ply_thickness: float | None = None,
    ) -> float | None:
        """The intrinsic fibre misalignment ``φ_C`` (radians), or ``None``.

        The angle at which a pristine ply loaded in pure fibre compression
        kinks exactly at ``σ₁₁ = −X_c``. ``None`` when no real angle does
        (shear strength high relative to ``X_c``); kinking then falls back
        to ``|σ₁₁| / X_c``. Uses the same (in-situ) longitudinal shear
        strength as the kink-band check, which is what makes the anchoring
        exact.
        """
        _, S12_is = self._in_situ_strengths(material, ply_thickness)
        phi_c = self._phi_c_array(material, np.array([S12_is]))
        return None if np.isnan(phi_c[0]) else float(phi_c[0])

    def _phi_c_array(
        self, material: OrthotropicMaterial, S12_is: np.ndarray
    ) -> np.ndarray:
        """Vectorised ``φ_C`` per point; NaN where it has no real value."""
        mu_L, _ = self._friction_coefficients(material)
        ratio = S12_is / material.Xc
        a = ratio + mu_L
        disc = 1.0 - 4.0 * a * ratio
        with np.errstate(invalid="ignore", divide="ignore"):
            phi_c = np.arctan((1.0 - np.sqrt(disc)) / (2.0 * a))
        return np.where(disc >= 0.0, phi_c, np.nan)

    def _kink_fi_planes(
        self,
        s: np.ndarray,
        material: OrthotropicMaterial,
        phi_extra: np.ndarray,
        Yt_is: np.ndarray,
        S12_is: np.ndarray,
        phi_c0: np.ndarray,
        psi: np.ndarray,
    ) -> np.ndarray:
        """Kink-band index on given planes: ``psi`` is ``(N, K)``."""
        s1 = s[:, 0][:, None]
        s2, s3 = s[:, 1][:, None], s[:, 2][:, None]
        t23, t13, t12 = s[:, 3][:, None], s[:, 4][:, None], s[:, 5][:, None]
        G12, Xc = material.G12, material.Xc
        mu_L, mu_T = self._friction_coefficients(material)

        cs, sn = np.cos(psi), np.sin(psi)
        cs2, sn2 = cs * cs, sn * sn
        sig2p = s2 * cs2 + s3 * sn2 + 2.0 * t23 * sn * cs
        tau23p = (s3 - s2) * sn * cs + t23 * (cs2 - sn2)
        tau12p = t12 * cs + t13 * sn
        tau13p = t13 * cs - t12 * sn

        # Misalignment under load (LaRC03 closed form). The initial
        # imperfection's sign is not known, so both are evaluated and the
        # worse kept: phi = (tau_12psi + s0 * G12 * phi_0) / den, s0 = +-1.
        # When the shear is large the worse sign is the one that amplifies
        # it (the published choice); where tau_12psi crosses zero, fixing
        # the sign to sign(tau_12psi) made the index jump with the plane
        # angle, which no plane search can maximise reliably.
        imperfection = (G12 - Xc) * phi_c0[:, None] + G12 * phi_extra[:, None]
        den = G12 + s1 - sig2p
        den_ok = np.where(den <= 0.0, 1.0, den)
        S12c, Ytc = S12_is[:, None], Yt_is[:, None]
        fi = np.full(np.broadcast_shapes(tau12p.shape, den.shape), -np.inf)
        for sign0 in (1.0, -1.0):
            phi = (tau12p + sign0 * imperfection) / den_ok
            # Two ways the band has already kinked: the shear instability
            # itself (den <= 0), or a rotation past the small-angle range
            # the closed form is valid in. Without the second guard the
            # rotated stresses wrap round and an element loaded far past
            # failure (phi ~ 76 deg at 5 x strength) reports FI < 1.
            unstable = (den <= 0.0) | (np.abs(phi) >= _PHI_KINKED)
            cp, sp = np.cos(phi), np.sin(phi)
            cp2, sp2 = cp * cp, sp * sp
            sig2m = s1 * sp2 + sig2p * cp2 - 2.0 * tau12p * sp * cp
            tau12m = -(s1 - sig2p) * sp * cp + tau12p * (cp2 - sp2)
            tau23m = tau23p * cp - tau13p * sp
            with np.errstate(divide="ignore", invalid="ignore"):
                fi_t = (
                    (tau12m / S12c) ** 2
                    + (sig2m / Ytc) ** 2
                    + (tau23m / material.S23) ** 2
                )
                denom_l = S12c + mu_L * np.abs(sig2m)
                denom_t = material.S23 + mu_T * np.abs(sig2m)
                fi_c = (tau12m / denom_l) ** 2 + (tau23m / denom_t) ** 2
            tension_band = sig2m >= 0.0
            fi_s = np.sqrt(
                np.maximum(np.where(tension_band, fi_t, fi_c), 0.0)
            )
            bad = unstable | (
                ~tension_band & ((denom_l <= 0) | (denom_t <= 0))
            )
            fi = np.maximum(fi, np.where(bad, np.inf, fi_s))
        out: np.ndarray = fi
        return out

    #: Golden-section steps refining the critical kink plane. Each step
    #: shrinks the bracket (one coarse grid spacing either side) by 0.618;
    #: 20 steps resolve the plane to ~1e-5 rad, where the index error (it
    #: is quadratic in the plane offset at the maximum) is ~1e-9.
    _PSI_REFINE_ITERS = 20

    def _kink_fi(
        self,
        s: np.ndarray,
        material: OrthotropicMaterial,
        phi_extra: np.ndarray,
        Yt_is: np.ndarray,
        S12_is: np.ndarray,
    ) -> np.ndarray:
        """Fibre-kinking index for ``(N, 6)`` local stresses.

        Shared by :meth:`evaluate` (N = 1) and the field path, so the two
        are bit-identical by construction. Meaningful for ``σ₁₁ < 0``;
        callers select it there. See the module docstring for the model.

        Always finite. Past the kinking instability the closed form has no
        value (:meth:`_kink_fi_raw` returns ``inf`` there), so those rows
        report ``1 / R`` with ``R`` the exact reserve factor: at least 1,
        and linear in a proportional load like a linear criterion's index.
        An ``inf`` would otherwise be dropped by every consumer that
        filters non-finite values, hiding the worst point in a field.
        """
        fi = self._kink_fi_raw(s, material, phi_extra, Yt_is, S12_is)
        kinked = np.isinf(fi)
        if kinked.any():
            rf = self._kink_reserve(
                s[kinked], material, phi_extra[kinked], Yt_is[kinked],
                S12_is[kinked],
            )
            fi = fi.copy()
            fi[kinked] = 1.0 / rf
        return fi

    def _kink_fi_raw(
        self,
        s: np.ndarray,
        material: OrthotropicMaterial,
        phi_extra: np.ndarray,
        Yt_is: np.ndarray,
        S12_is: np.ndarray,
    ) -> np.ndarray:
        """Kinking index, ``inf`` where the band has already kinked.

        The critical kink plane is found in two stages: a coarse grid over
        ``[0, π)`` plus the plane of maximum fibre-direction shear, then a
        golden-section refinement around the three best peaks of those. A
        grid alone under-reads the index between its points, and always in
        the unsafe direction (by up to 5 % at 36 planes). The refined value
        is never below the coarse one.
        """
        n = s.shape[0]
        phi_c = self._phi_c_array(material, S12_is)
        calibrated = ~np.isnan(phi_c)
        phi_c0 = np.where(calibrated, phi_c, 0.0)
        args = (s, material, phi_extra, Yt_is, S12_is, phi_c0)

        # Stage 1: coarse grid + the maximum-shear plane (pi-periodic).
        grid = np.linspace(0.0, np.pi, self.n_psi, endpoint=False)
        psi = np.empty((n, grid.size + 1))
        psi[:, :-1] = grid
        psi[:, -1] = np.arctan2(s[:, 4], s[:, 5])
        fi_grid = self._kink_fi_planes(*args, psi)
        fi_point = fi_grid.max(axis=1)

        # Stage 2: golden-section refinement within one grid spacing of the
        # THREE best coarse *peaks*. The index can have several lobes in
        # psi. Refining the best coarse points instead climbs the wrong
        # lobe: the two best points are usually neighbours on one lobe, so
        # which lobe won flipped with round-off (equal stress fields gave
        # indices 1e-4 apart) and the index was under-read by up to 1.4 %.
        # A candidate is a local maximum of the pi-periodic grid, or the
        # maximum-shear plane.
        ring = fi_grid[:, :-1]
        peak = (ring >= np.roll(ring, 1, axis=1)) & (
            ring >= np.roll(ring, -1, axis=1)
        )
        score = np.where(np.column_stack([peak, np.ones(n, bool)]),
                         fi_grid, -np.inf)
        top = np.argsort(score, axis=1, kind="stable")[:, -3:]
        half = np.pi / self.n_psi
        g = 0.5 * (np.sqrt(5.0) - 1.0)
        # All candidates refine together, one (N, n_candidates) evaluation
        # per step: per-candidate arithmetic is unchanged, but the scalar
        # path (N = 1, overhead-bound) makes a third of the calls.
        centre = np.take_along_axis(psi, top, axis=1)
        a, b = centre - half, centre + half
        c, d = b - g * (b - a), a + g * (b - a)
        fc = self._kink_fi_planes(*args, c)
        fd = self._kink_fi_planes(*args, d)
        for _ in range(self._PSI_REFINE_ITERS):
            # Keep the half holding the larger interior value; one new
            # evaluation per step (the other interior point is reused).
            left = fc >= fd
            a_new = np.where(left, a, c)
            b_new = np.where(left, d, b)
            c_new = np.where(left, b_new - g * (b_new - a_new), d)
            d_new = np.where(left, c, a_new + g * (b_new - a_new))
            f_eval = self._kink_fi_planes(
                *args, np.where(left, c_new, d_new)
            )
            fc, fd = np.where(left, f_eval, fd), np.where(left, fc, f_eval)
            a, b, c, d = a_new, b_new, c_new, d_new
        fi_point = np.maximum(
            fi_point, np.maximum(fc, fd).max(axis=1)
        )

        # No real phi_C: kinking cannot be calibrated to Xc for this
        # material, so fall back to the plain compressive-strength ratio.
        out: np.ndarray = np.where(
            calibrated, fi_point, np.abs(s[:, 0]) / material.Xc
        )
        return out

    #: Iteration cap for the kinking reserve-factor solve (bracketed
    #: secant / Illinois). It converges superlinearly, typically in under
    #: ten steps; the cap only bounds pathological rows.
    _RF_MAX_ITERS = 60
    #: Relative bracket width at which a row's reserve factor is final.
    _RF_RTOL = 1.0e-12

    def _kink_reserve(
        self,
        s: np.ndarray,
        material: OrthotropicMaterial,
        phi_extra: np.ndarray,
        Yt_is: np.ndarray,
        S12_is: np.ndarray,
    ) -> np.ndarray:
        """Exact load scale ``R`` with ``kinking FI(R · σ) = 1``, per row.

        The kinking index is nonlinear in load (the misalignment grows with
        it), so ``1 / FI`` at an arbitrary load is not the reserve factor —
        far past failure it is not even monotone. Below the first crossing
        ``FI < 1`` and from it on ``FI >= 1`` (checked over every library
        material, angle and random load direction in
        ``tests/test_failure/test_larc05.py``), so a bracketed root-find
        between a load scale below failure and one at-or-above it converges
        to the first crossing. The returned value is the bracket's failed
        end, so ``FI(R · σ) >= 1``. Rows never touch each other (a converged
        row is frozen), so a one-row call (the scalar path) is bit-identical
        to the same row in a field.
        """
        def fi_at(scale: np.ndarray, rows: np.ndarray) -> np.ndarray:
            return self._kink_fi_raw(
                s[rows] * scale[:, None], material, phi_extra[rows],
                Yt_is[rows], S12_is[rows],
            )

        return self._reserve(fi_at, s.shape[0])

    def _reserve(
        self,
        fi_at: Callable[[np.ndarray, np.ndarray], np.ndarray],
        n: int,
    ) -> np.ndarray:
        """Load scale ``R`` with ``FI(R · σ) = 1`` for ``n`` rows.

        ``fi_at(scale, rows)`` returns the index of rows ``rows`` with
        their stress scaled by ``scale``; it may return ``inf`` for a
        failed state. Requires ``FI < 1`` below the first crossing and
        ``FI >= 1`` from it on. Brackets by doubling, then Illinois. Returns
        the bracket's failed end (so ``FI(R · σ) >= 1``), and ``inf`` for
        a row that is unloaded or whose index never reaches 1.
        """
        def g(scale: np.ndarray, active: np.ndarray) -> np.ndarray:
            """``FI - 1`` at ``scale`` for the ``active`` rows only (rows
            are independent, so evaluating a subset is bit-identical);
            other rows get ``nan`` and are never read."""
            out = np.full(n, np.nan)
            rows = np.flatnonzero(active)
            if rows.size:
                fi = fi_at(scale[rows], rows)
                # inf (kinked) is simply "failed"; cap it so the secant
                # step stays finite.
                out[rows] = np.minimum(fi, 1.0e6) - 1.0
            return out

        fi_one = fi_at(np.ones(n), np.arange(n))
        unloaded = fi_one == 0.0
        positive = fi_one > 0.0
        # 1 / FI is the reserve factor for a linear index, so a good first
        # guess. A row already failed at scale 1 (FI = inf) starts its
        # bracket there instead of at 1 / inf = 0.
        first = positive & np.isfinite(fi_one)
        r_hi = np.where(first, 1.0 / np.where(first, fi_one, 1.0), 1.0)
        g_hi = np.where(unloaded, 0.0, g(r_hi, ~unloaded))
        r_lo = np.zeros(n)
        g_lo = np.full(n, -1.0)  # FI(0) = 0
        for _ in range(64):  # expand until the upper end is failed
            below = (g_hi < 0.0) & ~unloaded
            if not below.any():
                break
            r_lo = np.where(below, r_hi, r_lo)
            g_lo = np.where(below, g_hi, g_lo)
            r_hi = np.where(below, 2.0 * r_hi, r_hi)
            g_hi = np.where(below, g(r_hi, below), g_hi)
        # Still unfailed after 2**64 x the first guess: it never fails
        # (a friction-saturated matrix plane).
        never = (g_hi < 0.0) & ~unloaded

        # Illinois: regula falsi that halves the stale end's value, so it
        # cannot stall on one side.
        side = np.zeros(n)  # +1: hi moved last, -1: lo moved last
        done = unloaded | never | (r_hi - r_lo <= self._RF_RTOL * r_hi)
        for _ in range(self._RF_MAX_ITERS):
            if done.all():
                break
            denom = g_hi - g_lo
            r_new = np.where(
                denom > 0.0,
                r_hi - g_hi * (r_hi - r_lo) / np.where(denom > 0, denom, 1.0),
                0.5 * (r_lo + r_hi),
            )
            # Stay strictly inside the bracket.
            inside = (r_new > r_lo) & (r_new < r_hi)
            r_new = np.where(inside, r_new, 0.5 * (r_lo + r_hi))
            upd = ~done
            g_new = g(r_new, upd)
            failed = g_new >= 0.0
            hi_moves = upd & failed
            lo_moves = upd & ~failed
            r_hi = np.where(hi_moves, r_new, r_hi)
            g_hi = np.where(hi_moves, g_new, g_hi)
            r_lo = np.where(lo_moves, r_new, r_lo)
            g_lo = np.where(lo_moves, g_new, g_lo)
            g_lo = np.where(hi_moves & (side > 0), 0.5 * g_lo, g_lo)
            g_hi = np.where(lo_moves & (side < 0), 0.5 * g_hi, g_hi)
            side = np.where(hi_moves, 1.0, np.where(lo_moves, -1.0, side))
            done = done | (r_hi - r_lo <= self._RF_RTOL * r_hi) | (
                upd & (g_new == 0.0)
            )
        out: np.ndarray = np.where(unloaded | never, np.inf, r_hi)
        return out

    def _fibre_kinking(
        self,
        stress: np.ndarray,
        material: OrthotropicMaterial,
        Yt_is: float,
        S12_is: float,
        phi_extra: float,
    ) -> float:
        """Scalar fibre-kinking index (one-row call into :meth:`_kink_fi`)."""
        return float(
            self._kink_fi(
                np.asarray(stress, dtype=np.float64)[None, :],
                material,
                np.array([phi_extra], dtype=np.float64),
                np.array([Yt_is], dtype=np.float64),
                np.array([S12_is], dtype=np.float64),
            )[0]
        )

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

            # ``x * x`` rather than ``x ** 2`` to stay bit-identical with
            # the vectorised field path (issue #299).
            if sn >= 0:
                # Tensile fracture plane
                r_nt = tnt / S_T
                r_n1 = tn1 / S_L
                r_n = sn / Yt_is
                fi_arr[i] = r_nt * r_nt + r_n1 * r_n1 + r_n * r_n
            else:
                # Compressive fracture plane with friction
                denom_t = S_T + mu_T * abs(sn)
                denom_l = S_L + mu_L * abs(sn)
                if denom_t <= 0 or denom_l <= 0:
                    fi_arr[i] = float("inf")
                else:
                    r_nt = tnt / denom_t
                    r_n1 = tn1 / denom_l
                    fi_arr[i] = r_nt * r_nt + r_n1 * r_n1

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

            - ``'misalignment_angle'`` (float): *additional* initial
              fibre misalignment (radians) beyond the intrinsic ``φ_C``.
              Pass it only when the stress is NOT already in the
              misaligned fibre frame; FE local stresses are, so the FE
              path does not pass it.
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

        # Reserve factor: the load scale to first failure. Fibre tension is
        # linear in load, so 1 / FI is exact there; kinking is not, so its
        # reserve factor is solved for (see ``_kink_reserve``), and so is
        # the matrix one, which friction makes nonlinear too
        # (``_matrix_reserve``).
        if s1 >= 0:
            rf_fiber = 1.0 / fi_fiber if fi_fiber > 0 else float("inf")
        else:
            rf_fiber = float(
                self._kink_reserve(
                    stress_local[None, :],
                    material,
                    np.array([phi_0], dtype=np.float64),
                    np.array([Yt_is], dtype=np.float64),
                    np.array([S12_is], dtype=np.float64),
                )[0]
            )
        rf_matrix = float(
            self._matrix_reserve(
                stress_local[None, :],
                material,
                np.array([Yt_is], dtype=np.float64),
                np.array([S12_is], dtype=np.float64),
            )[0]
        )
        rf = min(rf_fiber, rf_matrix)

        detail = {
            "fi_fiber": float(fi_fiber),
            "fi_matrix": float(fi_matrix),
            "mode_fiber": mode_fiber,
            "mode_matrix": mode_matrix,
            # Caller-supplied extra imperfection, and the intrinsic,
            # Xc-calibrated misalignment (None if it has no real value).
            "phi_0": phi_0,
            "phi_C": self.intrinsic_misalignment(material, t_eff),
        }

        return FailureResult(
            index=float(fi),
            mode=mode,
            reserve_factor=float(rf),
            criterion_name=self.name,
            detail=detail,
        )

    # ------------------------------------------------------------------
    # Vectorised field evaluation (issue #299)
    # ------------------------------------------------------------------

    def evaluate_field(
        self,
        stress_field: np.ndarray,
        material: OrthotropicMaterial,
        contexts=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised LaRC04/05 evaluation across an array of stress states.

        Reproduces per-point :meth:`evaluate` exactly (issue #299): the
        fibre-tension, fibre-kinking (the shared :meth:`_kink_fi`, over an
        ``(N, n_psi + 1)`` kink-plane grid) and matrix fracture-plane-search
        sub-criteria are each broadcast over N points; the fracture-plane
        search additionally broadcasts over the ``(N, n_theta)`` angle
        grid. Per-point ``contexts`` supply the extra misalignment angle
        (and optional ply-thickness override) as arrays.

        Parameters
        ----------
        stress_field : np.ndarray
            Shape ``(N, 6)`` array of local stress vectors.
        material : OrthotropicMaterial
            Material with strength, elastic and LaRC properties; shared
            by all N points.
        contexts : list of dict, optional
            Per-point context dicts (``misalignment_angle``,
            ``ply_thickness``), same length as *stress_field*.

        Returns
        -------
        indices, modes, reserve_factors : np.ndarray
            Shape ``(N,)`` arrays matching :meth:`evaluate` per point.
            (The per-point ``detail`` diagnostics of :meth:`evaluate` are
            not materialised on the field path.)
        """
        indices, modes, reserve_factors = self._field(
            stress_field, material, contexts, want_rf=True
        )
        assert reserve_factors is not None
        return indices, modes, reserve_factors

    def evaluate_field_indices(
        self,
        stress_field: np.ndarray,
        material: OrthotropicMaterial,
        contexts: list | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`evaluate_field` without the reserve factors.

        Identical indices and modes. Skips the kinking reserve-factor solve,
        which is a root-find per compressive point; the FE failure fields
        (:class:`~wrinklefe.failure.evaluator.FailureEvaluator`) only use
        indices and modes.
        """
        indices, modes, _ = self._field(
            stress_field, material, contexts, want_rf=False
        )
        return indices, modes

    def _field(
        self,
        stress_field: np.ndarray,
        material: OrthotropicMaterial,
        contexts: list | None,
        *,
        want_rf: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Shared body of :meth:`evaluate_field` / ``_indices``."""
        s = np.asarray(stress_field, dtype=np.float64)
        if s.ndim != 2 or s.shape[1] != 6:
            raise ValueError(
                f"stress_field must have shape (N, 6), got {s.shape}"
            )
        n = s.shape[0]

        # --- Per-point context arrays -------------------------------
        phi_0 = np.zeros(n, dtype=np.float64)
        t_eff = np.full(n, self.ply_thickness, dtype=np.float64)
        if contexts is not None:
            for i, ctx in enumerate(contexts):
                if ctx is None:
                    continue
                phi_0[i] = ctx.get("misalignment_angle", 0.0)
                t_override = ctx.get("ply_thickness", None)
                if t_override is not None:
                    t_eff[i] = t_override

        # --- In-situ strengths --------------------------------------
        # Usually every point shares one ply thickness -> one scalar
        # evaluation; fall back to per-unique-thickness evaluation when
        # contexts carry overrides (rare).
        Yt_is = np.empty(n, dtype=np.float64)
        S12_is = np.empty(n, dtype=np.float64)
        for t_val in np.unique(t_eff):
            yt, s12i = self._in_situ_strengths(material, float(t_val))
            mask = t_eff == t_val
            Yt_is[mask] = yt
            S12_is[mask] = s12i

        # Process rows in cache-sized blocks: the fracture-plane search
        # materialises ~8 (block x n_theta) float64 temporaries, and
        # keeping them cache resident is measured ~3x faster than the
        # full-field grid on an 80k-point sample. Chunking does not
        # change any per-element arithmetic.
        indices = np.empty(n, dtype=np.float64)
        modes = np.empty(n, dtype="U32")
        reserve_factors = np.empty(n, dtype=np.float64) if want_rf else None
        for start in range(0, n, self._FIELD_CHUNK):
            b = slice(start, min(start + self._FIELD_CHUNK, n))
            fi_b, mode_b, rf_b = self._evaluate_field_block(
                s[b], material, phi_0[b], Yt_is[b], S12_is[b],
                want_rf=want_rf,
            )
            indices[b] = fi_b
            modes[b] = mode_b
            if reserve_factors is not None and rf_b is not None:
                reserve_factors[b] = rf_b
        return indices, modes, reserve_factors

    # Rows per block for the (N, n_theta) fracture-plane grid; see
    # ``evaluate_field``.
    _FIELD_CHUNK = 2048

    def _matrix_fi_rows(
        self,
        s: np.ndarray,
        material: OrthotropicMaterial,
        Yt_is: np.ndarray,
        S12_is: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised matrix fracture-plane search over ``(N, 6)`` rows.

        Returns the matrix index and whether its critical plane is in
        tension. Bit-identical per row to :meth:`_matrix_failure`.
        """
        n = s.shape[0]
        s2, s3 = s[:, 1], s[:, 2]
        t23, t13, t12 = s[:, 3], s[:, 4], s[:, 5]
        mu_L, mu_T = self._friction_coefficients(material)
        alpha_0_rad = np.radians(material.alpha_0)
        tan_2a = np.tan(2.0 * alpha_0_rad)
        S_T = material.Yc * np.cos(alpha_0_rad) * (
            np.sin(alpha_0_rad) + np.cos(alpha_0_rad) / tan_2a
        )

        thetas = np.linspace(-np.pi / 2, np.pi / 2, self.n_theta)
        cos_t = np.cos(thetas)[None, :]
        sin_t = np.sin(thetas)[None, :]
        s2c, s3c = s2[:, None], s3[:, None]
        t23c, t13c, t12c = t23[:, None], t13[:, None], t12[:, None]

        sigma_n = (
            s2c * cos_t**2 + s3c * sin_t**2 + 2.0 * t23c * sin_t * cos_t
        )
        tau_nt = (s3c - s2c) * sin_t * cos_t + t23c * (cos_t**2 - sin_t**2)
        tau_n1 = t12c * cos_t + t13c * sin_t

        S_L = S12_is[:, None]
        Yt_col = Yt_is[:, None]
        plane_tension = sigma_n >= 0

        fi_grid = np.empty_like(sigma_n)
        with np.errstate(divide="ignore", invalid="ignore"):
            fi_t = (
                (tau_nt / S_T) ** 2
                + (tau_n1 / S_L) ** 2
                + (sigma_n / Yt_col) ** 2
            )
            denom_tp = S_T + mu_T * np.abs(sigma_n)
            denom_lp = S_L + mu_L * np.abs(sigma_n)
            fi_c = (tau_nt / denom_tp) ** 2 + (tau_n1 / denom_lp) ** 2
        np.copyto(fi_grid, fi_c)
        np.copyto(fi_grid, fi_t, where=plane_tension)
        bad_plane = ~plane_tension & ((denom_tp <= 0) | (denom_lp <= 0))
        fi_grid[bad_plane] = np.inf

        rows = np.arange(n)
        idx_max = np.argmax(fi_grid, axis=1)
        fi_matrix = np.sqrt(np.maximum(fi_grid[rows, idx_max], 0.0))
        tension_at_max: np.ndarray = plane_tension[rows, idx_max]
        return fi_matrix, tension_at_max

    def _matrix_reserve(
        self,
        s: np.ndarray,
        material: OrthotropicMaterial,
        Yt_is: np.ndarray,
        S12_is: np.ndarray,
    ) -> np.ndarray:
        """Exact load scale ``R`` with ``matrix FI(R · σ) = 1``, per row.

        On a compressive fracture plane friction raises the strength with
        the load (``S + μ|σ_n|``), so the index grows more slowly than the
        load and ``1 / FI`` is not the reserve factor: below failure it
        under-reads it (by over half in measured states), past failure it
        over-reads it. On every plane the index still rises monotonically
        with load (``λτ / (S + μλ|σ_n|)`` is increasing for ``μ >= 0``),
        so the maximum over planes does too and the bracketed root-find
        applies. A row whose planes all saturate below 1 never fails
        (``inf``).
        """

        def fi_at(scale: np.ndarray, rows: np.ndarray) -> np.ndarray:
            return self._matrix_fi_rows(
                s[rows] * scale[:, None], material, Yt_is[rows], S12_is[rows],
            )[0]

        return self._reserve(fi_at, s.shape[0])

    def _evaluate_field_block(
        self,
        s: np.ndarray,
        material: OrthotropicMaterial,
        phi_0: np.ndarray,
        Yt_is: np.ndarray,
        S12_is: np.ndarray,
        *,
        want_rf: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """One cache-sized block of the vectorised evaluation."""
        s1, t13, t12 = s[:, 0], s[:, 4], s[:, 5]

        # --- Fibre tension (sigma_11 >= 0) ---------------------------
        tension = s1 >= 0
        fi_ft = (s1 / material.Xt) ** 2 + (t12 / material.S12) ** 2
        if material.S13 > 0:
            fi_ft = fi_ft + (t13 / material.S13) ** 2
        fi_ft = np.sqrt(np.maximum(fi_ft, 0.0))

        # --- Fibre kinking (sigma_11 < 0) ----------------------------
        # The same function evaluate() calls, so the paths agree exactly.
        fi_kink = self._kink_fi(s, material, phi_0, Yt_is, S12_is)

        fi_fiber = np.where(tension, fi_ft, fi_kink)
        modes_fiber = np.where(
            tension, "fiber_tension", "fiber_kinking"
        ).astype("U32")

        # --- Matrix failure (fracture-plane search, (N, n_theta)) ----
        fi_matrix, matrix_tension = self._matrix_fi_rows(
            s, material, Yt_is, S12_is
        )
        modes_matrix = np.where(
            matrix_tension, "matrix_tension", "matrix_compression",
        ).astype("U32")

        # --- Governing criterion -------------------------------------
        fiber_governs = fi_fiber >= fi_matrix
        indices = np.where(fiber_governs, fi_fiber, fi_matrix)
        modes = np.where(fiber_governs, modes_fiber, modes_matrix).astype(
            "U32"
        )
        # Reserve factors exactly as evaluate() forms them: 1 / FI for
        # fibre tension, the solved load scale for kinking and matrix.
        def _inv(x: np.ndarray) -> np.ndarray:
            return np.where(x > 0, 1.0 / np.where(x > 0, x, 1.0), np.inf)

        if not want_rf:
            return indices, modes, None
        rf_fiber = _inv(fi_ft)
        comp = ~tension
        if comp.any():
            rf_fiber[comp] = self._kink_reserve(
                s[comp], material, phi_0[comp], Yt_is[comp], S12_is[comp]
            )
        reserve_factors = np.minimum(
            rf_fiber, self._matrix_reserve(s, material, Yt_is, S12_is)
        )
        return indices, modes, reserve_factors
