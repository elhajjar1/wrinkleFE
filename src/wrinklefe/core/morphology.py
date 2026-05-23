"""Wrinkle morphology configuration and multi-wrinkle aggregate mechanics.

This module implements the wrinkle morphology configuration system for
composite laminates with fiber waviness defects. It manages multi-wrinkle
setups and computes morphology factors based on the physics-based model
from wavy plate theory.

The morphology factor M_f quantifies how the relative phase offset between
adjacent wrinkles amplifies or attenuates compression strength knockdown.
For dual-wrinkle configurations, the three canonical morphologies are:

- **Stack** (phi = 0): Aligned peaks/troughs, M_f = 1.0 (baseline)
- **Convex** (phi = pi/2): Outward bulge, M_f < 1 (best for compression)
- **Concave** (phi = -pi/2): Inward pinch, M_f > 1 (worst for compression)

The concave morphology is most critical because the inward pinching
amplifies kink-band formation under compression loading.

References
----------
- Jin, L. et al. (2026). Interlaminar damage analysis of dual-wrinkled
  composite laminates under multidirectional static loading. Thin-Walled
  Structures, 219, 114237.
- Elhajjar, R. (2025). Fat-tailed failure strength distributions and
  manufacturing defects in advanced composites. Scientific Reports, 15, 25977.
- Budiansky, B. & Fleck, N.A. (1993). Compressive failure of fibre
  composites. J. Mech. Phys. Solids, 41(1), 183-211.
- Timoshenko, S. & Woinowsky-Krieger, S. (1959). Theory of Plates and Shells.
  McGraw-Hill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Union

import numpy as np

from wrinklefe.core.wrinkle import WrinkleProfile, WrinkleSurface3D


# ======================================================================
# Predefined morphology phases
# ======================================================================

MORPHOLOGY_PHASES: Dict[str, float] = {
    "stack": 0.0,
    "convex": np.pi / 2,
    "concave": -np.pi / 2,
}

# Single-wrinkle morphology types (not phase-based)
SINGLE_WRINKLE_MODES = {"uniform", "graded"}
"""Canonical dual-wrinkle morphology phase offsets (radians).

- ``'stack'``: phi = 0 -- peaks and troughs aligned vertically.
- ``'convex'``: phi = pi/2 -- interface bulges outward (best under compression).
- ``'concave'``: phi = -pi/2 -- interface pinches inward (worst under compression).

These values correspond to the morphology classifications in Jin et al. (2026)
Table 3 and the analytical model in Elhajjar (2025) Eq. 8.
"""


# ======================================================================
# Loading-dependent model parameters
# ======================================================================

_LOADING_PARAMS: Dict[str, Dict[str, float]] = {
    "compression": {"alpha_asym": 0.288, "alpha_offset": 0.0},
    "tension": {"alpha_asym": 0.033, "alpha_offset": 0.183},
}
"""Morphology factor parameters calibrated from wavy plate theory.

For compression loading, the asymmetry parameter alpha_asym = 0.288 captures the
pinch/bulge asymmetry in kink-band formation. The offset parameter is zero because
kink-band initiation depends primarily on the asymmetric curvature interaction.

For tension loading, alpha_asym = 0.033 (weak asymmetry) and alpha_offset = 0.183
captures the stress-concentration and delamination-driven failure mode where
misaligned plies (phi != 0, pi) are penalised by the offset term.
"""


# ======================================================================
# WrinklePlacement
# ======================================================================

@dataclass
class WrinklePlacement:
    """A single wrinkle placed at a specific ply interface.

    Parameters
    ----------
    profile : WrinkleProfile or WrinkleSurface3D
        The wrinkle geometry profile defining out-of-plane displacement.
        A :class:`WrinkleProfile` provides a 1D cross-section z(x), while
        a :class:`WrinkleSurface3D` extends this to z(x, y).
    ply_interface : int
        Interface index between ply *i* and ply *i+1* where the wrinkle
        is located (dimensionless integer). For a laminate with *N*
        plies, valid indices are 0 through N-2 (inclusive).
    phase_offset : float
        Phase offset phi in **radians** relative to a reference wrinkle,
        in the range that maps naturally to ``[-pi, pi]``. The first
        wrinkle in a configuration typically uses phi = 0. Canonical
        dual-wrinkle morphologies: phi = 0 (stack, aligned crests),
        phi = +pi/2 (convex), phi = -pi/2 (concave). Sign convention
        follows :data:`MORPHOLOGY_PHASES`.

    Notes
    -----
    The phase offset is the key parameter that determines the dual-wrinkle
    morphology. It is a pure angle (radians) that relates to a *geometric*
    longitudinal offset Delta_x (mm) between wrinkle centres by:

        phi = 2 * pi * Delta_x / lambda      (radians)

    where lambda is the wrinkle wavelength (mm) (Jin et al., 2026, Eq. 4).
    Equivalently Delta_x = phi * lambda / (2 * pi) (mm). Because both
    Delta_x and lambda are in millimetres, phi is dimensionless.
    """

    profile: Union[WrinkleProfile, WrinkleSurface3D]
    ply_interface: int
    phase_offset: float = 0.0


# ======================================================================
# WrinkleConfiguration
# ======================================================================

class WrinkleConfiguration:
    """Manages a set of wrinkles and computes aggregate morphology effects.

    This class is the central object for multi-wrinkle analysis. It stores
    one or more :class:`WrinklePlacement` instances and provides methods
    to compute:

    - Pairwise and aggregate morphology factors
    - Effective fiber misalignment angles
    - Mesh deformation fields for FE analysis
    - Local fiber angle distributions

    A configuration is characterised by *two* independent choices: the
    set of :class:`WrinklePlacement` instances (one or two wrinkles, each
    at its own ply interface and phase) and the ``decay_mode`` /
    ``decay_floor`` pair that shapes the through-thickness amplitude
    envelope. The five named morphologies exposed via
    :meth:`from_morphology_name` map onto these two axes as follows:

    - ``"stack"``: dual-wrinkle, φ = 0, ``decay_mode="default"``
      (linear taper from the interface plies to zero at the outer
      surfaces). M_f = 1.0 dual-wrinkle baseline.
    - ``"convex"``: dual-wrinkle, φ = +π/2, ``decay_mode="default"``.
    - ``"concave"``: dual-wrinkle, φ = −π/2, ``decay_mode="default"``.
    - ``"uniform"``: single wrinkle, ``decay_mode="uniform"`` (every
      ply carries the full profile, *no* through-thickness decay).
      Differs from ``"stack"`` in both the wrinkle count and the
      through-thickness envelope even though both have M_f = 1.0.
    - ``"graded"``: single wrinkle, ``decay_mode="graded"`` (linear
      decay from mid-ply to surfaces, modulated by ``decay_floor``).

    The aggregate morphology factor for N wrinkles uses the geometric-mean
    normalisation from the N-wrinkle extension (Eq. 12)::

        M_f_agg = (prod_{i=1}^{N-1} M_f(phi_i))^{1/(N-1)}

    where phi_i is the phase offset between wrinkle *i* and wrinkle *i+1*.

    Parameters
    ----------
    wrinkles : list of WrinklePlacement
        One or more wrinkle placements sorted by ply interface. At least
        one wrinkle is required.
    decay_mode : str, optional
        Through-thickness amplitude distribution. ``"default"`` (linear
        decay from the wrinkle interface to the laminate surfaces),
        ``"uniform"`` (full amplitude at every ply), or ``"graded"``
        (linear decay from mid-ply to surfaces). Default ``"default"``.
    decay_floor : float, optional
        Dimensionless fraction in ``[0, 1]`` used only by the ``"graded"``
        decay mode: the minimum fraction of the wrinkle amplitude retained
        at the surface plies. ``0.0`` means full decay to zero amplitude
        at the surfaces (pure graded); ``1.0`` means no decay (equivalent
        to uniform). Values outside ``[0, 1]`` are clamped. Default 0.0.
    amplitude_profile : {"constant", "gaussian", "linear"}, optional
        Spatially varying in-plane amplitude modulation applied on top of
        the wrinkle's own longitudinal envelope. ``"constant"`` (default)
        preserves the legacy behaviour -- the amplitude *A* set on the
        wrinkle profile is used uniformly across the in-plane domain.
        ``"gaussian"`` multiplies *A* by ``exp(-(s/d)**2)`` where *s* is the
        coordinate (relative to the wrinkle centre) along the chosen axis
        and *d* is the decay length. ``"linear"`` multiplies *A* by
        ``max(0, 1 - |s|/d)`` (clipped at zero).
    amplitude_profile_decay_length : float, optional
        Length scale *d* (mm) controlling the Gaussian sigma or the
        linear-decay extent. When ``None`` (default) the helper falls back
        to the wrinkle profile's own ``width``, which is the natural choice
        because users typically want the amplitude to taper on the same
        length scale as the envelope. Must be positive when provided.
    amplitude_profile_axis : {"x", "y"}, optional
        In-plane axis along which the amplitude modulation runs. Default
        ``"x"``. NOTE: the existing longitudinal envelope (e.g. the
        Gaussian in :class:`GaussianSinusoidal`) already runs on *x*, so
        choosing ``amplitude_profile_axis="x"`` with ``"gaussian"`` will
        effectively square the Gaussian on *x*; this is a degenerate but
        allowed configuration -- pick ``"y"`` (transverse axis) to obtain
        an independent in-plane tapering of *A*.

    Raises
    ------
    ValueError
        If the wrinkle list is empty, the amplitude profile name is
        unknown, or the decay length is non-positive.

    Notes
    -----
    The wrinkle ``amplitude`` carried by each placement's
    :class:`~wrinklefe.core.wrinkle.WrinkleProfile` is the
    **half-amplitude** *A* (mm): the peak displacement of the wrinkled
    mid-surface from the flat reference, so
    ``z(x) = A * cos(2*pi*(x - x0) / lambda)`` (modulated by the
    envelope) and the peak-to-trough height is ``2A``. For a measured
    wrinkle, ``A = (z_max - z_min) / 2``.

    Examples
    --------
    Create a dual-wrinkle concave configuration::

        >>> from wrinklefe.core.wrinkle import GaussianSinusoidal
        >>> profile = GaussianSinusoidal(amplitude=0.366, wavelength=16.0, width=12.0)
        >>> config = WrinkleConfiguration.from_morphology_name(
        ...     "concave", profile, interface1=10, interface2=13
        ... )
        >>> config.n_wrinkles()
        2
        >>> config.aggregate_morphology_factor("compression")  # doctest: +ELLIPSIS
        1.336...

    References
    ----------
    - Jin et al. (2026), Thin-Walled Structures 219:114237, Eq. 2-6.
    - Elhajjar (2025), Scientific Reports 15:25977, Eq. 7-12.
    """

    _VALID_AMPLITUDE_PROFILES = ("constant", "gaussian", "linear")
    _VALID_AMPLITUDE_PROFILE_AXES = ("x", "y")

    def __init__(
        self,
        wrinkles: list[WrinklePlacement],
        decay_mode: str = "default",
        decay_floor: float = 0.0,
        amplitude_profile: Literal["constant", "gaussian", "linear"] = "constant",
        amplitude_profile_decay_length: Optional[float] = None,
        amplitude_profile_axis: Literal["x", "y"] = "x",
    ) -> None:
        if not wrinkles:
            raise ValueError("At least one WrinklePlacement is required.")
        # Store sorted by ply interface for consistent ordering
        self.wrinkles: list[WrinklePlacement] = sorted(
            wrinkles, key=lambda w: w.ply_interface
        )
        # Decay mode for through-thickness amplitude distribution:
        #   "default"  — standard linear decay from wrinkle interface
        #   "uniform"  — full amplitude at all plies (no decay)
        #   "graded"   — linear decay from mid-ply to surfaces
        self.decay_mode: str = decay_mode
        # Decay floor for graded mode: minimum fraction of amplitude at
        # surface plies (0.0 = full decay to zero, 1.0 = uniform).
        # Physically: surface plies retain some waviness even far from
        # the wrinkle core. Interpolates between graded (0) and uniform (1).
        self.decay_floor: float = max(0.0, min(1.0, decay_floor))

        # Spatially varying in-plane amplitude profile (issue #3). This is
        # a SEPARATE multiplicative modulation applied on top of the
        # wrinkle's own longitudinal envelope -- it lets a single wrinkle
        # be modelled with a tapered or localised amplitude.
        if amplitude_profile not in self._VALID_AMPLITUDE_PROFILES:
            raise ValueError(
                f"Unknown amplitude_profile '{amplitude_profile}'. "
                f"Choose from: {list(self._VALID_AMPLITUDE_PROFILES)}"
            )
        if amplitude_profile_axis not in self._VALID_AMPLITUDE_PROFILE_AXES:
            raise ValueError(
                f"Unknown amplitude_profile_axis '{amplitude_profile_axis}'. "
                f"Choose from: {list(self._VALID_AMPLITUDE_PROFILE_AXES)}"
            )
        if (
            amplitude_profile_decay_length is not None
            and amplitude_profile_decay_length <= 0.0
        ):
            raise ValueError(
                "amplitude_profile_decay_length must be positive, "
                f"got {amplitude_profile_decay_length}"
            )
        self.amplitude_profile: str = amplitude_profile
        self.amplitude_profile_decay_length: Optional[float] = (
            amplitude_profile_decay_length
        )
        self.amplitude_profile_axis: str = amplitude_profile_axis

    # ------------------------------------------------------------------
    # Phase & Morphology (static methods)
    # ------------------------------------------------------------------

    @staticmethod
    def phase_from_offset(delta_x: float, wavelength: float) -> float:
        """Convert a geometric longitudinal offset to a phase angle.

        Parameters
        ----------
        delta_x : float
            Longitudinal offset between wrinkle centres [mm].
        wavelength : float
            Wrinkle wavelength lambda [mm]. Must be positive.

        Returns
        -------
        float
            Phase offset phi = 2 * pi * delta_x / lambda [radians].

        Raises
        ------
        ValueError
            If wavelength is not positive.

        Notes
        -----
        This relation follows directly from the sinusoidal wrinkle profile
        z(x) = A * exp(...) * cos(2*pi*x / lambda). A shift of delta_x in
        the argument of the cosine produces a phase shift of 2*pi*delta_x/lambda.
        See Jin et al. (2026), Eq. 4.
        """
        if wavelength <= 0.0:
            raise ValueError(f"Wavelength must be positive, got {wavelength}")
        return 2.0 * np.pi * delta_x / wavelength

    @staticmethod
    def morphology_factor_analytical(phi: float, loading: str = "compression") -> float:
        """Physics-based morphology factor from wavy plate theory.

        Computes the morphology factor M_f that modifies the effective fiber
        misalignment angle based on the relative phase between adjacent
        wrinkles and the loading mode.

        Parameters
        ----------
        phi : float
            Phase offset between adjacent wrinkles [radians].
        loading : str
            Loading mode, either ``'compression'`` or ``'tension'``.

        Returns
        -------
        float
            Morphology factor M_f. Values < 1 indicate stabilisation
            (reduced knockdown), values > 1 indicate amplification.

        Raises
        ------
        ValueError
            If *loading* is not ``'compression'`` or ``'tension'``.

        Notes
        -----
        The morphology factor is defined as (Elhajjar, 2025, Eq. 8):

        .. math::

            M_f(\\phi, \\text{loading}) = \\exp\\bigl(
                -\\alpha_{\\text{asym}} \\sin(\\phi)
                - \\alpha_{\\text{offset}} (1 - |\\cos(\\phi)|)
            \\bigr)

        where the loading-dependent parameters are:

        +-------------+----------------+------------------+-----------------------------------+
        | Loading     | alpha_asym     | alpha_offset     | Physical Mechanism                |
        +=============+================+==================+===================================+
        | compression | 0.288          | 0.0              | Kink-band (pinch/bulge asymmetry) |
        +-------------+----------------+------------------+-----------------------------------+
        | tension     | 0.033          | 0.183            | Stress concentration + delam      |
        +-------------+----------------+------------------+-----------------------------------+

        The phase offset determines the morphology:

        - phi = 0 : Stack (aligned peaks/troughs), M_f = 1.0
        - phi = pi/2 : Convex (bulges outward), M_f < 1 (BEST for compression)
        - phi = -pi/2 : Concave (pinches inward), M_f > 1 (WORST for compression)

        The model derives from curved-beam interaction mechanics under axial
        load (Timoshenko & Gere, 1961) applied to the wavy plate interface.
        """
        if loading not in _LOADING_PARAMS:
            raise ValueError(
                f"Unsupported loading mode '{loading}'. "
                f"Choose from: {list(_LOADING_PARAMS.keys())}"
            )
        params = _LOADING_PARAMS[loading]
        alpha_asym = params["alpha_asym"]
        alpha_offset = params["alpha_offset"]

        mf = np.exp(
            -alpha_asym * np.sin(phi)
            - alpha_offset * (1.0 - np.abs(np.cos(phi)))
        )
        return float(mf)

    @staticmethod
    def curvature_correlation(phi: float) -> float:
        """Interface curvature correlation between adjacent wrinkled plies.

        Parameters
        ----------
        phi : float
            Phase offset between adjacent wrinkles [radians].

        Returns
        -------
        float
            Correlation C(phi) = cos(phi). Ranges from +1 (stack, perfectly
            correlated curvatures) to 0 (convex/concave, uncorrelated)
            to -1 (anti-stack, anti-correlated).

        Notes
        -----
        The curvature correlation quantifies the degree to which adjacent ply
        curvatures reinforce or cancel. It enters the interface normal stress
        calculation in the wavy plate theory framework (Timoshenko &
        Woinowsky-Krieger, 1959, adapted in Elhajjar, 2025, Eq. 6).
        """
        return float(np.cos(phi))

    # ------------------------------------------------------------------
    # Multi-wrinkle aggregate properties
    # ------------------------------------------------------------------

    def n_wrinkles(self) -> int:
        """Number of wrinkles in the configuration.

        Returns
        -------
        int
        """
        return len(self.wrinkles)

    def pairwise_phases(self) -> list[float]:
        """Phase offsets between adjacent wrinkle pairs.

        Computes the phase difference phi_{i+1} - phi_i for each consecutive
        pair of wrinkles, ordered by ply interface.

        Returns
        -------
        list of float
            Length N-1 list of pairwise phase offsets [radians].
            Empty list if only one wrinkle is present.
        """
        phases: list[float] = []
        for i in range(len(self.wrinkles) - 1):
            dphi = self.wrinkles[i + 1].phase_offset - self.wrinkles[i].phase_offset
            phases.append(dphi)
        return phases

    def pairwise_morphology_factors(self, loading: str = "compression") -> list[float]:
        """Morphology factors for each adjacent wrinkle pair.

        Parameters
        ----------
        loading : str
            Loading mode (``'compression'`` or ``'tension'``).

        Returns
        -------
        list of float
            Length N-1 list of M_f values, one per adjacent pair.
            Empty list if only one wrinkle is present.
        """
        return [
            self.morphology_factor_analytical(dphi, loading)
            for dphi in self.pairwise_phases()
        ]

    def aggregate_morphology_factor(self, loading: str = "compression") -> float:
        """Aggregate morphology factor for the full multi-wrinkle configuration.

        For N wrinkles there are N-1 pairwise interactions. The aggregate
        factor uses a geometric-mean normalisation so that the result is
        independent of the number of wrinkle pairs (Eq. 12 from the
        N-wrinkle extension):

        .. math::

            M_{f,\\text{agg}} = \\left(
                \\prod_{i=1}^{N-1} M_f(\\phi_i)
            \\right)^{1/(N-1)}

        Special cases:

        - Single wrinkle (N=1): returns 1.0 (no pairwise interaction).
        - Dual wrinkle (N=2): returns the single pairwise M_f directly.

        Parameters
        ----------
        loading : str
            Loading mode (``'compression'`` or ``'tension'``).

        Returns
        -------
        float
            Aggregate morphology factor M_f_agg.
        """
        if self.n_wrinkles() < 2:
            return 1.0

        mf_values = self.pairwise_morphology_factors(loading)
        n_pairs = len(mf_values)

        # Geometric mean: (prod M_f_i)^(1/(N-1))
        product = 1.0
        for mf in mf_values:
            product *= mf

        return float(product ** (1.0 / n_pairs))

    def max_angle(self) -> float:
        """Maximum fiber misalignment angle across all wrinkles.

        Calls ``max_angle()`` on each wrinkle profile and returns the
        largest value. This is the raw geometric angle before morphology
        correction.

        Returns
        -------
        float
            Maximum misalignment angle theta_max [radians].
        """
        return max(w.profile.max_angle() for w in self.wrinkles)

    def effective_angle(self, loading: str = "compression") -> float:
        """Effective fiber misalignment angle accounting for morphology.

        The effective angle combines the geometric maximum angle with
        the aggregate morphology factor:

        .. math::

            \\theta_{\\text{eff}} = \\theta_{\\max} \\times M_{f,\\text{agg}}

        This is the angle that enters the Budiansky-Fleck kink-band
        equation for compression strength prediction.

        Parameters
        ----------
        loading : str
            Loading mode (``'compression'`` or ``'tension'``).

        Returns
        -------
        float
            Effective misalignment angle theta_eff [radians].

        Notes
        -----
        The effective angle is the key input to the strength knockdown
        equation (Budiansky & Fleck, 1993):

        .. math::

            \\sigma / \\sigma_0 = 1 / (1 + \\theta_{\\text{eff}} / \\gamma_Y)

        For the concave morphology, M_f > 1 amplifies the angle and
        reduces strength. For convex, M_f < 1 attenuates the angle.
        """
        return self.max_angle() * self.aggregate_morphology_factor(loading)

    # ------------------------------------------------------------------
    # Mesh deformation
    # ------------------------------------------------------------------

    def _amplitude_scale(
        self,
        wrinkle: WrinklePlacement,
        x: float,
        y: float,
    ) -> float:
        """In-plane scale factor for the spatially varying amplitude profile.

        Returns a dimensionless multiplicative factor in ``[0, 1]`` that
        modulates the wrinkle's *A* in addition to the wrinkle's own
        longitudinal envelope. The factor depends on the coordinate
        *s* (along the chosen axis) relative to the wrinkle centre:

        - ``"constant"`` (default) -> ``1.0`` (legacy behaviour).
        - ``"gaussian"`` -> ``exp(-(s / d) ** 2)``.
        - ``"linear"`` -> ``max(0, 1 - |s| / d)`` (clipped at zero).

        The decay length *d* defaults to the wrinkle profile's own
        ``width`` when :attr:`amplitude_profile_decay_length` is ``None``.

        Parameters
        ----------
        wrinkle : WrinklePlacement
            The wrinkle whose centre defines *s = coord - centre*. For
            the x-axis the centre is ``profile.center + delta_x`` (so the
            phase-offset shift is respected). For the y-axis the centre is
            ``span_y / 2`` for a :class:`WrinkleSurface3D` (matching the
            transverse modes' convention) and ``0.0`` for a plain 1-D
            profile (which has no y-extent of its own).
        x, y : float
            World-frame coordinates of the node.

        Returns
        -------
        float
            Multiplicative scale factor applied on top of the wrinkle's
            own profile.displacement / profile.slope output. Both
            :meth:`apply_to_nodes` and :meth:`fiber_angles_at_nodes`
            call this helper so the displacement and angle fields stay
            mutually consistent (#18 invariant).

        Notes
        -----
        The new modulation stacks multiplicatively with the wrinkle's
        own longitudinal envelope. Choosing ``amplitude_profile_axis="x"``
        with ``"gaussian"`` therefore squares the Gaussian along *x* (the
        existing envelope plus this new one) -- a degenerate but allowed
        configuration. Pick the transverse axis (``"y"``) to get a
        decoupled tapering of *A*.
        """
        if self.amplitude_profile == "constant":
            return 1.0

        profile = wrinkle.profile
        # Resolve the per-wrinkle base profile and centre.
        if isinstance(profile, WrinkleSurface3D):
            base_profile = profile.profile
            y_center = profile.span_y / 2.0
        else:
            base_profile = profile
            y_center = 0.0

        # Longitudinal centre in the world frame includes the phase shift.
        delta_x = wrinkle.phase_offset * base_profile.wavelength / (2.0 * np.pi)
        x_center = base_profile.center + delta_x

        if self.amplitude_profile_axis == "x":
            s = x - x_center
        else:  # "y" -- validated in __init__
            s = y - y_center

        d = self.amplitude_profile_decay_length
        if d is None:
            d = base_profile.width

        if self.amplitude_profile == "gaussian":
            return float(np.exp(-((s / d) ** 2)))
        # "linear" -- validated in __init__
        return float(max(0.0, 1.0 - abs(s) / d))

    @staticmethod
    def _ply_decay(p: int, k: int, n_plies: int) -> float:
        """Through-thickness linear decay factor for the default mode.

        The decay is 1.0 at the wrinkle interface plies (``p == k`` and
        ``p == k + 1``), 0.0 at the laminate outer surfaces (``p == 0``
        and ``p == n_plies - 1``), and interpolates linearly in between.
        Specifically:

        - For ``p <= k``: ``decay = p / k`` (so ``p=0 -> 0``, ``p=k -> 1``).
        - For ``p > k``:  ``decay = (n_plies - 1 - p) / ((n_plies - 1) - (k + 1))``
          (so ``p=k+1 -> 1``, ``p=n_plies-1 -> 0``).

        Both call sites (:meth:`apply_to_nodes` and
        :meth:`fiber_angles_at_nodes`) MUST use this helper so the
        displacement field and the local fibre-angle field share one
        through-thickness profile (issues #17, #18).

        Degenerate edge cases:

        - If ``n_plies <= 1`` the helper returns 1.0 (single-ply laminate).
        - If ``k == 0`` (wrinkle at the bottom outer surface) the ``p <= k``
          branch collapses to ``1.0 at p=0``; the ``p > k`` side uses the
          normal top ramp ``(n_plies - 1 - p) / ((n_plies - 1) - 1)``.
        - If ``k + 1 == n_plies - 1`` (wrinkle at the top outer surface)
          the ``p > k`` branch collapses to ``1.0 at p=k+1``; the ``p <= k``
          side uses the normal bottom ramp ``p / k``.
        """
        if n_plies <= 1:
            return 1.0

        if p <= k:
            # Bottom side: p=0 -> 0, p=k -> 1
            if k <= 0:
                # Wrinkle is at the bottom outer surface
                return 1.0 if p == k else 0.0
            return p / k
        else:
            # Top side: p=k+1 -> 1, p=n_plies-1 -> 0
            denom = (n_plies - 1) - (k + 1)
            if denom <= 0:
                # Wrinkle is at the top outer surface
                return 1.0 if p == k + 1 else 0.0
            return (n_plies - 1 - p) / denom

    def _amplitude_scale_vec(
        self,
        wrinkle: WrinklePlacement,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Vectorised counterpart of :meth:`_amplitude_scale`.

        Returns a float array with the same shape as ``x`` / ``y``
        containing the in-plane amplitude-profile scale factor at every
        node. Mirrors the scalar implementation exactly so the legacy
        and vectorised mesh-deformation paths stay numerically identical.
        """
        if self.amplitude_profile == "constant":
            return np.ones_like(x, dtype=np.float64)

        profile = wrinkle.profile
        if isinstance(profile, WrinkleSurface3D):
            base_profile = profile.profile
            y_center = profile.span_y / 2.0
        else:
            base_profile = profile
            y_center = 0.0

        delta_x = wrinkle.phase_offset * base_profile.wavelength / (2.0 * np.pi)
        x_center = base_profile.center + delta_x

        if self.amplitude_profile_axis == "x":
            s = x - x_center
        else:  # "y"
            s = y - y_center

        d = self.amplitude_profile_decay_length
        if d is None:
            d = base_profile.width

        if self.amplitude_profile == "gaussian":
            return np.exp(-((s / d) ** 2))
        # "linear"
        return np.maximum(0.0, 1.0 - np.abs(s) / d)

    def _through_thickness_decay(
        self,
        ply_ids: np.ndarray,
        k: int,
        n_plies: int,
    ) -> np.ndarray:
        """Vectorised through-thickness decay for every node in ``ply_ids``.

        Implements the same branching logic as :meth:`_ply_decay` plus
        the ``"uniform"`` and ``"graded"`` decay modes used inside
        :meth:`apply_to_nodes` / :meth:`fiber_angles_at_nodes`, returning
        a ``(N,)`` float array.
        """
        p = np.asarray(ply_ids, dtype=np.int64)

        if self.decay_mode == "uniform":
            return np.ones(p.shape, dtype=np.float64)

        if self.decay_mode == "graded":
            if n_plies <= 1:
                return np.ones(p.shape, dtype=np.float64)
            p_mid = (n_plies - 1) / 2.0
            raw = 1.0 - np.abs(p - p_mid) / p_mid  # 1 at mid, 0 at surface
            decay = self.decay_floor + (1.0 - self.decay_floor) * raw
            return np.maximum(0.0, decay)

        # Default per-ply linear decay shared with ``_ply_decay``.
        if n_plies <= 1:
            return np.ones(p.shape, dtype=np.float64)

        # Bottom side: p <= k  -> p / k    (or {1 if p==k else 0} when k<=0)
        if k <= 0:
            bottom = np.where(p == k, 1.0, 0.0)
        else:
            bottom = p / float(k)

        # Top side: p > k  -> (n_plies-1-p) / ((n_plies-1)-(k+1))
        denom = (n_plies - 1) - (k + 1)
        if denom <= 0:
            top = np.where(p == k + 1, 1.0, 0.0)
        else:
            top = (n_plies - 1 - p) / float(denom)

        return np.where(p <= k, bottom, top).astype(np.float64, copy=False)

    def apply_to_nodes(
        self,
        nodes: np.ndarray,
        ply_ids: np.ndarray,
        n_plies: int,
    ) -> np.ndarray:
        """Deform mesh nodes according to the wrinkle configuration.

        Each wrinkle displaces nodes at and above its ply interface.
        The displacement decays linearly through plies above and below
        the wrinkle centre, reaching zero at the laminate surfaces.

        Parameters
        ----------
        nodes : np.ndarray
            Shape (N, 3) array of node coordinates [x, y, z].
        ply_ids : np.ndarray
            Shape (N,) integer array of ply indices for each node.
            Values range from 0 to ``n_plies - 1``.
        n_plies : int
            Total number of plies in the laminate.

        Returns
        -------
        np.ndarray
            Shape (N, 3) array of deformed node coordinates with
            z-values modified by the wrinkle displacement field.

        Notes
        -----
        The decay function ensures displacement continuity through the
        thickness. For a wrinkle at interface *k* (between ply *k* and
        *k+1*), the decay factor at ply *p* is:

        - For p <= k: decay = p / k
        - For p > k:  decay = (n_plies - 1 - p) / (n_plies - 1 - (k + 1))

        This produces unit displacement at the two interface plies
        (p = k and p = k + 1) and zero at the laminate outer surfaces
        (p = 0 and p = n_plies - 1), with linear interpolation in
        between. Degenerate cases (wrinkle interface coincident with an
        outer surface) collapse to 1.0 on the interface ply only.

        The implementation is fully vectorised over nodes (issue #185):
        each wrinkle contributes a single NumPy expression over the
        whole node array rather than a Python ``for`` loop. Profile
        ``displacement`` methods already accept ``ndarray`` input, so
        broadcasting carries the work down to C-level loops.
        """
        deformed = nodes.copy()
        x = nodes[:, 0]
        # ``y`` is always present in the (N, 3) node array; fall back to
        # zeros only if the user passes a thinner array.
        y = nodes[:, 1] if nodes.shape[1] >= 2 else np.zeros(len(nodes))

        for wrinkle in self.wrinkles:
            k = wrinkle.ply_interface
            profile = wrinkle.profile

            # φ = 2π·Δx/λ  →  Δx = φ·λ/(2π)
            if isinstance(profile, WrinkleSurface3D):
                wavelength = profile.profile.wavelength
            else:
                wavelength = profile.wavelength
            delta_x = wrinkle.phase_offset * wavelength / (2.0 * np.pi)
            x_shifted = x - delta_x

            if isinstance(profile, WrinkleSurface3D):
                dz = profile.displacement(x_shifted, y)
            else:
                dz = profile.displacement(x_shifted)

            dz = dz * self._amplitude_scale_vec(wrinkle, x, y)
            dz = dz * self._through_thickness_decay(ply_ids, k, n_plies)

            deformed[:, 2] += dz

        return deformed

    def fiber_angles_at_nodes(
        self,
        nodes: np.ndarray,
        ply_ids: np.ndarray,
        n_plies: int | None = None,
    ) -> np.ndarray:
        """Compute local fiber misalignment angle at each node.

        For nodes within the wrinkle zone, the angle is computed as
        arctan(dz/dx) from the wrinkle profile slope. Each node receives
        the angle contribution from the wrinkle affecting its ply
        interface. Nodes outside any wrinkle zone receive angle = 0.

        When multiple wrinkles affect the same ply (via through-thickness
        decay), the angles are combined as the root-sum-square, reflecting
        independent misalignment contributions.

        Parameters
        ----------
        nodes : np.ndarray
            Shape (N, 3) array of node coordinates [x, y, z].
        ply_ids : np.ndarray
            Shape (N,) integer array of ply indices for each node.
        n_plies : int, optional
            Total number of plies in the laminate. This is the same
            authoritative count that :meth:`apply_to_nodes` takes, and
            it sets the through-thickness decay basis. It MUST be passed
            (matching the value given to :meth:`apply_to_nodes`) whenever
            ``ply_ids`` does not span the full laminate, otherwise the
            displacement and angle decay fields desynchronise (issue
            #146). When omitted it falls back to ``ply_ids.max() + 1``,
            which is only correct when ``ply_ids`` reaches the top ply.

        Returns
        -------
        np.ndarray
            Shape (N,) array of fiber misalignment angles [radians].
            All values are non-negative.

        Notes
        -----
        The fiber angle at a point is determined by the local slope of
        the wrinkle profile (Jin et al., 2026, Eq. 3):

        .. math::

            \\theta(x) = \\arctan\\left|\\frac{dz}{dx}\\right|

        For multi-wrinkle configurations, the effective local angle uses
        root-sum-square combination to avoid double-counting while capturing
        the aggregate misalignment.
        """
        n_nodes = len(nodes)
        if n_plies is None:
            # Backward-compatible fallback: only correct when ply_ids
            # spans the full laminate. Callers with a partial ply_ids
            # array must pass the explicit n_plies (the same value given
            # to apply_to_nodes) so both decay fields stay synced (#146).
            n_plies = int(ply_ids.max()) + 1 if len(ply_ids) > 0 else 1
        angle_sq = np.zeros(n_nodes, dtype=np.float64)

        x = nodes[:, 0]
        y = nodes[:, 1] if nodes.shape[1] >= 2 else np.zeros(n_nodes)

        for wrinkle in self.wrinkles:
            profile = wrinkle.profile
            k = wrinkle.ply_interface

            # Convert phase offset to geometric longitudinal shift
            delta_x = wrinkle.phase_offset * profile.wavelength / (2.0 * np.pi)
            x_shifted = x - delta_x

            # 1-D profile slope is broadcast over all nodes in C.
            slope = profile.slope(x_shifted)

            amp_scale = self._amplitude_scale_vec(wrinkle, x, y)
            decay = self._through_thickness_decay(ply_ids, k, n_plies) * amp_scale

            angle = np.arctan(np.abs(slope)) * decay
            angle_sq += angle * angle

        return np.sqrt(angle_sq)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def dual_wrinkle(
        cls,
        profile: Union[WrinkleProfile, WrinkleSurface3D],
        interface1: int,
        interface2: int,
        phase: float = 0.0,
    ) -> WrinkleConfiguration:
        """Create a standard dual-wrinkle configuration.

        Both wrinkles share the same profile geometry but are placed at
        different ply interfaces with a specified phase offset.

        Parameters
        ----------
        profile : WrinkleProfile or WrinkleSurface3D
            Wrinkle geometry (shared by both wrinkles).
        interface1 : int
            Ply interface index for the first (reference) wrinkle.
        interface2 : int
            Ply interface index for the second wrinkle.
        phase : float
            Phase offset of the second wrinkle relative to the first
            [radians]. Shortcut morphology presets:

            - ``phase=0`` : Stack
            - ``phase=np.pi/2`` : Convex
            - ``phase=-np.pi/2`` : Concave

        Returns
        -------
        WrinkleConfiguration
            Configuration with two wrinkles.

        Examples
        --------
        Create a convex dual-wrinkle at interfaces 11 and 12::

            >>> import numpy as np
            >>> config = WrinkleConfiguration.dual_wrinkle(
            ...     profile, interface1=11, interface2=12, phase=np.pi/2
            ... )
            >>> config.aggregate_morphology_factor("compression")  # doctest: +ELLIPSIS
            0.749...
        """
        w1 = WrinklePlacement(
            profile=profile,
            ply_interface=interface1,
            phase_offset=0.0,
        )
        w2 = WrinklePlacement(
            profile=profile,
            ply_interface=interface2,
            phase_offset=phase,
        )
        return cls([w1, w2])

    @classmethod
    def from_morphology_name(
        cls,
        name: str,
        profile: Union[WrinkleProfile, WrinkleSurface3D],
        interface1: int,
        interface2: int,
        decay_floor: float = 0.0,
    ) -> WrinkleConfiguration:
        """Create a dual-wrinkle configuration from a morphology name.

        Maps the morphology name to a phase offset using the
        :data:`MORPHOLOGY_PHASES` dictionary and delegates to
        :meth:`dual_wrinkle`.

        Parameters
        ----------
        name : str
            Morphology name: ``'stack'``, ``'convex'``, or ``'concave'``.
            Case-insensitive.
        profile : WrinkleProfile or WrinkleSurface3D
            Wrinkle geometry (shared by both wrinkles).
        interface1 : int
            Ply interface index for the first (reference) wrinkle.
        interface2 : int
            Ply interface index for the second wrinkle.

        Returns
        -------
        WrinkleConfiguration
            Dual-wrinkle configuration with the corresponding phase offset.

        Raises
        ------
        ValueError
            If *name* is not a recognised morphology.

        Examples
        --------
        ::

            >>> config = WrinkleConfiguration.from_morphology_name(
            ...     "concave", profile, interface1=11, interface2=12
            ... )
        """
        key = name.lower().strip()

        # Single-wrinkle morphologies: one wrinkle at midplane
        if key in SINGLE_WRINKLE_MODES:
            mid_interface = (interface1 + interface2) // 2
            placement = WrinklePlacement(
                profile=profile,
                ply_interface=mid_interface,
                phase_offset=0.0,
            )
            return cls([placement], decay_mode=key, decay_floor=decay_floor)

        # Dual-wrinkle morphologies: phase-offset pair
        if key not in MORPHOLOGY_PHASES:
            all_names = list(MORPHOLOGY_PHASES.keys()) + list(SINGLE_WRINKLE_MODES)
            raise ValueError(
                f"Unknown morphology '{name}'. "
                f"Choose from: {all_names}"
            )
        phase = MORPHOLOGY_PHASES[key]
        return cls.dual_wrinkle(profile, interface1, interface2, phase)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = self.n_wrinkles()
        interfaces = [w.ply_interface for w in self.wrinkles]
        phases = [f"{w.phase_offset:.3f}" for w in self.wrinkles]
        return (
            f"WrinkleConfiguration(n_wrinkles={n}, "
            f"interfaces={interfaces}, phases=[{', '.join(phases)}])"
        )
