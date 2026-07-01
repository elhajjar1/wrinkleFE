"""Orthotropic material properties and material library for composite FE analysis.

This module provides:

- OrthotropicMaterial: A dataclass encapsulating the full set of orthotropic
  elastic constants, strength allowables, hygrothermal coefficients, and
  kink-band parameters for a single composite ply material.
- MaterialLibrary: A registry of named materials with JSON serialisation and
  eleven built-in fibre-reinforced systems plus an isotropic neat-epoxy card.

Compliance and stiffness matrices follow standard Voigt notation
(11, 22, 33, 23, 13, 12) consistent with most composite-mechanics texts.
"""

from __future__ import annotations

import functools
import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np


@dataclass
class OrthotropicMaterial:
    """Orthotropic composite ply material with elastic, strength, and hygrothermal properties.

    Default values correspond to IM7/8552 carbon/epoxy prepreg.

    Parameters
    ----------
    E1, E2, E3 : float
        Young's moduli in the fibre (1), in-plane transverse (2), and
        through-thickness (3) directions [MPa].
    G12, G13, G23 : float
        Shear moduli [MPa].
    nu12, nu13, nu23 : float
        Poisson's ratios.  The reciprocal ratios (nu21, nu31, nu32) are
        derived from symmetry of the compliance matrix.
    Xt, Xc : float
        Longitudinal tensile and compressive strengths [MPa].
    Yt, Yc : float
        Transverse tensile and compressive strengths [MPa].
    Zt, Zc : float
        Through-thickness tensile and compressive strengths [MPa].
    S12, S13, S23 : float
        Shear strengths [MPa].
    alpha1, alpha2, alpha3 : float
        Coefficients of thermal expansion [1/K].
    beta1, beta2, beta3 : float
        Coefficients of moisture expansion (hygroscopic swelling) [1/%M].
    gamma_Y : float
        Matrix yield shear strain for the Budiansky-Fleck kink-band model.
    name : str
        Human-readable material designation.

    Notes
    -----
    The compliance matrix in Voigt notation (engineering strains for shear)::

        S = [[1/E1,     -nu12/E1, -nu13/E1, 0,      0,      0     ],
             [-nu12/E1,  1/E2,    -nu23/E2, 0,      0,      0     ],
             [-nu13/E1, -nu23/E2,  1/E3,    0,      0,      0     ],
             [0,         0,        0,       1/G23,  0,      0     ],
             [0,         0,        0,       0,      1/G13,  0     ],
             [0,         0,        0,       0,      0,      1/G12 ]]
    """

    # --- Elastic constants (MPa / dimensionless) ---
    E1: float = 171_420.0
    E2: float = 9_080.0
    E3: float = 9_080.0
    G12: float = 5_290.0
    G13: float = 5_290.0
    G23: float = 3_970.0
    nu12: float = 0.32
    nu13: float = 0.32
    nu23: float = 0.43

    # --- Strength allowables (MPa) ---
    Xt: float = 2_326.0
    Xc: float = 1_200.0
    Yt: float = 62.3
    Yc: float = 199.8
    Zt: float = 62.3
    Zc: float = 199.8
    S12: float = 92.3
    S13: float = 92.3
    S23: float = 75.0

    # --- Hygrothermal coefficients ---
    alpha1: float = -0.5e-6
    alpha2: float = 26.0e-6
    alpha3: float = 26.0e-6
    beta1: float = 0.0
    beta2: float = 0.4
    beta3: float = 0.4

    # --- Kink-band parameter ---
    gamma_Y: float = 0.02

    # --- LaRC04/05 parameters ---
    GIc: float | None = None       # Mode I fracture toughness (N/mm)
    GIIc: float | None = None      # Mode II fracture toughness (N/mm)
    beta_shear: float = 3.2e-8        # Ramberg-Osgood nonlinear shear coeff (1/MPa³)
    # Fracture plane angle under pure transverse compression (degrees)
    alpha_0: float = 53.0

    # --- Cohesive-interface defaults (Mode I / Mode II for the
    # bilinear traction-separation law used by the CZM Phase 3 path).
    # ``sigma_max`` is the peak normal traction (Mode I) and ``tau_max``
    # is the peak shear traction (Mode II) on the ply-to-ply interface.
    # Values default to typical CFRP epoxy literature; per-material
    # overrides are supplied by ``MaterialLibrary._load_builtins``.
    sigma_max: float = 50.0           # MPa
    tau_max: float = 75.0             # MPa

    # --- Identifier ---
    name: str = "IM7_8552"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Run validation checks after initialisation."""
        self.validate()

    def validate(self) -> None:
        """Validate material property consistency.

        Checks
        ------
        1. All moduli and strengths are strictly positive.
        2. Optional LaRC04/05 properties are within their physical ranges.
        3. The 6x6 compliance matrix is positive-definite (all eigenvalues > 0).
           This implicitly enforces the orthotropic stability bounds on the
           Poisson ratios (e.g. ``|ν12| < sqrt(E1/E2)``); a user-supplied
           ν that drives the compliance non-physical fails here.

        Notes
        -----
        Poisson symmetry ``nu_ij / E_i == nu_ji / E_j`` is automatic — the
        dataclass only stores the major ratios (``nu12``, ``nu13``, ``nu23``)
        and derives the minor ratios from them via the symmetry relation
        (see :attr:`nu21`, :attr:`nu31`, :attr:`nu32`). A standalone
        symmetry check would compare a value against itself; the constraint
        is structurally guaranteed, not validated.

        Raises
        ------
        ValueError
            If any check fails.
        """
        # 1. Positivity
        for attr in ("E1", "E2", "E3", "G12", "G13", "G23",
                      "Xt", "Xc", "Yt", "Yc", "Zt", "Zc",
                      "S12", "S13", "S23", "gamma_Y",
                      "sigma_max", "tau_max"):
            val = getattr(self, attr)
            if val <= 0:
                raise ValueError(f"{attr} must be positive, got {val}")

        # 1b. Optional LaRC properties — positive when provided
        for attr in ("GIc", "GIIc"):
            val = getattr(self, attr)
            if val is not None and val <= 0:
                raise ValueError(f"{attr} must be positive when provided, got {val}")
        if self.beta_shear < 0:
            raise ValueError(f"beta_shear must be non-negative, got {self.beta_shear}")
        if not (0 < self.alpha_0 < 90):
            raise ValueError(f"alpha_0 must be in (0, 90) degrees, got {self.alpha_0}")

        # 2. Positive-definite compliance matrix. This catches Poisson values
        # that drive the elastic tensor non-physical (e.g. ν12 ≥ sqrt(E1/E2));
        # the standalone Poisson-symmetry check that used to live here was a
        # tautology — see the docstring above.
        S = self._build_compliance()
        eigvals = np.linalg.eigvalsh(S)
        if np.any(eigvals <= 0):
            raise ValueError(
                f"Compliance matrix is not positive-definite. "
                f"Eigenvalues: {eigvals}"
            )

    # ------------------------------------------------------------------
    # Matrix computations (cached)
    # ------------------------------------------------------------------

    def _build_compliance(self) -> np.ndarray:
        """Build the 6x6 compliance matrix [S] without caching.

        Returns
        -------
        np.ndarray
            Shape (6, 6). Voigt order: 11, 22, 33, 23, 13, 12.
        """
        S = np.zeros((6, 6), dtype=np.float64)

        S[0, 0] = 1.0 / self.E1
        S[1, 1] = 1.0 / self.E2
        S[2, 2] = 1.0 / self.E3
        S[3, 3] = 1.0 / self.G23
        S[4, 4] = 1.0 / self.G13
        S[5, 5] = 1.0 / self.G12

        S[0, 1] = S[1, 0] = -self.nu12 / self.E1
        S[0, 2] = S[2, 0] = -self.nu13 / self.E1
        S[1, 2] = S[2, 1] = -self.nu23 / self.E2

        return S

    @functools.cached_property
    def compliance_matrix(self) -> np.ndarray:
        """6x6 compliance matrix [S] in Voigt notation (11,22,33,23,13,12).

        Returns
        -------
        np.ndarray
            Shape (6, 6).  Units: 1/MPa for normal entries, 1/MPa for shear.
        """
        return self._build_compliance()

    @functools.cached_property
    def stiffness_matrix(self) -> np.ndarray:
        """6x6 stiffness matrix [C] = [S]^{-1}.

        Returns
        -------
        np.ndarray
            Shape (6, 6).  Units: MPa.
        """
        return np.linalg.inv(self.compliance_matrix)

    @functools.cached_property
    def reduced_stiffness(self) -> np.ndarray:
        """3x3 reduced stiffness matrix [Q] for plane-stress analysis.

        Plane-stress assumption: sigma_3 = tau_13 = tau_23 = 0.

        The non-zero components are::

            Q11 = E1 / (1 - nu12 * nu21)
            Q22 = E2 / (1 - nu12 * nu21)
            Q12 = nu12 * E2 / (1 - nu12 * nu21)
            Q66 = G12

        where nu21 = nu12 * E2 / E1.

        Returns
        -------
        np.ndarray
            Shape (3, 3) with ordering [11, 22, 12].
        """
        nu21 = self.nu12 * self.E2 / self.E1
        denom = 1.0 - self.nu12 * nu21

        Q = np.zeros((3, 3), dtype=np.float64)
        Q[0, 0] = self.E1 / denom
        Q[1, 1] = self.E2 / denom
        Q[0, 1] = Q[1, 0] = self.nu12 * self.E2 / denom
        Q[2, 2] = self.G12

        return Q

    # ------------------------------------------------------------------
    # Derived Poisson ratios
    # ------------------------------------------------------------------

    @property
    def nu21(self) -> float:
        """Minor Poisson's ratio nu21 from symmetry."""
        return self.nu12 * self.E2 / self.E1

    @property
    def nu31(self) -> float:
        """Minor Poisson's ratio nu31 from symmetry."""
        return self.nu13 * self.E3 / self.E1

    @property
    def nu32(self) -> float:
        """Minor Poisson's ratio nu32 from symmetry."""
        return self.nu23 * self.E3 / self.E2

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert material to a plain dictionary suitable for JSON serialisation."""
        return asdict(self)

    def blend(self, other: OrthotropicMaterial, w: float) -> OrthotropicMaterial:
        """Linear interpolation toward *other* by weight ``w`` in [0, 1].

        Returns a new material whose elastic constants and strength
        allowables are ``(1 - w) * self + w * other``.  Used for the
        graded resin-pocket transition (host fibre material at the lens
        boundary, ``w = 0``, smoothly to the neat resin at the lens
        centre, ``w = 1``), which removes the spurious stress
        concentration a binary fibre/resin jump produces.

        Parameters
        ----------
        other : OrthotropicMaterial
            Target material (reached at ``w = 1``).
        w : float
            Blend weight in [0, 1].

        Returns
        -------
        OrthotropicMaterial
            Interpolated material.  ``w = 0`` returns a copy of ``self``;
            ``w = 1`` a copy of ``other``.
        """
        w = float(min(max(w, 0.0), 1.0))
        if w <= 0.0:
            return replace(self)
        if w >= 1.0:
            return replace(other)
        a, b = self.to_dict(), other.to_dict()
        blended = dict(a)
        for k, av in a.items():
            bv = b.get(k)
            if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
                blended[k] = (1.0 - w) * av + w * bv
        blended["name"] = f"{self.name}~{other.name}@{w:.2f}"
        return OrthotropicMaterial.from_dict(blended)

    @classmethod
    def isotropic(
        cls,
        E: float,
        nu: float,
        *,
        name: str = "isotropic",
        St: float = 80.0,
        Sc: float = 120.0,
        Ss: float = 50.0,
        GIc: float | None = 0.25,
        GIIc: float | None = 0.75,
    ) -> OrthotropicMaterial:
        """Build an isotropic material as a degenerate orthotropic card.

        Used for the resin-pocket zone (bulk epoxy filling the lens the
        machined cosine insert leaves at a wrinkle crest, Li et al. 2024):
        a soft, fibre-free inclusion with matrix-level strengths.  All
        three Young's moduli equal ``E``, all Poisson ratios equal ``nu``,
        and the shear moduli follow the isotropic relation
        ``G = E / (2 (1 + nu))``.

        Parameters
        ----------
        E : float
            Young's modulus (MPa).  Must be > 0.
        nu : float
            Poisson's ratio.  Must satisfy ``-1 < nu < 0.5`` for the
            isotropic stiffness to stay positive-definite.
        name : str, optional
            Material identifier.  Default ``"isotropic"``.
        St, Sc, Ss : float, optional
            Tensile, compressive and shear strength allowables (MPa)
            applied uniformly to every direction.  Defaults are typical
            cured-epoxy values (80 / 120 / 50 MPa).
        GIc, GIIc : float or None, optional
            Mode I / II fracture toughnesses (N/mm).  Defaults 0.25 / 0.75.

        Returns
        -------
        OrthotropicMaterial
            Isotropic material card.
        """
        if not (E > 0):
            raise ValueError(f"isotropic E must be > 0, got {E}")
        if not (-1.0 < nu < 0.5):
            raise ValueError(
                f"isotropic nu must be in (-1, 0.5), got {nu}"
            )
        G = E / (2.0 * (1.0 + nu))
        return cls(
            E1=E, E2=E, E3=E,
            G12=G, G13=G, G23=G,
            nu12=nu, nu13=nu, nu23=nu,
            Xt=St, Xc=Sc, Yt=St, Yc=Sc, Zt=St, Zc=Sc,
            S12=Ss, S13=Ss, S23=Ss,
            GIc=GIc, GIIc=GIIc,
            name=name,
        )

    @classmethod
    def from_dict(cls, data: dict) -> OrthotropicMaterial:
        """Construct an OrthotropicMaterial from a dictionary.

        Parameters
        ----------
        data : dict
            Keys must match the dataclass field names.

        Returns
        -------
        OrthotropicMaterial
        """
        # Only pass keys that are valid field names to allow extra keys in JSON
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"OrthotropicMaterial(name={self.name!r}, "
            f"E1={self.E1:.0f}, E2={self.E2:.0f}, E3={self.E3:.0f}, "
            f"G12={self.G12:.0f}, nu12={self.nu12:.3f})"
        )


# ======================================================================
# MaterialLibrary
# ======================================================================

class MaterialLibrary:
    """Named collection of :class:`OrthotropicMaterial` instances.

    Provides eleven built-in fibre-reinforced composite systems (carbon,
    S-glass, and aramid / epoxy) plus an isotropic neat-epoxy card for the
    resin-pocket zone, and supports JSON serialisation for user-defined
    materials.

    Built-in materials
    ------------------
    - ``AS4_3501_6`` : Hercules AS4 / 3501-6 (standard aerospace epoxy)
    - ``IM7_8552`` : Hexcel IM7 / 8552 (toughened epoxy, widely characterised)
    - ``T300_914`` : Toray T300 / Hexcel 914 (legacy European aerospace)
    - ``T700_2510`` : Toray T700SC / Cytec 2510 (OOA VBO system)
    - ``AC318_S6C10`` : AC318 / S6C10-800 S-glass / epoxy, *moulded*
      realization (Li et al. 2026)
    - ``AC318_S6C10_vacbag`` : same prepreg, *vacuum-bag* realization
      (Li 2025; measured ``Xc = 335.5`` MPa, ``E1 = 50.8`` GPa)
    - ``T800S_M21`` : Hexcel T800S / M21 toughened epoxy (A350 / A400M primary)
    - ``IM10_8552`` : Hexcel IM10 / 8552 (high-strain toughened epoxy)
    - ``IM6G_3501_6`` : Hercules IM6G / 3501-6 carbon / epoxy (Hsiao &
      Daniel 1996 wavy-UD compression study; Dataset G)
    - ``S2_GLASS_EPOXY`` : Generic S-2 glass / epoxy (MIL-HDBK-17 / CMH-17)
    - ``KEVLAR49_EPOXY`` : Generic Kevlar 49 / epoxy aramid system
    - ``EPOXY_S6C10`` : Isotropic neat-epoxy card (the fibre-free resin
      lens at a machined wrinkle crest; built via
      :meth:`OrthotropicMaterial.isotropic`)

    Examples
    --------
    >>> lib = MaterialLibrary()
    >>> mat = lib.get("IM7_8552")
    >>> mat.E1
    171420.0
    >>> sorted(lib.list_names())  # doctest: +NORMALIZE_WHITESPACE
    ['AC318_S6C10', 'AC318_S6C10_vacbag', 'AS4_3501_6', 'EPOXY_S6C10',
     'IM10_8552', 'IM6G_3501_6', 'IM7_8552', 'KEVLAR49_EPOXY',
     'S2_GLASS_EPOXY', 'T300_914', 'T700_2510', 'T800S_M21']
    """

    def __init__(self) -> None:
        self._materials: dict[str, OrthotropicMaterial] = {}
        self._load_builtins()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, name: str) -> OrthotropicMaterial:
        """Retrieve a material by name.

        Parameters
        ----------
        name : str
            Case-sensitive material name.

        Returns
        -------
        OrthotropicMaterial

        Raises
        ------
        KeyError
            If the name is not in the library.
        """
        if name not in self._materials:
            raise KeyError(
                f"Material '{name}' not found. "
                f"Available: {self.list_names()}"
            )
        return self._materials[name]

    def add(self, material: OrthotropicMaterial) -> None:
        """Add or replace a material in the library.

        Parameters
        ----------
        material : OrthotropicMaterial
            The material to register.  Its ``name`` attribute is used as the
            dictionary key.
        """
        self._materials[material.name] = material

    def list_names(self) -> list[str]:
        """Return a sorted list of all material names in the library.

        Returns
        -------
        list of str
        """
        return sorted(self._materials.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._materials

    def __len__(self) -> int:
        return len(self._materials)

    # ------------------------------------------------------------------
    # JSON serialisation
    # ------------------------------------------------------------------

    def to_json(self, path: str | Path | None = None) -> str:
        """Serialise the entire library to a JSON string.

        Parameters
        ----------
        path : str or Path, optional
            If given, the JSON is also written to this file.

        Returns
        -------
        str
            JSON text.
        """
        data = {
            name: mat.to_dict() for name, mat in self._materials.items()
        }
        text = json.dumps(data, indent=2, sort_keys=True)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, source: str | Path) -> MaterialLibrary:
        """Create a library from a JSON file or string.

        If *source* is a path to an existing file it is read; otherwise
        it is interpreted as a JSON string.  The built-in materials are
        loaded first, then the JSON entries are added (overwriting
        built-ins with the same name).

        Parameters
        ----------
        source : str or Path
            File path or raw JSON text.

        Returns
        -------
        MaterialLibrary
        """
        lib = cls()
        p = Path(source) if len(str(source)) < 1024 else None
        if p is not None and p.is_file():
            text = p.read_text(encoding="utf-8")
        else:
            text = str(source)
        data = json.loads(text)
        for name, mat_dict in data.items():
            mat_dict.setdefault("name", name)
            lib.add(OrthotropicMaterial.from_dict(mat_dict))
        return lib

    # ------------------------------------------------------------------
    # Built-in materials
    # ------------------------------------------------------------------

    def _load_builtins(self) -> None:
        """Register the built-in material systems (carbon/epoxy, glass/epoxy,
        and aramid/epoxy reference systems)."""

        # 1. AS4 / 3501-6  (Soden et al. 1998; MIL-HDBK-17)
        #    Cohesive defaults: sigma_max=60, tau_max=80 (typical AS4-
        #    epoxy interface values; spec).  GIc/GIIc kept at the legacy
        #    library values to preserve backwards compatibility with
        #    onset-KD regression tests.
        self.add(OrthotropicMaterial(
            name="AS4_3501_6",
            E1=126_000.0, E2=11_000.0, E3=11_000.0,
            G12=6_600.0, G13=6_600.0, G23=3_930.0,
            nu12=0.28, nu13=0.28, nu23=0.40,
            Xt=1_950.0, Xc=1_480.0,
            Yt=48.0, Yc=200.0,
            Zt=48.0, Zc=200.0,
            S12=79.0, S13=79.0, S23=57.0,
            alpha1=-1.0e-6, alpha2=26.0e-6, alpha3=26.0e-6,
            beta1=0.0, beta2=0.4, beta3=0.4,
            gamma_Y=0.02,
            GIc=0.20, GIIc=0.80, beta_shear=2.5e-8, alpha_0=53.0,
            sigma_max=60.0, tau_max=80.0,
        ))

        # 2. IM7 / 8552  (Camanho et al. 2007; Hexcel data sheets)
        #    Also the default of OrthotropicMaterial.  Cohesive defaults
        #    sigma_max=80, tau_max=90 per Camanho & Davila (NASA/TM-
        #    2002-211737, Table 1).  GIc/GIIc kept at the legacy library
        #    values to preserve backwards compatibility.
        self.add(OrthotropicMaterial(
            GIc=0.28, GIIc=0.79,
            sigma_max=80.0, tau_max=90.0,
        ))

        # 3. T300 / 914  (Soden et al. 1998; WWFE-I benchmark)
        #    Cohesive defaults: sigma_max=50, tau_max=70 (typical T300-
        #    epoxy interface values; spec).  GIc/GIIc kept at the legacy
        #    library values.
        self.add(OrthotropicMaterial(
            name="T300_914",
            E1=138_000.0, E2=11_000.0, E3=11_000.0,
            G12=5_500.0, G13=5_500.0, G23=3_930.0,
            nu12=0.28, nu13=0.28, nu23=0.40,
            Xt=1_500.0, Xc=900.0,
            Yt=27.0, Yc=200.0,
            Zt=27.0, Zc=200.0,
            S12=80.0, S13=80.0, S23=60.0,
            alpha1=-0.7e-6, alpha2=25.0e-6, alpha3=25.0e-6,
            beta1=0.0, beta2=0.35, beta3=0.35,
            gamma_Y=0.02,
            GIc=0.15, GIIc=0.45, alpha_0=53.0,
            sigma_max=50.0, tau_max=70.0,
        ))

        # 4. T700SC / 2510  (Cytec CYCOM 2510 OOA VBO system)
        #    Cohesive defaults: generic CFRP epoxy — no published value
        #    found specifically for T700/2510; fall back to typical CFRP
        #    interface strengths (sigma_max=50, tau_max=75 from spec).
        self.add(OrthotropicMaterial(
            name="T700_2510",
            E1=132_000.0, E2=10_300.0, E3=10_300.0,
            G12=5_700.0, G13=5_700.0, G23=3_770.0,
            nu12=0.30, nu13=0.30, nu23=0.40,
            Xt=2_400.0, Xc=1_300.0,
            Yt=55.0, Yc=230.0,
            Zt=55.0, Zc=230.0,
            S12=89.0, S13=89.0, S23=65.0,
            alpha1=-0.6e-6, alpha2=28.0e-6, alpha3=28.0e-6,
            beta1=0.0, beta2=0.38, beta3=0.38,
            gamma_Y=0.02,
            GIc=0.23, GIIc=0.90, alpha_0=53.0,
            sigma_max=50.0, tau_max=75.0,
        ))

        # 5. AC318 / S6C10-800  (S-glass fiber / epoxy, Li et al. 2026)
        #    Constituent properties from Table 1 of Li et al. (2026),
        #    Composites Part A 205:109719.  Composite-level properties
        #    computed via micromechanics at Vf ≈ 0.60, validated against
        #    E1 ≈ 58 GPa from Fig. 12b (pristine compression modulus).
        #    Cohesive defaults: sigma_max=70, tau_max=90 (typical glass-
        #    fiber/epoxy interface; spec).  GIc/GIIc kept at the legacy
        #    library values.
        self.add(OrthotropicMaterial(
            name="AC318_S6C10",
            E1=58_000.0, E2=12_000.0, E3=12_000.0,
            G12=5_500.0, G13=5_500.0, G23=4_000.0,
            nu12=0.28, nu13=0.28, nu23=0.40,
            Xt=1_200.0, Xc=830.0,
            Yt=40.0, Yc=150.0,
            Zt=40.0, Zc=150.0,
            S12=60.0, S13=60.0, S23=45.0,
            alpha1=5.0e-6, alpha2=25.0e-6, alpha3=25.0e-6,
            beta1=0.0, beta2=0.3, beta3=0.3,
            gamma_Y=0.02,
            GIc=0.25, GIIc=0.75, alpha_0=53.0,
            sigma_max=70.0, tau_max=90.0,
        ))

        # 5b. S6C10-800 neat epoxy (resin-pocket zone for Li 2024/2025
        #     UD glass datasets).  The machined cosine resin insert that
        #     creates the wrinkle is co-cured bulk epoxy, so the lens it
        #     leaves at the crest is fibre-free matrix.  Isotropic:
        #     E ≈ 3.5 GPa, nu ≈ 0.35 (typical cured aerospace epoxy),
        #     matrix-level strengths consistent with the AC318 transverse
        #     allowables (Yt 40 / Yc 150 / S 60 MPa).  Built via the
        #     isotropic constructor so the degenerate-orthotropic card
        #     stays positive-definite.
        self.add(OrthotropicMaterial.isotropic(
            3_500.0, 0.35,
            name="EPOXY_S6C10",
            St=40.0, Sc=150.0, Ss=60.0,
            GIc=0.25, GIIc=0.75,
        ))

        # 5c. AC318 S-glass / S6C10-800 — vacuum-bag realization (Li 2025,
        #     Dataset F).  Same prepreg as the moulded AC318_S6C10 card
        #     (Dataset E) but oven-cured under a 1 bar vacuum bag rather
        #     than 2.9 MPa mould, giving lower consolidation: MEASURED
        #     pristine modulus E1 = 50.8 GPa and MEASURED pristine
        #     compressive strength Xc = 335.5 MPa (~half the moulded
        #     value).  E2/G12/Yt/... are inherited from the base AC318 card
        #     (no separate vacuum-bag measurements) and are approximate.
        #     The two realizations cannot share a normalization — F's
        #     335.5 plate gives KD > 1 for every E specimen (see
        #     VALIDATION_DATA section 3).
        self.add(OrthotropicMaterial(
            name="AC318_S6C10_vacbag",
            E1=50_800.0, E2=12_000.0, E3=12_000.0,
            G12=5_500.0, G13=5_500.0, G23=4_000.0,
            nu12=0.28, nu13=0.28, nu23=0.40,
            Xt=1_200.0, Xc=335.5,
            Yt=40.0, Yc=150.0,
            Zt=40.0, Zc=150.0,
            S12=60.0, S13=60.0, S23=45.0,
            alpha1=5.0e-6, alpha2=25.0e-6, alpha3=25.0e-6,
            beta1=0.0, beta2=0.3, beta3=0.3,
            gamma_Y=0.02,
            GIc=0.25, GIIc=0.75, alpha_0=53.0,
            sigma_max=70.0, tau_max=90.0,
        ))

        # 6. T800S / M21  (Hexcel T800S fibre / M21 toughened epoxy).
        #    Properties from Hexcel HexPly M21 product data sheet and the
        #    open characterisation in Catalanotti, Camanho & Marques (2013)
        #    "Three-dimensional failure criteria for fibre-reinforced
        #    laminates", Composite Structures 95:63-79 (Table 1) and
        #    Vogler, Camanho et al. (2013) "Modelling the inelastic
        #    deformation and fracture of polymer composites — Part II",
        #    Mech. Mater. 59:43-58.  Vf ≈ 0.59.
        self.add(OrthotropicMaterial(
            name="T800S_M21",
            E1=157_000.0, E2=8_500.0, E3=8_500.0,
            G12=4_200.0, G13=4_200.0, G23=2_800.0,
            nu12=0.35, nu13=0.35, nu23=0.50,
            Xt=2_950.0, Xc=1_680.0,
            Yt=70.0, Yc=290.0,
            Zt=70.0, Zc=290.0,
            S12=98.0, S13=98.0, S23=80.0,
            alpha1=-0.4e-6, alpha2=30.0e-6, alpha3=30.0e-6,
            beta1=0.0, beta2=0.4, beta3=0.4,
            gamma_Y=0.02,
            GIc=0.21, GIIc=0.77, alpha_0=53.0,
            sigma_max=80.0, tau_max=90.0,
        ))

        # 7. IM10 / 8552  (Hexcel IM10 fibre / 8552 toughened epoxy).
        #    Hexcel IM10 product data sheet (fibre Ef ≈ 310 GPa, Xt_fibre ≈
        #    6964 MPa) with HexPly 8552 matrix properties.  Lamina-level
        #    values are bracketed by Lopes et al. (2007) "Physically-sound
        #    simulation of low-velocity impact on fiber reinforced laminates"
        #    and Hexcel IM10/8552 prepreg datasheet (Table 1, 60% Vf cured
        #    ply properties).
        self.add(OrthotropicMaterial(
            name="IM10_8552",
            E1=185_000.0, E2=9_400.0, E3=9_400.0,
            G12=5_500.0, G13=5_500.0, G23=3_900.0,
            nu12=0.32, nu13=0.32, nu23=0.43,
            Xt=3_310.0, Xc=1_690.0,
            Yt=63.0, Yc=240.0,
            Zt=63.0, Zc=240.0,
            S12=95.0, S13=95.0, S23=75.0,
            alpha1=-0.4e-6, alpha2=26.0e-6, alpha3=26.0e-6,
            beta1=0.0, beta2=0.4, beta3=0.4,
            gamma_Y=0.02,
            GIc=0.28, GIIc=0.79, alpha_0=53.0,
            sigma_max=80.0, tau_max=90.0,
        ))

        # 7b. IM6G / 3501-6  (Hercules IM6G carbon / 3501-6 epoxy).
        #     Lamina properties from Hsiao & Daniel (1996) "Effect of fiber
        #     waviness on stiffness and strength reduction of unidirectional
        #     composites under compressive loading", Compos. Sci. Technol.
        #     56(5):581-593, Table 1 (Vf 0.66).  E3/G13/nu13 set by
        #     transverse isotropy; G23 ~= E2 / (2(1+nu23)); Z = transverse
        #     allowables; S23 estimated.  Cohesive defaults: brittle 3501-6
        #     epoxy interface (sigma_max=60, tau_max=90; spec).  This is the
        #     material card for Dataset G of the validation database.
        self.add(OrthotropicMaterial(
            name="IM6G_3501_6",
            E1=169_000.0, E2=9_000.0, E3=9_000.0,
            G12=6_500.0, G13=6_500.0, G23=3_200.0,
            nu12=0.31, nu13=0.31, nu23=0.40,
            Xt=2_236.0, Xc=1_682.0,
            Yt=46.2, Yc=213.0,
            Zt=46.2, Zc=213.0,
            S12=72.8, S13=72.8, S23=50.0,
            alpha1=-0.4e-6, alpha2=26.0e-6, alpha3=26.0e-6,
            beta1=0.0, beta2=0.4, beta3=0.4,
            gamma_Y=0.02,
            GIc=0.10, GIIc=0.45, alpha_0=53.0,
            sigma_max=60.0, tau_max=90.0,
        ))

        # 8. S-2 Glass / Epoxy  (generic).
        #    Reference values from MIL-HDBK-17-3F / CMH-17 Vol 2 (S-2 glass
        #    laminate tables) and Daniel & Ishai (2006) "Engineering
        #    Mechanics of Composite Materials", 2nd ed., Table A.4.
        #    Representative cured-ply values at Vf ≈ 0.60.
        #    Interlaminar fracture toughness: glass/epoxy is markedly tougher
        #    than carbon/epoxy (published GIc ≈ 0.4–1.0 N/mm, GIIc ≈ 2–4×
        #    GIc); GIc = 0.8, GIIc = 2.0 N/mm are representative values in
        #    that range (CMH-17 Vol 2 / Daniel & Ishai 2006), consistent
        #    with the ~2.5× GIIc/GIc ratio of the carbon presets above.
        self.add(OrthotropicMaterial(
            name="S2_GLASS_EPOXY",
            E1=52_000.0, E2=19_000.0, E3=19_000.0,
            G12=6_700.0, G13=6_700.0, G23=6_500.0,
            nu12=0.30, nu13=0.30, nu23=0.46,
            Xt=1_700.0, Xc=970.0,
            Yt=49.0, Yc=158.0,
            Zt=49.0, Zc=158.0,
            S12=83.0, S13=83.0, S23=60.0,
            alpha1=6.6e-6, alpha2=19.7e-6, alpha3=19.7e-6,
            beta1=0.0, beta2=0.3, beta3=0.3,
            gamma_Y=0.02,
            GIc=0.80, GIIc=2.00, alpha_0=53.0,
            sigma_max=70.0, tau_max=90.0,
        ))

        # 9. Kevlar 49 / Epoxy  (generic aramid).
        #    Reference values from Daniel & Ishai (2006) Table A.4 and
        #    Kaw (2005) "Mechanics of Composite Materials", 2nd ed.,
        #    Table 2.2.  Representative cured-ply values at Vf ≈ 0.60.
        #    Note: longitudinal compressive strength of aramid prepregs
        #    is markedly lower than tensile because of fibre kinking.
        #    Interlaminar fracture toughness: aramid/epoxy GIc ≈ 0.3–0.6
        #    N/mm (moderate matrix-dominated Mode I, raised by fibre
        #    bridging); GIc = 0.4, GIIc = 1.0 N/mm are representative values
        #    in that range (Daniel & Ishai 2006), at the ~2.5× GIIc/GIc
        #    ratio of the other presets.
        self.add(OrthotropicMaterial(
            name="KEVLAR49_EPOXY",
            E1=76_000.0, E2=5_500.0, E3=5_500.0,
            G12=2_300.0, G13=2_300.0, G23=1_800.0,
            nu12=0.34, nu13=0.34, nu23=0.50,
            Xt=1_400.0, Xc=235.0,
            Yt=12.0, Yc=53.0,
            Zt=12.0, Zc=53.0,
            S12=34.0, S13=34.0, S23=25.0,
            alpha1=-4.0e-6, alpha2=79.0e-6, alpha3=79.0e-6,
            beta1=0.0, beta2=0.6, beta3=0.6,
            gamma_Y=0.02,
            GIc=0.40, GIIc=1.00, alpha_0=53.0,
            sigma_max=40.0, tau_max=70.0,
        ))



    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"MaterialLibrary({self.list_names()})"
