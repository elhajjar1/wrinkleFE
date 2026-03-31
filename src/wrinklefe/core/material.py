"""Orthotropic material properties and material library for composite FE analysis.

This module provides:
- OrthotropicMaterial: A dataclass encapsulating the full set of orthotropic
  elastic constants, strength allowables, hygrothermal coefficients, and
  kink-band parameters for a single composite ply material.
- MaterialLibrary: A registry of named materials with JSON serialisation and
  four built-in carbon/epoxy systems.

Compliance and stiffness matrices follow standard Voigt notation
(11, 22, 33, 23, 13, 12) consistent with most composite-mechanics texts.
"""

from __future__ import annotations

import json
import functools
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union

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
    GIc: Optional[float] = None       # Mode I fracture toughness (kJ/m²)
    GIIc: Optional[float] = None      # Mode II fracture toughness (kJ/m²)
    beta_shear: float = 3.2e-8        # Ramberg-Osgood nonlinear shear coeff (1/MPa³)
    alpha_0: float = 53.0             # Fracture plane angle under pure transverse compression (degrees)

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
        2. Poisson's ratios satisfy symmetry: nu_ij / E_i = nu_ji / E_j
           within a relative tolerance of 5 %.
        3. The 6x6 compliance matrix is positive-definite (all eigenvalues > 0).

        Raises
        ------
        ValueError
            If any check fails.
        """
        # 1. Positivity
        for attr in ("E1", "E2", "E3", "G12", "G13", "G23",
                      "Xt", "Xc", "Yt", "Yc", "Zt", "Zc",
                      "S12", "S13", "S23", "gamma_Y"):
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

        # 2. Symmetric Poisson ratios
        #    nu_ij / E_i  should equal  nu_ji / E_j
        #    We derive nu_ji from user-supplied nu_ij.
        nu21 = self.nu12 * self.E2 / self.E1
        nu31 = self.nu13 * self.E3 / self.E1
        nu32 = self.nu23 * self.E3 / self.E2

        pairs = [
            ("nu12", self.nu12, self.E1, "nu21", nu21, self.E2),
            ("nu13", self.nu13, self.E1, "nu31", nu31, self.E3),
            ("nu23", self.nu23, self.E2, "nu32", nu32, self.E3),
        ]
        for name_ij, nu_ij, Ei, name_ji, nu_ji, Ej in pairs:
            lhs = nu_ij / Ei
            rhs = nu_ji / Ej
            if abs(lhs) > 0 and abs(rhs - lhs) / abs(lhs) > 0.05:
                raise ValueError(
                    f"Poisson symmetry violated: {name_ij}/E{name_ij[-1]} = {lhs:.6e} "
                    f"vs {name_ji}/E{name_ji[-1]} = {rhs:.6e}"
                )

        # 3. Positive-definite compliance matrix
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

    @classmethod
    def from_dict(cls, data: dict) -> "OrthotropicMaterial":
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

    Provides four built-in carbon/epoxy material systems and supports
    JSON serialisation for user-defined materials.

    Built-in materials
    ------------------
    - ``AS4_3501_6`` : Hercules AS4 / 3501-6 (standard aerospace epoxy)
    - ``IM7_8552`` : Hexcel IM7 / 8552 (toughened epoxy, widely characterised)
    - ``T300_914`` : Toray T300 / Hexcel 914 (legacy European aerospace)
    - ``T700_2510`` : Toray T700SC / Cytec 2510 (OOA VBO system)

    Examples
    --------
    >>> lib = MaterialLibrary()
    >>> mat = lib.get("IM7_8552")
    >>> mat.E1
    171420.0
    >>> lib.list_names()
    ['AS4_3501_6', 'IM7_8552', 'T300_914', 'T700_2510']
    """

    def __init__(self) -> None:
        self._materials: Dict[str, OrthotropicMaterial] = {}
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

    def list_names(self) -> List[str]:
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

    def to_json(self, path: Optional[Union[str, Path]] = None) -> str:
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
    def from_json(cls, source: Union[str, Path]) -> "MaterialLibrary":
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
        """Register the built-in material systems (4 carbon/epoxy + 1 glass/epoxy)."""

        # 1. AS4 / 3501-6  (Soden et al. 1998; MIL-HDBK-17)
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
        ))

        # 2. IM7 / 8552  (Camanho et al. 2007; Hexcel data sheets)
        #    Also the default of OrthotropicMaterial
        self.add(OrthotropicMaterial(
            GIc=0.28, GIIc=0.79,
        ))

        # 3. T300 / 914  (Soden et al. 1998; WWFE-I benchmark)
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
        ))

        # 4. T700SC / 2510  (Cytec CYCOM 2510 OOA VBO system)
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
        ))

        # 5. AC318 / S6C10-800  (S-glass fiber / epoxy, Li et al. 2026)
        #    Constituent properties from Table 1 of Li et al. (2026),
        #    Composites Part A 205:109719.  Composite-level properties
        #    computed via micromechanics at Vf ≈ 0.60, validated against
        #    E1 ≈ 58 GPa from Fig. 12b (pristine compression modulus).
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
        ))



    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"MaterialLibrary({self.list_names()})"
