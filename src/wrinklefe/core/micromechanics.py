"""Constituent-based micromechanics: fibre + matrix + fibre volume fraction to a ply.

The rest of the package treats a ply as a *given* set of orthotropic
constants (:class:`~wrinklefe.core.material.OrthotropicMaterial`).  This
module supplies the missing upstream step: given the **constituents** — a
transversely isotropic fibre, an isotropic matrix, and a fibre volume
fraction ``Vf`` — it predicts the cured-ply elastic constants and thermal
expansion coefficients.  The mixing rules and the cited constituent presets live here; the ply
constructor that applies them lands next.

Why it exists
-------------
It answers "what if my fibre volume fraction is 0.55 rather than 0.60?"
without hand-editing a material card, and it is the prerequisite for
modelling resin squeeze-out under a constrained wrinkle (issue #379),
where the local ``Vf`` varies from element to element.

Mixing rules
------------
========================  ==================================================
Property                  Rule
========================  ==================================================
``E1``                    Voigt rule of mixtures (:func:`e1_rule_of_mixtures`)
``nu12``                  Rule of mixtures (:func:`nu12_rule_of_mixtures`)
``E2``                    Halpin-Tsai, ``xi = 2`` (:func:`e2_halpin_tsai`)
``G12``                   Halpin-Tsai, ``xi = 1`` (:func:`g12_halpin_tsai`)
``nu23``                  Rule of mixtures (:func:`nu23_rule_of_mixtures`)
``G23``                   ``E2 / (2 (1 + nu23))`` (transverse isotropy,
                          :func:`g23_transverse_isotropy`)
``alpha1``, ``alpha2``    Schapery (:func:`alpha1_schapery`,
                          :func:`alpha2_schapery`)
========================  ==================================================

References: J. C. Halpin and J. L. Kardos, "The Halpin-Tsai equations: a
review", Polym. Eng. Sci. 16(5):344-352 (1976); R. A. Schapery, "Thermal
expansion coefficients of composite materials based on energy principles",
J. Compos. Mater. 2(3):380-404 (1968); I. M. Daniel and O. Ishai,
*Engineering Mechanics of Composite Materials*, 2nd ed., Oxford (2006),
chapter 3 and Tables A.2/A.3.

What is **not** predicted
-------------------------
**Strengths are not a function of ``Vf`` in any of these rules.**
``Xt``/``Xc``/``Yt``/``Yc``/``Zt``/``Zc``/``S12``/``S13``/``S23``, the
fracture toughnesses, the cohesive tractions and the kink-band parameter
are *carried over* from a reference ply via ``strengths_from=`` (or left at
the dataclass defaults).  A "``Vf``-scaled strength" model would be quietly
wrong: longitudinal strength is set by fibre strength statistics and
misalignment, and transverse/shear strengths by the matrix and the
fibre-matrix interface, none of which these stiffness-averaging rules
describe.  Scale strengths only with a model calibrated for the purpose.

Accuracy
--------
The rules are engineering approximations, and the measured spread against
the six shipped library presets that document their own ``Vf`` is wide:
``E1`` within 12 %, ``nu12`` within 26 %, ``E2`` within 33 %, ``G12``
within 32 % — except Kevlar 49 / epoxy, where Halpin-Tsai over-predicts
``G12`` by 84 % (published aramid ``G12f`` values scatter by an order of
magnitude).  ``nu23`` is under-predicted by 30-55 % throughout.
``tests/test_micromechanics.py`` pins every one of those deviations.

Use the predictions as a *trend* model — how properties move with ``Vf`` —
not as a replacement for measured allowables.  The intended workflow is to
anchor on a measured ply at a known ``Vf`` and read the *ratio* the model
gives between that ``Vf`` and the local one.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from wrinklefe.core.material import OrthotropicMaterial

logger = logging.getLogger(__name__)

__all__ = [
    "FiberProperties",
    "MatrixProperties",
    "FIBER_PRESETS",
    "MATRIX_PRESETS",
    "halpin_tsai",
    "e1_rule_of_mixtures",
    "nu12_rule_of_mixtures",
    "e2_halpin_tsai",
    "g12_halpin_tsai",
    "nu23_rule_of_mixtures",
    "g23_transverse_isotropy",
    "alpha1_schapery",
    "alpha2_schapery",
]


# ======================================================================
# Constituent property containers
# ======================================================================

@dataclass(frozen=True)
class FiberProperties:
    """Transversely isotropic reinforcing fibre.

    Parameters
    ----------
    E1f : float
        Axial (fibre-direction) Young's modulus [MPa].  This is the value
        quoted on fibre data sheets as "tensile modulus".
    E2f : float
        Transverse Young's modulus [MPa].  For PAN carbon and aramid this
        is *much* smaller than ``E1f`` (the graphitic/polymer chains align
        with the fibre axis) and is **never** published on a data sheet —
        it comes from handbook tables (see :data:`FIBER_PRESETS`).  For
        glass the fibre is isotropic and ``E2f == E1f``.
    G12f : float
        Axial shear modulus [MPa].
    nu12f : float
        Major Poisson's ratio of the fibre.
    alpha1f, alpha2f : float
        Axial and transverse coefficients of thermal expansion [1/K].
    G23f : float or None, optional
        Transverse shear modulus [MPa].  Published for only a few fibres;
        when ``None`` it is estimated as ``E2f / (2 (1 + nu12f))``, i.e.
        isotropy is assumed in the 2-3 plane.  That is a stated modelling
        assumption, not measured data.
    name : str
        Fibre designation.

    Raises
    ------
    ValueError
        If a modulus is non-positive or ``nu12f`` violates the
        transversely isotropic stability bound ``nu12f**2 < E1f / E2f``.
    """

    E1f: float
    E2f: float
    G12f: float
    nu12f: float
    alpha1f: float
    alpha2f: float
    G23f: float | None = None
    name: str = "fiber"

    def __post_init__(self) -> None:
        for attr in ("E1f", "E2f", "G12f"):
            val = float(getattr(self, attr))
            if val <= 0:
                raise ValueError(f"{attr} must be positive, got {val}")
        if self.G23f is not None and self.G23f <= 0:
            raise ValueError(
                f"G23f must be positive when provided, got {self.G23f}"
            )
        bound = (self.E1f / self.E2f) ** 0.5
        if not (-bound < self.nu12f < bound):
            raise ValueError(
                f"nu12f must satisfy |nu12f| < sqrt(E1f/E2f) = {bound:.3f}, "
                f"got {self.nu12f}"
            )

    @property
    def G23f_effective(self) -> float:
        """Transverse shear modulus [MPa], estimated when not supplied.

        Returns ``G23f`` when the preset carries a measured value; otherwise
        ``E2f / (2 (1 + nu12f))`` — 2-3 isotropy assumed.
        """
        if self.G23f is not None:
            return self.G23f
        return self.E2f / (2.0 * (1.0 + self.nu12f))

    @property
    def nu23f_effective(self) -> float:
        """Transverse Poisson's ratio of the fibre.

        ``E2f / (2 G23f) - 1``.  Equals ``nu12f`` when ``G23f`` was not
        supplied and 2-3 isotropy was assumed.
        """
        return self.E2f / (2.0 * self.G23f_effective) - 1.0


@dataclass(frozen=True)
class MatrixProperties:
    """Isotropic (neat resin) matrix.

    Parameters
    ----------
    Em : float
        Young's modulus [MPa].
    num : float
        Poisson's ratio.
    alpham : float
        Coefficient of thermal expansion [1/K].
    name : str
        Resin designation.

    Raises
    ------
    ValueError
        If ``Em <= 0`` or ``num`` is outside the isotropic stability range
        ``-1 < num < 0.5``.
    """

    Em: float
    num: float
    alpham: float
    name: str = "matrix"

    def __post_init__(self) -> None:
        if self.Em <= 0:
            raise ValueError(f"Em must be positive, got {self.Em}")
        if not (-1.0 < self.num < 0.5):
            raise ValueError(
                f"num must be in (-1, 0.5) for an isotropic solid, got {self.num}"
            )

    @property
    def Gm(self) -> float:
        """Shear modulus [MPa], ``Em / (2 (1 + num))``."""
        return self.Em / (2.0 * (1.0 + self.num))

    @classmethod
    def from_material(cls, material: OrthotropicMaterial) -> MatrixProperties:
        """Build matrix constituent data from an isotropic material card.

        Lets an existing neat-resin card in the library (e.g.
        ``EPOXY_S6C10``) be reused as the matrix constituent instead of
        duplicating the resin data here.

        Parameters
        ----------
        material : OrthotropicMaterial
            An isotropic card — typically one built with
            :meth:`OrthotropicMaterial.isotropic`.  ``E1``, ``nu12`` and
            ``alpha1`` are taken as ``Em``, ``num`` and ``alpham``.  A
            material whose directional constants differ is accepted but
            logged as a warning, since only the 1-direction values are used.

        Returns
        -------
        MatrixProperties
        """
        isotropic = (
            abs(material.E1 - material.E2) <= 1e-9 * abs(material.E1)
            and abs(material.E1 - material.E3) <= 1e-9 * abs(material.E1)
            and abs(material.nu12 - material.nu23) <= 1e-9
        )
        if not isotropic:
            logger.warning(
                "MatrixProperties.from_material: %s is not isotropic "
                "(E1=%.1f, E2=%.1f, nu12=%.3f, nu23=%.3f); using the "
                "1-direction constants as the matrix properties.",
                material.name, material.E1, material.E2,
                material.nu12, material.nu23,
            )
        return cls(
            Em=material.E1,
            num=material.nu12,
            alpham=material.alpha1,
            name=material.name,
        )


# ======================================================================
# Mixing rules
# ======================================================================

def _check_Vf(Vf: float) -> None:
    """Raise ``ValueError`` unless ``0 <= Vf <= 1``."""
    if not (0.0 <= Vf <= 1.0):
        raise ValueError(f"Vf must be in [0, 1], got {Vf}")


def halpin_tsai(Vf: float, Pf: float, Pm: float, xi: float) -> float:
    """Halpin-Tsai semi-empirical estimate of a matrix-dominated property.

    .. math::

        \\eta = \\frac{P_f / P_m - 1}{P_f / P_m + \\xi}, \\qquad
        P = P_m \\frac{1 + \\xi \\eta V_f}{1 - \\eta V_f}

    The reinforcement-geometry factor ``xi`` interpolates between the
    Reuss (series, ``xi -> 0``) and Voigt (parallel, ``xi -> inf``) bounds.

    Parameters
    ----------
    Vf : float
        Fibre volume fraction, in [0, 1].
    Pf, Pm : float
        The fibre and matrix values of the property (same units).
    xi : float
        Reinforcement geometry factor, ``xi >= 0``.

    Returns
    -------
    float
        The predicted composite property; ``Pm`` at ``Vf = 0`` and ``Pf``
        at ``Vf = 1`` exactly.

    References
    ----------
    Halpin & Kardos (1976), Polym. Eng. Sci. 16(5):344-352.
    """
    _check_Vf(Vf)
    if Pf <= 0 or Pm <= 0:
        raise ValueError(f"Halpin-Tsai needs positive properties, got Pf={Pf}, Pm={Pm}")
    if xi < 0:
        raise ValueError(f"xi must be non-negative, got {xi}")
    ratio = Pf / Pm
    eta = (ratio - 1.0) / (ratio + xi)
    return Pm * (1.0 + xi * eta * Vf) / (1.0 - eta * Vf)


def e1_rule_of_mixtures(
    Vf: float, fiber: FiberProperties, matrix: MatrixProperties
) -> float:
    """Longitudinal modulus [MPa] by the Voigt rule of mixtures.

    .. math:: E_1 = V_f E_{1f} + (1 - V_f) E_m

    Iso-strain in the fibre direction.  This is the least approximate of
    the rules here, but it is an *upper* bound: real laminae fall a few per
    cent below it because of fibre waviness and misalignment.

    References
    ----------
    Daniel & Ishai (2006), *Engineering Mechanics of Composite Materials*,
    2nd ed., eq. (3.20).
    """
    _check_Vf(Vf)
    return Vf * fiber.E1f + (1.0 - Vf) * matrix.Em


def nu12_rule_of_mixtures(
    Vf: float, fiber: FiberProperties, matrix: MatrixProperties
) -> float:
    """Major Poisson's ratio by the rule of mixtures.

    .. math:: \\nu_{12} = V_f \\nu_{12f} + (1 - V_f) \\nu_m

    References
    ----------
    Daniel & Ishai (2006), eq. (3.23).
    """
    _check_Vf(Vf)
    return Vf * fiber.nu12f + (1.0 - Vf) * matrix.num


def e2_halpin_tsai(
    Vf: float, fiber: FiberProperties, matrix: MatrixProperties, xi: float = 2.0
) -> float:
    """Transverse modulus [MPa] by Halpin-Tsai.

    ``xi = 2`` is the standard value for the transverse modulus of a
    unidirectional ply with circular fibres (Halpin & Kardos 1976); raise
    it for a more closely packed / less compliant transverse response.
    """
    return halpin_tsai(Vf, fiber.E2f, matrix.Em, xi)


def g12_halpin_tsai(
    Vf: float, fiber: FiberProperties, matrix: MatrixProperties, xi: float = 1.0
) -> float:
    """In-plane shear modulus [MPa] by Halpin-Tsai.

    ``xi = 1`` is the standard value for the axial shear modulus of a
    unidirectional ply with circular fibres (Halpin & Kardos 1976).  It is
    known to *under*-predict ``G12`` for well-consolidated aerospace
    prepreg; ``xi`` is exposed so it can be calibrated per system.
    """
    return halpin_tsai(Vf, fiber.G12f, matrix.Gm, xi)


def nu23_rule_of_mixtures(
    Vf: float, fiber: FiberProperties, matrix: MatrixProperties
) -> float:
    """Transverse Poisson's ratio by the rule of mixtures.

    .. math:: \\nu_{23} = V_f \\nu_{23f} + (1 - V_f) \\nu_m

    where ``nu23f`` is :attr:`FiberProperties.nu23f_effective`.  The same
    averaging as :func:`nu12_rule_of_mixtures`, applied in the transverse
    plane.

    .. warning::

       ``nu23`` is the least reliable of the predicted constants.  This
       rule keeps it bounded between ``nu23f`` and ``nu_m`` — always
       physically admissible — but it ignores the constrained,
       near-incompressible transverse response of the matrix and therefore
       **under**-predicts ``nu23``: measured unidirectional plies sit
       around 0.4-0.5, roughly 40 % above this estimate.  The Halpin-Tsai
       alternative (predict ``G23``, then back out ``nu23``) was rejected
       because with an isotropic fibre such as glass it returns
       ``nu23`` up to 0.94 — admissible but not credible.  Override
       ``nu23`` on the returned card when a measured value is available.
    """
    _check_Vf(Vf)
    return Vf * fiber.nu23f_effective + (1.0 - Vf) * matrix.num


def g23_transverse_isotropy(E2: float, nu23: float) -> float:
    """Transverse shear modulus [MPa] from transverse isotropy.

    .. math:: G_{23} = \\frac{E_2}{2 (1 + \\nu_{23})}

    In the 2-3 plane a transversely isotropic ply is isotropic, so ``G23``
    is not an independent constant once ``E2`` and ``nu23`` are known.
    """
    if E2 <= 0:
        raise ValueError(f"E2 must be positive, got {E2}")
    if nu23 <= -1.0:
        raise ValueError(f"nu23 must be > -1, got {nu23}")
    return E2 / (2.0 * (1.0 + nu23))


def alpha1_schapery(
    Vf: float, fiber: FiberProperties, matrix: MatrixProperties
) -> float:
    """Longitudinal CTE [1/K] by Schapery's energy-principles result.

    .. math::

        \\alpha_1 = \\frac{V_f E_{1f} \\alpha_{1f}
                           + (1 - V_f) E_m \\alpha_m}
                          {V_f E_{1f} + (1 - V_f) E_m}

    The stiffness-weighted average: the fibres dominate, which is why a
    carbon/epoxy ply has a near-zero (often slightly negative) axial CTE.

    References
    ----------
    Schapery (1968), J. Compos. Mater. 2(3):380-404.
    """
    _check_Vf(Vf)
    Vm = 1.0 - Vf
    num = Vf * fiber.E1f * fiber.alpha1f + Vm * matrix.Em * matrix.alpham
    return num / (Vf * fiber.E1f + Vm * matrix.Em)


def alpha2_schapery(
    Vf: float, fiber: FiberProperties, matrix: MatrixProperties
) -> float:
    """Transverse CTE [1/K] by Schapery's energy-principles result.

    .. math::

        \\alpha_2 = (1 + \\nu_{12f}) \\alpha_{2f} V_f
                  + (1 + \\nu_m) \\alpha_m (1 - V_f)
                  - \\alpha_1 \\nu_{12}

    References
    ----------
    Schapery (1968), J. Compos. Mater. 2(3):380-404.
    """
    _check_Vf(Vf)
    Vm = 1.0 - Vf
    alpha1 = alpha1_schapery(Vf, fiber, matrix)
    nu12 = nu12_rule_of_mixtures(Vf, fiber, matrix)
    return (
        (1.0 + fiber.nu12f) * fiber.alpha2f * Vf
        + (1.0 + matrix.num) * matrix.alpham * Vm
        - alpha1 * nu12
    )


# ======================================================================
# Constituent presets
# ======================================================================
#
# Sourcing policy (each entry states its own sources):
#
#   * ``E1f`` and ``alpha1f`` come from the fibre manufacturer's product
#     data sheet wherever one exists — those are the two fibre constants
#     manufacturers actually publish.
#   * ``E2f``, ``G12f``, ``nu12f``, ``alpha2f`` (and ``G23f`` where it is
#     tabulated) come from Daniel & Ishai (2006) Table A.2, "Mechanical and
#     Thermal Properties of Representative Fibers".  No manufacturer
#     publishes transverse fibre constants.
#   * For a fibre that Table A.2 does not tabulate, the transverse set is
#     taken from the table's nearest same-class column and the substitution
#     is named explicitly in the comment.  Nothing here is invented: every
#     number is either a data-sheet value or a handbook value, and where a
#     handbook value belongs to a *different* fibre of the same class, the
#     comment says so.
#
# Abbreviations used below:
#   D&I   = I. M. Daniel and O. Ishai, "Engineering Mechanics of Composite
#           Materials", 2nd ed., Oxford University Press (2006), Table A.2
#           (fibres) / Table A.3 (matrices).
#   PDS   = manufacturer product data sheet (revision as noted).

FIBER_PRESETS: dict[str, FiberProperties] = {
    # --- Standard-modulus PAN carbon -------------------------------------
    # Hexcel HexTow AS4 PDS (2023): tensile modulus 231 GPa (33.5 Msi),
    # tensile strength 4501 MPa, CTE -0.63 ppm/degC, density 1.79 g/cm3.
    # Transverse set from D&I Table A.2, column "AS-4 Carbon"
    # (E2f 15 GPa, G12f 27 GPa, G23f 7 GPa, nu12f 0.20, alpha2f 15e-6).
    "AS4": FiberProperties(
        E1f=231_000.0, E2f=15_000.0, G12f=27_000.0, G23f=7_000.0,
        nu12f=0.20, alpha1f=-0.63e-6, alpha2f=15.0e-6, name="AS4",
    ),
    # Toray TORAYCA T300 PDS: tensile modulus 230 GPa, strength 3530 MPa,
    # density 1.76 g/cm3 (D&I Table A.2 agrees: 230 GPa).  Transverse set
    # and CTEs from D&I Table A.2, column "T-300 Carbon".
    "T300": FiberProperties(
        E1f=230_000.0, E2f=15_000.0, G12f=27_000.0, G23f=7_000.0,
        nu12f=0.20, alpha1f=-0.7e-6, alpha2f=12.0e-6, name="T300",
    ),
    # Toray TORAYCA T700S PDS: tensile modulus 230 GPa, strength 4900 MPa,
    # density 1.80 g/cm3.  T700S is not tabulated in D&I Table A.2; the
    # transverse set and CTEs are taken from the "T-300 Carbon" column —
    # the same standard-modulus PAN class at an identical axial modulus.
    "T700S": FiberProperties(
        E1f=230_000.0, E2f=15_000.0, G12f=27_000.0, G23f=7_000.0,
        nu12f=0.20, alpha1f=-0.7e-6, alpha2f=12.0e-6, name="T700S",
    ),
    # --- Intermediate-modulus PAN carbon ---------------------------------
    # Hexcel HexTow IM7 PDS (2023): tensile modulus 276 GPa (40.0 Msi,
    # chord 6000-1000), strength 5670 MPa, CTE -0.64 ppm/degC.  Transverse
    # set from D&I Table A.2, column "IM7 Carbon" (E2f 21 GPa, G12f 14 GPa,
    # nu12f 0.20, alpha2f 10e-6); that column quotes E1f = 290 GPa, and the
    # data-sheet 276 GPa is preferred for the axial modulus.  D&I does not
    # tabulate G23f for IM7, so it is estimated (see G23f_effective).
    "IM7": FiberProperties(
        E1f=276_000.0, E2f=21_000.0, G12f=14_000.0,
        nu12f=0.20, alpha1f=-0.64e-6, alpha2f=10.0e-6, name="IM7",
    ),
    # Hexcel HexTow IM10 PDS (2023): tensile modulus 313 GPa (45.4 Msi),
    # strength 6826 MPa, CTE -0.70 ppm/degC.  Not in D&I Table A.2; the
    # transverse set is taken from the "IM7 Carbon" column, the nearest
    # tabulated intermediate-modulus PAN fibre.
    "IM10": FiberProperties(
        E1f=313_000.0, E2f=21_000.0, G12f=14_000.0,
        nu12f=0.20, alpha1f=-0.70e-6, alpha2f=10.0e-6, name="IM10",
    ),
    # Hexcel HexTow IM6 PDS (2023): tensile modulus 279 GPa (40.5 Msi),
    # strength 5860 MPa.  IM6G is the G-sized grade of the same fibre.  Not
    # in D&I Table A.2; transverse set and alpha1f from the "IM7 Carbon"
    # column (the PDS does not quote a CTE for IM6).
    "IM6G": FiberProperties(
        E1f=279_000.0, E2f=21_000.0, G12f=14_000.0,
        nu12f=0.20, alpha1f=-0.2e-6, alpha2f=10.0e-6, name="IM6G",
    ),
    # Toray TORAYCA T800S PDS: tensile modulus 294 GPa, strength 5880 MPa,
    # CTE -0.4e-6 /degC, density 1.80 g/cm3.  Not in D&I Table A.2;
    # transverse set from the "IM7 Carbon" column (nearest tabulated
    # intermediate-modulus PAN fibre).
    "T800S": FiberProperties(
        E1f=294_000.0, E2f=21_000.0, G12f=14_000.0,
        nu12f=0.20, alpha1f=-0.4e-6, alpha2f=10.0e-6, name="T800S",
    ),
    # --- Glass and aramid ------------------------------------------------
    # D&I Table A.2, column "S-Glass": E 86 GPa, G 35 GPa, nu 0.23,
    # alpha 5.6e-6 /degC, strength 4500 MPa.  Glass fibre is isotropic, so
    # E2f = E1f and G23f = G12f.  Cross-check: the AGY S-2 Glass data sheet
    # quotes 86.9 GPa and nu = 0.22.
    "S2_GLASS": FiberProperties(
        E1f=86_000.0, E2f=86_000.0, G12f=35_000.0, G23f=35_000.0,
        nu12f=0.23, alpha1f=5.6e-6, alpha2f=5.6e-6, name="S2_GLASS",
    ),
    # D&I Table A.2, column "Kevlar 49 Aramid": E1f 131 GPa, E2f 7 GPa,
    # G12f 21 GPa, nu12f 0.33, alpha1f -2e-6, alpha2f 60e-6 /degC.
    "KEVLAR49": FiberProperties(
        E1f=131_000.0, E2f=7_000.0, G12f=21_000.0,
        nu12f=0.33, alpha1f=-2.0e-6, alpha2f=60.0e-6, name="KEVLAR49",
    ),
}
"""Fibre constituent presets, keyed by fibre designation.

Every value is either a manufacturer data-sheet value or a handbook value;
see the sourcing policy in the module source above each entry.
"""


MATRIX_PRESETS: dict[str, MatrixProperties] = {
    # D&I Table A.3, column "EPOXY (3501-6)": Em 4.3 GPa, Gm 1.60 GPa,
    # nu 0.35, alpha 45e-6 /degC.
    "EPOXY_3501_6": MatrixProperties(
        Em=4_300.0, num=0.35, alpham=45.0e-6, name="EPOXY_3501_6",
    ),
    # Hexcel HexPly 8552 PDS (2023), "Typical Neat Resin Data": tensile
    # modulus 4670 MPa, tensile strength 121 MPa, density 1.301 g/cm3.
    # The PDS publishes no Poisson's ratio or CTE for the neat resin; both
    # are taken as the typical cured-epoxy values of D&I Table A.3
    # (nu = 0.35, alpha = 45e-6 /degC).
    "EPOXY_8552": MatrixProperties(
        Em=4_670.0, num=0.35, alpham=45.0e-6, name="EPOXY_8552",
    ),
}
"""Matrix (neat resin) constituent presets, keyed by resin designation.

An isotropic card already in :class:`~wrinklefe.core.material.MaterialLibrary`
— such as ``EPOXY_S6C10`` — can be used directly instead, via
:meth:`MatrixProperties.from_material`.
"""
