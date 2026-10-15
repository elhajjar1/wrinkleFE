"""Load-path eccentricity of a wrinkled laminate (issues #439, #452).

A wrinkle that moves the laminate's outer surfaces moves its load path:
the axial-stiffness-weighted centroid of the section at the wrinkle sits
off the straight line joining the centroids at the two ends, so an axial
load also bends the section there. Whole-thickness waves offset it by
roughly their amplitude, and a one-sided surface trough by roughly half
its depth. An embedded wrinkle with flat outer surfaces barely offsets it
at all: it misaligns fibres, which is what the kink-band model
represents, but it leaves the load path where it was.

The validation study behind issue #439
(``docs/internal/BENDING_MODE_INVESTIGATION.md``) scored 50 measured
compression cases against this offset, as a fraction ``e / t`` of the
laminate thickness. Below :data:`ECCENTRICITY_REGIME_THRESHOLD` the
angle-based analytical knockdown was never more than 15 % unsafe. Above
it the analytical model missed on the unsafe side by up to +115 % (Shi's
one-sided troughs, scored then on an embedded ``graded`` recipe; +52 %,
on Thor's whole-thickness wave, once issue #461 scored them as
``tool_flat``), while first-ply FE on the specimen's own geometry did
not. The ratio is
therefore reported with every analysis, and compression analyses above
the threshold are flagged as bending-dominated.

Issue #452 acts on the flag for whole-thickness waves, where both outer
surfaces follow the load path (:attr:`LoadPathEccentricity.whole_thickness`).
The wrinkled section is treated as an eccentric column: the outer fibres
carry ``sigma * (t_0 / t_c) * (1 + 6 e / t_c)``, so they reach the
pristine strength at :func:`bending_knockdown`, and the analytical
compression knockdown becomes the lower of the kink-band value and that
bound. A one-sided trough (one surface pinned flat by the tool) is not
covered: the bound did not explain its measured knockdown, so it keeps
the warning and the advice to run FE.

The measurement reads the geometry the user modelled, not the specimen
(issue #461). ``graded`` with ``decay_floor=0`` keeps both outer surfaces
flat, which is what an embedded wrinkle is. On a UD laminate the centroid
of such a section is exactly midway between its flat surfaces, so ``e``
is zero at any amplitude. On a multidirectional one only the stiffness
redistribution moves it (``e / t`` at most 0.03 on Shi's troughs). A
one-sided surface trough modelled as ``graded`` therefore gets no
warning. Its inputs are those of an embedded wrinkle: measured embedded
wrinkles reach the same amplitude-to-thickness ratio, peak angle and
amplitude-to-wavelength ratio as Shi's troughs. So nothing detects it,
and :data:`ONE_SIDED_TROUGH_ADVICE` steers users to ``tool_flat``
instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from wrinklefe.core.laminate import Laminate
    from wrinklefe.core.morphology import WrinkleConfiguration

#: ``e / t`` at and above which a compression wrinkle is flagged as
#: bending-dominated: outside the regime where the angle-based kink-band
#: knockdown was validated (issue #439). Below 0.10 the deployed model was
#: never more than 15 % unsafe on 39 measured cases; between 0.10 and 0.15
#: the evidence is mixed (3 cases), and above 0.15 it was unsafe on 7 of 8
#: (6 of 8 once issue #461 scored Shi's troughs on their one-sided geometry).
ECCENTRICITY_REGIME_THRESHOLD = 0.10

#: Fraction of the load-path offset ``e`` that EACH outer surface must
#: itself move, in the same direction and at the same section, for the
#: wave to count as whole-thickness (issue #452). A whole-thickness wave
#: moves both surfaces by about ``e`` (ratio ~1); a one-sided trough pins
#: one surface flat (ratio 0 on that side). The midpoint separates them.
WHOLE_THICKNESS_SURFACE_FRACTION = 0.5

#: Warning for a bending-dominated result the eccentric-column bound does
#: NOT cover (a one-sided trough, or any wave whose outer surfaces do not
#: both follow the load path). Shown wherever such a result is reported
#: (text summary, app, NCR summary), so they cannot disagree.
BENDING_REGIME_CAVEAT = (
    "This wrinkle moves the laminate's load path (load-path eccentricity "
    f"e/t at or above {ECCENTRICITY_REGIME_THRESHOLD:.2f}), so the section "
    "bends as well as kinks, but it is not a whole-thickness wave (one "
    "outer surface does not follow the load path, as over a flat tool "
    "face), so the eccentric-column bending bound is not applied. On the "
    "measured one-sided troughs of this kind the kink-band knockdown "
    "ranged from about 70% conservative to 5% unsafe, so it is usually "
    "excessively conservative here, for two reasons. It applies the "
    "trough's peak misalignment angle to the whole cross-section, although "
    "the misalignment fades to nothing toward the flat face. And it does "
    "not credit the straighter plies near that face with the load the "
    "wavy plies shed. It is not always conservative: on the deepest "
    "multidirectional trough it was 5% unsafe. First-ply FE on the measured "
    "geometry resolves both effects and was conservative by 1% to 40%. "
    "Prefer FE on the measured wrinkle profile (issues #439, #452, #461)."
)

#: Warning for a bending-dominated whole-thickness wave, where the
#: analytical knockdown is the lower of the kink-band value and the
#: eccentric-column bound (issue #452).
BENDING_BOUND_CAVEAT = (
    "This wrinkle moves the laminate's load path through its whole "
    "thickness (load-path eccentricity e/t at or above "
    f"{ECCENTRICITY_REGIME_THRESHOLD:.2f}), so the section bends as well "
    "as kinks. The analytical knockdown is therefore the lower of the "
    "kink-band knockdown and an eccentric-column bending bound, "
    "(t_c/t_0) / (1 + 6 e/t_c). On the measured whole-thickness waves in "
    "this regime the bound brought the worst unsafe miss from +52% to "
    "+15%, so it is a screening bound, not a validated allowable: check "
    "it against FE on the measured wrinkle profile (issue #452)."
)


#: Which morphology to pick for a one-sided surface trough (issue #461).
#: The load-path eccentricity measures the modelled geometry, and
#: ``graded`` (decay floor 0) models flat outer surfaces, so it cannot
#: see a trough. Shown where the morphology is chosen (the app's
#: morphology selector and the CLI's ``--morphology`` help), so they
#: cannot disagree. Plain text with no ``%``, because argparse
#: %-formats help strings.
ONE_SIDED_TROUGH_ADVICE = (
    "Model a one-sided surface trough (one outer surface follows the "
    "wrinkle, the other is flat on the tool) as 'tool_flat', pinning only "
    "the tool face (surface_pocket_side 'top' or 'bottom', not 'both'). "
    "Do not model it as 'graded'. 'graded' with decay floor 0 keeps both "
    "outer surfaces flat, which is an embedded wrinkle. The load-path "
    "eccentricity check then sees no offset, so a trough deep enough to "
    "bend the section gets no bending-dominated warning (issue #461)."
)


def bending_caveat(bending_dominated: bool, bound_applied: bool) -> str | None:
    """The bending warning that goes with a result, or ``None``.

    :data:`BENDING_BOUND_CAVEAT` when the eccentric-column bound was
    applied, :data:`BENDING_REGIME_CAVEAT` when the result is
    bending-dominated but the bound does not cover it, ``None`` otherwise.
    """
    if not bending_dominated:
        return None
    return BENDING_BOUND_CAVEAT if bound_applied else BENDING_REGIME_CAVEAT


#: x samples per shortest wavelength when measuring the offset.
_SAMPLES_PER_WAVELENGTH = 16
#: Bounds on the number of x samples.
_MIN_X_SAMPLES = 200
_MAX_X_SAMPLES = 4000
#: y samples (an odd count includes the domain centre line, where
#: transverse amplitude envelopes peak).
_Y_SAMPLES = 5


@dataclass(frozen=True)
class LoadPathEccentricity:
    """The largest offset of the load path from its far-field line.

    Attributes
    ----------
    e_mm : float
        Largest offset (mm) of the section's axial-stiffness-weighted
        centroid from the straight line joining the centroids at the two
        ends of the domain.
    ratio : float
        ``e_mm`` over the laminate thickness: the regime metric.
    thickness_mm : float
        Far-field laminate thickness (mm).
    x_mm : float
        Longitudinal position of the largest offset (mm).
    section_thickness_mm : float
        Laminate thickness at that position (mm); below ``thickness_mm``
        where the wrinkle thins the section (a one-sided trough).
    top_surface_offset_mm, bottom_surface_offset_mm : float
        How far each outer surface is itself displaced at that position
        (mm), from the straight line joining its own values at the two
        ends of the domain, signed positive in the direction of the
        load-path offset (issue #452). A whole-thickness wave moves both
        by about ``e_mm``; a surface pinned flat by the tool gives ~0;
        a negative value means the surface moved the other way.
    """

    e_mm: float
    ratio: float
    thickness_mm: float
    x_mm: float
    section_thickness_mm: float
    top_surface_offset_mm: float = 0.0
    bottom_surface_offset_mm: float = 0.0

    @property
    def whole_thickness(self) -> bool:
        """Whether both outer surfaces follow the load path (issue #452).

        ``True`` when the load path is offset at all and each outer
        surface moves, in the same direction at the same section, by at
        least :data:`WHOLE_THICKNESS_SURFACE_FRACTION` of that offset:
        the section is displaced bodily, as an eccentric column, rather
        than thinned or thickened on one side (a one-sided trough) or
        only misaligned inside flat surfaces (an embedded wrinkle).
        Measured on the generated ply boundaries, so it holds for every
        morphology, decay setting and multi-wrinkle layout alike.
        """
        if self.e_mm <= 0.0:
            return False
        least = min(self.top_surface_offset_mm, self.bottom_surface_offset_mm)
        return bool(least >= WHOLE_THICKNESS_SURFACE_FRACTION * self.e_mm)


def bending_knockdown(ecc: LoadPathEccentricity) -> float:
    """Eccentric-column bending knockdown of a wrinkled section (#452).

    The far-field load ``P = sigma * t_0`` acts along the load line, a
    distance ``e`` from the centroid of the section at the wrinkle, whose
    thickness is ``t_c``. The outer fibres there carry::

        P / t_c * (1 + 6 e / t_c)

    (axial plus ``M / Z`` with ``M = P e`` and ``Z = t_c**2 / 6`` per unit
    width), so they reach the pristine strength at::

        KD_bend = (t_c / t_0) / (1 + 6 e / t_c)

    Homogeneous-section beam theory, with no ``P``-delta amplification:
    it is meant as a screening bound alongside the kink-band knockdown,
    not as a column-stability check. ``1.0`` for a zero offset.
    """
    t_0 = float(ecc.thickness_mm)
    t_c = float(ecc.section_thickness_mm)
    if t_0 <= 0.0 or t_c <= 0.0:
        return 1.0
    return (t_c / t_0) / (1.0 + 6.0 * float(ecc.e_mm) / t_c)


def _axial_modulus(material, angle_deg: float) -> float:
    """In-plane axial modulus ``E_x`` of a ply at ``angle_deg`` (CLT)."""
    c = math.cos(math.radians(angle_deg))
    s = math.sin(math.radians(angle_deg))
    compliance = (
        c**4 / material.E1
        + (1.0 / material.G12 - 2.0 * material.nu12 / material.E1)
        * s**2 * c**2
        + s**4 / material.E2
    )
    return 1.0 / float(compliance)


def _shortest_wavelength(wrinkle_config: WrinkleConfiguration) -> float:
    lengths = []
    for w in getattr(wrinkle_config, "wrinkles", ()) or ():
        inner = getattr(w.profile, "profile", w.profile)
        lam = float(getattr(inner, "wavelength", 0.0))
        if lam > 0.0:
            lengths.append(lam)
    return min(lengths) if lengths else 0.0


def load_path_eccentricity(
    laminate: Laminate,
    wrinkle_config: WrinkleConfiguration | None,
    domain_length: float,
    domain_width: float,
) -> LoadPathEccentricity:
    """Measure the load-path offset of a wrinkled laminate.

    The wrinkle displaces the ply boundaries exactly as the FE mesh does
    (:meth:`wrinklefe.core.mesh.WrinkleMesh.deformed_nodes`), sampled on a
    fine x grid and a few y lines, without building or solving a mesh.
    For each (x, y) column the plies' mid-heights are weighted by their
    thickness and their CLT axial modulus to give the centroid; the result
    is the largest offset of that centroid from the straight line between
    its values at the two ends of the domain. The two outer surfaces are
    measured the same way, at that section, so the caller can tell a
    whole-thickness wave from a one-sided trough
    (:attr:`LoadPathEccentricity.whole_thickness`).

    Parameters
    ----------
    laminate : Laminate
        The laminate (ply angles, materials, thicknesses).
    wrinkle_config : WrinkleConfiguration or None
        The wrinkle geometry. ``None`` gives a zero offset.
    domain_length, domain_width : float
        The analysis domain (mm), as used by the FE mesh.

    Returns
    -------
    LoadPathEccentricity
    """
    from wrinklefe.core.mesh import WrinkleMesh

    thickness = float(laminate.total_thickness)
    if wrinkle_config is None or not getattr(wrinkle_config, "wrinkles", ()):
        return LoadPathEccentricity(0.0, 0.0, thickness, 0.0, thickness)

    lam_min = _shortest_wavelength(wrinkle_config)
    nx = _MIN_X_SAMPLES
    if lam_min > 0.0:
        nx = int(math.ceil(_SAMPLES_PER_WAVELENGTH * domain_length / lam_min))
    nx = min(max(nx, _MIN_X_SAMPLES), _MAX_X_SAMPLES)
    ny = _Y_SAMPLES - 1

    gen = WrinkleMesh(
        laminate=laminate, wrinkle_config=wrinkle_config,
        Lx=domain_length, Ly=domain_width, nx=nx, ny=ny, nz_per_ply=1,
    )
    nodes, _ = gen.deformed_nodes()
    n_plies = laminate.n_plies
    # Node order is k (z, slowest) -> j (y) -> i (x, fastest).
    z = nodes[:, 2].reshape(n_plies + 1, ny + 1, nx + 1)
    x = nodes[: nx + 1, 0]

    mid = 0.5 * (z[1:] + z[:-1])
    t = np.diff(z, axis=0)
    moduli = np.array(
        [_axial_modulus(p.material, p.angle) for p in laminate.plies]
    )[:, None, None]
    weight = moduli * t
    centroid = np.sum(weight * mid, axis=0) / np.sum(weight, axis=0)

    # Straight load line through the end centroids, per y line.
    frac = (x - x[0]) / (x[-1] - x[0])

    def _from_end_line(surface: np.ndarray) -> np.ndarray:
        line = surface[:, :1] + (surface[:, -1:] - surface[:, :1]) * frac
        return np.asarray(surface - line)

    signed = _from_end_line(centroid)
    offset = np.abs(signed)
    j, i = np.unravel_index(int(np.argmax(offset)), offset.shape)
    e = float(offset[j, i])
    # Each outer surface at the same section, positive in the direction
    # the load path moved (issue #452).
    direction = float(np.sign(signed[j, i]))
    top = direction * float(_from_end_line(z[-1])[j, i])
    bottom = direction * float(_from_end_line(z[0])[j, i])
    return LoadPathEccentricity(
        e_mm=e,
        ratio=e / thickness if thickness > 0.0 else 0.0,
        thickness_mm=thickness,
        x_mm=float(x[i]),
        section_thickness_mm=float(z[-1, j, i] - z[0, j, i]),
        top_surface_offset_mm=top,
        bottom_surface_offset_mm=bottom,
    )
