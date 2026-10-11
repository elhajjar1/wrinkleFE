"""Load-path eccentricity of a wrinkled laminate (issue #439).

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
it the analytical model missed on the unsafe side by up to +115 %, while
first-ply FE on the specimen's own geometry did not. The ratio is
therefore reported with every analysis, and compression analyses above
the threshold are flagged as outside the kink-band model's validated
regime.
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
#: the evidence is mixed (3 cases), and above 0.15 it was unsafe on 7 of 8.
ECCENTRICITY_REGIME_THRESHOLD = 0.10

#: The one-paragraph warning shown wherever a bending-dominated result is
#: reported (text summary, app, NCR summary), so they cannot disagree.
BENDING_REGIME_CAVEAT = (
    "This wrinkle moves the laminate's load path (load-path eccentricity "
    f"e/t at or above {ECCENTRICITY_REGIME_THRESHOLD:.2f}), so the section "
    "bends as well as kinks. The kink-band knockdown does not represent "
    "that: on measured compression cases in this regime it over-predicted "
    "retained strength by up to about 115%. Prefer FE on the measured "
    "wrinkle profile (issue #439)."
)

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
    """

    e_mm: float
    ratio: float
    thickness_mm: float
    x_mm: float
    section_thickness_mm: float


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
    its values at the two ends of the domain.

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
    line = centroid[:, :1] + (centroid[:, -1:] - centroid[:, :1]) * frac
    offset = np.abs(centroid - line)
    j, i = np.unravel_index(int(np.argmax(offset)), offset.shape)
    e = float(offset[j, i])
    return LoadPathEccentricity(
        e_mm=e,
        ratio=e / thickness if thickness > 0.0 else 0.0,
        thickness_mm=thickness,
        x_mm=float(x[i]),
        section_thickness_mm=float(z[-1, j, i] - z[0, j, i]),
    )
