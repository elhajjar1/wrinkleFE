"""Resin-pocket material zone for wrinkled laminates (Li et al. 2024/2025).

The machined cosine resin insert that creates the wrinkle in the Li
unidirectional glass/epoxy datasets is co-cured **bulk epoxy**, so the
lens it leaves at the wrinkle crest is a fibre-free, isotropic, soft
inclusion rather than homogenised composite.  The default WrinkleFE mesh
deforms every ply through the wrinkle but assigns each element its host
ply's (stiff, fibre-direction) material — it has no notion of the resin
pocket.  This module supplies the missing zone: given a generated
:class:`~wrinklefe.core.mesh.MeshData` and a :class:`ResinPocketSpec`, it
flags the hex elements whose centroids fall inside the lens so the
assembler / stress recovery can swap in the resin material and suppress
the (meaningless) fibre-misalignment angle there.

Geometry
--------
The pocket is a lens centred longitudinally at the wrinkle centre
``x_c`` and through-thickness at ``z_c = wrinkle_z_position * T``.  Its
longitudinal support is ``|x - x_c| <= half_length``; within that span
the through-thickness half-height tapers from ``h_center`` at the crest
to zero at the longitudinal edges following a raised cosine::

    half_height(x) = h_center * 0.5 * (1 + cos(pi * (x - x_c) / half_length))

An element is tagged as resin when its centroid ``(x_e, z_e)`` satisfies
``|x_e - x_c| <= half_length`` and ``|z_e - z_c| <= half_height(x_e)``.
The lens is independent of *y* (the wrinkle is prismatic across the
width), matching the structured mesh's transverse invariance.

References
----------
- Li, X., Ge, J., Chen, G., Zhang, B. & Liang, J. (2024). Composites
  Science and Technology 256, 110762.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from wrinklefe.core.mesh import MeshData


@dataclass(frozen=True)
class ResinPocketSpec:
    """Geometry of the resin lens at a wrinkle crest.

    Parameters
    ----------
    center_x : float
        Longitudinal centre of the pocket (mm); the wrinkle crest, i.e.
        ``domain_length / 2`` for the single-wrinkle pipeline.
    z_center : float
        Through-thickness centre of the pocket (mm), measured from the
        laminate bottom surface: ``wrinkle_z_position * T``.
    half_length : float
        Longitudinal half-extent of the lens (mm).  Elements beyond
        ``|x - center_x| > half_length`` are never resin.  Must be > 0.
    h_center : float
        Through-thickness half-height of the lens at the crest (mm).
        Must be > 0.  The lens tapers to zero half-height at the
        longitudinal edges.
    """

    center_x: float
    z_center: float
    half_length: float
    h_center: float

    def __post_init__(self) -> None:
        if not (self.half_length > 0 and math.isfinite(self.half_length)):
            raise ValueError(
                f"ResinPocketSpec.half_length must be a positive finite "
                f"float, got {self.half_length}"
            )
        if not (self.h_center > 0 and math.isfinite(self.h_center)):
            raise ValueError(
                f"ResinPocketSpec.h_center must be a positive finite "
                f"float, got {self.h_center}"
            )

    @classmethod
    def from_wrinkle(
        cls,
        *,
        amplitude: float,
        wavelength: float,
        center_x: float,
        z_center: float,
        height_scale: float = 1.0,
        length_scale: float = 1.0,
    ) -> ResinPocketSpec:
        """Derive a pocket spec from the wrinkle geometry.

        The lens crest half-height is ``height_scale * amplitude`` (the
        resin-rich zone scales with the wrinkle's amplitude — a deeper
        wrinkle leaves a thicker pocket) and the longitudinal half-extent
        is ``length_scale * (wavelength / 2)`` (the cosine insert's
        support).

        Parameters
        ----------
        amplitude : float
            Wrinkle half-amplitude *A* (mm).
        wavelength : float
            Wrinkle wavelength *lambda* / insert width *L* (mm).
        center_x, z_center : float
            Pocket centre (mm); see :class:`ResinPocketSpec`.
        height_scale : float, optional
            Multiplier on *A* setting the crest half-height.  Default 1.0.
        length_scale : float, optional
            Multiplier on *lambda/2* setting the longitudinal half-extent.
            Default 1.0.

        Returns
        -------
        ResinPocketSpec
        """
        return cls(
            center_x=center_x,
            z_center=z_center,
            half_length=length_scale * wavelength / 2.0,
            h_center=height_scale * amplitude,
        )


def compute_resin_mask(mesh: MeshData, spec: ResinPocketSpec) -> np.ndarray:
    """Boolean mask of elements inside the resin lens.

    Parameters
    ----------
    mesh : MeshData
        Generated hex8 mesh.
    spec : ResinPocketSpec
        Pocket geometry.

    Returns
    -------
    np.ndarray
        Shape ``(n_elements,)`` boolean array, ``True`` where the element
        centroid lies inside the lens.
    """
    # Element centroids (n_elem, 3) — mean of the 8 node coordinates.
    centroids = mesh.nodes[mesh.elements].mean(axis=1)
    xe = centroids[:, 0]
    ze = centroids[:, 2]

    dx = xe - spec.center_x
    within_x = np.abs(dx) <= spec.half_length

    # Raised-cosine half-height envelope; clamp the argument so elements
    # just outside the support (already excluded by within_x) don't wrap.
    arg = np.clip(dx / spec.half_length, -1.0, 1.0)
    half_height = spec.h_center * 0.5 * (1.0 + np.cos(math.pi * arg))

    within_z = np.abs(ze - spec.z_center) <= half_height
    return within_x & within_z


def compute_resin_blend(mesh: MeshData, spec: ResinPocketSpec) -> np.ndarray:
    """Graded resin-blend weight per element, in [0, 1].

    A binary fibre/resin jump at the lens boundary produces a spurious
    stress concentration that over-weakens the laminate (and double-counts
    the wrinkle defect, since the misaligned-fibre crest elements already
    carry the knockdown).  Grading the transition removes it: the weight
    is ``1`` at the lens centre (full neat resin, fibres displaced) and
    tapers smoothly to ``0`` at the lens boundary (host fibre material),
    using the same raised-cosine envelope as :func:`compute_resin_mask`
    for the longitudinal extent and a cosine taper through the thickness.

    Element materials are blended ``(1 - w) * host + w * resin`` and the
    fibre-misalignment angle is scaled by ``(1 - w)`` so the defect is
    represented once.

    Parameters
    ----------
    mesh : MeshData
        Generated hex8 mesh.
    spec : ResinPocketSpec
        Pocket geometry.

    Returns
    -------
    np.ndarray
        Shape ``(n_elements,)`` float array of blend weights in [0, 1];
        ``0`` outside the lens.
    """
    centroids = mesh.nodes[mesh.elements].mean(axis=1)
    xe = centroids[:, 0]
    ze = centroids[:, 2]

    dx = xe - spec.center_x
    within_x = np.abs(dx) <= spec.half_length

    arg = np.clip(dx / spec.half_length, -1.0, 1.0)
    half_height = spec.h_center * 0.5 * (1.0 + np.cos(math.pi * arg))

    # Through-thickness cosine taper: 1 at the lens centre-line, 0 at the
    # local lens edge.  Guard the divide where half_height -> 0.
    dz = np.abs(ze - spec.z_center)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac_z = np.where(half_height > 0, dz / half_height, 1.0)
    frac_z = np.clip(frac_z, 0.0, 1.0)
    w_z = 0.5 * (1.0 + np.cos(math.pi * frac_z))   # 1 at centre, 0 at edge

    # Longitudinal taper reuses the raised-cosine envelope, normalised.
    w_x = 0.5 * (1.0 + np.cos(math.pi * arg))       # 1 at crest, 0 at ends

    weight = w_x * w_z
    weight[~within_x] = 0.0
    weight[dz > half_height] = 0.0
    return weight
