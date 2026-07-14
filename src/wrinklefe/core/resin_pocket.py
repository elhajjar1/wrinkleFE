"""Resin-pocket material zones for wrinkled laminates.

Two physically distinct resin zones share this module because they share
the same *material* treatment (a fibre-free, isotropic, soft inclusion
that suppresses the meaningless fibre-misalignment angle) and the same
downstream mesh plumbing (``resin_mask`` / ``resin_blend`` +
``resin_blend_materials``, consumed by the assembler, stress recovery and
failure evaluator with **zero** solver changes):

1. **Crest lens** (:class:`ResinPocketSpec`, :func:`compute_resin_mask`,
   :func:`compute_resin_blend`; Li et al. 2024/2025) — a machined cosine
   epoxy insert at the wrinkle crest, mid-thickness.
2. **Surface resin pockets** (:class:`SurfacePocketSpec`,
   :func:`compute_surface_resin_blend`) — the neat-resin pockets that fill
   the wrinkle *troughs* just under a **tool-flat outer surface**.

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

Tool-flat surfaces and surface resin pockets
---------------------------------------------
Parts cured against rigid tooling (or under a caul sheet) have
**perfectly flat outer surfaces**: the fibre undulation is confined to
the interior and, where the outermost undulating ply dips *away* from the
flat tool surface, the volume deficit fills with **neat resin** —
surface-visible pockets over the wrinkle troughs, thinning to nothing
over the crests.  WrinkleFE's default through-thickness decay already
gets the *kinematics* right for free (the decay reaches exactly zero at
both outer surfaces for the ``stack``/``convex``/``concave`` morphologies
and for ``graded`` with ``decay_floor == 0`` — see
:func:`~wrinklefe.core.morphology.WrinkleConfiguration._through_thickness_decay`),
but the *material* is wrong: the sub-surface elements silently **stretch**
over the troughs while keeping their stiff fibre-direction material.
Physically that stretched volume is fibre-free isotropic resin — a
stiffness hole and a matrix-cracking initiation site where
fibre-misalignment failure criteria are meaningless.

:func:`compute_surface_resin_blend` recovers that zone from the deformed
mesh geometry.  For each column of elements it walks inward from the
chosen flat surface to the first element whose deformed height ``h``
departs from the nominal height ``h0`` (the transition element straddling
the outermost pinned/undulating interface), and assigns a resin blend
weight equal to the **excess-stretch fraction**::

    weight = max(0, (h - h0) / h)

This is exactly volume-conserving: the resin volume tagged in a column
equals ``weight * h * area == (h - h0) * area``, i.e. the integrated
kinematic gap ``-w(x) * decay_last`` between the flat surface and the
outermost undulating ply.  It is automatically zero over crests and
wherever the ply moves *toward* the tool (those elements compress; v1
leaves them as host material — the local fibre-volume increase is treated
as neutral).  This zone is the *complement* of the crest lens:
surface-bonded and trough-following rather than a mid-thickness crest
insert.  Both may be enabled together; their weights compose by
per-element maximum.

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
    mask: np.ndarray = within_x & within_z
    return mask


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

    weight: np.ndarray = w_x * w_z
    weight[~within_x] = 0.0
    weight[dz > half_height] = 0.0
    return weight


# =======================================================================
# Surface resin pockets (tool-flat outer surface, trough-following)
# =======================================================================


@dataclass(frozen=True)
class SurfacePocketSpec:
    """Geometry rule for surface resin pockets under a tool-flat surface.

    Unlike :class:`ResinPocketSpec` (a fixed cosine lens), the surface
    pocket is derived from the *deformed* mesh: it tags the transition
    element in each column where the outermost undulating ply pulls away
    from the flat surface, weighting it by the excess-stretch fraction.
    There is therefore no explicit position/size here — only which
    surface(s) to inspect and the flatness tolerance.

    Parameters
    ----------
    side : {"top", "bottom", "both"}
        Which tool-flat surface(s) to tag.  ``"top"`` is the ``+z`` face,
        ``"bottom"`` the ``-z`` face.  ``"both"`` tags each and composes
        the two weight fields by per-element maximum.
    min_gap_threshold : float or None, optional
        Absolute gap (mm) below which an element is treated as flat, so
        numerically-flat regions do not produce resin "dust".  ``None``
        (default) uses ``0.01 * ply_thickness`` inferred from the mesh.
    compaction_stiffening : bool, optional
        Reserved stretch-goal knob.  When ``False`` (default) the
        compression side (ply moving toward the tool) stays host material;
        no compaction stiffening is modelled in v1.  Present so the API is
        stable when the feature lands.
    """

    side: str = "top"
    min_gap_threshold: float | None = None
    compaction_stiffening: bool = False

    def __post_init__(self) -> None:
        if self.side not in ("top", "bottom", "both"):
            raise ValueError(
                f"SurfacePocketSpec.side must be 'top', 'bottom' or 'both', "
                f"got {self.side!r}"
            )
        if self.min_gap_threshold is not None and not (
            math.isfinite(self.min_gap_threshold) and self.min_gap_threshold >= 0.0
        ):
            raise ValueError(
                f"SurfacePocketSpec.min_gap_threshold must be a non-negative "
                f"finite float or None, got {self.min_gap_threshold!r}"
            )


def _tag_surface_side(
    gap: np.ndarray,
    height: np.ndarray,
    nonflat: np.ndarray,
    side: str,
    nz: int,
    ncol: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Tag the single transition element-layer nearest one flat surface.

    The surface pocket lives in exactly one horizontal element-layer: the
    layer straddling the outermost pinned/undulating interface (the pinned
    surface plies above it are flat, the interior below undulates).  That
    layer is the non-flat layer with the extreme k-index toward the chosen
    surface — fixed across all columns, so a column where the wrinkle
    happens to cross zero (locally flat) never leaks the tag onto a deeper
    internal ply interface.

    Returns the transition-layer element indices and their blend weights
    (0 where a column has no gap or the ply is in compression there).  When
    no layer is non-flat (flat mesh) the returned arrays are empty.
    """
    layer_nonflat = nonflat.any(axis=1)
    layers = np.flatnonzero(layer_nonflat)
    if layers.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, np.empty(0, dtype=np.float64)

    # Transition layer: closest non-flat layer to the chosen surface.
    k = int(layers.max()) if side == "top" else int(layers.min())

    cols = np.arange(ncol)
    g = gap[k, cols]
    h = height[k, cols]
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where((nonflat[k, cols]) & (g > 0.0), g / h, 0.0)
    elem_idx = k * ncol + cols
    return elem_idx, np.asarray(w, dtype=np.float64)


def compute_surface_resin_blend(
    mesh: MeshData,
    wrinkle_config: object,
    spec: SurfacePocketSpec,
) -> np.ndarray:
    """Per-element surface-pocket resin blend weight, in [0, 1].

    Computes the neat-resin blend weight for the surface pockets that fill
    the wrinkle troughs beneath a tool-flat outer surface.  The weight is
    the excess-stretch fraction ``max(0, (h - h0) / h)`` of the transition
    element in each column (the outermost element that departs from the
    nominal height ``h0`` as it spans the gap to the flat surface).  It is
    zero over crests and wherever the outermost undulating ply moves toward
    the tool (compression side stays host material in v1).

    The result is volume-conserving: ``weight * h == h - h0`` is exactly
    the kinematic gap ``-w(x) * decay_last`` between the flat surface and
    the outermost undulating ply, so the tagged resin volume equals the
    integrated gap.  The geometry is read from the *already-deformed* mesh,
    so multi-wrinkle configurations work for free (the gap comes from the
    composed displacement field).

    Parameters
    ----------
    mesh : MeshData
        Generated (deformed) hex8 mesh with a tool-flat outer surface.
    wrinkle_config : WrinkleConfiguration
        The wrinkle configuration that deformed the mesh.  Accepted for API
        symmetry with the crest-lens helpers and cross-checking; the blend
        is derived from the deformed geometry so a ``None`` is rejected to
        catch a mis-wired call (a surface pocket is meaningless without a
        wrinkle).
    spec : SurfacePocketSpec
        Which surface(s) to tag and the flatness tolerance.

    Returns
    -------
    np.ndarray
        Shape ``(n_elements,)`` float array of blend weights in [0, 1];
        ``0`` outside the tagged transition elements.
    """
    if wrinkle_config is None:
        raise ValueError(
            "compute_surface_resin_blend requires a wrinkle configuration; "
            "surface resin pockets are meaningless on an undeformed mesh."
        )

    nz = int(mesh.nz)
    ncol = int(mesh.nx) * int(mesh.ny)
    n_elem = int(mesh.n_elements)

    # Per-element deformed height from the 8 node z-coordinates.
    ez = mesh.nodes[mesh.elements][:, :, 2]
    height = ez.max(axis=1) - ez.min(axis=1)

    z_lo = float(mesh.nodes[:, 2].min())
    z_hi = float(mesh.nodes[:, 2].max())
    total_thickness = z_hi - z_lo
    # Structured mesh: every element has the same nominal height.
    h0 = total_thickness / nz if nz > 0 else 0.0

    n_plies = int(mesh.ply_ids.max()) + 1 if mesh.ply_ids.size else 1
    ply_thickness = total_thickness / n_plies if n_plies > 0 else total_thickness
    if spec.min_gap_threshold is not None:
        tol = float(spec.min_gap_threshold)
    else:
        tol = 0.01 * ply_thickness

    gap = height - h0
    weight = np.zeros(n_elem, dtype=np.float64)

    if n_elem != nz * ncol or nz == 0 or ncol == 0:
        # Non-structured or degenerate mesh: no surface pocket to tag.
        return weight

    gap_grid = gap.reshape(nz, ncol)
    height_grid = height.reshape(nz, ncol)
    nonflat = np.abs(gap_grid) > tol

    sides = ("top", "bottom") if spec.side == "both" else (spec.side,)
    for side in sides:
        elem_idx, w = _tag_surface_side(gap_grid, height_grid, nonflat, side, nz, ncol)
        np.maximum.at(weight, elem_idx, w)

    return weight
