"""Compaction-driven ply-thickness / local fibre-volume-fraction gradient.

A wrinkle cured against rigid tooling does not keep a constant ply
thickness.  The resin is the mobile phase: it is squeezed **out** of the
regions the tooling compacts and pools **into** the regions the geometry
opens up.  The defect is therefore as much a through-thickness ``Vf`` /
ply-thickness gradient as it is a fibre-angle field (issue #379).

The kinematic rule
------------------
Fibre content is conserved per element while the thickness change
absorbs or expels resin::

    Vf_local = Vf_nominal * h0 / h

where ``h`` is the tilt-invariant deformed element height and ``h0`` the
nominal one, both from
:func:`~wrinklefe.core.resin_pocket.element_heights` — the *same* metric
the surface resin pockets use, so the two features cannot drift apart.

* **Stretched** element (``h > h0``; the trough under a flat tool):
  ``Vf`` drops — the resin-rich end.  This is the continuous
  generalization of the binary surface-pocket tag.
* **Compacted** element (``h < h0``; the crest side): ``Vf`` rises — the
  resin-starved, over-consolidated end that the binary tag never modelled.

``Vf_local`` is clamped to ``[vf_min, vf_max]``.  The upper clamp matters:
at the ``tool_flat`` element-inversion bound a crest-side transition
element compresses to ``h / h0 ~ 0.2``, and the uncapped rule would return
three times the nominal ``Vf`` — beyond any achievable packing.  The
clamp stands in for the physics this model does **not** carry: lateral
resin flow along the ply, which relieves the compaction long before
hexagonal packing is reached.  Saturation is counted and logged once.

Ratio anchoring (why the micromechanics is used as a *ratio*)
-------------------------------------------------------------
:mod:`wrinklefe.core.micromechanics` predicts ply constants from
constituents, but its absolute accuracy against the shipped presets is
only 12-33 % (worse for aramid ``G12``).  Feeding those absolute numbers
into an analysis would replace a measured material card with a worse one.
So the local material is the **preset scaled by the model's own ``Vf``
sensitivity**::

    P_local = P_preset * P_micro(Vf_local) / P_micro(Vf_nominal)

for the stiffnesses (``E1``, ``E2``, ``E3``, ``G12``, ``G13``, ``G23``)
and the CTEs (``alpha1``, ``alpha2``, ``alpha3``).  The model's absolute
bias cancels in the ratio and only the trend survives.  At
``Vf_local == Vf_nominal`` every ratio is exactly ``1.0``, so an element
at nominal thickness keeps the preset card *bit for bit* — the zero-drift
anchor this feature is built on.

What is **not** scaled
----------------------
**Poisson's ratios and every strength allowable stay at the preset
value.**  The mixing rules do not predict strengths at all (see the
:mod:`~wrinklefe.core.micromechanics` module docs): longitudinal strength
is set by fibre-strength statistics and misalignment, transverse and shear
strengths by the matrix and the fibre-matrix interface.  Inventing a
``Vf``-strength law here would be quietly wrong, so it is left out — a
documented limitation of v1.  Local failure indices still move, because
the local stiffness change redistributes the stress field.

Quantization
------------
Materials are shared, not per element: ``Vf`` is quantized onto a grid of
``n_bins`` values *anchored on* ``Vf_nominal`` (so the nominal value is
always exactly on the grid) and one material is built per occupied bin.
A 10 000-element mesh therefore carries a few dozen material objects
rather than 10 000, which keeps both memory and the ``id()``-keyed
stiffness caches in the assembler small.

Examples
--------
>>> from wrinklefe.core.compaction import VfGradientSpec, scale_material_to_vf
>>> from wrinklefe.core.material import MaterialLibrary
>>> base = MaterialLibrary().get("IM7_8552")
>>> spec = VfGradientSpec.for_material("IM7_8552")
>>> spec.vf_nominal
0.6
>>> same = scale_material_to_vf(base, 0.60, spec)
>>> same is base                      # exact identity at the nominal Vf
True
>>> compacted = scale_material_to_vf(base, 0.66, spec)
>>> round(compacted.E1 / base.E1, 4)
1.0972

References
----------
- Issue #379 (this feature) and #371 (the tool-flat surface pockets whose
  height metric is reused here).
- Hubert, P. & Poursartip, A. (1998). "A review of flow and compaction
  modelling relevant to thermoset matrix laminate processing",
  J. Reinf. Plast. Compos. 17(4):286-318 — the consolidation/squeeze-flow
  process this kinematic rule abstracts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from wrinklefe.core.material import MaterialLibrary, OrthotropicMaterial
from wrinklefe.core.micromechanics import (
    FIBER_PRESETS,
    MATRIX_PRESETS,
    FiberProperties,
    MatrixProperties,
    alpha1_schapery,
    alpha2_schapery,
    e1_rule_of_mixtures,
    e2_halpin_tsai,
    g12_halpin_tsai,
    g23_transverse_isotropy,
    nu23_rule_of_mixtures,
)
from wrinklefe.core.resin_pocket import element_height_ratio

if TYPE_CHECKING:
    from wrinklefe.core.mesh import MeshData

logger = logging.getLogger(__name__)

__all__ = [
    "CONSTITUENT_DEFAULTS",
    "VfGradientSpec",
    "compute_vf_field",
    "build_vf_materials",
    "scale_material_to_vf",
    "resolve_constituents",
]


#: Material-preset name -> ``(fibre preset, matrix preset, documented Vf)``.
#:
#: Only the systems whose provenance comments in
#: :class:`~wrinklefe.core.material.MaterialLibrary` state a fibre volume
#: fraction are listed; the ``Vf`` is that documented value.  The matrix
#: entry names a key of
#: :data:`~wrinklefe.core.micromechanics.MATRIX_PRESETS` or an isotropic
#: card in the material library (resolved via
#: :meth:`~wrinklefe.core.micromechanics.MatrixProperties.from_material`).
#:
#: Substitutions are explicit rather than invented: ``T800S_M21`` has no
#: M21 constituent card, so the toughened-epoxy ``EPOXY_8552`` stands in,
#: and the generic ``S2_GLASS_EPOXY`` / ``KEVLAR49_EPOXY`` cards use the
#: handbook ``EPOXY_3501_6`` resin.  Because the properties are used as a
#: *ratio* about ``Vf_nominal``, a stand-in matrix shifts only the ``Vf``
#: sensitivity, not the absolute card.  Supply ``vf_fiber`` / ``vf_matrix``
#: explicitly to override any of this.
CONSTITUENT_DEFAULTS: dict[str, tuple[str, str, float]] = {
    "IM7_8552": ("IM7", "EPOXY_8552", 0.60),
    "AC318_S6C10": ("S2_GLASS", "EPOXY_S6C10", 0.60),
    "AC318_S6C10_vacbag": ("S2_GLASS", "EPOXY_S6C10", 0.60),
    "T800S_M21": ("T800S", "EPOXY_8552", 0.59),
    "IM10_8552": ("IM10", "EPOXY_8552", 0.60),
    "IM6G_3501_6": ("IM6G", "EPOXY_3501_6", 0.66),
    "S2_GLASS_EPOXY": ("S2_GLASS", "EPOXY_3501_6", 0.60),
    "KEVLAR49_EPOXY": ("KEVLAR49", "EPOXY_3501_6", 0.60),
}

#: Below this magnitude a reference (nominal-``Vf``) micromechanics value
#: is treated as "no usable ratio" and the preset property is carried over
#: unscaled.  Guards the CTE ratios, where the nominal-``Vf`` prediction
#: for a carbon/epoxy ``alpha1`` can pass through zero and turn a ratio
#: into a meaningless amplification.
_RATIO_GUARD = 1e-12


def resolve_constituents(
    fiber: str | FiberProperties,
    matrix: str | MatrixProperties,
) -> tuple[FiberProperties, MatrixProperties]:
    """Resolve constituent names to property objects.

    Parameters
    ----------
    fiber : str or FiberProperties
        A key of :data:`~wrinklefe.core.micromechanics.FIBER_PRESETS`, or
        a ready-made :class:`~wrinklefe.core.micromechanics.FiberProperties`.
    matrix : str or MatrixProperties
        A key of :data:`~wrinklefe.core.micromechanics.MATRIX_PRESETS`, the
        name of an isotropic card in
        :class:`~wrinklefe.core.material.MaterialLibrary` (converted with
        :meth:`~wrinklefe.core.micromechanics.MatrixProperties.from_material`),
        or a ready-made
        :class:`~wrinklefe.core.micromechanics.MatrixProperties`.

    Returns
    -------
    tuple of (FiberProperties, MatrixProperties)

    Raises
    ------
    ValueError
        If a name resolves to neither a constituent preset nor a library
        card; the message lists the valid names.
    """
    if isinstance(fiber, str):
        if fiber not in FIBER_PRESETS:
            raise ValueError(
                f"Unknown fibre preset {fiber!r}; available fibres are "
                f"{sorted(FIBER_PRESETS)}. Pass a FiberProperties instance "
                f"for a fibre that is not shipped."
            )
        fiber_props = FIBER_PRESETS[fiber]
    else:
        fiber_props = fiber

    if isinstance(matrix, str):
        if matrix in MATRIX_PRESETS:
            matrix_props = MATRIX_PRESETS[matrix]
        else:
            library = MaterialLibrary()
            try:
                card = library.get(matrix)
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"Unknown matrix {matrix!r}; available matrix presets "
                    f"are {sorted(MATRIX_PRESETS)} and isotropic material "
                    f"cards {sorted(library.list_names())}. Pass a "
                    f"MatrixProperties instance for a resin that is not "
                    f"shipped."
                ) from exc
            matrix_props = MatrixProperties.from_material(card)
    else:
        matrix_props = matrix

    return fiber_props, matrix_props


@dataclass(frozen=True)
class VfGradientSpec:
    """Constituents and bounds for the compaction ``Vf`` gradient.

    Parameters
    ----------
    fiber : str or FiberProperties
        Fibre constituent; see :func:`resolve_constituents`.
    matrix : str or MatrixProperties
        Matrix constituent; see :func:`resolve_constituents`.
    vf_nominal : float, optional
        The fibre volume fraction the *preset* material card represents —
        the anchor of the ratio.  Default 0.60.  Must satisfy
        ``0 < vf_nominal < vf_max``.
    vf_max : float, optional
        Upper clamp on the local fibre volume fraction (default 0.75, just
        under the 0.785 square-packing limit and short of the 0.907
        hexagonal one).  Stands in for the lateral resin flow this model
        does not carry.
    vf_min : float, optional
        Lower clamp (default 0.03), so a fully-stretched element
        approaches neat resin without reaching the ``Vf = 0`` edge cases of
        the ratio.
    n_bins : int, optional
        Number of quantization bins for the shared materials (default 64).
        The grid is anchored on ``vf_nominal``.

    Raises
    ------
    ValueError
        If the bounds are not ordered / finite, or ``n_bins < 2``.
    """

    fiber: str | FiberProperties
    matrix: str | MatrixProperties
    vf_nominal: float = 0.60
    vf_max: float = 0.75
    vf_min: float = 0.03
    n_bins: int = 64

    def __post_init__(self) -> None:
        for name in ("vf_nominal", "vf_max", "vf_min"):
            val = getattr(self, name)
            if not (isinstance(val, (int, float)) and np.isfinite(val)):
                raise ValueError(
                    f"VfGradientSpec.{name} must be a finite float, "
                    f"got {val!r}"
                )
        if not (0.0 < self.vf_min < self.vf_nominal < self.vf_max <= 0.9):
            raise ValueError(
                "VfGradientSpec requires 0 < vf_min < vf_nominal < vf_max "
                f"<= 0.9, got vf_min={self.vf_min}, "
                f"vf_nominal={self.vf_nominal}, vf_max={self.vf_max}"
            )
        if not isinstance(self.n_bins, int) or isinstance(self.n_bins, bool):
            raise ValueError(
                f"VfGradientSpec.n_bins must be an int, got {self.n_bins!r}"
            )
        if self.n_bins < 2:
            raise ValueError(
                f"VfGradientSpec.n_bins must be >= 2, got {self.n_bins}"
            )
        # Fail at construction rather than deep in the material build.
        resolve_constituents(self.fiber, self.matrix)

    @classmethod
    def for_material(
        cls,
        material_name: str,
        *,
        fiber: str | FiberProperties | None = None,
        matrix: str | MatrixProperties | None = None,
        vf_nominal: float | None = None,
        vf_max: float = 0.75,
        vf_min: float = 0.03,
        n_bins: int = 64,
    ) -> VfGradientSpec:
        """Build a spec for a library material, filling gaps from defaults.

        Explicit arguments always win; anything left ``None`` is taken from
        :data:`CONSTITUENT_DEFAULTS`.

        Parameters
        ----------
        material_name : str
            Name of the ply material card the analysis uses.
        fiber, matrix : str or constituent or None, optional
            Explicit constituents; ``None`` resolves from the defaults map.
        vf_nominal : float or None, optional
            Explicit nominal ``Vf``; ``None`` resolves from the defaults map.
        vf_max, vf_min, n_bins
            Passed through to :class:`VfGradientSpec`.

        Returns
        -------
        VfGradientSpec

        Raises
        ------
        ValueError
            If a needed value is neither supplied nor known for
            *material_name*; the message names the missing knobs.
        """
        default = CONSTITUENT_DEFAULTS.get(material_name)
        if default is None and (
            fiber is None or matrix is None or vf_nominal is None
        ):
            raise ValueError(
                f"No documented fibre/matrix/Vf for material "
                f"{material_name!r}; the fibre-volume-fraction gradient "
                f"cannot infer its anchor. Supply vf_fiber, vf_matrix and "
                f"vf_nominal explicitly, or use one of "
                f"{sorted(CONSTITUENT_DEFAULTS)}."
            )
        d_fiber, d_matrix, d_vf = (
            default if default is not None else ("", "", 0.0)
        )
        return cls(
            fiber=fiber if fiber is not None else d_fiber,
            matrix=matrix if matrix is not None else d_matrix,
            vf_nominal=vf_nominal if vf_nominal is not None else d_vf,
            vf_max=vf_max,
            vf_min=vf_min,
            n_bins=n_bins,
        )

    @property
    def constituents(self) -> tuple[FiberProperties, MatrixProperties]:
        """The resolved ``(fibre, matrix)`` property objects."""
        return resolve_constituents(self.fiber, self.matrix)


# =======================================================================
# The Vf field
# =======================================================================


def compute_vf_field(mesh: MeshData, spec: VfGradientSpec) -> np.ndarray:
    """Per-element local fibre volume fraction from the compaction.

    Applies ``Vf_local = Vf_nominal * h0 / h`` (fibre content conserved,
    thickness change absorbing or expelling resin) and clamps the result to
    ``[vf_min, vf_max]``.  Elements whose deformed height is non-positive
    (an inverted element, which the ``tool_flat`` amplitude bound already
    rejects) are treated as fully compacted, i.e. ``vf_max``.

    When any element saturates, **one** warning is logged naming the count
    and the un-modelled physics (lateral resin flow along the ply).

    Parameters
    ----------
    mesh : MeshData
        Generated (deformed) hex8 mesh.
    spec : VfGradientSpec
        Constituents, nominal ``Vf`` and the clamps.

    Returns
    -------
    np.ndarray
        Shape ``(n_elements,)`` float array of local fibre volume
        fractions.  Exactly ``spec.vf_nominal`` where the element keeps its
        nominal height.
    """
    ratio = element_height_ratio(mesh)
    with np.errstate(divide="ignore", invalid="ignore"):
        vf = np.where(ratio > 0.0, spec.vf_nominal / ratio, spec.vf_max)
    vf = np.asarray(vf, dtype=np.float64)

    n_high = int(np.count_nonzero(vf > spec.vf_max))
    n_low = int(np.count_nonzero(vf < spec.vf_min))
    vf = np.clip(vf, spec.vf_min, spec.vf_max)

    if n_high or n_low:
        logger.warning(
            "Vf gradient saturated on %d/%d elements (%d at the vf_max=%.3f "
            "compaction cap, %d at the vf_min=%.3f resin-rich floor). The "
            "kinematic rule Vf = Vf_nominal * h0/h has no lateral resin "
            "flow along the ply, so an extreme thickness change would "
            "otherwise return an unachievable packing fraction; the clamp "
            "stands in for that missing physics.",
            n_high + n_low, vf.size, n_high, spec.vf_max, n_low, spec.vf_min,
        )
    return vf


# =======================================================================
# Ratio-anchored local materials
# =======================================================================


def _micro_properties(
    vf: float, fiber: FiberProperties, matrix: MatrixProperties
) -> dict[str, float]:
    """Micromechanics predictions used as ratio numerators/denominators."""
    E2 = e2_halpin_tsai(vf, fiber, matrix)
    nu23 = nu23_rule_of_mixtures(vf, fiber, matrix)
    return {
        "E1": e1_rule_of_mixtures(vf, fiber, matrix),
        "E2": E2,
        "G12": g12_halpin_tsai(vf, fiber, matrix),
        "G23": g23_transverse_isotropy(E2, nu23),
        "alpha1": alpha1_schapery(vf, fiber, matrix),
        "alpha2": alpha2_schapery(vf, fiber, matrix),
    }


def _ratio(local: float, reference: float) -> float:
    """``local / reference``, or ``1.0`` when the reference is unusable."""
    if abs(reference) <= _RATIO_GUARD or not np.isfinite(local):
        return 1.0
    value = local / reference
    if not np.isfinite(value):
        return 1.0
    return float(value)


def scale_material_to_vf(
    base: OrthotropicMaterial, vf: float, spec: VfGradientSpec
) -> OrthotropicMaterial:
    """Ratio-anchor a preset ply card to a local fibre volume fraction.

    Returns ``P_preset * P_micro(vf) / P_micro(vf_nominal)`` for the
    stiffnesses and CTEs, with Poisson's ratios and **all** strengths left
    at their preset values (the mixing rules do not predict them — see the
    module docstring).  At ``vf == spec.vf_nominal`` the *same object* is
    returned, so a nominal-thickness element is bit-identical to the
    unscaled analysis.

    If the scaled card fails :meth:`OrthotropicMaterial.validate` (its
    compliance matrix must stay positive-definite — the Poisson ratios are
    held fixed while the moduli move, so an extreme ratio can in principle
    break the stability bounds), the local ``Vf`` is walked back halfway
    toward the nominal value until it validates, and a warning names the
    clamp.

    Parameters
    ----------
    base : OrthotropicMaterial
        The measured preset card, valid at ``spec.vf_nominal``.
    vf : float
        The local fibre volume fraction.
    spec : VfGradientSpec
        Constituents and the nominal anchor.

    Returns
    -------
    OrthotropicMaterial
        The scaled card (or *base* itself at the nominal ``Vf``).
    """
    if vf == spec.vf_nominal:
        return base

    fiber, matrix = spec.constituents
    ref = _micro_properties(spec.vf_nominal, fiber, matrix)

    vf_try = float(vf)
    for _attempt in range(24):
        loc = _micro_properties(vf_try, fiber, matrix)
        r_E1 = _ratio(loc["E1"], ref["E1"])
        r_E2 = _ratio(loc["E2"], ref["E2"])
        r_G12 = _ratio(loc["G12"], ref["G12"])
        r_G23 = _ratio(loc["G23"], ref["G23"])
        r_a1 = _ratio(loc["alpha1"], ref["alpha1"])
        r_a2 = _ratio(loc["alpha2"], ref["alpha2"])
        try:
            return replace(
                base,
                E1=base.E1 * r_E1,
                E2=base.E2 * r_E2,
                E3=base.E3 * r_E2,
                G12=base.G12 * r_G12,
                G13=base.G13 * r_G12,
                G23=base.G23 * r_G23,
                alpha1=base.alpha1 * r_a1,
                alpha2=base.alpha2 * r_a2,
                alpha3=base.alpha3 * r_a2,
                name=f"{base.name}_Vf{vf_try:.4f}",
            )
        except ValueError:
            vf_try = 0.5 * (vf_try + spec.vf_nominal)
            logger.warning(
                "Ratio-anchored ply at Vf=%.4f is not positive-definite "
                "(Poisson ratios are held at the preset values while the "
                "moduli scale); retrying at Vf=%.4f, halfway back to the "
                "nominal %.4f.",
                vf, vf_try, spec.vf_nominal,
            )
    logger.warning(
        "Could not build a positive-definite ply for Vf=%.4f; keeping the "
        "preset card %r unscaled for those elements.", vf, base.name,
    )
    return base


def _quantize(vf: np.ndarray, spec: VfGradientSpec) -> np.ndarray:
    """Snap ``Vf`` onto an ``n_bins`` grid anchored on ``vf_nominal``.

    The grid *contains* ``vf_nominal`` exactly, so an element at nominal
    thickness quantizes to the nominal value and its material stays the
    untouched preset (the zero-drift anchor).
    """
    step = (spec.vf_max - spec.vf_min) / (spec.n_bins - 1)
    snapped = spec.vf_nominal + np.round(
        (vf - spec.vf_nominal) / step
    ) * step
    return np.clip(snapped, spec.vf_min, spec.vf_max)


def build_vf_materials(
    base_material: OrthotropicMaterial,
    vf_field: np.ndarray,
    spec: VfGradientSpec,
    *,
    element_ids: np.ndarray | None = None,
) -> dict[int, OrthotropicMaterial]:
    """Per-element ratio-anchored materials, sharing one card per ``Vf`` bin.

    ``Vf`` is quantized onto ``spec.n_bins`` values (grid anchored on
    ``vf_nominal``) and one :class:`OrthotropicMaterial` is built per
    occupied bin, then shared by every element in that bin — so a large
    mesh carries a few dozen material objects, not one per element, and the
    ``id()``-keyed element-stiffness caches stay small.

    Elements that quantize to ``vf_nominal`` are **omitted** from the
    mapping: their material is unchanged, and leaving them out keeps the
    downstream resolution (and therefore the numbers) identical to a run
    with the feature off.

    Parameters
    ----------
    base_material : OrthotropicMaterial
        The preset ply card these elements would otherwise use.
    vf_field : np.ndarray
        Per-element local ``Vf`` from :func:`compute_vf_field`.
    spec : VfGradientSpec
        Constituents, nominal anchor and bin count.
    element_ids : np.ndarray or None, optional
        Restrict the mapping to these element indices (used when a laminate
        mixes materials, so each ply material is scaled on its own
        elements).  ``None`` (default) covers every element in *vf_field*.

    Returns
    -------
    dict of int -> OrthotropicMaterial
        Element index to its local material, for the elements whose ``Vf``
        differs from nominal.
    """
    vf = np.asarray(vf_field, dtype=np.float64)
    ids = (
        np.arange(vf.size, dtype=np.int64)
        if element_ids is None
        else np.asarray(element_ids, dtype=np.int64)
    )
    if ids.size == 0:
        return {}

    binned = _quantize(vf[ids], spec)
    materials: dict[int, OrthotropicMaterial] = {}
    cache: dict[float, OrthotropicMaterial] = {}
    for value in np.unique(binned):
        vf_bin = float(value)
        if vf_bin == spec.vf_nominal:
            continue
        cache[vf_bin] = scale_material_to_vf(base_material, vf_bin, spec)
    for elem, value in zip(ids.tolist(), binned.tolist(), strict=True):
        material = cache.get(float(value))
        if material is not None:
            materials[int(elem)] = material
    logger.debug(
        "Vf gradient: %d elements mapped onto %d shared materials "
        "(Vf %.3f-%.3f, nominal %.3f).",
        len(materials), len(cache),
        float(binned.min()), float(binned.max()), spec.vf_nominal,
    )
    return materials
