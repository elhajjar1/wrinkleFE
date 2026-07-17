"""Diagnostic spike for the ``tool_flat`` morphology (issue #371, Part A).

This module *protects the diagnosis* behind the ``tool_flat`` morphology
and pins the element-inversion bound the morphology's validation relies
on.  It deliberately depends only on primitives that predate the feature
(``WrinkleConfiguration`` decay helpers, a hand-rolled tool-flat decay,
the hex8 Jacobian check), so it stands on its own as the pre-implementation
spike.

Two findings are pinned:

1. **The bug (linear-decay morphologies give insignificant pockets).**
   At app defaults (24 plies, ``t = 0.183`` mm, ``A = 0.5`` mm) every
   tool-flat-*compatible* morphology (``stack`` and ``graded`` with
   ``decay_floor = 0``) decays the wave linearly across the whole
   thickness, so the outermost undulating ply displaces only
   ~0.043-0.046 mm — a trough gap of ~0.24 of a *single* ply thickness,
   invisible in the preview and mechanically negligible.  This is the
   user's report, encoded as a regression guard.

2. **The fix and its geometric limit.**  A uniform-amplitude core with a
   short linear transition to an exactly-flat pinned surface makes the
   trough gap ≈ the *full* amplitude ``A`` (≈2.7 ply thicknesses at
   defaults).  But on the crest side the ``S`` transition elements each
   *compress* by ``A / S`` over an element height ``t / nz_per_ply``, so
   they invert (negative Jacobian) once

       A / S >= t / nz_per_ply    <=>    A >= S * t / nz_per_ply.

   The spike verifies this bound numerically: positive Jacobians at the
   bound, inversion just beyond.  ``AnalysisConfig._validate`` rejects
   ``tool_flat`` amplitudes above ``0.8 * S * t / nz_per_ply``.
"""

from __future__ import annotations

import numpy as np

from wrinklefe.core.laminate import Laminate
from wrinklefe.core.material import MaterialLibrary
from wrinklefe.core.mesh import MeshData, WrinkleMesh
from wrinklefe.core.morphology import WrinkleConfiguration
from wrinklefe.core.wrinkle import GaussianSinusoidal
from wrinklefe.elements.hex8 import _detJ_at_centroid_batch

# App defaults reproduced from the issue.
N_PLIES = 24
PLY_T = 0.183
A_DEFAULT = 0.5
LAM = 16.0
WIDTH = 12.0


# ---------------------------------------------------------------------------
# Part 1 — the bug: linear-decay morphologies give an insignificant gap
# ---------------------------------------------------------------------------


def _outer_ply_displacement(morphology: str, **kw) -> float:
    """Max trough displacement of the outermost *undulating* ply."""
    Lx = 3.0 * LAM
    prof = GaussianSinusoidal(
        amplitude=A_DEFAULT, wavelength=LAM, width=WIDTH, center=Lx / 2.0
    )
    wc = WrinkleConfiguration.from_morphology_name(
        morphology, prof, interface1=11, interface2=12, **kw
    )
    # Node ply-ids for a single through-thickness column (k = 0 .. n_plies).
    ply_ids = np.minimum(np.arange(N_PLIES + 1), N_PLIES - 1)
    k = wc.wrinkles[0].ply_interface if wc.n_wrinkles() == 1 else 11
    decay = wc._through_thickness_decay(ply_ids, k, N_PLIES)
    xs = np.linspace(0.0, Lx, 4000)
    dmax = float(np.max(np.abs(prof.displacement(xs))))
    # The outermost undulating ply is the node-layer one in from the (flat)
    # surface: ply-id N_PLIES - 2.
    return dmax * float(decay[N_PLIES - 2])


def test_linear_decay_gap_is_insignificant():
    """Both tool-flat-compatible morphologies leave a sub-quarter-ply gap."""
    stack = _outer_ply_displacement("stack")
    graded = _outer_ply_displacement("graded", decay_floor=0.0)
    # ~0.045 mm for stack, ~0.043 mm for graded (the issue's numbers).
    assert 0.040 < stack < 0.050
    assert 0.038 < graded < 0.048
    # ... i.e. ~a quarter of ONE ply thickness — the user's "invisible" gap.
    assert stack / PLY_T < 0.30
    assert graded / PLY_T < 0.30


# ---------------------------------------------------------------------------
# Part 2 — the fix: a tool-flat decay makes the gap ~= full amplitude,
# and the element-inversion bound A_crit = S * t / nz_per_ply.
# ---------------------------------------------------------------------------


def _toolflat_decay(
    ply_ids: np.ndarray, n_plies: int, s_trans: int, side: str = "top"
) -> np.ndarray:
    """Hand-rolled tool-flat decay: 1 in the core, linear ramp to 0 at
    the pinned surface(s) over ``s_trans`` plies.  Mirrors the production
    ``_through_thickness_decay(decay_mode="tool_flat")`` this spike predates.
    """
    p = np.asarray(ply_ids, dtype=np.float64)
    top_ply = n_plies - 1
    top = (
        np.clip((top_ply - p) / s_trans, 0.0, 1.0)
        if side in ("top", "both")
        else np.ones_like(p)
    )
    bot = (
        np.clip(p / s_trans, 0.0, 1.0)
        if side in ("bottom", "both")
        else np.ones_like(p)
    )
    return np.minimum(top, bot)


def _build_toolflat_mesh(
    amplitude: float, *, nz_per_ply: int = 1, s_trans: int = 2, nx: int = 16
) -> tuple[MeshData, np.ndarray]:
    """Coarse hand-rolled tool-flat mesh (uniform core, flat top surface).

    Returns the deformed mesh and the per-node z-displacement field.
    """
    mat = MaterialLibrary().get("AS4_3501_6")
    lam = Laminate.from_angles([0.0] * N_PLIES, mat, PLY_T)
    Lx = LAM
    flat = WrinkleMesh(
        lam, None, Lx=Lx, Ly=4.0, nx=nx, ny=1, nz_per_ply=nz_per_ply
    ).generate()
    nodes = flat.nodes.copy()
    per_layer = (nx + 1) * 2
    k_idx = np.arange(flat.n_nodes) // per_layer
    node_ply = np.minimum(k_idx // nz_per_ply, N_PLIES - 1)
    prof = GaussianSinusoidal(
        amplitude=amplitude, wavelength=LAM, width=1.0e6, center=Lx / 2.0
    )
    dz = prof.displacement(nodes[:, 0]) * _toolflat_decay(
        node_ply, N_PLIES, s_trans, "top"
    )
    nodes[:, 2] += dz
    mesh = MeshData(
        nodes=nodes,
        elements=flat.elements,
        ply_ids=flat.ply_ids,
        fiber_angles=flat.fiber_angles,
        ply_angles=flat.ply_angles,
        nx=nx,
        ny=1,
        nz=flat.nz,
    )
    return mesh, dz


def test_toolflat_gap_is_full_amplitude():
    """The trough pocket depth under the flat surface is ~= the full A."""
    A = 0.25  # a safe amplitude for S = 2, nz_per_ply = 1
    mesh, dz = _build_toolflat_mesh(A, nz_per_ply=1, s_trans=2)
    # Top surface node layer is pinned exactly flat.
    per_layer = (mesh.nx + 1) * 2
    k_idx = np.arange(mesh.n_nodes) // per_layer
    top = mesh.nodes[k_idx == mesh.nz, 2]
    assert float(top.max() - top.min()) < 1e-12
    # The pocket depth is the displacement the outermost undulating (core)
    # plies pull away from the pinned surface — the full amplitude A, since
    # the core carries decay = 1 while the surface is pinned at 0.
    pocket_depth = float(np.max(np.abs(dz)))
    assert np.isclose(pocket_depth, A, rtol=0.02)  # ~= full amplitude A
    assert pocket_depth / PLY_T > 1.0  # >= one ply thickness (vs ~0.25 linear)


def test_inversion_bound_is_S_times_t_over_nz():
    """Positive Jacobians at the bound; inversion just beyond it."""
    s_trans = 2
    for nz_per_ply in (1, 2):
        a_crit = s_trans * PLY_T / nz_per_ply
        # At the safe fraction and at the bound: every element is valid.
        for frac in (0.8, 1.0):
            mesh, _ = _build_toolflat_mesh(
                frac * a_crit, nz_per_ply=nz_per_ply, s_trans=s_trans
            )
            detj = _detJ_at_centroid_batch(mesh.nodes[mesh.elements])
            assert detj.min() > 0.0, (
                f"nz_per_ply={nz_per_ply} frac={frac}: min detJ "
                f"{detj.min():.3e} should be positive at/under the bound"
            )
        # Just beyond the bound: elements invert (negative Jacobian).
        mesh, _ = _build_toolflat_mesh(
            1.1 * a_crit, nz_per_ply=nz_per_ply, s_trans=s_trans
        )
        detj = _detJ_at_centroid_batch(mesh.nodes[mesh.elements])
        assert detj.min() < 0.0, (
            f"nz_per_ply={nz_per_ply}: amplitude 1.1x the bound must invert"
        )
