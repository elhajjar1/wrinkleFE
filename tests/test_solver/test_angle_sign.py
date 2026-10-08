"""The wrinkle material frame follows the SIGNED fibre angle.

The nodal fibre-angle field is the signed slope of the composed wrinkle
surface, ``theta = arctan(dz/dx)``, and the element stiffness / recovery
rotation is its negation: ``rotate_stiffness_3d(C, phi, axis='y')``
aligns the material 1-axis with ``(cos phi, 0, -sin phi)``, so a fibre
tilted toward +z (positive slope) needs ``phi = -theta``.

Before the sign fix the field was ``arctan|dz/dx|`` and the rotation
used it directly: the negative-slope flank of every wrinkle came out
right and the positive-slope flank mirror-handed, because the rotated
stiffness is odd in the angle (its sigma_11 <-> gamma_13 coupling flips
sign). These tests pin both the transform convention and the
assembler-to-geometry consistency that the old field broke.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from wrinklefe.core.laminate import Laminate
from wrinklefe.core.material import MaterialLibrary
from wrinklefe.core.mesh import WrinkleMesh
from wrinklefe.core.morphology import WrinkleConfiguration
from wrinklefe.core.transforms import rotate_stiffness_3d
from wrinklefe.core.wrinkle import GaussianSinusoidal
from wrinklefe.solver.assembler import GlobalAssembler

_VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def _direct_tensor_rotation(C6: np.ndarray, alpha: float) -> np.ndarray:
    """Stiffness of a ply whose fibre axis is (cos a, 0, sin a), built by
    plain 4th-order tensor rotation — no package transform conventions."""
    Ct = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(_VOIGT):
        for b, (k, m) in enumerate(_VOIGT):
            Ct[i, j, k, m] = Ct[j, i, k, m] = C6[a, b]
            Ct[i, j, m, k] = Ct[j, i, m, k] = C6[a, b]
    ca, sa = np.cos(alpha), np.sin(alpha)
    # Columns = material basis vectors in global coordinates.
    R = np.array([[ca, 0.0, -sa], [0.0, 1.0, 0.0], [sa, 0.0, ca]])
    Cg = np.einsum("ip,jq,kr,ls,pqrs->ijkl", R, R, R, R, Ct)
    out = np.zeros((6, 6))
    for a, (i, j) in enumerate(_VOIGT):
        for b, (k, m) in enumerate(_VOIGT):
            out[a, b] = Cg[i, j, k, m]
    return out


class TestRotationConvention:
    """rotate_stiffness_3d's y-axis sign convention, pinned numerically."""

    def test_fibre_toward_plus_z_is_the_negated_angle(self):
        m = MaterialLibrary().get("IM7_8552")
        C = np.asarray(m.stiffness_matrix)
        alpha = 0.3  # fibre tilted toward +z
        want = _direct_tensor_rotation(C, alpha)
        npt.assert_allclose(
            rotate_stiffness_3d(C, -alpha, axis="y"), want, rtol=1e-9
        )
        # And the positive angle is its mirror image, not the same thing:
        # the sigma_11 <-> gamma_13 coupling is odd in the angle.
        got_plus = rotate_stiffness_3d(C, alpha, axis="y")
        assert got_plus[0, 4] == pytest.approx(-want[0, 4], rel=1e-9)
        assert abs(want[0, 4]) > 1.0e3  # the coupling is far from zero


class TestAssemblerFollowsGeometry:
    """Element wrinkle rotations match the geometry the mesh draws.

    For interface elements (decay = 1) the rotation angle handed to the
    element must be ``-arctan`` of the geometric slope of the element's
    own displaced edges — on BOTH flanks. The old unsigned field failed
    this on the positive-slope flank.
    """

    def test_rotation_matches_displaced_edges_on_both_flanks(self):
        mat = MaterialLibrary().get("IM7_8552")
        lam = Laminate.from_angles([0.0] * 4, mat, ply_thickness=0.4)
        prof = GaussianSinusoidal(
            amplitude=0.4, wavelength=16.0, width=12.0, center=12.0,
        )
        wc = WrinkleConfiguration.from_morphology_name(
            "graded", prof, interface1=1, interface2=2, decay_floor=0.0,
        )
        mesh = WrinkleMesh(
            laminate=lam, wrinkle_config=wc, Lx=24.0, Ly=4.0,
            nx=48, ny=1, nz_per_ply=1,
        ).generate()
        asm = GlobalAssembler(mesh, lam)

        checked_pos = checked_neg = 0
        for e in range(mesh.n_elements):
            if int(mesh.ply_ids[e]) != 1:  # interface ply: decay = 1
                continue
            elem = asm.create_element(e)
            phi = float(np.mean(elem.wrinkle_angles))
            if abs(phi) < np.radians(1.0):
                continue  # crest / far field: slope too small to sign
            coords = mesh.nodes[mesh.elements[e]]
            order = np.argsort(coords[:, 0])
            lo, hi = coords[order[:4]], coords[order[4:]]
            slope = (hi[:, 2].mean() - lo[:, 2].mean()) / (
                hi[:, 0].mean() - lo[:, 0].mean()
            )
            # Rotation angle = NEGATED geometric angle (secant vs nodal
            # tangent slope differ by the mesh spacing, hence the loose
            # relative tolerance; the SIGN is the point of this test).
            assert phi == pytest.approx(-np.arctan(slope), rel=0.2), (
                f"element {e}: phi={phi:+.4f}, "
                f"geometric arctan(slope)={np.arctan(slope):+.4f}"
            )
            if slope > 0:
                checked_pos += 1
            else:
                checked_neg += 1
        # Both flanks genuinely exercised.
        assert checked_pos >= 5 and checked_neg >= 5
