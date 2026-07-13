"""Benchmark: batched laminate modulus knockdown (guards #301).

``_laminate_modulus_knockdown`` batches the per-(ply, x) off-axis
rotation and plane-stress condensation over the whole grid (#301, a ~52x
speedup over the former Python loop). This benchmark drives it on a
multi-ply, multi-wrinkle grid and checks the knockdown stays in the
physical ``(0, 1]`` band.
"""
from __future__ import annotations

import numpy as np
import pytest

from wrinklefe.analysis import _laminate_modulus_knockdown
from wrinklefe.core.laminate import Laminate
from wrinklefe.core.material import OrthotropicMaterial

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]

ANGLES = [0.0, 45.0, -45.0, 90.0, 90.0, -45.0, 45.0, 0.0]
N_X = 4000  # longitudinal stations
PLY_THICKNESS = 0.183
DOMAIN = 40.0


def _build_inputs():
    material = OrthotropicMaterial()
    n_plies = len(ANGLES)
    x = np.linspace(-DOMAIN / 2.0, DOMAIN / 2.0, N_X)

    # Two wrinkles at distinct interfaces, mirroring the production
    # slope-field / through-thickness-decay construction.
    specs = [
        # (amplitude, wavelength, width, z_center, decay_scale)
        (0.3, 8.0, 6.0, 3.5 * PLY_THICKNESS, 4.0),
        (0.2, 12.0, 8.0, 4.5 * PLY_THICKNESS, 6.0),
    ]
    n_w = len(specs)
    slope_field = np.empty((n_w, N_X), dtype=float)
    ply_decays = np.empty((n_plies, n_w, N_X), dtype=float)
    for w, (amp, lam, wid, z_center, ds) in enumerate(specs):
        gauss_env = np.exp(-(x**2) / (wid**2))
        k = 2.0 * np.pi / lam
        slope_field[w] = amp * gauss_env * (
            (-2.0 * x / (wid**2)) * np.cos(k * x) - k * np.sin(k * x)
        )
        sigma_sq2 = 2.0 * ds * ds
        for p in range(n_plies):
            z_p = (p + 0.5) * PLY_THICKNESS
            ply_decays[p, w, :] = np.exp(-((z_p - z_center) ** 2) / sigma_sq2)

    laminate = Laminate.from_angles(ANGLES, material, PLY_THICKNESS)
    return material, laminate, slope_field, ply_decays


def test_bench_laminate_modulus_knockdown(benchmark):
    material, laminate, slope_field, ply_decays = _build_inputs()

    def _knockdown():
        return _laminate_modulus_knockdown(
            slope_field=slope_field,
            ply_decays=ply_decays,
            angles=ANGLES,
            stiffness_3d=material.stiffness_matrix,
            ply_thickness=PLY_THICKNESS,
            E_x0=laminate.Ex,
        )

    kd = benchmark(_knockdown)

    # Correctness invariant: a wavy laminate is knocked down into (0, 1].
    assert np.isfinite(kd)
    assert 0.0 < kd <= 1.0
