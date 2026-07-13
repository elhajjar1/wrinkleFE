"""Shared deterministic fixtures for the performance benchmarks.

All inputs are small and fixed so a benchmark run is reproducible and
fast enough to sit inside the CI ``benchmarks`` job. The heavier
per-benchmark inputs (stress fields, node clouds, CZM meshes) are built
in their own modules.
"""
from __future__ import annotations

import numpy as np
import pytest

from wrinklefe.core.laminate import Laminate
from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.core.mesh import WrinkleMesh
from wrinklefe.core.morphology import WrinkleConfiguration
from wrinklefe.core.wrinkle import GaussianSinusoidal


@pytest.fixture(scope="session")
def coarse_material() -> OrthotropicMaterial:
    return OrthotropicMaterial()


@pytest.fixture(scope="session")
def coarse_wrinkled_mesh(coarse_material):
    """A small 8-ply wrinkled hex8 mesh reused by the FE kernels.

    Returns ``(mesh, laminate)``. Geometry is intentionally coarse
    (nx=8, ny=4, one element per ply) so assembly and a direct solve
    complete in well under a second.
    """
    laminate = Laminate.from_angles(
        [0.0] * 8, material=coarse_material, ply_thickness=0.183
    )
    wrinkle = GaussianSinusoidal(
        amplitude=0.2, wavelength=8.0, width=6.0, center=0.0
    )
    cfg = WrinkleConfiguration.dual_wrinkle(
        profile=wrinkle, interface1=3, interface2=4, phase=0.0
    )
    gen = WrinkleMesh(
        laminate=laminate,
        wrinkle_config=cfg,
        Lx=10.0,
        Ly=6.0,
        nx=8,
        ny=4,
        nz_per_ply=1,
    )
    return gen.generate(), laminate


def stress_sample(n: int, seed: int = 20260713) -> np.ndarray:
    """Deterministic ``(n, 6)`` stress field spanning the failure regimes."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(-800.0, 800.0, (n, 6))
    # Bias the fibre-direction component so both tension and compression
    # (and hence the fibre-tension / fibre-kinking branches) are exercised.
    s[:, 0] = rng.uniform(-2200.0, 2200.0, n)
    return s
