"""Per-node parity between the vectorised and legacy loop implementations of
:meth:`WrinkleConfiguration.apply_to_nodes` and
:meth:`WrinkleConfiguration.fiber_angles_at_nodes` (issue #185).

The current production code is a fully vectorised NumPy implementation. To
guard against future regressions we keep a faithful copy of the original
per-node Python loops here and assert element-by-element equality at
``atol=1e-12`` for every supported morphology / decay mode / amplitude
profile combination.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from wrinklefe.core.morphology import WrinkleConfiguration, WrinklePlacement
from wrinklefe.core.wrinkle import GaussianSinusoidal, WrinkleSurface3D


# ----------------------------------------------------------------------
# Legacy reference implementations (verbatim port of the pre-vectorised
# Python loops). Kept here so the production class can stay clean.
# ----------------------------------------------------------------------

def _legacy_apply_to_nodes(
    cfg: WrinkleConfiguration,
    nodes: np.ndarray,
    ply_ids: np.ndarray,
    n_plies: int,
) -> np.ndarray:
    deformed = nodes.copy()

    for wrinkle in cfg.wrinkles:
        k = wrinkle.ply_interface
        profile = wrinkle.profile

        if isinstance(profile, WrinkleSurface3D):
            wavelength = profile.profile.wavelength
        else:
            wavelength = profile.wavelength
        delta_x = wrinkle.phase_offset * wavelength / (2.0 * np.pi)

        for node_idx in range(len(nodes)):
            x = nodes[node_idx, 0]
            p = int(ply_ids[node_idx])
            x_shifted = x - delta_x

            if isinstance(profile, WrinkleSurface3D):
                y = nodes[node_idx, 1]
                dz = float(profile.displacement(
                    np.atleast_1d(x_shifted), np.atleast_1d(y)
                )[0])
            else:
                y = nodes[node_idx, 1]
                dz = float(profile.displacement(np.atleast_1d(x_shifted))[0])

            dz *= cfg._amplitude_scale(wrinkle, x, y)

            if cfg.decay_mode == "uniform":
                decay = 1.0
            elif cfg.decay_mode == "graded":
                if n_plies > 1:
                    p_mid = (n_plies - 1) / 2.0
                    raw = 1.0 - abs(p - p_mid) / p_mid
                    decay = cfg.decay_floor + (1.0 - cfg.decay_floor) * raw
                else:
                    decay = 1.0
                decay = max(0.0, decay)
            else:
                decay = cfg._ply_decay(p, k, n_plies)

            deformed[node_idx, 2] += dz * decay

    return deformed


def _legacy_fiber_angles_at_nodes(
    cfg: WrinkleConfiguration,
    nodes: np.ndarray,
    ply_ids: np.ndarray,
    n_plies: int,
) -> np.ndarray:
    n_nodes = len(nodes)
    angle_sq = np.zeros(n_nodes, dtype=np.float64)

    for wrinkle in cfg.wrinkles:
        profile = wrinkle.profile
        k = wrinkle.ply_interface
        delta_x = wrinkle.phase_offset * profile.wavelength / (2.0 * np.pi)

        for node_idx in range(n_nodes):
            x = nodes[node_idx, 0]
            y = nodes[node_idx, 1] if nodes.shape[1] >= 2 else 0.0
            p = int(ply_ids[node_idx])
            x_shifted = x - delta_x

            slope = profile.slope(x_shifted)
            amp_scale = cfg._amplitude_scale(wrinkle, x, y)

            if cfg.decay_mode == "uniform":
                decay = 1.0
            elif cfg.decay_mode == "graded":
                if n_plies > 1:
                    p_mid = (n_plies - 1) / 2.0
                    raw = max(0.0, 1.0 - abs(p - p_mid) / p_mid)
                    decay = cfg.decay_floor + (1.0 - cfg.decay_floor) * raw
                else:
                    decay = 1.0
            else:
                decay = cfg._ply_decay(p, k, n_plies)

            decay = decay * amp_scale
            angle = np.arctan(np.abs(slope)) * decay
            angle_sq[node_idx] += angle ** 2

    return np.sqrt(angle_sq)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

def _structured_mesh(nx=12, ny=6, nz=24):
    """Build a tensor-product mesh with shape (nx, ny, nz) flattened."""
    xs = np.linspace(-20.0, 20.0, nx)
    ys = np.linspace(0.0, 10.0, ny)
    n_nodes_per_ply = nx * ny
    nodes = []
    ply_ids = []
    for p in range(nz):
        for j in range(ny):
            for i in range(nx):
                nodes.append([xs[i], ys[j], 0.0])
                ply_ids.append(p)
    return (
        np.asarray(nodes, dtype=float),
        np.asarray(ply_ids, dtype=int),
        nz,
    )


def _profile():
    return GaussianSinusoidal(amplitude=0.366, wavelength=16.0, width=12.0, center=0.0)


def _config_for_morphology(name: str) -> WrinkleConfiguration:
    """Return a WrinkleConfiguration for each supported morphology / decay mode."""
    p = _profile()
    if name in ("stack", "convex", "concave"):
        return WrinkleConfiguration.from_morphology_name(
            name, p, interface1=8, interface2=14
        )
    if name in ("graded", "uniform"):
        return WrinkleConfiguration.from_morphology_name(
            name, p, interface1=8, interface2=14, decay_floor=0.25
        )
    if name == "single":
        return WrinkleConfiguration(
            [WrinklePlacement(p, ply_interface=10, phase_offset=0.0)]
        )
    raise ValueError(name)


MORPHOLOGIES = ["stack", "convex", "concave", "graded", "uniform", "single"]


# ----------------------------------------------------------------------
# Parity tests
# ----------------------------------------------------------------------

@pytest.mark.parametrize("morphology", MORPHOLOGIES)
def test_apply_to_nodes_parity(morphology):
    cfg = _config_for_morphology(morphology)
    nodes, ply_ids, n_plies = _structured_mesh()

    out_vec = cfg.apply_to_nodes(nodes, ply_ids, n_plies)
    out_legacy = _legacy_apply_to_nodes(cfg, nodes, ply_ids, n_plies)

    npt.assert_allclose(out_vec, out_legacy, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("morphology", MORPHOLOGIES)
def test_fiber_angles_at_nodes_parity(morphology):
    cfg = _config_for_morphology(morphology)
    nodes, ply_ids, n_plies = _structured_mesh()

    out_vec = cfg.fiber_angles_at_nodes(nodes, ply_ids, n_plies=n_plies)
    out_legacy = _legacy_fiber_angles_at_nodes(cfg, nodes, ply_ids, n_plies)

    npt.assert_allclose(out_vec, out_legacy, atol=1e-12, rtol=1e-12)


# Cross-check that the amplitude-profile modulation (issue #3) is also
# bit-identical for every supported axis + shape combination.

@pytest.mark.parametrize("amplitude_profile", ["constant", "gaussian", "linear"])
@pytest.mark.parametrize("axis", ["x", "y"])
def test_apply_to_nodes_amplitude_profile_parity(amplitude_profile, axis):
    profile = _profile()
    cfg = WrinkleConfiguration(
        [WrinklePlacement(profile, ply_interface=10, phase_offset=0.5)],
        amplitude_profile=amplitude_profile,
        amplitude_profile_axis=axis,
        amplitude_profile_decay_length=6.0,
    )
    nodes, ply_ids, n_plies = _structured_mesh(nx=8, ny=5, nz=12)

    out_vec = cfg.apply_to_nodes(nodes, ply_ids, n_plies)
    out_legacy = _legacy_apply_to_nodes(cfg, nodes, ply_ids, n_plies)
    npt.assert_allclose(out_vec, out_legacy, atol=1e-12, rtol=1e-12)

    ang_vec = cfg.fiber_angles_at_nodes(nodes, ply_ids, n_plies=n_plies)
    ang_legacy = _legacy_fiber_angles_at_nodes(cfg, nodes, ply_ids, n_plies)
    npt.assert_allclose(ang_vec, ang_legacy, atol=1e-12, rtol=1e-12)
