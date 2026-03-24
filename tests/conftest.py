"""Shared pytest fixtures for WrinkleFE Phase 1 core module tests."""

import numpy as np
import pytest

from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.core.laminate import Laminate, Ply
from wrinklefe.core.wrinkle import GaussianSinusoidal
from wrinklefe.core.morphology import WrinkleConfiguration, WrinklePlacement


@pytest.fixture
def x850_material():
    """IM7/8552 default OrthotropicMaterial."""
    return OrthotropicMaterial()


@pytest.fixture
def quasi_iso_laminate(x850_material):
    """[0/45/-45/90]2s laminate (16 plies) using X850 material, ply_thickness=0.183 mm."""
    half_angles = [0, 45, -45, 90, 0, 45, -45, 90]
    return Laminate.symmetric(half_angles, material=x850_material, ply_thickness=0.183)


@pytest.fixture
def gaussian_wrinkle():
    """GaussianSinusoidal wrinkle with A=0.366, lambda=16, w=12, x0=0."""
    return GaussianSinusoidal(amplitude=0.366, wavelength=16.0, width=12.0, center=0.0)


@pytest.fixture
def dual_wrinkle_config(gaussian_wrinkle):
    """WrinkleConfiguration with stack morphology (phase=0) at interfaces 7-8."""
    return WrinkleConfiguration.dual_wrinkle(
        profile=gaussian_wrinkle,
        interface1=7,
        interface2=8,
        phase=0.0,
    )
