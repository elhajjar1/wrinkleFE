"""The eccentricity measure behind the issue #439 regime split.

``validation/bending_mode_study.py`` classifies wrinkles by how far they
move the laminate's load path (the stiffness-weighted centroid of the
section). These tests pin the two facts the study's argument rests on:
a whole-thickness wave offsets the load path by its amplitude, and an
embedded wrinkle with flat outer surfaces does not offset it at all in a
homogeneous (UD) section.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from wrinklefe.analysis import AnalysisConfig
from wrinklefe.core.material import MaterialLibrary

_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def study():
    sys.path.insert(0, str(_ROOT / "validation"))
    sys.path.insert(0, str(_ROOT / "scripts"))
    import bending_mode_study

    return bending_mode_study


def _ud_config(morphology: str) -> AnalysisConfig:
    return AnalysisConfig(
        amplitude=0.5, wavelength=20.0, width=20.0, morphology=morphology,
        loading="compression", material=MaterialLibrary().get("IM7_8552"),
        angles=[0.0] * 16, ply_thickness=0.25, analytical_only=True,
    )


def test_whole_thickness_wave_offsets_the_load_path_by_its_amplitude(study):
    geom = study.section_geometry(_ud_config("uniform"))
    # Measured from the domain edge, where the load line runs: the
    # Gaussian-sinusoid's side lobe sits about 0.05 mm below the far-field
    # line there (envelope exp(-1.5**2) at 1.5 wavelengths), so e is a
    # little above the amplitude.
    assert 0.5 <= geom["e"] <= 0.5 * 1.12
    assert geom["t_c"] == pytest.approx(geom["t_0"], rel=0.01)


def test_embedded_flat_surface_wrinkle_leaves_the_load_path(study):
    geom = study.section_geometry(_ud_config("graded"))
    assert geom["e"] < 0.01 * geom["t_0"]


def test_bending_knockdown_is_the_eccentric_column_formula(study):
    geom = {"e": 0.5, "t_c": 4.0, "t_0": 4.0}
    assert study.kd_bend(geom) == pytest.approx(1.0 / (1.0 + 6.0 * 0.5 / 4.0))
