"""Load-path eccentricity and the bending-regime flag (issue #439).

The validation study behind #439 found that the angle-based kink-band
knockdown is safe while a wrinkle leaves the laminate's load path in
place, and unsafe by up to +115% once the wrinkle moves it (e/t >= 0.10).
These tests pin the measurement, the flag, and every place the flag must
reach: the text summary, the schema-versioned results JSON, the NCR
summary and the app's result dict.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.eccentricity import (
    BENDING_REGIME_CAVEAT,
    ECCENTRICITY_REGIME_THRESHOLD,
    load_path_eccentricity,
)
from wrinklefe.core.material import MaterialLibrary
from wrinklefe.core.mesh import WrinkleMesh
from wrinklefe.io.export import build_analysis_summary, render_summary_markdown
from wrinklefe.io.results import SCHEMA_VERSION, export_results_json

_ROOT = Path(__file__).resolve().parents[1]
_QI = [0, 45, -45, 90] * 5
_QI = _QI + _QI[::-1]


def _config(amplitude=1.19, wavelength=27.9, morphology="uniform",
            loading="compression", angles=None, ply_thickness=0.131):
    """Thor (2021) quasi-isotropic IM7/8552 by default (Dataset J)."""
    return AnalysisConfig(
        amplitude=amplitude, wavelength=wavelength, width=wavelength,
        morphology=morphology, loading=loading,
        material=MaterialLibrary().get("IM7_8552"),
        angles=_QI if angles is None else angles,
        ply_thickness=ply_thickness, analytical_only=True,
    )


def _run(cfg):
    return WrinkleAnalysis(cfg).run(analytical_only=True)


# ----------------------------------------------------------------------
# The measurement
# ----------------------------------------------------------------------


def test_deformed_nodes_are_the_nodes_the_fe_mesh_uses():
    res = _run(_config())
    cfg = res.config
    gen = WrinkleMesh(laminate=res.laminate, wrinkle_config=res.wrinkle_config,
                      Lx=cfg.domain_length, Ly=cfg.domain_width,
                      nx=24, ny=2, nz_per_ply=1)
    nodes, _ = gen.deformed_nodes()
    np.testing.assert_array_equal(nodes, gen.generate().nodes)


def test_whole_thickness_wave_offsets_the_load_path_by_its_amplitude():
    res = _run(_config(amplitude=0.5, wavelength=20.0, angles=[0] * 16,
                       ply_thickness=0.25))
    ecc = load_path_eccentricity(res.laminate, res.wrinkle_config,
                                 res.config.domain_length,
                                 res.config.domain_width)
    # The Gaussian side lobe at the domain ends puts the load line a
    # little below the far-field mid-plane, so e sits just above A.
    assert 0.5 <= ecc.e_mm <= 0.56
    assert ecc.ratio == pytest.approx(ecc.e_mm / 4.0)
    assert ecc.section_thickness_mm == pytest.approx(4.0, rel=1e-6)


def test_embedded_flat_surface_wrinkle_leaves_the_load_path_in_place():
    res = _run(_config(amplitude=0.5, wavelength=20.0, morphology="graded",
                       angles=[0] * 16, ply_thickness=0.25))
    assert res.load_path_eccentricity_ratio < 0.01
    assert not res.bending_dominated


def test_a_flat_laminate_has_no_offset():
    res = _run(_config(amplitude=0.0))
    assert res.load_path_eccentricity_mm == 0.0
    assert not res.bending_dominated


# ----------------------------------------------------------------------
# The flag
# ----------------------------------------------------------------------


def test_thor_wave_1_is_flagged_and_wave_2_is_not():
    """Thor (2021): wave 1 failed by bending (analytical +52% unsafe),
    wave 2 by kinking (analytical -8%)."""
    severe = _run(_config())
    mild = _run(_config(amplitude=0.29, wavelength=14.5))
    assert severe.load_path_eccentricity_ratio == pytest.approx(0.251, abs=0.005)
    assert severe.bending_dominated
    assert mild.load_path_eccentricity_ratio == pytest.approx(0.061, abs=0.005)
    assert not mild.bending_dominated


def test_tension_reports_the_ratio_but_is_not_flagged():
    res = _run(_config(loading="tension"))
    assert res.load_path_eccentricity_ratio > ECCENTRICITY_REGIME_THRESHOLD
    assert not res.bending_dominated


def test_the_flag_alone_does_not_change_the_knockdown():
    """The flag is reporting only; what moves Thor's wave 1 (Dataset J,
    J-QI-1) is the eccentric-column bound of issue #452, applied because
    it is also a whole-thickness wave (``tests/test_bending_bound.py``).
    The kink-band value it replaced is still reported, unchanged."""
    res = _run(_config())
    assert res.kink_band_knockdown == pytest.approx(0.5271, abs=5e-4)
    assert res.bending_bound_applied
    assert res.analytical_knockdown == pytest.approx(0.3990, abs=5e-4)


def test_validation_cases_fall_on_the_documented_sides():
    sys.path.insert(0, str(_ROOT / "scripts"))
    from validate import case_config

    ledger = json.loads(
        (_ROOT / "tests" / "test_validation" / "ledger.json").read_text())

    def flagged(prefix, case_name):
        ds = next(d for d in ledger["datasets"] if d["name"].startswith(prefix))
        case = next(c for c in ds["cases"] if c["case"] == case_name)
        return _run(case_config(ds, case)).bending_dominated

    assert flagged("thor_2021_qi", "J-QI-1")
    assert not flagged("thor_2021_qi", "J-QI-2")
    assert not flagged("pilato_2022", "K-6")
    assert not flagged("li_2025", "S-M-3")


# ----------------------------------------------------------------------
# Where the flag must reach
# ----------------------------------------------------------------------


def test_summary_reports_the_ratio_and_warns_only_when_flagged():
    severe = _run(_config()).summary()
    mild = _run(_config(amplitude=0.29, wavelength=14.5)).summary()
    assert "Load-path eccentricity: e/t = 0.251" in severe
    assert "WARNING (bending-dominated" in severe
    assert "Load-path eccentricity: e/t = 0.061" in mild
    assert "WARNING" not in mild


def test_results_json_carries_the_keys(tmp_path):
    out = tmp_path / "r.json"
    export_results_json(_run(_config()), out)
    data = json.loads(out.read_text())
    assert data["schema_version"] == SCHEMA_VERSION
    assert tuple(int(x) for x in SCHEMA_VERSION.split(".")) >= (1, 5)
    analytical = data["analytical"]
    assert analytical["bending_dominated"] is True
    assert analytical["load_path_eccentricity_ratio"] == pytest.approx(
        0.251, abs=0.005)
    assert analytical["load_path_eccentricity_mm"] > 0.0


_DEFECT = {
    "amplitude_mm": 1.19, "wavelength_mm": 27.9, "width_mm": 27.9,
    "morphology": "uniform", "loading": "compression",
    "ply_thickness_mm": 0.131, "n_plies": 40, "layup": _QI,
    "material_name": "IM7_8552",
}


def _ncr(**extra):
    eng = {"analytical_knockdown": 0.527, "damage_index": 0.3,
           "analytical_strength_MPa": 632.0, **extra}
    return build_analysis_summary(defect=dict(_DEFECT), engineering=eng)


def test_ncr_summary_carries_the_caveat_when_flagged():
    summary = _ncr(load_path_eccentricity_ratio=0.251, bending_dominated=True)
    ea = summary["engineering_analysis"]
    assert ea["bending_dominated"] is True
    assert ea["bending_regime_caveat"] == BENDING_REGIME_CAVEAT
    md = render_summary_markdown(summary)
    assert "Load-path eccentricity e/t: 0.251" in md
    assert f"> **Bending-dominated:** {BENDING_REGIME_CAVEAT}" in md


def test_ncr_summary_is_quiet_when_not_flagged_or_not_supplied():
    quiet = _ncr(load_path_eccentricity_ratio=0.061, bending_dominated=False)
    assert quiet["engineering_analysis"]["bending_regime_caveat"] is None
    assert "Bending-dominated" not in render_summary_markdown(quiet)
    legacy = _ncr()
    assert legacy["engineering_analysis"]["load_path_eccentricity_ratio"] is None
    assert "Load-path eccentricity" not in render_summary_markdown(legacy)


@pytest.mark.viz
def test_app_result_dict_carries_the_flag():
    pytest.importorskip("streamlit", reason="Streamlit not installed.")
    sys.path.insert(0, str(_ROOT))
    import app

    material = MaterialLibrary().get("IM7_8552").to_dict()
    payload = tuple({
        "amplitude": 1.19, "wavelength": 27.9, "width": 27.9,
        "morphology": "uniform", "decay_floor": 0.0,
        "amplitude_profile": "constant",
        "amplitude_profile_decay_length": 5.0,
        "amplitude_profile_axis": "x", "loading": "compression",
        "ply_thickness": 0.131, "angles_tuple": tuple(float(a) for a in _QI),
        "applied_strain": -0.01,
        "material_tuple": tuple(sorted(material.items())),
        "analytical_only": True,
    }.items())
    result = app._run_analysis(payload)
    assert result["bending_dominated"] is True
    assert result["load_path_eccentricity_ratio"] == pytest.approx(0.251, abs=0.005)
