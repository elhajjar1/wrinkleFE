"""UD kink-band strain budget anchored to the material card (issue #451).

A UD-dominated laminate (at least 75 % 0-deg plies) takes the kink-band
strain budget ``g = S12 / Xc`` on the peak-angle compression path, so its
knockdown is Budiansky-Fleck with intrinsic misalignment,
``KD_BF = g / (g + theta_eff)``, instead of the layup confinement model's
``gamma_Y_eff`` (0.032 for pure UD, floored at 0.016). These tests pin
the classification, the budget, where it applies and where it does not.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from wrinklefe.analysis import (
    _GAMMA_Y_UD,
    _UD_DOMINATED_FRACTION,
    AnalysisConfig,
    WrinkleAnalysis,
    _effective_gamma_Y,
    _is_ud_dominated,
    _ud_kink_strain_budget,
    _zero_ply_fraction,
)
from wrinklefe.core.layup import parse_layup
from wrinklefe.core.material import MaterialLibrary

_ROOT = Path(__file__).resolve().parents[1]
_LEDGER = json.loads(
    (_ROOT / "tests" / "test_validation" / "ledger.json").read_text())
_ML = MaterialLibrary()

#: The multidirectional layups of the validation set (Datasets A/B, C, D
#: from validation/plot_all_validation.py; H-MD, I, J-QI, M-MD from the
#: ledger). None may be classified UD-dominated: their results must stay
#: bit-for-bit what the confinement model gave.
_ELHAJJAR = [0, 45, 90, -45, 0, 45, -45, 0]
_MUKHO = [45, 45, 90, 90, -45, -45, 0, 0] * 3
_WANG = [45, 0, -45, 90, 45, 0, -45, 0, 45, 0]
_MD_LAYUPS = {
    "A/B Elhajjar": _ELHAJJAR + _ELHAJJAR[::-1],
    "C Mukhopadhyay": _MUKHO + _MUKHO[::-1],
    "D Wang": _WANG + _WANG[::-1],
    **{d["name"]: parse_layup(d["layup"]) for d in _LEDGER["datasets"]
       if d["name"].startswith(("shi_2025_md", "calvo_2023", "thor_2021_qi",
                                "elhajjar_shams_2016_md"))},
}


def _config(angles, *, morphology="uniform", loading="compression",
            material="IM7_8552", **kw):
    return AnalysisConfig(
        amplitude=0.29, wavelength=14.5, width=14.5, morphology=morphology,
        loading=loading, material=_ML.get(material), angles=list(angles),
        ply_thickness=0.131, analytical_only=True, **kw,
    )


def _run(cfg):
    return WrinkleAnalysis(cfg).run(analytical_only=True)


# ----------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------

def test_zero_ply_fraction():
    assert _zero_ply_fraction([0.0] * 8) == 1.0
    assert _zero_ply_fraction([0, 45, -45, 90]) == 0.25
    assert _zero_ply_fraction([]) == 0.0
    # 0 and 180 deg are the same fibre direction; +/-5 deg counts as 0.
    assert _zero_ply_fraction([180, -180, 5, -5, 90]) == pytest.approx(0.8)
    assert _zero_ply_fraction([6, 90]) == 0.0


def test_threshold_is_three_quarters_of_the_plies():
    assert _UD_DOMINATED_FRACTION == 0.75
    assert _is_ud_dominated([0, 0, 0, 90])          # exactly 75 %
    assert not _is_ud_dominated([0, 0, 90, 90])     # 50 %
    assert not _is_ud_dominated([0] * 74 + [90] * 26)
    assert _is_ud_dominated([0] * 75 + [90] * 25)
    assert not _is_ud_dominated([])


def test_pure_and_near_ud_laminates_are_ud_dominated():
    assert _is_ud_dominated([0.0] * 40)
    # Pilato (2022): 16 of 18 sub-plies at 0 deg (woven 0/90 pairs).
    pilato = next(d for d in _LEDGER["datasets"]
                  if d["name"].startswith("pilato_2022"))
    angles = parse_layup(pilato["layup"])
    assert _zero_ply_fraction(angles) == pytest.approx(16 / 18)
    assert _is_ud_dominated(angles)


@pytest.mark.parametrize("name", sorted(_MD_LAYUPS))
def test_no_multidirectional_validation_layup_is_ud_dominated(name):
    angles = _MD_LAYUPS[name]
    assert _zero_ply_fraction(angles) <= 0.40
    assert not _is_ud_dominated(angles)


# ----------------------------------------------------------------------
# The budget
# ----------------------------------------------------------------------

@pytest.mark.parametrize("card", ["IM7_8552", "UD_CFRP_PILATO2022",
                                  "YC300_LY5052", "IM6G_3501_6",
                                  "T700_SHI2025", "T700GC_2510_AGATE"])
def test_budget_is_s12_over_xc(card):
    mat = _ML.get(card)
    assert _ud_kink_strain_budget(mat) == mat.S12 / mat.Xc


def test_budget_values_quoted_in_the_docs():
    assert _ud_kink_strain_budget(_ML.get("IM7_8552")) == pytest.approx(
        0.0769, abs=5e-5)
    assert _ud_kink_strain_budget(
        _ML.get("UD_CFRP_PILATO2022")) == pytest.approx(0.1021, abs=5e-5)
    assert _ud_kink_strain_budget(_ML.get("IM6G_3501_6")) == pytest.approx(
        0.0433, abs=5e-5)
    assert _ud_kink_strain_budget(
        _ML.get("T700GC_2510_AGATE")) == pytest.approx(0.1071, abs=5e-5)


# ----------------------------------------------------------------------
# Where it applies: the peak-angle (non-graded) path
# ----------------------------------------------------------------------

def test_ud_peak_angle_knockdown_is_g_over_g_plus_theta():
    cfg = _config([0.0] * 40)
    res = _run(cfg)
    g = cfg.material.S12 / cfg.material.Xc
    assert res.gamma_Y_eff == g
    theta = res.effective_angle_rad
    assert theta == pytest.approx(math.atan(2 * math.pi * 0.29 / 14.5))
    # f_0 = 1 for pure UD.
    assert res.analytical_knockdown == pytest.approx(g / (g + theta),
                                                     rel=1e-12)


def test_near_ud_keeps_the_clt_weighting():
    angles = [0, 0, 0, 90] * 4
    cfg = _config(angles)
    res = _run(cfg)
    mat = cfg.material
    g = mat.S12 / mat.Xc
    f_0 = 12 * mat.E1 / (12 * mat.E1 + 4 * mat.E2)
    kd_bf = g / (g + res.effective_angle_rad)
    assert res.gamma_Y_eff == g
    assert res.analytical_knockdown == pytest.approx(
        f_0 * kd_bf + (1.0 - f_0), rel=1e-12)


def test_budget_follows_the_material_card():
    """Same geometry and layup: the card with the larger S12/Xc keeps
    more strength (the layup model gave both the same 0.032)."""
    lo = _run(_config([0.0] * 40, material="IM6G_3501_6"))      # 0.043
    hi = _run(_config([0.0] * 40, material="UD_CFRP_PILATO2022"))  # 0.102
    assert lo.gamma_Y_eff < hi.gamma_Y_eff
    assert lo.analytical_knockdown < hi.analytical_knockdown


def test_argon_fleck_quadratic_still_applies_on_top():
    """A user-set c_AF composes with g: KD = 1 / (1 + r + c_AF r^2),
    r = theta_eff / g. The default c_AF = 0 is the plain g / (g + theta)."""
    cfg = _config([0.0] * 40, kink_band_quadratic_coeff=0.5)
    res = _run(cfg)
    r = res.effective_angle_rad / (cfg.material.S12 / cfg.material.Xc)
    assert res.analytical_knockdown == pytest.approx(
        1.0 / (1.0 + r + 0.5 * r * r), rel=1e-12)
    assert res.analytical_knockdown < _run(
        _config([0.0] * 40)).analytical_knockdown


def test_ud_budget_is_less_conservative_than_the_layup_model():
    """The cards in the validation set all have S12/Xc above the layup
    model's UD value, so the knockdown rises on every one of them."""
    res = _run(_config([0.0] * 40))
    theta = res.effective_angle_rad
    assert res.gamma_Y_eff > _GAMMA_Y_UD
    assert res.analytical_knockdown > 1.0 / (1.0 + theta / _GAMMA_Y_UD)


def test_tension_does_not_take_the_ud_budget():
    """Tension no longer floors at the compression knockdown (issue #454),
    so the UD budget reaches only the compression result. The tension run
    reports the layup model's value and its knockdown is free to sit
    below the compression one; tests/test_tension_knockdown.py checks
    that the budget cannot move a tension result."""
    angles = [0.0] * 40
    comp = _run(_config(angles))
    tens = _run(_config(angles, loading="tension"))
    assert comp.gamma_Y_eff == _ud_kink_strain_budget(comp.config.material)
    assert tens.gamma_Y_eff == _effective_gamma_Y(angles)


# ----------------------------------------------------------------------
# Where it does not apply
# ----------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(_MD_LAYUPS))
def test_multidirectional_layups_keep_the_confinement_model(name):
    angles = _MD_LAYUPS[name]
    res = _run(_config(angles))
    assert res.gamma_Y_eff == _effective_gamma_Y(angles)


@pytest.mark.parametrize("loading", ["compression", "tension"])
def test_graded_path_keeps_the_confinement_model(loading):
    """The graded profile-proportional path keeps gamma_Y_eff: its budget
    is calibrated jointly with the profile average, and g there would
    push Dataset H (Shi, UD) and Dataset M's UD tension floor further
    unsafe."""
    res = _run(_config([0.0] * 40, morphology="graded", loading=loading))
    assert res.gamma_Y_eff == _GAMMA_Y_UD


def test_penetration_gate_path_is_untouched():
    from wrinklefe.core.penetration_gate import GATE_LI2024_MOULDED

    res = _run(_config([0.0] * 40, penetration_gate=GATE_LI2024_MOULDED))
    assert res.gamma_Y_eff == GATE_LI2024_MOULDED.gamma_Y
