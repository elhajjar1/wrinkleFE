"""The eccentric-column bending bound on whole-thickness waves (issue #452).

On a compression analysis whose load-path eccentricity is at or above
``ECCENTRICITY_REGIME_THRESHOLD`` (``bending_dominated``) AND whose outer
surfaces both follow the load path (a whole-thickness wave), the
analytical knockdown is ``min(kink-band, (t_c/t_0) / (1 + 6 e/t_c))``.
A one-sided trough (Shi, Dataset H) is not covered and keeps the FE
advice. These tests pin the formula, the two-sided test, the governing
bound wherever a result is reported, the validation numbers the docs
quote, and the two properties the change must not break: nothing moves
below the threshold, and tension is untouched.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

import wrinklefe.core.eccentricity as eccentricity
from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis, WrinkleSpec
from wrinklefe.core.eccentricity import (
    BENDING_BOUND_CAVEAT,
    BENDING_REGIME_CAVEAT,
    ECCENTRICITY_REGIME_THRESHOLD,
    WHOLE_THICKNESS_SURFACE_FRACTION,
    LoadPathEccentricity,
    bending_caveat,
    bending_knockdown,
    load_path_eccentricity,
)
from wrinklefe.core.material import MaterialLibrary
from wrinklefe.io.export import build_analysis_summary, render_summary_markdown
from wrinklefe.io.results import SCHEMA_VERSION, results_to_dict

_ROOT = Path(__file__).resolve().parents[1]
_QI = [0, 45, -45, 90] * 5
_QI = _QI + _QI[::-1]


def _config(amplitude=1.19, wavelength=27.9, morphology="uniform",
            loading="compression", angles=None, ply_thickness=0.131, **kw):
    """Thor (2021) quasi-isotropic IM7/8552 by default (Dataset J, J-QI-1)."""
    return AnalysisConfig(
        amplitude=amplitude, wavelength=wavelength, width=wavelength,
        morphology=morphology, loading=loading,
        material=MaterialLibrary().get("IM7_8552"),
        angles=_QI if angles is None else angles,
        ply_thickness=ply_thickness, analytical_only=True, **kw,
    )


def _run(cfg):
    return WrinkleAnalysis(cfg).run(analytical_only=True)


def _measure(res):
    cfg = res.config
    return load_path_eccentricity(res.laminate, res.wrinkle_config,
                                  cfg.domain_length, cfg.domain_width)


def _ledger():
    return json.loads(
        (_ROOT / "tests" / "test_validation" / "ledger.json").read_text())


def _ledger_case(prefix, name, **overrides):
    sys.path.insert(0, str(_ROOT / "scripts"))
    from validate import case_config

    ds = next(d for d in _ledger()["datasets"] if d["name"].startswith(prefix))
    case = next(c for c in ds["cases"] if c["case"] == name)
    cfg = case_config(ds, case)
    if overrides:
        kw = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
        kw.update(overrides)
        cfg = type(cfg)(**kw)
    return cfg, case


# ----------------------------------------------------------------------
# The bound
# ----------------------------------------------------------------------


def _ecc(e, t_c=4.0, t_0=4.0, top=None, bottom=None):
    return LoadPathEccentricity(
        e_mm=e, ratio=e / t_0, thickness_mm=t_0, x_mm=0.0,
        section_thickness_mm=t_c,
        top_surface_offset_mm=e if top is None else top,
        bottom_surface_offset_mm=e if bottom is None else bottom,
    )


def test_bound_is_the_eccentric_column_formula():
    assert bending_knockdown(_ecc(0.5)) == pytest.approx(1.0 / (1.0 + 0.75))
    # A thinned section loses area as well as section modulus.
    assert bending_knockdown(_ecc(0.5, t_c=3.0)) == pytest.approx(
        (3.0 / 4.0) / (1.0 + 6.0 * 0.5 / 3.0))


def test_bound_is_one_without_an_offset():
    assert bending_knockdown(_ecc(0.0)) == 1.0


def test_bound_falls_monotonically_with_the_offset():
    values = [bending_knockdown(_ecc(e)) for e in (0.1, 0.4, 0.8, 1.6)]
    assert values == sorted(values, reverse=True)


def test_bound_matches_the_validation_study_prototype():
    """The study that motivated the bound (#439) used the same formula."""
    sys.path.insert(0, str(_ROOT / "validation"))
    sys.path.insert(0, str(_ROOT / "scripts"))
    import bending_mode_study

    geom = {"e": 0.7, "t_c": 3.1, "t_0": 3.4}
    assert bending_knockdown(_ecc(0.7, t_c=3.1, t_0=3.4)) == pytest.approx(
        bending_mode_study.kd_bend(geom))


# ----------------------------------------------------------------------
# Both surfaces wavy: the whole-thickness test
# ----------------------------------------------------------------------


@pytest.mark.parametrize("top, bottom, expected", [
    (1.0, 1.0, True),           # whole-thickness wave
    (0.5, 0.5, True),           # exactly at the fraction
    (0.49, 1.51, False),        # one surface lags
    (0.0, 2.0, False),          # one-sided trough
    (-1.0, 3.0, False),         # surfaces move apart: a bulge
])
def test_whole_thickness_needs_both_surfaces_with_the_load_path(
        top, bottom, expected):
    assert WHOLE_THICKNESS_SURFACE_FRACTION == 0.5
    assert _ecc(1.0, top=top, bottom=bottom).whole_thickness is expected


def test_no_offset_is_never_whole_thickness():
    assert not _ecc(0.0, top=0.0, bottom=0.0).whole_thickness


def test_uniform_wave_moves_both_surfaces_with_the_load_path():
    ecc = _measure(_run(_config()))
    assert ecc.top_surface_offset_mm == pytest.approx(ecc.e_mm, rel=1e-6)
    assert ecc.bottom_surface_offset_mm == pytest.approx(ecc.e_mm, rel=1e-6)
    assert ecc.whole_thickness


@pytest.mark.parametrize("side, moving", [("top", "bottom"),
                                          ("bottom", "top")])
def test_one_sided_trough_is_not_whole_thickness(side, moving):
    """A face pinned flat by the tool does not follow the load path."""
    res = _run(_config(morphology="tool_flat", surface_pocket_side=side,
                       surface_transition_plies=14,
                       enable_surface_resin_pockets=False))
    ecc = _measure(res)
    assert ecc.ratio >= ECCENTRICITY_REGIME_THRESHOLD
    pinned = (ecc.top_surface_offset_mm if side == "top"
              else ecc.bottom_surface_offset_mm)
    free = (ecc.top_surface_offset_mm if moving == "top"
            else ecc.bottom_surface_offset_mm)
    assert abs(pinned) < 1e-9
    assert free == pytest.approx(2.0 * ecc.e_mm, rel=0.3)
    assert not ecc.whole_thickness
    assert res.bending_dominated and not res.whole_thickness_wave


@pytest.mark.parametrize("morphology", ["graded", "stack", "convex",
                                        "concave"])
def test_flat_surface_morphologies_are_not_whole_thickness(morphology):
    res = _run(_config(morphology=morphology))
    assert not res.whole_thickness_wave
    assert not res.bending_bound_applied


def test_multi_wrinkle_layout_with_flat_surfaces_is_not_whole_thickness():
    specs = [WrinkleSpec(amplitude=1.19, wavelength=27.9, width=27.9,
                         ply_interface=k) for k in (10, 25)]
    res = _run(_config(morphology="stack", wrinkles=specs))
    assert not res.whole_thickness_wave
    assert res.analytical_knockdown == res.kink_band_knockdown


def test_graded_wave_with_wavy_surfaces_is_whole_thickness():
    """``decay_floor`` keeps the surfaces wavy: once that moves the load
    path past the threshold, the section bends like any other
    whole-thickness wave. ``decay_floor=1`` is the uniform wave."""
    res = _run(_config(morphology="graded", decay_floor=1.0))
    assert res.whole_thickness_wave and res.bending_bound_applied
    assert res.bending_knockdown == pytest.approx(
        _run(_config()).bending_knockdown, rel=1e-9)


# ----------------------------------------------------------------------
# Applying the bound and reporting which one governed
# ----------------------------------------------------------------------


def test_thor_wave_1_is_governed_by_bending():
    res = _run(_config())
    ecc = _measure(res)
    assert res.bending_dominated and res.whole_thickness_wave
    assert res.bending_bound_applied
    assert res.governing_bound == "bending"
    assert res.kink_band_knockdown == pytest.approx(0.5271, abs=5e-4)
    assert res.bending_knockdown == pytest.approx(bending_knockdown(ecc))
    assert res.analytical_knockdown == res.bending_knockdown
    assert res.analytical_strength_MPa == pytest.approx(
        res.config.material.Xc * res.bending_knockdown)


def test_kink_band_governs_when_it_is_lower():
    """Thor's UD wave 1: same geometry, so the same bound, but the
    angle-only kink-band is already far lower and stays."""
    res = _run(_config(angles=[0] * 40))
    assert res.bending_bound_applied
    assert res.governing_bound == "kink_band"
    assert res.bending_knockdown > res.kink_band_knockdown
    assert res.analytical_knockdown == res.kink_band_knockdown


def test_below_the_threshold_the_bound_is_not_applied():
    res = _run(_config(amplitude=0.29, wavelength=14.5))  # J-QI-2
    assert res.load_path_eccentricity_ratio < ECCENTRICITY_REGIME_THRESHOLD
    assert res.whole_thickness_wave
    assert not res.bending_bound_applied
    assert res.bending_knockdown is None
    assert res.governing_bound == "kink_band"
    assert res.analytical_knockdown == res.kink_band_knockdown


def test_one_sided_trough_keeps_the_kink_band_and_the_warning():
    """Shi (2025), Dataset H, on its FE recipe: bending-dominated, but a
    one-sided trough, so the bound is not applied."""
    cfg, _ = _ledger_case(
        "shi_2025_md", "H-MD-30", morphology="tool_flat",
        surface_pocket_side="bottom", surface_transition_plies=10,
        enable_surface_resin_pockets=False, analytical_only=True)
    res = _run(cfg)
    assert res.bending_dominated and not res.whole_thickness_wave
    assert not res.bending_bound_applied
    assert res.governing_bound == "kink_band"
    assert res.analytical_knockdown == res.kink_band_knockdown
    summary = res.summary()
    assert "WARNING (bending-dominated):" in summary
    assert "bending bound applied" not in summary


def test_the_bound_applies_to_the_gate_value_too():
    """With a penetration gate the compression knockdown is the gate's;
    the bound is a separate failure mode and caps it the same way."""
    from wrinklefe.core.penetration_gate import GATE_LI2024_MOULDED

    res = _run(_config(angles=[0] * 40, penetration_gate=GATE_LI2024_MOULDED))
    assert res.bending_bound_applied
    assert res.analytical_knockdown == pytest.approx(
        min(res.kink_band_knockdown, res.bending_knockdown))


# ----------------------------------------------------------------------
# Properties: nothing below the threshold, nothing in tension
# ----------------------------------------------------------------------


def _without_bound(monkeypatch):
    """Make the bound absurdly low, so any path that reaches it shows."""
    monkeypatch.setattr(eccentricity, "bending_knockdown", lambda ecc: 0.0)


@pytest.mark.parametrize("amplitude", [0.05, 0.15, 0.3, 0.45])
def test_nothing_changes_below_the_threshold(amplitude, monkeypatch):
    """Whole-thickness waves up to just under e/t = 0.10 (Thor QI,
    t = 5.24 mm): the bound is never consulted."""
    reference = _run(_config(amplitude=amplitude))
    assert reference.load_path_eccentricity_ratio < (
        ECCENTRICITY_REGIME_THRESHOLD)
    _without_bound(monkeypatch)
    res = _run(_config(amplitude=amplitude))
    assert res.analytical_knockdown == reference.analytical_knockdown
    assert res.analytical_knockdown == res.kink_band_knockdown
    assert not res.bending_bound_applied


def test_the_threshold_is_where_the_bound_switches_on(monkeypatch):
    _without_bound(monkeypatch)
    below = _run(_config(amplitude=0.45))
    above = _run(_config(amplitude=0.6))
    assert below.load_path_eccentricity_ratio < ECCENTRICITY_REGIME_THRESHOLD
    assert above.load_path_eccentricity_ratio >= ECCENTRICITY_REGIME_THRESHOLD
    assert below.analytical_knockdown > 0.0
    assert above.analytical_knockdown == 0.0


def test_ledger_cases_below_the_threshold_are_untouched(monkeypatch):
    """Every compression case in the validation ledger below e/t = 0.10
    predicts exactly what it did before the bound existed."""
    sys.path.insert(0, str(_ROOT / "scripts"))
    from validate import case_config

    _without_bound(monkeypatch)
    checked = 0
    for ds in _ledger()["datasets"]:
        if ds["loading"] != "compression":
            continue
        for case in ds["cases"]:
            res = _run(case_config(ds, case))
            if res.load_path_eccentricity_ratio >= (
                    ECCENTRICITY_REGIME_THRESHOLD):
                continue
            checked += 1
            assert res.analytical_knockdown == pytest.approx(
                float(case["expected_analytical_kd"]), rel=1e-3), case["case"]
    assert checked >= 25


def test_tension_is_untouched(monkeypatch):
    """An axial pull straightens the wave; the bound is compression only."""
    reference = _run(_config(loading="tension"))
    _without_bound(monkeypatch)
    res = _run(_config(loading="tension"))
    assert res.load_path_eccentricity_ratio >= ECCENTRICITY_REGIME_THRESHOLD
    assert res.whole_thickness_wave
    assert not res.bending_dominated and not res.bending_bound_applied
    assert res.analytical_knockdown == reference.analytical_knockdown
    assert res.governing_bound is None
    assert res.kink_band_knockdown is None
    assert res.bending_knockdown is None


# ----------------------------------------------------------------------
# The validation numbers the docs and the caveat quote
# ----------------------------------------------------------------------


def _err(pred, meas):
    return (pred - meas) / meas * 100.0


def test_thor_wave_1_moves_from_plus_52_to_plus_15_percent():
    cfg, case = _ledger_case("thor_2021_qi", "J-QI-1")
    res = _run(cfg)
    meas = float(case["measured_kd"])
    assert _err(res.kink_band_knockdown, meas) == pytest.approx(52.1, abs=0.05)
    assert _err(res.analytical_knockdown, meas) == pytest.approx(15.1, abs=0.05)
    assert "from +52% to +15%" in BENDING_BOUND_CAVEAT


def test_elhajjar_severe_cases_move_within_15_percent():
    """Dataset A (Elhajjar 2025), the head-to-head recipe: the two cases
    where the bound governs, before and after."""
    sys.path.insert(0, str(_ROOT / "validation"))
    import plot_all_validation as pav

    expected = {0.6075: (18.5, 13.1), 0.729: (29.6, 10.4)}
    rows = dict(pav.A_ROWS)
    for amplitude, (before, after) in expected.items():
        cfg = pav.case_config(
            amplitude, pav.elhajjar_wavelength(amplitude),
            material="T700_2510", angles=pav.ELHAJJAR, t_ply=0.152,
            loading="compression")
        res = _run(cfg)
        meas = rows[amplitude]
        assert res.governing_bound == "bending"
        assert _err(res.kink_band_knockdown, meas) == pytest.approx(
            before, abs=0.05)
        assert _err(res.analytical_knockdown, meas) == pytest.approx(
            after, abs=0.05)


# ----------------------------------------------------------------------
# Where the governing bound must reach
# ----------------------------------------------------------------------


def test_summary_names_the_governing_bound_and_the_applied_caveat():
    applied = _run(_config()).summary()
    assert "Governing bound:        bending (kink-band 0.5271" in applied
    assert "WARNING (bending-dominated, bending bound applied)" in applied
    mild = _run(_config(amplitude=0.29, wavelength=14.5)).summary()
    assert "Governing bound:        kink_band" in mild
    assert "WARNING" not in mild
    assert "Governing bound" not in _run(_config(loading="tension")).summary()


def test_results_json_carries_the_bound():
    assert tuple(int(x) for x in SCHEMA_VERSION.split(".")) >= (1, 5)
    data = results_to_dict(_run(_config()))
    analytical = data["analytical"]
    assert analytical["whole_thickness_wave"] is True
    assert analytical["bending_bound_applied"] is True
    assert analytical["governing_bound"] == "bending"
    assert analytical["analytical_knockdown"] == analytical["bending_knockdown"]
    assert analytical["kink_band_knockdown"] == pytest.approx(0.5271, abs=5e-4)
    tension = results_to_dict(_run(_config(loading="tension")))["analytical"]
    assert tension["governing_bound"] is None
    assert tension["bending_knockdown"] is None
    assert tension["kink_band_knockdown"] is None
    json.dumps(data)


def test_caveat_choice():
    assert bending_caveat(False, False) is None
    assert bending_caveat(False, True) is None
    assert bending_caveat(True, False) == BENDING_REGIME_CAVEAT
    assert bending_caveat(True, True) == BENDING_BOUND_CAVEAT
    assert "not applied" in BENDING_REGIME_CAVEAT
    assert "Prefer FE" in BENDING_REGIME_CAVEAT


_DEFECT = {
    "amplitude_mm": 1.19, "wavelength_mm": 27.9, "width_mm": 27.9,
    "morphology": "uniform", "loading": "compression",
    "ply_thickness_mm": 0.131, "n_plies": 40, "layup": _QI,
    "material_name": "IM7_8552",
}


def _ncr(**extra):
    eng = {"analytical_knockdown": 0.399, "damage_index": 0.3,
           "analytical_strength_MPa": 479.0,
           "load_path_eccentricity_ratio": 0.251, **extra}
    return build_analysis_summary(defect=dict(_DEFECT), engineering=eng)


def test_ncr_summary_reports_the_applied_bound():
    summary = _ncr(bending_dominated=True, bending_bound_applied=True,
                   governing_bound="bending", bending_knockdown=0.399,
                   kink_band_knockdown=0.5271)
    ea = summary["engineering_analysis"]
    assert ea["bending_bound_applied"] is True
    assert ea["governing_bound"] == "bending"
    assert ea["bending_regime_caveat"] == BENDING_BOUND_CAVEAT
    assert any("Eccentric-column bending bound" in c and "bending." in c
               for c in summary["criteria_cited"])
    md = render_summary_markdown(summary)
    assert ("- Governing bound: bending (kink-band 0.5271, eccentric-column "
            "bending 0.399)") in md
    assert f"> **Bending-dominated:** {BENDING_BOUND_CAVEAT}" in md


def test_ncr_summary_without_the_bound_keeps_the_fe_advice():
    one_sided = _ncr(bending_dominated=True, governing_bound="kink_band")
    ea = one_sided["engineering_analysis"]
    assert ea["bending_regime_caveat"] == BENDING_REGIME_CAVEAT
    assert ea["bending_knockdown"] is None
    assert not any("Eccentric-column" in c
                   for c in one_sided["criteria_cited"])
    md = render_summary_markdown(one_sided)
    assert "- Governing bound: kink_band\n" in md
    legacy = _ncr()
    assert legacy["engineering_analysis"]["governing_bound"] is None
    assert "Governing bound" not in render_summary_markdown(legacy)


@pytest.mark.viz
def test_app_result_dict_carries_the_bound():
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
    assert result["bending_bound_applied"] is True
    assert result["governing_bound"] == "bending"
    assert result["analytical_knockdown"] == pytest.approx(
        result["bending_knockdown"])
    assert result["kink_band_knockdown"] == pytest.approx(0.5271, abs=5e-4)
