"""Regime-conditional validation margins: the module and its reports (#455).

The fitted table is guarded against the ledger in
``tests/test_validation/test_margins_ledger.py``; this file pins the fat-tail
fit, the regime classification, the lookups, and every place the margin
must reach: the results JSON (schema 1.6) and the NCR summary in all
three formats. The margin is reporting only, so no knockdown may move.
"""

from __future__ import annotations

import json
import math
from importlib import resources
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.margins import (
    COVERAGE,
    MARGIN_DEFINITION,
    MIN_CASES,
    SIGMA_FLOOR,
    UD_DOMINATED_FRACTION,
    MarginRegime,
    MarginResidual,
    analytical_path,
    build_margin_table,
    fit_gaussian_mixture,
    fit_margin,
    laminate_class,
    load_margin_table,
    lookup_margin,
    margin_for_results,
    regime_for,
    zero_ply_fraction,
)
from wrinklefe.core.material import MaterialLibrary
from wrinklefe.core.penetration_gate import GATE_LI2025_VACBAG
from wrinklefe.io.export import (
    build_analysis_summary,
    export_summary,
    render_summary_markdown,
    render_summary_pdf,
)
from wrinklefe.io.results import SCHEMA_VERSION, results_to_dict

_ROOT = Path(__file__).resolve().parents[1]
_QI = [0, 45, -45, 90] * 5
_QI = _QI + _QI[::-1]
_MD_TENSION = [0, 45, 90, -45, 0, 45, -45, 0]
_MD_TENSION = _MD_TENSION + _MD_TENSION[::-1]


# ----------------------------------------------------------------------
# The fat-tail fit
# ----------------------------------------------------------------------


def _two_modes(n=4000, seed=7):
    rng = np.random.default_rng(seed)
    k = int(0.3 * n)
    return np.concatenate([rng.normal(-0.4, 0.08, k),
                           rng.normal(0.1, 0.05, n - k)])


def test_mixture_recovers_a_known_two_component_sample():
    fit = fit_gaussian_mixture(_two_modes(), sigma_floor=1e-3)
    assert fit.weights == pytest.approx((0.3, 0.7), abs=0.03)
    assert fit.means == pytest.approx((-0.4, 0.1), abs=0.02)
    assert fit.sds == pytest.approx((0.08, 0.05), abs=0.01)


def test_mixture_fit_is_deterministic_and_order_free():
    y = _two_modes(n=40, seed=3)
    a = fit_gaussian_mixture(y)
    b = fit_gaussian_mixture(y[::-1].copy())
    assert a == b
    assert fit_gaussian_mixture(y) == a


def test_quantile_inverts_the_cdf():
    fit = fit_gaussian_mixture(_two_modes(n=60, seed=1))
    for p in (0.01, 0.05, 0.5, 0.95):
        assert fit.cdf(fit.quantile(p)) == pytest.approx(p, abs=1e-9)
    with pytest.raises(ValueError):
        fit.quantile(1.0)


def test_sigma_floor_holds_a_component_off_a_single_point():
    fit = fit_gaussian_mixture([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    assert min(fit.sds) == pytest.approx(SIGMA_FLOOR)


def test_mixture_sees_a_fat_tail_a_normal_misses():
    """The reason for the fat-tail model: 10% of residuals far on the
    unsafe side. A normal spreads them into a symmetric bell and puts the
    5% quantile well short of them; the mixture puts it among them."""
    rng = np.random.default_rng(11)
    y = np.concatenate([rng.normal(0.0, 0.05, 180), rng.normal(-0.5, 0.05, 20)])
    mixture_q = fit_gaussian_mixture(y).quantile(0.05)
    normal_q = y.mean() + stats.norm.ppf(0.05) * y.std(ddof=1)
    empirical_q = np.quantile(y, 0.05)
    assert mixture_q < normal_q - 0.1
    assert mixture_q == pytest.approx(empirical_q, abs=0.05)


@pytest.mark.parametrize("bad", [[0.1, 0.2, 0.3], [0.0, 1.0, math.nan, 2.0]])
def test_mixture_rejects_bad_samples(bad):
    with pytest.raises(ValueError):
        fit_gaussian_mixture(bad)


# ----------------------------------------------------------------------
# One regime's margin
# ----------------------------------------------------------------------


def test_too_few_cases_report_insufficient_evidence():
    entry = fit_margin([0.9] * (MIN_CASES - 1))
    assert entry["status"] == "insufficient_evidence"
    assert entry["margin_factor"] is None
    assert entry["ratio_quantile"] is None
    assert entry["n"] == MIN_CASES - 1
    assert entry["min_ratio"] == 0.9


def test_margin_is_the_mixture_quantile_of_the_ratio():
    rng = np.random.default_rng(5)
    ratios = np.exp(rng.normal(-0.1, 0.1, 30))
    entry = fit_margin(ratios)
    assert entry["status"] == "fitted"
    mix = fit_gaussian_mixture(np.log(ratios))
    expected = math.exp(mix.quantile(1.0 - COVERAGE))
    assert entry["ratio_quantile"] == pytest.approx(expected, abs=1e-4)
    assert entry["margin_factor"] == pytest.approx(min(1.0, expected),
                                                   abs=1e-4)
    assert entry["n_below_margin"] == int(np.sum(ratios < entry["margin_factor"]))


def test_a_margin_never_raises_a_knockdown():
    """Conservative predictions (every ratio > 1) earn no credit."""
    entry = fit_margin(np.linspace(1.5, 3.0, 12))
    assert entry["ratio_quantile"] > 1.0
    assert entry["margin_factor"] == 1.0


@pytest.mark.parametrize("bad", [[], [1.0, -0.5], [1.0, math.inf]])
def test_margin_rejects_bad_ratios(bad):
    with pytest.raises(ValueError):
        fit_margin(bad)


# ----------------------------------------------------------------------
# Regimes
# ----------------------------------------------------------------------


def test_laminate_class():
    assert laminate_class([0] * 16) == "ud"
    assert laminate_class([180.0, 0.0, -2.0]) == "ud"
    # Pilato (2022): two woven 90-degree plies in 18.
    assert laminate_class([0] * 9 + [90] + [0] * 5 + [90] + [0] * 2) == "ud"
    assert laminate_class(_QI) == "md"
    assert laminate_class(_MD_TENSION) == "md"
    # The threshold is inclusive.
    assert zero_ply_fraction([0, 0, 0, 90]) == UD_DOMINATED_FRACTION
    assert laminate_class([0, 0, 0, 90]) == "ud"
    assert laminate_class([0, 0, 90, 90]) == "md"
    with pytest.raises(ValueError):
        zero_ply_fraction([])


def test_regime_for_drops_the_bending_flag_in_tension():
    r = regime_for("three_mechanism", "Tension ", _QI, True)
    assert r == MarginRegime("three_mechanism", "tension", "md", False)
    c = regime_for("kink_band", "compression", [0] * 8, True)
    assert c.key == "kink_band/compression/ud/bending"
    assert "bending-dominated" in c.label and "UD-dominated" in c.label


@pytest.mark.parametrize("kwargs", [
    {"path": "magic"}, {"loading": "shear"}, {"laminate": "woven"},
    {"loading": "tension", "bending_dominated": True},
])
def test_regime_rejects_unknown_cells(kwargs):
    base = {"path": "kink_band", "loading": "compression", "laminate": "ud",
            "bending_dominated": False}
    with pytest.raises(ValueError):
        MarginRegime(**{**base, **kwargs})


def _config(**kw):
    base = dict(amplitude=0.3, wavelength=10.0, width=10.0,
                morphology="graded", angles=[0] * 16, ply_thickness=0.25,
                material=MaterialLibrary().get("IM7_8552"))
    base.update(kw)
    return AnalysisConfig(**base)


def test_analytical_path_follows_the_model_that_ran():
    assert analytical_path(_config()) == "kink_band"
    assert analytical_path(_config(loading="tension")) == "three_mechanism"
    assert analytical_path(
        _config(penetration_gate=GATE_LI2025_VACBAG)) == "penetration_gate"


# ----------------------------------------------------------------------
# The shipped table and lookups
# ----------------------------------------------------------------------


def test_the_table_ships_as_package_data():
    path = resources.files("wrinklefe.core").joinpath("validation_margins.json")
    assert path.is_file()
    assert '"wrinklefe.core" = ["validation_margins.json"]' in (
        _ROOT / "pyproject.toml").read_text()
    table = load_margin_table()
    assert table["coverage"] == COVERAGE
    assert table["min_cases"] == MIN_CASES
    # A fresh copy each time, so a caller cannot corrupt the cache.
    table["regimes"].clear()
    assert load_margin_table()["regimes"]


def test_a_regime_without_cases_is_insufficient_with_n_zero():
    regime = MarginRegime("fe_retention", "compression", "md", False)
    margin = lookup_margin(regime)
    assert margin.n == 0
    assert margin.status == "insufficient_evidence"
    assert not margin.fitted
    assert margin.apply(0.8) is None
    assert "insufficient evidence (0 validation cases" in margin.describe()


def test_lookup_reads_a_fitted_entry():
    residuals = [
        MarginResidual(MarginRegime("kink_band", "compression", "md", False),
                       r, "ds", f"c{i}")
        for i, r in enumerate(np.linspace(0.8, 1.2, MIN_CASES))
    ]
    table = build_margin_table(residuals, source="test")
    margin = lookup_margin(residuals[0].regime, table)
    assert margin.fitted and margin.n == MIN_CASES
    assert margin.datasets == ("ds",)
    assert margin.apply(0.5) == pytest.approx(0.5 * margin.margin_factor)
    d = margin.to_dict(knockdown=0.5)
    assert d["margined_knockdown"] == pytest.approx(
        0.5 * margin.margin_factor, abs=1e-6)
    assert d["definition"] == MARGIN_DEFINITION
    assert d["regime"]["key"] == "kink_band/compression/md/non-bending"


def test_definition_states_the_coverage_and_the_basis_disclaimer():
    assert f"{COVERAGE:.0%}" in MARGIN_DEFINITION
    assert "not an A- or B-basis value" in MARGIN_DEFINITION
    assert "changes no number" in MARGIN_DEFINITION


# ----------------------------------------------------------------------
# Where the margin must reach
# ----------------------------------------------------------------------


def _tension_md():
    """Elhajjar & Shams (2016) multidirectional tension: a fitted regime."""
    return WrinkleAnalysis(_config(
        amplitude=0.3, wavelength=4.0, width=4.0, loading="tension",
        angles=_MD_TENSION, ply_thickness=0.152,
        material=MaterialLibrary().get("T700GC_2510_AGATE"),
    )).run(analytical_only=True)


def _thor_wave_1():
    """Thor (2021) wave 1: quasi-isotropic, bending-dominated."""
    return WrinkleAnalysis(_config(
        amplitude=1.19, wavelength=27.9, width=27.9, morphology="uniform",
        angles=_QI, ply_thickness=0.131,
    )).run(analytical_only=True)


def test_the_margin_changes_no_knockdown():
    res = _tension_md()
    kd = res.analytical_knockdown
    margin = margin_for_results(res)
    results_to_dict(res)
    assert res.analytical_knockdown == kd
    assert margin.fitted
    assert margin.regime.key == "three_mechanism/tension/md/non-bending"
    assert margin.margin_factor < 1.0


def test_results_json_carries_the_margin_block():
    assert tuple(int(x) for x in SCHEMA_VERSION.split(".")) >= (1, 6)
    res = _tension_md()
    block = results_to_dict(res)["validation_margin"]
    assert block["applies_to"] == "analytical_knockdown"
    assert block["status"] == "fitted"
    assert block["n"] >= MIN_CASES
    assert block["knockdown"] == pytest.approx(res.analytical_knockdown)
    assert block["margined_knockdown"] == pytest.approx(
        res.analytical_knockdown * block["margin_factor"], abs=1e-6)
    assert block["regime"]["path"] == "three_mechanism"
    json.dumps(block)


def test_results_json_reports_insufficient_evidence_with_its_n():
    res = _thor_wave_1()
    assert res.bending_dominated
    block = results_to_dict(res)["validation_margin"]
    assert block["regime"]["key"] == "kink_band/compression/md/bending"
    assert block["status"] == "insufficient_evidence"
    assert 0 < block["n"] < MIN_CASES
    assert block["margin_factor"] is None
    assert block["margined_knockdown"] is None


def test_results_json_reports_a_multidirectional_compression_margin():
    """Issue #462: Thor (2021) wave 2, quasi-isotropic and not
    bending-dominated, the commonest regime in practice, now has enough
    validation cases (Datasets A, C, D in the ledger) for a margin."""
    res = WrinkleAnalysis(_config(
        amplitude=0.29, wavelength=14.5, width=14.5, morphology="uniform",
        angles=_QI, ply_thickness=0.131,
    )).run(analytical_only=True)
    assert not res.bending_dominated
    block = results_to_dict(res)["validation_margin"]
    assert block["regime"]["key"] == "kink_band/compression/md/non-bending"
    assert block["status"] == "fitted"
    assert block["n"] >= MIN_CASES
    assert 0.0 < block["margin_factor"] <= 1.0


def test_results_json_follows_the_gate_path():
    res = WrinkleAnalysis(_config(
        amplitude=0.75, wavelength=12.9, width=12.9, angles=[0] * 14,
        ply_thickness=0.44, material=MaterialLibrary().get("AC318_S6C10"),
        penetration_gate=GATE_LI2025_VACBAG,
    )).run(analytical_only=True)
    block = results_to_dict(res)["validation_margin"]
    assert block["regime"]["key"] == (
        "penetration_gate/compression/ud/non-bending")


_DEFECT = {
    "amplitude_mm": 0.3, "wavelength_mm": 4.0, "width_mm": 4.0,
    "morphology": "graded", "loading": "tension", "ply_thickness_mm": 0.152,
    "n_plies": 16, "layup": _MD_TENSION, "material_name": "T700GC_2510_AGATE",
}


def _ncr(defect=None, **engineering):
    eng = {"analytical_knockdown": 0.6, "damage_index": 0.1,
           "analytical_strength_MPa": 900.0, **engineering}
    return build_analysis_summary(defect=dict(defect or _DEFECT),
                                  engineering=eng)


def test_ncr_summary_carries_the_margin_and_its_regime():
    summary = _ncr(analytical_path="three_mechanism")
    block = summary["engineering_analysis"]["validation_margin"]
    expected = lookup_margin(
        MarginRegime("three_mechanism", "tension", "md", False))
    assert block["margin_factor"] == expected.margin_factor
    assert block["n"] == expected.n
    assert block["margined_knockdown"] == pytest.approx(
        0.6 * expected.margin_factor, abs=1e-6)
    # Reporting only: the disposition reads the unmargined knockdown.
    plain = _ncr(analytical_path="three_mechanism")
    assert (summary["disposition_recommendation"]
            == plain["disposition_recommendation"])
    assert summary["engineering_analysis"]["analytical_knockdown"] == 0.6

    md = render_summary_markdown(summary)
    assert f"- Validation margin: **{expected.margin_factor:.4g}**" in md
    assert f"(n = {expected.n})" in md
    assert expected.regime.label in md
    assert f"> **Validation margin:** {MARGIN_DEFINITION}" in md
    assert any("Regime-conditional validation margin" in c
               for c in summary["criteria_cited"])


def test_ncr_summary_infers_the_path_from_the_loading():
    assert (_ncr()["engineering_analysis"]["validation_margin"]["regime"]
            ["path"]) == "three_mechanism"
    gate = _ncr(defect={**_DEFECT, "loading": "compression",
                        "layup": [0] * 14},
                analytical_path="penetration_gate")
    assert (gate["engineering_analysis"]["validation_margin"]["regime"]["key"]
            == "penetration_gate/compression/ud/non-bending")


def test_ncr_summary_reports_insufficient_evidence():
    summary = _ncr(defect={**_DEFECT, "loading": "compression",
                           "layup": _QI},
                   bending_dominated=True)
    block = summary["engineering_analysis"]["validation_margin"]
    assert block["status"] == "insufficient_evidence"
    md = render_summary_markdown(summary)
    assert "- Validation margin: **insufficient evidence**" in md
    assert f"at least {MIN_CASES} needed" in md
    assert not any("Regime-conditional validation margin" in c
                   for c in summary["criteria_cited"])


def test_ncr_summary_without_a_layup_says_so():
    summary = _ncr(defect={**_DEFECT, "layup": None})
    assert summary["engineering_analysis"]["validation_margin"] is None
    md = render_summary_markdown(summary)
    assert "- Validation margin: not determined (no layup supplied)" in md


def test_ncr_summary_accepts_a_layup_string():
    summary = _ncr(defect={**_DEFECT, "loading": "compression",
                           "layup": "[0]_16"})
    assert (summary["engineering_analysis"]["validation_margin"]["regime"]
            ["key"]) == "kink_band/compression/ud/non-bending"


def test_ncr_summary_built_before_the_margin_renders_without_it():
    summary = _ncr()
    del summary["engineering_analysis"]["validation_margin"]
    assert "Validation margin" not in render_summary_markdown(summary)


def test_every_ncr_format_carries_the_margin(tmp_path):
    summary = _ncr(analytical_path="three_mechanism")
    export_summary(summary, tmp_path / "s.json", fmt="json")
    export_summary(summary, tmp_path / "s.md", fmt="md")
    data = json.loads((tmp_path / "s.json").read_text())
    assert data["engineering_analysis"]["validation_margin"]["status"] == (
        "fitted")
    assert "Validation margin" in (tmp_path / "s.md").read_text()
    # The PDF is laid out from the Markdown, line by line.
    pdf = render_summary_pdf(summary)
    assert pdf.startswith(b"%PDF")
