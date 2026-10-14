"""Dataset M, Elhajjar & Shams (2016): tension onset and ultimate (#436).

Thirteen press-claved T700GC/2510 tension coupons with a rod-made
one-sided wrinkle, each with a measured damage onset (the DIC limit
point) and a measured ultimate. These tests pin the transcription and
every claim ``docs/internal/VALIDATION.md`` ("Dataset M") and
``docs/interpreting_results.md`` make about it.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_LEDGER = json.loads(
    (_ROOT / "tests" / "test_validation" / "ledger.json").read_text())
_HALVES = {d["name"]: d for d in _LEDGER["datasets"]
           if d["name"].startswith("elhajjar_shams_2016")}

#: The paper's Table 2 "ratio of limit to ultimate stress (control)"
#: column, specimens 1-13, as printed (two decimals).
_TABLE2_RATIO = [0.11, 0.14, 0.28, 0.25, 0.27, 0.24, 0.30, 0.29, 0.22,
                 0.24, 0.23, 0.14, 0.14]


def _cases():
    return [c for name in sorted(_HALVES, key=lambda n: "_ud_" in n)
            for c in _HALVES[name]["cases"]]


def _err(pred, meas):
    return (pred - meas) / meas * 100.0


def test_two_halves_thirteen_specimens():
    assert len(_HALVES) == 2
    assert [len(d["cases"]) for d in _HALVES.values()] == [9, 4]


def test_onset_ratios_reproduce_the_papers_table_2():
    """The onset knockdown is the limit point over the control ultimate;
    with the control ultimates in the ledger it reproduces the paper's
    own ratio column, which is what pins those control values."""
    got = [round(c["measured_onset_kd"] + 1e-9, 2) for c in _cases()]
    assert got == _TABLE2_RATIO


def test_measured_angles_are_matched_where_the_paper_gives_them():
    import math

    for c in _cases()[:5]:  # specimens 1-5 (Fig. 2: 30 and 23 deg)
        a = c["amplitude_p2p_mm"] / 2.0
        theta = math.degrees(math.atan(2 * math.pi * a / c["wavelength_mm"]))
        assert theta == pytest.approx(c["peak_angle_deg"], abs=0.05)
        assert c["peak_angle_deg"] in (30.0, 23.0)


def test_only_the_preloaded_specimen_is_left_out_of_the_onset_score():
    excluded = [c["case"] for c in _cases() if c.get("onset_excluded")]
    assert excluded == ["M-UD-12"]


def test_analytical_three_mechanism_on_the_steep_coupons():
    """Against the ultimate (issue #454): signed mean -4.9% on the nine
    multidirectional coupons (23-36 deg), 3 unsafe, worst +14.2%; -2.8%
    on the four UD, 2 unsafe, worst +17.6%. Before #454 the compression
    floor set every one of the 13 predictions and was unsafe on 8 of 9
    (worst +29.6%) and 2 of 4 (worst +39.2%)."""
    md = next(d for n, d in _HALVES.items() if "_md_" in n)["cases"]
    ud = next(d for n, d in _HALVES.items() if "_ud_" in n)["cases"]
    md_err = [_err(c["expected_analytical_kd"], c["measured_kd"]) for c in md]
    ud_err = [_err(c["expected_analytical_kd"], c["measured_kd"]) for c in ud]
    assert min(c["peak_angle_deg"] for c in md) >= 23.0
    assert sum(e > 0.5 for e in md_err) == 3
    assert 13.0 < max(md_err) < 16.0
    assert -6.0 < statistics.mean(md_err) < -4.0
    assert all(-23.0 < e < 16.0 for e in md_err)
    assert sum(e > 0.5 for e in ud_err) == 2
    assert 16.0 < max(ud_err) < 19.0
    assert -4.0 < statistics.mean(ud_err) < -2.0
    assert all(-24.0 < e < 19.0 for e in ud_err)


def test_analytical_tension_is_not_held_up_by_the_compression_floor():
    """Issue #454: the tension KD used to be floored by the compression
    (kink-band) KD, which set all 13 Dataset M predictions. Each
    prediction is now the tension model's own and sits below the
    compression KD the floor would have imposed."""
    import dataclasses

    sys.path.insert(0, str(_ROOT / "scripts"))
    from validate import case_config

    from wrinklefe.analysis import WrinkleAnalysis

    for name, ds in _HALVES.items():
        for case in ds["cases"]:
            cfg = case_config(ds, case)
            tension = WrinkleAnalysis(cfg).run(analytical_only=True)
            comp_cfg = dataclasses.replace(
                cfg, loading="compression",
                applied_strain=-abs(cfg.applied_strain))
            comp = WrinkleAnalysis(comp_cfg).run(analytical_only=True)
            assert tension.analytical_knockdown < comp.analytical_knockdown
            assert "capped" not in tension.tension_mechanisms["mode"]


@pytest.fixture(scope="module")
def fe_onset():
    sys.path.insert(0, str(_ROOT / "validation"))
    sys.path.insert(0, str(_ROOT / "scripts"))
    import strength_error_summary as ses

    return ses.fe_onset_elhajjar_shams(), ses._m_onset_scored()


@pytest.mark.slow
def test_fe_first_ply_onset_errs_both_ways(fe_onset):
    """Thirteen FE solves. First-ply FE against the measured onset: unsafe
    on 9 of the 12 scored coupons, from about -22% to about +154%, worst
    on the largest wrinkles; the docs and the FE caveat quote this."""
    rows, scored = fe_onset
    errs = {k: _err(fe, meas) for k, (fe, meas, _kd) in rows.items()
            if k in scored}
    assert len(errs) == 12
    assert sum(e > 0.5 for e in errs.values()) == 9
    assert -25.0 < min(errs.values()) < -18.0
    assert 140.0 < max(errs.values()) < 170.0
    assert 20.0 < statistics.median(errs.values()) < 35.0
    # Within -25 .. +50% except the two largest-amplitude groups.
    small = [e for k, e in errs.items()
             if k not in ("M-MD-1", "M-MD-2", "M-UD-13")]
    assert all(-25.0 < e < 50.0 for e in small)
    from wrinklefe.io.export import FE_STRENGTH_CAVEAT

    assert "unsafe on 9 of 12, by up to about 150%" in FE_STRENGTH_CAVEAT


@pytest.mark.slow
def test_fe_retention_is_conservative_against_the_ultimate(fe_onset):
    """First-ply retention scored against the fibre-governed ultimate is
    conservative on every coupon, as for Dataset I."""
    rows, _scored = fe_onset
    meas = {c["case"]: c["measured_kd"] for c in _cases()}
    errs = [_err(kd, meas[k]) for k, (_fe, _m, kd) in rows.items()]
    assert all(e < -20.0 for e in errs)
