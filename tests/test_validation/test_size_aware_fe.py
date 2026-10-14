"""The size-aware FE study of issue #453 stays what the ledger pins.

Two candidate FE strength measures for embedded flat-surface wrinkles --
the progressive-damage peak load and the load at cohesive-zone (CZM)
delamination onset -- were scored on Datasets C, D, E, L-WM and M against
the first-ply FE they were meant to replace. Neither reached the issue's
bar (a worst case of at most +15 % on every set), so the docs and
``FE_STRENGTH_CAVEAT`` carry an explicit scope exclusion with the measured
numbers. ``validation/size_aware_fe_study.py`` regenerates the study; its
``--ledger`` output is the ``size_aware_fe`` section of ``ledger.json``.

Fast tests keep the exclusion's quoted numbers equal to the pinned ones.
The slow test re-solves a cheap subset through the driver so the pinned
values cannot drift silently: the CZM initiation load (one elastic
increment, seconds per case) on one case of every dataset, and the
progressive peak on one UD coupon and its pristine baseline.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
_LEDGER = json.loads((_HERE / "ledger.json").read_text())
_STUDY = _LEDGER["size_aware_fe"]
_DATASETS = _STUDY["datasets"]

_MEASURES = ("first_ply", "progressive_peak", "czm_initiation")


def _err(pred: float, meas: float) -> float:
    return (pred - meas) / meas * 100.0


def _errors(ds: str, measure: str) -> list[float]:
    return [_err(r[measure], r["measured"]) for r in _DATASETS[ds]["cases"]]


def _worst(ds: str, measure: str) -> float:
    return max(_errors(ds, measure))


def _unsafe(ds: str, measure: str) -> int:
    return sum(e > 0.5 for e in _errors(ds, measure))


# ----------------------------------------------------------------------
# The pinned section is complete and self-consistent
# ----------------------------------------------------------------------

def test_every_scored_case_is_pinned_for_every_measure():
    assert {ds: len(v["cases"]) for ds, v in _DATASETS.items()} == {
        "C": 3, "D": 4, "E": 9, "L-WM": 6, "M": 12,
    }
    for ds, block in _DATASETS.items():
        for row in block["cases"]:
            for m in _MEASURES:
                assert isinstance(row[m], float), (ds, row["case"], m)
            assert isinstance(row["progressive_converged"], bool)
        assert block["absolute_MPa"] == (ds == "M")


def test_knockdowns_are_the_pinned_stresses_over_the_pristine_peak():
    """A knockdown row must be its own MPa values over the dataset's
    pristine progressive peak (CZM initiation capped at 1)."""
    for ds, block in _DATASETS.items():
        if block["absolute_MPa"]:
            continue
        sp = block["pristine_progressive_MPa"]
        for row in block["cases"]:
            assert row["progressive_peak"] == pytest.approx(
                row["progressive_peak_MPa"] / sp, abs=2e-4)
            assert row["czm_initiation"] == pytest.approx(
                min(row["czm_initiation_MPa"], sp) / sp, abs=2e-4)


def test_dataset_m_rows_are_absolute_onset_stresses():
    for row in _DATASETS["M"]["cases"]:
        assert row["measured"] > 50.0  # MPa, not a knockdown
        assert row["czm_initiation"] == row["czm_initiation_MPa"]
        assert row["progressive_peak"] == row["progressive_peak_MPa"]


def test_the_study_scores_the_same_cases_as_the_head_to_head():
    """Same measurements as ``head_to_head_cases.csv`` (issue #433) for
    C, D and E, so the two tables can be read side by side."""
    import csv

    rows = list(csv.DictReader(
        (_REPO / "validation" / "head_to_head_cases.csv").open()))
    h2h = {(r["dataset"], r["case"]): float(r["measured"]) for r in rows}
    for ds, label in (("C", "C Mukhopadhyay comp"), ("D", "D Wang conc/conv"),
                      ("E", "E Li2024 UD comp")):
        for row in _DATASETS[ds]["cases"]:
            assert h2h[(label, row["case"])] == pytest.approx(row["measured"])


# ----------------------------------------------------------------------
# The exclusion's numbers
# ----------------------------------------------------------------------

def test_neither_measure_meets_the_fifteen_percent_bar():
    for m in ("progressive_peak", "czm_initiation"):
        worst = max(_worst(ds, m) for ds in _DATASETS)
        assert worst > 15.0, (m, worst)


def test_the_worst_cases_sit_where_the_exclusion_says():
    # Multidirectional compression (C, D): both measures as unsafe as, or
    # worse than, the first-ply FE they would replace.
    assert 100.0 <= max(_worst(ds, "progressive_peak") for ds in "CD") <= 120.0
    assert 130.0 <= max(_worst(ds, "czm_initiation") for ds in "CD") <= 150.0
    for ds in "CD":
        for m in _MEASURES:
            assert _unsafe(ds, m) == len(_DATASETS[ds]["cases"]), (ds, m)
    # Wet lay-up (quarantined) L-WM: every case unsafe under every measure.
    assert 55.0 <= _worst("L-WM", "progressive_peak") <= 75.0
    assert 65.0 <= _worst("L-WM", "czm_initiation") <= 85.0
    assert all(_unsafe("L-WM", m) == 6 for m in _MEASURES)
    # Tension damage onset (M): a peak load is not an onset (every coupon
    # unsafe); the CZM onset is conservative on average but still unsafe
    # on the deepest dip.
    assert 190.0 <= _worst("M", "progressive_peak") <= 215.0
    assert _unsafe("M", "progressive_peak") == 12
    assert 45.0 <= _worst("M", "czm_initiation") <= 60.0
    assert _unsafe("M", "czm_initiation") == 2
    assert sum(_errors("M", "czm_initiation")) / 12 < -15.0
    # UD glass (E): the one set where the peak load stays within the bar.
    assert _worst("E", "progressive_peak") <= 15.0
    assert _unsafe("E", "progressive_peak") <= 3
    assert _worst("E", "czm_initiation") > 15.0


def test_the_peak_load_does_not_respond_to_depth_on_dataset_e():
    """On E's fixed-angle cases (15.8-16.7 deg) the measured knockdown
    falls from 0.76 to 0.52 with depth; every FE measure moves by less
    than a fifth of that, and the peak load goes the wrong way."""
    rows = [r for r in _DATASETS["E"]["cases"] if 15.5 <= r["theta_deg"] <= 17.0]
    assert len(rows) == 6
    rows.sort(key=lambda r: r["size"])
    meas_spread = rows[0]["measured"] - rows[-1]["measured"]
    assert meas_spread > 0.2
    for m in _MEASURES:
        vals = [r[m] for r in rows]
        assert max(vals) - min(vals) < 0.2 * meas_spread, m
    assert rows[-1]["progressive_peak"] >= rows[0]["progressive_peak"]


def test_czm_initiation_cannot_see_the_wrinkle_on_multidirectional_coupons():
    """On C and D the interface reaches its strength at the free edges of
    the flat coupon, at a load the pristine coupon would not survive, so
    the capped knockdown is 1.0 on every case."""
    for ds in "CD":
        block = _DATASETS[ds]
        assert all(r["czm_initiation"] == 1.0 for r in block["cases"])
        assert block["near_flat_czm_initiation_MPa"] > block[
            "pristine_progressive_MPa"]
    # On the UD sets the flat coupon never initiates; on M the wrinkle
    # governs (the flat coupon initiates above every wrinkled value).
    for ds in ("E", "L-WM"):
        assert _DATASETS[ds]["near_flat_czm_initiation_MPa"] > 1.0e4
    assert _DATASETS["M"]["near_flat_czm_initiation_MPa"] > max(
        r["czm_initiation_MPa"] for r in _DATASETS["M"]["cases"])


def test_the_caveat_quotes_the_pinned_numbers():
    """The exclusion quotes six worst cases ("about 110% and 140%", ...)
    in a fixed order, then the bar and Dataset E's worst."""
    from wrinklefe.io.export import FE_STRENGTH_CAVEAT

    tail = FE_STRENGTH_CAVEAT.split("no FE strength path is offered")[1]
    numbers = [int(m) for m in re.findall(r"(\d+)%", tail)]
    assert len(numbers) == 8, numbers
    quoted = numbers[:6]
    expected = [
        max(_worst(ds, "progressive_peak") for ds in "CD"),
        max(_worst(ds, "czm_initiation") for ds in "CD"),
        _worst("L-WM", "progressive_peak"),
        _worst("L-WM", "czm_initiation"),
        _worst("M", "progressive_peak"),
        _worst("M", "czm_initiation"),
    ]
    for q, e in zip(quoted, expected, strict=True):
        assert abs(q - e) <= 6.0, (q, e)
    assert numbers[6] == _STUDY["bar_worst_error_pct"]
    assert numbers[7] == round(_worst("E", "progressive_peak"))
    assert f"(worst +{numbers[7]}%)" in tail
    assert "Use the analytical knockdown there." in FE_STRENGTH_CAVEAT


# ----------------------------------------------------------------------
# Re-solve a cheap subset through the driver
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def driver():
    sys.path.insert(0, str(_REPO / "validation"))
    sys.path.insert(0, str(_REPO / "scripts"))
    import size_aware_fe_study as module

    return module


def _row(ds: str, case: str) -> dict:
    return next(r for r in _DATASETS[ds]["cases"] if r["case"] == case)


@pytest.mark.slow
@pytest.mark.parametrize(("ds", "case"), [
    ("C", "A=0.492"), ("D", "concave A=0.76"), ("E", "E-7"),
    ("L-WM", "L-WM-2-3"), ("M", "M-MD-1"),
])
def test_czm_initiation_load_matches_the_pinned_value(driver, ds, case,
                                                       tmp_path):
    """One elastic CZM increment per case (seconds): the initiation load
    is exact, so the tolerance only absorbs BLAS noise."""
    c = next(c for c in driver.cases(ds) if c.case == case)
    cache = driver.Cache(tmp_path / "cache.json")
    driver.solve_init(c, cache)
    got = cache.get(driver._init_key(c))["sigma_MPa"]
    pinned = _row(ds, case)["czm_initiation_MPa"]
    assert got == pytest.approx(pinned, rel=_STUDY["rel_tolerance"])


@pytest.mark.slow
def test_progressive_peak_matches_the_pinned_value_on_a_ud_coupon(driver,
                                                                   tmp_path):
    """The wrinkled L-WM-4-3 coupon and the flat YC300 baseline (both
    converged runs, about a minute together)."""
    c = next(c for c in driver.cases("L-WM") if c.case == "L-WM-4-3")
    cache = driver.Cache(tmp_path / "cache.json")
    driver.solve_first_and_prog(c, cache, prog=True)
    block = _DATASETS["L-WM"]
    tol = _STUDY["rel_tolerance"]
    prist = cache.get(f"pristine|{c.pristine}")
    assert prist["converged"] is True
    assert prist["peak_MPa"] == pytest.approx(
        block["pristine_progressive_MPa"], rel=tol)
    prog = cache.get(f"prog|L-WM|{c.geometry}")
    assert prog["peak_MPa"] == pytest.approx(
        _row("L-WM", "L-WM-4-3")["progressive_peak_MPa"], rel=tol)
    kd = driver.predictions(c, cache)["(a) prog. peak"]
    assert kd == pytest.approx(_row("L-WM", "L-WM-4-3")["progressive_peak"],
                               rel=tol)
