"""The shipped validation margins stay in step with the ledger (issue #455).

The margins the NCR summary and the results JSON report are fitted from
the validation ledger by ``scripts/fit_margins.py`` and shipped as package
data, because the ledger itself is not installed. These tests re-fit them
from the ledger and fail when the shipped table is stale, the same way
``test_ledger.py`` guards the pinned knockdowns: after a deliberate model
change, ``python scripts/validate.py --update`` re-pins both.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from wrinklefe.core.margins import (
    MIN_CASES,
    STATUS_FITTED,
    STATUS_INSUFFICIENT,
    load_margin_table,
)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_LEDGER = json.loads((_HERE / "ledger.json").read_text())


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, _ROOT / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fit_margins = _load("fit_margins")
validate = _load("validate")


@pytest.fixture(scope="module")
def regenerated() -> dict:
    return fit_margins.build_table()


def test_shipped_margins_match_the_ledger(regenerated):
    """The drift guard: the shipped table is what the ledger implies."""
    diffs = fit_margins.check(regenerated)
    assert not diffs, (
        "The shipped validation margins (src/wrinklefe/core/"
        "validation_margins.json) are stale against the validation ledger:\n  "
        + "\n  ".join(diffs[:20])
        + "\nRe-run 'python scripts/fit_margins.py --write' (or "
        "'python scripts/validate.py --update' after a model change) and "
        "commit the regenerated table."
    )


def test_the_installed_table_is_the_file_the_generator_writes():
    shipped = json.loads(fit_margins.TABLE.read_text())
    assert load_margin_table() == shipped


def test_quarantined_datasets_are_left_out(regenerated):
    quarantined = sorted(d["name"] for d in _LEDGER["datasets"]
                         if "quarantine" in d)
    assert quarantined, "the ledger is expected to quarantine Dataset L"
    assert regenerated["excluded_datasets"] == quarantined
    for entry in regenerated["regimes"].values():
        assert not set(entry["datasets"]) & set(quarantined)


def test_every_non_quarantined_prediction_is_counted_once(regenerated):
    expected = 0
    for ds in _LEDGER["datasets"]:
        if "quarantine" in ds:
            continue
        for case in ds["cases"]:
            if case.get("measured_kd") is None:
                continue
            expected += 1 + ("expected_gate_kd" in case)
    calibration = [r for r in _LEDGER["progressive_damage"]["recipes"]
                   if r["role"].startswith("calibration")]
    expected += sum(len(r["cases"]) for r in calibration)
    counted = sum(e["n"] for e in regenerated["regimes"].values())
    assert counted == expected
    for entry in regenerated["regimes"].values():
        assert entry["n"] == len(entry["cases"])


def test_small_regimes_report_insufficient_evidence(regenerated):
    assert regenerated["min_cases"] == MIN_CASES
    for key, entry in regenerated["regimes"].items():
        if entry["n"] < MIN_CASES:
            assert entry["status"] == STATUS_INSUFFICIENT, key
            assert entry["margin_factor"] is None, key
        else:
            assert entry["status"] == STATUS_FITTED, key
            assert 0.0 < entry["margin_factor"] <= 1.0, key


def test_n_below_margin_counts_the_cases_a_margin_leaves_unsafe(regenerated):
    """The in-sample check the table reports next to each margin: cases
    still non-conservative after it (a fat-tail fit to a small sample
    can extrapolate either side of the smallest ratio)."""
    for key, entry in regenerated["regimes"].items():
        if entry["status"] != STATUS_FITTED:
            assert entry["n_below_margin"] is None, key
            continue
        below = sum(c["ratio"] < entry["margin_factor"]
                    for c in entry["cases"])
        assert below == entry["n_below_margin"], key
        assert entry["min_ratio"] == min(c["ratio"] for c in entry["cases"])


def _regime_of(table: dict, dataset_prefix: str, case: str) -> str:
    hits = [key for key, e in table["regimes"].items()
            for c in e["cases"]
            if c["dataset"].startswith(dataset_prefix) and c["case"] == case
            and not key.startswith(("penetration_gate", "progressive"))]
    assert len(hits) == 1, (dataset_prefix, case, hits)
    return hits[0]


@pytest.mark.parametrize(("prefix", "case", "key"), [
    # Whole-thickness waves: Thor wave 1 bends, wave 2 kinks (issue #439).
    ("thor_2021_qi", "J-QI-1", "kink_band/compression/md/bending"),
    ("thor_2021_qi", "J-QI-2", "kink_band/compression/md/non-bending"),
    # Shi's one-sided trough, measured on its own geometry through the
    # ledger's eccentricity_recipe: e/t 0.06 / 0.12 / 0.17.
    ("shi_2025_md", "H-MD-10", "kink_band/compression/md/non-bending"),
    ("shi_2025_md", "H-MD-20", "kink_band/compression/md/bending"),
    ("shi_2025_md", "H-MD-30", "kink_band/compression/md/bending"),
    # Embedded UD wrinkle (flat surfaces) and the near-UD industrial
    # laminate with two woven 90-degree plies.
    ("li_2025", "S-M-3", "kink_band/compression/ud/non-bending"),
    ("pilato_2022", "K-6", "kink_band/compression/ud/non-bending"),
    # Tension is never bending-dominated.
    ("elhajjar_shams_2016_md", "M-MD-1", "three_mechanism/tension/md/non-bending"),
    ("elhajjar_shams_2016_ud", "M-UD-10", "three_mechanism/tension/ud/non-bending"),
    ("calvo_2023", "I-B5", "three_mechanism/tension/md/non-bending"),
])
def test_cases_fall_in_the_documented_regimes(regenerated, prefix, case, key):
    assert _regime_of(regenerated, prefix, case) == key


def test_the_gate_and_crack_band_paths_are_their_own_regimes(regenerated):
    gate = regenerated["regimes"]["penetration_gate/compression/ud/non-bending"]
    assert gate["datasets"] == ["li_2025_multi_wrinkle_compression"]
    assert gate["n"] == 6
    crack_band = regenerated["regimes"][
        "progressive_damage/compression/ud/non-bending"]
    assert [c["case"] for c in crack_band["cases"]] == [
        "S-M-2 (nx16_calibrated)", "S-M-4 (nx16_calibrated)",
        "S-M-5 (nx16_calibrated)"]


def test_a_moved_prediction_is_reported_as_drift(regenerated):
    """What the guard sees after a model change that moves one pin."""
    ledger = copy.deepcopy(_LEDGER)
    ds = next(d for d in ledger["datasets"]
              if d["name"].startswith("elhajjar_shams_2016_md"))
    ds["cases"][0]["expected_analytical_kd"] *= 1.2
    moved = fit_margins.build_table(ledger)
    diffs = fit_margins.compare_tables(regenerated, moved)
    assert any(d.startswith("regimes.three_mechanism/tension/md/non-bending")
               for d in diffs), diffs
    assert not any("kink_band" in d for d in diffs), diffs


def test_compare_tables_tolerates_only_last_digit_rounding():
    base = {"a": 0.7715, "b": [1, "x"], "c": {"d": None}}
    assert fit_margins.compare_tables(base, {**base, "a": 0.7716}) == []
    assert fit_margins.compare_tables(base, {**base, "a": 0.772})
    assert fit_margins.compare_tables(base, {**base, "b": [1, "y"]})
    assert fit_margins.compare_tables(base, {**base, "e": 1})
    assert fit_margins.compare_tables(base, {"a": 0.7715, "b": [1, "x"]})


def test_validate_fails_while_the_margins_are_stale(monkeypatch):
    """``scripts/validate.py`` (and so CI's harness test) flags a stale
    table even when every pinned knockdown still matches."""

    class _Stale:
        @staticmethod
        def check():
            return ["regimes.x.margin_factor: shipped 0.7, regenerated 0.8"]

    monkeypatch.setattr(validate, "_margins_module", lambda: _Stale)
    assert validate.main([]) == 1


def test_validate_update_regenerates_the_margins(monkeypatch, tmp_path):
    """One command after a model change: re-pin, then re-fit from the
    re-pinned ledger."""
    ledger_copy = tmp_path / "ledger.json"
    ledger_copy.write_text((_HERE / "ledger.json").read_text())
    calls = {}

    class _Recorder:
        @staticmethod
        def build_table(ledger):
            calls["ledger"] = ledger
            return {"regimes": {}}

        @staticmethod
        def write(table):
            calls["written"] = table
            return tmp_path / "validation_margins.json"

        @staticmethod
        def format_table(table):
            return ""

    monkeypatch.setattr(validate, "REPO", tmp_path)
    monkeypatch.setattr(validate, "LEDGER", ledger_copy)
    monkeypatch.setattr(validate, "_margins_module", lambda: _Recorder)
    assert validate.main(["--update"]) == 0
    assert calls["written"] == {"regimes": {}}
    assert calls["ledger"] == json.loads(ledger_copy.read_text())
