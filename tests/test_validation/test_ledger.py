"""Validation-ledger regression harness (issue #254, deliverable 1).

Recomputes every case in ``ledger.json`` and asserts the analytical
knockdown matches the pinned baseline within the ledger's tolerance —
the committed, reproducible harness whose absence forced the revert of
the graded-decay fix (commit ``00584b4``). Also documents the two #254
defects as strict xfails so that fixing them flips these tests loudly.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis

_HERE = Path(__file__).resolve().parent
_LEDGER = json.loads((_HERE / "ledger.json").read_text())

_spec = importlib.util.spec_from_file_location(
    "validate", _HERE.parents[1] / "scripts" / "validate.py"
)
validate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validate)


def _all_cases():
    for dataset in _LEDGER["datasets"]:
        for case in dataset["cases"]:
            yield pytest.param(
                dataset, case, id=f"{dataset['name']}-{case['case']}"
            )


@pytest.mark.parametrize(("dataset", "case"), list(_all_cases()))
def test_case_matches_pinned_baseline(dataset, case):
    predicted = validate.run_case(dataset, case)
    pinned = case["expected_analytical_kd"]
    tol = _LEDGER["rel_tolerance"]
    assert predicted == pytest.approx(pinned, rel=tol), (
        f"{case['case']} analytical KD drifted from the pinned baseline "
        f"({predicted} vs {pinned}). If intentional, re-pin via "
        f"'python scripts/validate.py --update' and explain the move."
    )


def test_validate_script_passes():
    """The one-command harness exits 0 from a clean checkout."""
    assert validate.main([]) == 0


# --------------------------------------------------------------------------- #
# Issue #254 defect documentation — strict xfail flips when the fix lands
# --------------------------------------------------------------------------- #


def _graded_kd(loading: str, decay_floor: float) -> float:
    # Quasi-isotropic IM7/8552: the tension path's decay_floor handling
    # is exercised by multi-angle layups (a UD stack shows no effect on
    # either path), which makes the compression-side inertness visible
    # as an asymmetry rather than a shared no-op.
    cfg = AnalysisConfig(
        amplitude=0.75, wavelength=12.9, width=12.9,
        morphology="graded", decay_floor=decay_floor, loading=loading,
        angles=[0.0, 45.0, -45.0, 90.0, 90.0, -45.0, 45.0, 0.0],
    )
    return float(
        WrinkleAnalysis(cfg).run(analytical_only=True).analytical_knockdown
    )


@pytest.mark.xfail(
    strict=True,
    reason="issue #254: decay_floor is inert on the graded compression "
    "path (honored in tension). When the re-landed fix makes it "
    "effective, this xfail flips and must be converted to a real test.",
)
def test_decay_floor_affects_compression_issue_254():
    kd_floor0 = _graded_kd("compression", 0.0)
    kd_floor1 = _graded_kd("compression", 0.9)
    assert kd_floor0 != pytest.approx(kd_floor1), (
        "decay_floor=0.0 and 0.9 produce identical compression knockdown"
    )


def test_decay_floor_compression_tension_asymmetry_is_current_behavior():
    """Pin the asymmetry itself: tension honors decay_floor, compression
    does not. Documents the user-visible inconsistency named in #254;
    delete together with the xfail above when the fix lands."""
    comp_delta = abs(
        _graded_kd("compression", 0.0) - _graded_kd("compression", 0.9)
    )
    tens_delta = abs(
        _graded_kd("tension", 0.0) - _graded_kd("tension", 0.9)
    )
    assert comp_delta == pytest.approx(0.0, abs=1e-12)
    assert tens_delta > 1e-6


def test_ledger_recipe_round_trips_config():
    """The ledger carries everything needed to rebuild each config."""
    dataset = _LEDGER["datasets"][0]
    case = dataset["cases"][0]
    cfg = validate.case_config(dataset, case)
    assert cfg.amplitude == pytest.approx(case["amplitude_p2p_mm"] / 2.0)
    assert cfg.wavelength == pytest.approx(case["wavelength_mm"])
    assert len(cfg.angles) == 14 and set(cfg.angles) == {0.0}
    assert cfg.material.name == dataset["material"]


def test_replace_preserves_ledger_predictions():
    """dataclasses.replace on a ledger config does not perturb the KD
    (guards the sweep/convergence helpers that clone configs)."""
    dataset = _LEDGER["datasets"][0]
    case = dataset["cases"][0]
    cfg = validate.case_config(dataset, case)
    kd1 = WrinkleAnalysis(cfg).run(analytical_only=True).analytical_knockdown
    kd2 = WrinkleAnalysis(replace(cfg)).run(
        analytical_only=True
    ).analytical_knockdown
    assert kd1 == pytest.approx(kd2, rel=1e-12)
