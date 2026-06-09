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
# Issue #254 — decay_floor symmetric under sign-flipped load
# --------------------------------------------------------------------------- #


def _graded_kd(loading: str, decay_floor: float) -> float:
    # Quasi-isotropic layup on a config whose through-thickness Gaussian
    # actually decays inside the laminate: 24 plies (T = 4.39 mm) with an
    # explicit 0.5 mm decay scale, so the surface plies see raw ~ 0 and
    # the floor governs how much angle they retain.
    cfg = AnalysisConfig(
        amplitude=0.75, wavelength=12.9, width=12.9,
        morphology="graded", decay_floor=decay_floor, loading=loading,
        angles=[0.0, 45.0, -45.0, 90.0] * 3
        + [90.0, -45.0, 45.0, 0.0] * 3,
        through_thickness_decay_scale=0.5,
    )
    return float(
        WrinkleAnalysis(cfg).run(analytical_only=True).analytical_knockdown
    )


@pytest.mark.parametrize("loading", ["compression", "tension"])
def test_decay_floor_effective_under_both_loadings_issue_254(loading):
    """decay_floor produces the same envelope semantics under a
    sign-flipped load: raising the floor retains more misalignment in
    the outer plies and monotonically lowers the knockdown, in tension
    AND compression. (Before the #254 fix, compression ignored the
    floor entirely while tension honored it.)"""
    kds = [_graded_kd(loading, f) for f in (0.0, 0.5, 1.0)]
    assert kds[0] > kds[1] > kds[2], (
        f"{loading}: KD must decrease monotonically as decay_floor "
        f"retains more angle, got {kds}"
    )
    # The floor must have a material effect, not a numerical whisper.
    assert kds[0] - kds[2] > 0.01


def test_decay_floor_one_disables_compression_decay():
    """floor=1.0 means every ply keeps the full angle — identical to a
    decay scale so large the Gaussian never drops below ~1."""
    kd_floor1 = _graded_kd("compression", 1.0)
    cfg = AnalysisConfig(
        amplitude=0.75, wavelength=12.9, width=12.9,
        morphology="graded", decay_floor=0.0, loading="compression",
        angles=[0.0, 45.0, -45.0, 90.0] * 3
        + [90.0, -45.0, 45.0, 0.0] * 3,
        through_thickness_decay_scale=1.0e6,
    )
    kd_nodecay = float(
        WrinkleAnalysis(cfg).run(analytical_only=True).analytical_knockdown
    )
    assert kd_floor1 == pytest.approx(kd_nodecay, rel=1e-9)


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
