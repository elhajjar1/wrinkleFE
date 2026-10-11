"""The gate-transfer claims of issue #435, pinned.

``docs/internal/VALIDATION.md`` ("Gate transfer to new materials") states:

* the shipped penetration-gate presets do not transfer blind to UD carbon,
  on the pooled sets (Shi, Thor, Pilato) and on the quarantined one;
* re-fitting ``gamma_Y`` to one wrinkled test predicted Shi's and Pilato's
  other cases well;
* it failed on Krishnappa & Gururaja's 18 configurations, where the fitted
  ``gamma_Y`` is not a material constant, even within one wrinkle
  position. That dataset is quarantined (issue #445), so it is reported
  on its own and the one-test question stays open.

These tests fail if the gate or the ledger changes in a way that makes
any of those statements untrue.
"""

from __future__ import annotations

import dataclasses
import statistics
import sys
from pathlib import Path

import pytest

from wrinklefe.core.penetration_gate import GATE_LI2024_MOULDED, GATE_LI2025_VACBAG

_ROOT = Path(__file__).resolve().parents[2]
_LARGE = "L Krishnappa (2024)"


@pytest.fixture(scope="module")
def study():
    sys.path.insert(0, str(_ROOT / "validation"))
    import gate_transfer_study

    return gate_transfer_study


@pytest.fixture(scope="module")
def cases(study):
    data = study.carbon_cases()
    assert sum(len(v) for v in data.values()) == 28
    assert len(data[_LARGE]) == 18
    return data


@pytest.fixture(scope="module")
def pooled(study, cases):
    """The carbon sets the ledger does not quarantine."""
    held_out = study.quarantined_labels()
    return {k: v for k, v in cases.items() if k not in held_out}


def test_only_the_wet_layup_set_is_quarantined(study, pooled):
    assert study.quarantined_labels() == {_LARGE}
    assert sum(len(v) for v in pooled.values()) == 10


def test_blind_presets_are_unsafe_on_new_carbon(study, cases, pooled):
    vacbag = dataclasses.replace(GATE_LI2025_VACBAG, position_q=None)
    moulded = [e for c in pooled.values()
               for e in study.blind_errors(c, GATE_LI2024_MOULDED)]
    assert sum(e > 0.5 for e in moulded) == 7
    assert max(moulded) > 90.0
    assert all(e > 0.5 for c in pooled.values()
               for e in study.blind_errors(c, vacbag))
    # The quarantined set, on its own, agrees.
    held = study.blind_errors(cases[_LARGE], GATE_LI2024_MOULDED)
    assert sum(e > 0.5 for e in held) == 14
    assert all(e > 0.5 for e in study.blind_errors(cases[_LARGE], vacbag))


def test_one_test_calibration_works_on_shi_and_pilato(study, cases):
    """Shi and Pilato (3 and 5 cases): small errors, stable gamma_Y."""
    for label in ("H Shi (2025)", "K Pilato (2022)"):
        errs, gammas, failed = study.anchored_errors(
            cases[label], GATE_LI2024_MOULDED)
        assert failed == 0
        assert statistics.mean(abs(x) for x in errs) < 6.0
        assert max(gammas) / min(gammas) < 1.6


def test_one_test_calibration_is_mixed_on_thor(study, cases):
    """Thor's two cases: never more than 15% unsafe, but a 29% miss."""
    errs, _gammas, _failed = study.anchored_errors(
        cases["J Thor (2021)"], GATE_LI2024_MOULDED)
    assert len(errs) == 2
    assert max(errs) < 15.0
    assert min(errs) < -25.0


def test_one_test_calibration_fails_on_the_quarantined_set(study, cases):
    """Krishnappa & Gururaja (18 configurations, one material): no single
    test calibrates the gate, and gamma_Y is not a material constant."""
    errs, gammas, failed = study.anchored_errors(
        cases[_LARGE], GATE_LI2024_MOULDED)
    assert failed == 0 and len(errs) == 18 * 17
    assert statistics.mean(abs(x) for x in errs) > 12.0
    assert sum(x > 15.0 for x in errs) / len(errs) > 0.2
    assert max(errs) > 50.0
    assert max(gammas) / min(gammas) > 3.0


def test_even_one_position_of_the_quarantined_set_calibrates_poorly(
        study, cases):
    """Within WL, WM or WR alone the one-test error is still at least
    twice Shi's and Pilato's."""
    groups = study.position_groups(cases[_LARGE])
    assert sorted(groups) == ["WL", "WM", "WR"]
    for group in groups.values():
        errs, _gammas, _failed = study.anchored_errors(
            group, GATE_LI2024_MOULDED)
        assert statistics.mean(abs(x) for x in errs) > 8.0


def test_the_steep_vacuum_bag_shape_does_not_transfer(study, cases):
    vacbag = dataclasses.replace(GATE_LI2025_VACBAG, position_q=None)
    failed = sum(study.anchored_errors(c, vacbag)[2] for c in cases.values())
    assert failed >= 5  # its penetration term is ~0 on Pilato's wrinkles
