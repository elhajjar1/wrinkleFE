"""The gate-transfer claims of issue #435, pinned.

``docs/internal/VALIDATION.md`` ("Gate transfer to new materials") states:

* the shipped penetration-gate presets do not transfer blind to UD carbon;
* re-fitting ``gamma_Y`` to one wrinkled test looks adequate on the small
  carbon sets (2-5 cases per material), but fails on Krishnappa &
  Gururaja's 18 configurations of one material, where the fitted
  ``gamma_Y`` is not a material constant.

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


def test_blind_presets_are_unsafe_on_new_carbon(study, cases):
    moulded = [e for c in cases.values()
               for e in study.blind_errors(c, GATE_LI2024_MOULDED)]
    vacbag = dataclasses.replace(GATE_LI2025_VACBAG, position_q=None)
    vac = [e for c in cases.values() for e in study.blind_errors(c, vacbag)]
    assert sum(e > 0.5 for e in moulded) == 21
    assert max(moulded) > 90.0
    assert all(e > 0.5 for e in vac)


def test_one_test_calibration_looks_fine_on_the_small_sets(study, cases):
    """Shi and Pilato (3 and 5 cases): small errors, stable gamma_Y. This
    is the result that the large set overturns; it is pinned so the
    contrast stays documented."""
    for label in ("H Shi (2025)", "K Pilato (2022)"):
        errs, gammas, failed = study.anchored_errors(
            cases[label], GATE_LI2024_MOULDED)
        assert failed == 0
        assert statistics.mean(abs(x) for x in errs) < 6.0
        assert max(gammas) / min(gammas) < 1.6


def test_one_test_calibration_fails_on_the_large_set(study, cases):
    """Krishnappa & Gururaja (18 configurations, one material): no single
    test calibrates the gate."""
    errs, gammas, failed = study.anchored_errors(
        cases[_LARGE], GATE_LI2024_MOULDED)
    assert failed == 0 and len(errs) == 18 * 17
    assert statistics.mean(abs(x) for x in errs) > 12.0
    assert sum(x > 15.0 for x in errs) / len(errs) > 0.2
    assert max(errs) > 50.0
    # Not a material constant: a >3x spread within one material.
    assert max(gammas) / min(gammas) > 3.0


def test_the_steep_vacuum_bag_shape_does_not_transfer(study, cases):
    vacbag = dataclasses.replace(GATE_LI2025_VACBAG, position_q=None)
    failed = sum(study.anchored_errors(c, vacbag)[2] for c in cases.values())
    assert failed >= 5  # its penetration term is ~0 on Pilato's wrinkles
