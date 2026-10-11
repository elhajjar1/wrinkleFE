"""The gate-transfer claims of issue #435, pinned.

``docs/internal/VALIDATION.md`` ("Gate transfer to new materials") states
that the shipped penetration-gate presets do not transfer blind to UD
carbon, but that the moulded preset's shape does once ``gamma_Y`` is
re-fitted to one wrinkled test of the new material. These tests fail if
the gate or the ledger changes in a way that makes either claim untrue.
"""

from __future__ import annotations

import dataclasses
import statistics
import sys
from pathlib import Path

import pytest

from wrinklefe.core.penetration_gate import GATE_LI2024_MOULDED, GATE_LI2025_VACBAG

_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def study():
    sys.path.insert(0, str(_ROOT / "validation"))
    import gate_transfer_study

    return gate_transfer_study


@pytest.fixture(scope="module")
def cases(study):
    data = study.carbon_cases()
    assert sum(len(v) for v in data.values()) == 10
    return data


def test_blind_presets_are_unsafe_on_new_carbon(study, cases):
    moulded = [e for c in cases.values()
               for e in study.blind_errors(c, GATE_LI2024_MOULDED)]
    vacbag = dataclasses.replace(GATE_LI2025_VACBAG, position_q=None)
    vac = [e for c in cases.values() for e in study.blind_errors(c, vacbag)]
    assert sum(e > 0.5 for e in moulded) == 7
    assert max(moulded) > 90.0
    assert all(e > 0.5 for e in vac)


def test_one_test_calibration_of_the_moulded_shape_transfers(study, cases):
    errs = []
    for c in cases.values():
        e, _gammas, failed = study.anchored_errors(c, GATE_LI2024_MOULDED)
        assert failed == 0
        errs += e
    assert len(errs) == 28
    assert statistics.mean(abs(x) for x in errs) < 6.0
    assert max(errs) < 15.5  # never more than ~15 % unsafe


def test_the_steep_vacuum_bag_shape_does_not_transfer(study, cases):
    vacbag = dataclasses.replace(GATE_LI2025_VACBAG, position_q=None)
    failed = sum(study.anchored_errors(c, vacbag)[2] for c in cases.values())
    assert failed >= 5  # its penetration term is ~0 on Pilato's wrinkles


def test_calibrated_gamma_y_is_a_material_constant(study, cases):
    """Within a material the re-fitted gamma_Y is stable; between
    materials it differs several-fold, so it cannot be assumed."""
    per_material = {
        label: study.anchored_errors(c, GATE_LI2024_MOULDED)[1]
        for label, c in cases.items()
    }
    for gammas in per_material.values():
        assert max(gammas) / min(gammas) < 1.6
    means = [statistics.mean(g) for g in per_material.values()]
    assert max(means) / min(means) > 4.0
