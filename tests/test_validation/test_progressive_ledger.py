"""Regression harness for the crack-band progressive-damage predictions.

These used to live only in untracked `validation/li_progressive_*.csv`
artefacts. Between 2026-07-04 and 2026-09-29 they moved by up to 0.034
per case with nothing to catch it, and the mesh-sensitivity figures in
VALIDATION.md went stale by more than a factor of two — including a
reversal of sign. This pins them.

Two things make this harness different from the analytical ledger next
door:

- **It is slow.** A load-stepping Newton solve to peak load takes tens
  of seconds per case, so these are marked ``slow`` and run in the
  ``test-full`` lane, not on every push.
- **The numbers are mesh- and Gf-locked.** The `nx36_refined` recipe is
  deliberately *not* a second calibration; it is evidence that
  refinement changes the answer, pinned so that finding cannot quietly
  stop being true.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
_LEDGER = json.loads((_HERE / "ledger.json").read_text())
_PD = _LEDGER.get("progressive_damage")

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def driver():
    """The same driver the CSVs came from."""
    sys.path.insert(0, str(_REPO / "validation"))
    import validate_li_progressive as module

    return module


def _recipe_cases():
    for recipe in _PD["recipes"]:
        for case in recipe["cases"]:
            yield pytest.param(
                recipe, case, id=f"{recipe['name']}-{case['case']}"
            )


def _run_case(driver, recipe, case, tmp_path):
    """Recompute one case through the shipped validation driver."""
    spec = next(
        c for c in driver.LI2025 if c[0] == case["case"]
    )
    records = driver.run(
        [spec], dataset_label="F",
        csv_path=tmp_path / f"{recipe['name']}_{case['case']}.csv",
        done=set(),
        nx=recipe["nx"], ny=recipe["ny"], nz=recipe["nz_per_ply"],
        pocket=recipe["resin_pocket"],
        height_scale=1.0, length_scale=1.0,
        n_increments=recipe["n_increments"],
        residual_factor=recipe["residual_factor"],
        crack_band=recipe["crack_band"],
        Gc_fiber=recipe["Gf_fiber"],
    )
    assert len(records) == 1
    return float(records[0]["kd_pred"])


@pytest.mark.parametrize("recipe,case", list(_recipe_cases()))
def test_case_matches_pinned_baseline(recipe, case, driver, tmp_path):
    predicted = _run_case(driver, recipe, case, tmp_path)
    pinned = float(case["expected_progressive_kd"])
    tol = float(_PD["rel_tolerance"])
    assert predicted == pytest.approx(pinned, rel=tol), (
        f"{recipe['name']}/{case['case']}: progressive-damage knockdown "
        f"moved from the pinned {pinned} to {predicted}. These numbers are "
        f"mesh- and Gf-locked; if the change is intentional, re-pin the "
        f"ledger AND update the figures in docs/internal/VALIDATION.md, "
        f"which quote them."
    )


def test_the_calibrated_mesh_reproduces_the_measured_amplitude_ordering():
    """The property the crack band is credited with: at fixed peak angle,
    knockdown must rise as amplitude falls, like the measurement.

    Pinned from the ledger rather than recomputed — cheap, and it is the
    *pins* whose physical ordering matters.
    """
    recipe = next(r for r in _PD["recipes"] if r["name"] == "nx16_calibrated")
    by_case = {c["case"]: c for c in recipe["cases"]}
    order = ["S-M-2", "S-M-4", "S-M-5"]           # amplitude 1.5, 1.0, 0.5
    amps = [by_case[c]["amplitude_p2p_mm"] for c in order]
    assert amps == sorted(amps, reverse=True), "fixture order assumption"

    predicted = [by_case[c]["expected_progressive_kd"] for c in order]
    measured = [by_case[c]["measured_kd"] for c in order]
    assert measured == sorted(measured), "measured ordering assumption"
    assert predicted == sorted(predicted), (
        f"the calibrated mesh no longer reproduces the measured amplitude "
        f"ordering: {dict(zip(order, predicted))}"
    )


def test_refinement_still_changes_the_answer_materially():
    """The documented mesh-sensitivity finding, pinned.

    If this ever stops holding, the crack band has become mesh-objective
    and the caveat in VALIDATION.md should be rewritten — loudly, not by
    someone noticing a stale paragraph months later.
    """
    coarse = next(r for r in _PD["recipes"] if r["name"] == "nx16_calibrated")
    fine = next(r for r in _PD["recipes"] if r["name"] == "nx36_refined")
    shared = {c["case"] for c in coarse["cases"]} & {
        c["case"] for c in fine["cases"]
    }
    assert shared, "no case is pinned at both meshes"

    for name in shared:
        a = next(c for c in coarse["cases"] if c["case"] == name)
        b = next(c for c in fine["cases"] if c["case"] == name)
        gap = abs(
            b["expected_progressive_kd"] - a["expected_progressive_kd"]
        )
        assert gap > 0.05, (
            f"{name}: nx=16 and nx=36 now agree to within {gap:.3f}. That "
            f"would be good news, but VALIDATION.md documents them as "
            f"materially different — update it."
        )


def test_every_pinned_case_records_its_full_recipe():
    """A mesh-locked number is meaningless without the mesh."""
    required = {
        "nx", "ny", "nz_per_ply", "Gf_fiber", "crack_band",
        "n_increments", "residual_factor", "resin_pocket",
    }
    for recipe in _PD["recipes"]:
        missing = required - set(recipe)
        assert not missing, f"{recipe['name']} omits {sorted(missing)}"
        for case in recipe["cases"]:
            assert "expected_progressive_kd" in case
            assert "measured_kd" in case
