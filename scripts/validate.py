#!/usr/bin/env python3
"""Recompute every validation-ledger case and compare against baselines.

One command from a clean checkout:

    python scripts/validate.py            # recompute + compare, exit 1 on drift
    python scripts/validate.py --update   # re-pin baselines after a deliberate change

The ledger (tests/test_validation/ledger.json) carries, per case, the
full analysis recipe, the measured experimental knockdown, and a pinned
``expected_analytical_kd`` computed by the current code. This script is
the reproducible harness named in issue #254 as the precondition for
re-landing the graded compression decay fix (reverted in 00584b4): run
it before and after a model change and every row that moved is visible,
with measured-vs-predicted error statistics per dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "tests" / "test_validation" / "ledger.json"

sys.path.insert(0, str(REPO / "src"))


def case_config(dataset: dict, case: dict):
    """Build the AnalysisConfig for one ledger case (the reference recipe)."""
    from wrinklefe.analysis import AnalysisConfig
    from wrinklefe.core.layup import parse_layup
    from wrinklefe.core.material import MaterialLibrary

    return AnalysisConfig(
        amplitude=case["amplitude_p2p_mm"] / 2.0,
        wavelength=case["wavelength_mm"],
        width=case["wavelength_mm"],
        morphology=dataset["morphology"],
        loading=dataset["loading"],
        material=MaterialLibrary().get(dataset["material"]),
        angles=parse_layup(dataset["layup"]),
        ply_thickness=dataset["ply_thickness_mm"],
    )


def run_case(dataset: dict, case: dict) -> float:
    from wrinklefe.analysis import WrinkleAnalysis

    cfg = case_config(dataset, case)
    return float(
        WrinkleAnalysis(cfg).run(analytical_only=True).analytical_knockdown
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update", action="store_true",
        help="re-pin expected_analytical_kd to the current code's output",
    )
    args = parser.parse_args(argv)

    ledger = json.loads(LEDGER.read_text())
    tol = float(ledger["rel_tolerance"])
    failures = 0

    for dataset in ledger["datasets"]:
        print(f"\n=== {dataset['name']}  ({dataset['reference']})")
        print(
            f"{'case':<8} {'KD meas':>8} {'KD pred':>9} {'pinned':>9} "
            f"{'drift%':>8} {'meas err%':>10}"
        )
        abs_errors = []
        for case in dataset["cases"]:
            predicted = run_case(dataset, case)
            pinned = float(case["expected_analytical_kd"])
            drift = abs(predicted - pinned) / max(abs(pinned), 1e-30)
            meas_err = (
                100.0 * (predicted - case["measured_kd"]) / case["measured_kd"]
            )
            abs_errors.append(abs(predicted - case["measured_kd"]))
            flag = ""
            if drift > tol:
                flag = "  <-- DRIFT vs pinned baseline"
                failures += 1
            print(
                f"{case['case']:<8} {case['measured_kd']:>8.3f} "
                f"{predicted:>9.4f} {pinned:>9.4f} {100 * drift:>8.3f} "
                f"{meas_err:>10.1f}{flag}"
            )
            if args.update:
                case["expected_analytical_kd"] = round(predicted, 6)
        mae = sum(abs_errors) / len(abs_errors)
        print(f"MAE vs measured KD: {mae:.4f}  ({len(abs_errors)} cases)")

    if args.update:
        LEDGER.write_text(json.dumps(ledger, indent=2) + "\n")
        print(f"\nBaselines re-pinned in {LEDGER.relative_to(REPO)}")
        return 0
    if failures:
        print(
            f"\n{failures} case(s) drifted beyond rel_tolerance={tol}. "
            "If the change is intentional, re-pin with --update and "
            "explain the move in the commit message."
        )
        return 1
    print("\nAll cases match the pinned baselines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
