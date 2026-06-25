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

Issue #326 extends the same guard to the **stiffness** (axial-modulus)
knockdown: cases that also pin ``expected_analytical_modulus_kd`` have
their ``analytical_modulus_knockdown`` recomputed and compared the same
way, so the closed-form CLT modulus estimate (added in #324) can no
longer drift silently. ``--update`` re-pins both baselines.
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
    """Build the AnalysisConfig for one ledger case (the reference recipe).

    Recipe fields default to the dataset level, but a case may override
    ``morphology``, ``layup``, ``ply_thickness_mm``, ``material`` or
    ``loading`` (used by Dataset G, whose uniform and graded specimens
    differ in morphology, layup and ply thickness).
    """
    from wrinklefe.analysis import AnalysisConfig
    from wrinklefe.core.layup import parse_layup
    from wrinklefe.core.material import MaterialLibrary

    def field(key):
        return case[key] if key in case else dataset[key]

    return AnalysisConfig(
        amplitude=case["amplitude_p2p_mm"] / 2.0,
        wavelength=case["wavelength_mm"],
        width=case["wavelength_mm"],
        morphology=field("morphology"),
        loading=field("loading"),
        material=MaterialLibrary().get(field("material")),
        angles=parse_layup(field("layup")),
        ply_thickness=field("ply_thickness_mm"),
    )


def run_case(dataset: dict, case: dict) -> float:
    """Recompute one case's analytical *strength* knockdown."""
    from wrinklefe.analysis import WrinkleAnalysis

    cfg = case_config(dataset, case)
    return float(
        WrinkleAnalysis(cfg).run(analytical_only=True).analytical_knockdown
    )


def run_case_both(dataset: dict, case: dict) -> tuple[float, float]:
    """Recompute the analytical strength and modulus knockdowns in one solve.

    Returns ``(strength_kd, modulus_kd)``. The modulus knockdown is the
    closed-form CLT series-average ``analytical_modulus_knockdown`` (#324),
    which is 1.0 for any non-unidirectional / degenerate config.
    """
    from wrinklefe.analysis import WrinkleAnalysis

    cfg = case_config(dataset, case)
    result = WrinkleAnalysis(cfg).run(analytical_only=True)
    return (
        float(result.analytical_knockdown),
        float(result.analytical_modulus_knockdown),
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
            f"{'case':<10} {'KD meas':>8} {'KD pred':>9} {'pinned':>9} "
            f"{'drift%':>8} {'meas err%':>10}   "
            f"{'Em meas':>8} {'Em pred':>9} {'Em pin':>9} {'drift%':>8}"
        )
        str_errors = []
        mod_errors = []
        for case in dataset["cases"]:
            predicted, predicted_mod = run_case_both(dataset, case)

            # --- strength knockdown (pinned for every case) ---
            pinned = float(case["expected_analytical_kd"])
            drift = abs(predicted - pinned) / max(abs(pinned), 1e-30)
            flag = ""
            if drift > tol:
                flag = "  <-- DRIFT"
                failures += 1
            meas = case.get("measured_kd")
            if meas is not None:
                meas_err = 100.0 * (predicted - meas) / meas
                str_errors.append(abs(predicted - meas))
                meas_s = f"{meas:>8.3f}"
                meas_err_s = f"{meas_err:>10.1f}"
            else:
                meas_s = f"{'--':>8}"
                meas_err_s = f"{'--':>10}"

            # --- modulus knockdown (only where a baseline is pinned) ---
            if "expected_analytical_modulus_kd" in case:
                pinned_mod = float(case["expected_analytical_modulus_kd"])
                drift_mod = (
                    abs(predicted_mod - pinned_mod)
                    / max(abs(pinned_mod), 1e-30)
                )
                if drift_mod > tol:
                    flag += "  <-- MOD DRIFT"
                    failures += 1
                meas_mod = case.get("measured_kd_modulus")
                if meas_mod is not None:
                    mod_errors.append(abs(predicted_mod - meas_mod))
                    meas_mod_s = f"{meas_mod:>8.3f}"
                else:
                    meas_mod_s = f"{'--':>8}"
                mod_cols = (
                    f"   {meas_mod_s} {predicted_mod:>9.4f} "
                    f"{pinned_mod:>9.4f} {100 * drift_mod:>8.3f}"
                )
                if args.update:
                    case["expected_analytical_modulus_kd"] = round(
                        predicted_mod, 6
                    )
            else:
                mod_cols = ""

            print(
                f"{case['case']:<10} {meas_s} "
                f"{predicted:>9.4f} {pinned:>9.4f} {100 * drift:>8.3f} "
                f"{meas_err_s}{mod_cols}{flag}"
            )
            if args.update:
                case["expected_analytical_kd"] = round(predicted, 6)
        if str_errors:
            mae = sum(str_errors) / len(str_errors)
            print(
                f"MAE strength vs measured KD: {mae:.4f}  "
                f"({len(str_errors)} cases)"
            )
        if mod_errors:
            mae_m = sum(mod_errors) / len(mod_errors)
            print(
                f"MAE modulus  vs measured KD: {mae_m:.4f}  "
                f"({len(mod_errors)} cases)"
            )

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
