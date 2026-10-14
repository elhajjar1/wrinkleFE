#!/usr/bin/env python3
"""Fit the regime-conditional validation margins from the ledger (#455).

The validation ledger (``tests/test_validation/ledger.json``) is not
shipped with the package, so the margins the NCR summary and the results
JSON report are fitted here and shipped as package data
(``src/wrinklefe/core/validation_margins.json``)::

    python scripts/fit_margins.py            # check: exit 1 if the table is stale
    python scripts/fit_margins.py --write    # regenerate the shipped table

``python scripts/validate.py --update`` re-pins the ledger and then
regenerates this table, so after a deliberate model change one command
brings both up to date. ``python scripts/validate.py`` and
``tests/test_validation/test_margins_ledger.py`` fail while the table
is stale.

Residuals: every ledger case with a measured strength knockdown, in every
dataset without a ``quarantine`` entry, gives ``measured / predicted``
for each pinned prediction it carries: ``expected_analytical_kd``
(kink-band in compression, three-mechanism in tension) and
``expected_gate_kd`` (penetration gate). The calibration recipe of the
pinned progressive-damage block adds the crack-band FE path. The pinned
values are used, not recomputed: ``validate.py`` guarantees they equal
the current code's output.

Regime: model path, load, laminate type (``laminate_class`` in
``wrinklefe.core.margins``, from the case's layup) and the
bending-dominated flag. The flag is measured on the specimen's wrinkle
geometry: the case's recipe, plus the dataset's ``eccentricity_recipe``
overrides where the knockdown recipe does not reproduce that geometry
(Dataset H's one-sided trough).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "tests" / "test_validation" / "ledger.json"
TABLE = REPO / "src" / "wrinklefe" / "core" / "validation_margins.json"

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from validate import case_config  # noqa: E402

from wrinklefe.core.margins import (  # noqa: E402
    MarginResidual,
    build_margin_table,
    regime_for,
)

#: Absolute tolerance for numbers when comparing tables: the table rounds
#: to 4 decimals, so this absorbs a last-digit rounding flip between
#: platforms and nothing more.
_TOL = 2e-4

#: Fields of an ``eccentricity_recipe`` that are notes, not overrides.
_NOTE_KEYS = {"_comment"}


def _geometry_results(dataset: dict, case: dict):
    """Analytical results on the specimen's geometry, for its regime.

    The knockdown recipe, plus the dataset's (and the case's)
    ``eccentricity_recipe`` overrides where they exist. Surface resin
    pockets are a material zone, not geometry, and FE-only, so they are
    off here (as in ``validation/bending_mode_study.py``).
    """
    from wrinklefe.analysis import WrinkleAnalysis

    cfg = case_config(dataset, case)
    overrides = {
        k: v
        for src in (dataset.get("eccentricity_recipe", {}),
                    case.get("eccentricity_recipe", {}))
        for k, v in src.items() if k not in _NOTE_KEYS
    }
    if overrides:
        cfg = dataclasses.replace(
            cfg, analytical_only=True, enable_surface_resin_pockets=False,
            **overrides,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WrinkleAnalysis(cfg).run(analytical_only=True)


def ledger_residuals(ledger: dict) -> list[MarginResidual]:
    """Every non-quarantined ``measured / predicted`` ratio, classified."""
    out: list[MarginResidual] = []
    geometry: dict[tuple[str, str], Any] = {}
    for ds in ledger["datasets"]:
        if "quarantine" in ds:
            continue
        for case in ds["cases"]:
            measured = case.get("measured_kd")
            if measured is None:
                continue
            geo = _geometry_results(ds, case)
            geometry[(ds["name"], case["case"])] = geo
            loading = geo.config.loading
            angles = geo.config.angles
            path = ("three_mechanism" if loading == "tension"
                    else "kink_band")
            predictions = [(path, case["expected_analytical_kd"])]
            if "expected_gate_kd" in case:
                predictions.append(
                    ("penetration_gate", case["expected_gate_kd"]))
            for model, predicted in predictions:
                out.append(MarginResidual(
                    regime=regime_for(model, loading, angles,
                                      geo.bending_dominated),
                    ratio=float(measured) / float(predicted),
                    dataset=ds["name"], case=case["case"],
                ))

    # Crack-band FE: the calibration recipe only. The other recipes are
    # mesh-sensitivity sentinels on the same specimens, not predictions.
    pd = ledger.get("progressive_damage")
    if pd:
        for recipe in pd["recipes"]:
            if not str(recipe.get("role", "")).startswith("calibration"):
                continue
            for case in recipe["cases"]:
                geo = geometry[(pd["dataset"], case["case"])]
                out.append(MarginResidual(
                    regime=regime_for(
                        "progressive_damage", geo.config.loading,
                        geo.config.angles, geo.bending_dominated),
                    ratio=(float(case["measured_kd"])
                           / float(case["expected_progressive_kd"])),
                    dataset=pd["dataset"],
                    case=f"{case['case']} ({recipe['name']})",
                ))
    return out


def build_table(ledger: dict | None = None) -> dict:
    """The margin table the current ledger implies."""
    if ledger is None:
        ledger = json.loads(LEDGER.read_text())
    excluded = [ds["name"] for ds in ledger["datasets"] if "quarantine" in ds]
    return build_margin_table(
        ledger_residuals(ledger),
        source=("tests/test_validation/ledger.json (datasets without a "
                "'quarantine' entry)"),
        excluded_datasets=excluded,
    )


def compare_tables(expected: Any, actual: Any, path: str = "") -> list[str]:
    """Human-readable differences between two tables (empty = equal)."""
    where = path or "<root>"
    if isinstance(expected, dict) and isinstance(actual, dict):
        diffs = []
        for key in sorted(set(expected) | set(actual)):
            sub = f"{path}.{key}" if path else key
            if key not in actual:
                diffs.append(f"{sub}: missing from the regenerated table")
            elif key not in expected:
                diffs.append(f"{sub}: new in the regenerated table")
            else:
                diffs.extend(compare_tables(expected[key], actual[key], sub))
        return diffs
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{where}: {len(expected)} entries shipped, "
                    f"{len(actual)} regenerated"]
        diffs = []
        for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
            diffs.extend(compare_tables(e, a, f"{path}[{i}]"))
        return diffs
    numeric = (int, float)
    if (isinstance(expected, numeric) and isinstance(actual, numeric)
            and not isinstance(expected, bool)
            and not isinstance(actual, bool)):
        if math.isclose(float(expected), float(actual), rel_tol=0.0,
                        abs_tol=_TOL):
            return []
        return [f"{where}: shipped {expected}, regenerated {actual}"]
    if expected != actual:
        return [f"{where}: shipped {expected!r}, regenerated {actual!r}"]
    return []


def check(table: dict | None = None) -> list[str]:
    """Differences between the shipped table and the ledger's (empty = fresh)."""
    shipped = json.loads(TABLE.read_text())
    return compare_tables(shipped, table if table is not None else build_table())


def write(table: dict | None = None) -> Path:
    """Write the table the current ledger implies to the package data."""
    if table is None:
        table = build_table()
    TABLE.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n",
                     encoding="utf-8")
    return TABLE


def format_table(table: dict) -> str:
    """The regimes as a plain-text table."""
    lines = [
        f"Validation margins: {table['coverage']:.0%} of each regime's "
        f"cases conservative; at least {table['min_cases']} cases per "
        f"margin",
        f"{'regime':48s} {'n':>3s} {'margin':>7s} {'q(mix)':>7s} "
        f"{'q(norm)':>7s} {'min':>6s}",
    ]
    for key, e in table["regimes"].items():
        if e["status"] == "fitted":
            cols = (f"{e['margin_factor']:7.3f} {e['ratio_quantile']:7.3f} "
                    f"{e['normal_ratio_quantile']:7.3f}")
        else:
            cols = f"{'insufficient evidence':>23s}"
        lines.append(f"{key:48s} {e['n']:3d} {cols} {e['min_ratio']:6.3f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true",
        help="regenerate src/wrinklefe/core/validation_margins.json",
    )
    args = parser.parse_args(argv)

    table = build_table()
    print(format_table(table))
    if args.write:
        out = write(table)
        print(f"\nMargins written to {out.relative_to(REPO)}")
        return 0
    diffs = check(table)
    if diffs:
        print("\nThe shipped validation margins are stale:")
        for d in diffs:
            print(f"  {d}")
        print("Re-run 'python scripts/fit_margins.py --write' (or "
              "'python scripts/validate.py --update') and commit the table.")
        return 1
    print("\nThe shipped validation margins match the ledger.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
