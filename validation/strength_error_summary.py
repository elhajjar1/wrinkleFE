#!/usr/bin/env python3
"""How far each strength prediction can be trusted: signed errors vs test.

Regenerates every number behind the FE-strength caveat the app and the NCR
summary carry (``wrinklefe.io.export.FE_STRENGTH_CAVEAT``) and the tables
in ``docs/interpreting_results.md`` ("How far to trust each number").

Signed, not absolute, errors are the point. A prediction above the
measured knockdown says the wrinkled part keeps more strength than it
does, which is the unsafe direction for a disposition; a mean absolute
error hides which way a model misses. Error = (predicted - measured) /
measured.

Three sections:

1. **Analytical paths** on every single-wrinkle dataset, each predicted by
   the model that physically applies to it (the same cases and models as
   ``plot_all_validation.py``).
2. **FE LaRC05 strength** on the Li (2025) UD cases: wrinkled over
   pristine strength at first failure, ``(E_eff,w / E_eff,p) * (FI_p /
   FI_w)``, capped at 1 (stress at failure ~ E_eff * eps / FI). Possible
   since the kinking fix gave LaRC05 its Xc-calibrated intrinsic
   misalignment: before it, a flat UD pristine coupon could not fail
   (max FI ~1e-10) and the FE strengths had to be normalised to the
   near-pristine *wrinkle* S-M-5 instead.
3. **Progressive damage (crack band)**: the values pinned in the
   validation ledger, not re-run here (minutes per case; the slow test
   lane re-checks them).

Run::

    python validation/strength_error_summary.py
"""
from __future__ import annotations

import dataclasses
import json
import statistics
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "validation"))

import numpy as np  # noqa: E402

LEDGER = REPO / "tests" / "test_validation" / "ledger.json"

#: A prediction more than this far above the measurement counts as
#: non-conservative; below it, rounding noise.
_NON_CONSERVATIVE_PCT = 0.5


def _err(pred: float, meas: float) -> float:
    return (pred - meas) / meas * 100.0


def analytical_section() -> None:
    import plot_all_validation as pav

    print("1. Analytical paths (model that applies to each dataset)")
    print(f"   {'dataset':24s} {'method':24s} {'n':>3s} {'MAE%':>6s} "
          f"{'min%':>7s} {'max%':>7s} {'non-cons.':>9s}")
    total = outside = 0
    for label, (fn, _c, _m, method) in pav.DATASETS.items():
        errs = [_err(p, m) for m, p in fn()]
        total += len(errs)
        outside += sum(abs(e) > 20.0 for e in errs)
        nonc = sum(e > _NON_CONSERVATIVE_PCT for e in errs)
        print(f"   {label:24s} {method:24s} {len(errs):3d} "
              f"{statistics.mean(abs(e) for e in errs):6.1f} "
              f"{min(errs):+7.1f} {max(errs):+7.1f} {nonc:5d}/{len(errs)}")
    print(f"   -> {total - outside} of {total} cases within +/-20 %\n")


def fe_larc05_errors() -> dict[str, tuple[float, float, float]]:
    """``{case: (FE knockdown, measured, error %)}`` for Li (2025) UD.

    Also used by ``tests/test_fe_strength_caveat.py`` to keep the caveat's
    claims true.
    """
    from validate import case_config

    from wrinklefe.analysis import WrinkleAnalysis

    ledger = json.loads(LEDGER.read_text())
    ds = next(d for d in ledger["datasets"] if d["name"].startswith("li_2025"))
    out = {}
    for case in ds["cases"]:
        cfg = case_config(ds, case)
        kw = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
        kw.update(analytical_only=False, verbose=False, nx=16, ny=4,
                  nz_per_ply=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = WrinkleAnalysis(type(cfg)(**kw)).run()
        fi = np.asarray(r.failure_indices["larc05"]).mean(axis=-1)
        fi_w = float(fi[np.isfinite(fi)].max())
        fi_p = float(r.baseline_fi["larc05"])
        meas = float(case["measured_kd"])
        kd = min(float(r.modulus_retention_global) * fi_p / fi_w, 1.0)
        out[case["case"]] = (kd, meas, _err(kd, meas))
    return out


def fe_larc05_section() -> None:
    rows = fe_larc05_errors()
    print("2. FE LaRC05 strength, Li (2025) UD glass/epoxy "
          "(wrinkled / pristine)")
    print(f"   {'case':7s} {'FE KD':>7s} {'meas.':>7s} {'error%':>8s}")
    for name, (kd, meas, e) in rows.items():
        print(f"   {name:7s} {kd:7.3f} {meas:7.3f} {e:+8.1f}")
    errs = [e for _, _, e in rows.values()]
    print(f"   -> MAE {statistics.mean(abs(e) for e in errs):.1f} %, "
          f"non-conservative on "
          f"{sum(e > _NON_CONSERVATIVE_PCT for e in errs)} of {len(errs)}\n")


def progressive_section() -> None:
    pd = json.loads(LEDGER.read_text())["progressive_damage"]
    print("3. Progressive damage (crack band), pinned ledger values")
    print(f"   {'recipe':16s} {'case':7s} {'pred.':>7s} {'meas.':>7s} "
          f"{'error%':>8s}")
    for recipe in pd["recipes"]:
        for c in recipe["cases"]:
            pred, meas = c["expected_progressive_kd"], c["measured_kd"]
            print(f"   {recipe['name']:16s} {c['case']:7s} {pred:7.3f} "
                  f"{meas:7.3f} {_err(pred, meas):+8.1f}")


def main() -> None:
    analytical_section()
    fe_larc05_section()
    progressive_section()


if __name__ == "__main__":
    main()
