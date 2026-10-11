#!/usr/bin/env python3
"""Issue #435: does the penetration gate transfer to UD carbon?

The two shipped gate presets were fitted to one S-glass prepreg (Li 2024
moulded, Li 2025 vacuum-bag). Four UD carbon datasets in the ledger were
fitted by neither: Shi et al. (2025), Thor et al. (2021), Pilato et al.
(2022) and Krishnappa & Gururaja (2024), 28 cases in all. This study
asks:

1. **Blind.** Do the shipped presets, unchanged, predict them?
2. **One-test calibration.** Keep a preset's *shape* (its penetration
   transition ``dt0`` and exponent ``p``) and re-fit only ``gamma_Y`` to a
   single wrinkled test of the new material; how well does that predict
   the material's other cases? Every case is used as the anchor in turn.

On the three small carbon sets (2-5 cases each) the answer looked good;
on Krishnappa & Gururaja's 18 configurations of one material it is no:
``gamma_Y`` is not a material constant once the wrinkle's position and
width vary as well as its angle and depth.

The gate's ``gamma_Y`` is a fitted parameter, not the material's matrix
yield strain: the presets carry 0.26 and 0.62 for the same glass system,
against 0.02 on every material card.

Inputs come straight from the ledger: the peak angle, and the gate's own
penetration convention ``D/T = A / T`` with ``A`` the half-amplitude
(:func:`wrinklefe.core.penetration_gate.predict_from_geometry`).

Run::

    python validation/gate_transfer_study.py
"""
from __future__ import annotations

import dataclasses
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

LEDGER = REPO / "tests" / "test_validation" / "ledger.json"

#: The UD carbon datasets no shipped preset was fitted to.
CARBON_DATASETS = (
    ("shi_2025_ud", "H Shi (2025)"),
    ("thor_2021_ud", "J Thor (2021)"),
    ("pilato_2022", "K Pilato (2022)"),
    ("krishnappa_2024", "L Krishnappa (2024)"),
)


def carbon_cases() -> dict[str, list[dict]]:
    """``{label: [{case, theta_deg, dt, measured}]}`` from the ledger."""
    from wrinklefe.core.layup import parse_layup

    ledger = json.loads(LEDGER.read_text())
    out: dict[str, list[dict]] = {}
    for prefix, label in CARBON_DATASETS:
        ds = next(d for d in ledger["datasets"] if d["name"].startswith(prefix))
        thickness = len(parse_layup(ds["layup"])) * float(ds["ply_thickness_mm"])
        out[label] = [
            {"case": c["case"],
             "theta_deg": float(c["peak_angle_deg"]),
             "dt": float(c["amplitude_p2p_mm"]) / 2.0 / thickness,
             "measured": float(c["measured_kd"])}
            for c in ds["cases"]
        ]
    return out


def _err(pred: float, meas: float) -> float:
    return (pred - meas) / meas * 100.0


def fit_gamma_y(shape, theta_deg: float, dt: float, measured: float):
    """``gamma_Y`` that makes ``shape`` reproduce one measured case.

    Returns ``None`` when no ``gamma_Y`` can: the shape's penetration term
    is too small at this ``D/T`` for any angle floor to reach the
    measurement.
    """
    from scipy.optimize import brentq

    from wrinklefe.core.penetration_gate import penetration_gate_kd

    def residual(g):
        return penetration_gate_kd(
            theta_deg, dt, dataclasses.replace(shape, gamma_Y=g)) - measured

    try:
        return float(brentq(residual, 1e-4, 50.0))
    except ValueError:
        return None


def blind_errors(cases, preset) -> list[float]:
    from wrinklefe.core.penetration_gate import penetration_gate_kd

    return [_err(penetration_gate_kd(c["theta_deg"], c["dt"], preset),
                 c["measured"]) for c in cases]


def anchored_errors(cases, shape) -> tuple[list[float], list[float], int]:
    """Anchor on each case in turn; predict the others of the same set.

    Returns (errors of the predictions, fitted gamma_Y per anchor, number
    of anchors no gamma_Y could reproduce).
    """
    from wrinklefe.core.penetration_gate import penetration_gate_kd

    errs, gammas, failed = [], [], 0
    for k, anchor in enumerate(cases):
        g = fit_gamma_y(shape, anchor["theta_deg"], anchor["dt"],
                        anchor["measured"])
        if g is None:
            failed += 1
            continue
        gammas.append(g)
        fitted = dataclasses.replace(shape, gamma_Y=g)
        errs += [_err(penetration_gate_kd(c["theta_deg"], c["dt"], fitted),
                      c["measured"])
                 for i, c in enumerate(cases) if i != k]
    return errs, gammas, failed


def main() -> None:
    from wrinklefe.core.penetration_gate import (
        GATE_LI2024_MOULDED,
        GATE_LI2025_VACBAG,
    )

    data = carbon_cases()
    # The position factor is a property of the Li 2025 specimens; these
    # are mid-plane or whole-thickness wrinkles, so it is left out.
    presets = (("moulded", GATE_LI2024_MOULDED),
               ("vacuum-bag",
                dataclasses.replace(GATE_LI2025_VACBAG, position_q=None)))

    print("1. Blind: shipped presets, unchanged (signed error %)")
    for name, preset in presets:
        allerr = []
        for label, cases in data.items():
            e = blind_errors(cases, preset)
            allerr += e
            print(f"   {name:10s} {label:16s} "
                  + " ".join(f"{x:+6.1f}" for x in e))
        print(f"   {name:10s} {'all':16s} unsafe {sum(x > 0.5 for x in allerr)}"
              f"/{len(allerr)}, worst {max(allerr):+.1f}%")

    print("\n2. One-test calibration: re-fit gamma_Y on one case of a "
          "material, predict its other cases (every case as anchor)")
    for name, shape in presets:
        for label, cases in data.items():
            errs, gammas, failed = anchored_errors(cases, shape)
            if not errs:
                print(f"   {name:10s} {label:20s} no anchor reaches the data "
                      f"({failed} of {len(cases)} unreachable)")
                continue
            spread = max(gammas) / min(gammas)
            print(f"   {name:10s} {label:20s} {len(errs):3d} predictions: "
                  f"MAE {statistics.mean(abs(x) for x in errs):5.1f}%, "
                  f"range {min(errs):+6.1f}..{max(errs):+6.1f}%, "
                  f"> +15 %: {sum(x > 15 for x in errs):3d}; gamma_Y "
                  f"{min(gammas):.3f}-{max(gammas):.3f} ({spread:.1f}x)"
                  + (f"; {failed} unreachable" if failed else ""))


if __name__ == "__main__":
    main()
