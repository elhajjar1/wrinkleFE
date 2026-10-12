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

Blind, the presets are unsafe on the carbon sets. With one test, the
moulded shape predicted Shi's and Pilato's other cases well, and failed
on Krishnappa & Gururaja's 18 configurations, where the fitted
``gamma_Y`` varies more than threefold within one material. That
dataset is **quarantined** (issue #445): a low-quality wet lay-up
laminate whose pristine baseline fails its test standard. So it is
reported on its own, not pooled, and the one-test question stays open
until a good-quality dataset varies wrinkle position. Section 3 shows
that even within one wrinkle position of that set, one test calibrates
poorly.

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


def quarantined_labels() -> set[str]:
    """Labels of the carbon datasets the ledger quarantines (issue #445)."""
    ledger = json.loads(LEDGER.read_text())
    out = set()
    for prefix, label in CARBON_DATASETS:
        ds = next(d for d in ledger["datasets"] if d["name"].startswith(prefix))
        if "quarantine" in ds:
            out.add(label)
    return out


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
    held_out = quarantined_labels()
    # The position factor is a property of the Li 2025 specimens; these
    # are mid-plane or whole-thickness wrinkles, so it is left out.
    presets = (("moulded", GATE_LI2024_MOULDED),
               ("vacuum-bag",
                dataclasses.replace(GATE_LI2025_VACBAG, position_q=None)))

    def tag(label):
        return label + " [Q]" if label in held_out else label

    print("1. Blind: shipped presets, unchanged (signed error %)")
    for name, preset in presets:
        pooled, held = [], []
        for label, cases in data.items():
            e = blind_errors(cases, preset)
            (held if label in held_out else pooled).extend(e)
            print(f"   {name:10s} {tag(label):24s} "
                  + " ".join(f"{x:+6.1f}" for x in e))
        print(f"   {name:10s} {'pooled (excl. [Q])':24s} unsafe "
              f"{sum(x > 0.5 for x in pooled)}/{len(pooled)}, "
              f"worst {max(pooled):+.1f}%")
        if held:
            print(f"   {name:10s} {'quarantined [Q]':24s} unsafe "
                  f"{sum(x > 0.5 for x in held)}/{len(held)}, "
                  f"worst {max(held):+.1f}%")

    print("\n2. One-test calibration: re-fit gamma_Y on one case of a "
          "material, predict its other cases (every case as anchor)")
    for name, shape in presets:
        for label, cases in data.items():
            errs, gammas, failed = anchored_errors(cases, shape)
            if not errs:
                print(f"   {name:10s} {tag(label):24s} no anchor reaches the "
                      f"data ({failed} of {len(cases)} unreachable)")
                continue
            spread = max(gammas) / min(gammas)
            print(f"   {name:10s} {tag(label):24s} {len(errs):3d} predictions: "
                  f"MAE {statistics.mean(abs(x) for x in errs):5.1f}%, "
                  f"range {min(errs):+6.1f}..{max(errs):+6.1f}%, "
                  f"> +15 %: {sum(x > 15 for x in errs):3d}; gamma_Y "
                  f"{min(gammas):.3f}-{max(gammas):.3f} ({spread:.1f}x)"
                  + (f"; {failed} unreachable" if failed else ""))

    print("\n3. One-test calibration within one wrinkle position of each "
          "position-varying set (moulded shape)")
    for label, cases in data.items():
        positions = position_groups(cases)
        if len(positions) < 2:
            continue
        for pos, group in positions.items():
            errs, gammas, _failed = anchored_errors(group, GATE_LI2024_MOULDED)
            print(f"   {tag(label):24s} {pos}: {len(errs):2d} predictions, "
                  f"MAE {statistics.mean(abs(x) for x in errs):5.1f}%, "
                  f"gamma_Y spread {max(gammas) / min(gammas):.1f}x")
    if held_out:
        print("\n   [Q] = quarantined dataset (issue #445): reported on its "
              "own, not pooled.")


def position_groups(cases) -> dict[str, list[dict]]:
    """Group a set's cases by wrinkle position, from case names like
    ``L-WM-2-1`` (Dataset L's WL/WM/WR). Sets without one share a group."""
    out: dict[str, list[dict]] = {}
    for c in cases:
        parts = c["case"].split("-")
        pos = parts[1] if len(parts) == 4 and parts[1].startswith("W") else "-"
        out.setdefault(pos, []).append(c)
    return out


if __name__ == "__main__":
    main()
