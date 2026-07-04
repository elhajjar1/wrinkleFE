"""Validation-ledger regression harness (issue #254, deliverable 1).

Recomputes every case in ``ledger.json`` and asserts the analytical
knockdown matches the pinned baseline within the ledger's tolerance —
the committed, reproducible harness whose absence forced the revert of
the graded-decay fix (commit ``00584b4``). Also documents the two #254
defects as strict xfails so that fixing them flips these tests loudly.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis

_HERE = Path(__file__).resolve().parent
_LEDGER = json.loads((_HERE / "ledger.json").read_text())

_spec = importlib.util.spec_from_file_location(
    "validate", _HERE.parents[1] / "scripts" / "validate.py"
)
validate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validate)


def _all_cases():
    for dataset in _LEDGER["datasets"]:
        for case in dataset["cases"]:
            yield pytest.param(
                dataset, case, id=f"{dataset['name']}-{case['case']}"
            )


def _modulus_cases():
    for dataset in _LEDGER["datasets"]:
        for case in dataset["cases"]:
            if "expected_analytical_modulus_kd" in case:
                yield pytest.param(
                    dataset, case, id=f"{dataset['name']}-{case['case']}"
                )


@pytest.mark.parametrize(("dataset", "case"), list(_all_cases()))
def test_case_matches_pinned_baseline(dataset, case):
    predicted = validate.run_case(dataset, case)
    pinned = case["expected_analytical_kd"]
    tol = _LEDGER["rel_tolerance"]
    assert predicted == pytest.approx(pinned, rel=tol), (
        f"{case['case']} analytical KD drifted from the pinned baseline "
        f"({predicted} vs {pinned}). If intentional, re-pin via "
        f"'python scripts/validate.py --update' and explain the move."
    )


@pytest.mark.parametrize(("dataset", "case"), list(_modulus_cases()))
def test_modulus_matches_pinned_baseline(dataset, case):
    """Pin the closed-form CLT modulus knockdown (#324) for the UD datasets.

    A deliberate change to ``_profile_modulus_knockdown`` (or the recipe
    that feeds it) moves ``analytical_modulus_knockdown`` and is caught
    here, the stiffness counterpart of the strength guard above (#326).
    """
    _strength, predicted = validate.run_case_both(dataset, case)
    pinned = case["expected_analytical_modulus_kd"]
    tol = _LEDGER["rel_tolerance"]
    assert predicted == pytest.approx(pinned, rel=tol), (
        f"{case['case']} analytical MODULUS KD drifted from the pinned "
        f"baseline ({predicted} vs {pinned}). If intentional, re-pin via "
        f"'python scripts/validate.py --update' and explain the move."
    )
    # The UD modulus path must produce a genuine sub-unity knockdown
    # (a multidirectional layup would silently collapse it to 1.0).
    assert 0.0 < predicted <= 1.0


def test_validate_script_passes():
    """The one-command harness exits 0 from a clean checkout."""
    assert validate.main([]) == 0


# --------------------------------------------------------------------------- #
# Issue #254 — decay_floor symmetric under sign-flipped load
# --------------------------------------------------------------------------- #


def _graded_kd(loading: str, decay_floor: float) -> float:
    # Quasi-isotropic layup on a config whose through-thickness Gaussian
    # actually decays inside the laminate: 24 plies (T = 4.39 mm) with an
    # explicit 0.5 mm decay scale, so the surface plies see raw ~ 0 and
    # the floor governs how much angle they retain.
    cfg = AnalysisConfig(
        amplitude=0.75, wavelength=12.9, width=12.9,
        morphology="graded", decay_floor=decay_floor, loading=loading,
        angles=[0.0, 45.0, -45.0, 90.0] * 3
        + [90.0, -45.0, 45.0, 0.0] * 3,
        through_thickness_decay_scale=0.5,
    )
    return float(
        WrinkleAnalysis(cfg).run(analytical_only=True).analytical_knockdown
    )


@pytest.mark.parametrize("loading", ["compression", "tension"])
def test_decay_floor_effective_under_both_loadings_issue_254(loading):
    """decay_floor produces the same envelope semantics under a
    sign-flipped load: raising the floor retains more misalignment in
    the outer plies and monotonically lowers the knockdown, in tension
    AND compression. (Before the #254 fix, compression ignored the
    floor entirely while tension honored it.)"""
    kds = [_graded_kd(loading, f) for f in (0.0, 0.5, 1.0)]
    assert kds[0] > kds[1] > kds[2], (
        f"{loading}: KD must decrease monotonically as decay_floor "
        f"retains more angle, got {kds}"
    )
    # The floor must have a material effect, not a numerical whisper.
    assert kds[0] - kds[2] > 0.01


def test_decay_floor_one_disables_compression_decay():
    """floor=1.0 means every ply keeps the full angle — identical to a
    decay scale so large the Gaussian never drops below ~1."""
    kd_floor1 = _graded_kd("compression", 1.0)
    cfg = AnalysisConfig(
        amplitude=0.75, wavelength=12.9, width=12.9,
        morphology="graded", decay_floor=0.0, loading="compression",
        angles=[0.0, 45.0, -45.0, 90.0] * 3
        + [90.0, -45.0, 45.0, 0.0] * 3,
        through_thickness_decay_scale=1.0e6,
    )
    kd_nodecay = float(
        WrinkleAnalysis(cfg).run(analytical_only=True).analytical_knockdown
    )
    assert kd_floor1 == pytest.approx(kd_nodecay, rel=1e-9)


def test_ledger_recipe_round_trips_config():
    """The ledger carries everything needed to rebuild each config."""
    dataset = _LEDGER["datasets"][0]
    case = dataset["cases"][0]
    cfg = validate.case_config(dataset, case)
    assert cfg.amplitude == pytest.approx(case["amplitude_p2p_mm"] / 2.0)
    assert cfg.wavelength == pytest.approx(case["wavelength_mm"])
    assert len(cfg.angles) == 14 and set(cfg.angles) == {0.0}
    assert cfg.material.name == dataset["material"]


def test_replace_preserves_ledger_predictions():
    """dataclasses.replace on a ledger config does not perturb the KD
    (guards the sweep/convergence helpers that clone configs)."""
    dataset = _LEDGER["datasets"][0]
    case = dataset["cases"][0]
    cfg = validate.case_config(dataset, case)
    kd1 = WrinkleAnalysis(cfg).run(analytical_only=True).analytical_knockdown
    kd2 = WrinkleAnalysis(replace(cfg)).run(
        analytical_only=True
    ).analytical_knockdown
    assert kd1 == pytest.approx(kd2, rel=1e-12)


# --------------------------------------------------------------------------- #
# Issue #161 — penetration-gate strength path (UD amplitude effect)
# --------------------------------------------------------------------------- #


def _gate_cases():
    for dataset in _LEDGER["datasets"]:
        if "penetration_gate" not in dataset:
            continue
        for case in dataset["cases"]:
            yield pytest.param(
                dataset, case, id=f"{dataset['name']}-{case['case']}"
            )


@pytest.mark.parametrize(("dataset", "case"), list(_gate_cases()))
def test_gate_matches_pinned_baseline(dataset, case):
    """Pin the calibrated penetration-gate knockdown (issue #161).

    The gate is the package's best UD strength predictor — the only
    analytical path sensitive to amplitude and through-thickness
    position independently of the peak angle. Before this guard it was
    entirely unpinned: a drift in the gate formula, the preset
    constants, or the position factor was invisible to CI.
    """
    predicted = validate.run_case_gate(dataset, case)
    pinned = case["expected_gate_kd"]
    tol = _LEDGER["rel_tolerance"]
    assert predicted == pytest.approx(pinned, rel=tol), (
        f"{case['case']} penetration-gate KD drifted from the pinned "
        f"baseline ({predicted} vs {pinned}). If intentional, re-pin via "
        f"'python scripts/validate.py --update' and explain the move."
    )


def _gate_kd_by_case() -> dict:
    dataset = next(
        d for d in _LEDGER["datasets"] if "penetration_gate" in d
    )
    return {
        c["case"]: (validate.run_case_gate(dataset, c), c["measured_kd"])
        for c in dataset["cases"]
    }


def test_issue_161_amplitude_trend_at_constant_angle():
    """The #161 acceptance criterion, as a permanent regression guard.

    S-M-2 / S-M-4 / S-M-5 share the same 20-degree peak angle but span
    amplitude 1.5 / 1.0 / 0.5 mm (measured KD 0.629 / 0.943 / 1.000 —
    a ~60% strength spread invisible to any angle-only model). The
    penetration gate must reproduce each within +/-15% relative KD and
    preserve the monotonic amplitude ordering.
    """
    kd = _gate_kd_by_case()
    trio = ["S-M-2", "S-M-4", "S-M-5"]
    for name in trio:
        pred, meas = kd[name]
        assert abs(pred - meas) / meas <= 0.15, (
            f"{name}: gate KD {pred:.3f} vs measured {meas:.3f} exceeds "
            f"the +/-15% acceptance band of issue #161"
        )
    # Larger amplitude at the same angle => lower strength.
    assert kd["S-M-2"][0] < kd["S-M-4"][0] < kd["S-M-5"][0]


def test_issue_161_angle_trend_not_degraded():
    """Companion guard: the angle series S-M-1/2/3 (10/20/30 degrees at
    fixed amplitude 1.5 mm) stays monotonic under the gate, and every
    Li 2025 case stays within the +/-20% parity band the README claims
    for the UD datasets."""
    kd = _gate_kd_by_case()
    assert kd["S-M-1"][0] > kd["S-M-2"][0] > kd["S-M-3"][0]
    for name, (pred, meas) in kd.items():
        assert abs(pred - meas) / meas <= 0.20, (
            f"{name}: gate KD {pred:.3f} vs measured {meas:.3f} outside "
            f"the +/-20% parity band"
        )


def test_gate_position_factor_reproduces_near_surface_case():
    """S-A-2 is S-M-2's geometry at z = 10/14: the ledger's z_frac plus
    the gate's position factor P(z) must reproduce the measured
    near-surface mildness (KD 0.981 vs mid-plane 0.629)."""
    kd = _gate_kd_by_case()
    pred_mid, _ = kd["S-M-2"]
    pred_surf, meas_surf = kd["S-A-2"]
    assert pred_surf > pred_mid + 0.2
    assert pred_surf == pytest.approx(meas_surf, abs=0.02)
