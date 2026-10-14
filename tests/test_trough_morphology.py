"""One-sided surface troughs and the morphology that can see them (#461).

The load-path eccentricity of issue #439 is measured on the modelled
geometry. ``graded`` with decay floor 0 keeps both outer surfaces flat,
so a one-sided surface trough modelled that way reads as an embedded
wrinkle and gets no bending warning. ``tool_flat`` with only the tool face
pinned is flagged once the trough is deep enough. No detector was added
on ``graded``, because its inputs overlap the measured embedded wrinkles'.
Users are steered to ``tool_flat`` instead, and the validation ledger
scores Shi's troughs (Dataset H) on ``tool_flat`` too (issue #461, part
2); ``graded`` was its recipe before.

These tests pin that evidence (the numbers ``docs/interpreting_results.md``
quotes), the no-false-alarm property on the embedded datasets, and the
steering text where the morphology is chosen (CLI and app).
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

import pytest

from wrinklefe.analysis import WrinkleAnalysis
from wrinklefe.core.eccentricity import (
    BENDING_REGIME_CAVEAT,
    ECCENTRICITY_REGIME_THRESHOLD,
    ONE_SIDED_TROUGH_ADVICE,
)

_ROOT = Path(__file__).resolve().parents[1]
_LEDGER = json.loads(
    (_ROOT / "tests" / "test_validation" / "ledger.json").read_text())


def _dataset(prefix):
    return next(d for d in _LEDGER["datasets"] if d["name"].startswith(prefix))


def _case_config(prefix, name, **overrides):
    """The ledger's knockdown recipe for one case, with overrides."""
    sys.path.insert(0, str(_ROOT / "scripts"))
    from validate import case_config

    ds = _dataset(prefix)
    case = next(c for c in ds["cases"] if c["case"] == name)
    cfg = case_config(ds, case)
    if overrides:
        cfg = dataclasses.replace(cfg, analytical_only=True, **overrides)
    return cfg, case


#: Shi's troughs as the ledger modelled them before issue #461 (part 2):
#: an embedded ``graded`` profile at the default mid interfaces. The
#: ledger recipe itself is now the one-sided ``tool_flat`` trough.
_AS_GRADED = {"morphology": "graded"}


def _run(cfg):
    return WrinkleAnalysis(cfg).run(analytical_only=True)


def _err(pred, meas):
    return (pred - meas) / meas * 100.0


# ----------------------------------------------------------------------
# The gap, and what the steered morphology reports (the docs table)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prefix", "case", "graded_et", "graded_err", "trough_et", "trough_err",
     "flagged"),
    [
        ("shi_2025_ud", "H-UD-10", 0.000, 0.2, 0.050, -58.9, False),
        # e/t = depth / (2 T) on UD: 0.10001, just on the threshold.
        ("shi_2025_ud", "H-UD-20", 0.000, 11.8, 0.100, -68.4, True),
        ("shi_2025_ud", "H-UD-30", 0.000, 25.9, 0.150, -71.0, True),
        ("shi_2025_md", "H-MD-10", 0.012, 5.8, 0.058, -42.1, False),
        ("shi_2025_md", "H-MD-20", 0.022, 55.6, 0.117, -22.4, True),
        ("shi_2025_md", "H-MD-30", 0.030, 115.2, 0.175, 5.0, True),
    ],
)
def test_shi_trough_is_flagged_on_tool_flat_and_not_on_graded(
        prefix, case, graded_et, graded_err, trough_et, trough_err, flagged):
    trough_cfg, meta = _case_config(prefix, case)
    measured = float(meta["measured_kd"])

    graded_cfg, _ = _case_config(prefix, case, **_AS_GRADED)
    graded = _run(graded_cfg)
    assert graded.config.morphology == "graded"
    assert graded.load_path_eccentricity_ratio == pytest.approx(
        graded_et, abs=5e-4)
    assert not graded.bending_dominated
    assert "WARNING (bending-dominated" not in graded.summary()
    assert _err(graded.analytical_knockdown, measured) == pytest.approx(
        graded_err, abs=0.05)

    # The ledger recipe (issue #461, part 2), so these are the pinned
    # knockdowns the validation scores.
    trough = _run(trough_cfg)
    assert trough.config.morphology == "tool_flat"
    assert trough.config.surface_pocket_side == "bottom"
    assert trough.analytical_knockdown == pytest.approx(
        float(meta["expected_analytical_kd"]), abs=5e-7)
    assert trough.load_path_eccentricity_ratio == pytest.approx(
        trough_et, abs=5e-4)
    assert trough.bending_dominated is flagged
    # One face pinned flat: never a whole-thickness wave, so no bound and,
    # when flagged, the regime caveat's advice to run FE.
    assert not trough.whole_thickness_wave
    assert not trough.bending_bound_applied
    assert _err(trough.analytical_knockdown, measured) == pytest.approx(
        trough_err, abs=0.05)
    summary = " ".join(trough.summary().split())
    assert ("WARNING (bending-dominated):" in summary) is flagged
    if flagged:
        # The regime caveat (no bound, prefer FE), not the bound's.
        assert " ".join(BENDING_REGIME_CAVEAT.split()[:7]) in summary
        assert "bending bound applied" not in summary


def test_the_regime_caveat_quotes_the_flagged_troughs():
    """``BENDING_REGIME_CAVEAT`` is what a flagged one-sided trough gets.
    Its analytical range is the four flagged Shi troughs on their ledger
    recipe. Its FE range is those troughs' first-ply retention, section
    2b of ``validation/strength_error_summary.py`` (H-UD-20/30 -36.7 /
    -39.9 %, H-MD-20/30 -7.4 / -0.9 %), which needs FE and is not
    re-run here."""
    errs = []
    for prefix in ("shi_2025_ud", "shi_2025_md"):
        for case in _dataset(prefix)["cases"]:
            res = _run(_case_config(prefix, case["case"])[0])
            if res.bending_dominated:
                assert not res.whole_thickness_wave
                errs.append(_err(res.analytical_knockdown,
                                 float(case["measured_kd"])))
    assert len(errs) == 4
    assert min(errs) == pytest.approx(-71.0, abs=0.05)
    assert max(errs) == pytest.approx(5.0, abs=0.05)
    caveat = " ".join(BENDING_REGIME_CAVEAT.split())
    assert "about 70% conservative to 5% unsafe" in caveat


@pytest.mark.parametrize("case", ["H-UD-10", "H-UD-20", "H-UD-30"])
def test_on_ud_graded_reads_zero_and_tool_flat_reads_half_the_depth(case):
    """A UD section's centroid is midway between its outer surfaces, so
    ``graded`` (both flat) reads exactly zero at any depth and the
    one-sided trough reads depth / (2 T)."""
    cfg, _ = _case_config("shi_2025_ud", case)
    thickness = len(cfg.angles) * cfg.ply_thickness
    graded = _run(_case_config("shi_2025_ud", case, **_AS_GRADED)[0])
    assert graded.load_path_eccentricity_mm == pytest.approx(0.0, abs=1e-9)
    trough = _run(cfg)
    assert trough.load_path_eccentricity_ratio == pytest.approx(
        cfg.amplitude / (2.0 * thickness), rel=1e-3)


@pytest.mark.parametrize(
    ("label", "overrides", "flagged", "whole_thickness"),
    [
        ("uniform", {"morphology": "uniform"}, True, True),
        ("graded, decay centred on the top surface",
         {**_AS_GRADED, "wrinkle_z_position": 1.0}, True, False),
        ("graded, decay centred just below it",
         {**_AS_GRADED, "wrinkle_z_position": 0.9}, False, False),
        ("tool_flat on the top face",
         {"morphology": "tool_flat", "surface_pocket_side": "top",
          "surface_transition_plies": 10}, True, False),
        ("tool_flat pinned on both faces",
         {"morphology": "tool_flat", "surface_pocket_side": "both",
          "surface_transition_plies": 10}, False, False),
        ("stack", {"morphology": "stack"}, False, False),
        ("convex", {"morphology": "convex"}, False, False),
        ("concave", {"morphology": "concave"}, False, False),
    ],
)
def test_other_morphologies_for_the_same_trough(label, overrides, flagged,
                                               whole_thickness):
    """The docs' "Other choices" list, on Shi's 20% multidirectional
    trough (+55.6% unsafe on ``graded``): a morphology that frees one
    outer surface is flagged; one that keeps both flat is not."""
    cfg, _ = _case_config("shi_2025_md", "H-MD-20", **overrides)
    res = _run(cfg)
    assert res.bending_dominated is flagged, label
    assert res.whole_thickness_wave is whole_thickness, label
    assert res.bending_bound_applied is (flagged and whole_thickness), label


# ----------------------------------------------------------------------
# No false alarms on genuinely embedded wrinkles
# ----------------------------------------------------------------------


def _embedded_ledger_cases():
    """Every ledger compression case modelled as an embedded wrinkle:
    ``graded``, both outer surfaces flat (Hsiao & Daniel's graded coupon,
    Li 2025, Krishnappa's mid-plane WM strips)."""
    out = []
    for ds in _LEDGER["datasets"]:
        if ds["loading"] != "compression" or ds["name"].startswith("shi"):
            continue
        for case in ds["cases"]:
            if case.get("morphology", ds["morphology"]) == "graded":
                out.append((ds["name"].split("_")[0], case["case"]))
    return out


_EMBEDDED_LEDGER = _embedded_ledger_cases()


def test_the_embedded_ledger_set_is_the_documented_one():
    names = {f"{p}:{c}" for p, c in _EMBEDDED_LEDGER}
    assert "hsiao:HD-graded" in names
    assert {f"li:{c}" for c in ("S-M-1", "S-M-2", "S-M-3", "S-M-4",
                                "S-M-5", "S-A-2")} <= names
    assert {f"krishnappa:L-WM-{w}-{n}" for w in (2, 4)
            for n in (1, 2, 3)} <= names
    assert len(names) == 13


@pytest.mark.parametrize(("prefix", "case"), _EMBEDDED_LEDGER)
def test_embedded_ledger_wrinkles_are_not_flagged(prefix, case):
    res = _run(_case_config(prefix, case)[0])
    # All UD: flat surfaces put the centroid exactly mid-thickness.
    assert res.load_path_eccentricity_mm == pytest.approx(0.0, abs=1e-9)
    assert not res.bending_dominated
    assert "WARNING (bending-dominated" not in res.summary()


def _head_to_head_embedded():
    """Datasets C (Mukhopadhyay, ``graded``), D (Wang, ``convex`` and
    ``concave``) and E (Li 2024, ``graded``): the embedded compression
    wrinkles scored outside the ledger."""
    sys.path.insert(0, str(_ROOT / "validation"))
    import strength_error_summary as ses

    out = []
    for label, cases in ses._md_cases().items():
        if label in ("C Mukhopadhyay comp", "D Wang conc/conv"):
            out.extend((label, c, cfg) for c, _t, _kd, cfg in cases)
    out.extend(("E Li2024 UD comp", c, cfg)
               for c, _t, _dt, _kd, cfg in ses._li2024_configs())
    return out


def test_embedded_head_to_head_wrinkles_are_not_flagged():
    cases = _head_to_head_embedded()
    assert len(cases) == 3 + 4 + 9
    for label, case, cfg in cases:
        res = _run(cfg)
        assert res.load_path_eccentricity_ratio < 0.03, (label, case)
        assert not res.bending_dominated, (label, case)


def test_no_input_threshold_separates_the_trough_from_embedded_wrinkles():
    """Why there is no detector on ``graded``: measured embedded wrinkles
    reach Shi's 20% trough on each input a user gives (docs and
    VALIDATION.md quote these)."""
    shi, _ = _case_config("shi_2025_md", "H-MD-20")
    krishnappa, _ = _case_config("krishnappa", "L-WM-4-3")
    li, _ = _case_config("li_2025", "S-M-3")

    def a_over_t(cfg):
        return cfg.amplitude / (len(cfg.angles) * cfg.ply_thickness)

    def a_over_l(cfg):
        return cfg.amplitude / cfg.wavelength

    def angle(cfg):
        return math.degrees(math.atan(2 * math.pi * a_over_l(cfg)))

    assert a_over_t(shi) == pytest.approx(0.200, abs=5e-4)
    assert a_over_t(krishnappa) == pytest.approx(0.207, abs=5e-4)
    assert a_over_l(shi) == pytest.approx(0.069, abs=5e-4)
    assert a_over_l(li) == pytest.approx(0.093, abs=5e-4)
    assert angle(shi) == pytest.approx(23.5, abs=0.05)
    assert angle(li) == pytest.approx(30.2, abs=0.05)
    for metric in (a_over_t, a_over_l, angle):
        embedded = max(metric(krishnappa), metric(li))
        assert embedded > metric(shi), metric.__name__
    # Neither embedded case is bending-dominated (checked above), and the
    # trough entered as graded reads as embedded too.
    shi_graded, _ = _case_config("shi_2025_md", "H-MD-20", **_AS_GRADED)
    assert _run(shi_graded).load_path_eccentricity_ratio < (
        ECCENTRICITY_REGIME_THRESHOLD)


# ----------------------------------------------------------------------
# The steering, where the morphology is chosen
# ----------------------------------------------------------------------


def test_the_advice_names_both_morphologies_and_the_tool_face():
    assert "'tool_flat'" in ONE_SIDED_TROUGH_ADVICE
    assert "'graded'" in ONE_SIDED_TROUGH_ADVICE
    assert "not 'both'" in ONE_SIDED_TROUGH_ADVICE
    assert "issue #461" in ONE_SIDED_TROUGH_ADVICE
    # argparse %-formats help strings.
    assert "%" not in ONE_SIDED_TROUGH_ADVICE


def test_cli_morphology_help_carries_the_advice(capsys):
    from wrinklefe.cli import main as cli_main

    with pytest.raises(SystemExit):
        cli_main(["analyze", "--help"])
    out = capsys.readouterr().out
    # argparse re-wraps (and may break at hyphens): compare without
    # whitespace.
    assert "".join(ONE_SIDED_TROUGH_ADVICE.split()) in "".join(out.split())


@pytest.mark.viz
def test_app_morphology_help_carries_the_advice():
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(_ROOT / "app.py"), default_timeout=30)
    at.session_state["expert_mode"] = True
    at.session_state["sb_morphology"] = "graded"
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    morphology = next(sb for sb in at.selectbox if sb.key == "sb_morphology")
    assert ONE_SIDED_TROUGH_ADVICE in morphology.help
    assert "*tool_flat*" in morphology.help
    captions = " ".join(c.value for c in at.caption)
    assert "one-sided surface trough" in captions
    assert "**tool_flat**" in captions
