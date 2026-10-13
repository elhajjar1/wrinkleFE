"""Manufacturing process as an input, and its out-of-scope warnings (#445).

The knockdown models have been validated on a few manufacturing routes
only; ``AnalysisConfig.manufacturing_process`` names the laminate's route,
and a route outside the validated scope (wet lay-up today) gets a warning
everywhere the bending-regime flag goes: the text summary, the results
JSON, the NCR summary, the CLI and the app. A penetration-gate preset
used outside the material system and route it was fitted to gets its
own warning. Neither changes a number.

These tests pin the scope (derived from the validation evidence, and kept
in step with the ledger), the warnings' claims, and every place the
warnings must reach.
"""

from __future__ import annotations

import dataclasses
import json
import statistics
import sys
from pathlib import Path

import pytest

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.cli import main as cli_main
from wrinklefe.core.material import MaterialLibrary
from wrinklefe.core.penetration_gate import (
    GATE_LI2024_MOULDED,
    GATE_LI2025_VACBAG,
    GateParameters,
)
from wrinklefe.core.process import (
    VALIDATED_PROCESSES,
    VALIDATION_EVIDENCE,
    ManufacturingProcess,
    gate_scope_caveat,
    parse_process,
    process_in_scope,
    process_scope_caveat,
)
from wrinklefe.io.export import build_analysis_summary, render_summary_markdown
from wrinklefe.io.results import SCHEMA_VERSION, results_to_dict

_ROOT = Path(__file__).resolve().parents[1]
_LEDGER = json.loads(
    (_ROOT / "tests" / "test_validation" / "ledger.json").read_text())
MP = ManufacturingProcess


def _config(process="unspecified", **kw):
    base = dict(
        amplitude=0.3, wavelength=10.0, width=10.0, angles=[0] * 16,
        ply_thickness=0.25, material=MaterialLibrary().get("IM7_8552"),
        analytical_only=True, manufacturing_process=process,
    )
    base.update(kw)
    return AnalysisConfig(**base)


def _run(cfg):
    return WrinkleAnalysis(cfg).run(analytical_only=True)


# ----------------------------------------------------------------------
# The input
# ----------------------------------------------------------------------


@pytest.mark.parametrize("value", [
    MP.WET_LAYUP, "wet_layup", "wet-layup", " Wet-Layup "])
def test_parse_process_accepts_every_spelling(value):
    assert parse_process(value) is MP.WET_LAYUP


def test_parse_process_rejects_anything_else():
    with pytest.raises(ValueError, match="wet_layup"):
        parse_process("hand")


def test_config_defaults_to_unspecified_and_normalises():
    assert AnalysisConfig().manufacturing_process == "unspecified"
    assert _config("wet-layup").manufacturing_process == "wet_layup"
    assert _config(MP.INFUSION).manufacturing_process == "infusion"
    with pytest.raises(ValueError, match="AnalysisConfig.manufacturing_process"):
        _config("hand")


def test_config_round_trips_and_old_files_still_load():
    cfg = _config("wet_layup")
    assert AnalysisConfig.from_dict(cfg.to_dict()).manufacturing_process == (
        "wet_layup")
    legacy = _config().to_dict()
    legacy.pop("manufacturing_process")
    assert AnalysisConfig.from_dict(legacy).manufacturing_process == (
        "unspecified")


# ----------------------------------------------------------------------
# The scope
# ----------------------------------------------------------------------


def test_scope_is_the_three_prepreg_routes_with_a_dataset():
    assert VALIDATED_PROCESSES == {
        MP.PREPREG_AUTOCLAVE, MP.PREPREG_PRESS, MP.PREPREG_VACUUM_BAG}
    for p in (MP.PREPREG_AUTOCLAVE, MP.PREPREG_PRESS, MP.PREPREG_VACUUM_BAG,
              MP.UNSPECIFIED):
        assert process_in_scope(p) and process_scope_caveat(p) is None
    for p in (MP.WET_LAYUP, MP.INFUSION, MP.RTM, MP.OTHER):
        assert not process_in_scope(p)
        assert p.label in process_scope_caveat(p)


def test_a_quarantined_dataset_does_not_widen_the_scope():
    wet = [e for e in VALIDATION_EVIDENCE if MP.WET_LAYUP in e.processes]
    assert wet and all(e.quarantined for e in wet)
    assert MP.WET_LAYUP not in VALIDATED_PROCESSES


def test_each_validated_route_has_its_evidence():
    """Autoclave and press have two datasets each (Elhajjar A/B, with
    Thor and Li 2024 respectively); vacuum bag has Li 2025 alone."""
    def datasets(p):
        return sorted(e.dataset.split()[0] for e in VALIDATION_EVIDENCE
                      if p in e.processes and not e.quarantined)

    assert datasets(MP.PREPREG_AUTOCLAVE) == ["A/B", "J"]
    assert datasets(MP.PREPREG_PRESS) == ["A/B", "E"]
    assert datasets(MP.PREPREG_VACUUM_BAG) == ["F"]


# The ledger's dataset prefix -> the evidence table's dataset letter.
_LEDGER_LETTER = {
    "li_2025": "F", "hsiao_daniel_1996": "G", "shi_2025_ud": "H",
    "shi_2025_md": "H", "calvo_2023": "I", "thor_2021_qi": "J",
    "thor_2021_ud": "J", "pilato_2022": "K", "krishnappa_2024": "L",
}


def test_ledger_process_entries_match_the_evidence_table():
    by_letter = {e.dataset.split()[0]: e for e in VALIDATION_EVIDENCE}
    for ds in _LEDGER["datasets"]:
        prefix = next(k for k in _LEDGER_LETTER if ds["name"].startswith(k))
        evidence = by_letter[_LEDGER_LETTER[prefix]]
        route = ds["process"]["route"]
        expected = [p.value for p in evidence.processes]
        assert route == (expected[0] if len(expected) == 1 else None), (
            ds["name"])
        assert ("quarantine" in ds) == evidence.quarantined, ds["name"]


_QUALITY_KEYS = {"fibre_volume_fraction", "porosity_pct", "test_standard",
                 "baseline_meets_standard", "pristine_failure_strain_pct",
                 "source"}


def test_every_ledger_dataset_carries_quality_metadata():
    """Issue #445: what each source reports about laminate quality, with
    null where it reports nothing, so the gaps are recorded too."""
    for ds in _LEDGER["datasets"]:
        q = ds["quality"]
        assert set(q) == _QUALITY_KEYS, ds["name"]
        assert q["source"], ds["name"]
        vf = q["fibre_volume_fraction"]
        assert vf is None or 0.3 < vf < 0.75, ds["name"]
    wet = next(d for d in _LEDGER["datasets"]
               if d["name"].startswith("krishnappa_2024"))["quality"]
    assert wet["baseline_meets_standard"] is False
    assert wet["porosity_pct"] == 4.2
    # The pristine strength over the measured modulus (98.1 GPa).
    assert wet["pristine_failure_strain_pct"] == pytest.approx(
        432.7 / 98100 * 100, abs=0.005)


def test_the_wet_layup_warning_numbers_are_the_ledger_numbers():
    """The analytical half of the wet lay-up note: the kink-band was
    conservative by about 56% on average on Dataset L (the FE half is
    pinned by tests/test_fe_strength_caveat.py)."""
    ds = next(d for d in _LEDGER["datasets"]
              if d["name"].startswith("krishnappa_2024"))
    errs = [(c["expected_analytical_kd"] - c["measured_kd"])
            / c["measured_kd"] * 100.0 for c in ds["cases"]]
    assert statistics.mean(errs) == pytest.approx(-56.0, abs=1.0)
    caveat = process_scope_caveat(MP.WET_LAYUP)
    assert "about 56% on average" in caveat
    assert "up to about 100%" in caveat


# ----------------------------------------------------------------------
# The gate
# ----------------------------------------------------------------------


def test_presets_record_their_fit():
    for gate, process in ((GATE_LI2024_MOULDED, MP.PREPREG_PRESS),
                          (GATE_LI2025_VACBAG, MP.PREPREG_VACUUM_BAG)):
        assert gate.fitted_materials == ("AC318_S6C10", "AC318_S6C10_vacbag")
        assert gate.fitted_process is process


def test_the_validated_gate_pairing_is_quiet():
    """Dataset F runs the vacuum-bag preset on the AC318_S6C10 card."""
    assert gate_scope_caveat(
        GATE_LI2025_VACBAG, "AC318_S6C10", MP.PREPREG_VACUUM_BAG) is None
    assert gate_scope_caveat(
        GATE_LI2024_MOULDED, "AC318_S6C10", MP.UNSPECIFIED) is None
    assert gate_scope_caveat(None, "IM7_8552", MP.WET_LAYUP) is None


def test_a_gate_outside_its_fit_is_flagged():
    other_material = gate_scope_caveat(
        GATE_LI2024_MOULDED, "IM7_8552", MP.UNSPECIFIED)
    assert "the material (IM7_8552) differs" in other_material
    other_process = gate_scope_caveat(
        GATE_LI2024_MOULDED, "AC318_S6C10", MP.PREPREG_VACUUM_BAG)
    assert "here the process (prepreg, vacuum-bag" in other_process
    assert "the material (" not in other_process
    both = gate_scope_caveat(GATE_LI2025_VACBAG, "IM7_8552", MP.WET_LAYUP)
    assert "and the process" in both and "differ." in both


def test_a_users_own_gate_is_not_second_guessed():
    own = GateParameters(gamma_Y=0.1, dt0=0.1, p=1.0, name="mine")
    assert gate_scope_caveat(own, "IM7_8552", MP.WET_LAYUP) is None


def test_the_gate_warning_numbers_are_the_transfer_study_numbers():
    sys.path.insert(0, str(_ROOT / "validation"))
    import gate_transfer_study as study

    data = study.carbon_cases()
    pooled = [c for k, v in data.items()
              if k not in study.quarantined_labels() for c in v]
    moulded = study.blind_errors(pooled, GATE_LI2024_MOULDED)
    vacbag = study.blind_errors(
        pooled, dataclasses.replace(GATE_LI2025_VACBAG, position_q=None))
    assert len(pooled) == 10
    assert sum(e > 0.5 for e in moulded) == 7
    assert sum(e > 0.5 for e in vacbag) == 10
    assert round(max(moulded)) == 97
    assert round(max(vacbag), -1) == 180
    text = gate_scope_caveat(GATE_LI2024_MOULDED, "IM7_8552", MP.UNSPECIFIED)
    assert "7 (moulded) and 10 (vacuum-bag) of 10 cases" in text
    assert "about 97% and 180%" in text


# ----------------------------------------------------------------------
# Where the warnings must reach
# ----------------------------------------------------------------------


def test_reporting_only_no_knockdown_changes():
    plain = _run(_config())
    wet = _run(_config("wet_layup", penetration_gate=GATE_LI2024_MOULDED))
    gated = _run(_config(penetration_gate=GATE_LI2024_MOULDED))
    assert wet.analytical_knockdown == gated.analytical_knockdown
    assert plain.process_out_of_scope is False
    assert plain.process_scope_caveat is None
    assert wet.process_out_of_scope and wet.gate_out_of_scope
    assert wet.manufacturing_process == "wet_layup"


def test_summary_names_the_process_and_warns():
    quiet = _run(_config()).summary()
    assert "Manufacturing process" not in quiet and "out of scope" not in quiet
    autoclave = _run(_config("prepreg_autoclave")).summary()
    assert "Manufacturing process:  prepreg, autoclave cure" in autoclave
    assert "WARNING" not in autoclave
    wet = _run(_config("wet_layup",
                       penetration_gate=GATE_LI2024_MOULDED)).summary()
    assert "WARNING (process out of scope)" in wet
    assert "WARNING (gate out of scope)" in wet


def test_results_json_carries_the_scope_block():
    data = results_to_dict(_run(_config("infusion")))
    assert SCHEMA_VERSION == "1.4"
    scope = data["scope"]
    assert scope["manufacturing_process"] == "infusion"
    assert scope["process_out_of_scope"] is True
    assert scope["process_scope_caveat"] == process_scope_caveat("infusion")
    assert scope["gate_out_of_scope"] is False
    assert data["config"]["manufacturing_process"] == "infusion"
    json.dumps(data)


def _ncr(defect_extra=None, **engineering):
    defect = {"amplitude_mm": 0.3, "wavelength_mm": 10.0, "width_mm": 10.0,
              "morphology": "graded", "loading": "compression",
              "ply_thickness_mm": 0.25, "n_plies": 16, "layup": [0] * 16,
              "material_name": "IM7_8552", **(defect_extra or {})}
    eng = {"analytical_knockdown": 0.6, "damage_index": 0.2,
           "analytical_strength_MPa": 700.0, **engineering}
    return build_analysis_summary(defect=defect, engineering=eng)


def test_ncr_summary_warns_ahead_of_the_numbers():
    summary = _ncr({"manufacturing_process": "wet_layup"})
    ea = summary["engineering_analysis"]
    assert summary["laminate"]["manufacturing_process"] == "wet_layup"
    assert ea["process_out_of_scope"] is True
    assert ea["process_scope_caveat"] == process_scope_caveat("wet_layup")
    md = render_summary_markdown(summary)
    assert "- Manufacturing process: wet lay-up" in md
    warning = md.index("> **Process outside the validated scope:**")
    assert warning < md.index("- Analytical knockdown")


def test_ncr_summary_carries_the_gate_warning_and_is_quiet_by_default():
    text = gate_scope_caveat(GATE_LI2024_MOULDED, "IM7_8552", MP.UNSPECIFIED)
    gated = _ncr(gate_scope_caveat=text)
    assert gated["engineering_analysis"]["gate_out_of_scope"] is True
    assert f"> **Gate outside its fit:** {text}" in render_summary_markdown(
        gated)
    legacy = _ncr()
    ea = legacy["engineering_analysis"]
    assert legacy["laminate"]["manufacturing_process"] == "unspecified"
    assert not ea["process_out_of_scope"] and not ea["gate_out_of_scope"]
    md = render_summary_markdown(legacy)
    assert "Manufacturing process" not in md and "validated scope" not in md


def test_cli_process_flag(capsys):
    cli_main(["analyze", "--analytical-only", "--process", "wet-layup"])
    out = capsys.readouterr().out
    assert "Manufacturing process:  wet lay-up" in out
    assert "WARNING (process out of scope)" in out


def test_cli_rejects_an_unknown_process(capsys):
    with pytest.raises(SystemExit) as exc:
        cli_main(["analyze", "--analytical-only", "--process", "hand"])
    assert exc.value.code == 2
    assert "--process" in capsys.readouterr().err


@pytest.mark.viz
def test_app_threads_the_process_and_flags_it():
    pytest.importorskip("streamlit", reason="Streamlit not installed.")
    sys.path.insert(0, str(_ROOT))
    import app

    assert app.DEFAULTS["sb_manufacturing_process"] == "unspecified"
    material = MaterialLibrary().get("IM7_8552").to_dict()
    items = {
        "amplitude": 0.3, "wavelength": 10.0, "width": 10.0,
        "morphology": "graded", "decay_floor": 0.0,
        "amplitude_profile": "constant",
        "amplitude_profile_decay_length": 5.0,
        "amplitude_profile_axis": "x", "loading": "compression",
        "ply_thickness": 0.25, "angles_tuple": tuple([0.0] * 16),
        "applied_strain": -0.01,
        "material_tuple": tuple(sorted(material.items())),
        "analytical_only": True,
    }
    default = app._run_analysis(tuple(sorted(items.items())))
    assert default["process_out_of_scope"] is False
    items["manufacturing_process"] = "wet_layup"
    payload = tuple(sorted(items.items()))
    assert app._config_from_payload(payload).manufacturing_process == (
        "wet_layup")
    result = app._run_analysis(payload)
    assert result["process_out_of_scope"] is True
    assert result["process_scope_caveat"] == process_scope_caveat("wet_layup")
