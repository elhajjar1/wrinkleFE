"""The analytical delamination-onset estimate is flagged as unvalidated (#463).

Against the measured damage onset of Elhajjar & Shams (2016), Dataset M,
the estimate is above the measurement on all nine multidirectional
coupons and 95-99% below it on the UD ones. The UD collapse comes from
the notional interface flaw, which spans the whole block of adjacent
0-degree plies: on a ``[0]16`` laminate that is the full thickness.
These tests pin the cause and that the caveat travels with the value
everywhere it leaves the package: the results object, the results JSON,
the critical-value search and the CLI.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from wrinklefe.analysis import (
    ONSET_ESTIMATE_CAVEAT,
    AnalysisConfig,
    WrinkleAnalysis,
    _max_consecutive_zero_plies,
)
from wrinklefe.io.results import SCHEMA_VERSION, results_to_dict

_ROOT = Path(__file__).resolve().parents[1]
_LEDGER = json.loads(
    (_ROOT / "tests" / "test_validation" / "ledger.json").read_text())
sys.path.insert(0, str(_ROOT / "scripts"))
import validate  # noqa: E402


def _m_case(half: str, index: int = 0):
    dataset = next(d for d in _LEDGER["datasets"]
                   if d["name"].startswith(f"elhajjar_shams_2016_{half}"))
    case = dataset["cases"][index]
    return validate.case_config(dataset, case), case


def _run(cfg):
    return WrinkleAnalysis(cfg).run(analytical_only=True)


# ----------------------------------------------------------------------
# The cause
# ----------------------------------------------------------------------

def test_ud_flaw_spans_the_whole_thickness():
    """The flaw is the run of adjacent 0-deg plies: all 16 on the UD
    coupons, a short block on the multidirectional ones."""
    ud_cfg, _ = _m_case("ud")
    md_cfg, _ = _m_case("md")
    assert _max_consecutive_zero_plies(list(ud_cfg.angles)) == len(ud_cfg.angles)
    assert _max_consecutive_zero_plies(list(md_cfg.angles)) <= 2


def test_ud_onset_collapses_and_md_onset_overshoots():
    ud_cfg, ud_case = _m_case("ud")
    md_cfg, md_case = _m_case("md")
    ud = _run(ud_cfg).analytical_onset_knockdown
    md = _run(md_cfg).analytical_onset_knockdown
    assert ud is not None and md is not None
    assert ud < 0.01 * 2  # about 0.2% of pristine strength
    assert ud < 0.05 * ud_case["measured_onset_kd"]
    assert md > md_case["measured_onset_kd"]  # unsafe


# ----------------------------------------------------------------------
# The flag on the results object and in the results JSON
# ----------------------------------------------------------------------

def test_tension_result_carries_the_caveat():
    res = _run(_m_case("md")[0])
    assert res.analytical_onset_knockdown is not None
    assert res.analytical_onset_caveat == ONSET_ESTIMATE_CAVEAT
    assert "#463" in ONSET_ESTIMATE_CAVEAT
    assert "acceptance limit" in ONSET_ESTIMATE_CAVEAT


def test_compression_result_has_no_onset_and_no_caveat():
    res = _run(AnalysisConfig())
    assert res.analytical_onset_knockdown is None
    assert res.analytical_onset_caveat is None


def test_results_json_flags_the_onset():
    assert tuple(int(x) for x in SCHEMA_VERSION.split(".")) >= (1, 7)
    tension = results_to_dict(_run(_m_case("ud")[0]))["analytical"]
    assert tension["analytical_onset_validated"] is False
    assert tension["analytical_onset_caveat"] == ONSET_ESTIMATE_CAVEAT
    compression = results_to_dict(_run(AnalysisConfig()))["analytical"]
    assert compression["analytical_onset_knockdown"] is None
    assert compression["analytical_onset_validated"] is None
    assert compression["analytical_onset_caveat"] is None


# ----------------------------------------------------------------------
# The critical-value search and the CLI
# ----------------------------------------------------------------------

def test_onset_search_warns_and_returns_the_caveat(caplog):
    from wrinklefe.goalseek import find_critical_value

    cfg = _m_case("md")[0]
    # The onset estimate is undefined at zero amplitude, so the scan
    # starts above it (the default range begins at zero).
    with caplog.at_level(logging.WARNING, logger="wrinklefe.goalseek"):
        result = find_critical_value(
            cfg, objective="onset_knockdown", target_knockdown=0.3,
            scan_points=5, analytical_only=True,
            bracket=(0.2 * cfg.amplitude, 2.0 * cfg.amplitude),
        )
    assert result.caveat == ONSET_ESTIMATE_CAVEAT
    assert any("#463" in rec.getMessage() for rec in caplog.records)
    assert ONSET_ESTIMATE_CAVEAT in result.summary()


def test_knockdown_search_has_no_caveat():
    from wrinklefe.goalseek import find_critical_value

    result = find_critical_value(
        AnalysisConfig(), target_knockdown=0.8, scan_points=5,
        analytical_only=True,
    )
    assert result.caveat is None


def test_cli_prints_and_exports_the_caveat(capsys, tmp_path):
    from wrinklefe.cli import main as cli_main
    from wrinklefe.goalseek import CriticalValueResult, GoalSeekEvaluation

    rows = [
        GoalSeekEvaluation(
            index=i, value=v, objective=o, knockdown=o,
            strength_MPa=1200.0 * o, residual=o - 0.3, phase="scan",
            runtime_s=0.0,
        )
        for i, (v, o) in enumerate([(0.0, 1.0), (0.1, 0.5), (0.2, 0.2)])
    ]
    fake = CriticalValueResult(
        parameter="amplitude", objective_name="onset_knockdown", target=0.3,
        target_is_strength=False, direction="decreasing",
        status="converged", message="", critical_value=0.15,
        critical_value_root=0.151, achieved_objective=0.3001,
        achieved_knockdown=0.6, achieved_strength_MPa=720.0,
        critical_config=AnalysisConfig(amplitude=0.15),
        bounds=(0.0, 2.0), bounds_source=("zero", "laminate_thickness"),
        scan_values=[r.value for r in rows],
        scan_objectives=[r.objective for r in rows],
        sign_changes=1, log_spaced=True, pristine_reference_MPa=1200.0,
        evaluations=rows, n_evaluations=len(rows), rtol=1e-3,
        analytical_only=True, runtime_s=0.1, caveat=ONSET_ESTIMATE_CAVEAT,
    )
    out_json = tmp_path / "critical.json"
    with patch("wrinklefe.goalseek.find_critical_value",
               new=lambda base_config, **kw: fake), \
            pytest.raises(SystemExit):
        cli_main([
            "critical", "--target-knockdown", "0.3",
            "--objective", "onset_knockdown", "--output-json", str(out_json),
        ])
    assert "CAVEAT: " + ONSET_ESTIMATE_CAVEAT in capsys.readouterr().out
    assert ONSET_ESTIMATE_CAVEAT in out_json.read_text()


def test_cli_objective_help_says_unvalidated(capsys):
    from wrinklefe.cli import main as cli_main

    with pytest.raises(SystemExit):
        cli_main(["critical", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "unvalidated" in help_text
