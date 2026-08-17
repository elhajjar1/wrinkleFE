"""Tests for the ``wrinklefe`` command-line interface.

These tests focus on how CLI flags map into :class:`AnalysisConfig` and how
they propagate into :meth:`WrinkleAnalysis.run` and the comparison/sweep
helpers. To keep them cheap, we patch the engine entry points and capture
the configs/kwargs the CLI hands them.
"""

from __future__ import annotations

import functools
from unittest.mock import MagicMock, patch

import pytest

from wrinklefe.cli import main as cli_main

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _stub_analysis_run():
    """Build a context manager that stubs ``WrinkleAnalysis.run``.

    Returns a tuple ``(patcher, captured)`` where ``captured`` is a dict
    that, after the CLI call, holds ``"config"`` (the AnalysisConfig the
    handler built) and ``"analytical_only"`` (the kwarg passed to
    :meth:`WrinkleAnalysis.run`).
    """
    captured: dict = {}

    def fake_run(self, analytical_only=None):
        captured["analytical_only"] = analytical_only
        captured["config"] = self.config
        result = MagicMock()
        result.summary.return_value = "<stubbed summary>"
        return result

    return captured, patch(
        "wrinklefe.analysis.WrinkleAnalysis.run", new=fake_run
    )


def _valid_interfaces(interface_1: int, interface_2: int):
    """Patch ``AnalysisConfig`` so the CLI builds it with valid interfaces.

    ``_cmd_analyze`` does not expose ``--interface-1/--interface-2`` and
    always relies on the dataclass defaults (11/12), which are only valid
    for layups with >=13 plies. Tests that exercise *small* layups purely
    to verify layup parsing must therefore supply an interface pair that
    is valid for that ply count; otherwise ``AnalysisConfig.__post_init__``
    (added in #147) rejects the config before parsing can be asserted.

    This binds valid interface defaults via ``functools.partial`` so the
    test stays test-only and does not alter CLI defaults or #147's
    validation.
    """
    from wrinklefe import analysis as _analysis

    bound = functools.partial(
        _analysis.AnalysisConfig,
        interface_1=interface_1,
        interface_2=interface_2,
    )
    return patch("wrinklefe.analysis.AnalysisConfig", new=bound)


# --------------------------------------------------------------------------- #
# analyze: flag -> config plumbing
# --------------------------------------------------------------------------- #


def test_analyze_default_runs_full_fe_solve():
    """No flags should run a full FE solve, not analytical-only."""
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main(["analyze"])

    cfg = captured["config"]
    # The CLI used to hard-code analytical_only=True. Default is now
    # full FE.
    assert captured["analytical_only"] is False
    assert cfg.analytical_only is False


def test_analyze_analytical_only_flag_skips_fe():
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main(["analyze", "--analytical-only"])

    cfg = captured["config"]
    assert captured["analytical_only"] is True
    assert cfg.analytical_only is True


def test_analyze_no_fe_flag_skips_fe():
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main(["analyze", "--no-fe"])

    assert captured["analytical_only"] is True


def test_analyze_fe_flag_runs_full_fe():
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main(["analyze", "--fe"])

    assert captured["analytical_only"] is False


def test_analyze_passes_solver_and_mesh_options():
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main([
            "analyze",
            "--nx", "8",
            "--ny", "4",
            "--strain", "-0.005",
            "--solver", "iterative",
        ])

    cfg = captured["config"]
    assert cfg.nx == 8
    assert cfg.ny == 4
    assert cfg.applied_strain == pytest.approx(-0.005)
    assert cfg.solver == "iterative"


# --------------------------------------------------------------------------- #
# compare: --analytical-only / --no-analytical-only
# --------------------------------------------------------------------------- #


def _make_fake_result():
    r = MagicMock()
    r.morphology_factor = 1.0
    r.max_angle_rad = 0.0
    r.effective_angle_rad = 0.0
    r.damage_index = 0.0
    r.analytical_knockdown = 1.0
    r.analytical_strength_MPa = 100.0
    return r


def test_compare_default_is_analytical_only():
    captured: dict = {}

    def fake_compare(base_config, morphologies, analytical_only):
        captured["analytical_only"] = analytical_only
        return {m: _make_fake_result() for m in morphologies}

    with patch(
        "wrinklefe.analysis.WrinkleAnalysis.compare_morphologies",
        new=staticmethod(fake_compare),
    ):
        cli_main(["compare"])

    assert captured["analytical_only"] is True


def test_compare_no_analytical_only_runs_full_fe():
    captured: dict = {}

    def fake_compare(base_config, morphologies, analytical_only):
        captured["analytical_only"] = analytical_only
        return {m: _make_fake_result() for m in morphologies}

    with patch(
        "wrinklefe.analysis.WrinkleAnalysis.compare_morphologies",
        new=staticmethod(fake_compare),
    ):
        cli_main(["compare", "--no-analytical-only"])

    assert captured["analytical_only"] is False


# --------------------------------------------------------------------------- #
# sweep: --analytical-only / --no-analytical-only
# --------------------------------------------------------------------------- #


def test_sweep_default_is_analytical_only():
    captured: dict = {}

    def fake_sweep(base_config, parameter, values, analytical_only,
                   n_workers=1):
        captured["analytical_only"] = analytical_only
        captured["parameter"] = parameter
        captured["values"] = list(values)
        captured["n_workers"] = n_workers
        return [_make_fake_result() for _ in values]

    with patch(
        "wrinklefe.analysis.WrinkleAnalysis.parametric_sweep",
        new=staticmethod(fake_sweep),
    ):
        cli_main([
            "sweep",
            "--parameter", "amplitude",
            "--min", "0.1",
            "--max", "0.3",
            "--steps", "3",
        ])

    assert captured["analytical_only"] is True
    assert captured["parameter"] == "amplitude"
    assert len(captured["values"]) == 3
    assert captured["n_workers"] == 1  # sequential by default (issue #260)


def test_sweep_no_analytical_only_runs_full_fe():
    captured: dict = {}

    def fake_sweep(base_config, parameter, values, analytical_only,
                   n_workers=1):
        captured["analytical_only"] = analytical_only
        return [_make_fake_result() for _ in values]

    with patch(
        "wrinklefe.analysis.WrinkleAnalysis.parametric_sweep",
        new=staticmethod(fake_sweep),
    ):
        cli_main([
            "sweep",
            "--parameter", "amplitude",
            "--min", "0.1",
            "--max", "0.3",
            "--steps", "2",
            "--no-analytical-only",
        ])

    assert captured["analytical_only"] is False


# --------------------------------------------------------------------------- #
# sweep: input validation and error handling (issue #298)
# --------------------------------------------------------------------------- #


def test_sweep_unknown_parameter_exits_cleanly(capsys):
    """A bad --parameter prints a one-line error and exits non-zero —
    no raw traceback (the AttributeError from parametric_sweep is
    caught)."""
    with pytest.raises(SystemExit) as exc:
        cli_main([
            "sweep", "--parameter", "not_a_field",
            "--min", "0.1", "--max", "0.5",
        ])
    assert exc.value.code != 0
    err = capsys.readouterr().err
    assert "error:" in err and "not_a_field" in err
    assert "Traceback" not in err


def test_sweep_min_not_less_than_max_rejected(capsys):
    for lo, hi in (("0.5", "0.1"), ("0.2", "0.2")):
        with pytest.raises(SystemExit) as exc:
            cli_main([
                "sweep", "--parameter", "amplitude",
                "--min", lo, "--max", hi,
            ])
        assert exc.value.code == 2
        assert "--min" in capsys.readouterr().err


def test_sweep_steps_below_two_rejected(capsys):
    for steps in ("1", "0", "-3"):
        with pytest.raises(SystemExit) as exc:
            cli_main([
                "sweep", "--parameter", "amplitude",
                "--min", "0.1", "--max", "0.5", "--steps", steps,
            ])
        assert exc.value.code == 2
        assert "--steps" in capsys.readouterr().err


def test_sweep_validation_runs_before_solve():
    """A degenerate range is rejected before parametric_sweep is ever
    called (no compute burned)."""
    called = {"n": 0}

    def fake_sweep(*a, **k):
        called["n"] += 1
        return []

    with patch(
        "wrinklefe.analysis.WrinkleAnalysis.parametric_sweep",
        new=staticmethod(fake_sweep),
    ), pytest.raises(SystemExit):
        cli_main([
            "sweep", "--parameter", "amplitude",
            "--min", "0.5", "--max", "0.1",
        ])
    assert called["n"] == 0


# --------------------------------------------------------------------------- #
# sweep/compare: --output-json / --output-csv (issue #266)
# --------------------------------------------------------------------------- #


def test_sweep_output_csv_and_json(tmp_path, capsys):
    """A coarse analytical sweep writes a tidy CSV and a JSON array of
    per-run objects matching the analyze --output-json schema; the
    stdout table is still printed."""
    import csv
    import json

    csv_path = tmp_path / "sweep.csv"
    json_path = tmp_path / "sweep.json"
    cli_main([
        "sweep", "--parameter", "amplitude",
        "--min", "0.1", "--max", "0.3", "--steps", "3",
        "--output-csv", str(csv_path),
        "--output-json", str(json_path),
    ])

    # stdout table unchanged by default.
    out = capsys.readouterr().out
    assert "WrinkleFE Parametric Sweep" in out

    rows = list(csv.DictReader(csv_path.open()))
    assert [r["parameter_name"] for r in rows] == ["amplitude"] * 3
    assert [float(r["parameter_value"]) for r in rows] == [0.1, 0.2, 0.3]
    assert all(0.0 < float(r["knockdown"]) <= 1.0 for r in rows)
    # Full precision in files (more digits than the 4dp stdout table).
    assert len(rows[0]["knockdown"].split(".")[-1]) > 4
    # Analytical-only: FE-derived columns are empty, not bogus.
    assert rows[0]["max_failure_index"] == ""

    arr = json.loads(json_path.read_text())
    assert len(arr) == 3
    # Per-run schema parity with analyze --output-json.
    for entry in arr:
        assert {"wrinklefe_version", "provenance", "configuration",
                "analytical_predictions"} <= set(entry)
    assert [e["configuration"]["amplitude_mm"] for e in arr] == [0.1, 0.2, 0.3]


def test_compare_output_csv_and_json(tmp_path):
    import csv
    import json

    csv_path = tmp_path / "cmp.csv"
    json_path = tmp_path / "cmp.json"
    cli_main([
        "compare",
        "--output-csv", str(csv_path),
        "--output-json", str(json_path),
    ])

    rows = list(csv.DictReader(csv_path.open()))
    assert [r["morphology"] for r in rows] == ["stack", "convex", "concave"]
    assert all(r["parameter_name"] == "morphology" for r in rows)

    arr = json.loads(json_path.read_text())
    assert [e["configuration"]["morphology"] for e in arr] == [
        "stack", "convex", "concave"
    ]


def test_sweep_without_output_flags_writes_nothing(tmp_path, monkeypatch):
    """Default behaviour (stdout only) is preserved: no files appear."""
    monkeypatch.chdir(tmp_path)
    cli_main([
        "sweep", "--parameter", "amplitude",
        "--min", "0.1", "--max", "0.2", "--steps", "2",
    ])
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# argparse exit codes for invalid input
# --------------------------------------------------------------------------- #


def test_unknown_morphology_exits_with_code_2():
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["analyze", "--morphology", "bogus"])
    assert exc_info.value.code == 2


def test_invalid_numeric_value_exits_with_code_2():
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["analyze", "--amplitude", "not-a-number"])
    assert exc_info.value.code == 2


# --------------------------------------------------------------------------- #
# issue #83: contracted layup notation + uniform/graded morphologies
# --------------------------------------------------------------------------- #


def test_cli_morphology_choices_sourced_from_core():
    """CLI choices must equal the canonical core.morphology set, not a
    hard-coded literal (acceptance criterion of issue #83)."""
    from wrinklefe.cli import MORPHOLOGY_CHOICES
    from wrinklefe.core.morphology import (
        MORPHOLOGY_PHASES,
        SINGLE_WRINKLE_MODES,
    )

    expected = sorted(set(MORPHOLOGY_PHASES.keys()) | SINGLE_WRINKLE_MODES)
    assert MORPHOLOGY_CHOICES == expected
    # The two formerly-unreachable modes are now present.
    assert "uniform" in MORPHOLOGY_CHOICES
    assert "graded" in MORPHOLOGY_CHOICES


@pytest.mark.parametrize("morph", ["uniform", "graded"])
def test_analyze_accepts_uniform_and_graded(morph):
    """--morphology uniform/graded are accepted and plumbed into the
    AnalysisConfig the engine receives."""
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main(["analyze", "--morphology", morph, "--analytical-only"])

    cfg = captured["config"]
    assert cfg.morphology == morph


# --------------------------------------------------------------------------- #
# issue #371: tool_flat morphology + surface-transition-plies
# --------------------------------------------------------------------------- #


def test_tool_flat_is_a_cli_morphology_choice():
    """``tool_flat`` reaches the CLI choices via the canonical core set."""
    from wrinklefe.cli import MORPHOLOGY_CHOICES

    assert "tool_flat" in MORPHOLOGY_CHOICES


@pytest.mark.parametrize("spelling", ["tool_flat", "tool-flat"])
def test_analyze_accepts_tool_flat_and_hyphen_alias(spelling):
    """Both ``tool_flat`` and the hyphenated ``tool-flat`` alias map to the
    canonical ``tool_flat`` morphology in the AnalysisConfig."""
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main([
            "analyze", "--morphology", spelling,
            "--amplitude", "0.2", "--analytical-only",
        ])

    cfg = captured["config"]
    assert cfg.morphology == "tool_flat"


def test_analyze_surface_transition_plies_maps_into_config():
    """--surface-transition-plies threads into AnalysisConfig for tool_flat."""
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main([
            "analyze", "--morphology", "tool-flat",
            "--amplitude", "0.3", "--surface-transition-plies", "4",
            "--analytical-only",
        ])

    cfg = captured["config"]
    assert cfg.morphology == "tool_flat"
    assert cfg.surface_transition_plies == 4


def test_analyze_tool_flat_amplitude_bound_rejected():
    """An amplitude above the tool_flat inversion bound exits with code 2
    (the actionable ValueError from AnalysisConfig validation)."""
    # 0.8 * 2 * 0.183 / 1 = 0.293 mm bound at the default transition plies.
    with pytest.raises(SystemExit) as exc_info:
        cli_main([
            "analyze", "--morphology", "tool-flat",
            "--amplitude", "0.5", "--surface-transition-plies", "2",
        ])
    assert exc_info.value.code == 2


def test_analyze_tool_flat_default_fe_auto_enables_pockets():
    """A default tool_flat FE run auto-enables surface resin pockets (they
    are the morphology's defining physics)."""
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main([
            "analyze", "--morphology", "tool-flat", "--amplitude", "0.2",
        ])

    cfg = captured["config"]
    assert cfg.morphology == "tool_flat"
    assert captured["analytical_only"] is False
    assert cfg.enable_surface_resin_pockets is True


def test_analyze_contracted_layup_expands_to_24_plies():
    """'[0/45/-45/90]_3s' must reach AnalysisConfig as the same 24-ply
    list the Streamlit app produces."""
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main([
            "analyze",
            "--layup", "[0/45/-45/90]_3s",
            "--analytical-only",
        ])

    cfg = captured["config"]
    quarter = [0.0, 45.0, -45.0, 90.0]
    expected = quarter * 3
    expected = expected + expected[::-1]
    assert cfg.angles == expected
    assert len(cfg.angles) == 24


def test_analyze_contracted_layup_matches_shared_parser():
    """The CLI path must agree exactly with the shared parser used by
    app.py for the same input."""
    from wrinklefe.core.layup import parse_layup

    captured, patcher = _stub_analysis_run()
    # "[0/±45/90]s" expands to 8 plies; the CLI's default interfaces
    # (11/12) are out of range for it, so supply a valid interior pair.
    with patcher, _valid_interfaces(3, 4):
        cli_main([
            "analyze",
            "--layup", "[0/±45/90]s",
            "--analytical-only",
        ])

    assert captured["config"].angles == parse_layup("[0/±45/90]s")


def test_analyze_explicit_comma_list_still_works():
    """Backwards compatibility: the original comma-separated form."""
    captured, patcher = _stub_analysis_run()
    # This layup has 4 plies; the CLI's default interfaces (11/12) are
    # out of range for it, so supply an interface pair valid at 4 plies.
    with patcher, _valid_interfaces(1, 2):
        cli_main([
            "analyze",
            "--angles", "0, 45, -45, 90",
            "--analytical-only",
        ])

    assert captured["config"].angles == [0.0, 45.0, -45.0, 90.0]


def test_analyze_invalid_layup_exits_nonzero_with_message(capsys):
    """A malformed layup must produce a clear error and a non-zero exit."""
    captured, patcher = _stub_analysis_run()
    with patcher, pytest.raises(SystemExit) as exc_info:
        cli_main(["analyze", "--layup", "[0/45/", "--analytical-only"])

    assert exc_info.value.code != 0
    err = capsys.readouterr().err
    assert "layup" in err.lower()


# --------------------------------------------------------------------------- #
# issues #154 / #156: small layups + explicit --interface-1/--interface-2
# --------------------------------------------------------------------------- #


def test_analyze_small_layup_no_interface_flags_succeeds():
    """Issue #154: ``analyze --layup '[0/±45/90]s'`` (8 plies) must no
    longer crash. With auto-derivation in ``AnalysisConfig`` the CLI now
    runs successfully without the user having to supply interface flags.
    """
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main([
            "analyze",
            "--layup", "[0/±45/90]s",
            "--analytical-only",
        ])

    cfg = captured["config"]
    # 8-ply layup auto-derives to (3, 4) and both must satisfy the
    # validator (which previously rejected the hard-coded 11/12).
    assert len(cfg.angles) == 8
    assert cfg.interface_1 == 3
    assert cfg.interface_2 == 4


def test_analyze_interface_flags_override_auto_derivation():
    """Issue #154: ``--interface-1`` / ``--interface-2`` flags exist on
    ``analyze`` and override the auto-derived defaults."""
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main([
            "analyze",
            "--layup", "[0/90]_4",
            "--interface-1", "2",
            "--interface-2", "3",
            "--analytical-only",
        ])

    cfg = captured["config"]
    assert cfg.interface_1 == 2
    assert cfg.interface_2 == 3


def test_analyze_24_ply_default_interfaces_unchanged():
    """The canonical 24-ply default layup still resolves to (11, 12) —
    backwards-compat with the pre-#154 hard-coded dataclass defaults."""
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main(["analyze", "--analytical-only"])

    cfg = captured["config"]
    assert len(cfg.angles) == 24
    assert cfg.interface_1 == 11
    assert cfg.interface_2 == 12


def test_sweep_accepts_graded_morphology():
    """sweep --morphology graded must be accepted (was argparse-rejected)."""
    captured: dict = {}

    def fake_sweep(base_config, parameter, values, analytical_only,
                   n_workers=1):
        captured["config"] = base_config
        return [_make_fake_result() for _ in values]

    with patch(
        "wrinklefe.analysis.WrinkleAnalysis.parametric_sweep",
        new=staticmethod(fake_sweep),
    ):
        cli_main([
            "sweep",
            "--parameter", "amplitude",
            "--min", "0.1",
            "--max", "0.3",
            "--steps", "2",
            "--morphology", "graded",
        ])

    assert captured["config"].morphology == "graded"


# --------------------------------------------------------------------------- #
# sweep: --parallel (issue #260)
# --------------------------------------------------------------------------- #


def test_sweep_parallel_flag_passthrough():
    """--parallel N reaches parametric_sweep as n_workers (0 = all
    cores is resolved downstream, so it must pass through verbatim)."""
    for flag_val, expected in (("4", 4), ("0", 0)):
        captured: dict = {}

        def fake_sweep(base_config, parameter, values, analytical_only,
                       n_workers=1):
            captured["n_workers"] = n_workers
            return [_make_fake_result() for _ in values]

        with patch(
            "wrinklefe.analysis.WrinkleAnalysis.parametric_sweep",
            new=staticmethod(fake_sweep),
        ):
            cli_main([
                "sweep", "--parameter", "amplitude",
                "--min", "0.1", "--max", "0.3", "--steps", "2",
                "--parallel", flag_val,
            ])
        assert captured["n_workers"] == expected


def test_sweep_negative_parallel_rejected(capsys):
    with pytest.raises(SystemExit) as exc:
        cli_main([
            "sweep", "--parameter", "amplitude",
            "--min", "0.1", "--max", "0.3",
            "--parallel", "-1",
        ])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--parallel" in err and "Traceback" not in err


# --------------------------------------------------------------------------- #
# analyze: wrinkle-defect capabilities (issue #346)
# --------------------------------------------------------------------------- #

_UD_LAYUP = "0,0,0,0,0,0,0,0"


def test_analyze_wrinkle_z_position_maps_to_config():
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main(["analyze", "--analytical-only", "--wrinkle-z-position", "0.7"])
    assert captured["config"].wrinkle_z_position == pytest.approx(0.7)


def test_analyze_wrinkle_z_position_out_of_range_rejected(capsys):
    with pytest.raises(SystemExit) as exc:
        cli_main(["analyze", "--analytical-only", "--wrinkle-z-position", "1.5"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "wrinkle_z_position" in err and "[0, 1]" in err
    assert "Traceback" not in err


def test_analyze_gate_maps_to_preset_object():
    from wrinklefe.core.penetration_gate import GATE_LI2025_VACBAG

    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main([
            "analyze", "--analytical-only", "--morphology", "uniform",
            "--angles", _UD_LAYUP, "--interface-1", "3", "--interface-2", "4",
            "--gate", "li2025-vacbag",
        ])
    assert captured["config"].penetration_gate is GATE_LI2025_VACBAG


def test_analyze_gate_moulded_maps_to_preset_object():
    from wrinklefe.core.penetration_gate import GATE_LI2024_MOULDED

    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main([
            "analyze", "--analytical-only", "--morphology", "uniform",
            "--angles", _UD_LAYUP, "--interface-1", "3", "--interface-2", "4",
            "--gate", "li2024-moulded",
        ])
    assert captured["config"].penetration_gate is GATE_LI2024_MOULDED


def test_analyze_resin_pocket_maps_and_forces_fe():
    """--resin-pocket enables the FE-only zone and overrides --analytical-only."""
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main(["analyze", "--analytical-only", "--resin-pocket"])
    cfg = captured["config"]
    assert cfg.enable_resin_pocket is True
    # FE-only feature must force the FE path despite --analytical-only.
    assert cfg.analytical_only is False
    assert captured["analytical_only"] is False


def test_analyze_surface_resin_pockets_maps_side():
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main([
            "analyze", "--surface-resin-pockets",
            "--surface-pocket-side", "both",
        ])
    cfg = captured["config"]
    assert cfg.enable_surface_resin_pockets is True
    assert cfg.surface_pocket_side == "both"
    assert cfg.analytical_only is False


def test_analyze_progressive_maps_increments_and_forces_fe():
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main([
            "analyze", "--analytical-only",
            "--progressive", "--increments", "7",
        ])
    cfg = captured["config"]
    assert cfg.enable_progressive_damage is True
    assert cfg.progressive_n_increments == 7
    assert cfg.analytical_only is False
    assert captured["analytical_only"] is False


def test_analyze_gate_overrides_config_file(tmp_path):
    """A --gate flag overrides the penetration_gate from a --config file."""
    from wrinklefe.analysis import AnalysisConfig
    from wrinklefe.core.penetration_gate import (
        GATE_LI2024_MOULDED,
        GATE_LI2025_VACBAG,
    )

    base = AnalysisConfig(
        morphology="uniform",
        angles=[0.0] * 8,
        interface_1=3,
        interface_2=4,
        penetration_gate=GATE_LI2024_MOULDED,
        analytical_only=True,
    )
    path = tmp_path / "case.json"
    base.save_json(path)

    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main([
            "analyze", "--config", str(path), "--gate", "li2025-vacbag",
        ])
    # The flag wins; other file values are preserved.
    assert captured["config"].penetration_gate is GATE_LI2025_VACBAG
    assert captured["config"].morphology == "uniform"


def test_analyze_gate_reproduces_api_penetration_gate_kd():
    """`analyze --gate li2025-vacbag` reproduces the API gate KD (UD)."""
    import math

    from wrinklefe.analysis import WrinkleAnalysis
    from wrinklefe.core.penetration_gate import (
        GATE_LI2025_VACBAG,
        penetration_gate_kd,
    )

    amplitude, wavelength = 0.5, 16.0
    angles = [0.0] * 8

    # Capture the config the CLI builds from --gate, then run it, so the
    # flag→config mapping (not just a hand-built config) is under test.
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main([
            "analyze", "--analytical-only", "--morphology", "uniform",
            "--angles", ",".join("0" for _ in angles),
            "--interface-1", "3", "--interface-2", "4",
            "--amplitude", str(amplitude), "--wavelength", str(wavelength),
            "--gate", "li2025-vacbag",
        ])
    cfg = captured["config"]
    assert cfg.penetration_gate is GATE_LI2025_VACBAG
    kd_cli = WrinkleAnalysis(cfg).run(analytical_only=True).analytical_knockdown

    theta_deg = math.degrees(math.atan(2 * math.pi * amplitude / wavelength))
    dt = amplitude / (cfg.ply_thickness * len(angles))
    kd_api = penetration_gate_kd(
        theta_deg, dt, GATE_LI2025_VACBAG, z_position=0.5
    )
    assert kd_cli == pytest.approx(kd_api)


@pytest.mark.slow
def test_analyze_progressive_reports_knockdown(capsys):
    """`analyze --progressive` completes a real FE solve and reports KD."""
    cli_main([
        "analyze", "--progressive", "--increments", "3",
        "--morphology", "stack", "--angles", _UD_LAYUP,
        "--interface-1", "3", "--interface-2", "4",
        "--amplitude", "0.3", "--wavelength", "16",
        "--nx", "4", "--ny", "2",
    ])
    out = capsys.readouterr().out
    assert "Progressive Damage" in out
    assert "Strength knockdown:" in out


# --------------------------------------------------------------------------- #
# sweep: new base-config parameters (issue #346)
# --------------------------------------------------------------------------- #


def test_sweep_wrinkle_z_position_end_to_end(capsys):
    """`sweep --parameter wrinkle_z_position` runs and yields distinct KDs."""
    cli_main([
        "sweep", "--parameter", "wrinkle_z_position",
        "--min", "0.2", "--max", "0.8", "--steps", "3",
        "--morphology", "graded",
    ])
    out = capsys.readouterr().out
    assert "wrinkle_z_position" in out
    # Three data rows should be present between the header rules.
    assert out.count("\n") > 5


# --------------------------------------------------------------------------- #
# analyze: transverse modes + ergonomics (issue #375)
# --------------------------------------------------------------------------- #


def test_analyze_transverse_mode_maps_and_forces_fe():
    """A non-uniform --transverse-mode reaches the config and forces FE."""
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main([
            "analyze", "--transverse-mode", "gaussian_decay",
            "--transverse-span", "20.0", "--transverse-width", "5.0",
        ])
    cfg = captured["config"]
    assert cfg.transverse_mode == "gaussian_decay"
    assert cfg.transverse_span == pytest.approx(20.0)
    assert cfg.transverse_width == pytest.approx(5.0)
    # FE-only feature forces the full FE path over the analytical default.
    assert captured["analytical_only"] is False


def test_analyze_transverse_uniform_default_untouched():
    """Without the flag the config keeps the uniform (x-only) default."""
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main(["analyze", "--analytical-only"])
    assert captured["config"].transverse_mode == "uniform"
    assert captured["analytical_only"] is True


def test_analyze_transverse_czm_conflict_is_clean(capsys):
    """--transverse-mode combined with --enable-czm surfaces a clean error,
    not a raw traceback (the config validation rejects the combination)."""
    with pytest.raises(SystemExit) as exc:
        cli_main([
            "analyze", "--transverse-mode", "gaussian_decay", "--enable-czm",
        ])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "error:" in err and "Traceback" not in err


def test_analyze_nz_per_ply_and_ply_thickness_map():
    captured, patcher = _stub_analysis_run()
    with patcher:
        cli_main([
            "analyze", "--analytical-only",
            "--nz-per-ply", "3", "--ply-thickness", "0.25",
        ])
    cfg = captured["config"]
    assert cfg.nz_per_ply == 3
    assert cfg.ply_thickness == pytest.approx(0.25)


def test_analyze_output_csv(tmp_path):
    """`analyze --output-csv` writes a one-row tidy CSV matching the
    sweep/compare schema."""
    import csv

    csv_path = tmp_path / "run.csv"
    cli_main([
        "analyze", "--analytical-only",
        "--amplitude", "0.3", "--wavelength", "16",
        "--output-csv", str(csv_path),
    ])
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 1
    assert {"parameter_name", "parameter_value", "morphology",
            "knockdown", "predicted_strength_MPa"} <= set(rows[0])
    assert 0.0 < float(rows[0]["knockdown"]) <= 1.0


# --------------------------------------------------------------------------- #
# stochastic subcommand (issue #375 CLI exposure of #301)
# --------------------------------------------------------------------------- #


def test_stochastic_reports_percentiles_and_is_reproducible(capsys):
    """`stochastic` prints percentiles and a fixed --seed reproduces them."""
    argv = [
        "stochastic",
        "--distribution", "amplitude:normal:0.4:0.05",
        "--n-samples", "128", "--seed", "11",
    ]
    cli_main(argv)
    out_a = capsys.readouterr().out
    assert "Probabilistic Analysis" in out_a
    assert "P5=" in out_a and "P50=" in out_a and "P95=" in out_a

    cli_main(argv)
    out_b = capsys.readouterr().out
    # Same seed -> byte-identical percentile summary.
    assert out_a == out_b


def test_stochastic_json_and_csv_outputs(tmp_path):
    import csv
    import json

    json_path = tmp_path / "prob.json"
    csv_path = tmp_path / "prob.csv"
    cli_main([
        "stochastic",
        "--distribution", "amplitude:normal:0.4:0.05",
        "--distribution", "wavelength:uniform:14:18",
        "--n-samples", "64", "--seed", "3",
        "--output-json", str(json_path),
        "--output-csv", str(csv_path),
    ])

    payload = json.loads(json_path.read_text())
    assert payload["n_samples"] == 64
    assert payload["seed"] == 3
    assert "5.0" in payload["knockdown"]["percentiles"]
    assert len(payload["samples"]["knockdown"]) == 64

    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 64
    assert {"input_amplitude", "input_wavelength", "knockdown",
            "strength_MPa", "modulus_knockdown"} <= set(rows[0])


def test_stochastic_reproducible_percentile_values(tmp_path):
    """Two seeded runs produce identical percentile knockdowns in JSON."""
    import json

    def run(dst):
        cli_main([
            "stochastic", "--distribution", "amplitude:normal:0.4:0.05",
            "--n-samples", "100", "--seed", "42", "--output-json", str(dst),
        ])
        return json.loads(dst.read_text())["knockdown"]["percentiles"]

    a = run(tmp_path / "a.json")
    b = run(tmp_path / "b.json")
    assert a == b
    # A genuine spread was produced (not a degenerate constant).
    assert a["5.0"] < a["95.0"]


@pytest.mark.parametrize("bad", [
    "amplitude:normal:0.4",          # too few fields
    "amplitude:normal:0.4:0.05:1",   # too many fields
    "amplitude:weibull:0.4:0.05",    # unknown distribution
    "amplitude:normal:x:0.05",       # non-numeric parameter
    ":normal:0.4:0.05",              # empty field name
])
def test_stochastic_bad_distribution_spec_exits_cleanly(bad, capsys):
    with pytest.raises(SystemExit) as exc:
        cli_main(["stochastic", "--distribution", bad, "--n-samples", "10"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "error:" in err and "--distribution" in err
    assert "Traceback" not in err


def test_stochastic_uses_config_base(tmp_path):
    """`stochastic --config` samples over the file's base laminate."""
    from wrinklefe.analysis import AnalysisConfig

    base = AnalysisConfig(
        angles=[0.0] * 8, interface_1=3, interface_2=4,
        analytical_only=True,
    )
    path = tmp_path / "base.json"
    base.save_json(path)

    from wrinklefe.stochastic import (
        probabilistic_analysis as _real_prob,
    )

    captured: dict = {}

    def fake_prob(base_config, distributions, **kwargs):
        captured["base_config"] = base_config
        captured["distributions"] = distributions
        return _real_prob(base_config, distributions, **kwargs)

    with patch("wrinklefe.stochastic.probabilistic_analysis", new=fake_prob):
        cli_main([
            "stochastic", "--config", str(path),
            "--distribution", "amplitude:normal:0.3:0.02",
            "--n-samples", "16", "--seed", "1",
        ])

    assert captured["base_config"].angles == [0.0] * 8
    assert captured["distributions"] == {"amplitude": ("normal", 0.3, 0.02)}


# --------------------------------------------------------------------------- #
# --version (issue #375)
# --------------------------------------------------------------------------- #


def test_version_matches_installed_metadata(capsys):
    """`wrinklefe --version` reports the installed distribution version, not
    a hardcoded string."""
    from importlib.metadata import version as _pkg_version

    with pytest.raises(SystemExit) as exc:
        cli_main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert _pkg_version("wrinklefe") in out


# --------------------------------------------------------------------------- #
# sweep / compare: --config base (issue #375)
# --------------------------------------------------------------------------- #


def test_sweep_config_uses_files_laminate_not_default(tmp_path):
    """`sweep --config` runs over the file's UD laminate/gate, not the
    default IM7/8552 quasi-iso laminate."""
    from wrinklefe.analysis import AnalysisConfig
    from wrinklefe.core.penetration_gate import GATE_LI2025_VACBAG

    base = AnalysisConfig(
        morphology="graded",
        angles=[0.0] * 14,
        ply_thickness=0.44,
        wavelength=12.9,
        width=6.45,
        penetration_gate=GATE_LI2025_VACBAG,
        analytical_only=True,
    )
    path = tmp_path / "ud_gate.json"
    base.save_json(path)

    captured: dict = {}

    def fake_sweep(base_config, parameter, values, analytical_only,
                   n_workers=1):
        captured["base_config"] = base_config
        return [_make_fake_result() for _ in values]

    with patch(
        "wrinklefe.analysis.WrinkleAnalysis.parametric_sweep",
        new=staticmethod(fake_sweep),
    ):
        cli_main([
            "sweep", "--config", str(path),
            "--parameter", "amplitude",
            "--min", "0.3", "--max", "0.9", "--steps", "4",
        ])

    cfg = captured["base_config"]
    # The file's UD laminate + gate reached the sweep engine, NOT the
    # IM7/8552 quasi-iso default (angles=None, ply_thickness=0.183).
    assert cfg.angles == [0.0] * 14
    assert cfg.ply_thickness == pytest.approx(0.44)
    assert cfg.morphology == "graded"
    assert cfg.penetration_gate is GATE_LI2025_VACBAG


def test_sweep_config_explicit_flag_overrides_file(tmp_path):
    """A geometry flag on the command line overrides the --config file."""
    from wrinklefe.analysis import AnalysisConfig

    base = AnalysisConfig(morphology="graded", analytical_only=True)
    path = tmp_path / "case.json"
    base.save_json(path)

    captured: dict = {}

    def fake_sweep(base_config, parameter, values, analytical_only,
                   n_workers=1):
        captured["base_config"] = base_config
        return [_make_fake_result() for _ in values]

    with patch(
        "wrinklefe.analysis.WrinkleAnalysis.parametric_sweep",
        new=staticmethod(fake_sweep),
    ):
        cli_main([
            "sweep", "--config", str(path), "--morphology", "convex",
            "--parameter", "amplitude", "--min", "0.1", "--max", "0.3",
            "--steps", "2",
        ])

    assert captured["base_config"].morphology == "convex"


def test_sweep_bad_config_path_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        cli_main([
            "sweep", "--config", "/no/such/file.json",
            "--parameter", "amplitude", "--min", "0.1", "--max", "0.3",
        ])
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_compare_config_uses_files_laminate(tmp_path):
    """`compare --config` runs the three morphologies over the file's
    laminate/material, not the default."""
    from wrinklefe.analysis import AnalysisConfig

    base = AnalysisConfig(
        angles=[0.0] * 8,
        interface_1=3,
        interface_2=4,
        ply_thickness=0.25,
        analytical_only=True,
    )
    path = tmp_path / "case.json"
    base.save_json(path)

    captured: dict = {}

    def fake_compare(base_config, morphologies, analytical_only):
        captured["base_config"] = base_config
        return {m: _make_fake_result() for m in morphologies}

    with patch(
        "wrinklefe.analysis.WrinkleAnalysis.compare_morphologies",
        new=staticmethod(fake_compare),
    ):
        cli_main(["compare", "--config", str(path)])

    cfg = captured["base_config"]
    assert cfg.angles == [0.0] * 8
    assert cfg.ply_thickness == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# critical subcommand (issue #280)
# --------------------------------------------------------------------------- #


def _fake_critical_result(**overrides):
    """A minimal ``CriticalValueResult`` stand-in for handler tests."""
    from wrinklefe.analysis import AnalysisConfig
    from wrinklefe.goalseek import CriticalValueResult, GoalSeekEvaluation

    rows = [
        GoalSeekEvaluation(
            index=i, value=v, objective=o, knockdown=o,
            strength_MPa=1200.0 * o, residual=o - 0.85, phase="scan",
            runtime_s=0.0,
        )
        for i, (v, o) in enumerate([(0.0, 1.0), (0.1, 0.8), (0.2, 0.6)])
    ]
    defaults = dict(
        parameter="amplitude", objective_name="knockdown", target=0.85,
        target_is_strength=False, direction="decreasing",
        status="converged", message="", critical_value=0.0665,
        critical_value_root=0.0666, achieved_objective=0.8500007,
        achieved_knockdown=0.8500007, achieved_strength_MPa=1020.0,
        critical_config=AnalysisConfig(amplitude=0.0665),
        bounds=(0.0, 4.392), bounds_source=("zero", "laminate_thickness"),
        scan_values=[r.value for r in rows],
        scan_objectives=[r.objective for r in rows],
        sign_changes=1, log_spaced=True, pristine_reference_MPa=1200.0,
        evaluations=rows, n_evaluations=len(rows), rtol=1e-3,
        analytical_only=True, runtime_s=0.5,
    )
    defaults.update(overrides)
    return CriticalValueResult(**defaults)


def _capture_critical():
    """Patch the engine and record the kwargs the handler passes it."""
    captured: dict = {}

    def fake(base_config, **kwargs):
        captured["config"] = base_config
        captured.update(kwargs)
        return _fake_critical_result()

    return captured, patch("wrinklefe.goalseek.find_critical_value", new=fake)


def test_critical_defaults_reach_the_engine():
    """T-60: the documented defaults are what the engine actually sees."""
    captured, patcher = _capture_critical()
    with patcher, pytest.raises(SystemExit) as excinfo:
        cli_main(["critical", "--target-knockdown", "0.85"])
    assert excinfo.value.code == 0
    assert captured["parameter"] == "amplitude"
    assert captured["target_knockdown"] == pytest.approx(0.85)
    assert captured["target_strength_MPa"] is None
    assert captured["analytical_only"] is None
    assert captured["rtol"] == pytest.approx(1e-3)
    assert captured["scan_points"] == 9
    assert captured["bracket"] is None


def test_critical_no_analytical_only_warns(capsys):
    """T-61: the FE opt-in reaches the engine and is announced."""
    captured, patcher = _capture_critical()
    with patcher, pytest.raises(SystemExit):
        cli_main([
            "critical", "--target-knockdown", "0.85", "--no-analytical-only",
        ])
    assert captured["analytical_only"] is False
    assert "warning:" in capsys.readouterr().err


def test_critical_config_file_precedence(tmp_path):
    """T-62: an explicit geometry flag beats the file; an omitted one
    inherits it (the SUPPRESS invariant)."""
    from wrinklefe.analysis import AnalysisConfig

    base = AnalysisConfig(
        angles=[0.0] * 8, interface_1=3, interface_2=4,
        ply_thickness=0.25, wavelength=20.0, loading="compression",
    )
    path = tmp_path / "base.json"
    base.save_json(path)

    captured, patcher = _capture_critical()
    with patcher, pytest.raises(SystemExit):
        cli_main([
            "critical", "--config", str(path),
            "--target-knockdown", "0.85", "--loading", "tension",
        ])
    cfg = captured["config"]
    assert cfg.angles == [0.0] * 8          # inherited from the file
    assert cfg.ply_thickness == pytest.approx(0.25)
    assert cfg.wavelength == pytest.approx(20.0)
    assert cfg.loading == "tension"         # overridden on the command line


def test_critical_tension_without_a_config_file():
    """T-63: tension is reachable from plain flags, not only via JSON."""
    captured, patcher = _capture_critical()
    with patcher, pytest.raises(SystemExit):
        cli_main([
            "critical", "--target-knockdown", "0.9", "--loading", "tension",
        ])
    assert captured["config"].loading == "tension"


@pytest.mark.parametrize("argv", [
    ["critical"],
    ["critical", "--target-knockdown", "0.85",
     "--target-strength", "1020.0"],
])
def test_critical_target_flags_are_exclusive_and_required(argv):
    """T-64: argparse enforces exactly one target."""
    with pytest.raises(SystemExit) as excinfo:
        cli_main(argv)
    assert excinfo.value.code == 2


def test_critical_unknown_parameter_exits_2(capsys):
    """T-65: a bad field name is a clean exit 2, not a traceback."""
    with pytest.raises(SystemExit) as excinfo:
        cli_main([
            "critical", "--parameter", "amplitud",
            "--target-knockdown", "0.85",
        ])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "Traceback" not in err


def test_critical_integer_parameter_exits_2(capsys):
    """T-66: integer fields are refused by name."""
    with pytest.raises(SystemExit) as excinfo:
        cli_main([
            "critical", "--parameter", "nz_per_ply",
            "--target-knockdown", "0.85",
        ])
    assert excinfo.value.code == 2
    assert "integer AnalysisConfig field" in capsys.readouterr().err


def test_critical_inverted_bracket_exits_before_compute(capsys):
    """T-67."""
    with pytest.raises(SystemExit) as excinfo:
        cli_main([
            "critical", "--target-knockdown", "0.85",
            "--bracket", "0.5", "0.1",
        ])
    assert excinfo.value.code == 2
    assert "--bracket" in capsys.readouterr().err


def test_critical_no_crossing_exits_1_with_the_table(capsys):
    """T-68: a shallow target still prints the curve context."""
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["critical", "--target-knockdown", "0.30"])
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Critical-Value Search" in captured.out
    assert "Knockdown" in captured.out
    assert "no_crossing" in captured.out
    assert "not necessarily an error" in captured.err


@pytest.mark.parametrize("exc", ["memory", "runtime", "linalg"])
def test_critical_solver_failure_exits_1(capsys, exc):
    """T-69: a blown-up forward solve is exit 1, never a traceback."""
    import numpy as np

    errors = {
        "memory": MemoryError("out of memory"),
        "runtime": RuntimeError("solver diverged"),
        "linalg": np.linalg.LinAlgError("singular matrix"),
    }

    def boom(base_config, **kwargs):
        raise errors[exc]

    with patch("wrinklefe.goalseek.find_critical_value", new=boom):
        with pytest.raises(SystemExit) as excinfo:
            cli_main(["critical", "--target-knockdown", "0.85"])
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "Traceback" not in err


def test_critical_converged_run_prints_the_criterion_line(capsys):
    """T-70: the visible proof of the safe-side back-off."""
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["critical", "--target-knockdown", "0.85"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "Largest acceptable amplitude" in out
    assert "Criterion satisfied: knockdown" in out
    assert "Strength (MPa)" in out


def test_critical_save_config_round_trips(tmp_path):
    """T-71: the saved config is the run to attach to a disposition."""
    from wrinklefe.analysis import AnalysisConfig

    path = tmp_path / "critical.json"
    with pytest.raises(SystemExit) as excinfo:
        cli_main([
            "critical", "--target-knockdown", "0.85",
            "--save-config", str(path),
        ])
    assert excinfo.value.code == 0
    restored = AnalysisConfig.load(path)
    assert 0.05 < restored.amplitude < 0.09
    assert restored.analytical_only is True


def test_critical_save_plot(tmp_path, monkeypatch):
    """T-72."""
    monkeypatch.setenv("MPLBACKEND", "Agg")
    path = tmp_path / "critical.png"
    with pytest.raises(SystemExit):
        cli_main([
            "critical", "--target-knockdown", "0.85",
            "--save-plot", str(path),
        ])
    assert path.exists() and path.stat().st_size > 0


def test_critical_json_and_csv_outputs(tmp_path, capsys):
    """T-73: both payloads land and the stdout summary still prints."""
    import csv
    import json

    json_path = tmp_path / "critical.json"
    csv_path = tmp_path / "critical.csv"
    with pytest.raises(SystemExit):
        cli_main([
            "critical", "--target-knockdown", "0.85",
            "--output-json", str(json_path), "--output-csv", str(csv_path),
        ])

    payload = json.loads(json_path.read_text())
    assert isinstance(payload, dict)
    for key in (
        "parameter", "objective", "target", "status", "critical_value",
        "critical_value_root", "achieved_knockdown", "bounds",
        "bounds_source", "scan_values", "scan_objectives", "sign_changes",
        "critical_config", "evaluations", "runs",
    ):
        assert key in payload
    assert payload["status"] == "converged"
    assert payload["critical_config"]["amplitude"] == pytest.approx(
        payload["critical_value"]
    )

    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == payload["n_evaluations"]
    assert {"parameter_name", "parameter_value", "morphology",
            "knockdown", "predicted_strength_MPa"} <= set(rows[0])
    assert "Critical-Value Search" in capsys.readouterr().out
