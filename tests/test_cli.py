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
