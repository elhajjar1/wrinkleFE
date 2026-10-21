"""``wrinklefe compare --morphologies`` (issue #471).

``compare`` was hard-wired to stack, convex and concave while the app
compares any of the six morphologies. ``--morphologies`` takes a
comma-separated list; the default keeps the original three.
"""

from __future__ import annotations

import json

import pytest

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.cli import MORPHOLOGY_CHOICES, _parse_morphologies
from wrinklefe.cli import main as cli_main


def _table_rows(out: str) -> list[str]:
    """Morphology names in the comparison table, in printed order."""
    lines = out.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("-" * 20))
    rows = []
    for ln in lines[start + 1:]:
        if ln.startswith("=" * 20):
            break
        rows.append(ln.split()[0])
    return rows


def test_default_compares_the_original_three(capsys):
    cli_main(["compare"])
    assert _table_rows(capsys.readouterr().out) == ["stack", "convex", "concave"]


def test_all_six_morphologies_including_tool_flat(capsys):
    # tool_flat's FE mesh bounds the amplitude (0.8 * 2 plies * 0.183 mm),
    # so compare at an amplitude every morphology accepts.
    names = "stack,convex,concave,uniform,graded,tool_flat"
    cli_main(["compare", "--morphologies", names, "--amplitude", "0.25"])
    out = capsys.readouterr()
    assert _table_rows(out.out) == names.split(",")
    # On the analytical path tool_flat must not switch on (and then trip
    # over) its FE-only surface pockets.
    assert "auto-enabled" not in out.err


def test_names_are_case_insensitive_and_deduplicated():
    assert _parse_morphologies(" Uniform, uniform ,GRADED,") == ("uniform", "graded")


def test_unknown_name_exits_2_and_lists_the_choices(capsys):
    with pytest.raises(SystemExit) as exc:
        cli_main(["compare", "--morphologies", "stack,wavy"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "'wavy'" in err
    for name in MORPHOLOGY_CHOICES:
        assert name in err


def test_empty_list_exits_2():
    with pytest.raises(SystemExit) as exc:
        cli_main(["compare", "--morphologies", " , "])
    assert exc.value.code == 2


def test_tool_flat_above_its_amplitude_bound_exits_2_with_the_remedy(capsys):
    with pytest.raises(SystemExit) as exc:
        cli_main(["compare", "--morphologies", "tool_flat"])
    assert exc.value.code == 2
    assert "surface_transition_plies" in capsys.readouterr().err


def test_output_json_holds_one_entry_per_selected_morphology(tmp_path):
    path = tmp_path / "cmp.json"
    cli_main(["compare", "--morphologies", "graded,uniform",
              "--output-json", str(path)])
    runs = json.loads(path.read_text())
    assert len(runs) == 2


def test_api_compare_tool_flat_on_the_analytical_path():
    """``compare_morphologies(..., analytical_only=True)`` builds each config
    on the analytical path, so tool_flat's auto-enabled surface pockets do
    not make ``run(analytical_only=True)`` refuse the run."""
    base = AnalysisConfig(amplitude=0.25, wavelength=16.0, width=12.0)
    res = WrinkleAnalysis.compare_morphologies(
        base, morphologies=("uniform", "tool_flat"), analytical_only=True
    )
    assert set(res) == {"uniform", "tool_flat"}
    assert 0.0 < res["tool_flat"].analytical_knockdown <= 1.0
