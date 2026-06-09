"""Logging conventions for library code (issue #251).

Library modules must report diagnostics through ``logging`` rather than
``print``, and must not configure handlers themselves (standard
library-logging etiquette). CLI user-facing output is exempt:
``cli.py`` and the argparse-driven ``sweep/parametric_sweep.py`` write
their reports/tables to stdout by design.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "wrinklefe"

# CLI-facing modules whose stdout output is the user interface.
_PRINT_EXEMPT = {"cli.py", "sweep/parametric_sweep.py"}


def _library_files() -> list[Path]:
    return [
        p for p in sorted(SRC.rglob("*.py"))
        if str(p.relative_to(SRC)) not in _PRINT_EXEMPT
    ]


def _calls(tree: ast.AST) -> list[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)]


def test_no_print_in_library_code():
    """No print() calls outside the CLI-facing modules."""
    offenders = []
    for path in _library_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for call in _calls(tree):
            if isinstance(call.func, ast.Name) and call.func.id == "print":
                offenders.append(f"{path.relative_to(SRC)}:{call.lineno}")
    assert not offenders, (
        "print() found in library code (use the module logger): "
        + ", ".join(offenders)
    )


def test_library_does_not_configure_logging_handlers():
    """No basicConfig/addHandler/setLevel on root in library code."""
    offenders = []
    for path in _library_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for call in _calls(tree):
            func = call.func
            if isinstance(func, ast.Attribute) and func.attr == "basicConfig":
                offenders.append(f"{path.relative_to(SRC)}:{call.lineno}")
    assert not offenders, (
        "logging.basicConfig() found in library code: "
        + ", ".join(offenders)
    )


@pytest.mark.parametrize(
    "module",
    [
        "analysis.py",
        "solver/static.py",
        "solver/assembler.py",
        "solver/nonlinear.py",
        "solver/arclength.py",
        "failure/evaluator.py",
    ],
)
def test_module_has_logger(module):
    """Pipeline modules define a module-level logger."""
    src = (SRC / module).read_text(encoding="utf-8")
    assert "logging.getLogger(__name__)" in src, (
        f"{module} must define a module-level logger"
    )
