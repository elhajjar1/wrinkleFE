"""The requirements files must never contradict ``pyproject.toml``.

``pyproject.toml`` is the source of truth for *install metadata* — the
constraints a plain ``pip install wrinklefe`` resolves against. The
``requirements*.txt`` files are the deploy/CI pin files (Streamlit Cloud,
the test lane); they deliberately carry **tighter** floors and upper
bounds than the library metadata, so the two are not required to be
identical.

What must hold is that they cannot *contradict*: every version the
requirements files allow has to be a version ``pyproject`` also allows.
Until this test existed that was enforced only by a
"keep versions in sync" comment, which is how ``requirements.txt``
drifted to ``numpy>=2.1`` while ``pyproject`` still declared
``numpy>=1.24`` (issue #376).

The check is deliberately narrow: packages that appear only in a
requirements file (the optional Google-Sheets usage-logging deps, which
are imported fail-soft and are intentionally absent from the install
metadata) are skipped rather than flagged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

packaging_requirements = pytest.importorskip("packaging.requirements")
packaging_specifiers = pytest.importorskip("packaging.specifiers")
packaging_version = pytest.importorskip("packaging.version")

Requirement = packaging_requirements.Requirement
SpecifierSet = packaging_specifiers.SpecifierSet
Version = packaging_version.Version

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on the 3.10 matrix cell only
    tomllib = pytest.importorskip(
        "tomli",
        reason="TOML parsing needs stdlib tomllib (3.11+) or the tomli backport",
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
REQUIREMENTS = REPO_ROOT / "requirements.txt"
REQUIREMENTS_TEST = REPO_ROOT / "requirements-test.txt"


def _canonical(name: str) -> str:
    """Normalise a distribution name for comparison (PEP 503)."""
    return name.lower().replace("_", "-").replace(".", "-")


def _parse_requirements_file(path: Path) -> dict[str, Requirement]:
    """Parse a ``requirements.txt``-style file into {canonical name: Requirement}."""
    parsed: dict[str, Requirement] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        req = Requirement(line)
        parsed[_canonical(req.name)] = req
    return parsed


def _pyproject_constraints() -> dict[str, SpecifierSet]:
    """Collect every constraint ``pyproject`` declares, keyed by package.

    Covers ``project.dependencies`` and every ``optional-dependencies``
    extra. When a package appears in more than one place the specifier
    sets are intersected (``&``), which is what an installer would
    resolve against.
    """
    data = tomllib.loads(PYPROJECT.read_text())
    project = data["project"]

    groups = [project.get("dependencies", [])]
    groups.extend(project.get("optional-dependencies", {}).values())

    constraints: dict[str, SpecifierSet] = {}
    for group in groups:
        for entry in group:
            req = Requirement(entry)
            # Self-referential extras such as `wrinklefe[streamlit,...]`
            # carry no version information worth checking.
            if _canonical(req.name) == "wrinklefe":
                continue
            key = _canonical(req.name)
            constraints[key] = (
                req.specifier if key not in constraints
                else constraints[key] & req.specifier
            )
    return constraints


def _floor(specifier: SpecifierSet) -> Version | None:
    """Return the lowest version a specifier set admits, if it states one."""
    lows = [
        Version(spec.version)
        for spec in specifier
        if spec.operator in (">=", "==", "~=")
    ]
    return max(lows) if lows else None


@pytest.mark.parametrize("requirements_path", [REQUIREMENTS, REQUIREMENTS_TEST],
                         ids=["requirements.txt", "requirements-test.txt"])
def test_requirements_floor_is_allowed_by_pyproject(requirements_path):
    """Every requirements floor must satisfy the pyproject constraint.

    A floor *below* pyproject's would mean the deploy file installs a
    version the library metadata declares unsupported.
    """
    pyproject = _pyproject_constraints()
    reqs = _parse_requirements_file(requirements_path)

    checked = 0
    for name, req in reqs.items():
        if name not in pyproject:
            # Optional, fail-soft deps (gspread / google-auth) that are
            # intentionally not part of the install metadata.
            continue
        floor = _floor(req.specifier)
        assert floor is not None, (
            f"{requirements_path.name} pins {name} with no lower bound "
            f"({req}); give it a floor so this check has something to verify"
        )
        assert floor in pyproject[name], (
            f"{requirements_path.name} allows {name}=={floor}, which "
            f"pyproject.toml's constraint '{name}{pyproject[name]}' rejects. "
            f"pyproject is the source of truth for install metadata — raise "
            f"its floor or the requirements floor so the two agree."
        )
        checked += 1

    assert checked, (
        f"no package in {requirements_path.name} was matched against "
        f"pyproject.toml — the parsing has probably broken"
    )


def test_requirements_floors_are_at_least_pyproject_floors():
    """requirements.txt may tighten pyproject's floors, never loosen them."""
    pyproject = _pyproject_constraints()
    for name, req in _parse_requirements_file(REQUIREMENTS).items():
        if name not in pyproject:
            continue
        req_floor = _floor(req.specifier)
        meta_floor = _floor(pyproject[name])
        if meta_floor is None or req_floor is None:
            continue
        assert req_floor >= meta_floor, (
            f"requirements.txt floors {name} at {req_floor}, below "
            f"pyproject.toml's {meta_floor}"
        )


def test_streamlit_extra_packages_are_all_pinned_in_requirements():
    """The `streamlit` extra is what the deployed app runs — keep it listed.

    The pyproject comment promises requirements.txt tracks the extra; this
    turns that promise into a check.
    """
    data = tomllib.loads(PYPROJECT.read_text())
    extra = data["project"]["optional-dependencies"]["streamlit"]
    listed = _parse_requirements_file(REQUIREMENTS)
    for entry in extra:
        name = _canonical(Requirement(entry).name)
        assert name in listed, (
            f"pyproject's `streamlit` extra declares {name} but "
            f"requirements.txt does not pin it"
        )


def test_requirements_test_does_not_hard_pin_pytest():
    """A single ``pytest==X.Y.Z`` pin silently blocks routine upgrades.

    Issue #376: the file pinned ``pytest==9.1.1`` while pyproject's `dev`
    extra allowed ``pytest>=7.0``, so the two lanes could install
    different majors with no signal. A bounded range keeps the lanes
    compatible without freezing the patch version.
    """
    pytest_req = _parse_requirements_file(REQUIREMENTS_TEST)["pytest"]
    operators = {spec.operator for spec in pytest_req.specifier}
    assert "==" not in operators, (
        "requirements-test.txt hard-pins pytest; use a bounded range "
        "(e.g. 'pytest>=9,<10') so patch upgrades are not blocked"
    )
    assert ">=" in operators, "pytest needs a lower bound in requirements-test.txt"
