"""The software DOI is cited consistently (issue #284).

Zenodo minted the concept DOI on the 1.2.0 release. It now appears in four
places: ``CITATION.cff`` (``doi`` and ``identifiers``) and the README
(badge, plain-text citation, BibTeX). A citation that names a different DOI
in each place splits whatever citation count the project accrues, and
nothing visible breaks when one copy is edited and the others are not, so
this pins them to each other rather than to a literal.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_ZENODO_DOI = re.compile(r"10\.5281/zenodo\.\d+")


def _cff_text() -> str:
    return (_ROOT / "CITATION.cff").read_text(encoding="utf-8")


def _concept_doi() -> str:
    match = re.search(r'^doi:\s*"?([^"\s]+)"?\s*$', _cff_text(), re.MULTILINE)
    assert match, "CITATION.cff has no top-level doi"
    return match.group(1)


def test_the_cff_doi_is_a_zenodo_doi():
    assert _ZENODO_DOI.fullmatch(_concept_doi()), _concept_doi()


def test_the_cff_identifier_matches_its_doi():
    values = re.findall(r'^\s+value:\s*"?([^"\s]+)"?\s*$', _cff_text(), re.MULTILINE)
    assert values == [_concept_doi()], values


def test_the_readme_cites_the_same_doi_everywhere():
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    doi = _concept_doi()
    assert f"zenodo.org/badge/DOI/{doi}.svg" in readme, "badge"
    assert f"Zenodo. https://doi.org/{doi}" in readme, "plain-text citation"
    assert f"doi = {{{doi}}}" in readme, "BibTeX entry"
    # Any other Zenodo DOI in the README must be a version DOI named as
    # such, never a second, competing concept DOI.
    others = set(_ZENODO_DOI.findall(readme)) - {doi}
    for other in others:
        context = readme[readme.index(other) - 200: readme.index(other)]
        assert "1.2.0 is" in context or "version" in context.lower(), other


def test_no_placeholder_or_stale_note_remains():
    for name in ("README.md", "CITATION.cff", "CONTRIBUTING.md"):
        text = (_ROOT / name).read_text(encoding="utf-8")
        assert "zenodo.XXXXXXX" not in text, name
        assert "does not have one yet" not in text, name
