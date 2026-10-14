"""Claims the docs make about the material-anchored UD kink-band budget (#451).

``docs/internal/VALIDATION.md`` ("UD kink-band strain budget") and
``docs/interpreting_results.md`` quote the kink-band's signed errors on
the UD carbon datasets after the budget moved to ``S12 / Xc``. These
tests recompute them from the ledger so the text cannot drift from the
code.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_LEDGER = json.loads(
    (_ROOT / "tests" / "test_validation" / "ledger.json").read_text())


def _errors(prefix: str, *, case_prefix: tuple[str, ...] = ("",)) -> list[float]:
    ds = next(d for d in _LEDGER["datasets"] if d["name"].startswith(prefix))
    return [(c["expected_analytical_kd"] - c["measured_kd"])
            / c["measured_kd"] * 100.0
            for c in ds["cases"] if c["case"].startswith(case_prefix)]


def test_thor_ud_and_pilato_within_the_acceptance_band():
    """Issue #451 acceptance: signed mean within -25 %, none unsafe."""
    thor = _errors("thor_2021_ud")
    pilato = _errors("pilato_2022")
    assert statistics.mean(thor) == pytest.approx(-16.6, abs=0.05)
    assert statistics.mean(pilato) == pytest.approx(-20.3, abs=0.05)
    assert min(thor) == pytest.approx(-23.2, abs=0.05)
    assert max(thor) == pytest.approx(-10.0, abs=0.05)
    assert min(pilato) == pytest.approx(-22.9, abs=0.05)
    assert max(pilato) == pytest.approx(-16.7, abs=0.05)
    assert all(e < 0.0 for e in thor + pilato)


def test_quarantined_krishnappa_stays_conservative():
    errs = _errors("krishnappa_2024")
    assert statistics.mean(errs) == pytest.approx(-31.7, abs=0.05)
    assert max(errs) < 0.0
    whole = _errors("krishnappa_2024", case_prefix=("L-WL", "L-WR"))
    assert len(whole) == 12
    assert statistics.mean(whole) == pytest.approx(-37.5, abs=0.05)
    assert min(whole) == pytest.approx(-48.0, abs=0.05)
    assert max(whole) == pytest.approx(-23.4, abs=0.05)


def test_graded_ud_sets_keep_their_errors():
    """Shi (H) and Dataset M's UD tension half are UD-dominated but take
    the graded path, which keeps the layup budget: their errors are the
    ones quoted before the change."""
    shi = _errors("shi_2025_ud")
    assert [round(e, 1) for e in shi] == [0.2, 11.8, 25.9]
    m_ud = _errors("elhajjar_shams_2016_ud")
    assert statistics.mean(m_ud) == pytest.approx(18.8, abs=0.05)
    assert sum(e > 0.5 for e in m_ud) == 2
    assert max(m_ud) == pytest.approx(39.2, abs=0.05)


def test_analytical_parity_count():
    """60 of the 77 analytical cases (quarantined L excluded) are within
    +/-20 %, as docs/interpreting_results.md says (55 before #451, 58
    after it, 60 once the bending bound of #452 is also applied)."""
    sys.path.insert(0, str(_ROOT / "validation"))
    import plot_all_validation as pav

    held_out = pav.quarantined()
    total = inside = 0
    for label, (fn, _c, _m, _method) in pav.DATASETS.items():
        if "(FE)" in label or label in held_out:
            continue
        errs = [abs(p - m) / m for m, p in fn()]
        total += len(errs)
        inside += sum(e <= 0.20 for e in errs)
    assert (inside, total) == (60, 77)
    text = (_ROOT / "docs" / "interpreting_results.md").read_text()
    assert "60 of the 77 analytical cases fall within ±20%" in text


def test_docs_quote_the_moves():
    validation = (_ROOT / "docs" / "internal" / "VALIDATION.md").read_text()
    for claim in ("−57.9% → −16.6%", "−77.5% → −20.3%",
                  "−73.6% → −37.5% (dataset −55.8% → −31.7%)",
                  "55 → 58 of 77"):
        assert claim in validation, claim
    guide = (_ROOT / "docs" / "interpreting_results.md").read_text()
    assert "| J Thor, UD carbon (2) | −16.6%, 0 unsafe |" in guide
    assert "| K Pilato, near-UD carbon (5) | −20.3%, 0 unsafe |" in guide
