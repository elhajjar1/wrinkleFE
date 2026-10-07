"""The FE-strength caveat reaches users, and its claims stay true.

Validation showed the FE strength outputs err on the unsafe side where it
matters most: FE LaRC05 over-predicted retained strength on the most severe
Li (2025) wrinkles, and the crack-band progressive-damage model
over-predicted the most severe wrinkle. That was documented only internally
while the app and the NCR summary showed FE strength numbers with no
qualification, next to a tool pitched for scrap/repair/accept decisions.

Two kinds of test:

* **Delivery** (fast): the caveat appears in the NCR summary (data,
  Markdown and therefore PDF) whenever an FE or progressive-damage block
  does, and in the app beside the FE strength metric.
* **Truth** (the FE half is ``slow``): the numbers the caveat quotes are
  still what the code produces. If someone improves the FE path, these fail
  and the caveat gets rewritten instead of going stale in the
  conservative-looking direction, or worse, the other one.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from wrinklefe.io.export import (
    FE_STRENGTH_CAVEAT,
    PROGRESSIVE_DAMAGE_CAVEAT,
    build_analysis_summary,
    render_summary_markdown,
    render_summary_pdf,
)

_ROOT = Path(__file__).resolve().parent.parent

_DEFECT = {
    "amplitude_mm": 0.5, "wavelength_mm": 16.0, "width_mm": 12.0,
    "morphology": "stack", "loading": "compression",
    "ply_thickness_mm": 0.183, "n_plies": 8,
    "layup": [0, 45, -45, 90, 90, -45, 45, 0], "material_name": "IM7_8552",
}
_ANALYTICAL = {
    "analytical_knockdown": 0.62, "analytical_strength_MPa": 990.0,
    "damage_index": 0.38, "max_angle_deg": 11.0,
    "effective_angle_deg": 9.0, "morphology_factor": 1.0,
}
_FE = {
    "modulus_retention": 0.97, "modulus_retention_global": 0.96,
    "retention_factors": {"larc05": 0.81, "hashin": 0.85},
    "critical_criterion": "larc05", "critical_mode": "fiber_kinking",
    "critical_ply": 3,
}
_PROGRESSIVE = {
    "knockdown": 0.74, "strength_MPa": 1180.0,
    "pristine_strength_MPa": 1590.0,
}


def _summary(*, fe=None, progressive=None) -> dict:
    eng = dict(_ANALYTICAL, fe=fe)
    if progressive is not None:
        eng["progressive"] = progressive
    return build_analysis_summary(defect=dict(_DEFECT), engineering=eng)


def _assessment(summary: dict) -> dict:
    return summary["engineering_analysis"]


# ----------------------------------------------------------------------
# Delivery: the NCR summary
# ----------------------------------------------------------------------

class TestNcrSummary:
    def test_fe_block_carries_the_caveat(self):
        fe_block = _assessment(_summary(fe=dict(_FE)))["finite_element"]
        assert fe_block["caveat"] == FE_STRENGTH_CAVEAT

    def test_fe_caveat_is_rendered_beside_the_fe_numbers(self):
        md = render_summary_markdown(
            _summary(fe=dict(_FE), progressive=dict(_PROGRESSIVE))
        )
        fe_at = md.index("**Finite-element evaluation**")
        caveat_at = md.index(f"> **Caveat:** {FE_STRENGTH_CAVEAT}")
        next_block_at = md.index("**Progressive-damage evaluation**")
        # Inside the FE block: after its numbers, before the next block.
        assert fe_at < caveat_at < next_block_at

    def test_progressive_block_carries_its_own_caveat(self):
        s = _summary(fe=dict(_FE), progressive=dict(_PROGRESSIVE))
        prog = _assessment(s)["progressive_damage"]
        assert prog["caveat"] == PROGRESSIVE_DAMAGE_CAVEAT
        assert f"> **Caveat:** {PROGRESSIVE_DAMAGE_CAVEAT}" in (
            render_summary_markdown(s)
        )

    def test_an_analytical_only_summary_carries_no_fe_caveat(self):
        """The warning qualifies the FE number; with no FE number it would
        only teach readers to skip it."""
        md = render_summary_markdown(_summary(fe=None))
        assert "Caveat:" not in md

    def test_a_summary_from_before_the_caveat_still_renders_it(self):
        """A summary dict saved by an older version has no ``caveat`` key;
        re-rendering it must still warn, because the number is the same."""
        s = _summary(fe=dict(_FE), progressive=dict(_PROGRESSIVE))
        del _assessment(s)["finite_element"]["caveat"]
        del _assessment(s)["progressive_damage"]["caveat"]
        md = render_summary_markdown(s)
        assert FE_STRENGTH_CAVEAT in md
        assert PROGRESSIVE_DAMAGE_CAVEAT in md

    def test_the_pdf_renders_with_the_caveat(self):
        """The PDF is built from the Markdown, so this guards the renderer
        against choking on the quote line rather than re-checking text."""
        pdf = render_summary_pdf(_summary(fe=dict(_FE)))
        assert pdf[:5] == b"%PDF-" and len(pdf) > 1000

    def test_the_caveat_never_changes_the_disposition(self):
        """The disposition stays on the analytical knockdown: adding (or
        omitting) an FE block must not move it."""
        with_fe = _summary(fe=dict(_FE))["disposition_recommendation"]
        without = _summary(fe=None)["disposition_recommendation"]
        assert with_fe["severity"] == without["severity"]


# ----------------------------------------------------------------------
# Delivery: the app
# ----------------------------------------------------------------------

@pytest.mark.viz
def test_the_app_shows_the_caveat_with_fe_results():
    pytest.importorskip("streamlit", reason="Streamlit not installed.")
    import matplotlib

    matplotlib.use("Agg")
    from streamlit.testing.v1 import AppTest

    from wrinklefe.analysis import AnalysisConfig

    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    at = AppTest.from_file(str(_ROOT / "app.py"), default_timeout=300)
    at.session_state["_wf_acknowledged"] = True
    at.run()
    # A coarse FE case, seeded the way an uploaded config is.
    at.session_state["_pending_config"] = AnalysisConfig(
        amplitude=0.4, wavelength=16.0, analytical_only=False,
        nx=8, ny=4, nz_per_ply=1,
    )
    at.run()
    for button in at.button:
        if button.label == "Run analysis":
            button.click()
            break
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.session_state["results"].get("fe") is not None
    warnings_shown = [w.value for w in at.warning]
    assert any(FE_STRENGTH_CAVEAT in w for w in warnings_shown), warnings_shown


@pytest.mark.viz
def test_the_app_shows_no_fe_caveat_on_an_analytical_run():
    pytest.importorskip("streamlit", reason="Streamlit not installed.")
    import matplotlib

    matplotlib.use("Agg")
    from streamlit.testing.v1 import AppTest

    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    at = AppTest.from_file(str(_ROOT / "app.py"), default_timeout=300)
    at.session_state["_wf_acknowledged"] = True
    at.run()
    for checkbox in at.checkbox:
        if checkbox.key == "sb_analytical_only":
            checkbox.set_value(True)
    at.run()
    for button in at.button:
        if button.label == "Run analysis":
            button.click()
            break
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert not any(FE_STRENGTH_CAVEAT in w.value for w in at.warning)


# ----------------------------------------------------------------------
# Truth: the caveat's numbers are still what the code produces
# ----------------------------------------------------------------------

def _pct(text: str) -> list[int]:
    return [int(m) for m in re.findall(r"[+]?(\d+)%", text)]


class TestProgressiveClaimsMatchTheLedger:
    """Fast: the progressive caveat quotes pinned ledger values."""

    @pytest.fixture(scope="class")
    def recipes(self):
        ledger = json.loads(
            (_ROOT / "tests" / "test_validation" / "ledger.json").read_text()
        )
        return {r["name"]: r for r in ledger["progressive_damage"]["recipes"]}

    @staticmethod
    def _err(case: dict) -> float:
        pred, meas = case["expected_progressive_kd"], case["measured_kd"]
        return (pred - meas) / meas * 100.0

    def test_the_quoted_overshoots_are_the_pinned_ones(self, recipes):
        calibrated = next(
            c for c in recipes["nx16_calibrated"]["cases"]
            if c["case"] == "S-M-2"
        )
        refined = next(
            c for c in recipes["nx36_refined"]["cases"]
            if c["case"] == "S-M-2"
        )
        assert _pct(PROGRESSIVE_DAMAGE_CAVEAT) == [
            round(self._err(calibrated)), round(self._err(refined)),
        ]

    def test_the_milder_wrinkles_really_are_under_predicted(self, recipes):
        milder = [
            c for c in recipes["nx16_calibrated"]["cases"]
            if c["case"] != "S-M-2"
        ]
        assert milder and all(self._err(c) < 0 for c in milder)

    def test_the_quoted_mesh_is_the_calibrated_one(self, recipes):
        r = recipes["nx16_calibrated"]
        assert f"nx = {r['nx']}, nz_per_ply = {r['nz_per_ply']}" in (
            PROGRESSIVE_DAMAGE_CAVEAT
        )


@pytest.mark.slow
def test_the_fe_larc05_claims_still_hold():
    """Six FE solves. The caveat says the FE over-predicts the two most
    severe wrinkles "by up to about 30%" and under-predicts the milder
    ones; when the FE path changes, this fails and the caveat is rewritten
    to match."""
    sys.path.insert(0, str(_ROOT / "validation"))
    import strength_error_summary as ses

    rows = ses.fe_larc05_errors()
    assert len(rows) == 6
    non_conservative = {k for k, (_kd, _m, e) in rows.items() if e > 0.5}
    # The two most severe: 1.5 mm at 20 deg, and 30 deg.
    assert non_conservative == {"S-M-2", "S-M-3"}, rows
    worst = max(e for _kd, _m, e in rows.values())
    assert 25.0 <= worst <= 35.0, rows
    assert "about 30%" in FE_STRENGTH_CAVEAT
    milder = [e for k, (_kd, _m, e) in rows.items() if k not in non_conservative]
    assert all(e < 0.5 for e in milder), rows
