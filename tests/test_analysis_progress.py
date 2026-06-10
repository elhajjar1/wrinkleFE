"""Tests for the progress_callback parameter on WrinkleAnalysis.run().

Verifies the fix for issue #74: ``WrinkleAnalysis.run()`` accepts an
optional ``progress_callback`` so the Streamlit UI (and other callers)
can display a granular per-phase progress bar instead of an opaque
spinner.  The callback must be invoked multiple times with
monotonically non-decreasing fractions in ``[0, 1]`` and must finish at
exactly ``1.0``.  Calls without the kwarg must continue to work
identically.
"""

from __future__ import annotations

import pytest

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary


def _tiny_analytical_config() -> AnalysisConfig:
    """Smallest possible analytical-only config — runs in well under 1 s."""
    return AnalysisConfig(
        amplitude=0.183,
        wavelength=16.0,
        width=12.0,
        morphology="stack",
        loading="compression",
        material=MaterialLibrary().get("IM7_8552"),
        angles=[0, 45, -45, 90, 90, -45, 45, 0],
        interface_1=3,
        interface_2=4,
        ply_thickness=0.183,
        applied_strain=-0.005,
        analytical_only=True,
        verbose=False,
    )


def _tiny_fe_config() -> AnalysisConfig:
    """Smallest viable FE config so the test stays under a second."""
    return AnalysisConfig(
        amplitude=0.183,
        wavelength=16.0,
        width=12.0,
        morphology="stack",
        loading="compression",
        material=MaterialLibrary().get("IM7_8552"),
        angles=[0, 45, -45, 90, 90, -45, 45, 0],
        interface_1=3,
        interface_2=4,
        ply_thickness=0.183,
        applied_strain=-0.005,
        nx=4,
        ny=2,
        nz_per_ply=1,
        domain_width=8.0,
        analytical_only=False,
        verbose=False,
    )


def _assert_progress_trace_valid(trace: list[tuple[str, float]]) -> None:
    """Common invariants every progress trace must satisfy."""
    assert len(trace) >= 2, (
        f"progress_callback should fire at least twice (got {len(trace)}): {trace}"
    )

    fractions = [frac for _, frac in trace]
    for frac in fractions:
        assert 0.0 <= frac <= 1.0, (
            f"progress fraction out of [0, 1]: {frac} in {trace}"
        )

    # Monotonically non-decreasing.
    for prev, nxt in zip(fractions, fractions[1:]):
        assert nxt >= prev, (
            f"progress fraction decreased: {prev} -> {nxt} in {trace}"
        )

    # Final report must be exactly 1.0.
    assert fractions[-1] == pytest.approx(1.0), (
        f"final progress must be 1.0, got {fractions[-1]}: {trace}"
    )

    # Every label must be a non-empty string.
    for label, _ in trace:
        assert isinstance(label, str) and label, (
            f"progress label must be a non-empty string: {label!r}"
        )


def test_progress_callback_analytical_only():
    """Callback fires through the analytical-only path and ends at 1.0."""
    trace: list[tuple[str, float]] = []

    def cb(label: str, fraction: float) -> None:
        trace.append((label, fraction))

    cfg = _tiny_analytical_config()
    WrinkleAnalysis(cfg).run(progress_callback=cb)

    _assert_progress_trace_valid(trace)


def test_progress_callback_full_fe_path():
    """Callback fires through every FE phase boundary and ends at 1.0."""
    trace: list[tuple[str, float]] = []

    def cb(label: str, fraction: float) -> None:
        trace.append((label, fraction))

    cfg = _tiny_fe_config()
    WrinkleAnalysis(cfg).run(progress_callback=cb)

    _assert_progress_trace_valid(trace)
    # The FE path must surface strictly more phases than analytical-only
    # (mesh, solve, failure, retention all in addition to laminate/analytical).
    assert len(trace) >= 5, (
        f"FE path should report >=5 phases, got {len(trace)}: {trace}"
    )


def test_run_without_callback_is_unaffected():
    """Existing callers that do not pass progress_callback see no change."""
    cfg = _tiny_analytical_config()
    # Must not raise and must return a populated result.
    result = WrinkleAnalysis(cfg).run()
    assert result is not None
    assert result.analytical_strength_MPa > 0.0


def test_progress_callback_keyword_only_is_optional():
    """progress_callback defaults to None and may be passed positionally as None."""
    cfg = _tiny_analytical_config()
    # Explicit None is equivalent to omitting the argument.
    result = WrinkleAnalysis(cfg).run(progress_callback=None)
    assert result is not None
