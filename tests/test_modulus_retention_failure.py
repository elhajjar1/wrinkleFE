"""Regression tests for issue #374: a failed FE modulus-retention
computation must be *loud* and *distinguishable* from a genuinely computed
``1.0``.

Previously both the local (σ₁₁ proxy) and global (reaction-based) modulus-
retention blocks swallowed every exception and silently set the no-knockdown
value ``1.0`` (the local block logged nothing at all), so any bug in the FE
stiffness-knockdown path read as a clean bill of health.

The fix keeps the field a plain ``float`` (many consumers call ``float(...)``
/ format it) but

* logs a ``WARNING`` with ``exc_info`` when the computation raises, and
* sets a companion boolean flag (``modulus_retention_failed`` /
  ``modulus_retention_global_failed``) that summary(), the JSON export and
  these tests use to tell a fallback ``1.0`` from a real one.

The flag is only serialized when it is ``True`` (valid runs stay byte-
identical — ledger zero-drift).
"""
from __future__ import annotations

import logging

import pytest

from wrinklefe.analysis import AnalysisConfig, AnalysisResults, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary
from wrinklefe.io.results import _knockdown_factors

ML = MaterialLibrary()


# --------------------------------------------------------------------------
# Fast unit tests — defaults and serialization (no FE run)
# --------------------------------------------------------------------------

def test_failure_flags_default_false() -> None:
    """A freshly constructed result has neither failure flag set, so a
    default ``1.0`` is unambiguously *not* a failure."""
    res = AnalysisResults(config=AnalysisConfig(material=ML.get("IM6G_3501_6")))
    assert res.modulus_retention_failed is False
    assert res.modulus_retention_global_failed is False


def test_serialization_omits_flags_when_not_failed() -> None:
    """For a valid run the failure keys are absent from the export — this
    is what preserves ledger zero-drift."""
    res = AnalysisResults(config=AnalysisConfig(material=ML.get("IM6G_3501_6")))
    out = _knockdown_factors(res)
    assert "modulus_retention_failed" not in out
    assert "modulus_retention_global_failed" not in out


def test_serialization_surfaces_flags_when_failed() -> None:
    """A failed computation is machine-distinguishable: the failure keys
    appear in the export even though the value itself is the ``1.0``
    fallback."""
    res = AnalysisResults(config=AnalysisConfig(material=ML.get("IM6G_3501_6")))
    res.modulus_retention = 1.0
    res.modulus_retention_failed = True
    res.modulus_retention_global = 1.0
    res.modulus_retention_global_failed = True
    out = _knockdown_factors(res)
    assert out["modulus_retention"] == pytest.approx(1.0)
    assert out["modulus_retention_failed"] is True
    assert out["modulus_retention_global_failed"] is True


# --------------------------------------------------------------------------
# Integration tests — the except paths actually fire (monkeypatched raise)
# --------------------------------------------------------------------------

pytestmark_slow = pytest.mark.slow

_MAT = "IM6G_3501_6"


def _small_cfg() -> AnalysisConfig:
    wavelength = 8.1
    return AnalysisConfig(
        amplitude=0.75,
        wavelength=wavelength,
        width=wavelength / 2.0,
        morphology="uniform",
        loading="compression",
        material=ML.get(_MAT),
        angles=[0.0] * 14,
        ply_thickness=0.44,
        nx=24,
        ny=2,
        nz_per_ply=2,
        domain_length=wavelength,
        domain_width=10.0,
        applied_strain=-0.01,
        wrinkle_z_position=0.5,
    )


@pytest.mark.slow
def test_local_failure_is_logged_and_flagged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """When the LOCAL σ₁₁ computation raises, the run warns (with a
    traceback) and flags the fallback rather than silently reporting a
    healthy ``1.0``."""
    def _boom(self, *a, **k):  # noqa: ANN001, ANN002, ANN003
        raise RuntimeError("injected local-modulus failure")

    monkeypatch.setattr(WrinkleAnalysis, "_local_modulus_retention", _boom)
    with caplog.at_level(logging.WARNING, logger="wrinklefe.analysis"):
        res = WrinkleAnalysis(_small_cfg()).run()

    assert res.modulus_retention_failed is True
    assert res.modulus_retention == pytest.approx(1.0)
    assert any(
        "modulus-retention computation failed" in r.message
        and r.levelno == logging.WARNING
        for r in caplog.records
    )
    # the traceback of the injected error is attached (exc_info=True)
    assert any(r.exc_info is not None for r in caplog.records)
    # summary() renders the failure so it is human-distinguishable from 1.0
    assert "computation failed" in res.summary()


@pytest.mark.slow
def test_global_failure_is_logged_and_flagged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """When the GLOBAL reaction-based computation raises, the run warns and
    flags the fallback."""
    def _boom(self, *a, **k):  # noqa: ANN001, ANN002, ANN003
        raise RuntimeError("injected global-modulus failure")

    monkeypatch.setattr(WrinkleAnalysis, "_reaction_modulus", _boom)
    with caplog.at_level(logging.WARNING, logger="wrinklefe.analysis"):
        res = WrinkleAnalysis(_small_cfg()).run()

    assert res.modulus_retention_global_failed is True
    assert res.modulus_retention_global == pytest.approx(1.0)
    assert any(
        "Global reaction-based modulus-retention computation failed"
        in r.message
        and r.levelno == logging.WARNING
        for r in caplog.records
    )
