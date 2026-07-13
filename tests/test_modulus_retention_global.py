"""Regression tests for issue #328: reaction-based (global) FE modulus
retention.

``AnalysisResults.modulus_retention`` is the FE axial-modulus knockdown
from a *local* fibre-direction-stress proxy (``<sigma_11> / strain``).
That proxy averages the local fibre stress rather than the coupon's
global axial response, so it over-predicts the modulus retention vs
experiment.  Issue #328 adds ``modulus_retention_global``: a true
coupon-level ``E_x / E_x0`` from the global reaction response,
``E_eff = sigma_nominal / strain`` with ``sigma_nominal = R / A`` (total
axial reaction on the loaded ``x_max`` face over the cross-section area),
wrinkled vs pristine — reusing the progressive-damage solver's
``reaction = sum((K @ u)[xmax_dofs])`` pattern.

These tests pin:

* the new field is populated and physically bounded ``(0, 1]``;
* it equals ~1.0 for a flat / pristine (zero-amplitude) coupon;
* it is **strictly below** the local ``sigma_11`` proxy for a
  representative wrinkled UD case — the documented over-prediction
  direction (the local proxy reads too high);
* the original ``modulus_retention`` proxy is unchanged (backward
  compatibility — existing pins keep working).
"""

from __future__ import annotations

import pytest

from wrinklefe.analysis import AnalysisConfig, AnalysisResults, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary

pytestmark = pytest.mark.slow

ML = MaterialLibrary()

# A through-thickness ("uniform") UD wrinkle: the wrinkle penetrates the
# full laminate so the coupon's global axial stiffness genuinely drops,
# making the local-vs-global gap robust and sign-stable.  Carbon/epoxy
# IM6G/3501-6 (dataset G) at a single localized wavelength.
_MAT = "IM6G_3501_6"
_NX, _NY, _NZ = 24, 2, 2


def _run(amplitude: float, *, morphology: str = "uniform") -> AnalysisResults:
    wavelength = 8.1
    cfg = AnalysisConfig(
        amplitude=amplitude,
        wavelength=wavelength,
        width=wavelength / 2.0,
        morphology=morphology,
        loading="compression",
        material=ML.get(_MAT),
        angles=[0.0] * 14,
        ply_thickness=0.44,
        nx=_NX,
        ny=_NY,
        nz_per_ply=_NZ,
        domain_length=wavelength,
        domain_width=10.0,
        applied_strain=-0.01,
        wrinkle_z_position=0.5,
    )
    return WrinkleAnalysis(cfg).run()


@pytest.fixture(scope="module")
def flat_result() -> AnalysisResults:
    return _run(amplitude=0.0)


@pytest.fixture(scope="module")
def wrinkled_result() -> AnalysisResults:
    return _run(amplitude=0.75)


def test_global_field_default_is_one() -> None:
    """A freshly constructed ``AnalysisResults`` defaults the new global
    modulus retention to 1.0 (no knockdown), so analytical-only / un-run
    results stay backward compatible."""
    res = AnalysisResults(config=AnalysisConfig(material=ML.get(_MAT)))
    assert res.modulus_retention_global == 1.0


def test_global_modulus_populated_and_bounded(
    wrinkled_result: AnalysisResults,
) -> None:
    """The global reaction-based modulus retention is populated and lies
    in the physical range ``(0, 1]`` for a wrinkled coupon."""
    mr = wrinkled_result.modulus_retention_global
    assert mr is not None
    assert 0.0 < mr <= 1.0


def test_global_modulus_flat_is_unity(flat_result: AnalysisResults) -> None:
    """For a flat (zero-amplitude / pristine) coupon the wrinkled and
    pristine reaction-based moduli are the same mesh, so the global
    retention is ~1.0."""
    assert flat_result.modulus_retention_global == pytest.approx(1.0, abs=1e-3)
    # The local proxy is likewise ~1.0 on a flat coupon.
    assert flat_result.modulus_retention == pytest.approx(1.0, abs=1e-3)


def test_global_modulus_below_local_proxy_for_wrinkle(
    wrinkled_result: AnalysisResults,
) -> None:
    """Core acceptance of #328: for a representative wrinkled UD case the
    global (reaction-based) modulus retention is strictly **below** the
    local sigma_11 proxy — the proxy over-predicts the retained stiffness,
    which is exactly the bug the global estimator corrects."""
    local = wrinkled_result.modulus_retention
    glob = wrinkled_result.modulus_retention_global
    # Both should register a real knockdown for this through-thickness
    # wrinkle (sanity: the case actually exercises the difference).
    assert local < 1.0
    assert glob < 1.0
    assert glob < local, (
        f"global modulus retention {glob:.4f} should be below the local "
        f"sigma_11 proxy {local:.4f} (the proxy over-predicts)"
    )


def test_local_proxy_backward_compatible(
    wrinkled_result: AnalysisResults,
) -> None:
    """Adding the global field must not disturb the existing local
    ``modulus_retention`` proxy: it stays populated and bounded."""
    local = wrinkled_result.modulus_retention
    assert 0.0 < local <= 1.0
