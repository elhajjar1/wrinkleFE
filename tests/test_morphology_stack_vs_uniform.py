"""Pin the documented relationship between the ``stack`` and ``uniform``
morphology names (GitHub issue #159).

``stack`` is a *dual-wrinkle inter-ply phase* morphology (phi = 0), while
``uniform`` is a *single-wrinkle through-thickness amplitude decay* mode
(no decay). They live on two orthogonal model axes. In the analytical
pipeline neither the aggregate morphology factor (phi=0 -> 1.0 for stack;
N=1 -> 1.0 for uniform) nor the geometry-only angle
``theta = arctan(2*pi*A/lambda)`` depends on the through-thickness decay
mode, so every analytical scalar is *identical* for the two names. They
diverge only in the FE/mesh path, where ``decay_mode`` changes the
through-thickness amplitude field. These tests pin the real observed
behavior, not a guess. See ARCHITECTURE.md "Morphology Axes".
"""

import numpy as np
import pytest

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.morphology import (
    MORPHOLOGY_PHASES,
    SINGLE_WRINKLE_MODES,
    WrinkleConfiguration,
)


# Representative (amplitude, wavelength, width) configs in mm.
_CONFIGS = [
    pytest.param(0.366, 16.0, 12.0, id="A=0.366_lam=16_w=12"),
    pytest.param(0.6, 20.0, 10.0, id="A=0.6_lam=20_w=10"),
    pytest.param(0.2, 12.0, 8.0, id="A=0.2_lam=12_w=8"),
]

# Metrics compared between the two names (attr -> human label).
_METRICS = {
    "morphology_factor": "morphology_factor",
    "max_angle_rad": "max_angle",
    "effective_angle_rad": "effective_angle",
    "analytical_knockdown": "analytical_knockdown",
    "damage_index": "damage_index",
}


def _analytical(morphology, amplitude, wavelength, width):
    cfg = AnalysisConfig(
        morphology=morphology,
        amplitude=amplitude,
        wavelength=wavelength,
        width=width,
        analytical_only=True,
    )
    return WrinkleAnalysis(cfg).run()


# ----------------------------------------------------------------------
# Axis classification: the two names belong to different model axes.
# ----------------------------------------------------------------------

def test_stack_is_a_dual_wrinkle_phase_morphology():
    """``stack`` is a phase entry (phi = 0), not a decay mode."""
    assert "stack" in MORPHOLOGY_PHASES
    assert MORPHOLOGY_PHASES["stack"] == 0.0
    assert "stack" not in SINGLE_WRINKLE_MODES


def test_uniform_is_a_single_wrinkle_decay_mode():
    """``uniform`` is a through-thickness decay mode, not a phase entry."""
    assert "uniform" in SINGLE_WRINKLE_MODES
    assert "uniform" not in MORPHOLOGY_PHASES


def test_resolved_configurations_have_different_wrinkle_counts():
    """stack -> 2 wrinkles (phase pair); uniform -> 1 wrinkle (decay)."""
    from wrinklefe.core.wrinkle import GaussianSinusoidal

    profile = GaussianSinusoidal(amplitude=0.366, wavelength=16.0, width=12.0)
    stack_cfg = WrinkleConfiguration.from_morphology_name(
        "stack", profile, interface1=10, interface2=13
    )
    uniform_cfg = WrinkleConfiguration.from_morphology_name(
        "uniform", profile, interface1=10, interface2=13
    )
    assert stack_cfg.n_wrinkles() == 2
    assert stack_cfg.decay_mode == "default"
    assert uniform_cfg.n_wrinkles() == 1
    assert uniform_cfg.decay_mode == "uniform"


# ----------------------------------------------------------------------
# Analytical pipeline: stack and uniform are bit-identical.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("amplitude, wavelength, width", _CONFIGS)
@pytest.mark.parametrize("metric", sorted(_METRICS))
def test_stack_and_uniform_analytical_metrics_are_identical(
    metric, amplitude, wavelength, width
):
    """Every analytical scalar is identical for stack vs uniform.

    This is exact equality (Δ = 0), not a tolerance: both resolve to an
    aggregate morphology factor of exactly 1.0 and share the same
    geometry-only ``theta_max``; the through-thickness decay mode never
    enters the analytical computation.
    """
    s = _analytical("stack", amplitude, wavelength, width)
    u = _analytical("uniform", amplitude, wavelength, width)
    sv = getattr(s, metric)
    uv = getattr(u, metric)
    assert sv == uv, (
        f"{_METRICS[metric]} differs: stack={sv!r} uniform={uv!r} "
        f"(expected exact equality on the analytical axis)"
    )


@pytest.mark.parametrize("amplitude, wavelength, width", _CONFIGS)
def test_both_names_give_unit_morphology_factor(
    amplitude, wavelength, width
):
    """stack (phi=0) and uniform (N=1) both yield M_f == 1.0 exactly."""
    s = _analytical("stack", amplitude, wavelength, width)
    u = _analytical("uniform", amplitude, wavelength, width)
    assert s.morphology_factor == 1.0
    assert u.morphology_factor == 1.0


# ----------------------------------------------------------------------
# stack is non-trivial on its own (phase) axis.
# ----------------------------------------------------------------------

def test_stack_differs_from_other_phase_morphologies():
    """``stack`` is meaningful on the phase axis: convex/concave differ.

    Pins the documented contrast (convex M_f ~ 0.750, concave ~ 1.334
    versus stack 1.0) so a regression that collapses the phase axis is
    caught.
    """
    stack = _analytical("stack", 0.366, 16.0, 12.0)
    convex = _analytical("convex", 0.366, 16.0, 12.0)
    concave = _analytical("concave", 0.366, 16.0, 12.0)

    assert stack.morphology_factor == 1.0
    assert convex.morphology_factor < stack.morphology_factor
    assert concave.morphology_factor > stack.morphology_factor
    # Pin the measured values reported in ARCHITECTURE.md.
    assert convex.morphology_factor == pytest.approx(0.7498, abs=1e-3)
    assert concave.morphology_factor == pytest.approx(1.3338, abs=1e-3)


# ----------------------------------------------------------------------
# FE/mesh path: the two names DO diverge (decay mode matters there).
# ----------------------------------------------------------------------

def test_stack_and_uniform_diverge_in_fe_mesh_path():
    """In the FE path the decay mode changes the mesh fiber-angle field.

    ``stack`` (default linear decay, two interface plies at full
    amplitude, RSS-combined) yields a larger mesh max fiber angle than
    ``uniform`` (single wrinkle, no decay) for this coarse mesh. We pin
    the qualitative inequality plus a loose band around the observed
    values, so the documented divergence is locked without being brittle
    to minor meshing changes.
    """
    common = dict(
        amplitude=0.366,
        wavelength=16.0,
        width=12.0,
        analytical_only=False,
        nx=12,
        ny=2,
        nz_per_ply=1,
    )
    s = WrinkleAnalysis(AnalysisConfig(morphology="stack", **common)).run(
        analytical_only=False
    )
    u = WrinkleAnalysis(AnalysisConfig(morphology="uniform", **common)).run(
        analytical_only=False
    )
    s_deg = np.degrees(s.mesh_max_angle_rad)
    u_deg = np.degrees(u.mesh_max_angle_rad)

    # They must differ in the FE path (unlike the analytical path).
    assert s_deg != pytest.approx(u_deg, abs=0.5)
    # stack peaks higher than uniform for this configuration.
    assert s_deg > u_deg
    # Loose bands around the observed values (~10.36 and ~7.33 deg).
    assert s_deg == pytest.approx(10.36, abs=1.5)
    assert u_deg == pytest.approx(7.33, abs=1.5)
