"""Tests for the inverse goal-seek helper (issue #280).

The load-bearing property under test is *not* "brentq finds a root" — it
is that the returned value satisfies the criterion under a real forward
evaluation, and that a curve which does not admit a unique root is
refused rather than answered.  Assertions are on structure (status,
direction, sign_changes, bounds_source) and on the forward round-trip
inequality; physics constants appear only as loose sanity bounds,
because the repo's sanctioned pinning mechanism is
``tests/test_validation/ledger.json`` + ``scripts/validate.py --update``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import wrinklefe.goalseek as gs
from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis, WrinkleSpec
from wrinklefe.core.penetration_gate import GATE_LI2025_VACBAG
from wrinklefe.goalseek import find_critical_value

# --------------------------------------------------------------------------- #
# Stubbed forward model (fast, deterministic, closed-form root)
# --------------------------------------------------------------------------- #

#: Law parameters, monkeypatched per test.
_LAW: dict[str, object] = {}


@dataclass
class _FakeResults:
    config: AnalysisConfig
    analytical_knockdown: float
    analytical_modulus_knockdown: float = 1.0
    analytical_onset_knockdown: float | None = None
    analytical_strength_MPa: float = 0.0
    modulus_retention_global: float = 1.0
    modulus_retention_global_failed: bool = False
    mesh: object | None = None


class _FakeAnalysis:
    """Evaluates the law registered in ``_LAW`` at ``cfg.amplitude``."""

    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, analytical_only=None):
        law = _LAW["law"]
        parameter = _LAW.get("parameter", "amplitude")
        value = getattr(self.cfg, parameter)
        kd = float(law(value))  # type: ignore[operator]
        return _FakeResults(
            config=self.cfg,
            analytical_knockdown=kd,
            analytical_modulus_knockdown=kd,
            analytical_strength_MPa=1200.0 * kd,
        )


@pytest.fixture()
def stubbed(monkeypatch):
    """Route every forward solve to ``_LAW['law']``."""
    monkeypatch.setattr(gs, "WrinkleAnalysis", _FakeAnalysis)
    _LAW.clear()
    _LAW["law"] = _decreasing_law()
    yield _LAW
    _LAW.clear()


def _decreasing_law(floor: float = 0.0, a0: float = 0.2):
    """``kd(A) = floor + (1 - floor) / (1 + A / a0)`` — root in closed form."""

    def law(value: float) -> float:
        return floor + (1.0 - floor) / (1.0 + value / a0)

    return law


def _decreasing_root(target: float, floor: float = 0.0, a0: float = 0.2) -> float:
    """Exact inverse of :func:`_decreasing_law`."""
    return a0 * ((1.0 - floor) / (target - floor) - 1.0)


def _increasing_law(a0: float = 0.2):
    """Mirror of the decreasing law: safer as the parameter grows."""

    def law(value: float) -> float:
        return 1.0 - 1.0 / (1.0 + value / a0)

    return law


def _increasing_root(target: float, a0: float = 0.2) -> float:
    return a0 * (1.0 / (1.0 - target) - 1.0)


def _base(**overrides) -> AnalysisConfig:
    defaults = dict(
        amplitude=0.366, wavelength=16.0, width=12.0,
        morphology="stack", loading="compression",
    )
    defaults.update(overrides)
    return AnalysisConfig(**defaults)


# --------------------------------------------------------------------------- #
# T-1..T-13: stubbed logic
# --------------------------------------------------------------------------- #


def test_closed_form_root_decreasing(stubbed):
    """T-1: the answer matches the analytic root within rtol."""
    result = find_critical_value(_base(), target_knockdown=0.5)
    assert result.status == "converged"
    assert result.direction == "decreasing"
    assert result.critical_value == pytest.approx(
        _decreasing_root(0.5), rel=1e-3
    )


def test_closed_form_root_increasing(stubbed):
    """T-2: an increasing objective returns the *smallest* safe value."""
    stubbed["law"] = _increasing_law()
    result = find_critical_value(_base(), target_knockdown=0.5)
    assert result.status == "converged"
    assert result.direction == "increasing"
    assert result.critical_value == pytest.approx(
        _increasing_root(0.5), rel=1e-3
    )
    assert "Smallest acceptable amplitude" in result.summary()


@pytest.mark.parametrize(
    "target", [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
)
@pytest.mark.parametrize("increasing", [False, True])
def test_safe_side_guarantee(stubbed, target, increasing):
    """T-3: the returned value satisfies the criterion, never merely
    approaches it — the guarantee the whole feature rests on."""
    stubbed["law"] = _increasing_law() if increasing else _decreasing_law()
    result = find_critical_value(_base(), target_knockdown=target)
    assert result.status == "converged"
    assert result.achieved_objective >= target
    assert abs(result.achieved_objective - target) <= result.rtol * target


def test_backoff_actually_runs(stubbed):
    """T-3 (cont.): at least one target needs a real back-off step."""
    backed_off = 0
    for target in (0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3):
        result = find_critical_value(_base(), target_knockdown=target)
        if result.critical_value != result.critical_value_root:
            backed_off += 1
            assert any(
                ev.phase == "backoff" for ev in result.evaluations
            )
    assert backed_off >= 1


def test_backoff_budget_exhausted_is_flat(stubbed):
    """T-4: a near-vertical law cannot be resolved — refuse, don't hang."""

    def cliff(value: float) -> float:
        return 0.9 if value < 1.0 else 0.1

    stubbed["law"] = cliff
    result = find_critical_value(_base(), target_knockdown=0.5)
    assert result.status in {"flat", "non_monotonic"}
    assert result.critical_value is None


def test_target_below_floor_is_no_crossing(stubbed):
    """T-5: nothing in range violates the criterion."""
    stubbed["law"] = _decreasing_law(floor=0.4)
    result = find_critical_value(_base(), target_knockdown=0.3)
    assert result.status == "no_crossing"
    assert result.critical_value is None
    assert result.bounds_source[1] == "laminate_thickness"
    assert "laminate_thickness" in result.message
    assert f"{min(result.scan_objectives):.6g}" in result.message


@pytest.mark.parametrize("increasing", [False, True])
def test_target_above_everything_is_unreachable(stubbed, increasing):
    """T-6: the (sign, direction) table is direction-independent."""
    # Both laws stay well below the target across the whole range, so
    # the residual never changes sign in either direction.
    stubbed["law"] = (
        (lambda value: 0.20 + 0.02 * value)
        if increasing
        else (lambda value: 0.30 - 0.02 * value)
    )
    result = find_critical_value(_base(), target_knockdown=0.95)
    assert result.status == "target_unreachable"
    assert result.critical_value is None
    assert "meets the criterion" in result.message


def test_constant_law_is_flat(stubbed):
    """T-7: a parameter with no effect is refused, not root-found."""
    stubbed["law"] = lambda value: 0.7
    result = find_critical_value(_base(), target_knockdown=0.5)
    assert result.status == "flat"
    assert result.critical_value is None


def test_bit_exact_plateau_at_target_is_flat(stubbed):
    """T-8: a plateau sitting on the target has no unique root."""

    def plateau(value: float) -> float:
        if value < 1.0:
            return 0.5
        return 0.5 - 0.3 * (value - 1.0)

    stubbed["law"] = plateau
    result = find_critical_value(_base(), target_knockdown=0.5)
    assert result.status == "flat"
    assert "0.5" in result.message


def test_plateau_away_from_target_still_converges(stubbed):
    """T-9: a bit-exact tie is not a direction reversal."""

    def plateau(value: float) -> float:
        if value < 0.5:
            return 0.9
        return max(0.9 - 1.5 * (value - 0.5), 0.05)

    stubbed["law"] = plateau
    result = find_critical_value(_base(), target_knockdown=0.6)
    assert result.status == "converged"
    assert result.achieved_objective >= 0.6


def _u_shaped(value: float) -> float:
    """Minimum at 0.05 — inside the first 2 % of a [0, 4.392] range.

    The curve dips to 0.1 and has fully recovered by 0.4, so a linear
    grid over the same range never sees the dip at all.
    """
    if value <= 0.05:
        return 1.0 - 18.0 * value
    return min(0.1 + 2.4 * (value - 0.05), 0.95)


def test_u_shape_is_non_monotonic(stubbed):
    """T-10: an interior minimum is detected and refused."""
    stubbed["law"] = _u_shaped
    result = find_critical_value(_base(), target_knockdown=0.5)
    assert result.status == "non_monotonic"
    assert result.sign_changes >= 2
    assert result.critical_value is None
    assert "not monotonic" in result.message


def test_linear_spacing_would_miss_the_u_shape(stubbed, monkeypatch):
    """T-11: pins the reason the scan is log-spaced.

    With a linear grid over ``[0, 4.392]`` the first interior probe sits
    at 0.549, well past the minimum at 0.05 and past the recovery, so
    the scan sees a monotone curve and the refusal above never fires.
    """
    stubbed["law"] = _u_shaped

    def linear_grid(lo, cap, n):
        import numpy as np

        return [float(v) for v in np.linspace(lo, cap, n)], False

    monkeypatch.setattr(gs, "_scan_grid", linear_grid)
    result = find_critical_value(_base(), target_knockdown=0.5)
    assert result.status != "non_monotonic"
    assert result.sign_changes <= 1


def test_explicit_bracket_is_respected(stubbed):
    """T-12: no evaluation escapes an explicit bracket."""
    result = find_critical_value(
        _base(), target_knockdown=0.5, bracket=(0.05, 0.5)
    )
    assert result.bounds_source == ("user_bracket", "user_bracket")
    assert result.bounds == (0.05, 0.5)
    for ev in result.evaluations:
        assert 0.05 <= ev.value <= 0.5


def test_evaluation_ledger(stubbed):
    """T-13: the ledger is complete, ordered and phase-tagged."""
    result = find_critical_value(
        _base(), target_knockdown=0.5, scan_points=7
    )
    assert result.n_evaluations == len(result.evaluations)
    assert [ev.index for ev in result.evaluations] == list(
        range(result.n_evaluations)
    )
    assert {ev.phase for ev in result.evaluations} <= {
        "scan", "root", "verify", "backoff"
    }
    assert any(ev.phase == "verify" for ev in result.evaluations)
    assert len(result.scan_values) == 7
    assert len(result.scan_objectives) == 7


# --------------------------------------------------------------------------- #
# T-14..T-18: validation
# --------------------------------------------------------------------------- #

_INTEGER_FIELDS = [
    "interface_1",
    "interface_2",
    "nx",
    "ny",
    "nz_per_ply",
    "iterative_maxiter",
    "surface_transition_plies",
    "progressive_n_increments",
    "czm_n_load_increments",
]


@pytest.mark.parametrize("parameter", _INTEGER_FIELDS)
def test_integer_fields_are_refused_by_name(parameter):
    """T-14a: every integer field gets the integer-specific message."""
    with pytest.raises(ValueError, match="integer AnalysisConfig field"):
        find_critical_value(
            _base(), parameter=parameter, target_knockdown=0.9
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(parameter="amplitud", target_knockdown=0.9),
         "no continuously-searchable field"),
        (dict(parameter="morphology", target_knockdown=0.9),
         "is a str, not a float"),
        (dict(parameter="analytical_only", target_knockdown=0.9),
         "boolean AnalysisConfig field"),
        (dict(parameter="phase", target_knockdown=0.9), "is None"),
        (dict(), "exactly one of"),
        (dict(target_knockdown=0.9, target_strength_MPa=1000.0),
         "exactly one of"),
        (dict(target_knockdown=1.0), "degenerate"),
        (dict(target_knockdown=1.5), r"\(0.0, 1.0\)"),
        (dict(target_knockdown=0.0), r"\(0.0, 1.0\)"),
        (dict(target_strength_MPa=-1.0), "must be a finite value > 0"),
        (dict(target_knockdown=0.9, rtol=0.0), "rtol must be"),
        (dict(target_knockdown=0.9, rtol=1e-16), "rtol must be"),
        (dict(target_knockdown=0.9, rtol=0.5), "rtol must be"),
        (dict(target_knockdown=0.9, scan_points=2), "scan_points must be"),
        (dict(target_knockdown=0.9, objective="bogus"), "unknown objective"),
        (dict(target_knockdown=0.9, objective="modulus_retention"),
         "over-predicts"),
        (dict(target_knockdown=0.9, objective="retention_factors"),
         "hard-clamped"),
        (dict(target_knockdown=0.9, objective="progressive_knockdown"),
         "load-stepped"),
        (dict(target_knockdown=0.9, bracket=(0.5, 0.1)), "lo < hi"),
        (dict(target_knockdown=0.9, bracket=(-1.0, 0.1)), "must be >= 0"),
        (dict(target_strength_MPa=100.0, objective="modulus_knockdown"),
         "cannot be combined"),
    ],
)
def test_validation_table(kwargs, match):
    """T-14b: every input error raises ValueError before any compute."""
    with pytest.raises(ValueError, match=match):
        find_critical_value(_base(), **kwargs)


def test_zero_valued_base_parameter_needs_a_bracket():
    """T-15: a scaled range around zero is degenerate."""
    with pytest.raises(ValueError, match="bracket"):
        find_critical_value(
            _base(), parameter="decay_floor", target_knockdown=0.9
        )


def test_negative_base_parameter_stays_negative(stubbed):
    """T-16: applied_strain defaults to -0.01; the range must not flip
    to the tension half-line."""
    _LAW["parameter"] = "applied_strain"
    _LAW["law"] = lambda value: 0.5
    result = find_critical_value(
        _base(), parameter="applied_strain", target_knockdown=0.9
    )
    lo, cap = result.bounds
    assert lo < 0.0 and cap < 0.0 and lo < cap
    assert all(ev.value < 0.0 for ev in result.evaluations)


def test_fe_feature_ladder_refuses_analytical_only():
    """T-17a: an FE-only feature plus analytical_only=True is an error."""
    with pytest.raises(ValueError, match="enable_czm"):
        find_critical_value(
            _base(enable_czm=True), target_knockdown=0.9,
            analytical_only=True,
        )


def test_fe_feature_ladder_resolves_none_to_fe(caplog):
    """T-17b: analytical_only=None resolves to the FE path, loudly."""
    cfg = _base(enable_czm=True)
    with caplog.at_level("WARNING"):
        assert gs._resolve_analytical_only(cfg, None) is False
    assert "enable_czm" in caplog.text


def test_transverse_mode_refusal_names_both_remedies():
    """T-17c: the through-width mode is FE-only too."""
    cfg = _base(transverse_mode="gaussian_decay", transverse_width=6.0)
    with pytest.raises(ValueError) as excinfo:
        find_critical_value(cfg, target_knockdown=0.9, analytical_only=True)
    message = str(excinfo.value)
    assert "transverse_mode" in message
    assert "analytical_only=False" in message


@pytest.mark.parametrize("analytical_only", [None, True, False])
def test_progressive_damage_is_refused_outright(analytical_only):
    """T-17d: ~30 load-stepped solves per evaluation is hours."""
    cfg = _base(enable_progressive_damage=True)
    with pytest.raises(ValueError, match="parametric_sweep"):
        find_critical_value(
            cfg, target_knockdown=0.9, analytical_only=analytical_only
        )


def test_tool_flat_cannot_be_searched_analytically():
    """T-17e: tool_flat is an FE morphology (it auto-enables surface
    resin pockets, which AnalysisConfig rejects analytically)."""
    cfg = _base(amplitude=0.05, morphology="tool_flat")
    with pytest.raises(ValueError, match="analytical_only"):
        find_critical_value(cfg, target_knockdown=0.9, analytical_only=True)


def test_wrinkles_list_makes_amplitude_inert():
    """T-18: cfg.amplitude is ignored when 'wrinkles' is set."""
    cfg = _base(
        wrinkles=[
            WrinkleSpec(
                amplitude=0.3, wavelength=16.0, width=12.0, ply_interface=11
            )
        ]
    )
    with pytest.raises(ValueError, match="is ignored when 'wrinkles' is set"):
        find_critical_value(cfg, target_knockdown=0.9)


def test_summary_contains_the_search_evidence(stubbed):
    """T-19: summary() is the text a user quotes in a disposition."""
    result = find_critical_value(_base(), target_knockdown=0.5)
    text = result.summary()
    assert "amplitude" in text
    assert "Criterion satisfied:" in text
    assert "0.500000" in text
    assert result.status == "converged"
    for value in result.scan_values:
        assert f"{value:.6g}" in text


def test_replace_swept_parity_with_parametric_sweep():
    """T-20: the shared clone helper matches the sweep it was lifted from."""
    from wrinklefe.analysis import _replace_swept

    base = _base()
    assert base.domain_length == pytest.approx(48.0)
    cloned = _replace_swept(base, "wavelength", 64.0)
    assert cloned.domain_length == pytest.approx(192.0)

    pinned = _base(domain_length=99.0)
    kept = _replace_swept(pinned, "wavelength", 64.0)
    assert kept.domain_length == pytest.approx(99.0)


def test_resolved_analytical_only_reaches_every_clone(stubbed):
    """T-21: a saved result records the path it actually ran on."""
    result = find_critical_value(
        _base(), target_knockdown=0.5, keep_results=True
    )
    assert result.analytical_only is True
    assert result.results is not None
    assert len(result.results) == result.n_evaluations
    for run in result.results:
        assert run.config.analytical_only is True
    assert result.critical_config.analytical_only is True


# --------------------------------------------------------------------------- #
# T-30..T-44: the real analytical model (~40 ms per forward evaluation)
# --------------------------------------------------------------------------- #


def _readme_default(**overrides) -> AnalysisConfig:
    return _base(**overrides)


def test_readme_default_reaches_the_target():
    """T-30 (AC1): a forward run at the answer reproduces the target."""
    result = find_critical_value(_readme_default(), target_knockdown=0.9)
    assert result.status == "converged"
    forward = WrinkleAnalysis(result.critical_config).run(
        analytical_only=True
    )
    assert forward.analytical_knockdown >= 0.9
    assert forward.analytical_knockdown == pytest.approx(0.9, rel=1e-3)
    assert result.critical_value == pytest.approx(0.040, rel=0.05)


def test_compression_acceptance_limit():
    """T-31 (AC3a)."""
    result = find_critical_value(_readme_default(), target_knockdown=0.85)
    assert result.status == "converged"
    forward = WrinkleAnalysis(result.critical_config).run(
        analytical_only=True
    )
    assert forward.analytical_knockdown >= 0.85
    assert result.critical_value == pytest.approx(0.0665, rel=0.05)


def test_tension_acceptance_limit():
    """T-32 (AC3b): the same question on the tension path."""
    result = find_critical_value(
        _readme_default(loading="tension"), target_knockdown=0.9
    )
    assert result.status == "converged"
    forward = WrinkleAnalysis(result.critical_config).run(
        analytical_only=True
    )
    assert forward.analytical_knockdown >= 0.9


def test_no_crossing_is_clean_not_a_traceback():
    """T-33 (AC2): a shallow target refuses cleanly."""
    result = find_critical_value(_readme_default(), target_knockdown=0.30)
    assert result.status == "no_crossing"
    assert result.critical_value is None
    assert result.bounds_source[1] == "laminate_thickness"
    assert min(result.scan_objectives) == pytest.approx(0.41, rel=0.05)
    assert "not necessarily an error" in result.message


@pytest.mark.parametrize("target", [0.9, 0.85, 0.6, 0.5])
def test_safe_side_on_the_real_model(target):
    """T-34: regression for the measured 3-of-4 raw-root violation."""
    result = find_critical_value(
        _readme_default(), target_knockdown=target
    )
    assert result.status == "converged"
    forward = WrinkleAnalysis(result.critical_config).run(
        analytical_only=True
    )
    assert forward.analytical_knockdown >= target


def test_strength_target_matches_the_knockdown_target():
    """T-35: an MPa target is the same question in strength units."""
    by_mpa = find_critical_value(
        _readme_default(), target_strength_MPa=1020.0
    )
    by_kd = find_critical_value(_readme_default(), target_knockdown=0.85)
    assert by_mpa.status == "converged"
    assert by_mpa.target_is_strength is True
    assert by_mpa.objective_name == "strength_MPa"
    assert by_mpa.critical_value == pytest.approx(
        by_kd.critical_value, rel=1e-3
    )
    assert by_mpa.pristine_reference_MPa == pytest.approx(1200.0, rel=1e-6)


def test_modulus_knockdown_objective():
    """T-36: a stiffness-driven acceptance limit."""
    result = find_critical_value(
        _readme_default(), objective="modulus_knockdown",
        target_knockdown=0.8,
    )
    assert result.status == "converged"
    forward = WrinkleAnalysis(result.critical_config).run(
        analytical_only=True
    )
    assert forward.analytical_modulus_knockdown >= 0.8


def test_fe_only_objective_refused_on_the_analytical_path():
    """T-37."""
    with pytest.raises(ValueError, match="modulus_retention_global"):
        find_critical_value(
            _readme_default(), objective="modulus_retention_global",
            target_knockdown=0.9,
        )


def test_onset_objective_refused_under_compression():
    """T-38."""
    with pytest.raises(ValueError, match="tension"):
        find_critical_value(
            _readme_default(), objective="onset_knockdown",
            target_knockdown=0.9,
        )


def test_wavelength_goal_seek_resets_the_domain_sentinel():
    """T-39: the auto-derived domain follows the swept wavelength."""
    result = find_critical_value(
        _readme_default(), parameter="wavelength", target_knockdown=0.8
    )
    assert result.status == "converged"
    assert result.direction == "increasing"
    assert result.critical_config.domain_length == pytest.approx(
        3.0 * result.critical_config.wavelength
    )


def test_graded_wavelength_u_shape_is_refused():
    """T-40: the measured counterexample to #280's monotonicity premise.

    Configuration: amplitude=0.5, angles=[0]*24, morphology='graded'.
    The through-thickness Gaussian widens with wavelength, so the
    knockdown curve has an interior minimum near lambda ~ 3 mm.
    Reporting a root here would hand back a safe-sounding limit for a
    catastrophically failing geometry.
    """
    cfg = _base(
        amplitude=0.5, angles=[0.0] * 24, morphology="graded"
    )
    result = find_critical_value(
        cfg, parameter="wavelength", bracket=(1.0, 128.0),
        target_knockdown=0.40,
    )
    assert result.status == "non_monotonic"
    assert result.sign_changes >= 2
    assert result.critical_value is None
    assert min(result.scan_objectives) < 0.20
    assert "graded" in result.message


def test_saturated_penetration_gate_is_flat():
    """T-41: a near-surface wrinkle under the gate barely responds."""
    cfg = _base(
        morphology="graded", penetration_gate=GATE_LI2025_VACBAG,
        wrinkle_z_position=0.042,
    )
    result = find_critical_value(cfg, target_knockdown=0.9)
    assert result.status == "flat"
    assert result.critical_value is None
    span = max(result.scan_objectives) - min(result.scan_objectives)
    assert span < result.rtol * 0.9


def test_gate_ply_thickness_plateau_still_converges():
    """T-42: a plateau of exact ties away from the target is not a
    reversal, so the well-posed problem is answered, not refused."""
    cfg = _base(angles=[0.0] * 24, penetration_gate=GATE_LI2025_VACBAG)
    result = find_critical_value(
        cfg, parameter="ply_thickness", bracket=(0.001, 0.183),
        target_knockdown=0.90,
    )
    assert result.status == "converged"
    assert result.critical_value == pytest.approx(0.147, rel=0.10)
    forward = WrinkleAnalysis(result.critical_config).run(
        analytical_only=True
    )
    assert forward.analytical_knockdown >= 0.90


def test_tension_knockdown_is_monotone_in_amplitude():
    """T-43: pins the compression cap that keeps the tension analytical
    knockdown monotone over a dense amplitude scan."""
    cfg = _base(angles=[0.0] * 24, loading="tension")
    previous = 2.0
    for i in range(25):
        amplitude = 0.02 * (i + 1)
        run = WrinkleAnalysis(
            AnalysisConfig(
                amplitude=amplitude, wavelength=16.0, width=12.0,
                angles=cfg.angles, loading="tension",
            )
        ).run(analytical_only=True)
        assert run.analytical_knockdown <= previous + 1e-12
        previous = run.analytical_knockdown


def test_search_is_deterministic():
    """T-44: no RNG, no cache, no module state."""
    first = find_critical_value(_readme_default(), target_knockdown=0.85)
    second = find_critical_value(_readme_default(), target_knockdown=0.85)
    assert first.critical_value == second.critical_value
    assert first.scan_objectives == second.scan_objectives


# --------------------------------------------------------------------------- #
# T-50..T-52: the real FE path
# --------------------------------------------------------------------------- #

# Each of these runs ~19 forward evaluations, and one FE evaluation is 4
# full linear solves (~1.1 s on the 8-ply mesh below), so budget ~25 s
# each plus the verifying re-run. They are the only place the FE clause
# of acceptance criterion 3 is non-vacuous: 'modulus_retention_global'
# is the one listed objective whose value differs between the analytical
# and FE paths (analytical_knockdown is computed before the
# analytical_only branch, so it is bit-identical on both).


def _fe_config(**overrides) -> AnalysisConfig:
    defaults = dict(
        amplitude=0.05, wavelength=16.0, width=12.0,
        angles=[0.0, 90.0, 90.0, 0.0, 0.0, 90.0, 90.0, 0.0],
        nx=8, ny=2, nz_per_ply=1, analytical_only=False,
    )
    defaults.update(overrides)
    return AnalysisConfig(**defaults)


@pytest.mark.slow
def test_fe_path_compression_acceptance_limit():
    """T-50 (AC3c): a real FE round-trip on the FE-only objective."""
    result = find_critical_value(
        _fe_config(), objective="modulus_retention_global",
        target_knockdown=0.99, bracket=(0.01, 0.45), scan_points=3,
        rtol=1e-2, analytical_only=False,
    )
    assert result.analytical_only is False
    assert result.status == "converged"
    forward = WrinkleAnalysis(result.critical_config).run(
        analytical_only=False
    )
    assert forward.modulus_retention_global >= 0.99


@pytest.mark.slow
def test_fe_path_tension_acceptance_limit():
    """T-51 (AC3d): the same question with loading='tension'.

    The global stiffness retention is load-direction independent, so the
    answer matches the compression case; what this covers is that the
    whole tension config path survives an FE goal-seek.
    """
    result = find_critical_value(
        _fe_config(loading="tension"),
        objective="modulus_retention_global", target_knockdown=0.99,
        bracket=(0.01, 0.45), scan_points=3, rtol=1e-2,
        analytical_only=False,
    )
    assert result.status == "converged"
    forward = WrinkleAnalysis(result.critical_config).run(
        analytical_only=False
    )
    assert forward.modulus_retention_global >= 0.99


def test_fe_path_clamps_the_amplitude_bound(monkeypatch):
    """T-52: the mesh-inversion limit caps the search on the FE path.

    The bound resolution is pure arithmetic, so this needs no solve.
    """
    cfg = _readme_default(analytical_only=False)
    lo, cap, source = gs._resolve_bounds(
        cfg, "amplitude", None, None, analytical_only=False
    )
    assert source[1] == "mesh_amplitude_safe"
    assert cap == pytest.approx(2.5 * cfg.ply_thickness / cfg.nz_per_ply)
    assert lo == 0.0
