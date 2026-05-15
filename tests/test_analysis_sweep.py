"""Regression tests for AnalysisConfig field preservation in sweeps.

Verifies the fix for issue #13: ``compare_morphologies`` and
``parametric_sweep`` previously rebuilt a fresh :class:`AnalysisConfig`
field-by-field from ``base_config``, silently dropping fields like
``decay_floor``.  Both helpers must now preserve every existing field
and only override the swept parameters.
"""

from __future__ import annotations

import math
from dataclasses import fields, replace
from unittest.mock import patch

import pytest

from wrinklefe.analysis import AnalysisConfig, AnalysisResults, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary


@pytest.fixture
def base_config_with_decay_floor():
    """Tiny config with a non-default decay_floor (issue #13)."""
    return AnalysisConfig(
        amplitude=0.366,
        wavelength=16.0,
        width=12.0,
        morphology="stack",
        decay_floor=0.3,  # non-default
        loading="compression",
        material=MaterialLibrary().get("IM7_8552"),
        angles=[0, 90, 0, 90, 0, 90, 90, 0, 90, 0, 90, 0],
        interface_1=5,
        interface_2=6,
        nx=4,
        ny=2,
        nz_per_ply=1,
        domain_width=10.0,
        applied_strain=-0.005,
        verbose=False,
    )


def _make_stub_result(config):
    """Build a bare AnalysisResults capturing the config it was given."""
    return AnalysisResults(config=config)


class TestDecayFloorPreservedInSweep:
    """Issue #13: swept configs must retain user-set decay_floor."""

    def test_parametric_sweep_preserves_decay_floor(
        self, base_config_with_decay_floor
    ):
        """Sweeping amplitude must not reset decay_floor to 0.0."""
        captured_configs = []

        def fake_run(self, analytical_only=None):
            captured_configs.append(self.config)
            return _make_stub_result(self.config)

        amps = [0.183, 0.366, 0.549]
        with patch.object(WrinkleAnalysis, "run", fake_run):
            results = WrinkleAnalysis.parametric_sweep(
                base_config_with_decay_floor, "amplitude", amps
            )

        assert len(results) == 3
        assert len(captured_configs) == 3
        # All swept configs preserve the user-set decay_floor.
        for cfg, expected_amp in zip(captured_configs, amps):
            assert cfg.decay_floor == pytest.approx(0.3), (
                f"decay_floor was silently dropped during sweep "
                f"(got {cfg.decay_floor}, expected 0.3)"
            )
            assert cfg.amplitude == pytest.approx(expected_amp)

        # The same invariant holds when inspecting the returned results.
        for res in results:
            assert res.config.decay_floor == pytest.approx(0.3)

    def test_compare_morphologies_preserves_decay_floor(
        self, base_config_with_decay_floor
    ):
        """compare_morphologies must not reset decay_floor across morphologies."""
        captured_configs = []

        def fake_run(self, analytical_only=None):
            captured_configs.append(self.config)
            return _make_stub_result(self.config)

        morphologies = ("stack", "convex", "concave")
        with patch.object(WrinkleAnalysis, "run", fake_run):
            results = WrinkleAnalysis.compare_morphologies(
                base_config_with_decay_floor, morphologies=morphologies
            )

        assert set(results.keys()) == set(morphologies)
        assert len(captured_configs) == len(morphologies)
        for cfg in captured_configs:
            assert cfg.decay_floor == pytest.approx(0.3), (
                f"decay_floor was silently dropped during compare "
                f"(got {cfg.decay_floor}, expected 0.3)"
            )
        # Each morphology slot keeps its own decay_floor.
        for morph in morphologies:
            assert results[morph].config.morphology == morph
            assert results[morph].config.decay_floor == pytest.approx(0.3)

    def test_parametric_sweep_preserves_other_nondefault_fields(
        self, base_config_with_decay_floor
    ):
        """Other user-set fields must also survive the sweep."""

        def fake_run(self, analytical_only=None):
            return _make_stub_result(self.config)

        with patch.object(WrinkleAnalysis, "run", fake_run):
            results = WrinkleAnalysis.parametric_sweep(
                base_config_with_decay_floor, "amplitude", [0.2, 0.4]
            )

        for res in results:
            cfg = res.config
            assert cfg.morphology == "stack"
            assert cfg.interface_1 == 5
            assert cfg.interface_2 == 6
            assert cfg.applied_strain == pytest.approx(-0.005)
            assert cfg.nx == 4
            assert cfg.ny == 2
            assert cfg.domain_width == pytest.approx(10.0)
            assert cfg.decay_floor == pytest.approx(0.3)


class TestSweepDomainLengthDerivation:
    """Issue #44: sweeping wavelength must re-run __post_init__ so the
    auto-derived ``domain_length`` (3 * wavelength) tracks the swept value.

    Before the fix, ``parametric_sweep`` patched ``cfg.domain_length``
    directly after ``dataclasses.replace``, bypassing ``__post_init__``
    and leaving derived state stale for any base config where
    ``domain_length`` was originally auto-derived.
    """

    def _base_auto_domain(self):
        # ``domain_length=0.0`` → __post_init__ sets it to 3 * wavelength.
        return AnalysisConfig(
            amplitude=0.366,
            wavelength=10.0,
            width=12.0,
            morphology="stack",
            loading="compression",
            material=MaterialLibrary().get("IM7_8552"),
            angles=[0, 90, 0, 90],
            interface_1=1,
            interface_2=2,
            nx=4,
            ny=2,
            nz_per_ply=1,
            domain_length=0.0,  # auto-derive
            domain_width=10.0,
            applied_strain=-0.005,
            verbose=False,
        )

    def test_wavelength_sweep_redrives_domain_length(self):
        """Each swept wavelength must produce domain_length == 3 * wavelength."""
        base = self._base_auto_domain()
        # Sanity: base auto-derivation worked.
        assert base.domain_length == pytest.approx(30.0)

        captured = []

        def fake_run(self, analytical_only=None):
            captured.append(self.config)
            return _make_stub_result(self.config)

        wavelengths = [5.0, 12.0, 25.0]
        with patch.object(WrinkleAnalysis, "run", fake_run):
            WrinkleAnalysis.parametric_sweep(base, "wavelength", wavelengths)

        assert len(captured) == 3
        for cfg, wl in zip(captured, wavelengths):
            assert cfg.wavelength == pytest.approx(wl)
            assert cfg.domain_length == pytest.approx(3.0 * wl), (
                f"domain_length stale after sweep: got {cfg.domain_length}, "
                f"expected {3.0 * wl} for wavelength {wl}"
            )

    def test_wavelength_sweep_respects_explicit_domain_length(self):
        """If the user pinned domain_length, sweeping wavelength must keep it."""
        base = self._base_auto_domain()
        # Override domain_length explicitly with a value the auto-derivation
        # would never produce (3 * 10.0 == 30.0; we pick 99.0).
        pinned = replace(base, domain_length=99.0)
        assert pinned.domain_length == pytest.approx(99.0)
        # Quick sanity: ``fields(pinned)`` is non-empty (used to ensure the
        # `fields` import is exercised; also documents the contract).
        assert any(f.name == "domain_length" for f in fields(pinned))

        captured = []

        def fake_run(self, analytical_only=None):
            captured.append(self.config)
            return _make_stub_result(self.config)

        with patch.object(WrinkleAnalysis, "run", fake_run):
            WrinkleAnalysis.parametric_sweep(pinned, "wavelength", [5.0, 12.0])

        for cfg in captured:
            assert cfg.domain_length == pytest.approx(99.0), (
                "Explicit domain_length must not be overwritten by sweep"
            )

    def test_invalid_parameter_rejects_non_field_attributes(self):
        """Sweep must reject method/descriptor names, not just unknown attrs."""
        base = self._base_auto_domain()
        # ``__post_init__`` exists as an attribute on the dataclass instance
        # but is not a field; the stricter check must reject it.
        with pytest.raises(AttributeError, match="has no field"):
            WrinkleAnalysis.parametric_sweep(base, "__post_init__", [1.0])


class TestPhaseSweep:
    """Issue #49: sweeping ``phase`` must change the dual-wrinkle geometry.

    Previously ``AnalysisConfig`` had no ``phase`` field, so a phase
    sweep silently produced N identical configurations (every point used
    the same named-morphology phase).  ``phase`` is now a real numeric
    field that overrides the named-morphology phase and is threaded into
    the wrinkle profile via ``WrinkleConfiguration.dual_wrinkle``.
    """

    def _phase_base(self):
        return AnalysisConfig(
            amplitude=0.366,
            wavelength=16.0,
            width=12.0,
            morphology="stack",
            loading="compression",
            material=MaterialLibrary().get("IM7_8552"),
            angles=[0, 90, 0, 90, 0, 90, 90, 0, 90, 0, 90, 0],
            interface_1=5,
            interface_2=6,
            nx=4,
            ny=2,
            nz_per_ply=1,
            domain_width=10.0,
            applied_strain=-0.005,
            analytical_only=True,
            verbose=False,
        )

    def test_phase_is_a_sweepable_field(self):
        """``phase`` must be a real AnalysisConfig field (no AttributeError)."""
        assert any(f.name == "phase" for f in fields(AnalysisConfig))
        # Default leaves phase unset so named morphology still governs.
        assert AnalysisConfig().phase is None

    def test_parametric_sweep_phase_produces_distinct_configs(self):
        """Each swept phase must reach AnalysisConfig.phase as a distinct value."""
        captured = []

        def fake_run(self, analytical_only=None):
            captured.append(self.config)
            return _make_stub_result(self.config)

        phases = [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]
        with patch.object(WrinkleAnalysis, "run", fake_run):
            results = WrinkleAnalysis.parametric_sweep(
                self._phase_base(), "phase", phases
            )

        assert len(results) == len(phases)
        for cfg, expected in zip(captured, phases):
            assert cfg.phase == pytest.approx(expected)
        # All swept phase values are distinct (no silent collapse).
        assert len({round(c.phase, 9) for c in captured}) == len(phases)

    def test_parametric_sweep_phase_changes_knockdown(self):
        """Distinct phases must yield distinct morphology factors / knockdowns."""
        base = self._phase_base()
        # Phases chosen so sin(phi) is strictly increasing (the morphology
        # factor is exp(-alpha*sin(phi) - ...)); avoid points that share a
        # sin value, e.g. phi=0/pi or phi=pi/4 vs 3pi/4, which are
        # *physically* identical (this is correct morphology physics, not
        # a no-op).  Within [0, pi/2] sin is monotonic.
        phases = [0.0, math.pi / 6, math.pi / 3, math.pi / 2]
        results = WrinkleAnalysis.parametric_sweep(
            base, "phase", phases, analytical_only=True
        )

        mfs = [r.morphology_factor for r in results]
        kds = [r.analytical_knockdown for r in results]
        # N distinct morphology factors and knockdowns — not a no-op.
        assert len({round(m, 8) for m in mfs}) == len(phases), (
            f"phase sweep produced non-distinct morphology factors: {mfs}"
        )
        assert len({round(k, 8) for k in kds}) == len(phases), (
            f"phase sweep produced non-distinct knockdowns (silent no-op): {kds}"
        )

    def test_explicit_phase_matches_named_morphology(self):
        """phase=+pi/2 must reproduce the named 'convex' morphology exactly."""
        base = self._phase_base()
        named_convex = WrinkleAnalysis(
            replace(base, morphology="convex")
        ).run(analytical_only=True)
        explicit = WrinkleAnalysis(
            replace(base, morphology="stack", phase=math.pi / 2)
        ).run(analytical_only=True)
        assert explicit.analytical_knockdown == pytest.approx(
            named_convex.analytical_knockdown
        )
        assert explicit.morphology_factor == pytest.approx(
            named_convex.morphology_factor
        )

    def test_phase_none_preserves_named_morphology(self):
        """phase=None (default) must keep using the named-morphology phase."""
        base = self._phase_base()
        concave = WrinkleAnalysis(
            replace(base, morphology="concave", phase=None)
        ).run(analytical_only=True)
        stack = WrinkleAnalysis(
            replace(base, morphology="stack", phase=None)
        ).run(analytical_only=True)
        # Named morphologies still differentiate when phase is unset.
        assert concave.morphology_factor != pytest.approx(
            stack.morphology_factor
        )
        # concave amplifies (M_f > 1), stack is baseline (M_f == 1).
        assert concave.morphology_factor > 1.0
        assert stack.morphology_factor == pytest.approx(1.0)
