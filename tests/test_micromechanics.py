"""Tests for wrinklefe.core.micromechanics.

The headline test is :class:`TestPresetReproduction`: it rebuilds the
shipped library presets from their constituents at the fibre volume
fraction each preset documents in its own source comment, and pins how far
the prediction lands from the shipped card.  Those deviations are the
honest accuracy statement for this module — the tolerances below were read
off the measurement, not chosen first and fitted to.
"""

import numpy as np
import pytest

from wrinklefe.core.material import MaterialLibrary, OrthotropicMaterial
from wrinklefe.core.micromechanics import (
    FIBER_PRESETS,
    MATRIX_PRESETS,
    FiberProperties,
    MatrixProperties,
    alpha1_schapery,
    alpha2_schapery,
    e1_rule_of_mixtures,
    e2_halpin_tsai,
    g12_halpin_tsai,
    g23_transverse_isotropy,
    halpin_tsai,
    nu12_rule_of_mixtures,
    nu23_rule_of_mixtures,
)


@pytest.fixture(scope="module")
def library():
    return MaterialLibrary()


@pytest.fixture(scope="module")
def generic_epoxy(library):
    """The library's neat-epoxy card as a matrix constituent (3.5 GPa)."""
    return MatrixProperties.from_material(library.get("EPOXY_S6C10"))


# ======================================================================
# Preset reproduction — the headline validation
# ======================================================================

# One entry per library preset whose own source comment documents the
# fibre volume fraction it was built at (material.py, the _load_builtins
# block).  Presets that do not state a Vf are excluded: without the Vf the
# comparison would be a fit, not a prediction.
#
#   system            fibre       matrix           Vf     source of the Vf
PRESET_CASES = [
    # "computed via micromechanics at Vf ~ 0.60"
    ("AC318_S6C10", "S2_GLASS", None, 0.60),
    # "Vf ~ 0.59".  HexPly M21 publishes no neat-resin modulus, so the
    # library's generic 3.5 GPa epoxy card stands in for the matrix; the
    # matrix-dominated predictions (E2, G12) are indicative only here.
    ("T800S_M21", "T800S", None, 0.59),
    # "Table 1, 60% Vf cured ply properties"
    ("IM10_8552", "IM10", "EPOXY_8552", 0.60),
    # "Hsiao & Daniel (1996) ... Table 1 (Vf 0.66)"
    ("IM6G_3501_6", "IM6G", "EPOXY_3501_6", 0.66),
    # "Representative cured-ply values at Vf ~ 0.60"
    ("S2_GLASS_EPOXY", "S2_GLASS", None, 0.60),
    # "Representative cured-ply values at Vf ~ 0.60"
    ("KEVLAR49_EPOXY", "KEVLAR49", None, 0.60),
]

# Measured deviation of the prediction from the shipped card, in per cent
# (positive = the model is stiffer than the preset).  These are recorded
# measurements, re-derived by running the module; they are pinned so that a
# change to a mixing rule or a constituent constant has to be looked at.
MEASURED_DEVIATION_PCT = {
    ("AC318_S6C10", "E1"): -8.6,
    ("AC318_S6C10", "E2"): 28.7,
    ("AC318_S6C10", "G12"): -17.1,
    ("AC318_S6C10", "nu12"): -0.7,
    ("T800S_M21", "E1"): 11.4,
    ("T800S_M21", "E2"): 13.3,
    ("T800S_M21", "G12"): -9.8,
    ("T800S_M21", "nu12"): -25.3,
    ("IM10_8552", "E1"): 2.5,
    ("IM10_8552", "E2"): 20.8,
    ("IM10_8552", "G12"): -13.2,
    ("IM10_8552", "nu12"): -18.8,
    ("IM6G_3501_6", "E1"): 9.8,
    ("IM6G_3501_6", "E2"): 32.8,
    ("IM6G_3501_6", "G12"): -21.3,
    ("IM6G_3501_6", "nu12"): -19.0,
    ("S2_GLASS_EPOXY", "E1"): 1.9,
    ("S2_GLASS_EPOXY", "E2"): -18.7,
    ("S2_GLASS_EPOXY", "G12"): -32.0,
    ("S2_GLASS_EPOXY", "nu12"): -7.3,
    ("KEVLAR49_EPOXY", "E1"): 5.3,
    ("KEVLAR49_EPOXY", "E2"): -2.7,
    # Halpin-Tsai with the handbook aramid G12f = 21 GPa over-predicts the
    # ply shear modulus by a factor of ~1.8.  See the xfail below.
    ("KEVLAR49_EPOXY", "G12"): 83.6,
    ("KEVLAR49_EPOXY", "nu12"): -0.6,
}

# Tolerances read off the measurements above, with margin.  They are wide
# on purpose: these are semi-empirical mixing rules compared against cards
# assembled from test data and several different literature sources, and
# quoting a tight tolerance would misrepresent the model.
TOLERANCE_PCT = {"E1": 15.0, "E2": 40.0, "G12": 40.0, "nu12": 30.0}

# (system, property) pairs the model does not reproduce, kept visible as
# xfail rather than excluded or tuned away.
KNOWN_FAILURES = {
    ("KEVLAR49_EPOXY", "G12"): (
        "Halpin-Tsai over-predicts aramid ply shear by 84 %: Daniel & Ishai "
        "Table A.2 quotes G12f = 21 GPa for Kevlar 49, but the measured "
        "axial shear modulus of aramid fibre is contested (published values "
        "span ~2-21 GPa) because the fibrils shear so readily.  Bending the "
        "fibre constant to close the gap would make the module useless at a "
        "Vf other than this one, so the discrepancy stays visible."
    ),
}


def _build(case, generic_epoxy):
    system, fiber_key, matrix_key, Vf = case
    matrix = generic_epoxy if matrix_key is None else MATRIX_PRESETS[matrix_key]
    return OrthotropicMaterial.from_constituents(
        FIBER_PRESETS[fiber_key], matrix, Vf, name=f"{system}_predicted"
    )


def _deviation_pct(predicted: float, reference: float) -> float:
    return 100.0 * (predicted - reference) / reference


def _preset_params():
    params = []
    for case in PRESET_CASES:
        for prop in ("E1", "E2", "G12", "nu12"):
            marks = []
            reason = KNOWN_FAILURES.get((case[0], prop))
            if reason is not None:
                marks.append(pytest.mark.xfail(strict=False, reason=reason))
            params.append(
                pytest.param(case, prop, marks=marks, id=f"{case[0]}-{prop}")
            )
    return params


class TestPresetReproduction:
    """Rebuild the shipped presets from constituents at their documented Vf."""

    @pytest.mark.parametrize(("case", "prop"), _preset_params())
    def test_within_tolerance(self, case, prop, library, generic_epoxy):
        predicted = getattr(_build(case, generic_epoxy), prop)
        reference = getattr(library.get(case[0]), prop)
        deviation = _deviation_pct(predicted, reference)
        assert abs(deviation) <= TOLERANCE_PCT[prop], (
            f"{case[0]} {prop}: predicted {predicted:.4g} vs preset "
            f"{reference:.4g} ({deviation:+.1f} %)"
        )

    @pytest.mark.parametrize(
        ("case", "prop"),
        [
            pytest.param(case, prop, id=f"{case[0]}-{prop}")
            for case in PRESET_CASES
            for prop in ("E1", "E2", "G12", "nu12")
        ],
    )
    def test_measured_deviation_is_pinned(self, case, prop, library, generic_epoxy):
        """The recorded deviations are the module's accuracy claim.

        A change to a mixing rule or a constituent constant moves these, and
        that has to be a deliberate, re-measured decision — including for
        the pairs marked xfail above.
        """
        predicted = getattr(_build(case, generic_epoxy), prop)
        reference = getattr(library.get(case[0]), prop)
        deviation = _deviation_pct(predicted, reference)
        expected = MEASURED_DEVIATION_PCT[(case[0], prop)]
        assert deviation == pytest.approx(expected, abs=0.1)

    @pytest.mark.parametrize("case", PRESET_CASES, ids=lambda c: c[0])
    def test_predicted_card_is_valid(self, case, generic_epoxy):
        """Every predicted card survives OrthotropicMaterial validation."""
        ply = _build(case, generic_epoxy)
        ply.validate()
        assert np.all(np.linalg.eigvalsh(ply.compliance_matrix) > 0)


# ======================================================================
# Physical limits and monotonicity
# ======================================================================

class TestLimits:
    def test_Vf_zero_returns_matrix_properties(self, generic_epoxy):
        matrix = generic_epoxy
        ply = OrthotropicMaterial.from_constituents(
            FIBER_PRESETS["IM7"], matrix, 0.0
        )
        assert ply.E1 == pytest.approx(matrix.Em)
        assert ply.E2 == pytest.approx(matrix.Em)
        assert ply.E3 == pytest.approx(matrix.Em)
        assert ply.G12 == pytest.approx(matrix.Gm)
        assert ply.G23 == pytest.approx(matrix.Gm)
        assert ply.nu12 == pytest.approx(matrix.num)
        assert ply.nu23 == pytest.approx(matrix.num)
        assert ply.alpha1 == pytest.approx(matrix.alpham)
        assert ply.alpha2 == pytest.approx(matrix.alpham)

    def test_Vf_one_returns_fiber_properties(self, generic_epoxy):
        fiber = FIBER_PRESETS["IM7"]
        ply = OrthotropicMaterial.from_constituents(fiber, generic_epoxy, 1.0)
        assert ply.E1 == pytest.approx(fiber.E1f)
        assert ply.E2 == pytest.approx(fiber.E2f)
        assert ply.G12 == pytest.approx(fiber.G12f)
        assert ply.nu12 == pytest.approx(fiber.nu12f)
        assert ply.nu23 == pytest.approx(fiber.nu23f_effective)
        assert ply.alpha1 == pytest.approx(fiber.alpha1f)

    def test_halpin_tsai_is_exact_at_both_limits(self):
        assert halpin_tsai(0.0, 200.0, 3.5, 2.0) == pytest.approx(3.5)
        assert halpin_tsai(1.0, 200.0, 3.5, 2.0) == pytest.approx(200.0)

    @pytest.mark.parametrize("fiber_key", sorted(FIBER_PRESETS))
    def test_E1_increases_monotonically_with_Vf(self, fiber_key, generic_epoxy):
        fiber = FIBER_PRESETS[fiber_key]
        Vfs = np.linspace(0.0, 1.0, 21)
        E1 = [e1_rule_of_mixtures(float(v), fiber, generic_epoxy) for v in Vfs]
        assert np.all(np.diff(E1) > 0.0)

    @pytest.mark.parametrize("fiber_key", sorted(FIBER_PRESETS))
    def test_E2_and_G12_increase_monotonically_with_Vf(
        self, fiber_key, generic_epoxy
    ):
        fiber = FIBER_PRESETS[fiber_key]
        Vfs = np.linspace(0.0, 1.0, 21)
        E2 = [e2_halpin_tsai(float(v), fiber, generic_epoxy) for v in Vfs]
        G12 = [g12_halpin_tsai(float(v), fiber, generic_epoxy) for v in Vfs]
        assert np.all(np.diff(E2) > 0.0)
        assert np.all(np.diff(G12) > 0.0)

    def test_stiffness_ratio_between_two_Vf_is_the_intended_use(self):
        """Compaction raises Vf, which must raise E1 and E2."""
        fiber = FIBER_PRESETS["IM10"]
        matrix = MATRIX_PRESETS["EPOXY_8552"]
        nominal = OrthotropicMaterial.from_constituents(fiber, matrix, 0.60)
        compacted = OrthotropicMaterial.from_constituents(fiber, matrix, 0.68)
        assert compacted.E1 > nominal.E1
        assert compacted.E2 > nominal.E2
        assert compacted.G12 > nominal.G12


class TestPositiveDefiniteness:
    """Bad mixing constants can silently yield a non-physical ply."""

    @pytest.mark.parametrize("fiber_key", sorted(FIBER_PRESETS))
    @pytest.mark.parametrize(
        "matrix_key", [*sorted(MATRIX_PRESETS), "EPOXY_S6C10"]
    )
    def test_every_preset_pair_is_positive_definite_over_the_Vf_range(
        self, fiber_key, matrix_key, generic_epoxy
    ):
        fiber = FIBER_PRESETS[fiber_key]
        matrix = (
            generic_epoxy if matrix_key == "EPOXY_S6C10"
            else MATRIX_PRESETS[matrix_key]
        )
        for Vf in np.linspace(0.30, 0.75, 19):
            ply = OrthotropicMaterial.from_constituents(fiber, matrix, float(Vf))
            eigvals = np.linalg.eigvalsh(ply.compliance_matrix)
            assert np.all(eigvals > 0), f"non-PD at Vf={Vf:.3f}"
            assert -1.0 < ply.nu23 < 0.5


# ======================================================================
# Strengths are not predicted
# ======================================================================

class TestStrengthsAreNotPredicted:
    def test_strengths_do_not_track_Vf(self, generic_epoxy):
        """Executable statement of the documented limitation."""
        fiber = FIBER_PRESETS["IM10"]
        low = OrthotropicMaterial.from_constituents(fiber, generic_epoxy, 0.40)
        high = OrthotropicMaterial.from_constituents(fiber, generic_epoxy, 0.70)
        for attr in ("Xt", "Xc", "Yt", "Yc", "S12"):
            assert getattr(low, attr) == getattr(high, attr)

    def test_strengths_from_carries_the_reference_allowables(
        self, library, generic_epoxy
    ):
        reference = library.get("IM10_8552")
        ply = OrthotropicMaterial.from_constituents(
            FIBER_PRESETS["IM10"], MATRIX_PRESETS["EPOXY_8552"], 0.55,
            strengths_from=reference,
        )
        for attr in ("Xt", "Xc", "Yt", "Yc", "Zt", "Zc", "S12", "S13", "S23",
                     "beta1", "beta2", "beta3", "gamma_Y", "GIc", "GIIc",
                     "beta_shear", "alpha_0", "sigma_max", "tau_max"):
            assert getattr(ply, attr) == getattr(reference, attr)
        # ... but the elastic constants are the predicted ones, not copied.
        assert ply.E1 != reference.E1

    def test_default_name_records_the_constituents_and_Vf(self, generic_epoxy):
        ply = OrthotropicMaterial.from_constituents(
            FIBER_PRESETS["T300"], generic_epoxy, 0.6
        )
        assert ply.name == "T300_EPOXY_S6C10_Vf0.60"


# ======================================================================
# Validation
# ======================================================================

class TestValidation:
    @pytest.mark.parametrize("Vf", [-0.01, 1.01, 2.0, -1.0])
    def test_Vf_outside_unit_interval_raises(self, Vf, generic_epoxy):
        with pytest.raises(ValueError, match=r"Vf must be in \[0, 1\]"):
            OrthotropicMaterial.from_constituents(
                FIBER_PRESETS["IM7"], generic_epoxy, Vf
            )

    @pytest.mark.parametrize("attr", ["E1f", "E2f", "G12f"])
    def test_non_positive_fiber_modulus_raises(self, attr):
        kwargs = dict(
            E1f=230_000.0, E2f=15_000.0, G12f=27_000.0, nu12f=0.2,
            alpha1f=-0.5e-6, alpha2f=15e-6,
        )
        kwargs[attr] = -1.0
        with pytest.raises(ValueError, match=f"{attr} must be positive"):
            FiberProperties(**kwargs)

    def test_non_positive_fiber_G23f_raises(self):
        with pytest.raises(ValueError, match="G23f must be positive"):
            FiberProperties(
                E1f=230_000.0, E2f=15_000.0, G12f=27_000.0, G23f=0.0,
                nu12f=0.2, alpha1f=-0.5e-6, alpha2f=15e-6,
            )

    def test_inadmissible_fiber_poisson_ratio_raises(self):
        with pytest.raises(ValueError, match="nu12f must satisfy"):
            FiberProperties(
                E1f=15_000.0, E2f=230_000.0, G12f=27_000.0, nu12f=0.9,
                alpha1f=-0.5e-6, alpha2f=15e-6,
            )

    def test_non_positive_matrix_modulus_raises(self):
        with pytest.raises(ValueError, match="Em must be positive"):
            MatrixProperties(Em=0.0, num=0.35, alpham=45e-6)

    @pytest.mark.parametrize("num", [0.5, 0.7, -1.0, -2.0])
    def test_inadmissible_matrix_poisson_ratio_raises(self, num):
        with pytest.raises(ValueError, match=r"num must be in \(-1, 0.5\)"):
            MatrixProperties(Em=3500.0, num=num, alpham=45e-6)

    def test_negative_halpin_tsai_xi_raises(self):
        with pytest.raises(ValueError, match="xi must be non-negative"):
            halpin_tsai(0.5, 200.0, 3.5, -1.0)

    def test_non_positive_halpin_tsai_property_raises(self):
        with pytest.raises(ValueError, match="positive properties"):
            halpin_tsai(0.5, -200.0, 3.5, 2.0)

    def test_g23_from_negative_E2_raises(self):
        with pytest.raises(ValueError, match="E2 must be positive"):
            g23_transverse_isotropy(-1.0, 0.3)

    def test_g23_from_inadmissible_nu23_raises(self):
        with pytest.raises(ValueError, match="nu23 must be"):
            g23_transverse_isotropy(9000.0, -1.5)

    @pytest.mark.parametrize(
        "func",
        [e1_rule_of_mixtures, nu12_rule_of_mixtures, nu23_rule_of_mixtures,
         alpha1_schapery, alpha2_schapery],
    )
    def test_mixing_rules_reject_out_of_range_Vf(self, func, generic_epoxy):
        with pytest.raises(ValueError, match=r"Vf must be in \[0, 1\]"):
            func(1.5, FIBER_PRESETS["IM7"], generic_epoxy)


# ======================================================================
# Constituent containers
# ======================================================================

class TestConstituentContainers:
    def test_matrix_shear_modulus_is_the_isotropic_relation(self):
        matrix = MatrixProperties(Em=3500.0, num=0.35, alpham=45e-6)
        assert matrix.Gm == pytest.approx(3500.0 / (2.0 * 1.35))

    def test_from_material_reads_an_isotropic_card(self, library):
        card = library.get("EPOXY_S6C10")
        matrix = MatrixProperties.from_material(card)
        assert matrix.Em == card.E1
        assert matrix.num == card.nu12
        assert matrix.alpham == card.alpha1
        assert matrix.name == "EPOXY_S6C10"
        assert matrix.Gm == pytest.approx(card.G12)

    def test_from_material_warns_on_an_orthotropic_card(self, library, caplog):
        with caplog.at_level("WARNING"):
            matrix = MatrixProperties.from_material(library.get("IM7_8552"))
        assert "not isotropic" in caplog.text
        assert matrix.Em == library.get("IM7_8552").E1

    def test_G23f_defaults_to_transverse_isotropy(self):
        fiber = FiberProperties(
            E1f=230_000.0, E2f=15_000.0, G12f=27_000.0, nu12f=0.2,
            alpha1f=-0.5e-6, alpha2f=15e-6,
        )
        assert fiber.G23f_effective == pytest.approx(15_000.0 / 2.4)
        assert fiber.nu23f_effective == pytest.approx(0.2)

    def test_G23f_is_used_when_supplied(self):
        fiber = FiberProperties(
            E1f=230_000.0, E2f=15_000.0, G12f=27_000.0, G23f=7_000.0,
            nu12f=0.2, alpha1f=-0.5e-6, alpha2f=15e-6,
        )
        assert fiber.G23f_effective == 7_000.0
        assert fiber.nu23f_effective == pytest.approx(15_000.0 / 14_000.0 - 1.0)

    @pytest.mark.parametrize("fiber_key", sorted(FIBER_PRESETS))
    def test_every_fiber_preset_is_named_and_physical(self, fiber_key):
        fiber = FIBER_PRESETS[fiber_key]
        assert fiber.name == fiber_key
        assert fiber.E1f >= fiber.E2f  # no fibre is stiffer across than along
        assert 0.0 < fiber.nu12f < 0.5

    @pytest.mark.parametrize("matrix_key", sorted(MATRIX_PRESETS))
    def test_every_matrix_preset_is_named_and_physical(self, matrix_key):
        matrix = MATRIX_PRESETS[matrix_key]
        assert matrix.name == matrix_key
        assert 2_000.0 < matrix.Em < 6_000.0  # cured aerospace epoxy
        assert 0.3 <= matrix.num < 0.5


# ======================================================================
# Thermal expansion
# ======================================================================

class TestThermalExpansion:
    def test_carbon_epoxy_axial_CTE_is_near_zero(self, generic_epoxy):
        """The fibre dominates alpha1, so a CFRP ply barely grows axially."""
        alpha1 = alpha1_schapery(0.60, FIBER_PRESETS["IM7"], generic_epoxy)
        assert abs(alpha1) < 1.0e-6

    def test_transverse_CTE_is_matrix_dominated(self, generic_epoxy):
        alpha2 = alpha2_schapery(0.60, FIBER_PRESETS["IM7"], generic_epoxy)
        assert alpha2 > 5.0 * abs(
            alpha1_schapery(0.60, FIBER_PRESETS["IM7"], generic_epoxy)
        )

    def test_axial_CTE_falls_towards_the_fibre_value_as_Vf_rises(
        self, generic_epoxy
    ):
        fiber = FIBER_PRESETS["IM7"]
        low = alpha1_schapery(0.30, fiber, generic_epoxy)
        high = alpha1_schapery(0.75, fiber, generic_epoxy)
        assert high < low
        assert abs(high - fiber.alpha1f) < abs(low - fiber.alpha1f)

    def test_glass_epoxy_axial_CTE_matches_the_preset(self, library, generic_epoxy):
        """A CTE check the presets can actually adjudicate.

        S2_GLASS_EPOXY documents alpha1 = 6.6e-6 /K at Vf ~ 0.60; Schapery
        predicts 5.4e-6, i.e. within 1.2e-6 /K.
        """
        predicted = alpha1_schapery(0.60, FIBER_PRESETS["S2_GLASS"], generic_epoxy)
        reference = library.get("S2_GLASS_EPOXY").alpha1
        assert abs(predicted - reference) < 1.5e-6
