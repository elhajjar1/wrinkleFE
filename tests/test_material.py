"""Tests for wrinklefe.core.material module."""


import numpy as np
import numpy.testing as npt
import pytest

from wrinklefe.core.material import MaterialLibrary, OrthotropicMaterial


class TestOrthotropicMaterialDefaults:
    """Test OrthotropicMaterial creation with IM7/8552 defaults."""

    def test_default_creation(self):
        mat = OrthotropicMaterial()
        assert mat.name == "IM7_8552"
        assert mat.E1 == 171_420.0
        assert mat.E2 == 9_080.0
        assert mat.E3 == 9_080.0
        assert mat.G12 == 5_290.0
        assert mat.nu12 == 0.32
        assert mat.gamma_Y == 0.02

    def test_custom_creation(self):
        mat = OrthotropicMaterial(E1=140_000.0, E2=10_000.0, name="custom")
        assert mat.E1 == 140_000.0
        assert mat.name == "custom"


class TestComplianceMatrix:
    """Test compliance matrix properties."""

    def test_shape_is_6x6(self, x850_material):
        S = x850_material.compliance_matrix
        assert S.shape == (6, 6)

    def test_symmetric(self, x850_material):
        S = x850_material.compliance_matrix
        npt.assert_allclose(S, S.T, atol=1e-20)

    def test_diagonal_positive(self, x850_material):
        S = x850_material.compliance_matrix
        assert np.all(np.diag(S) > 0)


class TestStiffnessMatrix:
    """Test stiffness matrix properties."""

    def test_shape_is_6x6(self, x850_material):
        C = x850_material.stiffness_matrix
        assert C.shape == (6, 6)

    def test_inverse_of_compliance(self, x850_material):
        """S @ C should be the identity matrix."""
        S = x850_material.compliance_matrix
        C = x850_material.stiffness_matrix
        product = S @ C
        npt.assert_allclose(product, np.eye(6), atol=1e-10)

    def test_positive_definite(self, x850_material):
        """All eigenvalues of stiffness matrix must be positive."""
        C = x850_material.stiffness_matrix
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals > 0), f"Non-positive eigenvalues: {eigvals}"


class TestReducedStiffness:
    """Test reduced stiffness (plane-stress) matrix."""

    def test_shape_is_3x3(self, x850_material):
        Q = x850_material.reduced_stiffness
        assert Q.shape == (3, 3)

    def test_symmetric(self, x850_material):
        Q = x850_material.reduced_stiffness
        npt.assert_allclose(Q, Q.T, atol=1e-20)

    def test_Q11_value(self, x850_material):
        """Q11 = E1 / (1 - nu12 * nu21)."""
        mat = x850_material
        nu21 = mat.nu12 * mat.E2 / mat.E1
        expected_Q11 = mat.E1 / (1.0 - mat.nu12 * nu21)
        npt.assert_allclose(mat.reduced_stiffness[0, 0], expected_Q11, rtol=1e-12)

    def test_Q22_value(self, x850_material):
        """Q22 = E2 / (1 - nu12 * nu21)."""
        mat = x850_material
        nu21 = mat.nu12 * mat.E2 / mat.E1
        expected_Q22 = mat.E2 / (1.0 - mat.nu12 * nu21)
        npt.assert_allclose(mat.reduced_stiffness[1, 1], expected_Q22, rtol=1e-12)

    def test_Q66_is_G12(self, x850_material):
        """Q66 = G12."""
        npt.assert_allclose(
            x850_material.reduced_stiffness[2, 2],
            x850_material.G12,
            rtol=1e-12,
        )


class TestPoissonSymmetry:
    """Test Poisson ratio symmetry relationships."""

    def test_nu12_E1_equals_nu21_E2(self, x850_material):
        """nu12 / E1 should equal nu21 / E2."""
        mat = x850_material
        lhs = mat.nu12 / mat.E1
        rhs = mat.nu21 / mat.E2
        npt.assert_allclose(lhs, rhs, rtol=1e-12)

    def test_nu13_E1_equals_nu31_E3(self, x850_material):
        mat = x850_material
        lhs = mat.nu13 / mat.E1
        rhs = mat.nu31 / mat.E3
        npt.assert_allclose(lhs, rhs, rtol=1e-12)

    def test_nu23_E2_equals_nu32_E3(self, x850_material):
        mat = x850_material
        lhs = mat.nu23 / mat.E2
        rhs = mat.nu32 / mat.E3
        npt.assert_allclose(lhs, rhs, rtol=1e-12)


class TestMaterialValidation:
    """Test validation checks."""

    def test_negative_modulus_raises(self):
        with pytest.raises(ValueError, match="E1 must be positive"):
            OrthotropicMaterial(E1=-1.0)

    def test_zero_gamma_Y_raises(self):
        with pytest.raises(ValueError, match="gamma_Y must be positive"):
            OrthotropicMaterial(gamma_Y=0.0)

    def test_unphysical_poisson_caught_by_pos_definite(self):
        """A Poisson value above the orthotropic stability bound
        (|ν12| ≥ sqrt(E1/E2)) drives the compliance matrix non-physical,
        and validate() should reject it via the positive-definiteness check.

        This is the contract that replaces the old (tautological)
        Poisson-symmetry check — see issue #82.
        """
        with pytest.raises(ValueError, match="not positive-definite"):
            # ν12 = 4.5 with E1=171420, E2=9080 gives ν12^2 * E2/E1 ≈ 1.07 > 1,
            # which violates the orthotropic stability bound
            # (1 - ν12 * ν21) > 0 and produces a non-PD compliance.
            OrthotropicMaterial(nu12=4.5)

    def test_validate_docstring_does_not_claim_symmetry_check(self):
        """Regression guard for issue #82: the docstring used to claim that
        validate() enforced Poisson symmetry, but the check was a tautology.
        After the fix, the docstring must not promise a check that doesn't exist.
        """
        doc = OrthotropicMaterial.validate.__doc__ or ""
        # The string "tolerance of 5" was the giveaway in the old docstring.
        assert "tolerance of 5" not in doc, (
            "validate() docstring still references the removed 5% Poisson "
            "symmetry check — see issue #82."
        )


BUILTIN_NAMES = (
    "AS4_3501_6",
    "IM7_8552",
    "T300_914",
    "T700_2510",
    "AC318_S6C10",
    "T800S_M21",
    "IM10_8552",
    "S2_GLASS_EPOXY",
    "KEVLAR49_EPOXY",
)


class TestMaterialLibrary:
    """Test MaterialLibrary functionality."""

    def test_has_nine_builtins(self):
        lib = MaterialLibrary()
        assert len(lib) == len(BUILTIN_NAMES)

    def test_list_names_sorted(self):
        lib = MaterialLibrary()
        names = lib.list_names()
        assert names == sorted(names)
        for n in BUILTIN_NAMES:
            assert n in names, f"missing built-in {n!r} in {names}"

    def test_get_existing_material(self):
        lib = MaterialLibrary()
        mat = lib.get("IM7_8552")
        assert mat.E1 == 171_420.0
        assert mat.name == "IM7_8552"

    def test_get_missing_raises(self):
        lib = MaterialLibrary()
        with pytest.raises(KeyError, match="not found"):
            lib.get("nonexistent")

    def test_add_material(self):
        lib = MaterialLibrary()
        custom = OrthotropicMaterial(name="custom_mat")
        lib.add(custom)
        assert "custom_mat" in lib
        assert lib.get("custom_mat") is custom
        assert len(lib) == len(BUILTIN_NAMES) + 1

    def test_contains(self):
        lib = MaterialLibrary()
        assert "IM7_8552" in lib
        assert "nonexistent" not in lib


class TestNewBuiltinMaterials:
    """Verify each newly-added built-in is retrievable with sensible values.

    Covers the aerospace-prepreg expansion of issue #88: T800S/M21, IM10/8552,
    S-2 glass/epoxy, and Kevlar-49/epoxy.
    """

    NEW_NAMES = ("T800S_M21", "IM10_8552", "S2_GLASS_EPOXY", "KEVLAR49_EPOXY")

    @pytest.mark.parametrize("name", NEW_NAMES)
    def test_retrievable_by_name(self, name):
        lib = MaterialLibrary()
        mat = lib.get(name)
        assert mat.name == name

    @pytest.mark.parametrize("name", NEW_NAMES)
    def test_positive_moduli(self, name):
        lib = MaterialLibrary()
        mat = lib.get(name)
        for attr in ("E1", "E2", "E3", "G12", "G13", "G23"):
            val = getattr(mat, attr)
            assert val > 0, f"{name}: {attr} = {val} not positive"
        # Fibre-direction stiffness should dominate transverse stiffness
        # for the unidirectional prepregs we ship.
        assert mat.E1 > mat.E2

    @pytest.mark.parametrize("name", NEW_NAMES)
    def test_positive_strengths(self, name):
        lib = MaterialLibrary()
        mat = lib.get(name)
        for attr in ("Xt", "Xc", "Yt", "Yc", "Zt", "Zc",
                     "S12", "S13", "S23"):
            val = getattr(mat, attr)
            assert val > 0, f"{name}: {attr} = {val} not positive"

    @pytest.mark.parametrize("name", NEW_NAMES)
    def test_compliance_positive_definite(self, name):
        lib = MaterialLibrary()
        mat = lib.get(name)
        eigvals = np.linalg.eigvalsh(mat.compliance_matrix)
        assert np.all(eigvals > 0), (
            f"{name}: non-PD compliance, eigenvalues={eigvals}"
        )


class TestMaterialSerialization:
    """Test JSON round-trip serialization."""

    def test_to_dict_from_dict_roundtrip(self):
        original = OrthotropicMaterial()
        d = original.to_dict()
        restored = OrthotropicMaterial.from_dict(d)
        assert restored.E1 == original.E1
        assert restored.E2 == original.E2
        assert restored.nu12 == original.nu12
        assert restored.G12 == original.G12
        assert restored.name == original.name
        assert restored.gamma_Y == original.gamma_Y

    def test_json_roundtrip(self, tmp_path):
        lib = MaterialLibrary()
        json_path = tmp_path / "materials.json"
        lib.to_json(path=json_path)

        # Verify file was written
        assert json_path.exists()

        # Restore from file
        lib2 = MaterialLibrary.from_json(json_path)
        assert lib2.list_names() == lib.list_names()

        # Verify a material survived the round-trip
        orig = lib.get("IM7_8552")
        restored = lib2.get("IM7_8552")
        npt.assert_allclose(orig.E1, restored.E1)
        npt.assert_allclose(orig.nu12, restored.nu12)

    def test_json_roundtrip_from_string(self):
        lib = MaterialLibrary()
        json_str = lib.to_json()

        lib2 = MaterialLibrary.from_json(json_str)
        assert len(lib2) >= 4

    def test_to_dict_contains_all_fields(self):
        mat = OrthotropicMaterial()
        d = mat.to_dict()
        assert "E1" in d
        assert "E2" in d
        assert "nu12" in d
        assert "gamma_Y" in d
        assert "name" in d
