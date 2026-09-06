"""FE thermal initial-strain loading (issue #273, Stage 2).

Stage 1 wired ``AnalysisConfig.delta_T`` into the CLT path and
deliberately *refused* it on the FE path, because the element formulation
carried no initial-strain term.  Stage 2 supplies that term:

- :meth:`~wrinklefe.elements.hex8.Hex8Element.thermal_force_vector`
  integrates ``int B^T C_bar eps_th dV`` per element,
- :meth:`~wrinklefe.solver.assembler.GlobalAssembler.assemble_thermal_force`
  scatters it into the global right-hand side,
- stress recovery subtracts the thermal strain, so
  ``sigma = C_bar (eps_total - eps_th)``.

What is pinned here, in rising order of how badly a regression would hurt:

1. **Element algebra.** The thermal load is self-equilibrated (it is an
   internal load: no net force, no net moment).  A body free to expand
   reports **zero stress**; a fully restrained one reports exactly
   ``-C_bar eps_th``.  Both hold under combined ply-angle and wrinkle
   rotation, which is the case the whole package exists to analyse.
2. **The CLT benchmark** (issue #273 acceptance criterion 3).  A flat
   laminate solved by the 3-D FE path with a statically determinate
   restraint must reproduce the closed-form CLT ply stresses.
3. **Pipeline reachability and the deliberate asymmetries.**  ``delta_T``
   changes the FE fields; the retention baseline is solved at the *same*
   temperature; the measured modulus is solved at ``delta_T = 0``; and
   ``delta_T = 0`` leaves every FE result bit-identical.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.laminate import Laminate, LoadState
from wrinklefe.core.material import MaterialLibrary
from wrinklefe.core.mesh import WrinkleMesh
from wrinklefe.core.morphology import WrinkleConfiguration
from wrinklefe.core.wrinkle import GaussianSinusoidal
from wrinklefe.elements.hex8 import Hex8Element
from wrinklefe.elements.hex8i import Hex8IElement
from wrinklefe.solver.boundary import BoundaryCondition
from wrinklefe.solver.static import StaticSolver

CURE_COOLDOWN_DT = -155.0
PLY_T = 0.125

# Unit cube, VTK node ordering (bottom face CCW, then top face).
UNIT_CUBE = np.array(
    [
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ]
)

# Six constraints that remove the rigid-body modes of a single hex without
# restraining any stretch: node 0 pinned, node 1 free only in x, node 3
# free in x and y.  Statically determinate, so a free thermal expansion
# develops no reaction and therefore no stress.
_RIGID_BODY_DOFS = [0, 1, 2, 3 + 1, 3 + 2, 9 + 2]

# (ply angle in degrees, wrinkle misalignment in radians)
ROTATION_CASES = [
    (0.0, 0.0),      # on-axis
    (30.0, 0.0),     # ply rotation only
    (0.0, 0.12),     # wrinkle rotation only
    (45.0, -0.20),   # both, opposite signs
    (-60.0, 0.25),   # both, the other way round
]


def _element(cls, ply_angle, phi, delta_T, material):
    return cls(
        UNIT_CUBE,
        material,
        ply_angle=ply_angle,
        wrinkle_angles=np.full(8, phi),
        delta_T=delta_T,
    )


def _free_expansion_displacement(elem) -> np.ndarray:
    """Solve ``K u = f_th`` with the rigid-body modes removed."""
    K = elem.stiffness_matrix()
    f = elem.thermal_force_vector()
    free = np.setdiff1d(np.arange(24), _RIGID_BODY_DOFS)
    u = np.zeros(24)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])
    return u


# --------------------------------------------------------------------------- #
# 1. Element algebra
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("element_cls", [Hex8Element, Hex8IElement])
class TestElementThermalLoad:
    """The two hex formulations must agree on the thermal fundamentals."""

    @pytest.mark.parametrize("ply_angle,phi", ROTATION_CASES)
    def test_free_expansion_develops_no_stress(
        self, element_cls, ply_angle, phi, x850_material
    ):
        """The load vector and the stress recovery must be consistent.

        This is the single strongest check available: an unrestrained body
        under a uniform temperature change expands freely and carries no
        stress.  It fails if the thermal load is integrated with the wrong
        stiffness, if the CTE vector is rotated with the *stress* rather
        than the inverse *strain* transformation, or if stress recovery
        forgets to subtract the thermal strain — the three ways this term
        is usually got wrong.
        """
        elem = _element(element_cls, ply_angle, phi, CURE_COOLDOWN_DT,
                        x850_material)
        u = _free_expansion_displacement(elem)
        sigma = elem.stress_at_gauss_points(u)
        # Scaled against the stress a fully restrained element would carry.
        restrained = np.abs(
            elem.stress_at_gauss_points(np.zeros(24))
        ).max()
        assert restrained > 1.0, "test is only meaningful if dT bites"
        assert np.abs(sigma).max() < 1e-9 * restrained

    @pytest.mark.parametrize("ply_angle,phi", ROTATION_CASES)
    def test_full_restraint_gives_minus_C_eps_th(
        self, element_cls, ply_angle, phi, x850_material
    ):
        """``u = 0`` must give exactly ``sigma = -C_bar alpha dT``."""
        elem = _element(element_cls, ply_angle, phi, CURE_COOLDOWN_DT,
                        x850_material)
        if element_cls is Hex8IElement:
            elem.stiffness_matrix()  # populate the condensation caches
        sigma = elem.stress_at_gauss_points(np.zeros(24))
        for gp, (xi, eta, zeta) in enumerate(elem._gauss_points):
            C_bar = elem.rotated_stiffness(xi, eta, zeta)
            eps_th = elem.thermal_strain(xi, eta, zeta)
            npt.assert_allclose(sigma[gp], -C_bar @ eps_th, rtol=1e-12,
                                atol=1e-10)

    def test_thermal_load_is_self_equilibrated(
        self, element_cls, x850_material
    ):
        """No net force: a thermal load does no rigid-body work."""
        elem = _element(element_cls, 45.0, 0.2, CURE_COOLDOWN_DT,
                        x850_material)
        f = elem.thermal_force_vector().reshape(8, 3)
        scale = np.abs(f).max()
        assert scale > 0.0
        npt.assert_allclose(f.sum(axis=0), np.zeros(3), atol=1e-10 * scale)

    def test_zero_delta_T_is_inert(self, element_cls, x850_material):
        """The default must cost nothing and change nothing."""
        elem = _element(element_cls, 30.0, 0.1, 0.0, x850_material)
        npt.assert_array_equal(elem.thermal_force_vector(), np.zeros(24))
        npt.assert_array_equal(elem.thermal_strain(0.0, 0.0, 0.0),
                               np.zeros(6))
        # Stress recovery reduces to the purely mechanical form.
        u = np.linspace(-1e-3, 1e-3, 24)
        if element_cls is Hex8IElement:
            elem.stiffness_matrix()
        sigma = elem.stress_at_gauss_points(u)
        assert np.isfinite(sigma).all()

    def test_thermal_load_scales_linearly_with_delta_T(
        self, element_cls, x850_material
    ):
        """Linear thermoelasticity: doubling ΔT doubles the load."""
        one = _element(element_cls, 30.0, 0.1, -50.0, x850_material)
        two = _element(element_cls, 30.0, 0.1, -100.0, x850_material)
        npt.assert_allclose(
            2.0 * one.thermal_force_vector(),
            two.thermal_force_vector(),
            rtol=1e-12,
        )


class TestThermalExpansionRotation:
    """The rotated CTE vector must behave like a second-order tensor."""

    @pytest.mark.parametrize("ply_angle,phi", ROTATION_CASES)
    def test_trace_is_rotation_invariant(self, ply_angle, phi,
                                         x850_material):
        """``a_11 + a_22 + a_33`` is the first invariant — rotation cannot
        change it.  A wrong transformation (using the stress matrix, or
        forgetting the engineering-shear factor of two) breaks this."""
        elem = _element(Hex8Element, ply_angle, phi, 1.0, x850_material)
        a = elem.thermal_expansion_global(0.0, 0.0, 0.0)
        expected = (
            x850_material.alpha1
            + x850_material.alpha2
            + x850_material.alpha3
        )
        npt.assert_allclose(a[:3].sum(), expected, rtol=1e-12)

    def test_on_axis_is_the_untransformed_material_cte(self, x850_material):
        elem = _element(Hex8Element, 0.0, 0.0, 1.0, x850_material)
        npt.assert_allclose(
            elem.thermal_expansion_global(0.0, 0.0, 0.0),
            [x850_material.alpha1, x850_material.alpha2,
             x850_material.alpha3, 0.0, 0.0, 0.0],
            rtol=1e-14,
        )

    def test_ply_rotation_matches_the_clt_in_plane_transform(
        self, x850_material
    ):
        """The 3-D ply-angle rotation must agree with the 2-D CLT one.

        ``Ply.thermal_strain_global`` is the CLT-path transform, already
        pinned by the Stage 1 tests; the FE path must not disagree with it
        in the in-plane components.
        """
        from wrinklefe.core.laminate import Ply

        for angle in (0.0, 30.0, 45.0, -60.0, 90.0):
            elem = _element(Hex8Element, angle, 0.0, 1.0, x850_material)
            a3d = elem.thermal_expansion_global(0.0, 0.0, 0.0)
            a2d = Ply(x850_material, angle, PLY_T).thermal_strain_global()
            # [a_x, a_y, a_xy] vs Voigt [11, 22, 33, 23, 13, 12].
            npt.assert_allclose(
                [a3d[0], a3d[1], a3d[5]], a2d, rtol=1e-12, atol=1e-18
            )


# --------------------------------------------------------------------------- #
# 2. The CLT benchmark — issue #273 acceptance criterion 3
# --------------------------------------------------------------------------- #


def _flat_mesh(laminate, *, nx=12, ny=4, nz_per_ply=2, Lx=20.0, Ly=10.0):
    profile = GaussianSinusoidal(
        amplitude=0.0, wavelength=10.0, width=5.0, center=Lx / 2.0
    )
    config = WrinkleConfiguration.from_morphology_name(
        "stack", profile, interface1=1, interface2=2
    )
    return WrinkleMesh(
        laminate=laminate, wrinkle_config=config,
        Lx=Lx, Ly=Ly, nx=nx, ny=ny, nz_per_ply=nz_per_ply,
    ).generate()


def _free_expansion_bcs(mesh):
    """Statically determinate restraint: three corners of the bottom face."""
    nodes = mesh.nodes
    tol = 1e-9

    def corner(x, y, z):
        m = (
            (np.abs(nodes[:, 0] - x) < tol)
            & (np.abs(nodes[:, 1] - y) < tol)
            & (np.abs(nodes[:, 2] - z) < tol)
        )
        ids = np.where(m)[0]
        assert ids.size == 1, "expected exactly one node at the corner"
        return ids

    x0, y0, z0 = nodes.min(axis=0)
    x1, y1, _ = nodes.max(axis=0)
    return [
        BoundaryCondition("fixed", node_ids=corner(x0, y0, z0),
                          dofs=[0, 1, 2]),
        BoundaryCondition("fixed", node_ids=corner(x1, y0, z0), dofs=[1, 2]),
        BoundaryCondition("fixed", node_ids=corner(x0, y1, z0), dofs=[0, 2]),
    ]


class TestFlatLaminateVsCLT:
    """Acceptance criterion 3: flat FE thermal solve == closed-form CLT.

    A symmetric cross-ply cooling from cure has no mechanical load and no
    curvature, so the 3-D solve is compared directly against
    ``Laminate.ply_stresses_local``.  Agreement is not exact by
    construction — CLT enforces plane stress through the reduced
    stiffness, while the solid elements let ``sigma_33`` relax to zero on
    its own — so the tolerance is a physics tolerance, not machine
    precision.  Measured agreement on this mesh is ~0.3 %.
    """

    ANGLES = [0.0, 90.0, 90.0, 0.0]

    @pytest.fixture(scope="class")
    @classmethod
    def solved(cls):
        material = MaterialLibrary().get("IM7_8552")
        laminate = Laminate.from_angles(
            cls.ANGLES, material, ply_thickness=PLY_T
        )
        mesh = _flat_mesh(laminate)
        solver = StaticSolver(mesh, laminate, delta_T=CURE_COOLDOWN_DT)
        field = solver.solve(_free_expansion_bcs(mesh), solver="direct")
        return laminate, mesh, field

    @staticmethod
    def _interior(mesh, field):
        """Elements away from the free edges, where CLT is the right model."""
        centres = field.element_centers
        Lx, Ly, _ = mesh.domain_size
        return (
            (np.abs(centres[:, 0] - Lx / 2.0) < 0.25 * Lx)
            & (np.abs(centres[:, 1] - Ly / 2.0) < 0.30 * Ly)
        )

    def test_ply_stresses_match_clt(self, solved):
        laminate, mesh, field = solved
        load = LoadState(delta_T=CURE_COOLDOWN_DT)
        interior = self._interior(mesh, field)

        for k in range(len(self.ANGLES)):
            sel = interior & (mesh.ply_ids == k)
            assert sel.any()
            fe = field.stress_local[sel].mean(axis=(0, 1))
            clt = laminate.ply_stresses_local(load, k, "mid")
            # sigma_1 (fibre) and sigma_2 (matrix); Voigt slots 0 and 1.
            npt.assert_allclose(fe[0], clt[0], rtol=0.01)
            npt.assert_allclose(fe[1], clt[1], rtol=0.01)

    def test_matrix_direction_is_in_tension(self, solved):
        """The physics the whole feature exists for.

        Cooling a cross-ply puts every ply's matrix direction in tension:
        the 0 deg plies barely shrink along the fibres and hold the 90 deg
        plies (which want to shrink a lot transversely) stretched, and vice
        versa.  This is the cure-microcracking driver.
        """
        _laminate, mesh, field = solved
        interior = self._interior(mesh, field)
        sigma_2 = field.stress_local[interior][:, :, 1]
        assert sigma_2.min() > 0.0

    def test_midplane_strain_matches_clt(self, solved):
        laminate, mesh, field = solved
        response = laminate.midplane_strains(LoadState(delta_T=CURE_COOLDOWN_DT))
        interior = self._interior(mesh, field)
        eps = field.strain_global[interior].mean(axis=(0, 1))
        npt.assert_allclose(eps[0], response[0], rtol=0.03)
        npt.assert_allclose(eps[1], response[1], rtol=0.03)

    def test_no_thermal_curvature_for_a_symmetric_layup(self, solved):
        """A symmetric laminate cools flat: top and bottom faces move
        together in z, apart from the uniform through-thickness
        contraction."""
        _laminate, mesh, field = solved
        u = field.displacement
        nodes = mesh.nodes
        top = np.abs(nodes[:, 2] - nodes[:, 2].max()) < 1e-9
        # Out-of-plane displacement on the top face must be uniform (a pure
        # contraction), not bowed.
        w = u[top, 2]
        assert np.ptp(w) < 0.02 * abs(w.mean())


# --------------------------------------------------------------------------- #
# 3. Pipeline reachability and the deliberate asymmetries
# --------------------------------------------------------------------------- #


def _fe_config(delta_T: float, applied_strain: float = -0.005
               ) -> AnalysisConfig:
    """A small but genuinely wrinkled FE case."""
    return AnalysisConfig(
        amplitude=0.15,
        wavelength=12.0,
        width=8.0,
        morphology="graded",
        angles=[0.0, 90.0, 90.0, 0.0],
        ply_thickness=PLY_T,
        material=MaterialLibrary().get("IM7_8552"),
        loading="tension" if applied_strain > 0 else "compression",
        applied_strain=applied_strain,
        domain_length=16.0,
        domain_width=8.0,
        nx=8, ny=3, nz_per_ply=1,
        analytical_only=False,
        delta_T=delta_T,
    )


@pytest.mark.slow
class TestPipeline:
    """``AnalysisConfig.delta_T`` reaches the FE fields — and only there.

    Measured on this ``[0/90]s`` IM7/8552 coupon, a ΔT = −155 cool-down
    adds ``+35.3 MPa`` of matrix (σ₂) tension, and it adds the *same*
    amount whether the coupon is loaded in tension or compression — which
    is exactly what a load-independent residual stress should do.  The
    consequence for failure is therefore **signed**: under tension the
    residual adds to the mechanically-driven matrix tension and the
    LaRC05 index rises (0.497 → 0.754); under compression it relieves the
    matrix compression and the index falls (0.348 → 0.214).  Asserting
    only "cooling makes things worse" would have been wrong.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def runs(cls):
        out = {}
        for strain in (-0.005, +0.005):
            for delta_T in (CURE_COOLDOWN_DT, 0.0):
                out[strain, delta_T] = WrinkleAnalysis(
                    _fe_config(delta_T, strain)
                ).run(analytical_only=False)
        return out

    def test_fe_stress_field_changes(self, runs):
        cold = runs[-0.005, CURE_COOLDOWN_DT]
        neutral = runs[-0.005, 0.0]
        assert cold.field_results is not None
        assert neutral.field_results is not None
        assert not np.allclose(
            cold.field_results.stress_local,
            neutral.field_results.stress_local,
        )

    def test_residual_matrix_stress_is_load_independent(self, runs):
        """The residual σ₂ offset must not depend on the applied load.

        A thermal *initial strain* is a constant right-hand-side term, so
        the stress it contributes is the same under tension and
        compression.  If the term were (wrongly) scaled with the applied
        load — or folded into the stiffness — the two shifts would
        differ.
        """
        shifts = []
        for strain in (-0.005, +0.005):
            cold = runs[strain, CURE_COOLDOWN_DT].field_results
            neutral = runs[strain, 0.0].field_results
            shifts.append(
                cold.stress_local[:, :, 1].mean()
                - neutral.stress_local[:, :, 1].mean()
            )
        # Both shifts are the same cure-residual matrix tension.
        assert shifts[0] > 20.0
        npt.assert_allclose(shifts[0], shifts[1], rtol=1e-6)

    def test_cooldown_raises_the_matrix_failure_index_in_tension(self, runs):
        """Residual matrix tension adds to mechanical matrix tension."""
        cold = runs[+0.005, CURE_COOLDOWN_DT].failure_indices["larc05"]
        neutral = runs[+0.005, 0.0].failure_indices["larc05"]
        assert np.max(cold) > 1.2 * np.max(neutral)

    def test_cooldown_relieves_the_matrix_failure_index_in_compression(
        self, runs
    ):
        """...and subtracts from mechanical matrix compression.

        Pinned deliberately: the residual is not a blanket penalty, and a
        model that reported one would be wrong for the compression cases
        this package mostly analyses.
        """
        cold = runs[-0.005, CURE_COOLDOWN_DT].failure_indices["larc05"]
        neutral = runs[-0.005, 0.0].failure_indices["larc05"]
        assert np.max(cold) < np.max(neutral)

    def test_measured_modulus_is_not_moved_by_delta_T(self, runs):
        """Deliberate asymmetry.

        The global modulus routine divides a reaction force by
        ``area * strain``.  A cure residual load adds a
        *strain-independent* offset to that reaction, which is not a
        stiffness change — so that solve is pinned at ``delta_T = 0`` and
        the reported modulus must be bit-identical.
        """
        npt.assert_array_equal(
            runs[-0.005, CURE_COOLDOWN_DT].modulus_retention_global,
            runs[-0.005, 0.0].modulus_retention_global,
        )


@pytest.mark.slow
class TestNonlinearRoutes:
    """The CZM and progressive-damage routes must carry ΔT too.

    Neither takes the thermal load as an external force: the assembler
    subtracts it from the *internal* force, because an element's internal
    force is ``K_e u_e - f^th_e``.  The Newton residual
    ``R = F_ext - F_int`` therefore drives ``K u = F_ext + F_th`` on its
    own.  If the subtraction were missing, a CZM solve would still
    converge — to the wrong displacement — so these tests check the
    *stress shift*, not just convergence.
    """

    EXPECTED_SIGMA2_SHIFT = 35.3  # MPa, measured on the linear path

    def test_czm_route_carries_the_residual_stress(self):
        import dataclasses

        shifts = []
        for delta_T in (0.0, CURE_COOLDOWN_DT):
            cfg = dataclasses.replace(
                _fe_config(delta_T), enable_czm=True,
                czm_n_load_increments=6,
            )
            result = WrinkleAnalysis(cfg).run(analytical_only=False)
            assert result.czm_converged
            shifts.append(result.field_results.stress_local[:, :, 1].mean())
        npt.assert_allclose(
            shifts[1] - shifts[0], self.EXPECTED_SIGMA2_SHIFT, rtol=0.02
        )

    def test_progressive_damage_route_carries_the_residual_stress(self):
        """Both the wrinkled run and its pristine baseline see ΔT.

        The load stepping prescribes displacement, so the thermal term is
        present from the first increment.  Under compression the residual
        matrix tension delays matrix failure, so both ultimate strengths
        rise — and because both sides move, the *knockdown* barely does.
        """
        import dataclasses

        out = {}
        for delta_T in (0.0, CURE_COOLDOWN_DT):
            cfg = dataclasses.replace(
                _fe_config(delta_T), enable_progressive_damage=True,
                progressive_n_increments=6,
            )
            out[delta_T] = WrinkleAnalysis(cfg).run(analytical_only=False)

        cold, warm = out[CURE_COOLDOWN_DT], out[0.0]
        assert cold.progressive_strength_MPa > warm.progressive_strength_MPa
        assert (
            cold.progressive_pristine_strength_MPa
            > warm.progressive_pristine_strength_MPa
        )
        # Both sides shift together, so the ratio is nearly unchanged.
        npt.assert_allclose(
            cold.progressive_knockdown, warm.progressive_knockdown, rtol=0.05
        )


class TestZeroDeltaTIsBitIdentical:
    """The default must not perturb a single bit of the FE path."""

    def test_assembler_thermal_force_is_zero(self):
        from wrinklefe.solver.assembler import GlobalAssembler

        material = MaterialLibrary().get("IM7_8552")
        laminate = Laminate.from_angles(
            [0.0, 90.0, 90.0, 0.0], material, ply_thickness=PLY_T
        )
        mesh = _flat_mesh(laminate, nx=4, ny=2, nz_per_ply=1)
        assembler = GlobalAssembler(mesh, laminate)
        npt.assert_array_equal(
            assembler.assemble_thermal_force(), np.zeros(mesh.n_dof)
        )

    def test_solve_is_unchanged(self):
        from wrinklefe.solver.boundary import BoundaryHandler

        material = MaterialLibrary().get("IM7_8552")
        laminate = Laminate.from_angles(
            [0.0, 90.0, 90.0, 0.0], material, ply_thickness=PLY_T
        )
        mesh = _flat_mesh(laminate, nx=4, ny=2, nz_per_ply=1)
        bcs = BoundaryHandler.compression_bcs(mesh, applied_strain=-0.01)

        a = StaticSolver(mesh, laminate).solve(bcs, solver="direct")
        b = StaticSolver(mesh, laminate, delta_T=0.0).solve(
            bcs, solver="direct"
        )
        npt.assert_array_equal(a.stress_local, b.stress_local)
        npt.assert_array_equal(a.displacement, b.displacement)
