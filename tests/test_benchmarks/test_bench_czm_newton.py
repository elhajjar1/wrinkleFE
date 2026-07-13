"""Benchmark: a bounded cohesive (CZM) Newton solve on a coarse DCB.

A deliberately small Double-Cantilever-Beam (few elements, a handful of
displacement increments) exercises the cohesive-element + Newton-Raphson
path — the most expensive kernel in the suite — while staying well under
~30 s. The invariant is that the opening drives real cohesive damage and
a finite reaction load, so a broken CZM path cannot pass as "fast".
"""
from __future__ import annotations

import numpy as np
import pytest

from wrinklefe.core.cohesive_mesh import insert_cohesive_interface
from wrinklefe.core.laminate import Laminate, Ply
from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.core.mesh import WrinkleMesh
from wrinklefe.elements.cohesive8 import CohesiveProperties
from wrinklefe.solver.assembler import GlobalAssembler
from wrinklefe.solver.boundary import BoundaryCondition, BoundaryHandler
from wrinklefe.solver.nonlinear import NewtonRaphsonSolver

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]

L_TOTAL = 40.0
WIDTH = 25.0
H_ARM = 1.5
A0_PRECRACK = 20.0
NX = 10
NY = 1
NZ_PER_ARM = 2

MAT = OrthotropicMaterial(
    name="DCB_CFRP",
    E1=135_000.0, E2=9_000.0, E3=9_000.0,
    G12=5_000.0, G13=5_000.0, G23=3_000.0,
    nu12=0.30, nu13=0.30, nu23=0.40,
)
COH_PROPS = CohesiveProperties(
    K=1.0e6, sigma_max=25.0, tau_max=25.0,
    GIc=0.28, GIIc=0.79, eta_BK=1.45, beta=1.0,
)
DELTA_MAX = 2.5
N_INC = 5


def _laminate() -> Laminate:
    return Laminate([
        Ply(material=MAT, angle=0.0, thickness=H_ARM),
        Ply(material=MAT, angle=0.0, thickness=H_ARM),
    ])


def _build_mesh():
    wm = WrinkleMesh(
        laminate=_laminate(), wrinkle_config=None,
        Lx=L_TOTAL, Ly=WIDTH, nx=NX, ny=NY, nz_per_ply=NZ_PER_ARM,
    )
    base_mesh = wm.generate()
    z_mid = 0.5 * (
        float(base_mesh.nodes[:, 2].min()) + float(base_mesh.nodes[:, 2].max())
    )
    new_mesh, all_coh = insert_cohesive_interface(
        base_mesh, z_interface=z_mid, cohesive_props=COH_PROPS,
    )
    kept = [c for c in all_coh if float(c.node_coords[:4, 0].mean()) >= A0_PRECRACK]
    for k, c in enumerate(kept):
        c.elem_id = k
    return new_mesh, kept


def _build_bcs(mesh, delta):
    tol = 1e-6
    z, x, y = mesh.nodes[:, 2], mesh.nodes[:, 0], mesh.nodes[:, 1]
    z_min, z_max = float(z.min()), float(z.max())
    x_min_v, x_max_v = float(x.min()), float(x.max())
    on_z_min, on_z_max = np.abs(z - z_min) <= tol, np.abs(z - z_max) <= tol
    on_x_min, on_x_max = np.abs(x - x_min_v) <= tol, np.abs(x - x_max_v) <= tol
    on_y_min = np.abs(y - float(y.min())) <= tol
    on_y_max = np.abs(y - float(y.max())) <= tol

    def fn(m):
        return np.flatnonzero(m).astype(np.intp)

    half = 0.5 * float(delta)
    return [
        BoundaryCondition(
            bc_type="fixed", node_ids=fn(on_x_max & on_z_min), dofs=[0, 1, 2]
        ),
        BoundaryCondition(
            bc_type="fixed", node_ids=fn(on_x_max & on_z_max), dofs=[0, 1]
        ),
        BoundaryCondition(bc_type="fixed", node_ids=fn(on_y_min), dofs=[1]),
        BoundaryCondition(bc_type="fixed", node_ids=fn(on_y_max), dofs=[1]),
        BoundaryCondition(
            bc_type="displacement", node_ids=fn(on_x_min & on_z_max),
            dofs=[2], value=+half,
        ),
        BoundaryCondition(
            bc_type="displacement", node_ids=fn(on_x_min & on_z_min),
            dofs=[2], value=-half,
        ),
    ]


def _drive():
    mesh, cohesive_elements = _build_mesh()
    assembler = GlobalAssembler(
        mesh=mesh, laminate=_laminate(),
        cohesive_elements=[(c.elem_id, c) for c in cohesive_elements],
    )
    bc_handler = BoundaryHandler(mesh)
    solver = NewtonRaphsonSolver(
        assembler=assembler, bc_handler=bc_handler,
        boundary_conditions=_build_bcs(mesh, 0.0),
        n_increments=1, max_newton_iter=100,
        tol_residual=1e-4, tol_absolute=1e-8, tol_displacement=1e-9,
        line_search=False,
    )
    u = np.zeros(mesh.n_dof)
    step = DELTA_MAX / N_INC
    P_final = 0.0
    for i in range(N_INC):
        bcs = _build_bcs(mesh, (i + 1) * step)
        cons = bc_handler.get_constrained_dofs(bcs)
        F_ext = bc_handler.get_force_dofs(bcs)
        u_new, _n, ok = solver._newton_step(u, F_ext, cons, verbose=False, inc=1)
        if not ok:
            continue
        u = u_new
        solver._commit_state()
        F_int = assembler.assemble_internal_force(u)
        P_final = float(np.abs(F_int).max())
    max_d = max(
        (s.d for cid in range(len(cohesive_elements)) for s in assembler.cohesive_state[cid]),
        default=0.0,
    )
    n_coh = len(cohesive_elements)
    return max_d, P_final, n_coh


def test_bench_czm_newton_dcb(benchmark):
    max_d, P_final, n_coh = benchmark.pedantic(
        _drive, rounds=1, iterations=1, warmup_rounds=0
    )

    # Correctness invariant: cohesive elements exist, the opening drove
    # real damage (0 < d), and the reaction load is finite and non-zero.
    assert n_coh > 0
    assert 0.0 < max_d <= 1.0
    assert np.isfinite(P_final)
    assert P_final > 0.0
