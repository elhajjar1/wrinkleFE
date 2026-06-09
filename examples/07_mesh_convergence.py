"""Mesh-convergence study: is my mesh fine enough?

Runs the README quick-start case at successively refined meshes via
``mesh_convergence_study``, tabulating the peak failure index, DOF
count, and runtime per level, then recommends the coarsest mesh whose
QoI agrees with the finest level within tolerance. The same study is
available from the command line: ``wrinklefe converge --tolerance 0.02``.

Expected runtime: ~90 s (three FE solves at increasing density).
Expected output:  a 3-level convergence table with shrinking d% and
                  ``07_convergence.png`` (QoI vs DOF, log-x). At these
                  coarse densities the study correctly reports that the
                  peak FI is still mesh-sensitive ("refine further") —
                  exactly the situation the helper exists to expose.
"""

import matplotlib

matplotlib.use("Agg")  # headless-safe; remove to use an interactive backend

from wrinklefe.analysis import AnalysisConfig
from wrinklefe.convergence import mesh_convergence_study

base = AnalysisConfig(
    amplitude=0.366, wavelength=16.0, width=12.0,
    morphology="stack", loading="compression",
)

study = mesh_convergence_study(
    base, levels=3, refine=("nx", "nz_per_ply"),
    qoi="max_fi", tolerance=0.02,
)
print(study.summary())

if study.recommended_config is not None:
    rc = study.recommended_config
    print(
        f"\nUse nx={rc.nx}, ny={rc.ny}, nz_per_ply={rc.nz_per_ply} "
        "for this geometry."
    )

ax = study.plot()
ax.figure.savefig("07_convergence.png", dpi=150)
print("Saved: 07_convergence.png")
