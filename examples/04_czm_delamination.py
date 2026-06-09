"""Cohesive-zone delamination at a wrinkle under tension.

Enables the CZM pathway (``enable_czm=True``): cohesive interface
elements are inserted near the wrinkle crest and the incremental
Newton-Raphson solver is driven to the applied strain, tracking
interface damage. A concave tension wrinkle concentrates peel stress at
the crest and develops meaningful (but sub-critical) cohesive damage.

Expected runtime: ~10 s (small mesh, 20 load increments).
Expected output:  printed CZM summary with converged=True, non-zero
                  max interface damage, and crack length per interface.
"""

import numpy as np

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis
from wrinklefe.core.material import MaterialLibrary

mat = MaterialLibrary().get("IM7_8552")
layup = ([0, 90] * 4) + ([90, 0] * 4)  # [0/90]_4s, 16 plies

config = AnalysisConfig(
    amplitude=0.366, wavelength=16.0, width=12.0,
    morphology="concave", loading="tension",
    material=mat, angles=layup, ply_thickness=0.183,
    nx=12, ny=4, nz_per_ply=1,          # coarse mesh keeps this fast
    applied_strain=0.015,
    enable_czm=True,
    czm_n_load_increments=20,
)
result = WrinkleAnalysis(config).run()
print(result.summary())

print(f"\nConverged:            {result.czm_converged}")
print(f"Max interface damage: {float(np.max(result.czm_damage)):.4f}")
if result.czm_crack_length_per_interface:
    for iface, length in sorted(
        result.czm_crack_length_per_interface.items()
    ):
        print(f"Crack length, interface {iface}: {length:.4e} mm")
else:
    print("Crack length per interface: (none — damage is sub-critical)")
