"""Crest-to-crest delamination link-up between two adjacent wrinkles.

Combines the two headline capabilities — multi-wrinkle FE (a ``wrinkles``
list of :class:`WrinkleSpec`) and cohesive-zone delamination
(``enable_czm=True``) — that issue #283 made composable. Two wrinkles are
placed on the *same* ply interface with their crests only ~6 mm apart.
With ``czm_interfaces="near_crest"`` (the default) both wrinkles nominate
that one shared interface, which is deduplicated into a single continuous
cohesive surface running the full coupon length. Under tension the peel
stress fields of the two crests overlap, so cohesive damage initiates at
each crest AND bridges the gap between them — the crest-to-crest link-up
that is the recognised multi-wrinkle failure pattern (Li et al. 2025).

Contrast: two *far-separated* wrinkles behave like two independent
single-wrinkle solves with zero damage in the gap; the bridging below is
the genuine wrinkle-interaction signal.

Expected runtime: ~5 s (8 plies, coarse mesh, 8 load increments).
Expected output:  printed CZM summary with converged=True, the single
                  shared interface [3], and bridge damage between the
                  crests comparable to the crest damage itself.
"""

import numpy as np

from wrinklefe.analysis import AnalysisConfig, WrinkleAnalysis, WrinkleSpec
from wrinklefe.core.material import MaterialLibrary

mat = MaterialLibrary().get("IM7_8552")
angles = [0.0, 45.0, -45.0, 90.0, 90.0, -45.0, 45.0, 0.0]  # [0/45/-45/90]_s

# Two wrinkles on ply interface 3. phase = 2*pi*dx/lambda, so +/-0.75*pi
# offsets the crests by +/-3 mm about the coupon centre -> 6 mm apart,
# with overlapping Gaussian supports (width=2 mm).
wrinkles = [
    WrinkleSpec(amplitude=0.3, wavelength=8.0, width=2.0,
                ply_interface=3, phase_offset=-0.75 * np.pi),
    WrinkleSpec(amplitude=0.3, wavelength=8.0, width=2.0,
                ply_interface=3, phase_offset=+0.75 * np.pi),
]

config = AnalysisConfig(
    amplitude=0.3, wavelength=8.0, width=2.0,   # reference wrinkle metadata
    morphology="graded", loading="tension",
    material=mat, angles=angles, ply_thickness=0.183,
    nx=32, ny=2, nz_per_ply=1,                  # coarse mesh keeps this fast
    domain_length=28.0,
    applied_strain=0.03,                        # 3% tension opens the crests
    enable_czm=True,
    czm_interfaces="near_crest",                # -> the one shared interface
    czm_n_load_increments=8,
    wrinkles=wrinkles,
)
result = WrinkleAnalysis(config).run()
print(result.summary())

print(f"\nConverged:              {result.czm_converged}")
print(f"Interfaces used:        {result.czm_interfaces_used}")
print(f"Max interface damage:   {float(np.max(result.czm_damage)):.4f}")

# Quantify the link-up: peak damage in the gap between the crests vs at
# the crests. A bridge value comparable to the crest value is the
# link-up signal (a far-separated pair would read ~0 in the gap).
x = result.czm_element_centroids[:, 0]
d = result.czm_damage.max(axis=1)
L = config.domain_length
d_bridge = float(d[np.abs(x - L / 2.0) < 1.5].max())
d_crests = float(
    max(
        d[np.abs(x - (L / 2.0 - 3.0)) < 2.0].max(),
        d[np.abs(x - (L / 2.0 + 3.0)) < 2.0].max(),
    )
)
print(f"Crest damage:           {d_crests:.4f}")
print(f"Bridge (gap) damage:    {d_bridge:.4f}")
if d_bridge > 0.5 * d_crests:
    print("=> crest-to-crest link-up: damage bridges the gap between wrinkles.")
else:
    print("=> crests damaged independently (no significant bridging).")
