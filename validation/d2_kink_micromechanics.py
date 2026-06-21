#!/usr/bin/env python3
"""D.2 assessment: can kink-band micromechanics anchor the gate's gamma_Y?

The D.3 (theta, D/T) penetration gate uses a Budiansky-Fleck angle floor
``KD_angle = 1 / (1 + theta / gamma_Y)`` with a *fitted* ``gamma_Y``
(0.26 for E, 0.62 for F).  D.2 of the modelling plan proposed anchoring
that ``gamma_Y`` with a first-principles kink-band micromechanics model
(Pimenta-Pinho) instead of a fit.

This script tests whether that is possible.  It compares, on the Li 2025
F constant-penetration triple (S-M-1/2/3, D/T = 0.122 fixed so the gate
floor approaches the pure angle response), three predictions:

* **Argon / Budiansky-Fleck** with the *physical* matrix-shear-yield
  strain ``gamma_y = S12 / G12`` and a typical inherent misalignment:
  ``KD = (phi_inh + gamma_y) / (phi_inh + theta + gamma_y)``.  This is the
  leading-order behaviour of every foundation-based kink model, including
  Pimenta-Pinho.
* **Gate floor** with the fitted ``gamma_Y``.
* **Measured**.

Result (see ``__main__``): the physical kink micromechanics over-predicts
the knockdown by ~10x (gamma_y ~ 0.011 vs the fitted 0.62, a 57x gap), so
**Pimenta-Pinho cannot anchor the gate's gamma_Y** — the foundation models
all give catastrophic over-knock at moderate angles.  The missing physics
is the fibre *bending* stiffness / couple-stress length scale (Fleck-Shu,
consistent with the D.4 finding), which mitigates the knockdown and which
the fitted ``gamma_Y`` absorbs.  Compressive kinking strength is, in the
state of the art, a *calibrated* quantity — the gate is the right
modelling level.
"""
from __future__ import annotations

import numpy as np

# Li 2025 vacuum-bag AC318 (Dataset F).
S12, G12 = 60.0, 5500.0
GATE_GAMMA_Y_F = 0.6215           # fitted gate gamma_Y for F (D.3)
PHI_INH = np.radians(1.0)          # typical inherent fibre misalignment

# F constant-D/T triple: (theta_deg, measured KD).  At D/T = 0.122 the
# gate's penetration factor S ~ 1, so the floor ~ the pure angle response.
TRIPLE = [(10.3, 0.891), (20.1, 0.629), (30.2, 0.472)]


def kd_argon(theta_rad, gamma_y, phi_inh=PHI_INH):
    """Argon / Budiansky-Fleck knockdown with physical matrix yield."""
    return (phi_inh + gamma_y) / (phi_inh + theta_rad + gamma_y)


def kd_gate_floor(theta_rad, gamma_Y):
    return 1.0 / (1.0 + theta_rad / gamma_Y)


if __name__ == "__main__":
    gamma_y_phys = S12 / G12
    print(f"physical matrix-shear-yield strain  gamma_y = S12/G12 = "
          f"{gamma_y_phys:.4f}")
    print(f"fitted gate angle-floor parameter   gamma_Y          = "
          f"{GATE_GAMMA_Y_F:.4f}  ({GATE_GAMMA_Y_F / gamma_y_phys:.0f}x)\n")
    print(f"{'theta':>6} {'measured':>9} {'Argon/BF(phys)':>15} "
          f"{'gate floor':>11}")
    ea, eg = [], []
    for th_deg, kd in TRIPLE:
        th = np.radians(th_deg)
        a = kd_argon(th, gamma_y_phys)
        g = kd_gate_floor(th, GATE_GAMMA_Y_F)
        ea.append(abs(a - kd) / kd)
        eg.append(abs(g - kd) / kd)
        print(f"{th_deg:6.1f} {kd:9.3f} {a:15.3f} {g:11.3f}")
    print(f"\nMAE  Argon/BF (physical gamma_y): {100 * np.mean(ea):5.1f} %")
    print(f"MAE  gate floor (fitted gamma_Y): {100 * np.mean(eg):5.1f} %")
    print("\nConclusion: foundation kink micromechanics (incl. "
          "Pimenta-Pinho) over-predict the knockdown ~10x and cannot "
          "anchor the gate's gamma_Y; the fitted value absorbs the "
          "fibre-bending / couple-stress mitigation (Fleck-Shu). "
          "Compressive kinking is calibrated, not first-principles "
          "predicted -- the gate is the right modelling level.")
