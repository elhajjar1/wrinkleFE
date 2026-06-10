"""Delamination failure reporter built from cohesive-zone damage state.

Phase 3 of the cohesive-zone modelling (CZM) work surfaces interfacial
delamination as a first-class failure mode alongside the existing
fibre / matrix / kinking criteria.  This module is a thin reporter:
given the per-Gauss-point damage state of the inserted
:class:`~wrinklefe.elements.cohesive8.Cohesive8Element` instances, it
constructs a :class:`~wrinklefe.failure.evaluator.LaminateFailureReport`
shaped object so callers can query delamination uniformly with the
other failure modes.

The mapping from CZM state -> LaminateFailureReport is intentionally
coarse:

* The *failure index* for the delamination "criterion" is the maximum
  damage variable :math:`d \\in [0, 1]` across all Gauss points of the
  interface.  ``d = 1`` is fully open; the choice of ``d`` as a
  surrogate FI keeps the contract ``FI >= 1.0 => failed`` with no
  rescaling needed.
* The *load factor* is ``1 / max(FI, eps)`` (the standard
  reserve-factor inverse), capped to a sentinel ``inf`` when no damage
  has accumulated.
* The "ply" reported by FPF / LPF is the ply-interface index of the
  most-damaged interface (one row per inserted cohesive layer).

This module performs **no physics**; it only repackages results.  The
actual CZM physics lives in
:mod:`wrinklefe.elements.cohesive8` and
:mod:`wrinklefe.solver.nonlinear`.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from wrinklefe.failure.evaluator import LaminateFailureReport


def build_delamination_report(
    damage_per_interface: Mapping[int, np.ndarray],
    *,
    energy_per_interface: Mapping[int, float] | None = None,
    crack_length_per_interface: Mapping[int, float] | None = None,
) -> LaminateFailureReport:
    """Wrap per-interface CZM damage into a :class:`LaminateFailureReport`.

    Parameters
    ----------
    damage_per_interface : Mapping[int, np.ndarray]
        Map from ply-interface index to the cohesive damage array for
        that interface.  Each value has shape ``(n_iface_elems, n_gp)``
        or any shape with a meaningful maximum.
    energy_per_interface : Mapping[int, float], optional
        Per-interface dissipated energy in N*mm.  Stored on the report's
        FPF/LPF ``detail`` dicts for downstream introspection.
    crack_length_per_interface : Mapping[int, float], optional
        Per-interface crack length proxy in mm.  Stored on the report's
        FPF/LPF ``detail`` dicts.

    Returns
    -------
    LaminateFailureReport
        Report whose ``critical_criterion`` is ``"delamination"`` when
        at least one interface has damage > 0.  ``ply_failure_indices``
        contains a length-``n_iface`` array of per-interface max damage
        (indexed positionally by sorted interface index).
    """
    report = LaminateFailureReport()
    if not damage_per_interface:
        return report

    iface_indices = sorted(damage_per_interface.keys())
    max_damage_per_iface: list[float] = []
    for idx in iface_indices:
        arr = np.asarray(damage_per_interface[idx], dtype=float)
        max_damage_per_iface.append(float(arr.max()) if arr.size else 0.0)

    fi_array = np.asarray(max_damage_per_iface, dtype=float)
    report.ply_failure_indices = {"delamination": fi_array}

    # Identify the most-damaged interface.  When multiple interfaces tie
    # at zero, default to the first listed one — the report's FPF/LPF
    # then reflects "no delamination yet" with FI = 0.
    crit_pos = int(np.argmax(fi_array)) if fi_array.size else 0
    crit_iface = iface_indices[crit_pos]
    crit_fi = float(fi_array[crit_pos]) if fi_array.size else 0.0

    # Reserve factor / load factor: 1 / FI for non-zero damage, inf
    # otherwise.  We keep the report-level structure identical to the
    # other criteria so consumers can iterate uniformly.
    if crit_fi > 0.0:
        load_factor = 1.0 / crit_fi
    else:
        load_factor = float("inf")

    detail: dict = {
        "interface_indices": list(iface_indices),
        "max_damage_per_interface": list(max_damage_per_iface),
    }
    if energy_per_interface is not None:
        detail["energy_per_interface"] = {
            int(k): float(v) for k, v in energy_per_interface.items()
        }
    if crack_length_per_interface is not None:
        detail["crack_length_per_interface"] = {
            int(k): float(v) for k, v in crack_length_per_interface.items()
        }

    common = {
        "fi": crit_fi,
        "ply": int(crit_iface),
        "mode": "delamination",
        "load_factor": load_factor,
        "detail": detail,
    }
    report.fpf = {"delamination": dict(common)}
    report.lpf = {"delamination": dict(common)}
    report.critical_ply = int(crit_iface)
    report.critical_mode = "delamination"
    # Only mark delamination as the critical criterion when damage is
    # non-trivial; otherwise leave the field empty so a downstream
    # combiner can prefer a real (FI-driven) criterion instead.
    report.critical_criterion = "delamination" if crit_fi > 0.0 else ""
    return report


__all__ = ["build_delamination_report"]
