"""Manufacturing process: an analysis input and its validated scope (#445).

The knockdown models have been checked against laminates made by a few
manufacturing routes only. Laminate quality differs between routes
(fibre volume fraction, porosity, how straight the fibres are in the
undamaged laminate), and no model represents it. A wrinkle in a laminate
made some other way is therefore outside the validated scope, whatever
the models return.

:class:`ManufacturingProcess` names the route.
``AnalysisConfig.manufacturing_process`` takes one;
:func:`process_scope_caveat` returns the warning for a route outside the
scope, and :func:`gate_scope_caveat` the warning for a penetration-gate
preset used outside the material system and route it was fitted to.
Both are reporting only: no knockdown changes.

The scope is not a hand-kept list. It is derived from
:data:`VALIDATION_EVIDENCE`, the route of every validation dataset as
its source states it: a route is in scope when at least one dataset that
is not quarantined was made that way. Datasets whose source does not
state the route add nothing to the scope. A wrinkle dataset made by a new
route moves that route into scope when it is added here (and to the
validation ledger, whose ``process`` entries a test keeps in step with
this table).
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wrinklefe.core.penetration_gate import GateParameters


class ManufacturingProcess(str, Enum):
    """How the laminate was made."""

    PREPREG_AUTOCLAVE = "prepreg_autoclave"
    PREPREG_PRESS = "prepreg_press"
    PREPREG_VACUUM_BAG = "prepreg_vacuum_bag"
    RTM = "rtm"
    INFUSION = "infusion"
    WET_LAYUP = "wet_layup"
    OTHER = "other"
    UNSPECIFIED = "unspecified"

    @property
    def label(self) -> str:
        """Plain-language name, for messages and the app."""
        return _LABELS[self]


_LABELS = {
    ManufacturingProcess.PREPREG_AUTOCLAVE: "prepreg, autoclave cure",
    ManufacturingProcess.PREPREG_PRESS: "prepreg, press or closed-mould cure",
    ManufacturingProcess.PREPREG_VACUUM_BAG:
        "prepreg, vacuum-bag oven cure (out of autoclave)",
    ManufacturingProcess.RTM: "resin transfer moulding (RTM)",
    ManufacturingProcess.INFUSION: "vacuum infusion (VARTM)",
    ManufacturingProcess.WET_LAYUP: "wet lay-up",
    ManufacturingProcess.OTHER: "another process",
    ManufacturingProcess.UNSPECIFIED: "not specified",
}


def parse_process(value: ManufacturingProcess | str) -> ManufacturingProcess:
    """Return ``value`` as a :class:`ManufacturingProcess`.

    Accepts the enum, its value (``"wet_layup"``) or the same with
    hyphens (``"wet-layup"``, the CLI spelling).

    Raises
    ------
    ValueError
        For anything else, naming the accepted values.
    """
    if isinstance(value, ManufacturingProcess):
        return value
    if isinstance(value, str):
        key = value.strip().lower().replace("-", "_")
        try:
            return ManufacturingProcess(key)
        except ValueError:
            pass
    accepted = ", ".join(p.value for p in ManufacturingProcess)
    raise ValueError(
        f"manufacturing_process must be one of {accepted}; got {value!r}"
    )


@dataclass(frozen=True)
class ProcessEvidence:
    """The manufacturing route(s) of one validation dataset.

    ``processes`` lists every route the dataset's laminates were made by
    (some datasets used more than one), and is empty when the source does
    not state the route; ``source`` says where it is stated, or that it is
    not.
    """

    dataset: str
    processes: tuple[ManufacturingProcess, ...]
    source: str
    quarantined: bool = False


#: The route of every validation dataset, as its source states it.
VALIDATION_EVIDENCE: tuple[ProcessEvidence, ...] = (
    ProcessEvidence(
        "A/B Elhajjar (2025)",
        (ManufacturingProcess.PREPREG_AUTOCLAVE,
         ManufacturingProcess.PREPREG_PRESS),
        "T700/2510 prepreg, autoclave or hot-press cure, with high "
        "consolidation and fibre volume fraction (the author, R. Elhajjar, "
        "2026)",
    ),
    ProcessEvidence(
        "C Mukhopadhyay, Jones & Hallett (2015)", (),
        "IM7/8552 prepreg; cure route not recorded in the repository",
    ),
    ProcessEvidence(
        "D Wang (2021)", (),
        "T800S/M21 prepreg; cure route not recorded in the repository",
    ),
    ProcessEvidence(
        "E Li et al. (2024)", (ManufacturingProcess.PREPREG_PRESS,),
        "AC318/S6C10 prepreg moulded at 2.9 MPa",
    ),
    ProcessEvidence(
        "F Li, Li, Ge & Liang (2025)", (ManufacturingProcess.PREPREG_VACUUM_BAG,),
        "the same AC318/S6C10 prepreg cured under a ~1 bar vacuum bag",
    ),
    ProcessEvidence(
        "G Hsiao & Daniel (1996)", (),
        "IM6G/3501-6 prepreg; cure route not recorded in the repository",
    ),
    ProcessEvidence(
        "H Shi et al. (2025)", (), "not stated in the paper",
    ),
    ProcessEvidence(
        "I Calvo et al. (2023)", (),
        "M21E/IMA prepreg; cure route not stated in the paper",
    ),
    ProcessEvidence(
        "J Thor et al. (2021)", (ManufacturingProcess.PREPREG_AUTOCLAVE,),
        "IM7/8552 prepreg, 'fabricated in an autoclave following the "
        "recommended curing cycle'",
    ),
    ProcessEvidence(
        "K Pilato et al. (2022)", (),
        "industrial part, 'standard industry practices'; route not stated",
    ),
    ProcessEvidence(
        "M Elhajjar & Shams (2016)", (ManufacturingProcess.PREPREG_PRESS,),
        "T700GC/2510 prepreg, 'press-claved' (El-Hajjar & Petersen 2011, "
        "Sec. 3.1; Elhajjar & Shams 2016, Sec. 2.1)",
    ),
    ProcessEvidence(
        "L Krishnappa & Gururaja (2024)", (ManufacturingProcess.WET_LAYUP,),
        "wet lay-up of stitched fabric, room-temperature cure under a "
        "vacuum bag (quarantined, issue #445)",
        quarantined=True,
    ),
)

#: Routes at least one non-quarantined validation dataset was made by.
VALIDATED_PROCESSES: frozenset[ManufacturingProcess] = frozenset(
    p for e in VALIDATION_EVIDENCE if not e.quarantined for p in e.processes
)


def process_in_scope(process: ManufacturingProcess | str) -> bool:
    """``True`` unless ``process`` is specified and not validated.

    ``unspecified`` is in scope: it is the default and keeps the
    behaviour from before the input existed.
    """
    p = parse_process(process)
    return p is ManufacturingProcess.UNSPECIFIED or p in VALIDATED_PROCESSES


def _validated_labels() -> str:
    order = list(ManufacturingProcess)
    return "; ".join(p.label for p in sorted(VALIDATED_PROCESSES,
                                             key=order.index))


#: Added to the out-of-scope warning for wet lay-up: the one wrinkle
#: dataset made that way, and what the models did on it.
_WET_LAYUP_NOTE = (
    " The one wet lay-up wrinkle dataset (Krishnappa & Gururaja 2024) is "
    "quarantined: its undamaged laminate was itself weak (46% fibre "
    "volume, 4.2% porosity, a baseline outside the test standard). On it "
    "the kink-band "
    "knockdown was conservative by about 56% on average, while first-ply "
    "FE over-predicted retained strength by up to about 100%."
)


def process_scope_caveat(process: ManufacturingProcess | str) -> str | None:
    """The out-of-scope warning for ``process``, or ``None`` in scope.

    Plain text, so the summary, the exports and the app share one
    wording.
    """
    p = parse_process(process)
    if process_in_scope(p):
        return None
    text = (
        f"The laminate's manufacturing process ({p.label}) is outside the "
        f"validated scope. The knockdown models have been checked only on "
        f"laminates made by: {_validated_labels()}. Laminate quality "
        f"(fibre volume fraction, porosity, fibre straightness) differs "
        f"between processes and no model represents it, so treat every "
        f"knockdown here as unvalidated for this laminate and confirm it "
        f"by test."
    )
    if p is ManufacturingProcess.WET_LAYUP:
        text += _WET_LAYUP_NOTE
    return text


def gate_scope_caveat(
    gate: GateParameters | None,
    material_name: str | None,
    process: ManufacturingProcess | str,
) -> str | None:
    """Warning for a penetration-gate preset used outside its fit.

    ``None`` when no gate is set, the gate records no fit (a user's own
    calibration), or the material and the process both match it. An
    unspecified process is not held against the gate; a material name
    outside the gate's fitted material system is.
    """
    if gate is None or (not gate.fitted_materials
                        and gate.fitted_process is None):
        return None
    p = parse_process(process)
    reasons = []
    if gate.fitted_materials and material_name not in gate.fitted_materials:
        reasons.append(f"the material ({material_name})")
    if (gate.fitted_process is not None
            and p is not ManufacturingProcess.UNSPECIFIED
            and p is not gate.fitted_process):
        reasons.append(f"the process ({p.label})")
    if not reasons:
        return None
    fitted_process = (
        gate.fitted_process.label if gate.fitted_process is not None
        else "an unrecorded process"
    )
    materials = " / ".join(gate.fitted_materials) or "an unrecorded material"
    return (
        f"The penetration-gate preset {gate.name} was fitted to the "
        f"{materials} material cards (process: {fitted_process}); here "
        f"{' and '.join(reasons)} "
        f"{'differ' if len(reasons) > 1 else 'differs'}. Used on "
        f"unidirectional carbon they were not fitted to, the shipped "
        f"presets over-predicted retained strength on 7 (moulded) and 10 "
        f"(vacuum-bag) of 10 cases, by up to about 97% and 180% (issue "
        f"#435). Calibrate the gate on this material and process before "
        f"relying on its knockdown."
    )


def wrap_warning(prefix: str, text: str, width: int = 61) -> list[str]:
    """Indented, wrapped ``WARNING`` lines for the text summary."""
    return textwrap.wrap(
        f"WARNING ({prefix}): {text}", width=width,
        initial_indent="    ", subsequent_indent="    ",
    )
