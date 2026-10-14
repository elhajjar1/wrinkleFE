#!/usr/bin/env python3
"""Size-aware FE candidates for embedded flat-surface wrinkles (issue #453).

On embedded wrinkles under flat surfaces the first-ply FE strength path
(``strength_error_summary._fe_first_ply``) over-predicts retained strength:
it reads a wrinkle's angle but not its size, and ignores the delamination
that starts at the wrinkle. This driver scores two candidate replacements
on the datasets where that bites, on the same configs and FE meshes as the
head-to-head matrix of ``strength_error_summary.py``:

* **(a) progressive peak** -- the peak carried stress of the ply-discount
  :class:`~wrinklefe.solver.progressive_damage.ProgressiveDamageSolver`,
  run exactly as ``AnalysisConfig.enable_progressive_damage`` runs it
  (ramp to 1.8 x the 0-deg ply's ``Xc / E1``, its increment count and
  residual stiffness), over the same solver's pristine flat coupon.
* **(b) CZM initiation** -- the carried stress at which the first point of
  any ply interface reaches its mixed-mode interlaminar strength, from the
  CZM path with a cohesive layer at *every* interface (``czm_interfaces=
  "all"``), read from :class:`~wrinklefe.solver.czm_history.CzmDamageHistory`.
  Threshold: ``d > 0``, the bilinear law's own Benzeggagh-Kenane initiation
  condition. It adds no free parameter, ``d`` is exactly zero below it (no
  noise floor to tune), and the pre-initiation response is linear, so the
  onset load is exact and increment-independent (one elastic increment,
  scaled). Any later threshold (a damaged area, ``d = 1``) depends on how
  well these meshes resolve the cohesive zone (about one element here) and
  on the increment size. Dataset M's measured onset is itself
  interlaminar-damage onset (the DIC limit point). Normalised by the
  pristine progressive peak and capped at 1: initiation after the pristine
  coupon would have failed means no delamination-driven knockdown.
* **(c) = min(a, b)** -- whichever comes first.
* **separation** (sensitivity, ``--measures sep``) -- the first increment
  with a fully separated (``d = 1``) interface point, from a nonlinear
  ramp at the near-crest interface. Resolved to one increment, and
  mesh-dependent at these meshes; reported to show whether a later
  threshold changes the conclusion.

Datasets: C (Mukhopadhyay, compression), D (Wang), E (Li 2024), L-WM
(Krishnappa & Gururaja 2024, embedded wrinkle-middle; quarantined, issue
#445) as knockdowns; M (Elhajjar & Shams 2016) damage onset in absolute
stress against the measured DIC limit point, exactly as
``fe_onset_elhajjar_shams`` scores first-ply FE.

Scoring: signed error ``(pred - meas) / meas``; *unsafe* = above
``+0.5 %`` (``strength_error_summary._NON_CONSERVATIVE_PCT``). The bar of
issue #453 is a worst case of at most +15 % on every dataset. Size
tracking: Spearman rank correlation of each prediction with wrinkle size
(D/T for E, amplitude elsewhere), next to the measured one, and Dataset E's
fixed-angle series (15.8-16.7 deg, D/T 0.026-0.083) where the measured
knockdown falls from 0.758 to 0.523 at one angle.

FE solves are minutes each, so every result is cached per case in a JSON
file (``--cache``, default ``_parked/size_aware_fe_cache.json``, a
git-ignored directory) and a re-run only computes what is missing.

Run::

    python validation/size_aware_fe_study.py                     # all
    python validation/size_aware_fe_study.py --datasets E,L-WM --measures init
    python validation/size_aware_fe_study.py --report-only       # cache only
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "validation"))

import strength_error_summary as ses  # noqa: E402

DATASETS = ("C", "D", "E", "L-WM", "M")
MEASURES = ("first", "init", "prog", "sep")
DEFAULT_CACHE = REPO / "_parked" / "size_aware_fe_cache.json"

#: Elastic CZM increment for the initiation load: any strain below
#: initiation gives the same, exact answer (the response is linear there).
_INIT_STRAIN = 1.0e-4

#: Mesh overrides for (b) where the head-to-head mesh does not converge the
#: initiation load. Mesh check (one case per dataset, doubling nx, and
#: doubling nz_per_ply): C, D, L-WM and M move by at most 4 %; Dataset E's
#: nx = 16 (3 mm columns on E-7) does not -- E-7 656 -> 549 -> 526 -> 521
#: -> 517 MPa at nx = 16/32/48/64/96, E-3 591 -> 484 -> 478 -> 475 at
#: nx = 16/48/64/96 -- so E is scored at nx = 48 (within 2 % of nx = 96).
INIT_MESH_OVERRIDE: dict[str, dict] = {"E": {"nx": 48}}

#: Separation ramp: to this multiple of the initiation strain, in this many
#: equal increments (so one increment is 1/20 of the initiation strain).
SEP_RAMP = 3.0
SEP_INCREMENTS = 60

#: Issue #453's bar: worst signed error on every dataset at most this.
UNSAFE_BAR_PCT = 15.0

#: Dataset E's fixed-angle size series (degrees).
_E_FIXED_ANGLE = (15.5, 17.0)


@dataclasses.dataclass
class Case:
    """One scored case and its FE recipe."""

    dataset: str
    case: str
    cfg: object           # AnalysisConfig (analytical recipe)
    fe: dict              # FE overrides: mesh + FE-only recipe
    measured: float       # knockdown, or MPa when ``absolute``
    absolute: bool        # Dataset M onset: scored in MPa
    size: float           # D/T (E) or amplitude A (mm)
    theta: float          # peak angle (deg)
    geometry: str         # dedup key: M repeats geometries
    pristine: str         # pristine-coupon key (laminate + mesh)


def _ledger() -> dict:
    return json.loads(ses.LEDGER.read_text())


def _ledger_ds(prefix: str) -> list[dict]:
    return [d for d in _ledger()["datasets"] if d["name"].startswith(prefix)]


def _key(cfg, fe: dict) -> str:
    """Pristine-coupon key: laminate, loading sign and mesh density.

    A flat coupon under the end-displacement BCs is uniformly loaded, so
    its strength does not depend on its length; one solve serves every
    case that shares this key.
    """
    mat = getattr(cfg.material, "name", str(cfg.material))
    sign = "c" if cfg.applied_strain < 0 else "t"
    mesh = (fe.get("nx", cfg.nx), fe.get("ny", cfg.ny),
            fe.get("nz_per_ply", cfg.nz_per_ply))
    return (f"{mat}|{','.join(str(a) for a in cfg.angles)}|"
            f"{cfg.ply_thickness}|{sign}|{mesh}")


def cases(dataset: str) -> list[Case]:
    """The scored cases of one dataset, with the head-to-head FE recipe."""
    from validate import case_config

    out: list[Case] = []
    if dataset in ("C", "D"):
        label = "C Mukhopadhyay comp" if dataset == "C" else "D Wang conc/conv"
        fe = dict(ses._MD_FE_MESH)
        for name, theta, kd, cfg in ses._md_cases()[label]:
            out.append(Case(dataset, name, cfg, fe, kd, False,
                            float(cfg.amplitude), theta, name, _key(cfg, fe)))
    elif dataset == "E":
        fe = dict(ses._LI_FE_MESH)
        for name, theta, dt, kd, cfg in ses._li2024_configs():
            out.append(Case(dataset, name, cfg, fe, kd, False, dt, theta,
                            name, _key(cfg, fe)))
    elif dataset == "L-WM":
        (ds,) = _ledger_ds("krishnappa_2024")
        for c in ds["cases"]:
            if "-WM-" not in c["case"]:
                continue
            cfg = case_config(ds, c)
            fe = {**ses._JK_FE_MESH, **c.get("fe_recipe", {})}
            out.append(Case(dataset, c["case"], cfg, fe,
                            float(c["measured_kd"]), False,
                            float(cfg.amplitude), float(c["peak_angle_deg"]),
                            c["case"], _key(cfg, fe)))
    elif dataset == "M":
        scored = ses._m_onset_scored()
        fe = dict(ses._M_FE_RECIPE)
        for ds in _ledger_ds("elhajjar_shams_2016"):
            for c in ds["cases"]:
                if c["case"] not in scored:
                    continue
                cfg = case_config(ds, c)
                geom = (f"{ds['layup']}|{c['amplitude_p2p_mm']}|"
                        f"{c['wavelength_mm']}|{c.get('width_mm')}")
                out.append(Case(dataset, c["case"], cfg, fe,
                                float(c["measured_onset_MPa"]), True,
                                float(cfg.amplitude),
                                float(c["peak_angle_deg"]), geom,
                                _key(cfg, fe)))
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    return out


# ----------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------

class Cache:
    """Per-case results, persisted after every solve.

    Writes re-read the file under an exclusive lock and merge, so several
    driver processes (one per dataset) can share one cache.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict = (json.loads(path.read_text())
                           if path.exists() else {})

    def get(self, key: str):
        return self.data.get(key)

    def put(self, key: str, value) -> None:
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_suffix(".lock")
        with lock.open("w") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            if self.path.exists():
                self.data = {**json.loads(self.path.read_text()), **self.data}
            self.data[key] = value
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True))
            tmp.replace(self.path)


# ----------------------------------------------------------------------
# Solves
# ----------------------------------------------------------------------

def _progressive_peak(mesh, laminate, cfg) -> dict:
    """Ply-discount progressive peak, as ``_run_progressive_path`` runs it."""
    from wrinklefe.solver.progressive_damage import ProgressiveDamageSolver

    mat0 = laminate.plies[0].material
    target = (float(cfg.progressive_max_strain)
              if cfg.progressive_max_strain is not None
              else 1.8 * mat0.Xc / mat0.E1)
    sign = -1.0 if cfg.applied_strain <= 0 else 1.0
    t0 = time.time()
    res = ProgressiveDamageSolver(
        mesh, laminate, applied_strain=sign * abs(target),
        n_increments=cfg.progressive_n_increments,
        residual_factor=cfg.progressive_residual_factor,
        solver=cfg.solver, delta_T=cfg.delta_T,
    ).solve()
    return {"peak_MPa": float(res.peak_stress),
            "n_failed": int(res.n_failed_elements),
            "converged": bool(res.converged),
            "history": [[float(e), float(s)] for e, s in res.history],
            "seconds": round(time.time() - t0, 1)}


def solve_first_and_prog(c: Case, cache: Cache, *, prog: bool) -> None:
    """First-ply baseline (cheap) and, when asked, the progressive peak."""
    from wrinklefe.analysis import WrinkleAnalysis

    k_first = f"first|{c.dataset}|{c.geometry}"
    k_prog = f"prog|{c.dataset}|{c.geometry}"
    k_prist = f"pristine|{c.pristine}"
    need_first = cache.get(k_first) is None
    need_prog = prog and cache.get(k_prog) is None
    need_prist = prog and not c.absolute and cache.get(k_prist) is None
    if not (need_first or need_prog or need_prist):
        return
    t0 = time.time()
    r = ses._fe_run(c.cfg, **c.fe)
    if need_first:
        kd, onset = ses._first_ply_from_results(r)
        cache.put(k_first, {"kd": kd, "onset_MPa": onset,
                            "seconds": round(time.time() - t0, 1)})
    if need_prog:
        mesh = r.mesh
        saved = mesh.element_material_override
        cache.put(k_prog, _progressive_peak(mesh, r.laminate, r.config))
        mesh.element_material_override = saved
    if need_prist:
        flat = WrinkleAnalysis(r.config)._build_flat_mesh(r.laminate)
        cache.put(k_prist, _progressive_peak(flat, r.laminate, r.config))


def _czm_run(c: Case, **czm):
    sign = -1.0 if c.cfg.applied_strain < 0 else 1.0
    strain = czm.pop("strain")
    fe = {**c.fe, **czm.pop("fe", {})}
    return ses._fe_run(c.cfg, **fe, enable_czm=True,
                       applied_strain=sign * abs(strain), **czm)


def _init_key(c: Case) -> str:
    over = INIT_MESH_OVERRIDE.get(c.dataset, {})
    tag = "".join(f"|{k}{v}" for k, v in sorted(over.items()))
    return f"init|{c.dataset}|{c.geometry}{tag}"


def solve_init(c: Case, cache: Cache) -> None:
    """(b): exact initiation load from one elastic CZM increment."""
    key = _init_key(c)
    if cache.get(key) is not None:
        return
    t0 = time.time()
    r = _czm_run(c, strain=_INIT_STRAIN, czm_interfaces="all",
                 czm_n_load_increments=1,
                 fe=INIT_MESH_OVERRIDE.get(c.dataset, {}))
    h = r.czm_damage_history
    on = h.initiation()
    cache.put(key, {
        "sigma_MPa": None if on is None else on.nominal_stress,
        "strain": None if on is None else on.applied_strain,
        "exact": None if on is None else on.exact,
        "elastic": bool(on is not None and on.extrapolated),
        "n_interfaces": len(r.czm_interfaces_used or []),
        "seconds": round(time.time() - t0, 1),
    })


def solve_flat_init(c: Case, cache: Cache) -> None:
    """Diagnostic: (b) on a near-flat coupon (A = 1 um) of the dataset.

    Interlaminar stresses are singular where dissimilar plies meet a free
    edge or the clamped loaded ends, so a multidirectional coupon reaches
    interface "initiation" with no wrinkle at all. When that load is below
    the wrinkled one, (b) measures the coupon, not the wrinkle.
    """
    key = f"flatinit|{c.dataset}"
    if cache.get(key) is not None:
        return
    flat = dataclasses.replace(c, cfg=dataclasses.replace(c.cfg,
                                                          amplitude=1.0e-3))
    r = _czm_run(flat, strain=_INIT_STRAIN, czm_interfaces="all",
                 czm_n_load_increments=1,
                 fe=INIT_MESH_OVERRIDE.get(c.dataset, {}))
    on = r.czm_damage_history.initiation()
    cache.put(key, {"sigma_MPa": None if on is None else on.nominal_stress,
                    "case": c.case})


def solve_sep(c: Case, cache: Cache) -> None:
    """Separation (sensitivity): nonlinear ramp at the near-crest interface."""
    key = f"sep|{c.dataset}|{c.geometry}"
    init = cache.get(_init_key(c))
    if cache.get(key) is not None or not init or init["strain"] is None:
        return
    t0 = time.time()
    r = _czm_run(c, strain=SEP_RAMP * abs(init["strain"]),
                 czm_interfaces="near_crest",
                 czm_n_load_increments=SEP_INCREMENTS,
                 fe=INIT_MESH_OVERRIDE.get(c.dataset, {}))
    h = r.czm_damage_history
    ini = h.initiation()
    sep = h.separation()
    cache.put(key, {
        "sigma_MPa": None if sep is None else sep.nominal_stress,
        "increment": None if sep is None else sep.increment,
        "init_near_crest_MPa": None if ini is None else ini.nominal_stress,
        "peak_MPa": float(h.nominal_stress.max()) if h.n_recorded else None,
        "last_MPa": float(h.nominal_stress[-1]) if h.n_recorded else None,
        "recorded": h.n_recorded,
        "completed": h.completed,
        "max_damage": float(h.max_damage[-1]) if h.n_recorded else None,
        "separated_fraction": (float(h.separated_fraction[-1])
                               if h.n_recorded else None),
        "history": h.to_dict(),
        "seconds": round(time.time() - t0, 1),
    })


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------

def predictions(c: Case, cache: Cache) -> dict[str, float | None]:
    """Each measure's prediction for one case (knockdown, or MPa for M)."""
    first = cache.get(f"first|{c.dataset}|{c.geometry}")
    prog = cache.get(f"prog|{c.dataset}|{c.geometry}")
    init = cache.get(_init_key(c))
    sep = cache.get(f"sep|{c.dataset}|{c.geometry}")
    prist = cache.get(f"pristine|{c.pristine}")
    out: dict[str, float | None] = dict.fromkeys(
        ("first-ply", "(a) prog. peak", "(b) CZM init.", "(c) min(a,b)",
         "separation"))
    if first:
        out["first-ply"] = first["onset_MPa"] if c.absolute else first["kd"]
    s_a = prog["peak_MPa"] if prog else None
    s_b = init["sigma_MPa"] if init else None
    s_s = sep["sigma_MPa"] if sep else None
    if c.absolute:
        out["(a) prog. peak"] = s_a
        out["(b) CZM init."] = s_b
        out["separation"] = s_s
    elif prist:
        sp = prist["peak_MPa"]
        out["(a) prog. peak"] = None if s_a is None else s_a / sp
        out["(b) CZM init."] = None if s_b is None else min(s_b, sp) / sp
        out["separation"] = None if s_s is None else min(s_s, sp) / sp
    a, b = out["(a) prog. peak"], out["(b) CZM init."]
    if a is not None and b is not None:
        out["(c) min(a,b)"] = min(a, b)
    return out


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    from scipy.stats import spearmanr

    rho = float(spearmanr(x, y).statistic)
    return None if math.isnan(rho) else rho


def score(all_cases: list[Case], cache: Cache) -> list[dict]:
    """One row per (dataset, measure) with n, mean, worst, unsafe count."""
    rows = []
    measures = ("first-ply", "(a) prog. peak", "(b) CZM init.",
                "(c) min(a,b)", "separation")
    for ds in dict.fromkeys(c.dataset for c in all_cases):
        dcases = [c for c in all_cases if c.dataset == ds]
        for m in measures:
            pts = [(c, predictions(c, cache)[m]) for c in dcases]
            pts = [(c, p) for c, p in pts if p is not None]
            if not pts:
                continue
            errs = [ses._err(p, c.measured) for c, p in pts]
            worst_c, worst_e = max(zip([c.case for c, _ in pts], errs),
                                   key=lambda t: t[1])
            rho_p = _spearman([c.size for c, _ in pts], [p for _, p in pts])
            rho_m = _spearman([c.size for c, _ in pts],
                              [c.measured for c, _ in pts])
            rows.append({
                "dataset": ds, "measure": m, "n": len(pts),
                "of": len(dcases),
                "mean": statistics.mean(errs),
                "mae": statistics.mean(abs(e) for e in errs),
                "worst": worst_e, "worst_case": worst_c,
                "unsafe": sum(e > ses._NON_CONSERVATIVE_PCT for e in errs),
                "rho_pred": rho_p, "rho_meas": rho_m,
            })
    return rows


def fixed_angle_series(cache: Cache) -> list[tuple]:
    """Dataset E at ~16 deg: (case, D/T, measured, {measure: pred})."""
    lo, hi = _E_FIXED_ANGLE
    out = []
    for c in cases("E"):
        if lo <= c.theta <= hi:
            out.append((c.case, c.size, c.measured, predictions(c, cache)))
    return sorted(out, key=lambda t: t[1])


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

def _fmt(v, spec=".3f") -> str:
    return "-" if v is None else format(v, spec)


def report(all_cases: list[Case], cache: Cache) -> None:
    print("Per case (knockdown; Dataset M onset in MPa)")
    hdr = ("first-ply", "(a) prog. peak", "(b) CZM init.", "(c) min(a,b)",
           "separation")
    print(f"  {'dataset':5s} {'case':16s} {'size':>6s} {'theta':>5s} "
          f"{'meas.':>7s} " + " ".join(f"{h:>14s}" for h in hdr))
    for c in all_cases:
        p = predictions(c, cache)
        spec = ".1f" if c.absolute else ".3f"
        print(f"  {c.dataset:5s} {c.case:16s} {c.size:6.3f} {c.theta:5.1f} "
              f"{format(c.measured, spec):>7s} "
              + " ".join(f"{_fmt(p[h], spec):>14s}" for h in hdr))
    print()
    print("Scores (signed error %; unsafe = above +0.5 %; rho = Spearman "
          "rank correlation with size)")
    print(f"  {'dataset':5s} {'measure':15s} {'n':>5s} {'mean%':>7s} "
          f"{'MAE%':>6s} {'worst%':>7s} {'worst case':>16s} {'unsafe':>7s} "
          f"{'rho pred':>8s} {'rho meas':>8s}")
    for r in score(all_cases, cache):
        print(f"  {r['dataset']:5s} {r['measure']:15s} "
              f"{r['n']:2d}/{r['of']:<2d} {r['mean']:+7.1f} {r['mae']:6.1f} "
              f"{r['worst']:+7.1f} {r['worst_case']:>16s} "
              f"{r['unsafe']:3d}/{r['n']:<3d} {_fmt(r['rho_pred'], '+.2f'):>8s} "
              f"{_fmt(r['rho_meas'], '+.2f'):>8s}")
    print()
    series = fixed_angle_series(cache)
    if series:
        print(f"Dataset E fixed-angle size series "
              f"({_E_FIXED_ANGLE[0]}-{_E_FIXED_ANGLE[1]} deg): does the "
              f"prediction fall with D/T as the measurement does?")
        print(f"  {'case':5s} {'D/T':>6s} {'meas.':>6s} "
              + " ".join(f"{h:>14s}" for h in hdr))
        for name, dt, meas, p in series:
            print(f"  {name:5s} {dt:6.3f} {meas:6.3f} "
                  + " ".join(f"{_fmt(p[h]):>14s}" for h in hdr))
        meas_spread = max(m for _n, _d, m, _p in series) - min(
            m for _n, _d, m, _p in series)
        for h in hdr:
            vals = [p[h] for _n, _d, _m, p in series if p[h] is not None]
            if len(vals) == len(series):
                rho = _spearman([d for _n, d, _m, _p in series], vals)
                print(f"  {h:15s} spread {max(vals) - min(vals):.3f} "
                      f"(measured {meas_spread:.3f}), rho {_fmt(rho, '+.2f')}")
    print()
    rows = score(all_cases, cache)
    for m in ("(a) prog. peak", "(b) CZM init.", "(c) min(a,b)"):
        mrows = [r for r in rows if r["measure"] == m]
        complete = all(r["n"] == r["of"] for r in mrows) and len(mrows) == len(
            {c.dataset for c in all_cases})
        worst = max((r["worst"] for r in mrows), default=None)
        verdict = ("meets" if worst is not None and worst <= UNSAFE_BAR_PCT
                   else "fails")
        print(f"  {m}: worst {_fmt(worst, '+.1f')} % -> {verdict} the "
              f"+{UNSAFE_BAR_PCT:.0f} % bar"
              + ("" if complete else " (INCOMPLETE: not every case solved)"))


def ledger_section(all_cases: list[Case], cache: Cache) -> dict:
    """The ``size_aware_fe`` block of ``tests/test_validation/ledger.json``.

    Per case: the measurement and each measure's prediction exactly as the
    report scores them (knockdown, or MPa for Dataset M), plus the raw CZM
    initiation stress the slow test re-solves. Per dataset: the pristine
    progressive peak the knockdowns are normalised by.
    """
    out: dict = {}
    for ds in dict.fromkeys(c.dataset for c in all_cases):
        dcases = [c for c in all_cases if c.dataset == ds]
        prist = cache.get(f"pristine|{dcases[0].pristine}")
        flat = cache.get(f"flatinit|{ds}") or {}
        rows = []
        for c in dcases:
            p = predictions(c, cache)
            init = cache.get(_init_key(c)) or {}
            prog = cache.get(f"prog|{c.dataset}|{c.geometry}") or {}
            rows.append({
                "case": c.case,
                "size": round(c.size, 4),
                "theta_deg": round(c.theta, 2),
                "measured": c.measured,
                "first_ply": _round(p["first-ply"]),
                "progressive_peak": _round(p["(a) prog. peak"]),
                "progressive_peak_MPa": _round(prog.get("peak_MPa")),
                "progressive_converged": prog.get("converged"),
                "czm_initiation": _round(p["(b) CZM init."]),
                "czm_initiation_MPa": _round(init.get("sigma_MPa")),
            })
        out[ds] = {
            "absolute_MPa": dcases[0].absolute,
            "fe_mesh": {k: v for k, v in dcases[0].fe.items()
                        if k in ("nx", "ny", "nz_per_ply")},
            "czm_initiation_mesh_override": INIT_MESH_OVERRIDE.get(ds, {}),
            "pristine_progressive_MPa": (None if dcases[0].absolute or not prist
                                         else _round(prist["peak_MPa"])),
            "pristine_progressive_converged": (
                None if dcases[0].absolute or not prist
                else prist["converged"]),
            "near_flat_czm_initiation_MPa": _round(flat.get("sigma_MPa")),
            "cases": rows,
        }
    return out


def _round(v):
    return None if v is None else round(float(v), 4)


# ----------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--measures", default="first,init,prog")
    ap.add_argument("--cases", default="",
                    help="comma-separated case names to restrict the solves")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--ledger", action="store_true",
                    help="print the ledger's size_aware_fe datasets block "
                         "(from the cache) as JSON and exit")
    args = ap.parse_args(argv)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    measures = {m.strip() for m in args.measures.split(",") if m.strip()}
    unknown = measures - set(MEASURES)
    if unknown:
        ap.error(f"unknown measures {sorted(unknown)}")
    only = {c.strip() for c in args.cases.split(",") if c.strip()}
    cache = Cache(args.cache)
    all_cases = [c for d in datasets for c in cases(d)]
    if args.ledger:
        print(json.dumps(ledger_section(all_cases, cache), indent=1))
        return 0

    if not args.report_only:
        done: set[tuple[str, str]] = set()
        for c in all_cases:
            if only and c.case not in only:
                continue
            if (c.dataset, c.geometry) in done:
                continue
            done.add((c.dataset, c.geometry))
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if "first" in measures or "prog" in measures:
                    solve_first_and_prog(c, cache, prog="prog" in measures)
                if "init" in measures or "sep" in measures:
                    solve_init(c, cache)
                if "init" in measures:
                    solve_flat_init(c, cache)
                if "sep" in measures:
                    solve_sep(c, cache)
            print(f"  solved {c.dataset} {c.case} [{time.time() - t0:.0f}s]",
                  flush=True)
    report(all_cases, cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
