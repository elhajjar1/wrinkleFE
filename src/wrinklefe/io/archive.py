"""Lossless archive of an :class:`AnalysisResults` — save it, reload it later.

An FE or CZM run holds the expensive part of the computation: per-Gauss-point
stress and strain, failure indices and modes, cohesive damage, separation and
traction. None of it survived the Python session (issue #277). A CZM solve that
took an hour could not be reopened to render a different slice; a 75-point
sweep produced 75 result objects of which only scalar summaries could be kept;
sharing a result for someone else's post-processing meant sharing your
terminal.

This is the archive tier *underneath* the report exports. It is deliberately a
different thing from what :mod:`wrinklefe.io.results` and
:mod:`wrinklefe.io.export` produce:

===========================  ==========================================
:func:`save_results` here     the whole result, losslessly, for reloading
``export_results_json``       a few-KB document for a report, arrays
                              reduced to ``{min, max, mean, p95, n}``
===========================  ==========================================

Format
------
One compressed ``.npz``. Every array is an entry keyed by its dotted path in
the result graph (``field_results.stress_local``,
``failure_report.ply_failure_indices.larc05``); everything else — scalars,
strings, the config, the provenance block, and the structure that says how to
put it back together — is a JSON manifest stored as a ``uint8`` array under
``__manifest__``.

**No pickle.** Arrays are written as plain numeric or unicode dtypes and read
back with ``allow_pickle=False``, so an archive cannot execute anything on
open and does not depend on this package's class layout to be readable. That
also makes it inspectable with nothing but ``numpy`` and ``json``.

Compatibility
-------------
:data:`ARCHIVE_FORMAT_VERSION` is written into the manifest and checked on
load. A newer major version fails loudly rather than guessing: a silently
half-read result is worse than a refusal, because the numbers would look
plausible.

Examples
--------
>>> from wrinklefe.io.archive import save_results, load_results
>>> save_results(results, "run.wfr")                  # doctest: +SKIP
>>> restored = load_results("run.wfr")                # doctest: +SKIP
>>> plot_stress_field(restored.field_results)         # doctest: +SKIP
"""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from wrinklefe.analysis import AnalysisConfig, AnalysisResults
from wrinklefe.core.laminate import Laminate, Ply
from wrinklefe.core.material import OrthotropicMaterial
from wrinklefe.core.mesh import MeshData
from wrinklefe.core.morphology import WrinkleConfiguration, WrinklePlacement
from wrinklefe.failure.evaluator import LaminateFailureReport
from wrinklefe.io.export import build_provenance
from wrinklefe.solver.results import FieldResults

__all__ = [
    "ARCHIVE_FORMAT_VERSION",
    "ArchiveFormatError",
    "DEFAULT_SUFFIX",
    "load_results",
    "save_results",
]

#: Archive schema version, ``"major.minor"``. Bump the **minor** part for a
#: change a reader of this version can still load (a new optional field);
#: bump the **major** part when it cannot, because :func:`load_results`
#: refuses a major it does not know.
ARCHIVE_FORMAT_VERSION = "1.0"

#: Conventional suffix — "WrinkleFE result". Nothing enforces it; the file is
#: a ``.npz`` whatever it is called.
DEFAULT_SUFFIX = ".wfr"

_MANIFEST_KEY = "__manifest__"

# Manifest node tags. Explicit tags rather than inferring from the JSON shape:
# a dict of floats and a serialised object are both JSON objects, and guessing
# between them is how a loader silently returns the wrong type.
_T_ARRAY = "array"
_T_DATACLASS = "dataclass"
_T_DICT = "dict"
_T_LIST = "list"
_T_VALUE = "value"
_T_REF = "ref"
_T_CONFIG = "config"
_T_MATERIAL = "material"
_T_LAMINATE = "laminate"
_T_PLY = "ply"
_T_WRINKLE_CONFIG = "wrinkle_config"
_T_WRINKLE_PLACEMENT = "wrinkle_placement"
_T_WRINKLE_PROFILE = "wrinkle_profile"

#: Dataclasses walked field by field. Each must be reconstructible by keyword
#: from its own fields; anything needing more than that gets an explicit
#: handler instead (see ``Laminate``, whose A/B/D are derived).
_WALKED_DATACLASSES: dict[str, type] = {
    "MeshData": MeshData,
    "FieldResults": FieldResults,
    "LaminateFailureReport": LaminateFailureReport,
}


class ArchiveFormatError(ValueError):
    """An archive could not be read: wrong format, or a version too new."""


# ----------------------------------------------------------------------
# Wrinkle profiles
# ----------------------------------------------------------------------

def _profile_classes() -> dict[str, type]:
    """Name -> class for every concrete wrinkle profile.

    Built by walking the ``WrinkleProfile`` hierarchy rather than hard-coding
    a list, so a profile added later archives without touching this module.
    ``WrinkleSurface3D`` is included explicitly: a placement may hold one and
    it is not a ``WrinkleProfile`` subclass.
    """
    from wrinklefe.core.wrinkle import WrinkleProfile, WrinkleSurface3D

    out: dict[str, type] = {"WrinkleSurface3D": WrinkleSurface3D}

    def _walk(cls: type) -> None:
        # Annotated because mypy >= 2.4 cannot infer it from
        # ``type.__subclasses__()`` on a bare ``type``.
        sub: type
        for sub in cls.__subclasses__():
            out[sub.__name__] = sub
            _walk(sub)

    _walk(WrinkleProfile)
    return out


_PROFILE_PARAMS = ("amplitude", "wavelength", "width", "center")


# ----------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------

class _Encoder:
    """Walk a result graph into (JSON manifest, {array key: ndarray}).

    Object identity is preserved: ``results.mesh``, ``mesh.laminate`` and
    ``field_results.mesh`` are the same objects at runtime, so the first
    encounter is encoded and later ones become a ``ref`` to its path. That
    keeps the archive from carrying three copies of the mesh *and* keeps the
    reloaded graph wired the way viz code finds it.
    """

    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self._seen: dict[int, str] = {}

    def encode(self, value: Any, path: str) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return {"t": _T_VALUE, "v": value}

        if isinstance(value, np.ndarray):
            return self._array(value, path)

        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return {"t": _T_VALUE, "v": value.item()}

        # Everything below is an object that may be shared.
        key = id(value)
        if key in self._seen:
            return {"t": _T_REF, "path": self._seen[key]}

        if isinstance(value, dict):
            self._seen[key] = path
            return self._dict(value, path)
        if isinstance(value, (list, tuple)):
            self._seen[key] = path
            return {
                "t": _T_LIST,
                "tuple": isinstance(value, tuple),
                "items": [
                    self.encode(v, f"{path}.{i}") for i, v in enumerate(value)
                ],
            }

        self._seen[key] = path

        if isinstance(value, AnalysisConfig):
            return {"t": _T_CONFIG, "data": value.to_dict()}
        if isinstance(value, OrthotropicMaterial):
            return {"t": _T_MATERIAL, "data": value.to_dict()}
        if isinstance(value, Ply):
            return {
                "t": _T_PLY,
                "material": self.encode(value.material, f"{path}.material"),
                "angle": float(value.angle),
                "thickness": float(value.thickness),
            }
        if isinstance(value, Laminate):
            # Only the plies: A/B/D are derived in ``__init__`` and storing
            # them would let an archive disagree with its own plies.
            return {
                "t": _T_LAMINATE,
                "plies": [
                    self.encode(p, f"{path}.plies.{i}")
                    for i, p in enumerate(value.plies)
                ],
            }
        if isinstance(value, WrinklePlacement):
            return {
                "t": _T_WRINKLE_PLACEMENT,
                "profile": self.encode(value.profile, f"{path}.profile"),
                "ply_interface": int(value.ply_interface),
                "phase_offset": float(value.phase_offset),
            }
        if isinstance(value, WrinkleConfiguration):
            return self._wrinkle_config(value, path)

        profiles = _profile_classes()
        if type(value).__name__ in profiles:
            return {
                "t": _T_WRINKLE_PROFILE,
                "cls": type(value).__name__,
                "params": {
                    name: float(getattr(value, name))
                    for name in _PROFILE_PARAMS
                    if hasattr(value, name)
                },
            }

        name = type(value).__name__
        if name in _WALKED_DATACLASSES and is_dataclass(value):
            return {
                "t": _T_DATACLASS,
                "cls": name,
                "fields": {
                    f.name: self.encode(
                        getattr(value, f.name), f"{path}.{f.name}"
                    )
                    for f in fields(value)
                },
            }

        raise ArchiveFormatError(
            f"cannot archive {name!r} at {path!r}: no handler. Add one to "
            f"wrinklefe.io.archive rather than falling back to pickle — the "
            f"format guarantees an archive contains no executable objects."
        )

    def _wrinkle_config(self, value: WrinkleConfiguration, path: str) -> dict:
        # Reconstructed by keyword, so capture exactly the constructor's
        # parameters plus any extra public attribute the engine sets on it
        # (``wrinkle_z_position`` is set after construction).
        import inspect

        params = [
            p for p in inspect.signature(WrinkleConfiguration.__init__)
            .parameters if p != "self"
        ]
        ctor = {
            name: self.encode(getattr(value, name), f"{path}.{name}")
            for name in params
            if hasattr(value, name)
        }
        extra = {
            name: self.encode(getattr(value, name), f"{path}.{name}")
            for name in sorted(vars(value))
            if name not in params and not name.startswith("_")
        }
        return {"t": _T_WRINKLE_CONFIG, "ctor": ctor, "extra": extra}

    def _dict(self, value: dict, path: str) -> dict:
        # Key types survive the round trip: JSON object keys are always
        # strings, but ``czm_energy_per_interface`` is keyed by interface
        # *index*, and handing it back with "0" instead of 0 would break
        # every lookup silently.
        entries = []
        for k, v in value.items():
            if isinstance(k, str):
                kind, ks = "str", k
            elif isinstance(k, bool):
                kind, ks = "bool", str(k)
            elif isinstance(k, (int, np.integer)):
                kind, ks = "int", str(int(k))
            elif isinstance(k, (float, np.floating)):
                kind, ks = "float", repr(float(k))
            else:
                raise ArchiveFormatError(
                    f"dict key {k!r} at {path!r} is a {type(k).__name__}; "
                    f"only str/int/float/bool keys can be archived."
                )
            entries.append(
                {"k": ks, "kt": kind, "v": self.encode(v, f"{path}.{ks}")}
            )
        return {"t": _T_DICT, "entries": entries}

    def _array(self, value: np.ndarray, path: str) -> dict:
        if value.dtype == object:
            raise ArchiveFormatError(
                f"array at {path!r} has dtype=object, which npz can only "
                f"store by pickling. Archives are pickle-free by design."
            )
        key = f"a:{path}"
        self.arrays[key] = value
        return {"t": _T_ARRAY, "key": key}


# ----------------------------------------------------------------------
# Decoding
# ----------------------------------------------------------------------

class _Decoder:
    def __init__(self, npz: Any) -> None:
        self._npz = npz
        self._by_path: dict[str, Any] = {}
        self._pending: dict[str, dict] = {}

    def register(self, node: Any, path: str) -> None:
        """Index every node by path so ``ref`` nodes can be resolved."""
        if not isinstance(node, dict) or "t" not in node:
            return
        self._pending[path] = node

    def decode(self, node: Any, path: str) -> Any:
        if path in self._by_path:
            return self._by_path[path]

        tag = node.get("t")
        if tag == _T_VALUE:
            return node["v"]
        if tag == _T_ARRAY:
            return self._npz[node["key"]]
        if tag == _T_REF:
            target = node["path"]
            if target in self._by_path:
                return self._by_path[target]
            if target not in self._pending:
                raise ArchiveFormatError(
                    f"archive references {target!r} from {path!r}, but that "
                    f"path is not in the manifest."
                )
            return self.decode(self._pending[target], target)

        out: Any
        if tag == _T_DICT:
            out = {}
            self._by_path[path] = out
            for entry in node["entries"]:
                out[self._key(entry)] = self.decode(
                    entry["v"], f"{path}.{entry['k']}"
                )
        elif tag == _T_LIST:
            items = [
                self.decode(item, f"{path}.{i}")
                for i, item in enumerate(node["items"])
            ]
            out = tuple(items) if node.get("tuple") else items
            self._by_path[path] = out
        elif tag == _T_CONFIG:
            out = AnalysisConfig.from_dict(node["data"])
        elif tag == _T_MATERIAL:
            out = OrthotropicMaterial.from_dict(node["data"])
        elif tag == _T_PLY:
            out = Ply(
                material=self.decode(node["material"], f"{path}.material"),
                angle=node["angle"],
                thickness=node["thickness"],
            )
        elif tag == _T_LAMINATE:
            out = Laminate([
                self.decode(p, f"{path}.plies.{i}")
                for i, p in enumerate(node["plies"])
            ])
        elif tag == _T_WRINKLE_PROFILE:
            out = _profile_classes()[node["cls"]](**node["params"])
        elif tag == _T_WRINKLE_PLACEMENT:
            out = WrinklePlacement(
                profile=self.decode(node["profile"], f"{path}.profile"),
                ply_interface=node["ply_interface"],
                phase_offset=node["phase_offset"],
            )
        elif tag == _T_WRINKLE_CONFIG:
            ctor = {
                k: self.decode(v, f"{path}.{k}")
                for k, v in node["ctor"].items()
            }
            out = WrinkleConfiguration(**ctor)
            for k, v in node["extra"].items():
                setattr(out, k, self.decode(v, f"{path}.{k}"))
        elif tag == _T_DATACLASS:
            cls = _WALKED_DATACLASSES.get(node["cls"])
            if cls is None:
                raise ArchiveFormatError(
                    f"archive holds a {node['cls']!r} at {path!r}, which "
                    f"this version does not know how to rebuild."
                )
            kwargs = {
                name: self.decode(sub, f"{path}.{name}")
                for name, sub in node["fields"].items()
            }
            out = cls(**kwargs)
        else:
            raise ArchiveFormatError(
                f"unknown manifest node type {tag!r} at {path!r}."
            )

        self._by_path[path] = out
        return out

    @staticmethod
    def _key(entry: dict) -> Any:
        kind, raw = entry["kt"], entry["k"]
        if kind == "str":
            return raw
        if kind == "int":
            return int(raw)
        if kind == "float":
            return float(raw)
        if kind == "bool":
            return raw == "True"
        raise ArchiveFormatError(f"unknown dict key type {kind!r}")


def _index_manifest(node: Any, path: str, decoder: _Decoder) -> None:
    """Pre-register every node so forward ``ref`` nodes resolve."""
    if not isinstance(node, dict):
        return
    tag = node.get("t")
    if tag is None:
        return
    decoder.register(node, path)
    if tag == _T_DICT:
        for entry in node["entries"]:
            _index_manifest(entry["v"], f"{path}.{entry['k']}", decoder)
    elif tag == _T_LIST:
        for i, item in enumerate(node["items"]):
            _index_manifest(item, f"{path}.{i}", decoder)
    elif tag == _T_DATACLASS:
        for name, sub in node["fields"].items():
            _index_manifest(sub, f"{path}.{name}", decoder)
    elif tag == _T_LAMINATE:
        for i, p in enumerate(node["plies"]):
            _index_manifest(p, f"{path}.plies.{i}", decoder)
    elif tag == _T_PLY:
        _index_manifest(node["material"], f"{path}.material", decoder)
    elif tag == _T_WRINKLE_PLACEMENT:
        _index_manifest(node["profile"], f"{path}.profile", decoder)
    elif tag == _T_WRINKLE_CONFIG:
        for group in ("ctor", "extra"):
            for name, sub in node[group].items():
                _index_manifest(sub, f"{path}.{name}", decoder)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def save_results(results: AnalysisResults, path: str | Path) -> Path:
    """Archive a complete :class:`AnalysisResults` losslessly.

    Parameters
    ----------
    results : AnalysisResults
        The result to archive. Analytical-only results archive too — they
        simply carry fewer arrays.
    path : str or Path
        Destination. Parent directories are created. A ``.wfr`` suffix is
        conventional (see :data:`DEFAULT_SUFFIX`) but not required.

    Returns
    -------
    Path
        The file written.

    Raises
    ------
    ArchiveFormatError
        If the graph holds something this module has no handler for. That is
        deliberate: the alternative is pickling it, and the format's promise
        is that an archive contains nothing executable.

    See Also
    --------
    load_results : the inverse.
    wrinklefe.io.results.export_results_json : the report-tier export, which
        summarises large arrays instead of storing them.
    """
    encoder = _Encoder()
    tree = {
        name: encoder.encode(getattr(results, name), name)
        for name in (f.name for f in fields(results))
    }
    manifest = {
        "format_version": ARCHIVE_FORMAT_VERSION,
        "provenance": build_provenance(),
        "results": tree,
    }
    blob = json.dumps(manifest, sort_keys=True).encode("utf-8")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        _MANIFEST_KEY: np.frombuffer(blob, dtype=np.uint8)
    }
    payload.update(encoder.arrays)
    # Written through an open handle, not a path: given a path,
    # ``np.savez_compressed`` appends ``.npz`` unless the name already ends
    # that way, so ``save_results(r, "run.wfr")`` would silently produce
    # ``run.wfr.npz`` and ``load_results("run.wfr")`` would then fail on a
    # file the caller believes it just wrote. A handle writes exactly where
    # asked.
    #
    # ``allow_pickle=False`` makes the format's promise numpy's to enforce,
    # at write time: an object-dtype array raises here instead of being
    # quietly pickled in. The encoder rejects object dtype too, so this is
    # the second of two independent guards.
    #
    # The ignore is a numpy-stub artefact: ``savez_compressed`` is typed
    # ``(file, *args, allow_pickle=True, **kwds)``, so a ``**dict[str,
    # ndarray]`` expansion is matched against ``allow_pickle: bool``.
    with open(out, "wb") as handle:
        np.savez_compressed(  # type: ignore[arg-type]
            handle, allow_pickle=False, **payload
        )
    return out


def load_results(path: str | Path) -> AnalysisResults:
    """Rebuild an :class:`AnalysisResults` from an archive.

    The returned object is the real thing, not a view: every ``viz``
    function, exporter and failure-report accessor takes it unchanged, and
    the internal wiring (``field_results.mesh is results.mesh``) is restored
    rather than duplicated.

    Parameters
    ----------
    path : str or Path
        An archive written by :func:`save_results`.

    Returns
    -------
    AnalysisResults

    Raises
    ------
    ArchiveFormatError
        If the file is not an archive, or its major format version is newer
        than this version understands. Refusing beats half-reading: the
        numbers from a partially understood archive would look plausible.
    """
    src = Path(path)
    with np.load(src, allow_pickle=False) as npz:
        if _MANIFEST_KEY not in npz:
            raise ArchiveFormatError(
                f"{src}: not a WrinkleFE result archive (no "
                f"{_MANIFEST_KEY!r} entry). An export from "
                f"wrinklefe.io.results is a JSON report, not an archive."
            )
        manifest = json.loads(bytes(npz[_MANIFEST_KEY]).decode("utf-8"))
        _check_version(manifest.get("format_version"), src)

        decoder = _Decoder(npz)
        tree = manifest["results"]
        for name, node in tree.items():
            _index_manifest(node, name, decoder)
        kwargs = {
            name: decoder.decode(node, name) for name, node in tree.items()
        }

    known = {f.name for f in fields(AnalysisResults)}
    unknown = sorted(set(kwargs) - known)
    if unknown:
        raise ArchiveFormatError(
            f"{src}: archive carries fields this version of AnalysisResults "
            f"does not have: {unknown}. It was probably written by a newer "
            f"WrinkleFE."
        )
    try:
        return AnalysisResults(**kwargs)
    except TypeError as exc:
        # A field this version requires is absent — an archive from an older
        # WrinkleFE, written before the field existed. Say so, rather than
        # letting a bare ``TypeError`` about a missing keyword surface from
        # inside a loader the caller only asked to open a file.
        missing = sorted(
            f.name for f in fields(AnalysisResults)
            if f.name not in kwargs
        )
        raise ArchiveFormatError(
            f"{src}: archive is missing {len(missing)} field(s) this version "
            f"of AnalysisResults requires: {missing}. It was probably "
            f"written by an older WrinkleFE."
        ) from exc


def _check_version(version: Any, src: Path) -> None:
    if not isinstance(version, str):
        raise ArchiveFormatError(
            f"{src}: archive manifest has no format_version."
        )
    try:
        major = int(version.split(".")[0])
    except (ValueError, IndexError) as exc:
        raise ArchiveFormatError(
            f"{src}: unreadable format_version {version!r}."
        ) from exc
    ours = int(ARCHIVE_FORMAT_VERSION.split(".")[0])
    if major > ours:
        raise ArchiveFormatError(
            f"{src}: archive format {version} is newer than this version "
            f"understands ({ARCHIVE_FORMAT_VERSION}). Upgrade WrinkleFE to "
            f"read it — loading it here would risk a plausible-looking "
            f"partial result."
        )
