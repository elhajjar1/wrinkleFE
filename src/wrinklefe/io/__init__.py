"""Import/export: project files, Abaqus, VTK, material database.

Public API
----------
.. autofunction:: export_results_json
.. autofunction:: export_abaqus_inp
.. autofunction:: export_vtk
.. autofunction:: build_analysis_summary
.. autofunction:: recommend_disposition
.. autofunction:: render_summary_markdown
.. autofunction:: render_summary_pdf
.. autofunction:: export_summary
.. autofunction:: export_results_csv
.. autofunction:: results_to_dict

Two exporters, one name
-----------------------

The legacy ``export_results_json`` (in :mod:`wrinklefe.io.export`) and
the schema-versioned one in :mod:`wrinklefe.io.results` coexist
deliberately. The legacy entry remains what ``wrinklefe.io`` re-exports,
so existing callers do not break; consumers who want the structured
document (``per_ply`` table, ``first_ply_failure``,
``knockdown_factors``, ``schema_version``) import it directly from
:mod:`wrinklefe.io.results`.

Be aware of what that costs. The two documents are almost entirely
disjoint: on an FE run the legacy one emits 33 leaf paths and the
structured one 138, sharing only ``provenance`` and the three ``mesh``
counts, and on an analytical-only run nothing outside ``provenance``.
No result, prediction or configuration value is reachable by the same
path in both -- even the common blocks were renamed (``configuration``
against ``config``, ``analytical_predictions`` against
``analytical``). So these two imports::

    from wrinklefe.io import export_results_json           # legacy
    from wrinklefe.io.results import export_results_json    # structured

write files with nothing in common, and a consumer written against one
reads nothing from the other. They are told apart by their top-level
keys: the structured document has ``schema_version``, the legacy one
has ``wrinklefe_version``.

Both carry a ``provenance`` block built by
:func:`wrinklefe.io.export.build_provenance`, so whichever one a user
exports can support a reproducibility claim. (The structured document
gained it in schema 1.2; before that only the legacy one had it.)
"""

from wrinklefe.io.export import (
    build_analysis_summary,
    export_abaqus_inp,
    export_results_json,
    export_summary,
    export_vtk,
    recommend_disposition,
    render_summary_markdown,
    render_summary_pdf,
)
from wrinklefe.io.results import (
    SCHEMA_VERSION,
    export_results_csv,
    results_to_dict,
)

__all__ = [
    "export_results_json",
    "export_results_csv",
    "results_to_dict",
    "SCHEMA_VERSION",
    "export_abaqus_inp",
    "export_vtk",
    "build_analysis_summary",
    "recommend_disposition",
    "render_summary_markdown",
    "render_summary_pdf",
    "export_summary",
]
