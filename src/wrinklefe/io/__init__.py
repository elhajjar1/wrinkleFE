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

The legacy ``export_results_json`` (in :mod:`wrinklefe.io.export`) and
the schema-versioned, tabular pair in :mod:`wrinklefe.io.results`
(:func:`export_results_json`, :func:`export_results_csv`,
:func:`results_to_dict`) coexist. The legacy entry remains the default
under ``wrinklefe.io.export_results_json`` so existing callers do not
break; consumers who want the v1.0 schema (with ``per_ply`` table,
``first_ply_failure``, ``knockdown_factors``, ``schema_version``) should
import directly from :mod:`wrinklefe.io.results`.
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
