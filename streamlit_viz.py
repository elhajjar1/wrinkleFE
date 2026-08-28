"""DEPRECATED re-export shim — use :mod:`wrinklefe.viz` instead.

The Plotly figure builders that used to live in this repository-root
module now ship inside the package as
:mod:`wrinklefe.viz.plotly_figs`, so that ``pip install wrinklefe`` users
(notebooks in particular) can reach them at all — a root-level module is
not part of the wheel under the src layout. See issue #286.

Import them from the package::

    from wrinklefe.viz import mesh3d_figure, stress_contour_figure

This module is kept only so that ``import streamlit_viz`` keeps working
for the hosted Streamlit deployment and any external references; it
re-exports the package names unchanged and adds no behaviour. It emits
**no** ``DeprecationWarning``: the hosted app imports it on every script
rerun, and a warning per rerun would be pure log noise.

Requires plotly (the ``plotly`` / ``streamlit`` extras), exactly as
before — importing this shim without plotly raises the same actionable
:class:`ImportError` that :mod:`wrinklefe.viz.plotly_figs` raises.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit Cloud runs the app from a checkout it does not pip-install,
# so mirror app.py and make the src-layout package importable first.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from wrinklefe.viz.plotly_figs import (  # noqa: E402
    HEX_FACES,
    boundary_faces,
    compute_mesh3d_geometry,
    deformed_mesh_figure,
    fi_3d_figure,
    fi_y_slice_figure,
    mesh3d_figure,
    quads_to_triangles,
    stress_contour_figure,
    y_slice_figure,
)

__all__ = [
    "HEX_FACES",
    "boundary_faces",
    "quads_to_triangles",
    "compute_mesh3d_geometry",
    "mesh3d_figure",
    "stress_contour_figure",
    "deformed_mesh_figure",
    "fi_3d_figure",
    "y_slice_figure",
    "fi_y_slice_figure",
]
