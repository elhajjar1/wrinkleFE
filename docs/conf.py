"""Sphinx configuration for the WrinkleFE documentation site."""

import os
import sys

# Make the package importable without installation (autodoc).
sys.path.insert(0, os.path.abspath("../src"))

project = "WrinkleFE"
author = "Elhajjar Research Group"
copyright = "2026, Elhajjar Research Group"  # noqa: A001

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

# PyVista is an optional dependency (the `vtk` extra) imported lazily
# inside the 3-D CZM plot helpers, so the docs build never needs VTK
# installed. This mock is kept defensively: if the 3-D viz module is ever
# autodoc'd, the mock lets the build succeed without pulling in VTK.
autodoc_mock_imports = ["pyvista"]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"

napoleon_numpy_docstring = True
napoleon_google_docstring = False
# Render docstring "Attributes" sections as :ivar: fields so they do not
# collide with the attribute entries autodoc emits for the same names.
napoleon_use_ivar = True

myst_enable_extensions = ["dollarmath", "colon_fence"]
myst_heading_anchors = 3

# The root markdown docs link to each other by repository-relative
# paths (e.g. ``[VALIDATION.md](VALIDATION.md)``); those targets resolve
# on GitHub but not inside the Sphinx tree, so the missing-xref warning
# is suppressed rather than failing the -W build.
suppress_warnings = ["myst.xref_missing"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "WrinkleFE"
