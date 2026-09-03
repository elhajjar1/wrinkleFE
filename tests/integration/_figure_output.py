"""Output location for generated validation figures.

The Phase-7 experimental-validation tests render user-facing comparison
plots.  They used to write straight into the git-tracked ``figures/``
directory, so every local run dirtied the working tree with regenerated
PNGs and each commit needed a ``git checkout -- figures/`` first
(issue #278).

They now write to an untracked directory instead.  Set
``WRINKLEFE_FIGURE_DIR`` to render somewhere else — in particular, to
deliberately refresh the committed validation evidence::

    WRINKLEFE_FIGURE_DIR=figures pytest tests/integration -m integration

Committing a refreshed figure then becomes an explicit, reviewed act
rather than a side effect of running the suite.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Default (untracked, git-ignored) destination for generated figures.
DEFAULT_FIGURE_DIR = _REPO_ROOT / "figures" / "_generated"


def validation_figure_path(filename: str) -> Path:
    """Return the path a generated validation figure should be written to.

    Parameters
    ----------
    filename : str
        Bare file name, e.g. ``"phase7_dcb_validation.png"``.

    Returns
    -------
    Path
        Destination path; the parent directory is created if needed.
        Honours ``WRINKLEFE_FIGURE_DIR`` when set.
    """
    base = Path(os.environ.get("WRINKLEFE_FIGURE_DIR") or DEFAULT_FIGURE_DIR)
    base.mkdir(parents=True, exist_ok=True)
    return base / filename
