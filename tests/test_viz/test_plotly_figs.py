"""Package-path tests for :mod:`wrinklefe.viz.plotly_figs` (issue #286).

The figure builders' behaviour is covered in ``tests/test_streamlit_viz.py``
through the deprecated root shim.  What is pinned *here* is the packaging
contract that moving them into ``wrinklefe.viz`` was for:

- they are importable from the package namespace and build a real figure;
- ``import wrinklefe.viz`` does **not** drag plotly in, so a plain
  ``pip install wrinklefe`` environment keeps working (checked in a
  subprocess, since plotly is already imported in this one);
- requesting a figure builder without plotly installed fails with a
  message naming the extra, not a bare ``No module named 'plotly'``.

The wheel-level counterpart of the first two — install the built wheel
into a clean venv with and without plotly — runs in the ``build`` CI job.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

pytestmark = pytest.mark.viz


def _unit_cube_mesh():
    """A single hex8 element: the smallest mesh a Mesh3d can be built from."""
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    elements = np.arange(8, dtype=np.int64).reshape(1, 8)
    return nodes, elements


def _run(code: str) -> subprocess.CompletedProcess[str]:
    """Run a snippet in a fresh interpreter with this repo's src on the path."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# The package path works
# ---------------------------------------------------------------------------


def test_figure_builders_importable_from_package_namespace():
    """``from wrinklefe.viz import ...`` resolves the lazily-served names."""
    pytest.importorskip("plotly")
    from wrinklefe.viz import mesh3d_figure, stress_contour_figure
    from wrinklefe.viz import plotly_figs as pf

    assert mesh3d_figure is pf.mesh3d_figure
    assert stress_contour_figure is pf.stress_contour_figure


def test_mesh3d_figure_from_package_builds_a_figure():
    """The acceptance-criteria smoke test: a figure from a tiny mesh."""
    pytest.importorskip("plotly")
    from wrinklefe.viz import mesh3d_figure

    nodes, elements = _unit_cube_mesh()
    fig = mesh3d_figure(nodes, elements, cell_scalar=np.array([1.0]))
    assert len(fig.data) == 1
    mesh = fig.data[0]
    # One hex: 6 boundary quads -> 12 triangles over its 8 corner nodes.
    assert len(mesh.x) == 8
    assert len(mesh.i) == 12


def test_lazy_names_appear_in_dir_and_all():
    import wrinklefe.viz as viz

    assert "mesh3d_figure" in viz.__all__
    assert "mesh3d_figure" in dir(viz)


def test_unknown_attribute_still_raises_attribute_error():
    import wrinklefe.viz as viz

    with pytest.raises(AttributeError, match="no attribute 'not_a_figure'"):
        viz.not_a_figure


# ---------------------------------------------------------------------------
# Plotly stays optional (the lazy-import contract)
# ---------------------------------------------------------------------------


def test_importing_wrinklefe_viz_does_not_import_plotly():
    """Importing the viz package must not pull plotly into ``sys.modules``.

    Run in a subprocess: this test session has already imported plotly, so
    an in-process assertion would be meaningless.
    """
    proc = _run(
        """
        import sys
        import wrinklefe.viz
        assert "plotly" not in sys.modules, sorted(
            m for m in sys.modules if m.startswith("plotly")
        )
        print("OK")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_missing_plotly_raises_with_the_install_command():
    """With plotly unimportable, asking for a figure names the extra.

    ``plotly`` is blocked by a ``sys.meta_path`` finder rather than by
    uninstalling it, so the check runs in any environment.
    """
    proc = _run(
        """
        import sys

        class _BlockPlotly:
            def find_spec(self, name, path=None, target=None):
                if name == "plotly" or name.startswith("plotly."):
                    raise ImportError("No module named 'plotly'")
                return None

        sys.meta_path.insert(0, _BlockPlotly())
        for mod in [m for m in sys.modules if m.startswith("plotly")]:
            del sys.modules[mod]

        import wrinklefe.viz  # must succeed without plotly
        try:
            wrinklefe.viz.mesh3d_figure
        except ImportError as exc:
            print(f"RAISED: {exc}")
        else:
            raise AssertionError("expected ImportError")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "RAISED:" in proc.stdout
    assert "wrinklefe[plotly]" in proc.stdout
    assert "pip install" in proc.stdout
