wrinklefe.viz
=============

Interactive Plotly figures
--------------------------

Plotly figure builders for FE meshes and result fields — the boundary
surface of a hex mesh as a ``Mesh3d``, deformed-mesh and failure-index
views, and 2D y-slice scatters. They are what the Streamlit app draws,
and they are equally usable from a notebook.

Plotly is optional (the ``plotly`` extra, also pulled in by
``streamlit``), so these names are served from :mod:`wrinklefe.viz`
lazily: importing the package never imports plotly, and the import
happens on first attribute access.

.. code-block:: bash

   pip install 'wrinklefe[plotly]'

.. code-block:: python

   from wrinklefe.viz import mesh3d_figure, stress_contour_figure

   fig = stress_contour_figure(nodes, elements, stress_per_elem)
   fig.show()

Without plotly installed, that import raises an :class:`ImportError`
naming the command above; the rest of :mod:`wrinklefe.viz` (the
matplotlib plots) is unaffected.

.. automodule:: wrinklefe.viz.plotly_figs
   :members:
   :show-inheritance:
