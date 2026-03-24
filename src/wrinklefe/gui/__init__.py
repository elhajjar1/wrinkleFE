"""PyQt6 graphical user interface for WrinkleFE.

Launch the GUI with::

    from wrinklefe.gui import launch
    launch()

Or from the command line::

    python -m wrinklefe.gui

Requires PyQt6: ``pip install PyQt6``
"""

from wrinklefe.gui.main_window import launch

__all__ = ["launch"]
