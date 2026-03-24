#!/usr/bin/env python3
"""Bootstrap entry point for PyInstaller.

PyInstaller requires a concrete .py file as the analysis entry point.
This script simply imports and launches the WrinkleFE GUI.
"""

import sys
import os

# Ensure the src directory is on the path when running from source
_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src")
if os.path.isdir(_src):
    sys.path.insert(0, _src)

from wrinklefe.gui.main_window import launch

if __name__ == "__main__":
    launch()
