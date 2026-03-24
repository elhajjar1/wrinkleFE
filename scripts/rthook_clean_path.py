"""PyInstaller runtime hook to clean sys.path before pkg_resources loads.

Removes entries that contain spaces or non-standard characters that cause
pkg_resources to crash when parsing them as version strings.
"""
import sys
import os

# Keep only paths that are inside the bundle or standard library
_app_dir = os.path.dirname(sys.executable)
_clean = []
for p in sys.path:
    # Keep the app's own directories and stdlib
    if (
        p.startswith(_app_dir)
        or p == ""
        or p.startswith(sys.prefix)
        or "site-packages" not in p
    ):
        _clean.append(p)
sys.path[:] = _clean
