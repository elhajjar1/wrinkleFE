# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for building wrinklefe.exe (single-file Windows build).

Entry point:
    scripts/build_app.py - bootstrap that imports wrinklefe.gui.main_window.launch

Runtime hook:
    scripts/rthook_clean_path.py - cleans sys.path entries that break pkg_resources

Output:
    dist/wrinklefe.exe
"""

import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

# Project layout: spec lives at repo root, src/ holds the wrinklefe package.
project_root = os.path.abspath(os.getcwd())
src_path = os.path.join(project_root, "src")

# Bundle the entire wrinklefe package (submodules + any data files).
hiddenimports = collect_submodules("wrinklefe")
# GUI / scientific stack submodules PyInstaller can miss via static analysis.
for pkg in ("pyvista", "pyvistaqt", "vtkmodules", "matplotlib", "scipy"):
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        pass

datas = []
for pkg in ("wrinklefe", "pyvista", "pyvistaqt", "matplotlib"):
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass


a = Analysis(
    [os.path.join("scripts", "build_app.py")],
    pathex=[src_path, project_root],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join("scripts", "rthook_clean_path.py")],
    excludes=["tkinter", "pytest", "mypy", "ruff"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="wrinklefe",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
