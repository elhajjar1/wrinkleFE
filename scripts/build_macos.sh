#!/usr/bin/env bash
# build_macos.sh — Build WrinkleFE standalone macOS .app bundle
#
# Usage:
#   bash scripts/build_macos.sh
#
# Output:
#   dist/WrinkleFE.app

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================"
echo "  WrinkleFE macOS App Builder"
echo "========================================"
echo ""
echo "  Project:  $PROJECT_DIR"
echo "  Python:   $(python3 --version 2>&1)"
echo "  Arch:     $(uname -m)"
echo ""

# ------------------------------------------------------------------
# 1. Check / install dependencies
# ------------------------------------------------------------------
echo "[1/4] Checking dependencies..."

install_if_missing() {
    local pkg="$1"
    if ! python3 -c "import $pkg" 2>/dev/null; then
        echo "  Installing $pkg..."
        python3 -m pip install "$pkg" --quiet
    else
        echo "  $pkg: OK"
    fi
}

install_if_missing PyInstaller
install_if_missing PyQt6
install_if_missing numpy
install_if_missing scipy
install_if_missing matplotlib
install_if_missing pyvista
install_if_missing pyvistaqt

# Ensure wrinklefe itself is installed in development mode
echo "  Installing wrinklefe (editable)..."
python3 -m pip install -e "$PROJECT_DIR" --quiet

echo ""

# ------------------------------------------------------------------
# 2. Build
# ------------------------------------------------------------------
echo "[2/4] Building .app bundle..."
echo ""

cd "$PROJECT_DIR"
python3 -m PyInstaller wrinklefe.spec --clean --noconfirm 2>&1 | tail -20

echo ""

# ------------------------------------------------------------------
# 3. Verify output
# ------------------------------------------------------------------
echo "[3/4] Verifying output..."

APP_PATH="$PROJECT_DIR/dist/WrinkleFE.app"

if [ -d "$APP_PATH" ]; then
    APP_SIZE=$(du -sh "$APP_PATH" | cut -f1)
    echo "  SUCCESS: $APP_PATH"
    echo "  Size:    $APP_SIZE"
else
    echo "  ERROR: $APP_PATH not found!"
    echo "  Check the build output above for errors."
    exit 1
fi

echo ""

# ------------------------------------------------------------------
# 4. Summary
# ------------------------------------------------------------------
echo "[4/4] Done!"
echo ""
echo "========================================"
echo "  WrinkleFE.app built successfully"
echo "========================================"
echo ""
echo "  Location: $APP_PATH"
echo "  Size:     $APP_SIZE"
echo ""
echo "  To run:"
echo "    open \"$APP_PATH\""
echo ""
echo "  To distribute:"
echo "    zip -r WrinkleFE.zip dist/WrinkleFE.app"
echo ""
