@echo off
REM build_windows.bat - Build WrinkleFE standalone Windows executable
REM
REM Usage (from project root):
REM     scripts\build_windows.bat
REM
REM Output:
REM     dist\wrinklefe.exe

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."
pushd "%PROJECT_DIR%"
for %%I in ("%CD%") do set "PROJECT_DIR=%%~fI"

echo ========================================
echo   WrinkleFE Windows Executable Builder
echo ========================================
echo.
echo   Project:  %PROJECT_DIR%
for /f "delims=" %%V in ('python --version 2^>^&1') do echo   Python:   %%V
echo.

REM ------------------------------------------------------------------
REM 1. Check / install dependencies
REM ------------------------------------------------------------------
echo [1/4] Checking dependencies...

call :install_if_missing PyInstaller
if errorlevel 1 goto :fail
call :install_if_missing PyQt6
if errorlevel 1 goto :fail
call :install_if_missing numpy
if errorlevel 1 goto :fail
call :install_if_missing scipy
if errorlevel 1 goto :fail
call :install_if_missing matplotlib
if errorlevel 1 goto :fail
call :install_if_missing pyvista
if errorlevel 1 goto :fail
call :install_if_missing pyvistaqt
if errorlevel 1 goto :fail

echo   Installing wrinklefe (editable)...
python -m pip install -e "%PROJECT_DIR%" --quiet
if errorlevel 1 goto :fail

echo.

REM ------------------------------------------------------------------
REM 2. Build
REM ------------------------------------------------------------------
echo [2/4] Building wrinklefe.exe...
echo.

python -m PyInstaller wrinklefe_windows.spec --clean --noconfirm
if errorlevel 1 goto :fail

echo.

REM ------------------------------------------------------------------
REM 3. Verify output
REM ------------------------------------------------------------------
echo [3/4] Verifying output...

set "EXE_PATH=%PROJECT_DIR%\dist\wrinklefe.exe"

if exist "%EXE_PATH%" (
    for %%A in ("%EXE_PATH%") do set "EXE_SIZE=%%~zA"
    set /a EXE_MB=!EXE_SIZE! / 1048576
    echo   SUCCESS: %EXE_PATH%
    echo   Size:    !EXE_MB! MB
) else (
    echo   ERROR: %EXE_PATH% not found!
    echo   Check the build output above for errors.
    goto :fail
)

echo.

REM ------------------------------------------------------------------
REM 4. Summary
REM ------------------------------------------------------------------
echo [4/4] Done!
echo.
echo ========================================
echo   wrinklefe.exe built successfully
echo ========================================
echo.
echo   Location: %EXE_PATH%
echo.
echo   To run:
echo     "%EXE_PATH%"
echo.

popd
endlocal
exit /b 0

:install_if_missing
python -c "import %~1" >nul 2>&1
if errorlevel 1 (
    echo   Installing %~1...
    python -m pip install "%~1" --quiet
    if errorlevel 1 exit /b 1
) else (
    echo   %~1: OK
)
exit /b 0

:fail
echo.
echo   BUILD FAILED.
popd
endlocal
exit /b 1
