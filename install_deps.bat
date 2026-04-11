@echo off
setlocal EnableExtensions

if /I "%~1"=="--help" goto :help
if /I "%~1"=="/?" goto :help

for %%I in ("%~dp0.") do set "REPO_DIR=%%~fI"
set "WEB_DIR=%REPO_DIR%\web"
set "REQ_FILE=%REPO_DIR%\requirements.txt"

if not exist "%WEB_DIR%\package.json" (
  echo [install_deps] Could not find web\package.json.
  exit /b 1
)

if not exist "%REQ_FILE%" (
  echo [install_deps] Could not find requirements.txt.
  exit /b 1
)

where py >nul 2>nul
if not errorlevel 1 (
  set "PYTHON=py -3"
  goto :python_found
)

where python >nul 2>nul
if not errorlevel 1 (
  set "PYTHON=python"
  goto :python_found
)

echo [install_deps] Python was not found on PATH.
exit /b 1

:python_found
echo [install_deps] Installing Python dependencies...
call %PYTHON% -m pip install -r "%REQ_FILE%"
if errorlevel 1 exit /b 1

pushd "%WEB_DIR%"
if not exist "package-lock.json" (
  echo [install_deps] package-lock.json is missing, cannot run npm ci.
  popd
  exit /b 1
)
echo [install_deps] Installing web dependencies with npm ci...
call npm ci
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%

:help
echo Usage: install_deps.bat
echo.
echo Installs repo-managed dependencies:
echo   1. Python packages from requirements.txt
echo   2. Web packages from web\package-lock.json ^(npm ci^)
exit /b 0
