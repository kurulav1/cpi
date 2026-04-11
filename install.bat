@echo off
setlocal EnableExtensions

if /I "%~1"=="--help" goto :help
if /I "%~1"=="/?" goto :help

for %%I in ("%~dp0.") do set "REPO_DIR=%%~fI"
set "WEB_DIR=%REPO_DIR%\web"
set "BUILD_DIR=%REPO_DIR%\build"
set "INFER_BIN=%BUILD_DIR%\Release\llama_infer.exe"

if not exist "%WEB_DIR%\package.json" (
  echo [install] Could not find web\package.json.
  exit /b 1
)

echo [install] Installing Python and web dependencies...
call "%REPO_DIR%\install_deps.bat"
if errorlevel 1 exit /b 1

if not exist "%WEB_DIR%\.env" (
  if exist "%WEB_DIR%\.env.example" (
    echo [install] Creating web\.env from web\.env.example
    copy /Y "%WEB_DIR%\.env.example" "%WEB_DIR%\.env" >nul
  )
)

if not exist "%WEB_DIR%\config.json" (
  if exist "%WEB_DIR%\config.example.json" (
    echo [install] Creating web\config.json from web\config.example.json
    copy /Y "%WEB_DIR%\config.example.json" "%WEB_DIR%\config.json" >nul
  )
)

if not exist "%INFER_BIN%" (
  echo [install] Building llama_infer.exe...
  cmake --fresh -S "%REPO_DIR%" -B "%BUILD_DIR%" -A x64
  if errorlevel 1 exit /b 1
  cmake --build "%BUILD_DIR%" --config Release --target llama_infer
  if errorlevel 1 exit /b 1
)

echo [install] Repo is prepared.
echo [install] Next steps:
echo [install]   1. Put or convert models into artifacts\
echo [install]   2. Start dev UI with start_web.bat
echo [install]   3. Or run the local packaged app with start_local.bat
exit /b 0

:help
echo Usage: install.bat
echo.
echo Prepares the repo for first use:
echo   1. Installs Python dependencies from requirements.txt
echo   2. Installs web dependencies
echo   3. Creates web\.env if missing
echo   4. Creates web\config.json if missing
echo   5. Builds build\Release\llama_infer.exe if missing
exit /b 0
