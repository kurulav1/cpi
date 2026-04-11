@echo off
setlocal EnableExtensions

if /I "%~1"=="--help" goto :help
if /I "%~1"=="/?" goto :help

for %%I in ("%~dp0.") do set "REPO_DIR=%%~fI"
set "WEB_DIR=%REPO_DIR%\web"
set "INFER_BIN=%REPO_DIR%\build\Release\llama_infer.exe"

if not exist "%WEB_DIR%\package.json" (
  echo [start_local] Could not find web\package.json.
  exit /b 1
)

if not exist "%WEB_DIR%\.env" (
  echo [start_local] web\.env not found, copying from web\.env.example
  copy /Y "%WEB_DIR%\.env.example" "%WEB_DIR%\.env" >nul
)

if not exist "%WEB_DIR%\config.json" (
  if exist "%WEB_DIR%\config.example.json" (
    echo [start_local] web\config.json not found, copying from web\config.example.json
    copy /Y "%WEB_DIR%\config.example.json" "%WEB_DIR%\config.json" >nul
    echo [start_local] Edit web\config.json and set modelPath/tokenizerPath before generating.
  ) else (
    echo [start_local] web\config.json and web\config.example.json are missing.
  )
)

if not exist "%INFER_BIN%" (
  echo [start_local] llama_infer.exe is missing, building it now...
  cmake --fresh -S "%REPO_DIR%" -B "%REPO_DIR%\build" -A x64
  if errorlevel 1 exit /b 1
  cmake --build "%REPO_DIR%\build" --config Release --target llama_infer
  if errorlevel 1 exit /b 1
)

pushd "%WEB_DIR%"

if not exist "node_modules" (
  if not exist "package-lock.json" (
    echo [start_local] package-lock.json is missing, cannot run npm ci.
    popd
    exit /b 1
  )
  echo [start_local] Installing web dependencies with npm ci...
  call npm ci
  if errorlevel 1 (
    popd
    exit /b 1
  )
)

echo [start_local] Building web UI...
call npm run build
if errorlevel 1 (
  popd
  exit /b 1
)

echo [start_local] Starting local package on http://localhost:3001
echo [start_local] The API launches llama_infer on demand for each chat request.
node server/index.mjs
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%

:help
echo Usage: start_local.bat
echo.
echo Starts the non-Docker package:
echo   1. Copies web\.env from web\.env.example if needed
echo   2. Copies web\config.json from web\config.example.json if needed
echo   3. Builds build\Release\llama_infer.exe if missing
echo   4. Installs web dependencies with npm ci if needed
echo   5. Builds the React UI
echo   6. Starts the local server on http://localhost:3001
exit /b 0
