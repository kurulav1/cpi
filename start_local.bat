@echo off
setlocal EnableExtensions

if /I "%~1"=="--help" goto :help
if /I "%~1"=="/?" goto :help

set "REPO_DIR=%~dp0"
set "WEB_DIR=%REPO_DIR%web"
set "INFER_BIN=%REPO_DIR%build\Release\llama_infer.exe"

if not exist "%WEB_DIR%\package.json" (
  echo [start_local] Could not find web\package.json.
  exit /b 1
)

if not exist "%WEB_DIR%\.env" (
  echo [start_local] web\.env not found, copying from web\.env.example
  copy /Y "%WEB_DIR%\.env.example" "%WEB_DIR%\.env" >nul
)

if not exist "%INFER_BIN%" (
  echo [start_local] llama_infer.exe is missing, building it now...
  if not exist "%REPO_DIR%build\CMakeCache.txt" (
    cmake -S "%REPO_DIR%" -B "%REPO_DIR%build" -G "Visual Studio 17 2022" -A x64
    if errorlevel 1 exit /b 1
  )
  cmake --build "%REPO_DIR%build" --config Release --target llama_infer
  if errorlevel 1 exit /b 1
)

pushd "%WEB_DIR%"

if not exist "node_modules" (
  echo [start_local] Installing web dependencies...
  call npm install
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
echo   2. Builds build\Release\llama_infer.exe if missing
echo   3. Installs web dependencies if needed
echo   4. Builds the React UI
echo   5. Starts the local server on http://localhost:3001
exit /b 0
