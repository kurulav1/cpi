@echo off
setlocal EnableExtensions

if /I "%~1"=="--help" goto :help
if /I "%~1"=="/?" goto :help

set "REPO_DIR=%~dp0"
set "WEB_DIR=%REPO_DIR%web"
set "INFER_BIN=%REPO_DIR%build\Release\llama_infer.exe"

if not exist "%WEB_DIR%\package.json" (
  echo [start_web] Could not find web\package.json.
  exit /b 1
)

if not exist "%WEB_DIR%\.env" (
  if exist "%WEB_DIR%\.env.example" (
    echo [start_web] web\.env not found, copying from web\.env.example
    copy /Y "%WEB_DIR%\.env.example" "%WEB_DIR%\.env" >nul
  ) else (
    echo [start_web] web\.env and web\.env.example are missing.
    echo [start_web] Create web\.env manually if your setup requires it.
  )
)

if not exist "%WEB_DIR%\config.json" (
  if exist "%WEB_DIR%\config.example.json" (
    echo [start_web] web\config.json not found, copying from web\config.example.json
    copy /Y "%WEB_DIR%\config.example.json" "%WEB_DIR%\config.json" >nul
    echo [start_web] Edit web\config.json and set modelPath/tokenizerPath before generating.
  ) else (
    echo [start_web] web\config.json and web\config.example.json are missing.
  )
)

if not exist "%INFER_BIN%" (
  echo [start_web] Warning: %INFER_BIN% not found.
  echo [start_web] API will start, but inference requests will fail until llama_infer is built.
)

pushd "%WEB_DIR%"

if not exist "node_modules" (
  if exist "package-lock.json" (
    echo [start_web] Installing web dependencies with npm ci...
    call npm ci
  ) else (
    echo [start_web] Installing web dependencies with npm install...
    call npm install
  )
  if errorlevel 1 (
    popd
    exit /b 1
  )
)

echo [start_web] Starting dev server:
echo [start_web] API: http://localhost:3001
echo [start_web] UI : http://localhost:5173
call npm run dev
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%

:help
echo Usage: start_web.bat
echo.
echo Starts the web dev stack:
echo   1. Copies web\.env from web\.env.example if needed
echo   2. Copies web\config.json from web\config.example.json if needed
echo   3. Installs dependencies if node_modules is missing
echo   4. Runs npm run dev ^(API + Vite UI^)
exit /b 0
