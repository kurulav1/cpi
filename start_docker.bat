@echo off
setlocal EnableExtensions

if /I "%~1"=="--help" goto :help
if /I "%~1"=="/?" goto :help

set "MODELS_DIR=%~1"
if "%MODELS_DIR%"=="" set "MODELS_DIR=%~dp0models"

set "IMAGE_TAG=%~2"
if "%IMAGE_TAG%"=="" set "IMAGE_TAG=cpi-chat-ui"

if not exist "%MODELS_DIR%" (
  echo [start_docker] Models directory does not exist: %MODELS_DIR%
  exit /b 1
)

echo [start_docker] Building Docker image %IMAGE_TAG%...
docker build -t "%IMAGE_TAG%" "%~dp0"
if errorlevel 1 exit /b 1

echo [start_docker] Starting container on http://localhost:3001
echo [start_docker] Mounting host models from %MODELS_DIR%
docker run --rm -it -p 3001:3001 --gpus all ^
  -e LLAMA_MODEL_DIRS=/models ^
  -v "%MODELS_DIR%:/models:ro" ^
  "%IMAGE_TAG%"
exit /b %ERRORLEVEL%

:help
echo Usage: start_docker.bat [models_dir] [image_tag]
echo.
echo models_dir defaults to .\models (a "models" folder next to start_docker.bat)
echo image_tag defaults to cpi-chat-ui
exit /b 0
