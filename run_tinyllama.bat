@echo off
if "%~1"=="" (
    echo Usage: run_tinyllama.bat "your prompt here"
    echo.
    echo Required environment variables:
    echo   CPI_MODEL      - path to .ll2c model file  (e.g. C:\models\tinyllama-1.1b-chat.ll2c)
    echo   CPI_TOKENIZER  - path to tokenizer.json    (e.g. C:\models\tokenizer.json)
    exit /b 1
)
if "%CPI_MODEL%"=="" (
    echo [run_tinyllama] CPI_MODEL is not set. Set it to the path of your .ll2c file.
    exit /b 1
)
if "%CPI_TOKENIZER%"=="" (
    echo [run_tinyllama] CPI_TOKENIZER is not set. Set it to the path of your tokenizer file.
    exit /b 1
)
"%~dp0build\Release\llama_infer.exe" ^
    "%CPI_MODEL%" ^
    --tokenizer "%CPI_TOKENIZER%" ^
    --chat-template tinyllama ^
    --max-new 400 ^
    --prompt "%~1"
