@echo off
if "%~1"=="" (
    echo Usage: run_cpu_llama2.bat "your prompt here"
    echo.
    echo Required environment variables:
    echo   CPI_MODEL      - path to .ll2c model file  (e.g. C:\models\llama2-7b-chat.ll2c)
    echo   CPI_TOKENIZER  - path to tokenizer file    (e.g. C:\models\tokenizer.model)
    exit /b 1
)
if "%CPI_MODEL%"=="" (
    echo [run_cpu_llama2] CPI_MODEL is not set. Set it to the path of your .ll2c file.
    exit /b 1
)
if "%CPI_TOKENIZER%"=="" (
    echo [run_cpu_llama2] CPI_TOKENIZER is not set. Set it to the path of your tokenizer file.
    exit /b 1
)
"%~dp0build\Release\llama_infer.exe" ^
    "%CPI_MODEL%" ^
    --cpu ^
    --tokenizer "%CPI_TOKENIZER%" ^
    --chat-template llama2 ^
    --max-context 400 ^
    --max-new 400 ^
    --prompt "%~1"
