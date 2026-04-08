@echo off
if "%~1"=="" (
    echo Usage: run_cpu.bat "your prompt here"
    echo.
    echo Runs CPU inference on the bundled artifacts\tinyllama-streaming-rowwise.ll2c model.
    echo Set CPI_TOKENIZER to the path of your TinyLlama tokenizer.json or tokenizer.model.
    exit /b 1
)
if "%CPI_TOKENIZER%"=="" (
    echo [run_cpu] CPI_TOKENIZER is not set. Set it to the path of your tokenizer file.
    exit /b 1
)
"%~dp0build\Release\llama_infer.exe" ^
    "%~dp0artifacts\tinyllama-streaming-rowwise.ll2c" ^
    --cpu ^
    --tokenizer "%CPI_TOKENIZER%" ^
    --chat-template tinyllama ^
    --max-context 512 ^
    --max-new 200 ^
    --prompt "%~1"
