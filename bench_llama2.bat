@echo off
set BENCH_PROMPT=%~1
if "%BENCH_PROMPT%"=="" set BENCH_PROMPT=Tell me a short story
if "%CPI_MODEL%"=="" (
    echo [bench_llama2] CPI_MODEL is not set. Set it to the path of your .ll2c file.
    exit /b 1
)
if "%CPI_TOKENIZER%"=="" (
    echo [bench_llama2] CPI_TOKENIZER is not set. Set it to the path of your tokenizer file.
    exit /b 1
)
"%~dp0build\Release\llama_infer.exe" ^
    "%CPI_MODEL%" ^
    --int8-streaming ^
    --max-context 460 ^
    --tokenizer "%CPI_TOKENIZER%" ^
    --chat-template llama2 ^
    --max-new 100 ^
    --benchmark-reps 3 ^
    --prompt "%BENCH_PROMPT%"
