@echo off
cd /d "C:\Users\VIN~1\Downloads\cpi" || exit /b 1
set M=artifacts\hub\google__gemma-4-E2B-it
"C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe" profile -t cuda --cuda-graph-trace=node -o C:\Users\VIN~1\Downloads\cpi\prof2 --force-overwrite true build-cuda-ninja\cpi.exe %M%\gemma4-e2b.cpi --tokenizer %M%\hf\tokenizer.json --chat-template gemma --prompt "Write a very long detailed story about the sea." --max-new 120 --temp 0 --weight-quant int4
echo EXIT=%errorlevel%
