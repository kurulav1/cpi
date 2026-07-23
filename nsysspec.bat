@echo off
cd /d "C:\Users\VIN~1\Downloads\cpi" || exit /b 1
set M=artifacts\hub\google__gemma-4-E2B-it
set CPI_CUDA_SPEC=7
set CPI_CUDA_SPEC_STATS=1
"C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe" profile -t cuda --cuda-graph-trace=node -o C:\Users\VIN~1\Downloads\cpi\cpi_spec_prof --force-overwrite true build-cuda-ninja\llama_infer.exe %M%\gemma4-e2b.cpi --tokenizer %M%\hf\tokenizer.json --prompt "The lighthouse keeper climbed the spiral stairs, and the storm outside grew louder." --max-new 200 --temp 0 --weight-quant int4 --max-context 4096 --no-resource-limits --benchmark
echo NSYS_EXIT=%errorlevel%
