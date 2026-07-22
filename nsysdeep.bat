@echo off
cd /d "C:\Users\VIN~1\Downloads\cpi" || exit /b 1
set M=artifacts\hub\google__gemma-4-E2B-it
"C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe" profile -t cuda --cuda-graph-trace=node -o C:\Users\VIN~1\Downloads\cpi\profdeep --force-overwrite true build-cuda-ninja\gemma4_forward_test.exe --model %M%\gemma4-e2b.cpi --weight-quant 4 --graph-bench 60 --graph-pos 3500 --tokens 2,105,2364,107,6974,886,13315,1003,506,5442,236761,106,107,105,4368,107
echo EXIT=%errorlevel%
