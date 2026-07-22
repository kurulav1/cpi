@echo off
cd /d "C:\Users\VIN~1\Downloads\cpi\artifacts\llamacpp" || exit /b 1
"C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe" profile -t cuda --cuda-graph-trace=node -o C:\Users\VIN~1\Downloads\cpi\llamacpp_prof --force-overwrite true bin\llama-bench.exe -m e2b-Q4_0.gguf -n 200 -p 0 -r 1
echo NSYS_EXIT=%errorlevel%
