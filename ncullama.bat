@echo off
cd /d "C:\Users\VIN~1\Downloads\cpi\artifacts\llamacpp" || exit /b 1
set "USERPROFILE=C:\Users\VIN~1"
set "APPDATA=C:\Users\VIN~1\AppData\Roaming"
set "LOCALAPPDATA=C:\Users\VIN~1\AppData\Local"
"C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.0\target\windows-desktop-win7-x64\ncu.exe" --kernel-name regex:mul_mat_vec_q --launch-skip %1 --launch-count %2 --section SpeedOfLight --section Occupancy bin\llama-bench.exe -m e2b-Q4_0.gguf -n 8 -p 0 -r 1
echo NCU_EXIT=%errorlevel%
