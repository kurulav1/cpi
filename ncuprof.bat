@echo off
cd /d "C:\Users\VIN~1\Downloads\cpi" || exit /b 1
set "USERPROFILE=C:\Users\VIN~1"
set "APPDATA=C:\Users\VIN~1\AppData\Roaming"
set "LOCALAPPDATA=C:\Users\VIN~1\AppData\Local"
set M=artifacts\hub\google__gemma-4-E2B-it
"C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.0\target\windows-desktop-win7-x64\ncu.exe" --kernel-name regex:%1 --launch-skip %2 --launch-count %3 --section SpeedOfLight --section WarpStateStats --section SchedulerStats build-cuda-ninja\llama_infer.exe %M%\gemma4-e2b.cpi --tokenizer %M%\hf\tokenizer.json --chat-template gemma --prompt "Write one sentence about the sea." --max-new 6 --temp 0 --weight-quant int4
echo NCU_EXIT=%errorlevel%
