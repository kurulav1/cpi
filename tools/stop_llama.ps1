param(
  [switch]$AlsoWeb
)

$ErrorActionPreference = "Stop"

Write-Host "[stop-llama] Searching for llama_infer processes..."
$llama = Get-Process -Name "llama_infer" -ErrorAction SilentlyContinue
if (-not $llama) {
  Write-Host "[stop-llama] No llama_infer process found."
} else {
  $llama | ForEach-Object {
    Write-Host ("[stop-llama] Stopping PID={0}" -f $_.Id)
    Stop-Process -Id $_.Id -Force
  }
}

if ($AlsoWeb) {
  Write-Host "[stop-llama] Checking web server listeners on ports 3001 and 7860..."
  foreach ($port in @(3001, 7860)) {
    $listeners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
      Select-Object -ExpandProperty OwningProcess -Unique
    if ($listeners) {
      foreach ($pid in $listeners) {
        Write-Host ("[stop-llama] Stopping web PID={0} (port {1})" -f $pid, $port)
        Stop-Process -Id $pid -Force
      }
    }
  }
}

Start-Sleep -Milliseconds 150
$maxChecks = 10
for ($i = 0; $i -lt $maxChecks; $i++) {
  $left = Get-Process -Name "llama_infer" -ErrorAction SilentlyContinue
  if (-not $left) {
    Write-Host "[stop-llama] Done."
    exit 0
  }
  Start-Sleep -Milliseconds 120
}

Write-Host "[stop-llama] Warning: llama_infer still running." -ForegroundColor Yellow
$left | Select-Object Id, ProcessName | Format-Table -AutoSize
exit 1
