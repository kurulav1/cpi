Write-Host "=== GPU Stats ==="
try {
  nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
} catch {
  Write-Host "nvidia-smi not available"
}

Write-Host "\n=== Host Memory (best effort) ==="
try {
  Get-Counter "\Memory\Available MBytes" | ForEach-Object { $_.CounterSamples } | Select-Object -ExpandProperty CookedValue
} catch {
  Write-Host "Host memory counters unavailable in this environment"
}