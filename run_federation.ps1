# run_federation.ps1
# Bootstraps the FL simulation with staggered booting and strict memory management.

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================================"
Write-Host " Phase 2 -- Federated Learning Orchestration"
Write-Host "============================================================"

# 1. Partition Data
Write-Host ""
Write-Host "  [Step 1] Partitioning data by project ..."
& .venv\Scripts\python.exe src/partition_data.py
if ($LASTEXITCODE -ne 0) { throw "Partitioning failed." }

$clientJobs = @()
$serverJob = $null

try {
    # 2. Start Server
    Write-Host ""
    Write-Host "  [Step 2] Starting FL server (port 8080) ..."
    $serverJob = Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "src/server.py" -PassThru -NoNewWindow -RedirectStandardError "NUL"
    Start-Sleep -Seconds 5

    # 3. Start 16 Clients with 3-second staggers
    Write-Host ""
    Write-Host "  [Step 3] Launching 16 FL clients (3-second stagger) ..."
    for ($i = 0; $i -lt 16; $i++) {
        Write-Host "           Client $i launched"
        $job = Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "src/client.py --client_id $i" -PassThru -NoNewWindow -RedirectStandardError "NUL"
        $clientJobs += $job
        Start-Sleep -Seconds 3
    }

    Write-Host ""
    Write-Host "  [Step 4] All clients connected. Running 3 federation rounds ..."
    Write-Host "           (Server evaluation will appear below)"
    Write-Host ""
    Wait-Process -Id $serverJob.Id -ErrorAction SilentlyContinue
    
    Write-Host ""
    Write-Host "  Federation finished!"
}
finally {
    Write-Host ""
    Write-Host "  Cleaning up background processes ..."
    if ($serverJob -and -not $serverJob.HasExited) {
        Stop-Process -Id $serverJob.Id -Force -ErrorAction SilentlyContinue
    }
    
    $stopped = 0
    foreach ($job in $clientJobs) {
        if (-not $job.HasExited) {
            Stop-Process -Id $job.Id -Force -ErrorAction SilentlyContinue
            $stopped++
        }
    }
    Write-Host "  Stopped $stopped remaining client process(es)."
    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Done."
    Write-Host "============================================================"
}
