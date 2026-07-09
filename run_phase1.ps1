# run_phase1.ps1
# Bootstraps the Centralized Khattab Phase 1 Gold Standard calculation.

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================================"
Write-Host " Phase 1 -- Centralized Baseline (Khattab et al.)"
Write-Host "============================================================"

Write-Host ""
Write-Host "  [Executing] Centralized Training & Meta-Learner Stack ..."
& .venv\Scripts\python.exe src/CentralizedKhattab_phase1.py
if ($LASTEXITCODE -ne 0) { throw "Phase 1 execution failed." }

Write-Host ""
Write-Host "============================================================"
Write-Host " Done."
Write-Host "============================================================"
