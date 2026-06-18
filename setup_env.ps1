# ============================================================
# setup_env.ps1 — One-command environment bootstrap
# Usage: .\setup_env.ps1
# ============================================================

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Federated Agile Effort Estimation"      -ForegroundColor Cyan
Write-Host " Environment Setup"                      -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check Python version
$pythonVersion = python --version 2>&1
Write-Host "[1/4] Python detected: $pythonVersion" -ForegroundColor Yellow

# 2. Create virtual environment
if (Test-Path ".venv") {
    Write-Host "[2/4] Virtual environment already exists at .venv\" -ForegroundColor Green
} else {
    Write-Host "[2/4] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    Write-Host "      Created .venv\" -ForegroundColor Green
}

# 3. Activate and install
Write-Host "[3/4] Installing dependencies..." -ForegroundColor Yellow
& .venv\Scripts\python.exe -m pip install --upgrade pip --quiet
& .venv\Scripts\python.exe -m pip install -r requirements.txt --quiet

# 4. Create data directories
$dirs = @("data\raw", "data\processed", "data\features", "notebooks", "tests")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Host "[4/4] Project directories created." -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " Setup complete!"                        -ForegroundColor Green
Write-Host " Activate with: .venv\Scripts\Activate"  -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
