# Smart Safety Inspector — Windows Development Runner
# Run in PowerShell: .\scripts\run_windows_dev.ps1

param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent

Write-Host "=== Smart Safety Inspector — Windows Dev Mode ===" -ForegroundColor Cyan
Write-Host "Project: $ProjectRoot"
Write-Host "Port: $Port"
Write-Host ""

# Activate venv if exists
$venvActivate = "$ProjectRoot\venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    Write-Host "[*] Activating virtual environment..." -ForegroundColor Gray
    & $venvActivate
} else {
    Write-Host "[!] No venv found — using system Python" -ForegroundColor Yellow
}

# Check model
$modelPath = "$ProjectRoot\models\yolov8n_safety_v1.pt"
if (Test-Path $modelPath) {
    $size = [math]::Round((Get-Item $modelPath).Length / 1MB, 1)
    Write-Host "[*] Model found: $modelPath ($size MB)" -ForegroundColor Green
} else {
    Write-Host "[!] Model not found: $modelPath" -ForegroundColor Yellow
    Write-Host "    Run training first: python -m src.training.train_yolov8" -ForegroundColor Yellow
}

# Start API server in background
Write-Host ""
Write-Host "[*] Starting API server on http://localhost:$Port ..." -ForegroundColor Gray
Start-Process python -ArgumentList "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "$Port", "--reload" -WorkingDirectory $ProjectRoot -NoNewWindow

Start-Sleep 3

Write-Host ""
Write-Host "[*] API running at http://localhost:$Port" -ForegroundColor Green
Write-Host "[*] Dashboard:    http://localhost:$Port" -ForegroundColor Green
Write-Host "[*] API docs:    http://localhost:$Port/docs" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the server." -ForegroundColor Yellow

# Wait for user interrupt
try {
    while ($true) { Start-Sleep 1 }
} finally {
    Write-Host ""
    Write-Host "[*] Stopping server..." -ForegroundColor Gray
    Stop-Process -Name "python" -Force -ErrorAction SilentlyContinue
    Write-Host "[*] Done." -ForegroundColor Cyan
}
