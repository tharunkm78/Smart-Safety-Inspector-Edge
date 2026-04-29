# Smart Safety Inspector — Windows Development Setup
# Run in PowerShell: .\scripts\setup_windows.ps1

Write-Host "=== Smart Safety Inspector — Windows Setup ===" -ForegroundColor Cyan
Write-Host "Platform: Windows (Development)" -ForegroundColor Gray
Write-Host ""

$ErrorActionPreference = "Stop"

# Check Python
Write-Host "[1/6] Checking Python..." -NoNewline
try {
    $pyVersion = python --version 2>&1
    Write-Host " $pyVersion" -ForegroundColor Green
} catch {
    Write-Host " Python not found!" -ForegroundColor Red
    Write-Host " Install Python 3.9+ from https://python.org" -ForegroundColor Yellow
    exit 1
}

# Create virtual environment
Write-Host "[2/6] Creating virtual environment..." -NoNewline
if (-not (Test-Path "venv")) {
    python -m venv venv
    Write-Host " created" -ForegroundColor Green
} else {
    Write-Host " already exists" -ForegroundColor Gray
}

# Activate venv
& "venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "[3/6] Upgrading pip..." -NoNewline
python -m pip install --upgrade pip --quiet
Write-Host " done" -ForegroundColor Green

# Install CUDA-aware PyTorch (if CUDA available)
Write-Host "[4/6] Installing PyTorch..." -NoNewline
$cudaAvailable = python -c "import torch; print(torch.cuda.is_available())" 2>&1
if ($cudaAvailable -eq "True") {
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
    Write-Host " (CUDA)" -ForegroundColor Green
} else {
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
    Write-Host " (CPU)" -ForegroundColor Yellow
}

# Install core dependencies
Write-Host "[5/6] Installing dependencies..." -NoNewline
pip install -r requirements.windows.txt --quiet
Write-Host " done" -ForegroundColor Green

# Install Ultralytics
Write-Host "[6/6] Installing Ultralytics..." -NoNewline
pip install ultralytics --quiet
Write-Host " done" -ForegroundColor Green

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Cyan
Write-Host "Activate environment: .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Balance dataset:   python -m src.data.balance_dataset" -ForegroundColor White
Write-Host "  2. Train model:        python -m src.training.train_yolov8 --epochs 10" -ForegroundColor White
Write-Host "  3. Start API:          python -m src.api.main" -ForegroundColor White
Write-Host "  4. Open dashboard:      http://localhost:8000" -ForegroundColor White
