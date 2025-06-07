# build_all.ps1
# One-shot script: create venv with uv, activate, build Rust extension, install deps, run tests

# 1. Create .venv if not present
if (!(Test-Path ".venv")) {
    Write-Host "Creating virtual environment with uv..."
    uv venv
}

# 2. Activate the venv
Write-Host "Activating virtual environment..."
.venv/Scripts/Activate.ps1

# 3. Install Python dependencies
Write-Host "Installing Python dependencies..."
uv pip install -r requirements.txt

# 4. Build Rust extension with maturin (in venv)
Write-Host "Building Rust extension with maturin..."
maturin develop --release

# 5. Run tests
Write-Host "Running tests..."
pytest

Write-Host "Build and test complete!"
