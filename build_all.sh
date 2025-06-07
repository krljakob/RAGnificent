#!/usr/bin/env bash
# build_all.sh
# One-shot script: create venv with uv, activate, build Rust extension, install deps, run tests

set -e

# 1. Create .venv if not present
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with uv..."
    uv venv
fi

# 2. Set PATH to use virtual environment
export PATH=".venv/bin:$PATH"

# 3. Install Python dependencies
echo "Installing Python dependencies..."
uv pip install -r requirements.txt

# 4. Build Rust extension with maturin (in venv)
echo "Building Rust extension with maturin..."
maturin develop --release

# 5. Run tests
echo "Running tests..."
pytest

echo "Build and test complete!"
