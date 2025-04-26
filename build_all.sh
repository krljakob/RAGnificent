#!/usr/bin/env bash
# build_all.sh
# One-shot script: create venv with uv, activate, build Rust extension, install deps, run tests

set -e

# 1. Create .venv if not present
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment with uv..."
  uv venv
fi

# 2. Activate the venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Virtual environment activated."

# 3. Install Python dependencies
echo "Installing Python dependencies..."
uv pip install -r requirements.txt

# 4. Build Rust extension with maturin (in venv)
echo "Building Rust extension with maturin..."
maturin build --release
maturin develop --release

# 5. Run tests
echo "Running tests..."
pytest --maxfail=0 --disable-warnings

echo "Build and test complete!"
