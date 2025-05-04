#!/usr/bin/env python
"""
Test runner script for RAGnificent.

This script sets up the proper import paths before running pytest.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

# Create a symlink from tests/RAGnificent to the RAGnificent directory if needed
test_module_symlink = project_root / "tests" / "RAGnificent"
ragnificent_dir = project_root / "RAGnificent"

if not test_module_symlink.exists() and ragnificent_dir.exists():
    try:
        # For Windows, we need to use a directory junction instead of a symlink
        # which requires admin privileges
        if os.name == 'nt':
            # Create a directory junction
            os.system(f'mklink /J "{test_module_symlink}" "{ragnificent_dir}"')
        else:
            # For Unix systems, create a symlink
            os.symlink(ragnificent_dir, test_module_symlink, target_is_directory=True)
        print(f"Created symlink from {test_module_symlink} to {ragnificent_dir}")
    except Exception as e:
        print(f"Warning: Could not create symlink: {e}")
        print("Tests may fail due to import issues.")
        
# Modify Python path to include the project root for absolute imports
os.environ['PYTHONPATH'] = f"{project_root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"

# Run pytest with the specified arguments
def run_tests():
    """Run the tests with proper path configuration."""
    args = sys.argv[1:] or ["."]
    cmd = [sys.executable, "-m", "pytest"] + args
    print(f"Running command: {' '.join(cmd)}")
    return subprocess.call(cmd)

if __name__ == "__main__":
    sys.exit(run_tests())
