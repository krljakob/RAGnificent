"""
Test configuration and fixtures for RAGnificent.

This module is automatically loaded by pytest and ensures that all the necessary
Python paths are correctly set up for imports to work across test files.
"""

import sys
import importlib.util
from pathlib import Path

# Add the repository root to the Python path to ensure imports work correctly
repo_root = Path(__file__).parent.parent
ragnificent_path = repo_root / "RAGnificent"
utils_path = ragnificent_path / "utils"
core_path = ragnificent_path / "core"

# Clear any existing paths that might interfere with our imports
# but keep standard library and site-packages
sys.path = [p for p in sys.path if 'site-packages' in p or 'lib' in p.lower()]

# Add paths in priority order
sys.path.insert(0, str(repo_root))  # Highest priority - allows 'RAGnificent.x' imports
sys.path.insert(0, str(ragnificent_path))  # Next priority - allows 'utils.x' imports
sys.path.insert(0, str(utils_path.parent))  # Next - for direct imports from utils

# Helper function to dynamically import modules from file paths
def import_module_from_path(module_name, file_path):
    """Import a module directly from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
