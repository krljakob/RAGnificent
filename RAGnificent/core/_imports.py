"""
Common import utilities to reduce repetitive import fallback patterns.
"""

import sys
from pathlib import Path
from typing import Any, Optional

def ensure_path_in_sys(path: Path) -> None:
    """Ensure a path is in sys.path if not already present."""
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

def get_module_with_fallback(module_name: str, fallback_name: str, package: Optional[str] = None) -> Any:
    """Import module with fallback pattern commonly used in the codebase."""
    try:
        if package:
            return __import__(f"{package}.{module_name}", fromlist=[module_name])
        else:
            return __import__(module_name)
    except ImportError:
        return __import__(fallback_name)

def setup_core_imports() -> None:
    """Set up common paths for core module imports."""
    current_file = Path(__file__)
    core_path = current_file.parent
    utils_path = current_file.parent.parent / "utils"
    
    ensure_path_in_sys(core_path)
    ensure_path_in_sys(utils_path)