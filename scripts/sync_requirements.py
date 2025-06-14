#!/usr/bin/env python3
"""
Sync requirements.txt with [project.dependencies] from pyproject.toml.
Usage: python scripts/sync_requirements.py
"""
import os
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore

PYPROJECT_PATH = Path(__file__).parent.parent / "pyproject.toml"
REQUIREMENTS_PATH = Path(__file__).parent.parent / "requirements.txt"

HEADER = """# This file is auto-synced with pyproject.toml. For dev/test/typing/docs dependencies, see pyproject.toml.\n"""

def extract_dependencies(pyproject_path):
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    return data.get("project", {}).get("dependencies", [])

import tempfile


def write_requirements(requirements_path, dependencies):
    dir_name = os.path.dirname(requirements_path)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=dir_name, delete=False) as tmp_file:
        tmp_file.write(HEADER)
        for dep in dependencies:
            tmp_file.write(f"{dep}\n")
        temp_path = tmp_file.name
    os.replace(temp_path, requirements_path)

def main():
    if not PYPROJECT_PATH.exists():
        sys.exit(1)
    deps = extract_dependencies(PYPROJECT_PATH)
    if not deps:
        sys.exit(1)
    write_requirements(REQUIREMENTS_PATH, deps)

if __name__ == "__main__":
    main()
