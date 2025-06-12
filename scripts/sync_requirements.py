#!/usr/bin/env python3
"""
Sync requirements.txt with [project.dependencies] from pyproject.toml.
Usage: python scripts/sync_requirements.py
"""
import sys
import os
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

def write_requirements(requirements_path, dependencies):
    with open(requirements_path, "w", encoding="utf-8") as f:
        f.write(HEADER)
        for dep in dependencies:
            f.write(f"{dep}\n")

def main():
    if not PYPROJECT_PATH.exists():
        print(f"pyproject.toml not found at {PYPROJECT_PATH}", file=sys.stderr)
        sys.exit(1)
    deps = extract_dependencies(PYPROJECT_PATH)
    if not deps:
        print("No dependencies found in pyproject.toml", file=sys.stderr)
        sys.exit(1)
    write_requirements(REQUIREMENTS_PATH, deps)
    print(
        "requirements.txt synced with [project.dependencies] from pyproject.toml."
    )

if __name__ == "__main__":
    main()
