#!/usr/bin/env python
"""
Import fixer script for RAGnificent.

This script will recursively find all Python files in both tests and the RAGnificent package
and update their import statements to use direct relative imports instead of absolute imports.
"""

import os
import re
import sys
from pathlib import Path

# Get the project root path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
TESTS_ROOT = PROJECT_ROOT / "tests"
RAGNIFICENT_ROOT = PROJECT_ROOT / "RAGnificent"


def fix_imports_in_file(file_path, is_test_file=False):
    """Fix import statements in a Python file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if the file already contains the import fix
    if "# Import fix applied" in content:
        print(f"Skipping {file_path} (already fixed)")
        return 0

    # For source files, we only care about RAGnificent imports
    if not is_test_file and "from RAGnificent." not in content:
        print(f"Skipping {file_path} (no self-imports)")
        return 0
    elif is_test_file and "import RAGnificent" not in content and "from RAGnificent" not in content:
        print(f"Skipping {file_path} (no RAGnificent imports)")
        return 0

    # Determine path setup code based on file type
    if is_test_file:
        # For test files
        path_setup_code = """
# Use direct import path rather than relying on package structure
# This allows tests to run even with inconsistent Python package installation
# Import fix applied
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
"""
    else:
        # For source files
        path_setup_code = """
# Use relative imports for internal modules
# Import fix applied
sys.path.insert(0, str(Path(__file__).parent.parent))
"""

    # Add import section pattern
    import_section_pattern = r"((?:import|from)[^\n]*\n)+"

    # Check if Path and sys are already imported
    if "from pathlib import Path" not in content:
        path_setup_code = "from pathlib import Path\n" + path_setup_code

    if "import sys" not in content:
        path_setup_code = "import sys\n" + path_setup_code

    if imports_match := re.search(import_section_pattern, content):
        insert_pos = imports_match.end()
        modified_content = content[:insert_pos] + path_setup_code + content[insert_pos:]
    else:
        if docstring_match := re.search(r'"""[^"]*"""', content):
            insert_pos = docstring_match.end() + 1
            modified_content = content[:insert_pos] + "\n" + path_setup_code + content[insert_pos:]
        else:
            modified_content = path_setup_code + content

    # Replace "from RAGnificent.X import Y" with "from X import Y"
    modified_content = re.sub(
        r"from\s+RAGnificent\.(\w+(?:\.\w+)*)\s+import",
        r"from \1 import",
        modified_content
    )

    # Replace "import RAGnificent.X" with "import X"
    modified_content = re.sub(
        r"import\s+RAGnificent\.(\w+(?:\.\w+)*)",
        r"import \1",
        modified_content
    )

    # Save the modified content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)

    print(f"Fixed imports in {file_path}")
    return 1


def find_and_fix_test_files():
    """Find all Python test files and fix their imports."""
    fixed_count = 0
    
    for root, _, files in os.walk(TESTS_ROOT):
        for file in files:
            if file.endswith('.py') and file.startswith('test_'):
                file_path = Path(root) / file
                fixed_count += fix_imports_in_file(file_path, is_test_file=True)
    
    return fixed_count


def find_and_fix_source_files():
    """Find all Python source files in RAGnificent and fix their imports."""
    fixed_count = 0
    
    for root, _, files in os.walk(RAGNIFICENT_ROOT):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                fixed_count += fix_imports_in_file(file_path, is_test_file=False)
    
    return fixed_count


if __name__ == "__main__":
    print(f"Looking for test files under {TESTS_ROOT}")
    test_fixed_count = find_and_fix_test_files()
    print(f"Fixed imports in {test_fixed_count} test files")
    
    print(f"\nLooking for source files under {RAGNIFICENT_ROOT}")
    source_fixed_count = find_and_fix_source_files()
    print(f"Fixed imports in {source_fixed_count} source files")
    
    print(f"\nTotal files fixed: {test_fixed_count + source_fixed_count}")
